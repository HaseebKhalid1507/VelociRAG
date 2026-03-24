"""
Production embedding engine with intelligent caching.

High-throughput vector embedding system with MD5-based content caching,
lazy model loading, and LRU eviction. Wraps sentence-transformers models
with optimized batch processing and persistent cache storage.
Extracted from production Jawz vector search system.
"""

import hashlib
import json
import logging
import os
import threading
import time
import warnings
import atexit
import weakref
from typing import Any
from collections import OrderedDict
from contextlib import contextmanager
import numpy as np

# Constants
DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CACHE_SIZE = 10000
MIN_CACHE_SIZE = 100
MAX_CACHE_SIZE = 100000
CACHE_SAVE_INTERVAL = 200
CACHE_VERSION = 2
CACHE_FILENAME = "embedding_cache.json"

# Setup logging
logger = logging.getLogger("velocirag.embedder")


class Embedder:
    """
    Production embedding engine with intelligent caching.
    
    Features:
    - Lazy model loading (model loaded only on first embed() call)
    - MD5 content hash caching with LRU eviction
    - Atomic disk cache persistence 
    - Batch processing optimization
    - Configurable model and cache parameters
    - Warning suppression during model initialization
    """
    
    # Class-level tracking for atexit cleanup
    _instances: list[weakref.ReferenceType['Embedder']] = []
    _atexit_registered = False
    
    @classmethod
    def _register_atexit(cls):
        """Register atexit cleanup handler once at class level."""
        if not cls._atexit_registered:
            atexit.register(cls._cleanup_all)
            cls._atexit_registered = True
    
    @classmethod
    def _cleanup_all(cls):
        """Cleanup all live instances on exit."""
        for instance_ref in cls._instances[:]:  # Copy to avoid modification during iteration
            instance = instance_ref()
            if instance is not None:
                instance.save_cache()
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_dir: str | None = None,
        cache_size: int = DEFAULT_CACHE_SIZE,
        normalize: bool = False
    ):
        """
        Initialize embedding engine.
        
        Args:
            model_name: HuggingFace model identifier
            cache_dir: Directory for persistent cache storage. None = memory-only cache.
            cache_size: Maximum cache entries before LRU eviction
            normalize: Whether to L2-normalize embeddings
            
        Raises:
            ValueError: If cache_size is outside valid range
        """
        # Validate cache size
        if not MIN_CACHE_SIZE <= cache_size <= MAX_CACHE_SIZE:
            raise ValueError(
                f"cache_size must be between {MIN_CACHE_SIZE} and {MAX_CACHE_SIZE}, got {cache_size}"
            )
        
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.cache_size = cache_size
        self.normalize_embeddings = normalize
        
        # Internal state
        self._model = None
        self._cache = OrderedDict()  # MD5 hash -> embedding list (OrderedDict for O(1) LRU)
        self._cache_lock = threading.Lock()
        self._new_entries = 0  # Track entries since last save
        
        # Load existing cache
        self._load_cache()
        
        # Register instance for class-level cleanup
        self._instances.append(weakref.ref(self))
        self._register_atexit()
    
    def embed(self, texts: str | list[str]) -> np.ndarray:
        """
        Generate embeddings for text(s) with intelligent caching.
        
        Model loads lazily on first call. Subsequent calls check cache first,
        only computing embeddings for unseen content.
        
        Args:
            texts: Single text string or list of texts to embed
            
        Returns:
            NumPy array of embeddings:
            - Single text: shape (dimensions,)
            - Multiple texts: shape (num_texts, dimensions)
            - Empty list: shape (0, dimensions)
            
        Raises:
            ValueError: If any text is empty string
        """
        # Handle input types
        if isinstance(texts, str):
            single_input = True
            text_list = [texts]
        elif isinstance(texts, list):
            single_input = False
            text_list = texts
        else:
            raise ValueError(f"texts must be str or list[str], got {type(texts).__name__}")
        
        # Handle empty list
        if not text_list:
            dimensions = self._get_model_dimensions()
            return np.empty((0, dimensions))
        
        # Check for empty strings
        for i, text in enumerate(text_list):
            if not text or not text.strip():
                if single_input:
                    raise ValueError("Cannot embed empty text")
                else:
                    raise ValueError(f"Cannot embed empty text at index {i}")
        
        # Ensure model is loaded
        if self._model is None:
            self._load_model()
        
        # Batch process with caching
        embeddings = self._embed_batch(text_list)
        
        # Apply normalization if enabled
        if self.normalize_embeddings:
            embeddings = self.normalize(embeddings)
        
        # Return correct shape
        if single_input:
            return embeddings[0]  # 1D array
        else:
            return embeddings  # 2D array
    
    def normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """
        L2-normalize embeddings for cosine similarity search.
        
        Args:
            embeddings: Raw embeddings array
            
        Returns:
            L2-normalized embeddings (unit vectors)
        """
        if embeddings.ndim == 1:
            norm = np.linalg.norm(embeddings)
            return embeddings / norm if norm > 0 else embeddings
        else:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            # Avoid division by zero
            norms = np.where(norms > 0, norms, 1.0)
            return embeddings / norms
    
    def get_model_info(self) -> dict[str, Any]:
        """
        Get model metadata and cache statistics.
        
        Returns:
            Dictionary with model and cache info
        """
        dimensions = self._get_model_dimensions()
        max_seq_length = self._get_max_seq_length()
        
        with self._cache_lock:
            cache_size = len(self._cache)
        
        return {
            'model_name': self.model_name,
            'dimensions': dimensions,
            'max_seq_length': max_seq_length,
            'cache_size': cache_size,
            'cache_max_size': self.cache_size,
            'model_loaded': self._model is not None,
            'cache_dir': self.cache_dir,
            'normalize_embeddings': self.normalize_embeddings
        }
    
    def clear_cache(self) -> None:
        """
        Clear all cached embeddings (memory and disk).
        Model remains loaded if already initialized.
        """
        with self._cache_lock:
            self._cache.clear()
            self._new_entries = 0
        
        # Remove cache file if it exists
        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, CACHE_FILENAME)
            try:
                if os.path.exists(cache_path):
                    os.remove(cache_path)
                    logger.info(f"Cache file removed: {cache_path}")
            except OSError as e:
                logger.warning(f"Failed to remove cache file: {e}")
    
    def save_cache(self) -> None:
        """
        Force immediate cache persistence to disk.
        Normally called automatically during cleanup.
        """
        if not self.cache_dir:
            return
        
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_path = os.path.join(self.cache_dir, CACHE_FILENAME)
            
            with self._cache_lock:
                if not self._cache:
                    return  # No cache to save
                
                cache_data = {
                    "_version": CACHE_VERSION,
                    "_model": self.model_name,
                    "embeddings": self._cache.copy()
                }
            
            # Atomic write via temp file
            temp_path = cache_path + ".tmp"
            with open(temp_path, 'w') as f:
                json.dump(cache_data, f)
            
            os.replace(temp_path, cache_path)
            logger.info(f"Cache saved: {len(cache_data['embeddings'])} entries to {cache_path}")
            
        except OSError as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def cleanup(self) -> None:
        """
        Persist cache to disk and cleanup resources.
        Called automatically via atexit hook.
        """
        self.save_cache()
    
    @contextmanager
    def _model_loading_context(self):
        """Context manager for safe global state changes during model loading."""
        # Save original environment variables
        original_env = {}
        env_vars = ['HF_HUB_DISABLE_PROGRESS_BARS', 'TOKENIZERS_PARALLELISM', 'HF_HUB_DISABLE_TELEMETRY']
        
        for var in env_vars:
            original_env[var] = os.environ.get(var)
        
        try:
            # Set temporary environment variables
            os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
            
            # Use warnings context manager
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", module="urllib3")
                warnings.filterwarnings("ignore", module="transformers")
                
                yield
                
        finally:
            # Restore original environment variables
            for var, value in original_env.items():
                if value is None:
                    os.environ.pop(var, None)
                else:
                    os.environ[var] = value
    
    def _load_model(self) -> None:
        """Load the sentence transformer model with warning suppression."""
        logger.info(f"Loading model: {self.model_name}")
        
        try:
            with self._model_loading_context():
                # Import and load model (deferred heavy import)
                from sentence_transformers import SentenceTransformer
                
                self._model = SentenceTransformer(self.model_name)
                
            logger.info(f"Model loaded: {self.model_name} ({self._get_model_dimensions()} dims)")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def _load_cache(self) -> None:
        """Load cache from disk if cache_dir is set."""
        if not self.cache_dir:
            return
        
        cache_path = os.path.join(self.cache_dir, CACHE_FILENAME)
        
        if not os.path.exists(cache_path):
            return
        
        try:
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
            
            # Check cache version and model compatibility
            cache_version = cache_data.get("_version", 1)
            cache_model = cache_data.get("_model", "unknown")
            
            if cache_version != CACHE_VERSION:
                logger.warning(f"Cache version mismatch: expected {CACHE_VERSION}, got {cache_version}. Starting fresh.")
                return
            
            if cache_model != self.model_name:
                logger.warning(f"Cache model mismatch: expected {self.model_name}, got {cache_model}. Starting fresh.")
                return
            
            # Load embeddings
            embeddings = cache_data.get("embeddings", {})
            
            with self._cache_lock:
                self._cache = OrderedDict(embeddings)
            
            logger.info(f"Cache loaded: {len(embeddings)} entries from {cache_path}")
            
        except (json.JSONDecodeError, OSError, KeyError) as e:
            logger.warning(f"Failed to load cache from {cache_path}: {e}. Starting fresh.")
    
    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """
        Embed a batch of texts with caching.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            2D numpy array of embeddings
        """
        # Generate hashes for all texts
        text_hashes = [self._text_hash(text) for text in texts]
        
        # Check cache for hits/misses
        embeddings = []
        cache_misses = []
        miss_indices = []
        
        with self._cache_lock:
            for i, (text, text_hash) in enumerate(zip(texts, text_hashes)):
                if text_hash in self._cache:
                    # Cache hit - convert from list back to numpy
                    embedding = np.array(self._cache[text_hash])
                    embeddings.append(embedding)
                    
                    # Update LRU order - O(1) with OrderedDict
                    self._cache.move_to_end(text_hash)
                else:
                    # Cache miss - placeholder for now
                    embeddings.append(None)
                    cache_misses.append(text)
                    miss_indices.append(i)
        
        # Compute missing embeddings if any
        if cache_misses:
            new_embeddings = self._model.encode(cache_misses, convert_to_numpy=True)
            
            # Handle single embedding case (ensure 2D)
            if new_embeddings.ndim == 1:
                new_embeddings = new_embeddings.reshape(1, -1)
            
            # Store in cache and update results
            with self._cache_lock:
                for embedding, miss_idx in zip(new_embeddings, miss_indices):
                    text_hash = text_hashes[miss_idx]
                    
                    # Store in cache as list (JSON serializable)
                    self._cache[text_hash] = embedding.tolist()
                    
                    # Update results
                    embeddings[miss_idx] = embedding
                    self._new_entries += 1
                
                # Apply LRU eviction if needed
                self._evict_lru()
            
            # Periodic save
            if self._new_entries >= CACHE_SAVE_INTERVAL:
                self._new_entries = 0
                self.save_cache()
        
        # Convert to final numpy array
        return np.array(embeddings)
    
    def _evict_lru(self) -> None:
        """Evict least recently used cache entries if over limit."""
        while len(self._cache) > self.cache_size:
            if not self._cache:
                break
            
            # Remove oldest item (first item in OrderedDict)
            self._cache.popitem(last=False)
    
    def _text_hash(self, text: str) -> str:
        """Generate MD5 hash for text content."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _get_model_dimensions(self) -> int:
        """Get embedding dimensions, with fallback if model not loaded."""
        if self._model is not None:
            return self._model.get_sentence_embedding_dimension()
        else:
            # Common model dimensions for fallback
            model_dims = {
                "all-MiniLM-L6-v2": 384,
                "all-mpnet-base-v2": 768,
                "all-MiniLM-L12-v2": 384,
            }
            if self.model_name in model_dims:
                return model_dims[self.model_name]
            else:
                # For empty list case, we need dimensions but can't load model yet
                # Use 384 as fallback but log a warning
                logger.warning(f"Unknown model {self.model_name}, using 384 dimensions as fallback. "
                             f"Call embed() with non-empty input to load model and get exact dimensions.")
                return 384
    
    def _get_max_seq_length(self) -> int:
        """Get model's maximum sequence length."""
        if self._model is not None:
            return self._model.max_seq_length
        else:
            return 256  # Conservative fallback