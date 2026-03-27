"""
Production embedding engine with intelligent caching.

High-throughput vector embedding system with MD5-based content caching,
lazy model loading, and LRU eviction. Uses ONNX Runtime for fast CPU inference
without PyTorch dependency. Extracted from production Jawz vector search system.
"""

import hashlib
import json
import logging
import os
import threading
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
CACHE_VERSION = 3  # Increment for ONNX transition
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
    - ONNX Runtime for fast CPU inference (no PyTorch dependency)
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
            model_name: Model identifier (currently supports "all-MiniLM-L6-v2")
            cache_dir: Directory for persistent cache storage. None = memory-only cache.
            cache_size: Maximum cache entries before LRU eviction
            normalize: Whether to L2-normalize embeddings
            
        Raises:
            ValueError: If cache_size is outside valid range or model not supported
        """
        # Validate cache size
        if not MIN_CACHE_SIZE <= cache_size <= MAX_CACHE_SIZE:
            raise ValueError(
                f"cache_size must be between {MIN_CACHE_SIZE} and {MAX_CACHE_SIZE}, got {cache_size}"
            )
        
        # Validate model (for now, only support the optimized one)
        if model_name != "all-MiniLM-L6-v2":
            raise ValueError(f"Currently only 'all-MiniLM-L6-v2' is supported, got '{model_name}'")
        
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.cache_size = cache_size
        self.normalize_embeddings = normalize
        
        # Internal state
        self._model_session = None
        self._tokenizer = None
        self._cache = OrderedDict()  # MD5 hash -> embedding list (OrderedDict for O(1) LRU)
        self._cache_lock = threading.Lock()
        self._model_lock = threading.Lock()  # Protect model loading and usage
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
        
        # Ensure model is loaded (thread-safe)
        with self._model_lock:
            if self._model_session is None:
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
            'model_loaded': self._model_session is not None,
            'cache_dir': self.cache_dir,
            'normalize_embeddings': self.normalize_embeddings,
            'backend': 'onnx'
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
                    "_backend": "onnx",
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
                warnings.filterwarnings("ignore", module="onnxruntime")
                
                yield
                
        finally:
            # Restore original environment variables
            for var, value in original_env.items():
                if value is None:
                    os.environ.pop(var, None)
                else:
                    os.environ[var] = value
    
    def _load_model(self) -> None:
        """Load the ONNX model and tokenizer."""
        logger.info(f"Loading ONNX model: {self.model_name}")
        
        try:
            with self._model_loading_context():
                import onnxruntime as ort
                from tokenizers import Tokenizer
                from huggingface_hub import hf_hub_download
                
                # Create model cache directory
                model_cache_dir = os.path.expanduser(f"~/.cache/velocirag/models/{self.model_name}")
                os.makedirs(model_cache_dir, exist_ok=True)
                
                model_path = os.path.join(model_cache_dir, "model.onnx")
                tokenizer_path = os.path.join(model_cache_dir, "tokenizer.json")
                
                # Download model files if not cached
                if not os.path.exists(model_path):
                    logger.info("Downloading ONNX model (first run)...")
                    hf_hub_download(
                        repo_id="optimum/all-MiniLM-L6-v2",
                        filename="model.onnx",
                        local_dir=model_cache_dir,
                        local_dir_use_symlinks=False
                    )
                
                if not os.path.exists(tokenizer_path):
                    logger.info("Downloading tokenizer...")
                    hf_hub_download(
                        repo_id="optimum/all-MiniLM-L6-v2", 
                        filename="tokenizer.json",
                        local_dir=model_cache_dir,
                        local_dir_use_symlinks=False
                    )
                
                # Load ONNX model
                session_options = ort.SessionOptions()
                session_options.intra_op_num_threads = os.cpu_count()
                session_options.log_severity_level = 3  # Suppress warnings
                
                self._model_session = ort.InferenceSession(
                    model_path,
                    sess_options=session_options
                )
                
                # Load tokenizer
                self._tokenizer = Tokenizer.from_file(tokenizer_path)
                self._tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
                self._tokenizer.enable_truncation(max_length=256)
                
            logger.info(f"ONNX model loaded: {self.model_name} ({self._get_model_dimensions()} dims)")
            
        except Exception as e:
            logger.error(f"Failed to load ONNX model {self.model_name}: {e}")
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
            cache_backend = cache_data.get("_backend", "sentence-transformers")
            
            if cache_version < CACHE_VERSION:
                logger.warning(f"Cache version outdated: expected {CACHE_VERSION}, got {cache_version}. Starting fresh.")
                return
            
            if cache_model != self.model_name:
                logger.warning(f"Cache model mismatch: expected {self.model_name}, got {cache_model}. Starting fresh.")
                return
            
            if cache_backend != "onnx":
                logger.warning(f"Cache backend mismatch: expected onnx, got {cache_backend}. Starting fresh.")
                return
            
            # Load embeddings, validating values are lists
            raw_embeddings = cache_data.get("embeddings", {})
            valid_embeddings = {
                k: v for k, v in raw_embeddings.items()
                if isinstance(v, list) and len(v) > 0
            }
            skipped = len(raw_embeddings) - len(valid_embeddings)

            with self._cache_lock:
                self._cache = OrderedDict(valid_embeddings)

            if skipped:
                logger.warning(f"Cache: skipped {skipped} invalid entries from {cache_path}")
            logger.info(f"Cache loaded: {len(valid_embeddings)} entries from {cache_path}")
            
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
        should_save = False
        if cache_misses:
            new_embeddings = self._encode_onnx(cache_misses)
            
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

                # Periodic save (check inside lock to avoid race on _new_entries)
                should_save = self._new_entries >= CACHE_SAVE_INTERVAL
                if should_save:
                    self._new_entries = 0

            if should_save:
                self.save_cache()
        
        # Convert to final numpy array
        return np.array(embeddings)
    
    def _encode_onnx(self, texts: list[str]) -> np.ndarray:
        """
        Encode texts using ONNX model with mean pooling.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            2D numpy array of embeddings
        """
        # Thread-safe access to model and tokenizer
        with self._model_lock:
            # Tokenize
            encoded = self._tokenizer.encode_batch(texts)
            
            # Convert to arrays
            input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
            attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
            token_type_ids = np.zeros_like(input_ids, dtype=np.int64)
            
            # Run ONNX inference
            outputs = self._model_session.run(None, {
                "input_ids": input_ids,
                "attention_mask": attention_mask, 
                "token_type_ids": token_type_ids
            })
        
        # Mean pooling (same as sentence-transformers does)
        token_embeddings = outputs[0]  # (batch, seq_len, hidden_dim)
        input_mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
        embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1) / np.clip(
            input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None
        )
        
        return embeddings
    
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
        """Get embedding dimensions."""
        # all-MiniLM-L6-v2 always has 384 dimensions
        return 384
    
    def _get_max_seq_length(self) -> int:
        """Get model's maximum sequence length."""
        # all-MiniLM-L6-v2 uses 256 max sequence length
        return 256