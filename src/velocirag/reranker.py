"""
Cross-encoder reranking for Velocirag.

Uses sentence-transformers CrossEncoder for optimized relevance scoring
of query-document pairs. Provides smart document excerpting and graceful
degradation when models fail to load. Requires optional 'reranker' dependency.
"""

import logging
import threading
import warnings
from typing import Dict, List, Any

# Lazy import — sentence-transformers pulls in PyTorch (~2.5s).
# Defer until the model is actually needed.
HAS_CROSS_ENCODER = None  # Tri-state: None = unchecked, True/False = resolved
CrossEncoder = None
_import_lock = threading.Lock()

# Constants
DEFAULT_MODEL = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
MAX_EXCERPT_LENGTH = 2000
EXCERPT_HEAD = 1000
EXCERPT_TAIL = 1000

logger = logging.getLogger("velocirag.reranker")


class Reranker:
    """
    Cross-encoder reranker for search result relevance scoring.
    
    Uses a pre-trained cross-encoder model to score query-document pairs
    and rerank search results by relevance. Features lazy model loading,
    smart document excerpting for long content, and graceful degradation.
    """
    
    def __init__(self, model_name: str = DEFAULT_MODEL):
        """
        Initialize reranker with lazy model loading.
        
        Args:
            model_name: HuggingFace cross-encoder model identifier
        """
        self.model_name = model_name
        self._model = None
        self._loaded = False
        self._load_error = None
    
    def rerank(self, query: str, results: List[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Rerank search results using cross-encoder relevance scoring.
        
        Scores query-document pairs and returns results sorted by relevance.
        Adds rerank_score to each result's metadata.
        
        Args:
            query: Original search query
            results: List of results with 'content' key
            limit: Max results to return
            
        Returns:
            Reranked results with rerank_score in metadata, limited to specified count
        """
        if not results:
            return results
        
        # Ensure model is loaded
        if not self._loaded and self._load_error is None:
            self._load_model()
        
        # Graceful degradation if model unavailable
        if self._load_error:
            logger.warning(f"Reranker unavailable ({self._load_error}), returning unranked results")
            return results[:limit]
        
        try:
            # Build query-document pairs with smart excerpting
            pairs = []
            for result in results:
                content = result.get('content', '')
                excerpted_content = self._excerpt_content(content)
                pairs.append((query, excerpted_content))
            
            # Score all pairs at once (batched inference)
            scores = self._model.predict(pairs, show_progress_bar=False)
            
            # Attach scores to results and sort by relevance
            scored_results = []
            for i, result in enumerate(results):
                result_copy = result.copy()
                if 'metadata' not in result_copy:
                    result_copy['metadata'] = {}
                
                rerank_score = round(float(scores[i]), 4)
                result_copy['metadata']['rerank_score'] = rerank_score
                scored_results.append((rerank_score, result_copy))
            
            # Sort by score descending and return top results
            scored_results.sort(key=lambda x: x[0], reverse=True)
            return [result for _, result in scored_results[:limit]]
            
        except Exception as e:
            logger.warning(f"Reranking failed: {e}, returning unranked results")
            return results[:limit]
    
    def __call__(self, query: str, results: list[dict], limit: int = 5) -> list[dict]:
        """Make Reranker callable — enables passing directly to Searcher."""
        return self.rerank(query, results, limit)

    def get_status(self) -> Dict[str, Any]:
        """
        Get reranker status and configuration.
        
        Returns:
            Dictionary with model info, load status, and error state
        """
        return {
            'model_name': self.model_name,
            'loaded': self._loaded,
            'error': self._load_error
        }
    
    def _load_model(self) -> None:
        """Load cross-encoder model with warning suppression."""
        global HAS_CROSS_ENCODER, CrossEncoder

        if self._loaded or self._load_error:
            return

        # Lazy-resolve: first time, try importing sentence-transformers
        with _import_lock:
            if HAS_CROSS_ENCODER is None:
                try:
                    from sentence_transformers import CrossEncoder as _CE
                    CrossEncoder = _CE
                    HAS_CROSS_ENCODER = True
                except ImportError:
                    HAS_CROSS_ENCODER = False

        if not HAS_CROSS_ENCODER:
            self._load_error = "sentence-transformers not installed (install with: pip install velocirag[reranker])"
            logger.warning(f"Cross-encoder unavailable: {self._load_error}")
            return

        try:
            logger.info(f"Loading cross-encoder model: {self.model_name}")

            # Suppress warnings during model loading
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                # Temporarily suppress sentence-transformers logging
                st_logger = logging.getLogger("sentence_transformers")
                original_level = st_logger.level
                st_logger.setLevel(logging.CRITICAL)

                try:
                    self._model = CrossEncoder(self.model_name)
                    self._loaded = True
                    logger.info(f"Cross-encoder model loaded: {self.model_name}")
                finally:
                    st_logger.setLevel(original_level)
                    
        except Exception as e:
            self._load_error = str(e)
            logger.error(f"Failed to load cross-encoder model {self.model_name}: {e}")
    
    def _excerpt_content(self, content: str) -> str:
        """
        Create smart excerpt for long documents.
        
        For documents longer than MAX_EXCERPT_LENGTH, takes first EXCERPT_HEAD
        characters and last EXCERPT_TAIL characters to capture both introduction
        and conclusion, connected with ellipsis marker.
        
        Args:
            content: Full document content
            
        Returns:
            Excerpted content suitable for cross-encoder input
        """
        if len(content) <= MAX_EXCERPT_LENGTH:
            return content
        
        # Take head and tail to capture intro + conclusion
        head = content[:EXCERPT_HEAD].rstrip()
        tail = content[-EXCERPT_TAIL:].lstrip()
        
        return f"{head}\n...\n{tail}"