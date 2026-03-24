"""
Velociragtor Phase 4 Searcher - High-level search orchestration.

The apex predator of search orchestration, combining query variants, 
embedding, vector search, and reciprocal rank fusion into a single
dominant pipeline.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Callable

import numpy as np

from .variants import generate_variants
from .rrf import reciprocal_rank_fusion
from .store import VectorStore
from .embedder import Embedder

# Constants
MIN_QUERY_LENGTH = 1
MAX_QUERY_LENGTH = 1000
SINGLE_VARIANT_THRESHOLD = 3
DEFAULT_SEARCH_LIMIT = 5
MAX_SEARCH_LIMIT = 100
DEFAULT_SIMILARITY_THRESHOLD = 0.3
DEFAULT_RRF_K = 60
MIN_RRF_K = 1
MAX_RRF_K = 1000
VARIANT_SEARCH_MULTIPLIER = 2
MAX_VARIANT_LIMIT = 20
TIMING_PRECISION = 2

logger = logging.getLogger("velocirag.searcher")


class SearchError(Exception):
    """Base exception for search pipeline errors."""
    pass


class Searcher:
    """
    High-level search orchestration combining query variants, embeddings,
    vector search, and reciprocal rank fusion.
    
    The apex predator of the Velociragtor ecosystem.
    """
    
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        rrf_k: int = DEFAULT_RRF_K,
        reranker: Optional[Callable[[str, List[Dict]], List[Dict]]] = None
    ):
        """
        Initialize search orchestrator.
        
        Args:
            store: VectorStore instance for FAISS search operations
            embedder: Embedder instance for query vectorization
            rrf_k: RRF parameter for result fusion (default 60)
            reranker: Optional reranking function with signature:
                     (query: str, results: list[dict]) -> list[dict]
        """
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        
        # Validate RRF k parameter
        if not isinstance(rrf_k, int):
            raise ValueError(f"rrf_k must be an integer, got {type(rrf_k).__name__}")
        if not MIN_RRF_K <= rrf_k <= MAX_RRF_K:
            raise ValueError(f"rrf_k must be between {MIN_RRF_K} and {MAX_RRF_K}, got {rrf_k}")
        
        self.rrf_k = rrf_k
        
    def search(
        self,
        query: str,
        limit: int = DEFAULT_SEARCH_LIMIT,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        include_stats: bool = False
    ) -> Dict[str, Any]:
        """
        Execute complete search pipeline with query variants and RRF fusion.
        
        Args:
            query: Search query string
            limit: Maximum results to return
            threshold: Minimum similarity score threshold
            include_stats: Include performance statistics
            
        Returns:
            Dictionary with results, query, timing, and variant information
        """
        start_time = time.time()
        
        # Validate input parameters
        if not isinstance(query, str):
            raise ValueError(f"query must be a string, got {type(query).__name__}")
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError(f"limit must be a positive integer, got {limit}")
        if limit > MAX_SEARCH_LIMIT:
            limit = MAX_SEARCH_LIMIT
        if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
            raise ValueError(f"threshold must be between 0 and 1, got {threshold}")
        
        # Handle empty query
        if not query or not query.strip():
            return {
                'results': [],
                'query': query,
                'total_results': 0,
                'search_time_ms': 0.0,
                'variants_used': [],
                'stats': {} if include_stats else None
            }
        
        query = query.strip()
        
        # Validate query length
        if len(query) < MIN_QUERY_LENGTH:
            return {
                'results': [],
                'query': query,
                'total_results': 0,
                'search_time_ms': round((time.time() - start_time) * 1000, TIMING_PRECISION),
                'variants_used': [],
                'stats': {} if include_stats else None
            }
        if len(query) > MAX_QUERY_LENGTH:
            raise ValueError(f"Query too long: {len(query)} chars (max {MAX_QUERY_LENGTH})")
        
        stats = {} if include_stats else None
        
        try:
            # Step 1: Generate variants
            variant_start = time.time()
            if len(query) >= SINGLE_VARIANT_THRESHOLD:
                variants = generate_variants(query)
            else:
                variants = [query]
            
            if not variants:  # Handle empty variant generation
                variants = [query]
                
            variant_time = (time.time() - variant_start) * 1000
            if include_stats:
                stats['variant_generation_ms'] = round(variant_time, TIMING_PRECISION)
            
            # Step 2: Embed all variants
            embed_start = time.time()
            try:
                variant_embeddings = self.embedder.embed(variants)
                # Handle shape differences - embedder returns different shapes for single vs multiple
                if variant_embeddings.ndim == 1:
                    # Single embedding, reshape to 2D
                    variant_embeddings = variant_embeddings.reshape(1, -1)
                elif len(variants) == 1 and variant_embeddings.ndim == 2 and variant_embeddings.shape[0] == 1:
                    # Single variant but embedder returned 2D with one row
                    pass
                
                # Normalize each embedding
                normalized_embeddings = []
                for i in range(len(variants)):
                    embedding = variant_embeddings[i] if variant_embeddings.ndim == 2 else variant_embeddings
                    # Normalize for cosine similarity
                    norm = np.linalg.norm(embedding)
                    normalized_embedding = embedding / norm if norm > 0 else embedding
                    normalized_embeddings.append(normalized_embedding)
                    
            except Exception as e:
                raise SearchError(f"Embedding failed for query '{query}': {e}")
            
            embed_time = (time.time() - embed_start) * 1000
            if include_stats:
                stats['embedding_time_ms'] = round(embed_time, TIMING_PRECISION)
            
            # Step 3: Search store for each variant
            search_start = time.time()
            all_results = []
            variant_stats = []
            
            for i, (variant, embedding) in enumerate(zip(variants, normalized_embeddings)):
                variant_search_start = time.time()
                try:
                    variant_limit = min(limit * VARIANT_SEARCH_MULTIPLIER, MAX_VARIANT_LIMIT)
                    variant_results = self.store.search(embedding, limit=variant_limit, min_similarity=threshold)
                    all_results.append(variant_results)
                    
                    if include_stats:
                        variant_search_time = (time.time() - variant_search_start) * 1000
                        variant_stats.append({
                            'variant': variant,
                            'results_found': len(variant_results),
                            'search_time_ms': round(variant_search_time, TIMING_PRECISION)
                        })
                        
                except Exception as e:
                    logger.warning(f"Search failed for variant '{variant}': {e}")
                    all_results.append([])
                    if include_stats:
                        variant_stats.append({
                            'variant': variant,
                            'results_found': 0,
                            'search_time_ms': 0.0,
                            'error': str(e)
                        })
            
            search_time = (time.time() - search_start) * 1000
            if include_stats:
                stats['search_time_ms'] = round(search_time, TIMING_PRECISION)
                stats['variants'] = variant_stats
            
            # Step 4: RRF fusion
            rrf_start = time.time()
            try:
                # Use doc_id from store results for deduplication
                def doc_id_extractor(result: Dict) -> str:
                    return result.get('doc_id', f"content_hash_{hash(result.get('content', ''))}")
                
                fused_results = reciprocal_rank_fusion(
                    all_results,
                    k=self.rrf_k,
                    doc_id_fn=doc_id_extractor
                )
            except Exception as e:
                raise SearchError(f"RRF fusion failed: {e}")
            
            rrf_time = (time.time() - rrf_start) * 1000
            if include_stats:
                stats['rrf_fusion_ms'] = round(rrf_time, TIMING_PRECISION)
            
            # Step 4.5: File-level deduplication
            dedup_start = time.time()
            fused_results = self._deduplicate_by_file(fused_results)
            dedup_time = (time.time() - dedup_start) * 1000
            if include_stats:
                stats['deduplication_ms'] = round(dedup_time, TIMING_PRECISION)
            
            # Step 5: Optional reranking
            rerank_time = None
            if self.reranker:
                rerank_start = time.time()
                try:
                    reranked = self.reranker(query, fused_results)
                    
                    # Blend rerank score with original similarity
                    # This prevents the reranker from completely overriding
                    # strong cosine matches with weak cross-encoder scores
                    for result in reranked:
                        meta = result.get('metadata', {})
                        rerank_score = meta.get('rerank_score', 0)
                        cosine_sim = result.get('similarity', 0)
                        rrf_score = meta.get('rrf_score', 0)
                        
                        # Normalize rerank score to 0-1 range (raw is roughly -10 to +10)
                        norm_rerank = 1.0 / (1.0 + 2.718 ** (-rerank_score))  # sigmoid
                        
                        # Blended score: 60% cosine + 15% rerank + 25% RRF position
                        # Cosine dominates — TinyBERT assists but doesn't override
                        blended = (0.6 * cosine_sim) + (0.15 * norm_rerank) + (0.25 * min(rrf_score * 30, 1.0))
                        meta['blended_score'] = round(blended, 4)
                    
                    # Sort by blended score
                    fused_results = sorted(reranked, key=lambda r: r.get('metadata', {}).get('blended_score', 0), reverse=True)
                    rerank_time = (time.time() - rerank_start) * 1000
                except Exception as e:
                    logger.warning(f"Reranking failed: {e}")
                    rerank_time = 0.0
                
                if include_stats:
                    stats['rerank_time_ms'] = round(rerank_time, TIMING_PRECISION)
            
            # Step 6: Limit results and finalize
            final_results = fused_results[:limit]
            total_search_time = (time.time() - start_time) * 1000
            
            response = {
                'results': final_results,
                'query': query,
                'total_results': len(final_results),
                'search_time_ms': round(total_search_time, TIMING_PRECISION),
                'variants_used': variants
            }
            
            if include_stats:
                response['stats'] = stats
            
            return response
            
        except SearchError:
            raise
        except Exception as e:
            total_time = (time.time() - start_time) * 1000
            raise SearchError(f"Search pipeline failed for query '{query}': {e}")
    
    def search_embedding(
        self,
        embedding: np.ndarray,
        limit: int = DEFAULT_SEARCH_LIMIT,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        include_stats: bool = False
    ) -> Dict[str, Any]:
        """
        Search with pre-computed embedding vector (bypasses variant generation).
        
        Args:
            embedding: Pre-computed query embedding vector
            limit: Maximum results to return
            threshold: Minimum similarity score threshold
            include_stats: Include performance statistics
            
        Returns:
            Same format as search() but without variant processing stats
        """
        start_time = time.time()
        
        # Validate parameters
        if not isinstance(embedding, np.ndarray):
            raise ValueError(f"embedding must be numpy array, got {type(embedding).__name__}")
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError(f"limit must be a positive integer, got {limit}")
        if limit > MAX_SEARCH_LIMIT:
            limit = MAX_SEARCH_LIMIT
        if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
            raise ValueError(f"threshold must be between 0 and 1, got {threshold}")
        
        # Check embedding dimensions if store has data
        store_stats = self.store.stats()
        if store_stats['dimensions'] and len(embedding) != store_stats['dimensions']:
            raise ValueError(
                f"Embedding dimension {len(embedding)} doesn't match store dimension {store_stats['dimensions']}"
            )
        
        stats = {} if include_stats else None
        
        try:
            # Normalize embedding for cosine similarity
            norm = np.linalg.norm(embedding)
            normalized_embedding = embedding / norm if norm > 0 else embedding
            
            # Search store directly
            search_start = time.time()
            results = self.store.search(normalized_embedding, limit=limit, min_similarity=threshold)
            search_time = (time.time() - search_start) * 1000
            
            if include_stats:
                stats['search_time_ms'] = round(search_time, TIMING_PRECISION)
            
            # Optional reranking
            if self.reranker:
                rerank_start = time.time()
                try:
                    # Can't rerank without original query string
                    logger.warning("Cannot rerank with pre-computed embedding (no query string)")
                    rerank_time = 0.0
                except Exception as e:
                    logger.warning(f"Reranking failed: {e}")
                    rerank_time = 0.0
                
                if include_stats:
                    stats['rerank_time_ms'] = round(rerank_time, TIMING_PRECISION)
            
            total_search_time = (time.time() - start_time) * 1000
            
            response = {
                'results': results,
                'query': '[embedding]',  # Placeholder since no original query
                'total_results': len(results),
                'search_time_ms': round(total_search_time, TIMING_PRECISION),
                'variants_used': []  # No variants for embedding search
            }
            
            if include_stats:
                response['stats'] = stats
            
            return response
            
        except Exception as e:
            total_time = (time.time() - start_time) * 1000
            raise SearchError(f"Embedding search failed: {e}")
    
    def _deduplicate_by_file(self, results: List[Dict]) -> List[Dict]:
        """
        Deduplicate results by file path, keeping only the highest-ranked result per file.
        
        Args:
            results: List of search results to deduplicate
            
        Returns:
            Deduplicated list with only the first (highest-ranked) result per unique file
        """
        if not results:
            return results
        
        seen_files = set()
        deduplicated = []
        
        for result in results:
            # Extract file path from metadata
            metadata = result.get('metadata', {})
            file_path = metadata.get('file_path', '')
            
            if not file_path:
                # If no file path, keep the result (shouldn't filter unknown files)
                deduplicated.append(result)
                continue
            
            # Handle chunk-based paths (format: "prefix::file_path::suffix")
            if '::' in file_path:
                parts = file_path.split('::')
                if len(parts) >= 3:
                    # Take the middle part as the actual file path
                    base_file = parts[1]
                else:
                    # Fallback: use the first part
                    base_file = parts[0]
            else:
                base_file = file_path
            
            # Keep only first occurrence of each file
            if base_file not in seen_files:
                seen_files.add(base_file)
                deduplicated.append(result)
        
        return deduplicated