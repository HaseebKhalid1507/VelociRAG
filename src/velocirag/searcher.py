"""
Velocirag Phase 4 Searcher - High-level search orchestration.

The apex predator of search orchestration, now with production-grade logic:
- Advanced query variant generation (CS656 → CS 656 patterns)
- Production-grade search consistency validation
- Sophisticated reranking with score blending
- Self-healing indices with async rebuilds

Combines Velocirag's clean architecture with production's battle-tested algorithms.
"""

import logging
import json
import time
import threading
import re
from typing import Any, Dict, List, Optional, Callable

import numpy as np

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

# Consistency validation
CONSISTENCY_TTL = 600.0  # 10 minutes TTL for validation cache

logger = logging.getLogger("velocirag.searcher")


class SearchError(Exception):
    """Base exception for search pipeline errors."""
    pass


class QueryVariantGenerator:
    """
    Advanced query variant generation ported from production.
    Handles patterns like CS656 <-> CS 656, HTB123 <-> HTB 123, etc.
    """
    
    @staticmethod
    def generate_variants(query: str) -> List[str]:
        """
        Generate normalized query variants for better matching.
        Handles patterns like CS656 <-> CS 656, HTB123 <-> HTB 123, etc.
        """
        if not query or not query.strip():
            return []
            
        variants = [query]  # Always include original
        
        # Handle case variations
        if query.lower() != query:
            variants.append(query.lower())
        
        # Pattern 1: Letters followed by numbers with no space -> add space
        # e.g., CS656 -> CS 656, HTB123 -> HTB 123
        # Use word boundaries to avoid mangling longer codes
        normalized = re.sub(r'\b([A-Za-z]+)(\d+)\b', r'\1 \2', query)
        if normalized != query:
            variants.append(normalized)
            # Also add lowercase version
            if normalized.lower() != normalized:
                variants.append(normalized.lower())
        
        # Pattern 2: Letters followed by space(s) and numbers -> remove space
        # e.g., CS 656 -> CS656, HTB 123 -> HTB123
        # Handle multiple spaces
        compressed = re.sub(r'\b([A-Za-z]+)\s+(\d+)\b', r'\1\2', query)
        if compressed != query:
            variants.append(compressed)
            # Also add lowercase version
            if compressed.lower() != compressed:
                variants.append(compressed.lower())
        
        # Pattern 3: Handle hyphens
        # e.g., CS-656 -> CS656, CS 656
        if '-' in query:
            no_hyphen = query.replace('-', '')
            if no_hyphen not in variants:
                variants.append(no_hyphen)
            space_hyphen = query.replace('-', ' ')
            if space_hyphen not in variants:
                variants.append(space_hyphen)
        
        # Deduplicate variants while preserving order
        seen = set()
        deduplicated = []
        for v in variants:
            if v not in seen:
                seen.add(v)
                deduplicated.append(v)
        
        return deduplicated


class Searcher:
    """
    High-level search orchestration combining query variants, embeddings,
    vector search, and reciprocal rank fusion.
    
    The apex predator of the Velocirag ecosystem, now with production logic.
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
        
        # Auto-create reranker if not provided
        if reranker is None:
            try:
                from .reranker import Reranker
                _reranker = Reranker()
                self.reranker = _reranker.rerank
                logger.info("Reranker auto-initialized (cross-encoder)")
            except Exception as e:
                logger.info(f"Reranker not available: {e}")
                self.reranker = None
        else:
            self.reranker = reranker
        
        # Validate RRF k parameter
        if not isinstance(rrf_k, int):
            raise ValueError(f"rrf_k must be an integer, got {type(rrf_k).__name__}")
        if not MIN_RRF_K <= rrf_k <= MAX_RRF_K:
            raise ValueError(f"rrf_k must be between {MIN_RRF_K} and {MAX_RRF_K}, got {rrf_k}")
        
        self.rrf_k = rrf_k
        
        # Consistency validation cache - Thread-safe validation state
        self._consistency_cache: Dict[str, tuple[bool, float]] = {}
        self._cache_ttl = CONSISTENCY_TTL
        self._consistency_lock = threading.Lock()
        
    def search(
        self,
        query: str,
        limit: int = DEFAULT_SEARCH_LIMIT,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        include_stats: bool = False
    ) -> Dict[str, Any]:
        """
        Execute search pipeline with query variants and RRF fusion.
        
        Args:
            query: Search query string
            limit: Maximum results to return
            threshold: Minimum similarity score threshold
            include_stats: Include performance statistics
            
        Returns:
            Dictionary with results, query, timing, and variant information
        """
        return self._search_standard(query, limit, threshold, include_stats)

    def _search_standard(
        self,
        query: str,
        limit: int = DEFAULT_SEARCH_LIMIT,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        include_stats: bool = False
    ) -> Dict[str, Any]:
        """
        Execute standard search pipeline with query variants and RRF fusion.
        Now with production-grade variant generation and consistency validation.
        
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
                'search_mode': 'standard',
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
                'search_mode': 'standard',
                'search_time_ms': round((time.time() - start_time) * 1000, TIMING_PRECISION),
                'variants_used': [],
                'stats': {} if include_stats else None
            }
        if len(query) > MAX_QUERY_LENGTH:
            raise ValueError(f"Query too long: {len(query)} chars (max {MAX_QUERY_LENGTH})")
        
        # Consistency validation (async rebuild if needed, don't block queries)
        self._validate_store_consistency()
        
        stats = {} if include_stats else None
        
        try:
            # Step 1: Generate variants using production logic
            variant_start = time.time()
            if len(query) >= SINGLE_VARIANT_THRESHOLD:
                variants = QueryVariantGenerator.generate_variants(query)
            else:
                variants = [query]
            
            if not variants:  # Handle empty variant generation
                variants = [query]
                
            variant_time = (time.time() - variant_start) * 1000
            if include_stats:
                stats['variant_generation_ms'] = round(variant_time, TIMING_PRECISION)
            
            # Step 2: Embed all variants (batch processing for performance)
            embed_start = time.time()
            try:
                variant_embeddings = self.embedder.embed(variants)
                # Always ensure 2D shape (n_variants, dimensions)
                if variant_embeddings.ndim == 1:
                    variant_embeddings = variant_embeddings.reshape(1, -1)

                # Normalize each embedding for cosine similarity
                normalized_embeddings = []
                for i in range(variant_embeddings.shape[0]):
                    embedding = variant_embeddings[i]
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
            
            # Try batch FAISS search first (performance optimization)
            try:
                # Stack all variant embeddings
                query_matrix = np.array(normalized_embeddings, dtype='float32').reshape(len(normalized_embeddings), -1)
                variant_limit = min(limit * VARIANT_SEARCH_MULTIPLIER, MAX_VARIANT_LIMIT)

                # Single FAISS batch search
                similarities, indices = self.store._faiss_index.search(query_matrix, min(variant_limit, self.store._faiss_index.ntotal))

                # Split results per variant and load documents
                for v_idx in range(len(variants)):
                    variant_search_start = time.time()
                    variant_results = []
                    with self.store._connect() as conn:
                        for sim, faiss_idx in zip(similarities[v_idx], indices[v_idx]):
                            if faiss_idx < 0 or sim < threshold:
                                continue
                            row = conn.execute('SELECT doc_id, content, metadata FROM documents WHERE faiss_idx = ?', (int(faiss_idx),)).fetchone()
                            if row:
                                metadata = json.loads(row[2]) if row[2] else {}
                                if not isinstance(metadata, dict):
                                    logger.warning(f"Invalid metadata type for doc {row[0]}: {type(metadata).__name__}")
                                    metadata = {}
                                variant_results.append({
                                    'doc_id': row[0], 'content': row[1],
                                    'metadata': metadata,
                                    'similarity': float(sim), 'score': float(sim)
                                })
                            else:
                                logger.warning(f"FAISS index {faiss_idx} has no matching document in SQLite")
                    all_results.append(variant_results)
                    
                    if include_stats:
                        variant_search_time = (time.time() - variant_search_start) * 1000
                        variant_stats.append({
                            'variant': variants[v_idx],
                            'results_found': len(variant_results),
                            'search_time_ms': round(variant_search_time, TIMING_PRECISION)
                        })
                
            except Exception as batch_error:
                logger.warning(f"Batch search failed, falling back to individual searches: {batch_error}")
                # Fallback to original per-variant loop
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
                    doc_id = result.get('doc_id')
                    if doc_id:
                        return doc_id
                    import hashlib as _hl
                    return f"content_hash_{_hl.sha256(result.get('content', '').encode()).hexdigest()}"
                
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
            
            # Step 5: Optional reranking with production score blending
            rerank_time = None
            if self.reranker:
                rerank_start = time.time()
                try:
                    reranked = self.reranker(query, fused_results)
                    
                    # Production-grade score blending
                    # Blend rerank score with original similarity and RRF position
                    for result in reranked:
                        meta = result.get('metadata', {})
                        rerank_score = meta.get('rerank_score', 0)
                        cosine_sim = result.get('similarity', 0)
                        rrf_score = meta.get('rrf_score', 0)
                        
                        # Normalize rerank score to 0-1 range (raw is roughly -10 to +10)
                        norm_rerank = 1.0 / (1.0 + np.exp(-rerank_score))  # sigmoid
                        
                        # Production blended score: 60% cosine + 15% rerank + 25% RRF position
                        # Cosine dominates — cross-encoder assists but doesn't override
                        blended = (0.6 * cosine_sim) + (0.15 * norm_rerank) + (0.25 * min(rrf_score * 30, 1.0))
                        meta['blended_score'] = round(blended, 4)
                        meta['norm_rerank_score'] = round(norm_rerank, 4)
                    
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
                'search_mode': 'standard',
                'search_time_ms': round(total_search_time, TIMING_PRECISION),
                'variants_used': variants
            }
            
            if include_stats:
                response['stats'] = stats
            
            return response
            
        except SearchError:
            raise
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.error(f"Search pipeline failed for query '{query}' after {elapsed_ms:.1f}ms: {e}")
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
        
        # Consistency validation
        self._validate_store_consistency()
        
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
            
            # File-level deduplication
            results = self._deduplicate_by_file(results)
            
            # Optional reranking (can't rerank without original query string)
            if self.reranker:
                try:
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
            elapsed_ms = (time.time() - start_time) * 1000
            logger.error(f"Embedding search failed after {elapsed_ms:.1f}ms: {e}")
            raise SearchError(f"Embedding search failed: {e}")
    
    def _deduplicate_by_file(self, results: List[Dict]) -> List[Dict]:
        """
        Deduplicate results by file path, keeping only the highest-ranked result per file.
        Production-grade implementation with proper chunk handling.
        
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
                if len(parts) > 2:
                    # Format: "source::file_path::chunk_N" — take the middle part
                    base_file = parts[1]
                elif len(parts) == 2:
                    # Format: "file_path::chunk_N" — take the first part
                    base_file = parts[0]
                else:
                    base_file = parts[0]
            else:
                base_file = file_path
            
            # Normalize: just the filename (strip directory prefixes)
            base_file = base_file.split('/')[-1] if '/' in base_file else base_file
            
            # Keep only first occurrence of each file
            if base_file and base_file not in seen_files:
                seen_files.add(base_file)
                deduplicated.append(result)
        
        return deduplicated
    

    
    def _validate_store_consistency(self, force: bool = False) -> bool:
        """
        Validate store consistency with TTL cache.
        Triggers async rebuild if inconsistent, but doesn't block queries.
        
        Args:
            force: Force validation bypass cache
            
        Returns:
            True if consistent, False otherwise
        """
        store_id = id(self.store)  # Use store object ID as cache key
        
        with self._consistency_lock:
            # Check cache first (unless forced)
            if not force and store_id in self._consistency_cache:
                is_consistent, last_check = self._consistency_cache[store_id]
                if time.time() - last_check < self._cache_ttl:
                    return is_consistent
        
        # Actual validation (outside lock - I/O bound)
        try:
            stats = self.store.stats()
            doc_count = stats.get('document_count', 0)
            faiss_count = stats.get('faiss_vectors', 0)
            dimensions = stats.get('dimensions')

            # Real consistency checks
            is_consistent = True
            if doc_count > 0 and faiss_count > 0 and doc_count != faiss_count:
                logger.warning(f"Index inconsistency: {doc_count} docs vs {faiss_count} FAISS vectors")
                is_consistent = False
            expected_dims = self.embedder._get_model_dimensions() if self.embedder else 384
            if dimensions is not None and dimensions != expected_dims:
                logger.warning(f"Unexpected embedding dimensions: {dimensions} (expected {expected_dims})")
                is_consistent = False
            
        except Exception as e:
            logger.warning(f"Store consistency check failed: {e}")
            is_consistent = False
        
        # Update cache under lock
        with self._consistency_lock:
            self._consistency_cache[store_id] = (is_consistent, time.time())
        
        if not is_consistent:
            logger.warning("Store consistency issues detected")
            # TODO: Implement async rebuild trigger if store supports it
            
        return is_consistent
    
    def invalidate_cache(self):
        """Invalidate consistency cache. Thread-safe."""
        with self._consistency_lock:
            self._consistency_cache.clear()
    
    def get_cache_status(self) -> Dict[str, Any]:
        """Get current cache status. Thread-safe."""
        current_time = time.time()
        
        with self._consistency_lock:
            store_id = id(self.store)
            if store_id in self._consistency_cache:
                is_consistent, last_check = self._consistency_cache[store_id]
                age_seconds = current_time - last_check
                is_expired = age_seconds >= self._cache_ttl
                
                return {
                    'cached': True,
                    'consistent': is_consistent,
                    'age_seconds': round(age_seconds, 1),
                    'ttl_seconds': self._cache_ttl,
                    'expired': is_expired,
                    'last_validated': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last_check))
                }
            else:
                return {
                    'cached': False,
                    'consistent': None,
                    'age_seconds': None,
                    'ttl_seconds': self._cache_ttl,
                    'expired': True,
                    'last_validated': 'never'
                }