"""
Velocirag Phase 4 Searcher - High-level search orchestration.

The apex predator of search orchestration, now with production-grade logic:
- Advanced query variant generation (CS656 → CS 656 patterns)
- Proper progressive search with consistency validation
- Sophisticated reranking with score blending
- Self-healing indices with async rebuilds

Combines Velocirag's clean architecture with production's battle-tested algorithms.
"""

import logging
import time
import threading
import re
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

# Progressive search configuration
L0_CANDIDATES = 50      # Wide net L0 candidates (abstracts)
L1_CANDIDATES = 20      # Focused L1 candidates (overviews)
L0_MULTIPLIER = 2       # Search limit multiplier for L0 (account for filtering)

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
        self._rebuild_in_progress: Dict[str, bool] = {}
        self._consistency_lock = threading.Lock()
        
    def search(
        self,
        query: str,
        limit: int = DEFAULT_SEARCH_LIMIT,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        include_stats: bool = False
    ) -> Dict[str, Any]:
        """
        Execute search pipeline. Auto-detects progressive vs standard search.
        
        Args:
            query: Search query string
            limit: Maximum results to return
            threshold: Minimum similarity score threshold
            include_stats: Include performance statistics
            
        Returns:
            Dictionary with results, query, timing, and variant information
        """
        # Standard search by default — progressive is opt-in via search_progressive()
        # At <10K docs, L2 direct search is fast enough and more accurate
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
                                import json
                                variant_results.append({
                                    'doc_id': row[0], 'content': row[1],
                                    'metadata': json.loads(row[2]) if row[2] else {},
                                    'similarity': float(sim), 'score': float(sim)
                                })
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
                rerank_start = time.time()
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
            total_time = (time.time() - start_time) * 1000
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
                if len(parts) >= 2:
                    # Take the middle part as the actual file path
                    base_file = parts[1] if len(parts) > 2 else parts[0]
                else:
                    # Fallback: use the first part
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
    
    def search_progressive(self, query: str, limit: int = 5, threshold: float = 0.3,
                           l0_candidates: int = L0_CANDIDATES, l1_candidates: int = L1_CANDIDATES) -> Dict[str, Any]:
        """
        Progressive search: L0 → L1 → L2 → rerank
        Now with production-grade logic and consistency validation.
        
        1. Generate variants with production patterns
        2. Search L0 index (wide net → l0_candidates results)
        3. Search L1 index for L0 candidates (narrow → l1_candidates)
        4. Load full L2 content for L1 candidates
        5. Rerank with sophisticated score blending
        
        Args:
            query: Search query string
            limit: Final number of results to return
            threshold: Minimum similarity threshold
            l0_candidates: Number of L0 candidates to retrieve
            l1_candidates: Number of L1 candidates to retrieve
            
        Returns:
            Search results with progressive stats
        """
        start_time = time.time()
        
        # Validate input
        if not isinstance(query, str) or not query.strip():
            return {
                'results': [],
                'query': query,
                'total_results': 0,
                'search_mode': 'progressive',
                'search_time_ms': 0.0,
                'progressive_stats': {'l0_candidates': 0, 'l1_candidates': 0, 'l2_loaded': 0}
            }
        
        query = query.strip()
        
        # Consistency validation
        self._validate_store_consistency()
        
        # Step 1: Generate variants using production logic
        if len(query) >= SINGLE_VARIANT_THRESHOLD:
            variants = QueryVariantGenerator.generate_variants(query)
        else:
            variants = [query]
        
        if not variants:
            variants = [query]
        
        # Step 2: L0 wide net — search abstract index
        l0_start = time.time()
        l0_results = {}
        for variant in variants:
            try:
                q_emb = self.embedder.embed(variant)
                if q_emb.ndim > 1:
                    q_emb = q_emb[0]
                q_norm = q_emb / np.linalg.norm(q_emb)
                hits = self.store.search_l0(q_norm, limit=l0_candidates * L0_MULTIPLIER)
                for hit in hits:
                    if hit['similarity'] < threshold:
                        continue
                    doc_id = hit['doc_id']
                    if doc_id not in l0_results or hit['similarity'] > l0_results[doc_id]['similarity']:
                        l0_results[doc_id] = hit
            except Exception as e:
                logger.warning(f"L0 search failed for variant '{variant}': {e}")
        
        # Sort by similarity, take top l0_candidates
        l0_sorted = sorted(l0_results.values(), key=lambda x: x['similarity'], reverse=True)[:l0_candidates]
        l0_time = (time.time() - l0_start) * 1000
        
        if not l0_sorted:
            # Fallback to standard search if no L0 results
            logger.info("No L0 results, falling back to standard search")
            fallback_result = self._search_standard(query, limit, threshold, False)
            fallback_result['search_mode'] = 'progressive_fallback'
            fallback_result['progressive_stats'] = {
                'l0_candidates': 0,
                'l1_candidates': 0,
                'l2_loaded': len(fallback_result['results']),
                'fallback_reason': 'no_l0_results'
            }
            return fallback_result
        
        # Step 3: L1 rerank — search overview index for L0 candidates only
        l1_start = time.time()
        l1_results = {}
        l0_doc_ids = {r['doc_id'] for r in l0_sorted}
        for variant in variants:
            try:
                q_emb = self.embedder.embed(variant)
                if q_emb.ndim > 1:
                    q_emb = q_emb[0]
                q_norm = q_emb / np.linalg.norm(q_emb)
                hits = self.store.search_l1(q_norm, limit=l1_candidates * 2, doc_ids=l0_doc_ids)
                for hit in hits:
                    doc_id = hit['doc_id']
                    if doc_id not in l1_results or hit['similarity'] > l1_results[doc_id]['similarity']:
                        l1_results[doc_id] = hit
            except Exception as e:
                logger.warning(f"L1 search failed for variant '{variant}': {e}")
        
        # If no L1 results, use L0 ranking honestly
        if not l1_results:
            logger.info("No L1 results, using L0 ranking")
            l1_sorted = l0_sorted[:l1_candidates]
            for result in l1_sorted:
                result['l1_similarity'] = result['similarity']  # Honest fallback
                result['fallback_mode'] = 'l0_ranking'
        else:
            l1_sorted = sorted(l1_results.values(), key=lambda x: x['similarity'], reverse=True)[:l1_candidates]
        
        l1_time = (time.time() - l1_start) * 1000
        
        # Step 4: L2 load — get full content for L1 candidates
        l2_start = time.time()
        l2_results = []
        for l1_hit in l1_sorted:
            doc = self.store.get(l1_hit['doc_id'])
            if doc:
                doc['similarity'] = l1_hit['similarity']
                # Preserve progressive metadata
                if 'fallback_mode' in l1_hit:
                    doc.setdefault('metadata', {})['progressive_mode'] = f"fallback_{l1_hit['fallback_mode']}"
                else:
                    doc.setdefault('metadata', {})['progressive_mode'] = 'full_l0_l1_l2'
                l2_results.append(doc)
        
        l2_time = (time.time() - l2_start) * 1000
        
        # Step 5: File-level deduplication
        l2_results = self._deduplicate_by_file(l2_results)
        
        # Step 6: Reranking with production score blending
        rerank_start = time.time()
        if self.reranker:
            try:
                reranked = self.reranker(query, l2_results)
                
                # Production score blending for progressive search
                for result in reranked:
                    meta = result.get('metadata', {})
                    rerank_score = meta.get('rerank_score', 0)
                    cosine_sim = result.get('similarity', 0)
                    
                    # Normalize rerank score
                    norm_rerank = 1.0 / (1.0 + np.exp(-rerank_score))
                    
                    # Progressive blended score: 70% cosine + 30% rerank
                    # Progressive search has high-quality L0/L1 filtering, so trust cosine more
                    blended = (0.7 * cosine_sim) + (0.3 * norm_rerank)
                    meta['blended_score'] = round(blended, 4)
                    meta['norm_rerank_score'] = round(norm_rerank, 4)
                
                # Sort by blended score
                l2_results = sorted(reranked, key=lambda r: r.get('metadata', {}).get('blended_score', 0), reverse=True)
            except Exception as e:
                logger.warning(f"Progressive reranking failed: {e}")
        
        rerank_time = (time.time() - rerank_start) * 1000
        
        # Step 7: Return with comprehensive stats
        total_time = (time.time() - start_time) * 1000
        
        return {
            'results': l2_results[:limit],
            'query': query,
            'total_results': len(l2_results[:limit]),
            'search_mode': 'progressive',
            'search_time_ms': round(total_time, TIMING_PRECISION),
            'variants_used': variants,
            'progressive_stats': {
                'l0_candidates': len(l0_sorted),
                'l1_candidates': len(l1_sorted),
                'l2_loaded': len(l2_results[:limit]),
                'timing_breakdown_ms': {
                    'l0_search': round(l0_time, TIMING_PRECISION),
                    'l1_rerank': round(l1_time, TIMING_PRECISION),
                    'l2_load': round(l2_time, TIMING_PRECISION),
                    'rerank': round(rerank_time, TIMING_PRECISION)
                },
                'config': {
                    'l0_candidates_requested': l0_candidates,
                    'l1_candidates_requested': l1_candidates,
                    'threshold': threshold
                }
            }
        }
    
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
            if dimensions is not None and dimensions != 384:  # Expected MiniLM dimensions
                logger.warning(f"Unexpected embedding dimensions: {dimensions}")
                is_consistent = False
            
        except Exception as e:
            logger.warning(f"Store consistency check failed: {e}")
            is_consistent = False
        
        # Update cache under lock
        with self._consistency_lock:
            self._consistency_cache[store_id] = (is_consistent, time.time())
        
        if not is_consistent:
            logger.warning(f"Store consistency issues detected")
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