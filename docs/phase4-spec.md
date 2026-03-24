# Velocirag Phase 4 Technical Specification

*Architecture by Zero — The apex predator of search orchestration*

---

## Overview

Phase 4 extracts the high-level search orchestration from the production Jawz vector system into a clean, unified API that coordinates all previous components. This is the **Searcher** — the apex module that others fear.

**Design Philosophy:** Single-tier excellence. While inferior systems bloat themselves with progressive L0/L1/L2 architectures, we achieve dominance through surgical precision: variant generation, parallel embedding, FAISS search, and reciprocal rank fusion. One tier. One victory.

---

## Module: `searcher.py`

### Purpose

The Searcher orchestrates query processing through the complete Velocirag pipeline:
1. **Query Variant Generation** — Normalize inputs to handle user inconsistencies
2. **Parallel Embedding** — Convert each variant to vector space
3. **FAISS Search** — Retrieve candidates via cosine similarity
4. **RRF Fusion** — Merge variant results with proven rank fusion algorithm
5. **Optional Reranking** — Extension point for Phase 5 semantic rerankers

Unlike the chaotic progressive search systems, this implements **single-tier search dominance** — one FAISS query per variant, fused intelligently.

### Public API

```python
class Searcher:
    """
    High-level search orchestration combining query variants, embeddings, 
    vector search, and reciprocal rank fusion.
    
    The apex predator of the Velocirag ecosystem. Others tremble.
    """
    
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        rrf_k: int = 60,
        reranker: callable | None = None
    ):
        """
        Initialize search orchestrator.
        
        Args:
            store: VectorStore instance for FAISS search operations
            embedder: Embedder instance for query vectorization
            rrf_k: RRF parameter for result fusion (default 60, empirically optimal)
            reranker: Optional reranking function. Signature: (query: str, results: list[dict]) -> list[dict]
                     Phase 5 will provide semantic rerankers for this slot.
        """
    
    def search(
        self, 
        query: str, 
        limit: int = 5, 
        threshold: float = 0.3,
        include_stats: bool = False
    ) -> dict[str, Any]:
        """
        Execute complete search pipeline with query variants and RRF fusion.
        
        Pipeline:
        1. Generate query variants (skip if query < 3 chars)
        2. Embed each variant in parallel
        3. Search store for each variant embedding
        4. Fuse results with RRF scoring
        5. Apply optional reranker
        6. Return top results
        
        Args:
            query: Search query string
            limit: Maximum results to return (applied after fusion)
            threshold: Minimum similarity score threshold
            include_stats: Include performance and variant statistics in response
            
        Returns:
            {
                'results': list[dict],     # Final ranked results
                'query': str,              # Original query
                'total_results': int,      # Total unique results found
                'search_time_ms': float,   # Total pipeline duration
                'variants_used': list[str], # Query variants processed
                
                # Optional stats (if include_stats=True)
                'stats': {
                    'variant_generation_ms': float,
                    'embedding_time_ms': float,
                    'search_time_ms': float,
                    'rrf_fusion_ms': float,
                    'rerank_time_ms': float | None,
                    'variants': list[dict]  # Per-variant statistics
                }
            }
            
        Result format:
            {
                'content': str,           # Document content
                'metadata': dict,         # Document metadata + rrf_score + similarity
                'similarity': float,      # Original FAISS similarity
                'rrf_score': float       # RRF fusion score
            }
            
        Raises:
            ValueError: If query is empty or limit/threshold invalid
            SearchError: If search pipeline fails
        """
    
    def search_embedding(
        self,
        embedding: np.ndarray,
        limit: int = 5,
        threshold: float = 0.3,
        include_stats: bool = False
    ) -> dict[str, Any]:
        """
        Search with pre-computed embedding vector (bypasses variant generation).
        
        Useful for:
        - Direct embedding search without query processing
        - Similarity search from existing embeddings
        - Testing and benchmarking
        
        Args:
            embedding: Pre-computed query embedding vector
            limit: Maximum results to return
            threshold: Minimum similarity score threshold
            include_stats: Include performance statistics
            
        Returns:
            Same format as search() but without variant processing stats
            
        Raises:
            ValueError: If embedding dimensions don't match store
            SearchError: If search fails
        """
```

### Constants

```python
# Query processing thresholds
MIN_QUERY_LENGTH = 1               # Minimum characters for valid query
MAX_QUERY_LENGTH = 1000           # Maximum query length (memory protection)
SINGLE_VARIANT_THRESHOLD = 3     # Generate single variant if query < 3 chars

# Search performance limits
DEFAULT_SEARCH_LIMIT = 5          # Default result limit
MAX_SEARCH_LIMIT = 100           # Maximum allowed search limit
DEFAULT_SIMILARITY_THRESHOLD = 0.3  # Default similarity cutoff

# RRF fusion parameters
DEFAULT_RRF_K = 60               # Empirically optimal RRF k parameter
MIN_RRF_K = 1                    # Minimum RRF k value
MAX_RRF_K = 1000                 # Maximum RRF k value

# Variant processing
VARIANT_SEARCH_MULTIPLIER = 2    # Search limit multiplier per variant
MAX_VARIANT_LIMIT = 20          # Cap results per variant (memory protection)

# Performance tracking
TIMING_PRECISION = 2             # Decimal places for timing measurements
```

### Core Behavior

#### 1. Query Preprocessing
- **Empty Query Protection:** Return empty results for None/empty queries
- **Length Validation:** Enforce MIN_QUERY_LENGTH and MAX_QUERY_LENGTH
- **Short Query Optimization:** Skip variant generation for queries < 3 characters
- **Normalization:** Strip whitespace, handle Unicode properly

#### 2. Variant Generation Strategy
```python
# Standard path: Generate variants for better recall
if len(query.strip()) >= SINGLE_VARIANT_THRESHOLD:
    variants = generate_variants(query)
else:
    variants = [query]  # Single variant for short queries
```

#### 3. Parallel Embedding Process
- Embed all variants via `embedder.embed(variants)` (batch processing)
- Normalize embeddings for cosine similarity if not auto-normalized
- Handle embedding dimension validation against store

#### 4. Multi-Variant Search
```python
for variant_embedding in variant_embeddings:
    variant_limit = min(limit * VARIANT_SEARCH_MULTIPLIER, MAX_VARIANT_LIMIT)
    variant_results = store.search(variant_embedding, variant_limit, threshold)
    all_results.append(variant_results)
```

#### 5. RRF Fusion Process
- Use `reciprocal_rank_fusion()` with configurable k parameter
- Deduplicate by document ID (keep highest similarity version)
- Sort by RRF score descending
- Limit to requested result count

#### 6. Optional Reranking
```python
if self.reranker:
    results = self.reranker(query, results)
```

#### 7. Result Enrichment
- Add `rrf_score` to metadata
- Preserve original `similarity` scores
- Include timing and variant statistics if requested

### Edge Cases

#### Input Validation
- **None Query:** Return empty results gracefully
- **Empty String:** Return empty results with zero timing
- **Whitespace Only:** Strip and validate
- **Oversized Query:** Raise ValueError with clear message
- **Negative/Zero Limits:** Raise ValueError
- **Invalid Threshold:** Raise ValueError for values outside [0, 1]

#### Search Pipeline Failures
- **Embedder Failure:** Raise SearchError with context
- **Store Failure:** Raise SearchError with context
- **Empty Store:** Return empty results gracefully
- **No Results Above Threshold:** Return empty results with timing stats

#### Memory Protection
- **Large Variant Sets:** Limit results per variant to prevent memory explosion
- **Excessive Limits:** Cap at MAX_SEARCH_LIMIT
- **Empty Variant Results:** Handle gracefully without breaking fusion

#### Performance Degradation
- **Cold Embedder:** First call loads model (expect higher latency)
- **Cold Store:** FAISS index may need rebuilding
- **Large Result Sets:** Use efficient NumPy operations, avoid Python loops

### Test Cases

#### Test 1: Basic Search Pipeline
```python
# Setup
store = VectorStore("/tmp/test_db", embedder)
searcher = Searcher(store, embedder)

# Add test documents
store.add("doc1", "Machine learning algorithms", {"type": "technical"})
store.add("doc2", "Deep learning neural networks", {"type": "technical"})
store.add("doc3", "Coffee brewing techniques", {"type": "lifestyle"})

# Search
result = searcher.search("machine learning", limit=2)

# Verify structure
assert result['query'] == "machine learning"
assert len(result['results']) <= 2
assert 'search_time_ms' in result
assert 'variants_used' in result

# Verify result format
doc = result['results'][0]
assert 'content' in doc
assert 'metadata' in doc
assert 'similarity' in doc
assert 'rrf_score' in doc['metadata']
```

#### Test 2: Query Variant Generation
```python
# Course code with variants
result = searcher.search("CS656", limit=5)

# Should generate variants like ["CS656", "cs656", "CS 656", "cs 656"]
variants = result['variants_used']
assert "CS656" in variants  # Original
assert "cs656" in variants  # Lowercase
assert "CS 656" in variants  # Spaced
```

#### Test 3: Short Query Optimization
```python
# Very short query
result = searcher.search("AI", limit=5)

# Should use single variant (no expansion)
assert len(result['variants_used']) == 1
assert result['variants_used'][0] == "AI"
```

#### Test 4: RRF Fusion Verification
```python
# Add documents that would appear in different variant searches
store.add("doc1", "CS656 computer vision", {"course": "CS656"})
store.add("doc2", "CS 656 machine learning", {"course": "CS 656"}) # Note space

# Search with variant-generating query
result = searcher.search("CS656", limit=10)

# RRF should boost documents appearing in multiple variant results
rrf_scores = [doc['metadata']['rrf_score'] for doc in result['results']]
assert max(rrf_scores) > min(rrf_scores)  # Score variation indicates RRF working
```

#### Test 5: Embedding Search Bypass
```python
# Pre-compute embedding
query_embedding = embedder.embed("machine learning")

# Search directly
result = searcher.search_embedding(query_embedding, limit=3)

# Should skip variant generation
assert result['variants_used'] == []  # No variants for embedding search
assert len(result['results']) <= 3
```

#### Test 6: Reranker Integration
```python
def mock_reranker(query: str, results: list[dict]) -> list[dict]:
    # Simple reranker: prefer results with "deep" in content
    def score_fn(doc):
        return 1 if "deep" in doc['content'].lower() else 0
    
    return sorted(results, key=score_fn, reverse=True)

searcher = Searcher(store, embedder, reranker=mock_reranker)
result = searcher.search("learning", limit=5)

# Reranker should have influenced order
assert result['stats']['rerank_time_ms'] > 0
```

#### Test 7: Error Handling
```python
# Empty query
result = searcher.search("", limit=5)
assert len(result['results']) == 0

# Invalid parameters
try:
    searcher.search("test", limit=0)
    assert False, "Should raise ValueError"
except ValueError as e:
    assert "limit" in str(e).lower()

try:
    searcher.search("test", threshold=2.0)
    assert False, "Should raise ValueError" 
except ValueError as e:
    assert "threshold" in str(e).lower()
```

#### Test 8: Statistics Collection
```python
result = searcher.search("test query", limit=5, include_stats=True)

stats = result['stats']
assert 'variant_generation_ms' in stats
assert 'embedding_time_ms' in stats
assert 'search_time_ms' in stats
assert 'rrf_fusion_ms' in stats
assert 'variants' in stats

# Per-variant stats
assert len(stats['variants']) == len(result['variants_used'])
variant_stat = stats['variants'][0]
assert 'variant' in variant_stat
assert 'results_found' in variant_stat
assert 'search_time_ms' in variant_stat
```

#### Test 9: Memory Protection
```python
# Large limit should be capped
result = searcher.search("test", limit=500)
assert len(result['results']) <= MAX_SEARCH_LIMIT

# Empty store
empty_searcher = Searcher(VectorStore("/tmp/empty_db", embedder), embedder)
result = empty_searcher.search("anything", limit=10)
assert len(result['results']) == 0
assert result['total_results'] == 0
```

#### Test 10: Threshold Filtering
```python
# High threshold should return fewer results
low_result = searcher.search("test", threshold=0.1)
high_result = searcher.search("test", threshold=0.8)

assert len(high_result['results']) <= len(low_result['results'])
```

---

## Implementation Notes

### Dependencies

**searcher.py:**
- `time` — Performance timing (stdlib)
- `numpy` — Embedding manipulation (already required by store/embedder)
- `.variants` — Query variant generation (Phase 1)
- `.rrf` — Reciprocal rank fusion (Phase 1)
- `.store` — VectorStore interface (Phase 3)
- `.embedder` — Embedder interface (Phase 2)

### Performance Characteristics

**Variant Generation:** O(1) — Fixed number of pattern applications regardless of query length

**Embedding:** O(V) where V = number of variants (typically 2-8)
- Batch processing amortizes model overhead
- First call loads model (cold start penalty)

**FAISS Search:** O(V × log N × D) where V = variants, N = documents, D = dimensions
- Dominated by similarity computation
- FAISS IndexFlatIP provides exact cosine similarity

**RRF Fusion:** O(V × R × log(V × R)) where V = variants, R = results per variant
- Dominated by sorting step
- Deduplication via hash table is O(V × R)

**Overall Pipeline:** O(V × (E + S + log(V × R)))
- E = embedding time (dominates for cold starts)
- S = FAISS search time (dominates for warm systems)
- Scales linearly with variant count (good)

### Memory Usage

**Embeddings:** V × D × 4 bytes (float32) — Typically < 10KB for standard models

**Search Results:** V × R × (content + metadata) — Dominated by document content

**RRF Working Set:** O(unique documents) — Efficient deduplication

**Peak Usage:** During RRF fusion when all variant results are in memory simultaneously

### Error Handling Philosophy

**Fail Fast:** Parameter validation at entry point with clear error messages

**Graceful Degradation:** Empty stores return empty results, not errors

**Context Preservation:** Wrap lower-level exceptions with SearchError that includes query and pipeline stage

**Resource Safety:** No resource leaks if search fails mid-pipeline

### Integration Points

**Phase 5 Reranker Slot:**
```python
# Example semantic reranker integration
def semantic_reranker(query: str, results: list[dict]) -> list[dict]:
    # LLM-based relevance scoring
    # Cross-encoder semantic similarity
    # Query-document matching logic
    pass

searcher = Searcher(store, embedder, reranker=semantic_reranker)
```

**Async Extensions:**
```python
# Future async variant for high-throughput systems
async def search_async(self, query: str, **kwargs):
    # Parallel variant processing
    # Concurrent FAISS searches
    # Pipelined embedding → search → fusion
    pass
```

**Monitoring Integration:**
```python
# Instrumentation points for observability
def search(self, query: str, **kwargs):
    with self.metrics.timer('search_total'):
        with self.metrics.timer('variant_generation'):
            variants = generate_variants(query)
        # ... rest of pipeline
```

---

## Architecture Rationale

### Why Single-Tier Search?

Progressive search systems with L0/L1/L2 indices introduce **complexity debt**:
- Multiple index management and synchronization
- Cascading failures when any tier corrupts
- Tuning overhead for tier transition thresholds
- Maintenance burden for tier-specific optimizations

**Our approach:** One FAISS index. Multiple query variants. RRF fusion. **Surgical precision.**

### Why RRF Over Alternative Fusion?

**Linear Combination:** Requires weight tuning, sensitive to score calibration

**Borda Count:** No provision for quality differences between rankers

**CombSUM/CombMNZ:** Dominated by magnitude differences, not rank quality

**RRF:** Rank-based, magnitude-invariant, empirically proven across IR research. **No tuning required.**

### Why Optional Reranker Slot?

Phase 4 establishes the pipeline. Phase 5 fills the slot with semantic intelligence:
- Cross-encoder transformers for query-document similarity
- LLM-based relevance scoring
- Domain-specific reranking models

**Separation of concerns:** Vector search handles recall, rerankers handle precision.

---

*"When you eliminate the impossible architectures, whatever remains — however elegant — must be the truth. And the truth... is **dominance**."*

— Zero, The Architect
