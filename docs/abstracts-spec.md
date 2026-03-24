# Velocirag L0/L1 Abstract Generator
## Technical Specification

*A centroid-based extractive summarization system using existing MiniLM embeddings.*

---

## Executive Summary

The L0/L1 Abstract Generator implements hierarchical extractive summarization to enable progressive search in Velocirag. Rather than generating new text, it identifies the most representative sentences from each document chunk using centroid-based cosine similarity scoring.

**Key Architectural Principles:**
- **No new models**: Uses existing MiniLM embedder — zero additional memory footprint
- **Extractive, not generative**: Pulls sentences directly from source content 
- **Thread-safe**: Leverages embedder's existing concurrency controls
- **Integrated**: Seamlessly extends store schema and searcher pipeline

---

## System Architecture

### Algorithm: Centroid-Based Representative Sentence Extraction

```
Input: Document chunk content (raw text)
Output: L0 abstract (1 sentence), L1 overview (3-5 sentences)

Process:
1. Split text into sentences using robust regex patterns
2. Filter sentences: minimum 10 characters, exclude headers/metadata
3. Embed all sentences using existing Embedder (batch for efficiency)
4. Compute centroid vector = mean(all_sentence_embeddings)
5. Score each sentence = cosine_similarity(sentence_embedding, centroid)
6. L0 = highest-scoring sentence (most representative of entire chunk)
7. L1 = top N highest-scoring sentences, reordered by original position
8. Generate embeddings for L0 and L1 abstracts themselves
```

### Mathematical Foundation

```python
# Given sentences S = [s1, s2, ..., sn] with embeddings E = [e1, e2, ..., en]
centroid = (1/n) * Σ(ei) for i in 1..n

# Score each sentence against the centroid
score_i = cosine_similarity(ei, centroid) = (ei · centroid) / (||ei|| * ||centroid||)

# Select representatives
L0 = sentence with max(score_i)
L1 = top_k sentences by score, reordered by original position
```

---

## Module Design: `src/velocirag/abstracts.py`

### Core Classes

```python
class SentenceSplitter:
    """Robust regex-based sentence segmentation with abbreviation handling."""
    
    def __init__(self):
        """Initialize with regex patterns for English text."""
    
    def split(self, text: str) -> List[str]:
        """Split text into sentences, handling abbreviations, URLs, numbers."""
    
    def _filter_sentence(self, sentence: str) -> bool:
        """Filter out headers, metadata, very short sentences."""

class AbstractGenerator:
    """Centroid-based extractive summarization using existing embedder."""
    
    def __init__(self, embedder: Embedder):
        """Initialize with existing embedder instance."""
        self.embedder = embedder
        self.splitter = SentenceSplitter()
    
    def generate(self, content: str, l0_sentences: int = 1, 
                l1_sentences: int = 3) -> AbstractResult:
        """Generate L0 and L1 abstracts for single chunk."""
    
    def generate_batch(self, contents: List[str], 
                      l0_sentences: int = 1, 
                      l1_sentences: int = 3) -> List[AbstractResult]:
        """Efficient batch generation for bulk indexing."""
    
    def _compute_centroid_scores(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity scores against centroid."""
    
    def _select_representatives(self, sentences: List[str], 
                              scores: np.ndarray, 
                              l0_count: int, 
                              l1_count: int) -> Tuple[str, str]:
        """Select and format L0/L1 abstracts maintaining original order."""

@dataclass
class AbstractResult:
    """Container for abstract generation results."""
    l0_abstract: str          # Single most representative sentence
    l1_overview: str          # 3-5 sentences preserving original order
    l0_embedding: np.ndarray  # Embedding of L0 abstract
    l1_embedding: np.ndarray  # Embedding of L1 overview
    original_sentences: int   # Count of sentences in source
    generation_time_ms: float # Performance tracking
```

### Sentence Splitting Algorithm

```python
class SentenceSplitter:
    ABBREVIATIONS = {
        'Mr.', 'Mrs.', 'Ms.', 'Dr.', 'Prof.', 'Sr.', 'Jr.',
        'vs.', 'etc.', 'e.g.', 'i.e.', 'Fig.', 'Vol.',
        'No.', 'pp.', 'Ph.D.', 'M.D.', 'B.A.', 'M.A.',
        'U.S.', 'U.K.', 'N.Y.', 'L.A.'
    }
    
    # Regex patterns
    SENTENCE_ENDINGS = r'[.!?]+'
    QUOTATION_MARKS = r'["\']'
    PARENTHESES = r'[)\]]'
    
    def split(self, text: str) -> List[str]:
        # Handle abbreviations by temporary replacement
        # Split on sentence boundaries
        # Restore abbreviations
        # Filter short/invalid sentences
        # Return clean sentence list
```

---

## Store Integration

### Schema Extensions

New columns added to existing `documents` table:

```sql
-- L0/L1 abstract storage
ALTER TABLE documents ADD COLUMN l0_abstract TEXT;
ALTER TABLE documents ADD COLUMN l1_overview TEXT;
ALTER TABLE documents ADD COLUMN l0_embedding BLOB;
ALTER TABLE documents ADD COLUMN l1_embedding BLOB;
ALTER TABLE documents ADD COLUMN abstract_generated TIMESTAMP;

-- Performance tracking
ALTER TABLE documents ADD COLUMN sentence_count INTEGER;
ALTER TABLE documents ADD COLUMN generation_time_ms REAL;
```

### FAISS Index Extensions

New specialized indices for progressive search:

```
store/
├── index.faiss          # Existing L2 (full chunk) index
├── index_l0.faiss       # L0 abstract index (1 vector per document)
├── index_l1.faiss       # L1 overview index (1 vector per document)
└── store.db             # SQLite with extended schema
```

### Store API Extensions

```python
class VectorStore:
    def add_with_abstracts(self, documents: List[Dict]) -> None:
        """Add documents with automatic abstract generation."""
    
    def regenerate_abstracts(self, doc_filter: Optional[str] = None) -> Stats:
        """Regenerate abstracts for existing documents."""
    
    def get_abstract_stats(self) -> Dict[str, Any]:
        """Get statistics on abstract generation coverage."""
    
    def rebuild_abstract_indices(self) -> None:
        """Rebuild L0/L1 FAISS indices from SQLite embeddings."""
```

---

## Searcher Integration: Progressive Search

### Three-Tier Search Strategy

When L0/L1 indices exist, searcher implements progressive filtering:

```python
# Progressive search pipeline
def progressive_search(query: str, limit: int = 5) -> List[Dict]:
    """
    L0 → L1 → L2 progressive refinement for optimal performance.
    
    1. Search L0 index (1 vector per document) → top 50 candidates
    2. Search L1 index for those 50 documents → top 20 candidates  
    3. Load full L2 chunks for those 20 documents → rerank → final 5
    """
    
    # Step 1: L0 abstract search (coarse filter)
    l0_candidates = store.search_l0(query_embedding, limit=50)
    candidate_doc_ids = [r['doc_id'] for r in l0_candidates]
    
    # Step 2: L1 overview search (medium filter)  
    l1_results = store.search_l1(query_embedding, 
                                 doc_id_filter=candidate_doc_ids, 
                                 limit=20)
    
    # Step 3: Full L2 chunk search (fine filter + rerank)
    final_doc_ids = [r['doc_id'] for r in l1_results]
    l2_results = store.search_l2(query_embedding,
                                 doc_id_filter=final_doc_ids,
                                 limit=limit)
    
    return l2_results
```

### Performance Benefits

- **L0 search**: 1 vector per document (3000 docs → 3000 vectors)
- **L1 search**: Limited to 50 candidates (significant speedup)
- **L2 search**: Limited to 20 candidates (10x faster than full search)

---

## Implementation Details

### Edge Case Handling

```python
def generate_abstracts(self, content: str) -> AbstractResult:
    sentences = self.splitter.split(content)
    
    # Edge case: Very short content
    if len(sentences) <= 2:
        # L0 = L1 = full content (no summarization needed)
        return AbstractResult(
            l0_abstract=content.strip(),
            l1_overview=content.strip(),
            l0_embedding=self.embedder.embed(content),
            l1_embedding=self.embedder.embed(content),
            original_sentences=len(sentences)
        )
    
    # Edge case: Single meaningful sentence after filtering
    valid_sentences = [s for s in sentences if self._is_valid_sentence(s)]
    if len(valid_sentences) == 1:
        sentence = valid_sentences[0]
        return AbstractResult(
            l0_abstract=sentence,
            l1_overview=sentence,
            l0_embedding=self.embedder.embed(sentence),
            l1_embedding=self.embedder.embed(sentence),
            original_sentences=len(sentences)
        )
    
    # Normal case: Multiple sentences available
    # ... centroid-based selection
```

### Sentence Validation

```python
def _is_valid_sentence(self, sentence: str) -> bool:
    """Filter out non-content sentences."""
    sentence = sentence.strip()
    
    # Minimum length check
    if len(sentence) < 10:
        return False
    
    # Header detection (markdown)
    if sentence.startswith('#'):
        return False
    
    # Metadata detection
    if sentence.startswith('---') or sentence.startswith('```'):
        return False
    
    # URL-only lines
    if sentence.startswith('http') and ' ' not in sentence:
        return False
    
    # Must contain at least one alphabetic character
    if not any(c.isalpha() for c in sentence):
        return False
    
    return True
```

### Batch Processing Optimization

```python
def generate_batch(self, contents: List[str]) -> List[AbstractResult]:
    """Optimized batch processing with minimal embedding calls."""
    
    all_sentences = []
    sentence_to_doc_map = []
    doc_sentence_counts = []
    
    # Step 1: Collect all sentences from all documents
    for doc_idx, content in enumerate(contents):
        sentences = self.splitter.split(content)
        valid_sentences = [s for s in sentences if self._is_valid_sentence(s)]
        
        for sentence in valid_sentences:
            all_sentences.append(sentence)
            sentence_to_doc_map.append(doc_idx)
        
        doc_sentence_counts.append(len(valid_sentences))
    
    # Step 2: Single batch embedding call for ALL sentences
    all_embeddings = self.embedder.embed(all_sentences)  # One call for everything
    
    # Step 3: Distribute embeddings back to documents and process
    results = []
    embedding_idx = 0
    
    for doc_idx, content in enumerate(contents):
        sentence_count = doc_sentence_counts[doc_idx]
        
        if sentence_count == 0:
            # Handle empty documents
            results.append(self._empty_result(content))
            continue
        
        # Extract this document's embeddings
        doc_embeddings = all_embeddings[embedding_idx:embedding_idx + sentence_count]
        doc_sentences = all_sentences[embedding_idx:embedding_idx + sentence_count]
        
        # Process this document
        result = self._process_document_embeddings(doc_sentences, doc_embeddings)
        results.append(result)
        
        embedding_idx += sentence_count
    
    return results
```

---

## Performance Specifications

### Target Metrics

- **Single chunk processing**: < 50ms (including embedding time)
- **Batch processing (3,000 chunks)**: < 60 seconds total
- **Memory overhead**: Zero (reuses existing embedder)
- **Index rebuild time**: < 30 seconds for 3,000 documents

### Optimization Strategies

1. **Batch Embedding**: Process all sentences from multiple documents in single embedder call
2. **Sentence Reuse**: Cache sentence embeddings to avoid recomputation  
3. **Progressive Processing**: Process chunks in batches during indexing
4. **Lazy Index Builds**: Only rebuild FAISS indices when needed

### Benchmarking Framework

```python
class AbstractBenchmark:
    def benchmark_single(self, content: str) -> Dict[str, float]:
        """Benchmark single chunk processing."""
    
    def benchmark_batch(self, contents: List[str]) -> Dict[str, Any]:
        """Benchmark batch processing with detailed timing."""
    
    def benchmark_progressive_search(self, queries: List[str]) -> Dict[str, Any]:
        """Compare progressive vs full search performance."""
```

---

## Migration Strategy

### Phase 1: Implementation (Week 1)
- Implement `SentenceSplitter` with comprehensive regex patterns
- Implement `AbstractGenerator` with centroid algorithm
- Add unit tests for edge cases and performance

### Phase 2: Integration (Week 2)  
- Extend `VectorStore` schema and FAISS indices
- Implement batch abstract generation pipeline
- Add CLI tool for bulk abstract generation

### Phase 3: Progressive Search (Week 3)
- Extend `Searcher` with L0/L1 progressive pipeline
- Benchmark progressive vs traditional search
- Optimize performance based on real workload

### Phase 4: Production (Week 4)
- Migration scripts for existing document collections
- Monitoring and alerting for abstract generation
- Documentation and operator training

---

## Testing Strategy

### Unit Tests

```python
class TestSentenceSplitter:
    def test_basic_splitting(self):
        """Test standard sentence boundaries."""
    
    def test_abbreviation_handling(self):
        """Ensure abbreviations don't break sentences."""
    
    def test_edge_cases(self):
        """Test URLs, numbers, special characters."""

class TestAbstractGenerator:
    def test_centroid_computation(self):
        """Verify centroid calculation accuracy."""
    
    def test_score_ranking(self):
        """Ensure highest-scoring sentences are selected."""
    
    def test_order_preservation(self):
        """Verify L1 maintains original sentence order."""

class TestProgressiveSearch:
    def test_filtering_accuracy(self):
        """Ensure progressive search doesn't miss relevant results."""
    
    def test_performance_improvement(self):
        """Benchmark progressive vs full search."""
```

### Integration Tests

```python
def test_full_pipeline():
    """End-to-end test of document ingestion with abstracts."""
    
def test_migration_compatibility():
    """Ensure existing documents work with new abstract system."""
    
def test_concurrent_generation():
    """Verify thread safety during parallel processing."""
```

---

## Risk Assessment

### Technical Risks

1. **Sentence Splitting Accuracy**: Regex may fail on edge cases
   - *Mitigation*: Comprehensive test suite with real-world content
   
2. **Centroid Representativeness**: Mathematical assumption may not hold
   - *Mitigation*: A/B testing against manual human summaries
   
3. **Performance Degradation**: Additional processing overhead
   - *Mitigation*: Extensive benchmarking and optimization

### Operational Risks

1. **Index Corruption**: Multiple FAISS indices increase failure surface
   - *Mitigation*: Atomic rebuild operations and backup strategies
   
2. **Storage Growth**: 3x FAISS indices (L0, L1, L2)  
   - *Mitigation*: Index compression and retention policies

---

## Future Enhancements

### Advanced Sentence Selection

```python
# Replace simple centroid with TF-IDF weighted centroid
def compute_weighted_centroid(self, sentences: List[str], 
                             embeddings: np.ndarray) -> np.ndarray:
    """Weight sentences by TF-IDF scores before centroid computation."""
    
# Multi-objective optimization
def select_diverse_representatives(self, sentences: List[str],
                                 embeddings: np.ndarray,
                                 similarity_weight: float = 0.7,
                                 diversity_weight: float = 0.3) -> List[int]:
    """Balance representativeness with diversity in sentence selection."""
```

### Query-Aware Abstracts

```python
def generate_query_focused_abstract(self, content: str, 
                                  query_embedding: np.ndarray) -> str:
    """Generate abstracts biased toward specific query focus."""
```

### Hierarchical Document Structure

```python
def generate_hierarchical_abstracts(self, document: str) -> Dict[str, str]:
    """
    Generate abstracts at multiple levels:
    - Section abstracts (L0.5)
    - Chapter abstracts (L1.5)  
    - Document abstract (L0)
    """
```

---

## Conclusion

The L0/L1 Abstract Generator represents the culmination of extractive summarization principles applied to production vector search. By leveraging existing infrastructure and mathematical elegance, it delivers progressive search capabilities without architectural complexity.

This system transforms Velocirag from a brute-force vector search into an intelligent, hierarchical retrieval system worthy of its predatory namesake.

*"In the end, there are no shortcuts to dominance — only better algorithms."* — Zero

---

**Document Version**: 1.0  
**Author**: Zero (Architect)  
**Date**: 2026-03-24  
**Status**: Specification Complete — Ready for Implementation
