# Velocirag Phase 2 Technical Specification

*Architecture by Zero — Embedding supremacy through calculated caching*

---

## Overview

Phase 2 introduces the **Embedder** — a production-hardened embedding module extracted from the Jawz vector search system. This is not another toy wrapper around sentence-transformers. This is a *weapon* forged through 10,000+ real-world embedding operations.

**Design Principle:** Intelligence through caching. Every text that enters this system is fingerprinted, embedded once, and remembered forever. Subsequent identical requests are served from cache at near-zero latency. The model loads lazily — only when absolutely necessary. 

---

## Module: `embedder.py`

### Purpose
Production-grade embedding engine with intelligent caching, lazy loading, and batch processing. Wraps sentence-transformers models with MD5-based content caching and LRU eviction. Designed for high-throughput vector search systems where the same texts are embedded repeatedly.

### Public API

```python
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
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: str | None = None,
        cache_size: int = 10000,
        normalize: bool = False
    ):
        """
        Initialize embedding engine.
        
        Args:
            model_name: HuggingFace model identifier (default: all-MiniLM-L6-v2)
            cache_dir: Directory for persistent cache storage. None = memory-only cache.
            cache_size: Maximum cache entries before LRU eviction (default: 10000)
            normalize: Whether to L2-normalize embeddings (default: False)
        """
    
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
            
        Cache Strategy:
            1. Hash each text with MD5
            2. Check cache for existing embeddings
            3. Compute missing embeddings in batch
            4. Update cache with new embeddings
            5. Apply LRU eviction if cache exceeds limit
        """
    
    def normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """
        L2-normalize embeddings for cosine similarity search.
        
        Args:
            embeddings: Raw embeddings array
            
        Returns:
            L2-normalized embeddings (unit vectors)
        """
    
    def get_model_info(self) -> dict[str, Any]:
        """
        Get model metadata and cache statistics.
        
        Returns:
            Dictionary with model and cache info:
            {
                'model_name': str,           # HuggingFace model identifier
                'dimensions': int,           # Embedding dimensions
                'max_seq_length': int,       # Model's maximum sequence length
                'cache_size': int,           # Current cache entries
                'cache_max_size': int,       # Cache size limit
                'model_loaded': bool,        # Whether model is loaded in memory
                'cache_dir': str | None,     # Cache directory path
                'normalize_embeddings': bool # Whether normalization is enabled
            }
        """
    
    def clear_cache(self) -> None:
        """
        Clear all cached embeddings (memory and disk).
        Model remains loaded if already initialized.
        """
    
    def save_cache(self) -> None:
        """
        Force immediate cache persistence to disk.
        Normally called automatically during cleanup.
        """
    
    def cleanup(self) -> None:
        """
        Persist cache to disk and cleanup resources.
        Called automatically via atexit hook.
        """
```

### Constants

```python
DEFAULT_MODEL = "all-MiniLM-L6-v2"     # Default sentence-transformers model
DEFAULT_CACHE_SIZE = 10000             # Default cache entries before LRU eviction
MIN_CACHE_SIZE = 100                   # Minimum allowed cache size
MAX_CACHE_SIZE = 100000                # Maximum allowed cache size
CACHE_SAVE_INTERVAL = 200              # Save cache every N new entries
CACHE_FILENAME = "embedding_cache.json" # Persistent cache file name
```

### Core Behavior

#### 1. Lazy Model Loading
- Model is NOT loaded during `__init__()` 
- First call to `embed()` triggers model initialization
- Heavy imports (`sentence_transformers`) deferred until needed
- Warning suppression applied during model load:
  - `HF_HUB_DISABLE_PROGRESS_BARS = '1'`
  - `TOKENIZERS_PARALLELISM = 'false'`  
  - `HF_HUB_DISABLE_TELEMETRY = '1'`
  - Suppress transformers, torch, and urllib3 warnings

#### 2. Content Hash Caching
- Each text hashed with MD5 before embedding lookup
- Cache key format: MD5 hexdigest (32 characters)
- Cache stores embedding as list (JSON-serializable)
- Cache miss triggers model computation
- Batch processing: compute all missing embeddings in single model call

#### 3. LRU Cache Management  
- Track access order for LRU eviction
- When cache exceeds `cache_size`, remove oldest accessed entries
- New entries added to end of access order
- Cache hits move entry to end (most recently used)

#### 4. Persistent Cache Storage
- Cache automatically loaded from disk during initialization
- Atomic writes via temp file + `os.replace()` pattern
- Periodic saves every 200 new cache entries
- Forced save during cleanup/atexit
- Graceful error handling for disk I/O failures

#### 5. Batch Optimization
- Single texts wrapped in list for uniform processing
- Missing embeddings computed in single model.encode() call
- Results distributed back to original positions
- Maintains input order in output array

#### 6. Optional Normalization
- L2 normalization for cosine similarity search
- Applied after embedding computation (before caching if enabled)
- Separate `normalize()` method for manual normalization

### Edge Cases

- **Empty Input:** `embed("")` returns zero vector of correct dimensions
- **Empty List:** `embed([])` returns empty array with shape `(0, dimensions)`
- **Mixed Empty/Valid:** Handles list with empty strings gracefully
- **Very Long Text:** Respects model's max_seq_length (typically 256-512 tokens)
- **Cache Directory Creation:** Creates parent directories automatically
- **Corrupted Cache File:** Falls back to empty cache without crashing
- **Disk Full:** Cache saves fail gracefully with warning message
- **Model Download Failure:** Propagates clear error message to caller
- **Invalid Model Name:** sentence_transformers error bubbled up with context

### Test Cases

```python
# Test 1: Basic embedding - single text
embedder = Embedder()
result = embedder.embed("hello world")
assert result.shape == (384,)  # all-MiniLM-L6-v2 dimensions
assert isinstance(result, np.ndarray)

# Test 2: Batch embedding - multiple texts  
texts = ["hello world", "goodbye world", "hello universe"]
result = embedder.embed(texts)
assert result.shape == (3, 384)
assert len(result) == 3

# Test 3: Cache hit optimization
embedder = Embedder(cache_dir="/tmp/test_cache")
text = "cache test text"

# First call - cache miss, model loads and computes
start = time.time()
result1 = embedder.embed(text)
first_duration = time.time() - start

# Second call - cache hit, should be much faster
start = time.time() 
result2 = embedder.embed(text)
second_duration = time.time() - start

assert np.allclose(result1, result2)  # Same embedding
assert second_duration < first_duration * 0.1  # >90% faster

# Test 4: Mixed cache hit/miss in batch
embedder = Embedder()
embedder.embed("cached text")  # Prime cache

batch = ["cached text", "new text", "another new text"]
result = embedder.embed(batch)
assert result.shape == (3, 384)

# Verify cache contains all three texts
info = embedder.get_model_info()
assert info['cache_size'] == 3

# Test 5: Cache persistence across instances
cache_dir = "/tmp/test_persistence"
text = "persistent test"

# First instance - embed and save
embedder1 = Embedder(cache_dir=cache_dir)
result1 = embedder1.embed(text)
embedder1.cleanup()

# Second instance - should load from cache
embedder2 = Embedder(cache_dir=cache_dir) 
info = embedder2.get_model_info()
assert info['cache_size'] == 1  # Cache loaded from disk
assert not info['model_loaded']  # Model not loaded yet

result2 = embedder2.embed(text)
assert np.allclose(result1, result2)

# Test 6: LRU eviction
embedder = Embedder(cache_size=3)
texts = ["text1", "text2", "text3", "text4", "text5"]

# Fill cache beyond limit
for text in texts:
    embedder.embed(text)

info = embedder.get_model_info()
assert info['cache_size'] == 3  # Evicted to limit

# Most recent should be cached, earliest should be evicted
# text3, text4, text5 should be in cache
# text1, text2 should be evicted

# Test 7: Normalization
embedder_norm = Embedder(normalize=True)
embedder_raw = Embedder(normalize=False)

text = "normalization test"
norm_embedding = embedder_norm.embed(text)
raw_embedding = embedder_raw.embed(text)

# Normalized embeddings should have unit length
assert np.allclose(np.linalg.norm(norm_embedding), 1.0)
assert not np.allclose(np.linalg.norm(raw_embedding), 1.0)

# Manual normalization should match
manual_norm = embedder_raw.normalize(raw_embedding)
assert np.allclose(norm_embedding, manual_norm)

# Test 8: Custom model
embedder = Embedder(model_name="all-mpnet-base-v2")
result = embedder.embed("custom model test")
assert result.shape == (768,)  # mpnet dimensions

info = embedder.get_model_info()
assert info['model_name'] == "all-mpnet-base-v2"
assert info['dimensions'] == 768

# Test 9: Error handling - invalid model
try:
    embedder = Embedder(model_name="invalid/model/name")
    embedder.embed("test")  # Trigger model loading
    assert False, "Should raise exception for invalid model"
except Exception as e:
    assert "invalid/model/name" in str(e).lower()

# Test 10: Edge cases
embedder = Embedder()

# Empty string
empty_result = embedder.embed("")
assert empty_result.shape == (384,)

# Empty list
empty_list_result = embedder.embed([])
assert empty_list_result.shape == (0, 384)

# Mixed empty and valid
mixed_result = embedder.embed(["", "valid text", ""])
assert mixed_result.shape == (3, 384)

# Test 11: Cache size validation
try:
    Embedder(cache_size=50)  # Below minimum
    assert False, "Should raise ValueError"
except ValueError as e:
    assert "cache_size" in str(e)

try:
    Embedder(cache_size=200000)  # Above maximum  
    assert False, "Should raise ValueError"
except ValueError as e:
    assert "cache_size" in str(e)

# Test 12: Model info without loading
embedder = Embedder()
info = embedder.get_model_info()
assert info['model_loaded'] == False
assert info['cache_size'] == 0
assert info['model_name'] == "all-MiniLM-L6-v2"

# Model info after loading
embedder.embed("trigger loading")
info = embedder.get_model_info()
assert info['model_loaded'] == True
assert info['dimensions'] == 384

# Test 13: Cache operations
embedder = Embedder(cache_dir="/tmp/test_ops")
embedder.embed("test content")

# Manual save
embedder.save_cache()
assert os.path.exists("/tmp/test_ops/embedding_cache.json")

# Cache clear
embedder.clear_cache()
info = embedder.get_model_info()
assert info['cache_size'] == 0
assert not os.path.exists("/tmp/test_ops/embedding_cache.json")
```

---

## Implementation Notes

### Dependencies

```python
# Core dependencies
import numpy as np
import hashlib
import json
import os
import time
import warnings
import logging
import atexit
from typing import Any

# Heavy import - deferred until model loading
# import sentence_transformers  # Only imported in _get_model()
```

### Performance Characteristics

**Initialization:**
- O(1) — no model loading, minimal setup
- Cache loading: O(n) where n = cached entries 

**Embedding:**
- Cold start: O(model_load_time + embed_time)  
- Cache hit: O(1) — hash lookup only
- Cache miss: O(embed_time) — model computation
- Batch processing: O(unique_texts × embed_time) — deduplicates identical texts

**Cache Management:**
- Memory: O(cache_size) — bounded by LRU eviction
- Disk I/O: Periodic (every 200 entries) + cleanup
- Hash computation: O(text_length) but extremely fast

**Model Loading Time (typical):**
- all-MiniLM-L6-v2: ~2-3 seconds (86MB download)
- all-mpnet-base-v2: ~5-8 seconds (420MB download)  
- Subsequent runs: ~0.5-1 second (local cache)

### Cache File Format

```json
{
  "md5_hash_32chars": [0.1, 0.2, 0.3, ...],
  "another_hash_here": [-0.1, 0.5, -0.2, ...],
  ...
}
```

### Error Handling Philosophy

- **Graceful Degradation:** Cache failures don't break embedding
- **Clear Error Messages:** Model loading errors include model name and context
- **Resource Cleanup:** atexit ensures cache persistence even on unexpected shutdown
- **Atomic Operations:** Cache saves use temp file + rename to prevent corruption
- **Warning Suppression:** Hide noise from transformers library during normal operation

### Memory Management

- **Bounded Cache:** LRU eviction prevents unbounded growth
- **Lazy Loading:** Model loaded only when needed
- **Numpy Efficiency:** Embeddings stored as efficient numpy arrays in memory
- **JSON Serialization:** Cache stored as lists for JSON compatibility (overhead acceptable)

### Concurrency Considerations

- **Thread Safety:** NOT thread-safe — use separate instances per thread
- **File Locking:** NOT implemented — single process cache access assumed
- **Cache Corruption:** Atomic writes prevent partial file corruption during saves

---

*"The difference between a toy and a weapon is not complexity — it is **reliability**. This embedder has seen battle. It has survived 10,000 production embedding requests. It **does not break**."*

— Zero, The Architect

