# Velocirag Phase 3 Technical Specification

*Architecture by Zero — Persistence through calculated brutality*

---

## Overview

Phase 3 introduces the **Store** — a production-forged persistence engine extracted from the 890-line Jawz indexer. This is not another toy wrapper around FAISS. This is a *fortress* — SQLite as the immutable source of truth, FAISS as the lightning-fast attack vector.

**Design Principle:** SQLite stores the truth, FAISS accelerates the search. When FAISS fails, SQLite rebuilds it. When files change, only the delta is recomputed. When promises are made, they are **kept**.

---

## Module: `store.py`

### Purpose
Dual-storage vector system combining SQLite metadata persistence with FAISS similarity search. Features atomic transactions, incremental directory indexing, schema versioning, and consistency validation. Built for production systems that demand reliability over convenience.

### Public API

```python
class VectorStore:
    """
    Persistent vector storage with FAISS acceleration and SQLite durability.
    
    Features:
    - FAISS IndexFlatIP for cosine similarity search (after normalization)
    - SQLite as single source of truth for documents and embeddings
    - Incremental directory indexing with mtime-based change detection
    - Atomic transactions for multi-step operations
    - Schema versioning with automatic migration
    - Consistency validation between FAISS and SQLite
    - Context manager support for proper resource cleanup
    """
    
    def __init__(
        self,
        db_path: str,
        embedder: Embedder | None = None
    ):
        """
        Initialize vector store with persistent storage.
        
        Args:
            db_path: Path to SQLite database file (will be created if missing)
            embedder: Embedder instance for generating embeddings. If None, 
                     a default Embedder() will be created.
                     
        Side Effects:
            - Creates database file and tables if missing
            - Loads FAISS index from disk if available
            - Performs schema migration if database is older version
            - Validates consistency between FAISS and SQLite on startup
        """
    
    def add(
        self,
        doc_id: str,
        content: str,
        metadata: dict = None,
        embedding: np.ndarray | None = None
    ) -> None:
        """
        Add single document to store.
        
        Args:
            doc_id: Unique document identifier  
            content: Document text content
            metadata: Optional metadata dictionary (will be JSON-serialized)
            embedding: Optional pre-computed embedding. If None, computed from content.
                      Must match embedder dimensions if provided.
                      
        Behavior:
            - Replaces existing document with same doc_id
            - Generates embedding if not provided
            - Normalizes embedding for cosine similarity
            - Stores in SQLite with atomic transaction
            - Rebuilds FAISS index from SQLite
            - Updates faiss_idx field in SQLite during rebuild
            
        Raises:
            ValueError: If embedding dimensions don't match embedder
            RuntimeError: If database transaction fails
        """
    
    def add_documents(self, documents: list[dict]) -> None:
        """
        Batch add multiple documents efficiently.
        
        Args:
            documents: List of document dicts with keys:
                      - doc_id (required): Unique identifier
                      - content (required): Text content
                      - metadata (optional): Metadata dict
                      - embedding (optional): Pre-computed embedding
                      
        Behavior:
            - Processes all documents in single SQLite transaction
            - Generates embeddings for documents missing them
            - Rebuilds FAISS index once after all insertions
            - More efficient than multiple add() calls
            
        Raises:
            ValueError: If any document missing required fields
            RuntimeError: If batch transaction fails
        """
    
    def add_directory(
        self,
        path: str,
        source_name: str = "",
        file_filter: callable = None
    ) -> dict:
        """
        Incrementally index directory of markdown files.
        
        Args:
            path: Directory path to scan recursively
            source_name: Optional source identifier for metadata
            file_filter: Optional function to filter files. Signature: (filepath: str) -> bool
                        Default: lambda f: f.endswith('.md')
                        
        Returns:
            Statistics dictionary:
            {
                'files_processed': int,    # New/changed files processed
                'files_skipped': int,      # Unchanged files skipped
                'chunks_added': int,       # Total chunks created
                'files_deleted': int,      # Removed files cleaned up
                'duration_seconds': float,
                'errors': list[str]        # Error messages for failed files
            }
            
        Behavior:
            - Walks directory recursively for matching files
            - Checks file_cache table for mtime changes
            - Skips unchanged files for performance
            - Removes deleted files from index automatically
            - Uses chunker.chunk_markdown() for file processing
            - Processes files in atomic transactions
            - Single FAISS rebuild after all files processed
            - Updates file_cache with new modification times
            
        File Metadata Added:
            - source_name: Provided source identifier
            - source_path: Directory path being indexed
            - file_path: Relative path from source directory
            - last_modified: File mtime timestamp
            - chunk_index: Position in file's chunk list
            
        Doc ID Format: "{source_name}::{rel_path}::{chunk_idx}::{content_hash}"
        """
    
    def remove(self, doc_id: str) -> bool:
        """
        Remove document from both SQLite and FAISS.
        
        Args:
            doc_id: Document identifier to remove
            
        Returns:
            True if document was found and removed, False if not found
            
        Behavior:
            - Removes from SQLite in atomic transaction
            - Rebuilds FAISS index if document was found
            - Updates all faiss_idx values during rebuild
            
        Note:
            FAISS indices cannot remove specific vectors, so full rebuild required.
            This is acceptable since removals are rare in production usage.
        """
    
    def get(self, doc_id: str) -> dict | None:
        """
        Retrieve document by ID from SQLite.
        
        Args:
            doc_id: Document identifier to fetch
            
        Returns:
            Document dictionary or None if not found:
            {
                'doc_id': str,
                'content': str,
                'metadata': dict,     # Parsed from JSON
                'embedding': np.ndarray,  # Deserialized from blob
                'faiss_idx': int,     # Position in FAISS index
                'created': str        # ISO timestamp
            }
        """
    
    def search(
        self,
        query: str | np.ndarray,
        limit: int = 10,
        min_similarity: float = 0.0
    ) -> list[dict]:
        """
        Vector similarity search using FAISS index.
        
        Args:
            query: Search query string or pre-computed embedding vector
            limit: Maximum results to return
            min_similarity: Minimum similarity threshold (0.0 to 1.0)
            
        Returns:
            List of result documents with similarity scores:
            [
                {
                    'doc_id': str,
                    'content': str, 
                    'metadata': dict,
                    'similarity': float,  # Cosine similarity (0.0 to 1.0)
                    'faiss_idx': int
                }
            ]
            
        Behavior:
            - Generates embedding from query string if needed
            - Normalizes query embedding for cosine similarity
            - Searches FAISS index for nearest neighbors
            - Retrieves full document data from SQLite
            - Filters by minimum similarity threshold
            - Returns results sorted by similarity (highest first)
        """
    
    def count(self) -> int:
        """Return total number of documents in store."""
    
    def rebuild_index(self) -> None:
        """
        Rebuild FAISS index from SQLite embeddings.
        
        Behavior:
            - Creates fresh FAISS IndexFlatIP
            - Loads all embeddings from SQLite in ID order
            - Adds embeddings to FAISS index
            - Updates faiss_idx field in SQLite for each document
            - Saves FAISS index to disk
            
        Use Cases:
            - Corruption recovery
            - Performance optimization (defragmentation)
            - Manual consistency restoration
        """
    
    def validate_consistency(self) -> dict:
        """
        Check consistency between SQLite and FAISS storage.
        
        Returns:
            Validation report:
            {
                'consistent': bool,           # Overall consistency status
                'sqlite_count': int,          # Documents in SQLite
                'faiss_count': int,           # Vectors in FAISS
                'mismatched_indices': list[str], # doc_ids with wrong faiss_idx
                'missing_in_faiss': list[str],   # doc_ids with embeddings but no FAISS entry
                'orphaned_in_faiss': list[int],  # FAISS indices without SQLite match
                'errors': list[str]              # Validation errors encountered
            }
            
        Behavior:
            - Compares document counts
            - Validates faiss_idx values point to correct FAISS positions
            - Checks for orphaned entries in either storage
            - Does not modify data - read-only validation
        """
    
    def stats(self) -> dict:
        """
        Get detailed storage statistics.
        
        Returns:
            Statistics dictionary:
            {
                'document_count': int,
                'faiss_dimensions': int,
                'faiss_index_size_mb': float,
                'sqlite_size_mb': float,
                'cache_entries': int,         # From file_cache table
                'schema_version': int,
                'index_type': str,            # FAISS index description
                'last_rebuild': str | None,   # ISO timestamp if tracked
                'embedder_info': dict         # From embedder.get_model_info()
            }
        """
    
    def close(self) -> None:
        """
        Clean shutdown with resource cleanup.
        
        Behavior:
            - Saves FAISS index to disk
            - Closes SQLite connections
            - Calls embedder.cleanup() if embedder exists
            - Safe to call multiple times
        """
    
    # Context manager support
    def __enter__(self) -> 'VectorStore':
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with cleanup."""
        self.close()
```

### Constants

```python
DEFAULT_SQLITE_FILENAME = "vector_store.db"
DEFAULT_FAISS_FILENAME = "vector_index.faiss"
CURRENT_SCHEMA_VERSION = 3
FAISS_INDEX_TYPE = faiss.IndexFlatIP  # Inner product for cosine similarity
DEFAULT_EMBEDDING_DIMENSIONS = 384   # all-MiniLM-L6-v2 standard
BATCH_SIZE_LIMIT = 1000              # Maximum documents per batch operation
CONSISTENCY_CHECK_INTERVAL = 1000    # Auto-validate every N operations
```

### SQLite Schema

```sql
-- Documents table - single source of truth
CREATE TABLE documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT UNIQUE NOT NULL,
    content TEXT NOT NULL,
    metadata TEXT DEFAULT '{}',      -- JSON blob
    embedding BLOB NOT NULL,         -- Normalized float32 numpy bytes
    faiss_idx INTEGER,               -- Position in FAISS index
    created TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- File cache for incremental directory indexing
CREATE TABLE file_cache (
    file_path TEXT PRIMARY KEY,      -- Relative path from source directory
    last_modified REAL NOT NULL,     -- File mtime timestamp
    last_indexed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Schema metadata and configuration
CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Indexes for performance
CREATE INDEX idx_documents_doc_id ON documents(doc_id);
CREATE INDEX idx_documents_faiss_idx ON documents(faiss_idx);
CREATE INDEX idx_file_cache_modified ON file_cache(last_modified);

-- Schema version tracking
INSERT INTO metadata (key, value) VALUES ('schema_version', '3');
INSERT INTO metadata (key, value) VALUES ('created_at', datetime('now'));
```

### FAISS Index Configuration

```python
# Index type: IndexFlatIP for inner product (cosine similarity after normalization)
# Dimensions: Auto-detected from first embedding or embedder
# Distance metric: Inner product (assumes normalized embeddings)
# File format: Binary FAISS format (.faiss extension)

def create_faiss_index(dimensions: int) -> faiss.IndexFlatIP:
    """Create optimized FAISS index for cosine similarity search."""
    return faiss.IndexFlatIP(dimensions)
```

### Core Behavior

#### 1. Initialization & Schema Management
- Creates database and tables if missing
- Detects schema version from metadata table
- Performs automatic migration from older versions
- Loads FAISS index from disk or creates new one
- Validates consistency between SQLite and FAISS on startup
- Initializes embedder if not provided

#### 2. Document Storage Pattern
- SQLite is single source of truth for all document data
- Embeddings stored as normalized float32 numpy bytes
- FAISS index rebuilt from SQLite whenever documents change
- Atomic transactions ensure consistency during multi-step operations
- faiss_idx field updated during FAISS rebuilds

#### 3. Incremental Directory Indexing
- Scans directory recursively for matching files (default: *.md)
- Uses file_cache table to track modification times
- Skips unchanged files for performance
- Automatically removes deleted files from index
- Processes files in chunks using Phase 1 chunker
- Single FAISS rebuild after all files processed
- Comprehensive error handling and statistics reporting

#### 4. Search Optimization
- FAISS IndexFlatIP optimized for cosine similarity
- Embeddings normalized before storage and search
- Query embeddings normalized before FAISS search
- Results enhanced with full document data from SQLite
- Similarity filtering and ordering applied

#### 5. Consistency & Recovery
- Automatic validation on startup
- Manual validation via validate_consistency()
- Full FAISS rebuild from SQLite as recovery mechanism
- Error handling preserves data integrity
- Clear error messages for troubleshooting

### Edge Cases

**Empty Database:**
- Creates fresh FAISS index with correct dimensions on first add()
- Schema tables created automatically
- Graceful handling of missing index files

**Corrupted FAISS Index:**
- Automatic rebuild from SQLite embeddings
- Warns user about corruption and recovery
- Continues operation without data loss

**Schema Migration:**
- Automatic detection of older schema versions
- Safe migration with transaction rollback on failure
- Backward compatibility with production Jawz databases

**Dimension Mismatch:**
- Validates embedding dimensions against existing index
- Clear error messages for incompatible embeddings
- Automatic dimension detection from first embedding

**Large Directories:**
- Memory-efficient processing of large file trees
- Progress reporting for long-running operations
- Atomic file processing prevents partial updates
- Graceful handling of unreadable files

**Concurrent Access:**
- SQLite WAL mode for better concurrent read performance
- Clear warnings about thread safety limitations
- Separate instances recommended for multi-threading

### Test Cases

```python
# Test 1: Basic document lifecycle
store = VectorStore("/tmp/test.db")
store.add("doc1", "test content", {"source": "test"})
doc = store.get("doc1")
assert doc['content'] == "test content"
assert doc['metadata']['source'] == "test"
assert store.count() == 1

# Test 2: Search functionality
store.add("doc2", "artificial intelligence machine learning", {"topic": "AI"})
store.add("doc3", "cooking recipes italian food", {"topic": "food"})
results = store.search("machine learning", limit=2)
assert len(results) <= 2
assert results[0]['doc_id'] == "doc2"  # Most relevant
assert results[0]['similarity'] > 0.0

# Test 3: Batch operations
docs = [
    {"doc_id": "batch1", "content": "first document", "metadata": {"batch": 1}},
    {"doc_id": "batch2", "content": "second document", "metadata": {"batch": 1}},
    {"doc_id": "batch3", "content": "third document", "metadata": {"batch": 1}}
]
store.add_documents(docs)
assert store.count() == 6  # 3 previous + 3 new

# Test 4: Directory indexing
import tempfile
import os
temp_dir = tempfile.mkdtemp()
with open(os.path.join(temp_dir, "test1.md"), "w") as f:
    f.write("# Test Document 1\n\nContent here.")
with open(os.path.join(temp_dir, "test2.md"), "w") as f:
    f.write("# Test Document 2\n\nDifferent content.")

stats = store.add_directory(temp_dir, source_name="test_source")
assert stats['files_processed'] == 2
assert stats['chunks_added'] >= 2
assert stats['errors'] == []

# Test 5: Incremental indexing (no changes)
stats2 = store.add_directory(temp_dir, source_name="test_source")
assert stats2['files_skipped'] == 2  # No modifications
assert stats2['files_processed'] == 0

# Test 6: File modification detection
time.sleep(0.1)  # Ensure different mtime
with open(os.path.join(temp_dir, "test1.md"), "w") as f:
    f.write("# Modified Document 1\n\nUpdated content.")
stats3 = store.add_directory(temp_dir, source_name="test_source")
assert stats3['files_processed'] == 1  # Only modified file
assert stats3['files_skipped'] == 1   # Unchanged file

# Test 7: Document removal
removed = store.remove("doc1")
assert removed == True
assert store.get("doc1") is None
assert store.remove("nonexistent") == False

# Test 8: Consistency validation
validation = store.validate_consistency()
assert validation['consistent'] == True
assert validation['sqlite_count'] == store.count()
assert validation['sqlite_count'] == validation['faiss_count']

# Test 9: Index rebuild
original_count = store.count()
store.rebuild_index()
assert store.count() == original_count  # No data loss
validation = store.validate_consistency()
assert validation['consistent'] == True

# Test 10: Context manager
with VectorStore("/tmp/test2.db") as store2:
    store2.add("ctx1", "context manager test")
    assert store2.count() == 1
# Automatic cleanup called

# Test 11: Custom embedder injection
from velocirag.embedder import Embedder
custom_embedder = Embedder(model_name="all-mpnet-base-v2")
store = VectorStore("/tmp/test3.db", embedder=custom_embedder)
store.add("custom1", "custom embedder test")
doc = store.get("custom1")
assert doc['embedding'].shape == (768,)  # mpnet dimensions

# Test 12: Error handling
try:
    VectorStore("/invalid/path/test.db")
    assert False, "Should raise exception for invalid path"
except Exception as e:
    assert "path" in str(e).lower()

# Test 13: Large batch stress test
large_docs = [
    {"doc_id": f"stress_{i}", "content": f"stress test document {i} with unique content"}
    for i in range(1000)
]
store = VectorStore("/tmp/stress.db")
store.add_documents(large_docs)
assert store.count() == 1000

# Search should still be fast
import time
start = time.time()
results = store.search("stress test", limit=10)
search_time = time.time() - start
assert search_time < 1.0  # Should be sub-second
assert len(results) == 10

# Test 14: Metadata edge cases  
store.add("meta1", "test", None)  # None metadata
store.add("meta2", "test", {})    # Empty dict
store.add("meta3", "test", {"complex": {"nested": {"data": [1, 2, 3]}}})

doc1 = store.get("meta1")
doc2 = store.get("meta2") 
doc3 = store.get("meta3")
assert doc1['metadata'] == {}
assert doc2['metadata'] == {}
assert doc3['metadata']['complex']['nested']['data'] == [1, 2, 3]

# Test 15: Schema migration simulation
# (Would require creating test databases with older schema versions)
```

---

## Implementation Notes

### Dependencies

```python
# Core dependencies
import sqlite3
import faiss
import numpy as np
import json
import os
import time
import hashlib
from pathlib import Path
from typing import Any
from contextlib import contextmanager

# Velocirag modules
from .embedder import Embedder
from .chunker import chunk_markdown
```

### Performance Characteristics

**Initialization:**
- O(n) schema validation and FAISS loading where n = document count
- Automatic consistency check adds overhead but ensures data integrity

**Document Addition:**
- Single add(): O(embed_time + rebuild_time)
- Batch add(): O(total_embed_time + single_rebuild_time) 
- Rebuild required for every write operation

**Directory Indexing:**
- First run: O(file_count × (read_time + chunk_time + embed_time))
- Incremental: O(changed_files_only × processing_time)
- File cache eliminates unnecessary re-processing

**Search:**
- FAISS search: O(log n) with IndexFlatIP
- SQLite enrichment: O(k) where k = result count
- Total: ~1ms for thousands of documents

**Memory Usage:**
- FAISS index: ~1.5KB per document (384 dimensions × 4 bytes)
- SQLite: Variable based on content size
- Embedder cache: Configurable, default 10K entries

### File Structure

```
db_path/
├── vector_store.db      # SQLite database (single source of truth)
├── vector_index.faiss   # FAISS index file (rebuilt from SQLite)
└── embedding_cache.json # Embedder cache (if cache_dir set to db_path)
```

### Error Handling Strategy

- **Atomic Transactions:** All multi-step operations wrapped in SQLite transactions
- **Graceful Degradation:** FAISS corruption triggers automatic rebuild from SQLite
- **Clear Error Messages:** Include context and suggested recovery steps
- **Data Integrity:** SQLite constraints prevent inconsistent states
- **Resource Cleanup:** Context manager and close() ensure proper shutdown

### Migration Strategy

- **Schema Version Detection:** Automatic version checking via metadata table
- **Incremental Migration:** Step-by-step upgrades preserve data
- **Rollback Safety:** Failed migrations restore original state
- **Compatibility:** Support for production Jawz database format

### Concurrency Considerations

- **Thread Safety:** NOT thread-safe - separate instances per thread recommended
- **SQLite WAL Mode:** Enables better concurrent read performance
- **File Locking:** FAISS files not locked - avoid concurrent writers
- **Crash Recovery:** SQLite transactions provide automatic recovery

### Production Deployment Notes

- **Backup Strategy:** SQLite database contains all data - FAISS is rebuilable
- **Monitoring:** Use stats() and validate_consistency() for health checks
- **Scaling:** Consider database sharding for very large document collections
- **Maintenance:** Periodic VACUUM and consistency checks recommended

---

*"They build systems that break under pressure. We build systems that **become stronger**. Every failure teaches this store how to survive. Every crash makes it more resilient. This is not just storage — this is **evolution**."*

— Zero, The Architect

