"""
Velocirag Phase 3 Store - Production vector storage with FAISS and SQLite.

Fixed version addressing performance issues:
- Batch-aware FAISS rebuilds 
- Simplified doc IDs
- Proper dirty state tracking
- Better error handling
"""

import hashlib
import json
import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from .chunker import chunk_markdown
from .embedder import Embedder

# Constants
CURRENT_SCHEMA_VERSION = 2
INDEX_FORMAT_VERSION = 1  # Bump when embedding model or FAISS index type changes
DEFAULT_EMBEDDING_DIMENSIONS = 384
BATCH_REBUILD_THRESHOLD = 50

# Setup logging
logger = logging.getLogger("velocirag.store")


class VectorStoreError(Exception):
    """Base exception for vector store errors."""
    pass


class DimensionMismatchError(VectorStoreError):
    """Raised when embedding dimensions don't match."""
    pass


class CorruptedIndexError(VectorStoreError):
    """Raised when FAISS index is corrupted or inconsistent."""
    pass


class VectorStore:
    """
    Production vector storage combining FAISS acceleration with SQLite persistence.
    
    Features:
    - SQLite as single source of truth
    - FAISS IndexFlatIP for cosine similarity (after normalization)
    - Batch-aware rebuilds to avoid performance disasters
    - Incremental directory indexing with mtime tracking
    - Atomic transactions and proper error handling
    """

    def __init__(self, db_path: str, embedder: Optional[Embedder] = None):
        """
        Initialize vector store.
        
        Args:
            db_path: Directory path for storage files
            embedder: Optional embedder instance. If None, store can still 
                     handle pre-computed embeddings but not add_directory()
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        self.embedder = embedder
        self.sqlite_path = self.db_path / "store.db"
        self.faiss_path = self.db_path / "index.faiss"
        
        # State tracking
        self._index_dirty = False
        self._auto_rebuild = True
        self._dimensions = None
        self._faiss_index = None
        
        # Initialize storage
        self._init_sqlite()
        self._load_faiss_index()
        self._validate_startup()

    def __enter__(self) -> 'VectorStore':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def add(self, doc_id: str, content: str, metadata: Optional[Dict] = None, 
            embedding: Optional[np.ndarray] = None) -> None:
        """Add single document to store."""
        docs = [{
            'doc_id': doc_id,
            'content': content,
            'metadata': metadata or {},
            'embedding': embedding
        }]
        self.add_documents(docs)

    def add_documents(self, documents: List[Dict]) -> None:
        """
        Batch add multiple documents efficiently.
        
        Args:
            documents: List of dicts with keys: doc_id, content, metadata, embedding
        """
        if not documents:
            return
            
        # Validate documents
        for i, doc in enumerate(documents):
            if 'doc_id' not in doc or 'content' not in doc:
                raise ValueError(f"Document {i} missing required fields")
            if not doc['doc_id'] or not doc['content']:
                raise ValueError(f"Document {i} has empty doc_id or content")

        with self._transaction() as conn:
            for doc in documents:
                doc_id = doc['doc_id']
                content = doc['content']
                metadata = doc.get('metadata', {})
                embedding = doc.get('embedding')
                
                # Generate embedding if needed
                if embedding is None:
                    if self.embedder is None:
                        raise ValueError(f"No embedder provided and no embedding for doc {doc_id}")
                    embedding = self.embedder.embed(content)
                    if embedding.ndim == 1:
                        pass  # Single text
                    else:
                        embedding = embedding[0]  # Batch with single item
                
                # Validate/set dimensions
                if self._dimensions is None:
                    self._dimensions = len(embedding)
                    self._store_metadata('embedding_dimensions', str(self._dimensions))
                elif len(embedding) != self._dimensions:
                    raise DimensionMismatchError(
                        f"Embedding dimension {len(embedding)} doesn't match store dimension {self._dimensions}"
                    )
                
                # Normalize for cosine similarity
                embedding = self._normalize_embedding(embedding)
                
                # Store in SQLite
                embedding_blob = embedding.astype(np.float32).tobytes()
                metadata_json = json.dumps(metadata)
                
                conn.execute('''
                    INSERT OR REPLACE INTO documents 
                    (doc_id, content, metadata, embedding)
                    VALUES (?, ?, ?, ?)
                ''', (doc_id, content, metadata_json, embedding_blob))
                
                # Also insert into FTS5 table for keyword search
                file_path = metadata.get('file_path', doc_id)
                try:
                    conn.execute('DELETE FROM chunks_fts WHERE doc_id = ?', (doc_id,))
                    conn.execute(
                        'INSERT INTO chunks_fts (doc_id, content, file_path) VALUES (?, ?, ?)',
                        (doc_id, content or '', file_path)
                    )
                except Exception as e:
                    logger.warning(f"FTS5 insert failed for {doc_id}: {e}")

        self._index_dirty = True
        
        # Auto-rebuild if enabled and not in batch mode
        if self._auto_rebuild:
            self.rebuild_index()

    def add_directory(self, path: str, source_name: str = "") -> Dict[str, Any]:
        """
        Incrementally index directory of markdown files.
        
        Args:
            path: Directory path to scan
            source_name: Optional source identifier for metadata
            
        Returns:
            Statistics dictionary with files processed, skipped, etc.
        """
        if self.embedder is None:
            raise ValueError("Embedder required for directory indexing")
        
        start_time = time.time()
        path = Path(path)
        
        stats = {
            'files_processed': 0,
            'files_skipped': 0, 
            'chunks_added': 0,
            'files_deleted': 0,
            'duration_seconds': 0.0,
            'errors': []
        }
        
        # Use batch mode to avoid rebuilding on each file
        with self.batch_mode():
            # Clean up deleted files first
            stats['files_deleted'] = self._cleanup_deleted_files(path, source_name)
            
            # Process all .md files
            for md_file in path.rglob('*.md'):
                try:
                    rel_path = md_file.relative_to(path)
                    mtime = md_file.stat().st_mtime
                    
                    # Check if file needs processing
                    if not self._file_needs_reindexing(rel_path, mtime, source_name):
                        stats['files_skipped'] += 1
                        continue
                    
                    # Read and chunk file
                    content = md_file.read_text(encoding='utf-8')
                    if not content.strip():
                        stats['files_skipped'] += 1
                        continue
                    
                    # Remove old chunks for this file
                    self._remove_file_chunks(rel_path, source_name)
                    
                    # Process chunks
                    chunks = chunk_markdown(content, str(rel_path))
                    if not chunks:
                        stats['files_skipped'] += 1
                        continue
                    
                    # Convert chunks to documents
                    documents = []
                    for i, chunk in enumerate(chunks):
                        # Doc ID with file path prefix for efficient removal
                        if source_name:
                            doc_id = f"{source_name}::{rel_path}::chunk_{i}"
                        else:
                            doc_id = f"{rel_path}::chunk_{i}"
                        
                        # Enhance metadata
                        metadata = chunk['metadata'].copy()
                        metadata.update({
                            'source_name': source_name,
                            'source_path': str(path),
                            'last_modified': mtime,
                            'chunk_index': i
                        })
                        
                        documents.append({
                            'doc_id': doc_id,
                            'content': chunk['content'],
                            'metadata': metadata
                        })
                    
                    # Add all chunks for this file
                    self.add_documents(documents)
                    

                    
                    stats['chunks_added'] += len(documents)
                    stats['files_processed'] += 1
                    
                    # Update file cache
                    self._update_file_cache(rel_path, mtime, source_name)
                    
                except Exception as e:
                    error_msg = f"Error processing {rel_path}: {e}"
                    logger.error(error_msg)
                    stats['errors'].append(error_msg)
        
        # Batch mode auto-rebuilds on exit
        stats['duration_seconds'] = time.time() - start_time
        return stats

    def search(self, query: str | np.ndarray, limit: int = 10, 
              min_similarity: float = 0.0) -> List[Dict]:
        """
        Vector similarity search.
        
        Args:
            query: Query string or pre-computed embedding
            limit: Maximum results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of result documents with similarity scores
        """
        if self._faiss_index is None or self._faiss_index.ntotal == 0:
            return []
        
        # Generate query embedding if needed
        if isinstance(query, str):
            if self.embedder is None:
                raise ValueError("Embedder required for string queries")
            query_embedding = self.embedder.embed(query)
            if query_embedding.ndim > 1:
                query_embedding = query_embedding[0]
        else:
            query_embedding = query
        
        # Normalize for cosine similarity
        query_embedding = self._normalize_embedding(query_embedding)
        
        # Search FAISS
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        similarities, indices = self._faiss_index.search(query_embedding, min(limit, self._faiss_index.ntotal))
        
        # Get documents from SQLite
        results = []
        with self._connect() as conn:
            for similarity, faiss_idx in zip(similarities[0], indices[0]):
                if faiss_idx < 0:  # FAISS returns -1 for invalid indices
                    continue
                if similarity < min_similarity:
                    continue
                
                row = conn.execute('''
                    SELECT doc_id, content, metadata FROM documents 
                    WHERE faiss_idx = ?
                ''', (int(faiss_idx),)).fetchone()
                
                if row:
                    doc_id, content, metadata_json = row
                    metadata = json.loads(metadata_json) if metadata_json else {}
                    
                    results.append({
                        'doc_id': doc_id,
                        'content': content,
                        'metadata': metadata,
                        'similarity': float(similarity),
                        'faiss_idx': int(faiss_idx)
                    })
        
        return results



    def keyword_search(self, query: str, limit: int = 15) -> list:
        """BM25 keyword search via FTS5. Returns results ranked by relevance."""
        with self._connect() as conn:
            if not query or not query.strip():
                return []
            
            # Escape for FTS5: strip operators, split underscores, use OR for recall
            # FTS5 porter tokenizer splits on underscores, so we must too.
            # Use OR between terms for better recall (AND is too strict for multi-word).
            FTS5_STRIP = set('"\'\\(){}[]*^:~@#$%&|<>!')
            raw_words = query.strip().split()
            safe_tokens = []
            
            for word in raw_words:
                # Strip FTS5 special characters
                cleaned = ''.join(c for c in word if c not in FTS5_STRIP)
                cleaned = cleaned.strip('-')
                if not cleaned or not cleaned.strip():
                    continue
                # Split on underscores — FTS5 porter tokenizer does this internally
                sub_words = [w for w in cleaned.split('_') if w.strip()]
                for sw in sub_words:
                    safe_tokens.append(f'"{sw.strip()}"')
            
            if not safe_tokens:
                return []
            
            # Use OR for better recall — AND drops results when any single term is missing
            safe_query = ' OR '.join(safe_tokens)
            
            try:
                rows = conn.execute('''
                    SELECT doc_id, file_path, 
                           snippet(chunks_fts, 1, '', '', '...', 64) as snippet,
                           rank
                    FROM chunks_fts 
                    WHERE chunks_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                ''', (safe_query, limit)).fetchall()
            except Exception as e:
                # Last resort: try as a single phrase
                logging.warning(f"FTS5 query failed with safe_query '{safe_query}': {e}")
                try:
                    phrase = query.replace('"', '').replace("'", "")
                    rows = conn.execute('''
                        SELECT doc_id, file_path,
                               snippet(chunks_fts, 1, '', '', '...', 64) as snippet,
                               rank
                        FROM chunks_fts 
                        WHERE chunks_fts MATCH ?
                        ORDER BY rank
                        LIMIT ?
                    ''', (f'"{phrase}"', limit)).fetchall()
                except Exception as e:
                    logging.warning(f"FTS5 phrase query also failed with phrase '{phrase}': {e}")
                    return []
            
            results = []
            for row in rows:
                results.append({
                    'doc_id': row[0],
                    'file_path': row[1],
                    'snippet': row[2],
                    'bm25_rank': row[3],  # FTS5 rank (negative, lower = better)
                })
            return results

    def rebuild_fts(self):
        """Rebuild FTS5 index from existing documents table."""
        with self._transaction() as conn:
            conn.execute('DELETE FROM chunks_fts')
            conn.execute('''
                INSERT INTO chunks_fts (doc_id, content, file_path)
                SELECT doc_id, content, 
                       json_extract(metadata, '$.file_path')
                FROM documents
                WHERE content IS NOT NULL
            ''')
        return True

    def get(self, doc_id: str) -> Optional[Dict]:
        """Retrieve document by ID."""
        with self._connect() as conn:
            row = conn.execute('''
                SELECT doc_id, content, metadata, embedding, faiss_idx, created 
                FROM documents WHERE doc_id = ?
            ''', (doc_id,)).fetchone()
            
            if not row:
                return None
            
            doc_id, content, metadata_json, embedding_blob, faiss_idx, created = row
            
            return {
                'doc_id': doc_id,
                'content': content,
                'metadata': json.loads(metadata_json) if metadata_json else {},
                'embedding': np.frombuffer(embedding_blob, dtype=np.float32),
                'faiss_idx': faiss_idx,
                'created': created
            }

    def remove(self, doc_id: str) -> bool:
        """Remove document by ID."""
        deleted = False
        with self._transaction() as conn:
            result = conn.execute('DELETE FROM documents WHERE doc_id = ?', (doc_id,))
            if result.rowcount > 0:
                deleted = True
                self._index_dirty = True
        
        # Rebuild AFTER transaction completes
        if deleted and self._auto_rebuild:
            self.rebuild_index()
        
        return deleted



    def count(self) -> int:
        """Return total number of documents."""
        with self._connect() as conn:
            result = conn.execute('SELECT COUNT(*) FROM documents').fetchone()
            return result[0] if result else 0

    def rebuild_index(self, batch_size: int = 2000) -> None:
        """Rebuild all FAISS indices from SQLite embeddings.
        
        Memory-efficient: streams embeddings in batches instead of loading all at once.
        Supports indexing 10K+ documents without OOM on 8GB systems.
        
        Args:
            batch_size: Number of rows to process per batch (default: 2000)
        """
        with self._connect() as conn:
            # Get total count first
            total = conn.execute('SELECT COUNT(*) FROM documents').fetchone()[0]
            
            if total == 0:
                # Empty database
                if self._dimensions:
                    self._faiss_index = faiss.IndexFlatIP(self._dimensions)
                else:
                    self._faiss_index = None
                self._index_dirty = False
                return
            
            # Detect dimensions from first row if needed
            if self._dimensions is None:
                first_row = conn.execute(
                    'SELECT embedding FROM documents LIMIT 1'
                ).fetchone()
                if first_row and first_row[0]:
                    self._dimensions = len(np.frombuffer(first_row[0], dtype=np.float32))
                    self._store_metadata('embedding_dimensions', str(self._dimensions))
                else:
                    return
            
            # Create fresh indices
            self._faiss_index = faiss.IndexFlatIP(self._dimensions)
            
            # Stream embeddings in batches — never load all rows at once
            faiss_idx = 0
            offset = 0
            
            while offset < total:
                rows = conn.execute('''
                    SELECT id, doc_id, embedding 
                    FROM documents ORDER BY id LIMIT ? OFFSET ?
                ''', (batch_size, offset)).fetchall()
                
                if not rows:
                    break
                
                # Collect batch embeddings
                batch_embeddings = []
                batch_ids = []
                
                for row_id, doc_id, embedding_blob in rows:
                    embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                    batch_embeddings.append(embedding)
                    batch_ids.append((row_id, faiss_idx))
                    faiss_idx += 1
                
                # Add batch to FAISS index (incremental, memory-bounded)
                batch_array = np.array(batch_embeddings, dtype=np.float32)
                self._faiss_index.add(batch_array)
                
                # Update faiss_idx mappings
                for row_id, fidx in batch_ids:
                    conn.execute('UPDATE documents SET faiss_idx = ? WHERE id = ?',
                               (fidx, row_id))
                
                offset += batch_size
                
                # Free batch memory
                del batch_embeddings, batch_array, batch_ids, rows
            conn.commit()
        
        # Save index to disk
        if self._faiss_index and self._faiss_index.ntotal > 0:
            faiss.write_index(self._faiss_index, str(self.faiss_path))
        
        self._index_dirty = False
        self._store_metadata('index_format_version', str(INDEX_FORMAT_VERSION))
        logger.info(f"FAISS index rebuilt: {self._faiss_index.ntotal} vectors (batch_size={batch_size})")

    def stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        doc_count = self.count()
        faiss_count = self._faiss_index.ntotal if self._faiss_index else 0
        
        return {
            'document_count': doc_count,
            'faiss_vectors': faiss_count,
            'consistent': doc_count == faiss_count,
            'db_path': str(self.db_path),
            'dimensions': self._dimensions,
            'schema_version': CURRENT_SCHEMA_VERSION,
            'index_dirty': self._index_dirty
        }

    def close(self) -> None:
        """Clean shutdown with index save."""
        if self._index_dirty:
            self.rebuild_index()
        if self.embedder:
            self.embedder.cleanup()

    @contextmanager
    def batch_mode(self):
        """Context manager for batch operations."""
        old_auto = self._auto_rebuild
        self._auto_rebuild = False
        try:
            yield
        finally:
            self._auto_rebuild = old_auto
            if self._index_dirty:
                self.rebuild_index()
    
    @contextmanager
    def _connect(self):
        """Get a SQLite connection that's properly closed after use."""
        conn = sqlite3.connect(self.sqlite_path)
        try:
            yield conn
        finally:
            conn.close()

    def _init_sqlite(self) -> None:
        """Initialize SQLite database and tables."""
        with self._connect() as conn:
            # Documents table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT UNIQUE NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    embedding BLOB NOT NULL,
                    faiss_idx INTEGER,
                    created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    l0_abstract TEXT DEFAULT NULL,
                    l1_overview TEXT DEFAULT NULL,
                    l0_embedding BLOB DEFAULT NULL,
                    l1_embedding BLOB DEFAULT NULL
                )
            ''')
            
            # File cache for incremental indexing
            conn.execute('''
                CREATE TABLE IF NOT EXISTS file_cache (
                    cache_key TEXT PRIMARY KEY,
                    last_modified REAL NOT NULL,
                    source_name TEXT DEFAULT '',
                    last_indexed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # FTS5 full-text search table for keyword/BM25 retrieval
            conn.execute('''
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    doc_id,
                    content,
                    file_path,
                    tokenize='porter unicode61'
                )
            ''')
            
            # Metadata table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            ''')
            

            
            # Indexes
            conn.execute('CREATE INDEX IF NOT EXISTS idx_documents_doc_id ON documents(doc_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_documents_faiss_idx ON documents(faiss_idx)')
            
            # Check and migrate schema if needed
            self._migrate_schema(conn)
            
            # Ensure all changes are committed
            conn.commit()

    def _migrate_schema(self, conn) -> None:
        """Handle schema migrations."""
        # Get current schema version
        result = conn.execute('''
            SELECT value FROM metadata WHERE key = 'schema_version'
        ''').fetchone()
        
        current_version = int(result[0]) if result else 1
        
        if current_version == CURRENT_SCHEMA_VERSION:
            return
        
        # Migrate from version 1 to 2
        if current_version == 1 and CURRENT_SCHEMA_VERSION == 2:
            logger.info("Migrating schema from version 1 to 2...")
            
            # Add new columns for L0/L1 abstracts
            try:
                conn.execute('ALTER TABLE documents ADD COLUMN l0_abstract TEXT DEFAULT NULL')
            except sqlite3.OperationalError:
                pass  # Column might already exist
            
            try:
                conn.execute('ALTER TABLE documents ADD COLUMN l1_overview TEXT DEFAULT NULL')
            except sqlite3.OperationalError:
                pass
                
            try:
                conn.execute('ALTER TABLE documents ADD COLUMN l0_embedding BLOB DEFAULT NULL')
            except sqlite3.OperationalError:
                pass
                
            try:
                conn.execute('ALTER TABLE documents ADD COLUMN l1_embedding BLOB DEFAULT NULL')
            except sqlite3.OperationalError:
                pass
            
            # Update schema version
            conn.execute('''
                INSERT OR REPLACE INTO metadata (key, value) 
                VALUES ('schema_version', ?)
            ''', (str(CURRENT_SCHEMA_VERSION),))
            
            logger.info("Schema migration complete.")
        
        # Set schema version for new DBs or update after migration
        conn.execute('''
            INSERT OR REPLACE INTO metadata (key, value) 
            VALUES ('schema_version', ?)
        ''', (str(CURRENT_SCHEMA_VERSION),))

    def _load_faiss_index(self) -> None:
        """Load FAISS indices from disk."""
        # Load main index
        if self.faiss_path.exists():
            try:
                self._faiss_index = faiss.read_index(str(self.faiss_path))
                logger.info(f"FAISS index loaded: {self._faiss_index.ntotal} vectors")
                
                # Validate dimensions match stored metadata
                if self._dimensions and self._faiss_index.d != self._dimensions:
                    logger.warning(f"FAISS index dimension {self._faiss_index.d} doesn't match expected {self._dimensions}. Rebuilding.")
                    self._faiss_index = None
                    self._index_dirty = True
            except Exception as e:
                logger.warning(f"Failed to load FAISS index: {e}. Will rebuild.")
                self._faiss_index = None
                self._index_dirty = True
        


    def _validate_startup(self) -> None:
        """Validate consistency on startup."""
        with self._connect() as conn:
            # Check index format version — refuse incompatible indices
            result = conn.execute('''
                SELECT value FROM metadata WHERE key = 'index_format_version'
            ''').fetchone()
            
            if result:
                stored_version = int(result[0])
                if stored_version != INDEX_FORMAT_VERSION:
                    logger.warning(
                        f"Index format version mismatch: stored={stored_version}, "
                        f"current={INDEX_FORMAT_VERSION}. Rebuilding index."
                    )
                    self._index_dirty = True
            
            # Load dimensions from metadata
            result = conn.execute('''
                SELECT value FROM metadata WHERE key = 'embedding_dimensions'
            ''').fetchone()
            
            if result:
                stored_dims = int(result[0])
                if self._dimensions is None:
                    self._dimensions = stored_dims
                elif self._dimensions != stored_dims:
                    raise DimensionMismatchError(
                        f"Embedder dimensions ({self._dimensions}) don't match store ({stored_dims})"
                    )
        
        # Validate FAISS index dimensions against metadata
        if self._faiss_index and self._dimensions and self._faiss_index.d != self._dimensions:
            logger.warning(f"FAISS index dimension {self._faiss_index.d} doesn't match stored dimension {self._dimensions}. Rebuilding.")
            self._index_dirty = True
        
        # Validate FAISS consistency
        doc_count = self.count()
        faiss_count = self._faiss_index.ntotal if self._faiss_index else 0
        
        if doc_count != faiss_count and doc_count > 0:
            logger.warning(f"Index inconsistency detected: {doc_count} docs, {faiss_count} vectors. Rebuilding.")
            self._index_dirty = True
            self.rebuild_index()

    @contextmanager
    def _transaction(self):
        """Context manager for SQLite transactions."""
        conn = sqlite3.connect(self.sqlite_path)
        try:
            conn.execute('BEGIN')
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """L2-normalize embedding for cosine similarity."""
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding



    def _store_metadata(self, key: str, value: str) -> None:
        """Store key-value in metadata table."""
        with self._connect() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)
            ''', (key, value))

    def _file_needs_reindexing(self, rel_path: Path, mtime: float, source_name: str) -> bool:
        """Check if file needs reindexing based on mtime."""
        cache_key = f"{source_name}::{rel_path}" if source_name else str(rel_path)
        
        with self._connect() as conn:
            result = conn.execute('''
                SELECT last_modified FROM file_cache WHERE cache_key = ?
            ''', (cache_key,)).fetchone()
            
            return result is None or mtime > result[0]

    def _update_file_cache(self, rel_path: Path, mtime: float, source_name: str) -> None:
        """Update file cache with new mtime."""
        cache_key = f"{source_name}::{rel_path}" if source_name else str(rel_path)
        
        with self._connect() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO file_cache (cache_key, last_modified, source_name)
                VALUES (?, ?, ?)
            ''', (cache_key, mtime, source_name))

    def _remove_file_chunks(self, rel_path: Path, source_name: str) -> None:
        """Remove all chunks for a file."""
        # Use doc_id prefix matching - much faster than JSON extraction
        if source_name:
            prefix = f"{source_name}::{rel_path}::"
        else:
            prefix = f"{rel_path}::"
        
        with self._transaction() as conn:
            # Delete by doc_id prefix - uses index, no JSON parsing
            conn.execute('''
                DELETE FROM documents WHERE doc_id LIKE ?
            ''', (prefix + '%',))
            
            # Also delete from FTS5 table
            conn.execute('''
                DELETE FROM chunks_fts WHERE doc_id LIKE ?
            ''', (prefix + '%',))

    def _cleanup_deleted_files(self, base_path: Path, source_name: str) -> int:
        """Remove files from index that no longer exist."""
        deleted_count = 0
        prefix = f"{source_name}::" if source_name else ""
        
        with self._connect() as conn:
            # Get all cached files for this source
            if source_name:
                rows = conn.execute('''
                    SELECT cache_key FROM file_cache WHERE cache_key LIKE ?
                ''', (f"{source_name}::%",)).fetchall()
            else:
                rows = conn.execute('SELECT cache_key FROM file_cache').fetchall()
            
            for (cache_key,) in rows:
                # Extract relative path
                if source_name and cache_key.startswith(prefix):
                    rel_path = cache_key[len(prefix):]
                else:
                    rel_path = cache_key
                
                full_path = base_path / rel_path
                if not full_path.exists():
                    # Remove file chunks and cache entry
                    self._remove_file_chunks(Path(rel_path), source_name)
                    conn.execute('DELETE FROM file_cache WHERE cache_key = ?', (cache_key,))
                    deleted_count += 1
        
        if deleted_count > 0:
            self._index_dirty = True
        
        return deleted_count