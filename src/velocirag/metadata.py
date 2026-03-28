"""
Velocirag Metadata Store - Structured SQLite metadata layer for documents.

Document-level structured queries with tags, cross-references, usage tracking,
and flexible filtering. Third layer of the Velocirag search stack.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("velocirag.metadata")


class MetadataStoreError(Exception):
    """Base exception for metadata store errors."""
    pass


class MetadataStore:
    """
    SQLite-backed metadata store for document-level structured queries.
    
    Stores document metadata, tags, cross-references, and usage statistics.
    Provides flexible querying capabilities for document organization and discovery.
    """
    
    def __init__(self, db_path: str):
        """
        Initialize metadata store.
        
        Args:
            db_path: Path to SQLite database file (created if not exists)
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self.initialize()
        logger.info(f"MetadataStore initialized: {self.db_path}")
    
    def __enter__(self) -> 'MetadataStore':
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
    
    def initialize(self) -> None:
        """Create tables and indexes if they don't exist."""
        try:
            with self._transaction() as conn:
                # Documents table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS documents (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename TEXT UNIQUE NOT NULL,
                        title TEXT NOT NULL,
                        category TEXT,
                        status TEXT DEFAULT 'active',
                        project TEXT,
                        created_date TEXT,
                        updated_date TEXT,
                        meta JSON,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Tags table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS tags (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT UNIQUE NOT NULL
                    )
                ''')
                
                # Document-tags junction table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS document_tags (
                        doc_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                        tag_id INTEGER REFERENCES tags(id) ON DELETE CASCADE,
                        PRIMARY KEY (doc_id, tag_id)
                    )
                ''')
                
                # Cross-references table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS cross_refs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        doc_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                        ref_type TEXT NOT NULL,
                        ref_target TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Usage log table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS usage_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        doc_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                        action TEXT NOT NULL,
                        source TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Performance indexes
                conn.execute('CREATE INDEX IF NOT EXISTS idx_documents_category ON documents(category)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_documents_project ON documents(project)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_documents_created_date ON documents(created_date)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_documents_filename ON documents(filename)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_tags_name ON tags(name)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_usage_log_timestamp ON usage_log(timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_usage_log_doc_id ON usage_log(doc_id)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_cross_refs_doc_id ON cross_refs(doc_id)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_cross_refs_target ON cross_refs(ref_target)')
                
                # Enable foreign key constraints
                conn.execute('PRAGMA foreign_keys = ON')
                
        except sqlite3.Error as e:
            raise MetadataStoreError(f"Failed to initialize database: {e}")
    
    def upsert_document(self, filename: str, title: str, metadata: Dict) -> int:
        """
        Insert or update document metadata.
        
        Args:
            filename: Unique filename identifier
            title: Document title
            metadata: Dictionary with optional keys: category, status, project, 
                     created_date, updated_date, plus any custom fields in 'meta'
                     
        Returns:
            Document ID (integer)
        """
        if not filename or not filename.strip():
            raise ValueError("Filename cannot be empty")
        if not title or not title.strip():
            raise ValueError("Title cannot be empty")
        
        try:
            with self._transaction() as conn:
                # Extract known fields
                category = metadata.get('category')
                status = metadata.get('status', 'active')
                project = metadata.get('project')
                created_date = metadata.get('created_date')
                updated_date = metadata.get('updated_date')
                
                # Store custom fields in meta JSON
                meta_fields = {k: v for k, v in metadata.items() 
                              if k not in ('category', 'status', 'project', 'created_date', 'updated_date')}
                meta_json = json.dumps(meta_fields) if meta_fields else None
                
                # Check if document exists
                existing = conn.execute(
                    'SELECT id FROM documents WHERE filename = ?', (filename,)
                ).fetchone()
                
                if existing:
                    # Update existing document
                    conn.execute('''
                        UPDATE documents 
                        SET title = ?, category = ?, status = ?, project = ?, 
                            created_date = ?, updated_date = ?, meta = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE filename = ?
                    ''', (title, category, status, project, created_date, updated_date, meta_json, filename))
                    return existing[0]
                else:
                    # Insert new document
                    cursor = conn.execute('''
                        INSERT INTO documents 
                        (filename, title, category, status, project, created_date, updated_date, meta, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ''', (filename, title, category, status, project, created_date, updated_date, meta_json))
                    return cursor.lastrowid
                
        except sqlite3.Error as e:
            raise MetadataStoreError(f"Failed to upsert document '{filename}': {e}")
    
    def add_tags(self, doc_id: int, tags: List[str]) -> None:
        """
        Add tags to document.
        
        Args:
            doc_id: Document ID
            tags: List of tag names
        """
        if not tags:
            return
            
        # Normalize tags
        tags = [tag.strip().lower() for tag in tags if tag.strip()]
        if not tags:
            return
        
        try:
            with self._transaction() as conn:
                # Verify document exists
                doc_exists = conn.execute(
                    'SELECT 1 FROM documents WHERE id = ?', (doc_id,)
                ).fetchone()
                if not doc_exists:
                    raise MetadataStoreError(f"Document with ID {doc_id} does not exist")
                
                for tag in tags:
                    # Insert tag if not exists
                    conn.execute(
                        'INSERT OR IGNORE INTO tags (name) VALUES (?)', (tag,)
                    )
                    
                    # Get tag ID
                    tag_row = conn.execute(
                        'SELECT id FROM tags WHERE name = ?', (tag,)
                    ).fetchone()
                    
                    if tag_row:
                        tag_id = tag_row[0]
                        
                        # Insert document-tag relationship
                        conn.execute(
                            'INSERT OR IGNORE INTO document_tags (doc_id, tag_id) VALUES (?, ?)',
                            (doc_id, tag_id)
                        )
                        
        except sqlite3.Error as e:
            raise MetadataStoreError(f"Failed to add tags to document {doc_id}: {e}")
    
    def add_cross_ref(self, doc_id: int, ref_target: str, ref_type: str = 'references') -> None:
        """
        Add cross-reference to document.
        
        Args:
            doc_id: Source document ID
            ref_target: Target reference (filename, URL, etc.)
            ref_type: Type of reference (references, mentions, etc.)
        """
        if not ref_target or not ref_target.strip():
            raise ValueError("Reference target cannot be empty")
            
        try:
            with self._transaction() as conn:
                # Verify document exists
                doc_exists = conn.execute(
                    'SELECT 1 FROM documents WHERE id = ?', (doc_id,)
                ).fetchone()
                if not doc_exists:
                    raise MetadataStoreError(f"Document with ID {doc_id} does not exist")
                
                # Insert cross-reference
                conn.execute('''
                    INSERT INTO cross_refs (doc_id, ref_type, ref_target)
                    VALUES (?, ?, ?)
                ''', (doc_id, ref_type, ref_target.strip()))
                
        except sqlite3.Error as e:
            raise MetadataStoreError(f"Failed to add cross-reference to document {doc_id}: {e}")
    
    def log_usage(self, doc_id: int, action: str, source: str = None) -> None:
        """
        Log document usage event.
        
        Args:
            doc_id: Document ID
            action: Action type (search_hit, read, update, etc.)
            source: Optional source identifier
        """
        if not action or not action.strip():
            raise ValueError("Action cannot be empty")
            
        try:
            with self._transaction() as conn:
                # Verify document exists
                doc_exists = conn.execute(
                    'SELECT 1 FROM documents WHERE id = ?', (doc_id,)
                ).fetchone()
                if not doc_exists:
                    raise MetadataStoreError(f"Document with ID {doc_id} does not exist")
                
                # Insert usage log entry
                conn.execute('''
                    INSERT INTO usage_log (doc_id, action, source)
                    VALUES (?, ?, ?)
                ''', (doc_id, action.strip(), source))
                
        except sqlite3.Error as e:
            raise MetadataStoreError(f"Failed to log usage for document {doc_id}: {e}")
    
    def query(self, **filters) -> List[Dict]:
        """
        Flexible document query with multiple filter options.
        
        Args:
            tags: List[str] - documents with ANY of these tags
            status: str - filter by status
            category: str - filter by category
            project: str - filter by project
            created_after: str - date string YYYY-MM-DD
            created_before: str - date string YYYY-MM-DD
            stale_days: int - not accessed in N days
            limit: int - max results (default 50)
            
        Returns:
            List of document dictionaries
        """
        limit = filters.get('limit', 50)
        
        try:
            with self._connect() as conn:
                # Build query dynamically
                conditions = []
                params = []
                joins = []
                
                # Tag filter
                if 'tags' in filters and filters['tags']:
                    tag_names = [tag.strip().lower() for tag in filters['tags']]
                    placeholders = ','.join(['?'] * len(tag_names))
                    joins.append('''
                        JOIN document_tags dt ON d.id = dt.doc_id
                        JOIN tags t ON dt.tag_id = t.id
                    ''')
                    conditions.append(f't.name IN ({placeholders})')
                    params.extend(tag_names)
                
                # Basic filters
                if 'status' in filters and filters['status']:
                    conditions.append('d.status = ?')
                    params.append(filters['status'])
                
                if 'category' in filters and filters['category']:
                    conditions.append('d.category = ?')
                    params.append(filters['category'])
                
                if 'project' in filters and filters['project']:
                    conditions.append('d.project = ?')
                    params.append(filters['project'])
                
                # Date filters
                if 'created_after' in filters and filters['created_after']:
                    conditions.append('d.created_date >= ?')
                    params.append(filters['created_after'])
                
                if 'created_before' in filters and filters['created_before']:
                    conditions.append('d.created_date <= ?')
                    params.append(filters['created_before'])
                
                # Staleness filter
                if 'stale_days' in filters and filters['stale_days']:
                    stale_threshold = (datetime.now() - timedelta(days=filters['stale_days'])).isoformat()
                    conditions.append('''
                        d.id NOT IN (
                            SELECT DISTINCT doc_id FROM usage_log 
                            WHERE timestamp >= ?
                        )
                    ''')
                    params.append(stale_threshold)
                
                # Build final query
                query = 'SELECT DISTINCT d.* FROM documents d'
                if joins:
                    query += ' ' + ' '.join(joins)
                if conditions:
                    query += ' WHERE ' + ' AND '.join(conditions)
                query += ' ORDER BY d.updated_at DESC'
                if limit:
                    query += ' LIMIT ?'
                    params.append(limit)
                
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                # Convert to dictionaries
                columns = [desc[0] for desc in cursor.description]
                results = []
                
                for row in rows:
                    doc_dict = dict(zip(columns, row))
                    
                    # Parse JSON meta field
                    if doc_dict.get('meta'):
                        try:
                            doc_dict['meta'] = json.loads(doc_dict['meta'])
                        except json.JSONDecodeError:
                            doc_dict['meta'] = {}
                    else:
                        doc_dict['meta'] = {}
                    
                    results.append(doc_dict)
                
                return results
                
        except sqlite3.Error as e:
            raise MetadataStoreError(f"Query failed: {e}")
    
    def get_document(self, filename: str) -> Optional[Dict]:
        """
        Get full document information including tags and usage stats.
        
        Args:
            filename: Document filename
            
        Returns:
            Document dictionary with tags and usage stats, or None if not found
        """
        try:
            with self._connect() as conn:
                # Get document
                cursor = conn.execute('''
                    SELECT * FROM documents WHERE filename = ?
                ''', (filename,))
                doc_row = cursor.fetchone()
                
                if not doc_row:
                    return None
                
                # Convert to dict
                columns = [desc[0] for desc in cursor.description]
                doc_dict = dict(zip(columns, doc_row))
                
                # Parse JSON meta field
                if doc_dict.get('meta'):
                    try:
                        doc_dict['meta'] = json.loads(doc_dict['meta'])
                    except json.JSONDecodeError:
                        doc_dict['meta'] = {}
                else:
                    doc_dict['meta'] = {}
                
                doc_id = doc_dict['id']
                
                # Get tags
                tag_rows = conn.execute('''
                    SELECT t.name FROM tags t
                    JOIN document_tags dt ON t.id = dt.tag_id
                    WHERE dt.doc_id = ?
                    ORDER BY t.name
                ''', (doc_id,)).fetchall()
                doc_dict['tags'] = [row[0] for row in tag_rows]
                
                # Get cross-references
                ref_rows = conn.execute('''
                    SELECT ref_type, ref_target, created_at FROM cross_refs
                    WHERE doc_id = ?
                    ORDER BY created_at DESC
                ''', (doc_id,)).fetchall()
                doc_dict['cross_refs'] = [
                    {'type': row[0], 'target': row[1], 'created_at': row[2]}
                    for row in ref_rows
                ]
                
                # Get usage stats
                try:
                    usage_stats = conn.execute('''
                        SELECT 
                            COUNT(*) as total_usage,
                            COUNT(CASE WHEN action = 'search_hit' THEN 1 END) as search_hits,
                            COUNT(CASE WHEN action = 'read' THEN 1 END) as reads,
                            MAX(timestamp) as last_accessed
                        FROM usage_log
                        WHERE doc_id = ?
                    ''', (doc_id,)).fetchone()
                    
                    doc_dict['usage_stats'] = {
                        'total_usage': usage_stats[0] or 0,
                        'search_hits': usage_stats[1] or 0,
                        'reads': usage_stats[2] or 0,
                        'last_accessed': usage_stats[3]
                    }
                except sqlite3.OperationalError:
                    # Handle missing usage_log table
                    doc_dict['usage_stats'] = {
                        'total_usage': 0,
                        'search_hits': 0,
                        'reads': 0,
                        'last_accessed': None
                    }
                
                return doc_dict
                
        except sqlite3.Error as e:
            raise MetadataStoreError(f"Failed to get document '{filename}': {e}")
    
    def get_stale(self, days: int = 90) -> List[Dict]:
        """
        Get documents not accessed in N days.
        
        Args:
            days: Number of days for staleness threshold
            
        Returns:
            List of stale document dictionaries
        """
        if days <= 0:
            raise ValueError("Days must be positive")
            
        stale_threshold = (datetime.now() - timedelta(days=days)).isoformat()
        
        try:
            with self._connect() as conn:
                cursor = conn.execute('''
                    SELECT d.* FROM documents d
                    WHERE d.id NOT IN (
                        SELECT DISTINCT doc_id FROM usage_log 
                        WHERE timestamp >= ?
                    )
                    ORDER BY d.created_at DESC
                ''', (stale_threshold,))
                rows = cursor.fetchall()
                
                # Convert to dictionaries
                columns = [desc[0] for desc in cursor.description]
                results = []
                
                for row in rows:
                    doc_dict = dict(zip(columns, row))
                    
                    # Parse JSON meta field
                    if doc_dict.get('meta'):
                        try:
                            doc_dict['meta'] = json.loads(doc_dict['meta'])
                        except json.JSONDecodeError:
                            doc_dict['meta'] = {}
                    else:
                        doc_dict['meta'] = {}
                    
                    results.append(doc_dict)
                
                return results
                
        except sqlite3.Error as e:
            raise MetadataStoreError(f"Failed to get stale documents: {e}")
    
    def get_recent(self, days: int = 7) -> List[Dict]:
        """
        Get recently created documents.
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of recent document dictionaries
        """
        if days <= 0:
            raise ValueError("Days must be positive")
            
        recent_threshold = (datetime.now() - timedelta(days=days)).isoformat()
        
        try:
            with self._connect() as conn:
                cursor = conn.execute('''
                    SELECT * FROM documents 
                    WHERE created_at >= ?
                    ORDER BY created_at DESC
                ''', (recent_threshold,))
                rows = cursor.fetchall()
                
                # Convert to dictionaries
                columns = [desc[0] for desc in cursor.description]
                results = []
                
                for row in rows:
                    doc_dict = dict(zip(columns, row))
                    
                    # Parse JSON meta field
                    if doc_dict.get('meta'):
                        try:
                            doc_dict['meta'] = json.loads(doc_dict['meta'])
                        except json.JSONDecodeError:
                            doc_dict['meta'] = {}
                    else:
                        doc_dict['meta'] = {}
                    
                    results.append(doc_dict)
                
                return results
                
        except sqlite3.Error as e:
            raise MetadataStoreError(f"Failed to get recent documents: {e}")
    
    def stats(self) -> Dict:
        """
        Get overview statistics.
        
        Returns:
            Dictionary with total docs, tags, categories, usage counts
        """
        try:
            with self._connect() as conn:
                # Basic counts
                total_docs = conn.execute('SELECT COUNT(*) FROM documents').fetchone()[0]
                total_tags = conn.execute('SELECT COUNT(*) FROM tags').fetchone()[0]
                total_usage = conn.execute('SELECT COUNT(*) FROM usage_log').fetchone()[0]
                total_cross_refs = conn.execute('SELECT COUNT(*) FROM cross_refs').fetchone()[0]
                
                # Category breakdown
                category_rows = conn.execute('''
                    SELECT category, COUNT(*) FROM documents 
                    WHERE category IS NOT NULL 
                    GROUP BY category
                    ORDER BY COUNT(*) DESC
                ''').fetchall()
                categories = {row[0]: row[1] for row in category_rows}
                
                # Status breakdown  
                status_rows = conn.execute('''
                    SELECT status, COUNT(*) FROM documents 
                    GROUP BY status
                    ORDER BY COUNT(*) DESC
                ''').fetchall()
                statuses = {row[0]: row[1] for row in status_rows}
                
                # Project breakdown
                project_rows = conn.execute('''
                    SELECT project, COUNT(*) FROM documents 
                    WHERE project IS NOT NULL 
                    GROUP BY project
                    ORDER BY COUNT(*) DESC
                ''').fetchall()
                projects = {row[0]: row[1] for row in project_rows}
                
                # Top tags
                top_tag_rows = conn.execute('''
                    SELECT t.name, COUNT(*) as usage_count FROM tags t
                    JOIN document_tags dt ON t.id = dt.tag_id
                    GROUP BY t.name
                    ORDER BY usage_count DESC
                    LIMIT 10
                ''').fetchall()
                top_tags = {row[0]: row[1] for row in top_tag_rows}
                
                # Recent activity
                recent_activity = conn.execute('''
                    SELECT COUNT(*) FROM usage_log 
                    WHERE timestamp >= datetime('now', '-7 days')
                ''').fetchone()[0]
                
                # Database file size
                db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
                
                return {
                    'total_documents': total_docs,
                    'total_tags': total_tags,
                    'total_usage_events': total_usage,
                    'total_cross_refs': total_cross_refs,
                    'categories': categories,
                    'statuses': statuses,
                    'projects': projects,
                    'top_tags': top_tags,
                    'recent_activity_7d': recent_activity,
                    'db_path': str(self.db_path),
                    'db_size_bytes': db_size,
                    'db_size_mb': round(db_size / (1024 * 1024), 2)
                }
                
        except sqlite3.Error as e:
            raise MetadataStoreError(f"Failed to get stats: {e}")
    
    def remove_document(self, filename: str) -> bool:
        """Remove a document and all related data (tags, cross-refs, usage log).
        
        Foreign key CASCADE handles junction table cleanup automatically.
        Orphan tags (no remaining documents) are pruned.
        
        Args:
            filename: Document filename to remove
            
        Returns:
            True if document existed and was removed, False if not found
        """
        try:
            with self._transaction() as conn:
                conn.execute('PRAGMA foreign_keys = ON')
                result = conn.execute(
                    'DELETE FROM documents WHERE filename = ?', (filename,)
                )
                if result.rowcount > 0:
                    # Prune orphan tags (no documents reference them)
                    conn.execute('''
                        DELETE FROM tags WHERE id NOT IN (
                            SELECT DISTINCT tag_id FROM document_tags
                        )
                    ''')
                return result.rowcount > 0
        except sqlite3.Error as e:
            raise MetadataStoreError(f"Failed to remove document '{filename}': {e}")

    def remove_documents_by_prefix(self, prefix: str) -> int:
        """Remove all documents matching a filename prefix.
        
        Useful for bulk cleanup when a source directory is removed.
        
        Args:
            prefix: Filename prefix to match (e.g., 'mikoshi::')
            
        Returns:
            Number of documents removed
        """
        try:
            with self._transaction() as conn:
                conn.execute('PRAGMA foreign_keys = ON')
                result = conn.execute(
                    'DELETE FROM documents WHERE filename LIKE ?', (prefix + '%',)
                )
                return result.rowcount
        except sqlite3.Error as e:
            raise MetadataStoreError(f"Failed to remove documents with prefix '{prefix}': {e}")

    def close(self) -> None:
        """Clean shutdown."""
        # No persistent connections to close
        pass
    
    @contextmanager
    def _connect(self):
        """Get a SQLite connection that's properly closed after use."""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()
    
    @contextmanager
    def _transaction(self):
        """Context manager for SQLite transactions."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute('PRAGMA foreign_keys = ON')
            conn.execute('BEGIN')
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()