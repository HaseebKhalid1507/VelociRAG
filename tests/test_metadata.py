"""Tests for metadata.py module."""

import pytest
import json
import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from velocirag.metadata import MetadataStore, MetadataStoreError


class TestConstructorAndSetup:
    """Test MetadataStore initialization and setup."""
    
    def test_default_parameters(self, tmp_path):
        """Default constructor creates store with expected settings."""
        db_path = tmp_path / "test_metadata.db"
        store = MetadataStore(str(db_path))
        
        assert store.db_path == db_path
        assert store.db_path.exists()
        
        # Verify tables were created
        with sqlite3.connect(db_path) as conn:
            tables = conn.execute('''
                SELECT name FROM sqlite_master WHERE type='table'
            ''').fetchall()
            table_names = {row[0] for row in tables}
            
            expected_tables = {'documents', 'tags', 'document_tags', 'cross_refs', 'usage_log'}
            assert table_names >= expected_tables
        
        store.close()
    
    def test_directory_creation(self, tmp_path):
        """Creates parent directories if they don't exist."""
        db_path = tmp_path / "nested" / "dirs" / "metadata.db"
        assert not db_path.parent.exists()
        
        store = MetadataStore(str(db_path))
        assert db_path.parent.exists()
        assert db_path.exists()
        store.close()
    
    def test_context_manager(self, tmp_path):
        """Context manager properly opens and closes store."""
        db_path = tmp_path / "test_metadata.db"
        
        with MetadataStore(str(db_path)) as store:
            doc_id = store.upsert_document("test.md", "Test Document", {})
            assert doc_id > 0
        
        # After context exit, data should be persisted
        store2 = MetadataStore(str(db_path))
        doc = store2.get_document("test.md")
        assert doc is not None
        assert doc['title'] == "Test Document"
        store2.close()
    
    def test_foreign_key_constraints(self, tmp_path):
        """Foreign key constraints are enabled."""
        db_path = tmp_path / "test_metadata.db"
        store = MetadataStore(str(db_path))
        
        with pytest.raises(MetadataStoreError):
            # Try to add tag to non-existent document
            store.add_tags(999, ["test"])
        
        with pytest.raises(MetadataStoreError):
            # Try to log usage for non-existent document
            store.log_usage(999, "read")
        
        store.close()


class TestDocumentOperations:
    """Test document CRUD operations."""
    
    def test_upsert_document_basic(self, tmp_path):
        """Basic document insertion works."""
        db_path = tmp_path / "test_metadata.db"
        store = MetadataStore(str(db_path))
        
        doc_id = store.upsert_document(
            "test.md", 
            "Test Document",
            {
                'category': 'notes',
                'status': 'active',
                'project': 'test-project',
                'custom_field': 'custom_value'
            }
        )
        
        assert doc_id > 0
        
        # Verify document was stored
        doc = store.get_document("test.md")
        assert doc is not None
        assert doc['filename'] == "test.md"
        assert doc['title'] == "Test Document"
        assert doc['category'] == 'notes'
        assert doc['status'] == 'active'
        assert doc['project'] == 'test-project'
        assert doc['meta']['custom_field'] == 'custom_value'
        
        store.close()
    
    def test_upsert_document_update(self, tmp_path):
        """Upserting existing document updates it."""
        db_path = tmp_path / "test_metadata.db"
        store = MetadataStore(str(db_path))
        
        # Insert first version
        doc_id1 = store.upsert_document("test.md", "Original Title", {'category': 'draft'})
        
        # Update same filename
        doc_id2 = store.upsert_document("test.md", "Updated Title", {'category': 'published'})
        
        # Should be same document ID
        assert doc_id1 == doc_id2
        
        # Verify update
        doc = store.get_document("test.md")
        assert doc['title'] == "Updated Title"
        assert doc['category'] == 'published'
        
        store.close()
    
    def test_upsert_document_validation(self, tmp_path):
        """Document validation rejects invalid input."""
        db_path = tmp_path / "test_metadata.db"
        store = MetadataStore(str(db_path))
        
        # Empty filename
        with pytest.raises(ValueError, match="Filename cannot be empty"):
            store.upsert_document("", "Title", {})
        
        with pytest.raises(ValueError, match="Filename cannot be empty"):
            store.upsert_document("   ", "Title", {})
        
        # Empty title
        with pytest.raises(ValueError, match="Title cannot be empty"):
            store.upsert_document("file.md", "", {})
        
        with pytest.raises(ValueError, match="Title cannot be empty"):
            store.upsert_document("file.md", "   ", {})
        
        store.close()
    
    def test_get_document_not_found(self, tmp_path):
        """Getting non-existent document returns None."""
        db_path = tmp_path / "test_metadata.db"
        store = MetadataStore(str(db_path))
        
        doc = store.get_document("nonexistent.md")
        assert doc is None
        
        store.close()
    
    def test_get_document_full_info(self, tmp_path):
        """get_document returns complete document information."""
        db_path = tmp_path / "test_metadata.db"
        store = MetadataStore(str(db_path))
        
        # Create document with metadata
        doc_id = store.upsert_document("test.md", "Test Doc", {
            'category': 'notes',
            'custom': 'value'
        })
        
        # Add tags
        store.add_tags(doc_id, ["tag1", "tag2"])
        
        # Add cross-refs
        store.add_cross_ref(doc_id, "other.md", "references")
        store.add_cross_ref(doc_id, "https://example.com", "external_link")
        
        # Add usage events
        store.log_usage(doc_id, "search_hit", "test_query")
        store.log_usage(doc_id, "read", "user")
        
        # Get full document
        doc = store.get_document("test.md")
        
        assert doc['filename'] == "test.md"
        assert doc['title'] == "Test Doc"
        assert doc['category'] == 'notes'
        assert doc['meta']['custom'] == 'value'
        
        # Check tags
        assert set(doc['tags']) == {"tag1", "tag2"}
        
        # Check cross-refs
        assert len(doc['cross_refs']) == 2
        ref_targets = {ref['target'] for ref in doc['cross_refs']}
        assert ref_targets == {"other.md", "https://example.com"}
        
        # Check usage stats
        assert doc['usage_stats']['total_usage'] == 2
        assert doc['usage_stats']['search_hits'] == 1
        assert doc['usage_stats']['reads'] == 1
        assert doc['usage_stats']['last_accessed'] is not None
        
        store.close()


class TestTagOperations:
    """Test tag management operations."""
    
    def test_add_tags_basic(self, tmp_path):
        """Basic tag addition works."""
        db_path = tmp_path / "test_metadata.db"
        store = MetadataStore(str(db_path))
        
        doc_id = store.upsert_document("test.md", "Test", {})
        store.add_tags(doc_id, ["python", "programming", "tutorial"])
        
        doc = store.get_document("test.md")
        assert set(doc['tags']) == {"python", "programming", "tutorial"}
        
        store.close()
    
    def test_add_tags_normalization(self, tmp_path):
        """Tags are normalized (lowercased, trimmed)."""
        db_path = tmp_path / "test_metadata.db"
        store = MetadataStore(str(db_path))
        
        doc_id = store.upsert_document("test.md", "Test", {})
        store.add_tags(doc_id, ["  Python  ", "PROGRAMMING", "Tutorial"])
        
        doc = store.get_document("test.md")
        assert set(doc['tags']) == {"python", "programming", "tutorial"}
        
        store.close()
    
    def test_add_tags_deduplication(self, tmp_path):
        """Duplicate tags are handled gracefully."""
        db_path = tmp_path / "test_metadata.db"
        store = MetadataStore(str(db_path))
        
        doc_id = store.upsert_document("test.md", "Test", {})
        
        # Add tags multiple times
        store.add_tags(doc_id, ["python", "tutorial"])
        store.add_tags(doc_id, ["python", "programming"])  # python already exists
        store.add_tags(doc_id, ["tutorial", "advanced"])   # tutorial already exists
        
        doc = store.get_document("test.md")
        assert set(doc['tags']) == {"python", "tutorial", "programming", "advanced"}
        
        store.close()
    
    def test_add_tags_empty_lists(self, tmp_path):
        """Empty tag lists are handled gracefully."""
        db_path = tmp_path / "test_metadata.db"
        store = MetadataStore(str(db_path))
        
        doc_id = store.upsert_document("test.md", "Test", {})
        
        # These should not crash
        store.add_tags(doc_id, [])
        store.add_tags(doc_id, ["", "  ", ""])
        store.add_tags(doc_id, None)  # This would be handled by type checking
        
        doc = store.get_document("test.md")
        assert doc['tags'] == []
        
        store.close()
    
    def test_add_tags_invalid_doc_id(self, tmp_path):
        """Adding tags to non-existent document raises error."""
        db_path = tmp_path / "test_metadata.db"
        store = MetadataStore(str(db_path))
        
        with pytest.raises(MetadataStoreError, match="Document with ID 999 does not exist"):
            store.add_tags(999, ["tag"])
        
        store.close()


class TestCrossReferences:
    """Test cross-reference operations."""
    
    def test_add_cross_ref_basic(self, tmp_path):
        """Basic cross-reference addition works."""
        db_path = tmp_path / "test_metadata.db"
        store = MetadataStore(str(db_path))
        
        doc_id = store.upsert_document("test.md", "Test", {})
        
        store.add_cross_ref(doc_id, "other.md", "references")
        store.add_cross_ref(doc_id, "https://example.com", "external_link")
        
        doc = store.get_document("test.md")
        assert len(doc['cross_refs']) == 2
        
        # Check content
        targets = {ref['target'] for ref in doc['cross_refs']}
        types = {ref['type'] for ref in doc['cross_refs']}
        
        assert targets == {"other.md", "https://example.com"}
        assert types == {"references", "external_link"}
        
        store.close()
    
    def test_add_cross_ref_default_type(self, tmp_path):
        """Cross-reference uses default type when not specified."""
        db_path = tmp_path / "test_metadata.db"
        store = MetadataStore(str(db_path))
        
        doc_id = store.upsert_document("test.md", "Test", {})
        store.add_cross_ref(doc_id, "other.md")  # No type specified
        
        doc = store.get_document("test.md")
        assert doc['cross_refs'][0]['type'] == 'references'
        
        store.close()
    
    def test_add_cross_ref_validation(self, tmp_path):
        """Cross-reference validation rejects invalid input."""
        db_path = tmp_path / "test_metadata.db"
        store = MetadataStore(str(db_path))
        
        doc_id = store.upsert_document("test.md", "Test", {})
        
        # Empty target
        with pytest.raises(ValueError, match="Reference target cannot be empty"):
            store.add_cross_ref(doc_id, "")
        
        with pytest.raises(ValueError, match="Reference target cannot be empty"):
            store.add_cross_ref(doc_id, "   ")
        
        # Invalid doc ID
        with pytest.raises(MetadataStoreError, match="Document with ID 999 does not exist"):
            store.add_cross_ref(999, "target")
        
        store.close()


class TestUsageLogging:
    """Test usage event logging."""
    
    def test_log_usage_basic(self, tmp_path):
        """Basic usage logging works."""
        db_path = tmp_path / "test_metadata.db"
        store = MetadataStore(str(db_path))
        
        doc_id = store.upsert_document("test.md", "Test", {})
        
        store.log_usage(doc_id, "search_hit", "test query")
        store.log_usage(doc_id, "read", "user")
        store.log_usage(doc_id, "update")  # No source
        
        doc = store.get_document("test.md")
        stats = doc['usage_stats']
        
        assert stats['total_usage'] == 3
        assert stats['search_hits'] == 1
        assert stats['reads'] == 1
        assert stats['last_accessed'] is not None
        
        store.close()
    
    def test_log_usage_validation(self, tmp_path):
        """Usage logging validation rejects invalid input."""
        db_path = tmp_path / "test_metadata.db"
        store = MetadataStore(str(db_path))
        
        doc_id = store.upsert_document("test.md", "Test", {})
        
        # Empty action
        with pytest.raises(ValueError, match="Action cannot be empty"):
            store.log_usage(doc_id, "")
        
        with pytest.raises(ValueError, match="Action cannot be empty"):
            store.log_usage(doc_id, "   ")
        
        # Invalid doc ID
        with pytest.raises(MetadataStoreError, match="Document with ID 999 does not exist"):
            store.log_usage(999, "read")
        
        store.close()


class TestQueryOperations:
    """Test document querying capabilities."""
    
    @pytest.fixture
    def populated_store(self, tmp_path):
        """Create a store with test data."""
        db_path = tmp_path / "test_metadata.db"
        store = MetadataStore(str(db_path))
        
        # Create test documents
        docs = [
            ("doc1.md", "Python Tutorial", {
                'category': 'tutorial', 'status': 'published', 'project': 'learning'
            }),
            ("doc2.md", "JavaScript Guide", {
                'category': 'tutorial', 'status': 'draft', 'project': 'learning'
            }),
            ("doc3.md", "Project Notes", {
                'category': 'notes', 'status': 'active', 'project': 'work'
            }),
            ("doc4.md", "Old Draft", {
                'category': 'tutorial', 'status': 'archived', 'project': 'old'
            })
        ]
        
        for filename, title, metadata in docs:
            doc_id = store.upsert_document(filename, title, metadata)
            
            # Add some tags
            if 'python' in title.lower():
                store.add_tags(doc_id, ['python', 'programming'])
            elif 'javascript' in title.lower():
                store.add_tags(doc_id, ['javascript', 'programming'])
            elif 'notes' in title.lower():
                store.add_tags(doc_id, ['notes', 'work'])
        
        return store
    
    def test_query_by_category(self, populated_store):
        """Query by category filter."""
        results = populated_store.query(category='tutorial')
        assert len(results) == 3
        
        titles = {doc['title'] for doc in results}
        assert 'Python Tutorial' in titles
        assert 'JavaScript Guide' in titles
        assert 'Old Draft' in titles
        
        populated_store.close()
    
    def test_query_by_status(self, populated_store):
        """Query by status filter."""
        results = populated_store.query(status='published')
        assert len(results) == 1
        assert results[0]['title'] == 'Python Tutorial'
        
        populated_store.close()
    
    def test_query_by_project(self, populated_store):
        """Query by project filter."""
        results = populated_store.query(project='learning')
        assert len(results) == 2
        
        titles = {doc['title'] for doc in results}
        assert titles == {'Python Tutorial', 'JavaScript Guide'}
        
        populated_store.close()
    
    def test_query_by_tags(self, populated_store):
        """Query by tags filter."""
        # Single tag
        results = populated_store.query(tags=['python'])
        assert len(results) == 1
        assert results[0]['title'] == 'Python Tutorial'
        
        # Multiple tags (ANY match)
        results = populated_store.query(tags=['python', 'work'])
        assert len(results) == 2
        
        titles = {doc['title'] for doc in results}
        assert titles == {'Python Tutorial', 'Project Notes'}
        
        populated_store.close()
    
    def test_query_combined_filters(self, populated_store):
        """Query with multiple filters combined."""
        results = populated_store.query(
            category='tutorial',
            status='published',
            tags=['programming']
        )
        assert len(results) == 1
        assert results[0]['title'] == 'Python Tutorial'
        
        populated_store.close()
    
    def test_query_limit(self, populated_store):
        """Query respects limit parameter."""
        results = populated_store.query(limit=2)
        assert len(results) <= 2
        
        populated_store.close()
    
    def test_query_no_results(self, populated_store):
        """Query with no matching results."""
        results = populated_store.query(category='nonexistent')
        assert results == []
        
        populated_store.close()


class TestStalenessOperations:
    """Test staleness detection operations."""
    
    def test_get_stale_basic(self, tmp_path):
        """Basic staleness detection works."""
        db_path = tmp_path / "test_metadata.db"
        store = MetadataStore(str(db_path))
        
        # Create documents
        doc1_id = store.upsert_document("fresh.md", "Fresh Doc", {})
        doc2_id = store.upsert_document("stale.md", "Stale Doc", {})
        
        # Log recent usage for one document
        store.log_usage(doc1_id, "read")
        
        # Get stale documents (1 day threshold)
        stale_docs = store.get_stale(days=1)
        
        # Should find the document without recent usage
        assert len(stale_docs) == 1
        assert stale_docs[0]['filename'] == 'stale.md'
        
        store.close()
    
    def test_get_stale_validation(self, tmp_path):
        """Staleness validation rejects invalid input."""
        db_path = tmp_path / "test_metadata.db"
        store = MetadataStore(str(db_path))
        
        with pytest.raises(ValueError, match="Days must be positive"):
            store.get_stale(days=0)
        
        with pytest.raises(ValueError, match="Days must be positive"):
            store.get_stale(days=-1)
        
        store.close()
    
    def test_get_recent_basic(self, tmp_path):
        """Basic recent document detection works."""
        db_path = tmp_path / "test_metadata.db"
        store = MetadataStore(str(db_path))
        
        # Create document
        store.upsert_document("recent.md", "Recent Doc", {})
        
        # Get recent documents
        recent_docs = store.get_recent(days=1)
        
        # Should find the just-created document
        assert len(recent_docs) == 1
        assert recent_docs[0]['filename'] == 'recent.md'
        
        store.close()
    
    def test_get_recent_validation(self, tmp_path):
        """Recent document validation rejects invalid input."""
        db_path = tmp_path / "test_metadata.db"
        store = MetadataStore(str(db_path))
        
        with pytest.raises(ValueError, match="Days must be positive"):
            store.get_recent(days=0)
        
        with pytest.raises(ValueError, match="Days must be positive"):
            store.get_recent(days=-1)
        
        store.close()


class TestStatistics:
    """Test statistics and reporting."""
    
    def test_stats_empty_database(self, tmp_path):
        """Statistics work on empty database."""
        db_path = tmp_path / "test_metadata.db"
        store = MetadataStore(str(db_path))
        
        stats = store.stats()
        
        assert stats['total_documents'] == 0
        assert stats['total_tags'] == 0
        assert stats['total_usage_events'] == 0
        assert stats['total_cross_refs'] == 0
        assert stats['categories'] == {}
        assert stats['statuses'] == {}
        assert stats['projects'] == {}
        assert stats['top_tags'] == {}
        assert stats['recent_activity_7d'] == 0
        assert 'db_path' in stats
        assert 'db_size_mb' in stats
        
        store.close()
    
    def test_stats_populated_database(self, tmp_path):
        """Statistics work on populated database."""
        db_path = tmp_path / "test_metadata.db"
        store = MetadataStore(str(db_path))
        
        # Create test data
        doc1_id = store.upsert_document("doc1.md", "Doc 1", {
            'category': 'tutorial', 'status': 'published', 'project': 'learning'
        })
        doc2_id = store.upsert_document("doc2.md", "Doc 2", {
            'category': 'notes', 'status': 'published', 'project': 'learning'  
        })
        
        # Add tags
        store.add_tags(doc1_id, ['python', 'programming'])
        store.add_tags(doc2_id, ['python', 'notes'])
        
        # Add usage events
        store.log_usage(doc1_id, 'read')
        store.log_usage(doc2_id, 'search_hit')
        
        # Add cross-refs
        store.add_cross_ref(doc1_id, 'other.md')
        
        stats = store.stats()
        
        assert stats['total_documents'] == 2
        assert stats['total_tags'] == 3  # python, programming, notes
        assert stats['total_usage_events'] == 2
        assert stats['total_cross_refs'] == 1
        
        assert stats['categories'] == {'tutorial': 1, 'notes': 1}
        assert stats['statuses'] == {'published': 2}
        assert stats['projects'] == {'learning': 2}
        
        # Top tags should include python (used 2 times)
        assert 'python' in stats['top_tags']
        assert stats['top_tags']['python'] == 2
        
        assert stats['recent_activity_7d'] == 2
        
        store.close()


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_database_error_handling(self, tmp_path):
        """Database errors are properly wrapped."""
        db_path = tmp_path / "test_metadata.db"
        store = MetadataStore(str(db_path))
        
        # Manually corrupt the database to trigger errors
        with sqlite3.connect(db_path) as conn:
            conn.execute("DROP TABLE documents")
        
        # Operations should raise MetadataStoreError
        with pytest.raises(MetadataStoreError):
            store.upsert_document("test.md", "Test", {})
        
        store.close()
    
    def test_json_parsing_robustness(self, tmp_path):
        """JSON parsing handles corrupted data gracefully."""
        db_path = tmp_path / "test_metadata.db"
        store = MetadataStore(str(db_path))
        
        # Manually insert document with invalid JSON
        with sqlite3.connect(db_path) as conn:
            conn.execute('''
                INSERT INTO documents (filename, title, meta)
                VALUES (?, ?, ?)
            ''', ("test.md", "Test", "invalid json {"))
        
        # Should handle corrupted JSON gracefully
        doc = store.get_document("test.md")
        assert doc is not None
        assert doc['meta'] == {}  # Should default to empty dict
        
        store.close()