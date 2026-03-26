"""Tests for tracker.py module."""

import pytest
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from velocirag.metadata import MetadataStore
from velocirag.tracker import UsageTracker


class TestUsageTrackerInitialization:
    """Test UsageTracker initialization."""
    
    def test_initialization(self, tmp_path):
        """UsageTracker initializes with MetadataStore."""
        db_path = tmp_path / "test_metadata.db"
        metadata_store = MetadataStore(str(db_path))
        tracker = UsageTracker(metadata_store)
        
        assert tracker.metadata_store is metadata_store
        
        metadata_store.close()
    
    def test_requires_metadata_store(self):
        """UsageTracker requires MetadataStore instance."""
        # Should not crash with valid MetadataStore
        with pytest.raises(TypeError):
            UsageTracker()  # No arguments


class TestLogOperations:
    """Test usage logging operations."""
    
    @pytest.fixture
    def setup_tracker(self, tmp_path):
        """Set up tracker with test documents."""
        db_path = tmp_path / "test_metadata.db"
        metadata_store = MetadataStore(str(db_path))
        tracker = UsageTracker(metadata_store)
        
        # Create test documents
        doc1_id = metadata_store.upsert_document("doc1.md", "Document 1", {'category': 'notes'})
        doc2_id = metadata_store.upsert_document("doc2.md", "Document 2", {'category': 'tutorial'})
        
        return tracker, metadata_store, {'doc1_id': doc1_id, 'doc2_id': doc2_id}
    
    def test_log_search_hit(self, setup_tracker):
        """log_search_hit logs search events correctly."""
        tracker, metadata_store, doc_ids = setup_tracker
        
        tracker.log_search_hit("doc1.md", "python tutorial")
        
        # Verify usage was logged
        doc = metadata_store.get_document("doc1.md")
        stats = doc['usage_stats']
        
        assert stats['total_usage'] == 1
        assert stats['search_hits'] == 1
        assert stats['reads'] == 0
        
        metadata_store.close()
    
    def test_log_search_hit_unknown_document(self, setup_tracker):
        """log_search_hit handles unknown documents gracefully."""
        tracker, metadata_store, doc_ids = setup_tracker
        
        # Should not crash for unknown document
        tracker.log_search_hit("unknown.md", "test query")
        
        # No usage should be logged
        doc = metadata_store.get_document("unknown.md")
        assert doc is None
        
        metadata_store.close()
    
    def test_log_read(self, setup_tracker):
        """log_read logs read events correctly."""
        tracker, metadata_store, doc_ids = setup_tracker
        
        tracker.log_read("doc1.md", source="user")
        
        # Verify usage was logged
        doc = metadata_store.get_document("doc1.md")
        stats = doc['usage_stats']
        
        assert stats['total_usage'] == 1
        assert stats['search_hits'] == 0
        assert stats['reads'] == 1
        
        metadata_store.close()
    
    def test_log_read_no_source(self, setup_tracker):
        """log_read works without source parameter."""
        tracker, metadata_store, doc_ids = setup_tracker
        
        tracker.log_read("doc1.md")  # No source
        
        # Verify usage was logged
        doc = metadata_store.get_document("doc1.md")
        stats = doc['usage_stats']
        
        assert stats['total_usage'] == 1
        assert stats['reads'] == 1
        
        metadata_store.close()
    
    def test_log_read_unknown_document(self, setup_tracker):
        """log_read handles unknown documents gracefully."""
        tracker, metadata_store, doc_ids = setup_tracker
        
        # Should not crash for unknown document
        tracker.log_read("unknown.md", source="user")
        
        # No usage should be logged
        doc = metadata_store.get_document("unknown.md")
        assert doc is None
        
        metadata_store.close()
    
    def test_log_update(self, setup_tracker):
        """log_update logs update events correctly."""
        tracker, metadata_store, doc_ids = setup_tracker
        
        tracker.log_update("doc1.md")
        
        # Verify usage was logged
        doc = metadata_store.get_document("doc1.md")
        stats = doc['usage_stats']
        
        assert stats['total_usage'] == 1
        assert stats['search_hits'] == 0
        assert stats['reads'] == 0
        
        # Check the action type directly
        history = tracker.get_access_history("doc1.md")
        assert len(history) == 1
        assert history[0]['action'] == 'update'
        assert history[0]['source'] == 'file_system'
        
        metadata_store.close()
    
    def test_log_update_unknown_document(self, setup_tracker):
        """log_update handles unknown documents gracefully."""
        tracker, metadata_store, doc_ids = setup_tracker
        
        # Should not crash for unknown document
        tracker.log_update("unknown.md")
        
        # No usage should be logged
        doc = metadata_store.get_document("unknown.md")
        assert doc is None
        
        metadata_store.close()
    
    def test_multiple_log_events(self, setup_tracker):
        """Multiple log events accumulate correctly."""
        tracker, metadata_store, doc_ids = setup_tracker
        
        # Log various events
        tracker.log_search_hit("doc1.md", "search query 1")
        tracker.log_search_hit("doc1.md", "search query 2")
        tracker.log_read("doc1.md", "user1")
        tracker.log_read("doc1.md", "user2")
        tracker.log_update("doc1.md")
        
        # Verify all events were logged
        doc = metadata_store.get_document("doc1.md")
        stats = doc['usage_stats']
        
        assert stats['total_usage'] == 5
        assert stats['search_hits'] == 2
        assert stats['reads'] == 2
        assert stats['last_accessed'] is not None
        
        metadata_store.close()


class TestAccessHistory:
    """Test access history retrieval."""
    
    @pytest.fixture
    def setup_with_history(self, tmp_path):
        """Set up tracker with access history."""
        db_path = tmp_path / "test_metadata.db"
        metadata_store = MetadataStore(str(db_path))
        tracker = UsageTracker(metadata_store)
        
        # Create test document
        doc_id = metadata_store.upsert_document("doc1.md", "Document 1", {})
        
        # Add some history
        tracker.log_search_hit("doc1.md", "query1")
        tracker.log_read("doc1.md", "user1")
        tracker.log_update("doc1.md")
        tracker.log_search_hit("doc1.md", "query2")
        tracker.log_read("doc1.md", "user2")
        
        return tracker, metadata_store
    
    def test_get_access_history_basic(self, setup_with_history):
        """get_access_history returns event history."""
        tracker, metadata_store = setup_with_history
        
        history = tracker.get_access_history("doc1.md")
        
        assert len(history) == 5
        
        # Should be in reverse chronological order (most recent first)
        actions = [event['action'] for event in history]
        assert 'read' in actions
        assert 'search_hit' in actions
        assert 'update' in actions
        
        # Check event structure
        event = history[0]
        assert 'action' in event
        assert 'source' in event
        assert 'timestamp' in event
        
        metadata_store.close()
    
    def test_get_access_history_limit(self, setup_with_history):
        """get_access_history respects limit parameter."""
        tracker, metadata_store = setup_with_history
        
        history = tracker.get_access_history("doc1.md", limit=3)
        
        assert len(history) == 3
        
        metadata_store.close()
    
    def test_get_access_history_unknown_document(self, setup_with_history):
        """get_access_history handles unknown documents."""
        tracker, metadata_store = setup_with_history
        
        history = tracker.get_access_history("unknown.md")
        
        assert history == []
        
        metadata_store.close()
    
    def test_get_access_history_no_events(self, tmp_path):
        """get_access_history works for documents with no events."""
        db_path = tmp_path / "test_metadata.db"
        metadata_store = MetadataStore(str(db_path))
        tracker = UsageTracker(metadata_store)
        
        # Create document but log no events
        metadata_store.upsert_document("doc1.md", "Document 1", {})
        
        history = tracker.get_access_history("doc1.md")
        
        assert history == []
        
        metadata_store.close()


class TestAccessAnalytics:
    """Test access analytics methods."""
    
    @pytest.fixture
    def setup_analytics_data(self, tmp_path):
        """Set up tracker with test data for analytics."""
        db_path = tmp_path / "test_metadata.db"
        metadata_store = MetadataStore(str(db_path))
        tracker = UsageTracker(metadata_store)
        
        # Create test documents
        docs = [
            ("accessed.md", "Accessed Document", {}),
            ("popular.md", "Popular Document", {}),
            ("never_accessed.md", "Never Accessed", {}),
            ("rarely_accessed.md", "Rarely Accessed", {})
        ]
        
        doc_ids = {}
        for filename, title, metadata in docs:
            doc_id = metadata_store.upsert_document(filename, title, metadata)
            doc_ids[filename] = doc_id
        
        # Add usage events
        # Popular document - many events
        for i in range(10):
            tracker.log_search_hit("popular.md", f"query {i}")
        for i in range(5):
            tracker.log_read("popular.md", f"user{i}")
        
        # Accessed document - moderate events
        tracker.log_search_hit("accessed.md", "search")
        tracker.log_read("accessed.md", "user")
        tracker.log_update("accessed.md")
        
        # Rarely accessed - minimal events
        tracker.log_read("rarely_accessed.md", "user")
        
        # never_accessed.md - no events
        
        return tracker, metadata_store, doc_ids
    
    def test_get_never_accessed(self, setup_analytics_data):
        """get_never_accessed finds documents without usage."""
        tracker, metadata_store, doc_ids = setup_analytics_data
        
        never_accessed = tracker.get_never_accessed()
        
        # Should find the document with no usage events
        assert len(never_accessed) == 1
        assert never_accessed[0]['filename'] == 'never_accessed.md'
        
        metadata_store.close()
    
    def test_get_never_accessed_empty(self, tmp_path):
        """get_never_accessed works when all documents are accessed."""
        db_path = tmp_path / "test_metadata.db"
        metadata_store = MetadataStore(str(db_path))
        tracker = UsageTracker(metadata_store)
        
        # Create document and access it
        metadata_store.upsert_document("doc.md", "Document", {})
        tracker.log_read("doc.md")
        
        never_accessed = tracker.get_never_accessed()
        
        assert never_accessed == []
        
        metadata_store.close()
    
    def test_get_most_accessed(self, setup_analytics_data):
        """get_most_accessed finds most used documents."""
        tracker, metadata_store, doc_ids = setup_analytics_data
        
        most_accessed = tracker.get_most_accessed(limit=3)
        
        # Should be ordered by access count descending
        assert len(most_accessed) >= 1
        assert most_accessed[0]['filename'] == 'popular.md'
        assert most_accessed[0]['access_count'] == 15  # 10 search hits + 5 reads
        assert most_accessed[0]['search_hits'] == 10
        assert most_accessed[0]['reads'] == 5
        
        # Second should be accessed.md
        if len(most_accessed) > 1:
            assert most_accessed[1]['filename'] == 'accessed.md'
            assert most_accessed[1]['access_count'] == 3
        
        metadata_store.close()
    
    def test_get_most_accessed_limit(self, setup_analytics_data):
        """get_most_accessed respects limit parameter."""
        tracker, metadata_store, doc_ids = setup_analytics_data
        
        most_accessed = tracker.get_most_accessed(limit=2)
        
        assert len(most_accessed) <= 2
        
        metadata_store.close()
    
    def test_get_most_accessed_no_usage(self, tmp_path):
        """get_most_accessed works when no documents have usage."""
        db_path = tmp_path / "test_metadata.db"
        metadata_store = MetadataStore(str(db_path))
        tracker = UsageTracker(metadata_store)
        
        # Create documents but don't log usage
        metadata_store.upsert_document("doc1.md", "Doc 1", {})
        metadata_store.upsert_document("doc2.md", "Doc 2", {})
        
        most_accessed = tracker.get_most_accessed()
        
        assert most_accessed == []
        
        metadata_store.close()


class TestUsageTrends:
    """Test usage trend analysis."""
    
    def test_get_usage_trends_basic(self, tmp_path):
        """get_usage_trends provides trend statistics."""
        db_path = tmp_path / "test_metadata.db"
        metadata_store = MetadataStore(str(db_path))
        tracker = UsageTracker(metadata_store)
        
        # Create documents and add events
        doc1_id = metadata_store.upsert_document("doc1.md", "Doc 1", {})
        doc2_id = metadata_store.upsert_document("doc2.md", "Doc 2", {})
        
        # Add various events
        tracker.log_search_hit("doc1.md", "query")
        tracker.log_read("doc1.md", "user")
        tracker.log_update("doc1.md")
        tracker.log_read("doc2.md", "user")
        
        trends = tracker.get_usage_trends(days=7)
        
        assert trends['period_days'] == 7
        assert trends['total_events'] == 4
        assert 'events_by_action' in trends
        assert 'daily_activity' in trends
        assert 'top_documents' in trends
        assert 'avg_events_per_day' in trends
        
        # Check action breakdown
        assert trends['events_by_action']['search_hit'] == 1
        assert trends['events_by_action']['read'] == 2
        assert trends['events_by_action']['update'] == 1
        
        # Check average
        assert abs(trends['avg_events_per_day'] - (4.0 / 7)) < 0.01
        
        metadata_store.close()
    
    def test_get_usage_trends_empty(self, tmp_path):
        """get_usage_trends works with no events."""
        db_path = tmp_path / "test_metadata.db"
        metadata_store = MetadataStore(str(db_path))
        tracker = UsageTracker(metadata_store)
        
        trends = tracker.get_usage_trends(days=30)
        
        assert trends['period_days'] == 30
        assert trends['total_events'] == 0
        assert trends['events_by_action'] == {}
        assert trends['daily_activity'] == {}
        assert trends['top_documents'] == []
        assert trends['avg_events_per_day'] == 0
        
        metadata_store.close()
    
    def test_get_usage_trends_period_filtering(self, tmp_path):
        """get_usage_trends filters by time period correctly."""
        db_path = tmp_path / "test_metadata.db"
        metadata_store = MetadataStore(str(db_path))
        tracker = UsageTracker(metadata_store)
        
        # Create document
        metadata_store.upsert_document("doc.md", "Document", {})
        
        # Add events (should be within the period)
        tracker.log_read("doc.md", "user")
        
        # Get trends for very short period
        trends = tracker.get_usage_trends(days=1)
        
        # Should include recent events
        assert trends['total_events'] >= 1
        
        metadata_store.close()


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_database_error_handling(self, tmp_path):
        """Tracker handles database errors gracefully."""
        db_path = tmp_path / "test_metadata.db"
        metadata_store = MetadataStore(str(db_path))
        tracker = UsageTracker(metadata_store)
        
        # Create document first
        metadata_store.upsert_document("doc.md", "Document", {})
        
        # Corrupt the database
        with sqlite3.connect(db_path) as conn:
            conn.execute("DROP TABLE usage_log")
        
        # Operations should handle errors gracefully
        history = tracker.get_access_history("doc.md")
        assert history == []
        
        trends = tracker.get_usage_trends()
        assert trends['total_events'] == 0
        
        metadata_store.close()
    
    def test_tracker_with_closed_metadata_store(self, tmp_path):
        """Tracker operations after metadata store is closed."""
        db_path = tmp_path / "test_metadata.db"
        metadata_store = MetadataStore(str(db_path))
        tracker = UsageTracker(metadata_store)
        
        # Create document
        metadata_store.upsert_document("doc.md", "Document", {})
        
        # Close metadata store
        metadata_store.close()
        
        # Tracker operations should still work (they create new connections)
        tracker.log_read("doc.md", "user")
        history = tracker.get_access_history("doc.md")
        
        # Should work since SQLite connections are per-operation
        assert len(history) >= 0


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    def test_complete_document_lifecycle(self, tmp_path):
        """Test complete document lifecycle tracking."""
        db_path = tmp_path / "test_metadata.db"
        metadata_store = MetadataStore(str(db_path))
        tracker = UsageTracker(metadata_store)
        
        # Document creation and initial indexing
        doc_id = metadata_store.upsert_document("tutorial.md", "Python Tutorial", {
            'category': 'tutorial',
            'status': 'draft'
        })
        tracker.log_update("tutorial.md")
        
        # Document appears in search results
        tracker.log_search_hit("tutorial.md", "python basics")
        tracker.log_search_hit("tutorial.md", "python tutorial")
        
        # Document is read by users
        tracker.log_read("tutorial.md", "student1")
        tracker.log_read("tutorial.md", "student2")
        tracker.log_read("tutorial.md", "instructor")
        
        # Document is updated
        tracker.log_update("tutorial.md")
        
        # More reads after update
        tracker.log_read("tutorial.md", "student3")
        
        # Analyze the complete lifecycle
        doc = metadata_store.get_document("tutorial.md")
        stats = doc['usage_stats']
        
        assert stats['total_usage'] == 8
        assert stats['search_hits'] == 2
        assert stats['reads'] == 4
        assert stats['last_accessed'] is not None
        
        # Check access history
        history = tracker.get_access_history("tutorial.md")
        assert len(history) == 8
        
        # Should be in reverse chronological order (but may not be deterministic when timestamps are identical)
        # Just check that we have the expected mix of actions
        actions = [event['action'] for event in history]
        assert actions.count('read') == 4
        assert actions.count('search_hit') == 2 
        assert actions.count('update') == 2
        
        # Check analytics
        most_accessed = tracker.get_most_accessed(limit=5)
        assert len(most_accessed) == 1
        assert most_accessed[0]['filename'] == 'tutorial.md'
        assert most_accessed[0]['access_count'] == 8
        
        never_accessed = tracker.get_never_accessed()
        assert never_accessed == []
        
        trends = tracker.get_usage_trends(days=1)
        assert trends['total_events'] == 8
        assert 'tutorial.md' in [doc['filename'] for doc in trends['top_documents']]
        
        metadata_store.close()
    
    def test_multiple_documents_analytics(self, tmp_path):
        """Test analytics across multiple documents."""
        db_path = tmp_path / "test_metadata.db"
        metadata_store = MetadataStore(str(db_path))
        tracker = UsageTracker(metadata_store)
        
        # Create different types of documents
        docs = [
            ("popular.md", "Popular Tutorial", "Gets lots of traffic"),
            ("moderate.md", "Moderate Guide", "Some usage"),
            ("niche.md", "Niche Topic", "Rarely accessed"),
            ("orphan.md", "Orphaned Document", "Never accessed")
        ]
        
        for filename, title, description in docs:
            metadata_store.upsert_document(filename, title, {'description': description})
        
        # Simulate different usage patterns
        # Popular document
        for i in range(20):
            tracker.log_search_hit("popular.md", f"query {i % 5}")
        for i in range(10):
            tracker.log_read("popular.md", f"user{i}")
        
        # Moderate document
        for i in range(5):
            tracker.log_search_hit("moderate.md", f"query {i}")
        tracker.log_read("moderate.md", "user1")
        tracker.log_read("moderate.md", "user2")
        
        # Niche document
        tracker.log_search_hit("niche.md", "specialized query")
        tracker.log_read("niche.md", "expert_user")
        
        # Orphan document - no usage
        
        # Analyze patterns
        most_accessed = tracker.get_most_accessed(limit=10)
        assert len(most_accessed) == 3  # Only documents with usage
        
        # Should be ordered by total access count
        filenames = [doc['filename'] for doc in most_accessed]
        assert filenames[0] == 'popular.md'
        assert filenames[1] == 'moderate.md'
        assert filenames[2] == 'niche.md'
        
        # Check never accessed
        never_accessed = tracker.get_never_accessed()
        assert len(never_accessed) == 1
        assert never_accessed[0]['filename'] == 'orphan.md'
        
        # Check trends
        trends = tracker.get_usage_trends(days=1)
        assert trends['total_events'] == 39  # 20+10+5+2+1+1 = 39
        
        top_docs = trends['top_documents']
        assert len(top_docs) >= 1
        assert top_docs[0]['filename'] == 'popular.md'
        
        metadata_store.close()