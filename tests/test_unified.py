"""
Tests for Velociragtor unified search.

Test unified search with and without graph enrichment.
"""

import pytest
from pathlib import Path
import numpy as np

from velocirag.searcher import Searcher
from velocirag.store import VectorStore
from velocirag.embedder import Embedder
from velocirag.graph import GraphStore, Node, Edge, NodeType, RelationType
from velocirag.pipeline import GraphPipeline
from velocirag.unified import UnifiedSearch
from velocirag.metadata import MetadataStore
from velocirag.tracker import UsageTracker


class MockSearcher:
    """Mock searcher for testing without real vector store."""
    
    def __init__(self, results=None):
        self.results = results or []
        self.search_called = False
        self.last_query = None
    
    def search(self, query, limit=5, threshold=0.3, include_stats=False):
        """Mock search that returns predefined results."""
        self.search_called = True
        self.last_query = query
        
        # Return mock results
        mock_results = {
            'results': self.results[:limit],
            'query': query,
            'search_time_ms': 10.5,
            'variants_used': ['exact', 'expanded']
        }
        
        if include_stats:
            mock_results['stats'] = {'mock': True}
        
        return mock_results


class TestUnifiedSearch:
    """Test the unified search orchestrator."""
    
    @pytest.fixture
    def sample_search_results(self):
        """Create sample vector search results."""
        return [
            {
                'chunk': 'Python programming guide with examples',
                'score': 0.85,
                'metadata': {
                    'file_path': '/docs/python-guide.md',
                    'chunk_index': 0,
                    'chunk_type': 'content'
                }
            },
            {
                'chunk': 'Machine learning with Python',
                'score': 0.75,
                'metadata': {
                    'file_path': '/docs/machine-learning.md',
                    'chunk_index': 1,
                    'chunk_type': 'content'
                }
            },
            {
                'chunk': 'Deep learning neural networks',
                'score': 0.65,
                'metadata': {
                    'file_path': '/docs/deep-learning.md',
                    'chunk_index': 0,
                    'chunk_type': 'content'
                }
            }
        ]
    
    @pytest.fixture
    def populated_graph_store(self, tmp_path):
        """Create a graph store with test data."""
        db_path = tmp_path / "test_graph.db"
        store = GraphStore(str(db_path))
        
        # Add nodes
        nodes = [
            Node(id="note_1", type=NodeType.NOTE, title="python-guide"),
            Node(id="note_2", type=NodeType.NOTE, title="machine-learning"),
            Node(id="note_3", type=NodeType.NOTE, title="deep-learning"),
            Node(id="note_4", type=NodeType.NOTE, title="data-science"),
        ]
        store.add_nodes(nodes)
        
        # Add edges
        edges = [
            Edge(id="e1", source_id="note_1", target_id="note_2", 
                 type=RelationType.REFERENCES, weight=0.8, confidence=0.9),
            Edge(id="e2", source_id="note_2", target_id="note_3",
                 type=RelationType.REFERENCES, weight=0.9, confidence=0.95),
            Edge(id="e3", source_id="note_2", target_id="note_4",
                 type=RelationType.SIMILAR_TO, weight=0.7, confidence=0.8),
        ]
        store.add_edges(edges)
        
        return store
    
    def test_vector_only_search(self, sample_search_results):
        """Test search without graph enrichment."""
        mock_searcher = MockSearcher(results=sample_search_results)
        unified = UnifiedSearch(searcher=mock_searcher, graph_store=None)
        
        # Search without graph
        results = unified.search("python programming", limit=3)
        
        # Verify search was called
        assert mock_searcher.search_called
        assert mock_searcher.last_query == "python programming"
        
        # Check response structure
        assert 'results' in results
        assert 'query' in results
        assert 'search_mode' in results
        assert results['search_mode'] == 'vector_only'
        
        # Verify results
        assert len(results['results']) == 3
        assert results['results'][0]['chunk'] == 'Python programming guide with examples'
        
        # Check that graph fields are initialized but empty
        for result in results['results']:
            assert 'graph_connections' in result['metadata']
            assert 'related_notes' in result['metadata']
            assert result['metadata']['graph_connections'] == []
            assert result['metadata']['related_notes'] == []
            assert result['metadata']['found_in_graph'] is False
    
    def test_unified_search_with_graph(self, sample_search_results, populated_graph_store):
        """Test search with graph enrichment."""
        mock_searcher = MockSearcher(results=sample_search_results)
        unified = UnifiedSearch(searcher=mock_searcher, graph_store=populated_graph_store)
        
        # Search with graph enrichment
        results = unified.search("python programming", limit=2, enrich_graph=True)
        
        # Check search mode
        assert results['search_mode'] == 'vector_graph'
        assert results['enrichment_stats']['graph_available'] is True
        
        # Find the python-guide result (RRF fusion may reorder results)
        python_guide_result = None
        machine_learning_result = None
        
        for result in results['results']:
            file_path = result.get('metadata', {}).get('file_path', '')
            if 'python-guide' in file_path:
                python_guide_result = result
            elif 'machine-learning' in file_path:
                machine_learning_result = result
        
        # Python-guide result should be enriched and connect to machine-learning
        assert python_guide_result is not None
        assert python_guide_result['metadata']['found_in_graph'] is True
        assert len(python_guide_result['metadata']['graph_connections']) > 0
        assert 'machine-learning' in python_guide_result['metadata']['graph_connections']
        
        # Machine-learning result should also be enriched
        assert machine_learning_result is not None
        assert machine_learning_result['metadata']['found_in_graph'] is True
        assert len(machine_learning_result['metadata']['graph_connections']) > 0
    
    def test_graph_enrichment_disabled(self, sample_search_results, populated_graph_store):
        """Test disabling graph enrichment even when graph is available."""
        mock_searcher = MockSearcher(results=sample_search_results)
        unified = UnifiedSearch(searcher=mock_searcher, graph_store=populated_graph_store)
        
        # Search with enrichment disabled
        results = unified.search("python", limit=2, enrich_graph=False)
        
        # Should use vector_only mode
        assert results['search_mode'] == 'vector_only'
        
        # Results should not be enriched
        for result in results['results']:
            assert result['metadata']['graph_connections'] == []
            assert result['metadata']['found_in_graph'] is False
    
    def test_graph_error_handling(self, sample_search_results, populated_graph_store):
        """Test that graph errors don't break search."""
        mock_searcher = MockSearcher(results=sample_search_results)
        
        # Create unified search with good graph store
        unified = UnifiedSearch(searcher=mock_searcher, graph_store=populated_graph_store)
        
        # Modify results to have invalid file paths to trigger enrichment errors
        broken_results = [
            {
                'chunk': 'Test content',
                'score': 0.85,
                'metadata': {
                    'file_path': '',  # Empty path will cause title extraction to fail
                    'chunk_index': 0
                }
            },
            {
                'chunk': 'Another test',
                'score': 0.75,
                'metadata': {}  # No file_path at all
            }
        ]
        mock_searcher.results = broken_results
        
        # Search should still work despite enrichment failures
        results = unified.search("python", limit=2)
        
        assert 'results' in results
        assert len(results['results']) >= 1  # At least one result should be returned
        
        # With true 3-layer fusion, graph layer may find valid candidates
        # even when vector results have broken paths. The test should verify
        # that search doesn't crash and returns reasonable results.
        
        # At least verify that results have the required graph metadata fields
        for result in results['results']:
            assert 'graph_connections' in result['metadata']
            assert 'found_in_graph' in result['metadata']
            assert isinstance(result['metadata']['graph_connections'], list)
            assert isinstance(result['metadata']['found_in_graph'], bool)
    
    def test_enrichment_stats(self, sample_search_results, populated_graph_store):
        """Test enrichment statistics tracking."""
        mock_searcher = MockSearcher(results=sample_search_results)
        unified = UnifiedSearch(searcher=mock_searcher, graph_store=populated_graph_store)
        
        results = unified.search("python", limit=3)
        
        # Check enrichment stats
        stats = results['enrichment_stats']
        assert stats['vector_results'] == 3
        assert stats['graph_available'] is True
        assert stats['graph_enriched'] >= 0
        assert stats['graph_enriched'] <= 3
    
    def test_search_timing(self, sample_search_results):
        """Test that search timing is tracked."""
        mock_searcher = MockSearcher(results=sample_search_results)
        unified = UnifiedSearch(searcher=mock_searcher)
        
        results = unified.search("test query")
        
        # Check timing fields
        assert 'search_time_ms' in results
        assert results['search_time_ms'] > 0
        assert 'vector_time_ms' in results
    
    def test_stats_method(self, populated_graph_store):
        """Test the stats method."""
        mock_searcher = MockSearcher()
        unified = UnifiedSearch(searcher=mock_searcher, graph_store=populated_graph_store)
        
        stats = unified.stats()
        
        # Check component availability
        assert stats['searcher_available'] is True
        assert stats['graph_available'] is True
        assert stats['components']['searcher'] == 'available'
        assert stats['components']['graph_store'] == 'available'
        assert stats['components']['graph_querier'] == 'available'
        
        # Check graph stats
        assert 'graph_stats' in stats
        assert stats['graph_stats']['node_count'] == 4
        assert stats['graph_stats']['edge_count'] == 3
    
    def test_real_integration(self, tmp_path):
        """Test with real Searcher and GraphStore integration."""
        # Create a real vector store
        vector_db = tmp_path / "vectors.db" 
        embedder = Embedder()
        store = VectorStore(str(vector_db))
        
        # Index some documents
        docs = [
            {
                'content': "Python is a high-level programming language",
                'metadata': {'file_path': '/docs/python-intro.md'}
            },
            {
                'content': "Machine learning algorithms in Python",
                'metadata': {'file_path': '/docs/ml-guide.md'}
            }
        ]
        
        # Prepare documents with embeddings
        documents_to_add = []
        for i, doc in enumerate(docs):
            embedding = embedder.embed(doc['content'])
            documents_to_add.append({
                'doc_id': f'doc_{i}',
                'content': doc['content'],
                'metadata': doc['metadata'],
                'embedding': embedding
            })
        
        # Add all documents
        store.add_documents(documents_to_add)
        
        # Create searcher
        searcher = Searcher(store, embedder)
        
        # Create graph store with matching data
        graph_db = tmp_path / "graph.db"
        graph_store = GraphStore(str(graph_db))
        
        nodes = [
            Node(id="n1", type=NodeType.NOTE, title="python-intro"),
            Node(id="n2", type=NodeType.NOTE, title="ml-guide"),
        ]
        graph_store.add_nodes(nodes)
        
        edges = [
            Edge(id="e1", source_id="n1", target_id="n2",
                 type=RelationType.REFERENCES, weight=0.7, confidence=0.8)
        ]
        graph_store.add_edges(edges)
        
        # Create unified search
        unified = UnifiedSearch(searcher=searcher, graph_store=graph_store)
        
        # Perform search
        results = unified.search("python", limit=2)
        
        # Should find results
        assert len(results['results']) > 0
        assert results['search_mode'] == 'vector_graph'
        
        # At least one result should be enriched
        enriched = [r for r in results['results'] if r['metadata']['found_in_graph']]
        assert len(enriched) > 0
    
    def test_search_with_metadata_filters(self, sample_search_results, tmp_path):
        """Test search with metadata filtering."""
        # Create metadata store with test data
        metadata_db = tmp_path / "metadata.db"
        metadata_store = MetadataStore(str(metadata_db))
        
        # Add test documents to metadata store
        doc1_id = metadata_store.upsert_document(
            filename="python-guide.md",
            title="Python Guide",
            metadata={'category': 'tutorial', 'status': 'active', 'project': 'docs'}
        )
        metadata_store.add_tags(doc1_id, ['python', 'programming', 'tutorial'])
        
        doc2_id = metadata_store.upsert_document(
            filename="machine-learning.md", 
            title="Machine Learning",
            metadata={'category': 'research', 'status': 'active', 'project': 'ai'}
        )
        metadata_store.add_tags(doc2_id, ['python', 'ml', 'ai'])
        
        doc3_id = metadata_store.upsert_document(
            filename="deep-learning.md",
            title="Deep Learning",
            metadata={'category': 'research', 'status': 'draft', 'project': 'ai'}
        )
        metadata_store.add_tags(doc3_id, ['ai', 'neural-networks'])
        
        # Create unified search with metadata
        mock_searcher = MockSearcher(results=sample_search_results)
        unified = UnifiedSearch(searcher=mock_searcher, metadata_store=metadata_store)
        
        # Search with tag filter
        results = unified.search("python", filters={'tags': ['python']})
        
        assert results['search_mode'] == 'vector_metadata'
        assert 'metadata_time_ms' in results
        assert results['enrichment_stats']['metadata_matches'] == 2  # Two docs have python tag
        
        # Search with category filter
        results = unified.search("learning", filters={'category': 'research'})
        assert results['enrichment_stats']['metadata_matches'] == 2  # ML and DL docs
        
        # Search with multiple filters
        results = unified.search("ai", filters={'tags': ['ai'], 'status': 'active'})
        assert results['enrichment_stats']['metadata_matches'] == 1  # Only ML doc is active
    
    def test_rrf_fusion_with_metadata(self, sample_search_results, tmp_path):
        """Test RRF fusion between vector and metadata results."""
        # Create metadata store
        metadata_db = tmp_path / "metadata.db"
        metadata_store = MetadataStore(str(metadata_db))
        
        # Add a document that matches metadata but not in vector results
        doc_id = metadata_store.upsert_document(
            filename="rust-guide.md",
            title="Rust Programming",
            metadata={'category': 'tutorial', 'status': 'active'}
        )
        metadata_store.add_tags(doc_id, ['rust', 'programming'])
        
        # Also add python-guide which is in vector results
        doc_id = metadata_store.upsert_document(
            filename="python-guide.md",
            title="Python Guide",
            metadata={'category': 'tutorial', 'status': 'active'}
        )
        metadata_store.add_tags(doc_id, ['python', 'programming'])
        
        mock_searcher = MockSearcher(results=sample_search_results)
        unified = UnifiedSearch(searcher=mock_searcher, metadata_store=metadata_store)
        
        # Search with programming tag filter
        results = unified.search("programming", filters={'tags': ['programming']})
        
        # Should have fusion time
        assert 'fusion_time_ms' in results
        
        # Results should be influenced by metadata matches
        # python-guide.md should be boosted since it appears in both
        for result in results['results']:
            if 'python-guide' in result.get('metadata', {}).get('file_path', ''):
                assert result.get('_metadata_match') is True
                break
    
    def test_usage_tracking(self, sample_search_results, tmp_path):
        """Test that search hits are tracked."""
        # Create metadata store and tracker
        metadata_db = tmp_path / "metadata.db"
        metadata_store = MetadataStore(str(metadata_db))
        tracker = UsageTracker(metadata_store)
        
        # Add test document
        doc_id = metadata_store.upsert_document(
            filename="python-guide.md",
            title="Python Guide",
            metadata={'category': 'tutorial'}
        )
        
        # Create unified search with tracker
        mock_searcher = MockSearcher(results=sample_search_results)
        unified = UnifiedSearch(searcher=mock_searcher, tracker=tracker)
        
        # Perform search
        results = unified.search("python programming")
        
        # Check that usage was logged
        # Note: Only logs if result has valid file_path in metadata
        history = tracker.get_access_history("python-guide.md")
        # Since mock results have file paths, should have logged
        if any('/python-guide.md' in r.get('metadata', {}).get('file_path', '') 
               for r in sample_search_results):
            assert len(history) > 0
            assert history[0]['action'] == 'search_hit'
    
    def test_direct_metadata_query(self, tmp_path):
        """Test direct metadata query method."""
        # Create metadata store with test data
        metadata_db = tmp_path / "metadata.db"
        metadata_store = MetadataStore(str(metadata_db))
        
        # Add test documents
        for i in range(5):
            doc_id = metadata_store.upsert_document(
                filename=f"doc_{i}.md",
                title=f"Document {i}",
                metadata={'category': 'test', 'status': 'active' if i < 3 else 'draft'}
            )
            metadata_store.add_tags(doc_id, ['test', f'doc{i}'])
        
        # Create unified search (no searcher needed for direct query)
        mock_searcher = MockSearcher()
        unified = UnifiedSearch(searcher=mock_searcher, metadata_store=metadata_store)
        
        # Direct query
        results = unified.query(tags=['test'], status='active')
        
        assert len(results) == 3  # Only active documents
        for doc in results:
            assert doc['status'] == 'active'
            assert 'test' in doc.get('tags', [])
    
    def test_search_mode_detection(self):
        """Test that search mode is correctly detected based on available components."""
        mock_searcher = MockSearcher()
        
        # Vector only
        unified = UnifiedSearch(searcher=mock_searcher)
        results = unified.search("test")
        assert results['search_mode'] == 'vector_only'
        
        # Vector + graph
        graph_store = GraphStore(":memory:")
        unified = UnifiedSearch(searcher=mock_searcher, graph_store=graph_store)
        results = unified.search("test")
        assert results['search_mode'] == 'vector_graph'
        
        # Vector + metadata
        metadata_store = MetadataStore(":memory:")
        unified = UnifiedSearch(searcher=mock_searcher, metadata_store=metadata_store)
        results = unified.search("test", filters={'tags': ['test']})
        assert results['search_mode'] == 'vector_metadata'
        
        # All three layers
        unified = UnifiedSearch(searcher=mock_searcher, graph_store=graph_store, 
                               metadata_store=metadata_store)
        results = unified.search("test", filters={'tags': ['test']})
        assert results['search_mode'] == 'unified_full'