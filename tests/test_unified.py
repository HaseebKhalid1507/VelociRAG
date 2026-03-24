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
        assert results['search_mode'] == 'unified'
        assert results['enrichment_stats']['graph_available'] is True
        
        # First result should be enriched (python-guide)
        first_result = results['results'][0]
        assert first_result['metadata']['found_in_graph'] is True
        assert len(first_result['metadata']['graph_connections']) > 0
        assert 'machine-learning' in first_result['metadata']['graph_connections']
        
        # Second result should also be enriched (machine-learning)
        second_result = results['results'][1]
        assert second_result['metadata']['found_in_graph'] is True
        assert len(second_result['metadata']['graph_connections']) > 0
    
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
        assert len(results['results']) == 2
        
        # Graph fields should be empty due to errors
        for result in results['results']:
            assert result['metadata']['graph_connections'] == []
            assert result['metadata']['found_in_graph'] is False
    
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
        assert results['search_mode'] == 'unified'
        
        # At least one result should be enriched
        enriched = [r for r in results['results'] if r['metadata']['found_in_graph']]
        assert len(enriched) > 0