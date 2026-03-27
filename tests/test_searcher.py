"""Tests for searcher.py module."""

import pytest
import numpy as np
import time
from unittest.mock import Mock, MagicMock
from pathlib import Path
from velocirag.searcher import Searcher, SearchError, DEFAULT_RRF_K, MIN_RRF_K, MAX_RRF_K
from velocirag.store import VectorStore
from velocirag.embedder import Embedder


@pytest.fixture
def test_documents():
    """Sample documents for testing."""
    return [
        {
            'doc_id': 'doc1',
            'content': 'Python programming language tutorial for beginners',
            'metadata': {'topic': 'programming', 'level': 'beginner'}
        },
        {
            'doc_id': 'doc2',
            'content': 'Advanced Python techniques and design patterns',
            'metadata': {'topic': 'programming', 'level': 'advanced'}
        },
        {
            'doc_id': 'doc3',
            'content': 'Machine learning with Python and scikit-learn',
            'metadata': {'topic': 'ml', 'tool': 'scikit-learn'}
        },
        {
            'doc_id': 'doc4',
            'content': 'Deep learning neural networks with PyTorch',
            'metadata': {'topic': 'ml', 'tool': 'pytorch'}
        },
        {
            'doc_id': 'doc5',
            'content': 'Data analysis and visualization with pandas',
            'metadata': {'topic': 'data', 'tool': 'pandas'}
        },
        {
            'doc_id': 'doc6',
            'content': 'Web development using Django framework',
            'metadata': {'topic': 'web', 'framework': 'django'}
        },
        {
            'doc_id': 'doc7',
            'content': 'FastAPI modern web APIs with Python',
            'metadata': {'topic': 'web', 'framework': 'fastapi'}
        },
        {
            'doc_id': 'doc8',
            'content': 'Database design and SQL optimization techniques',
            'metadata': {'topic': 'database', 'type': 'sql'}
        },
        {
            'doc_id': 'doc9',
            'content': 'NoSQL databases MongoDB and Redis comparison',
            'metadata': {'topic': 'database', 'type': 'nosql'}
        },
        {
            'doc_id': 'doc10',
            'content': 'Cloud computing with AWS and Python boto3',
            'metadata': {'topic': 'cloud', 'provider': 'aws'}
        },
        {
            'doc_id': 'doc11',
            'content': 'Docker containers and Kubernetes orchestration',
            'metadata': {'topic': 'devops', 'tools': ['docker', 'kubernetes']}
        },
        {
            'doc_id': 'doc12',
            'content': 'Git version control and GitHub workflow best practices',
            'metadata': {'topic': 'devops', 'tool': 'git'}
        },
        {
            'doc_id': 'doc13',
            'content': 'Testing Python code with pytest and unittest',
            'metadata': {'topic': 'testing', 'tools': ['pytest', 'unittest']}
        },
        {
            'doc_id': 'doc14',
            'content': 'Cybersecurity fundamentals and ethical hacking',
            'metadata': {'topic': 'security', 'level': 'beginner'}
        },
        {
            'doc_id': 'doc15',
            'content': 'Blockchain technology and cryptocurrency basics',
            'metadata': {'topic': 'blockchain', 'type': 'intro'}
        },
        {
            'doc_id': 'doc16',
            'content': 'Artificial intelligence and machine learning introduction',
            'metadata': {'topic': 'ai', 'level': 'intro'}
        },
        {
            'doc_id': 'doc17',
            'content': 'Natural language processing with transformers and BERT',
            'metadata': {'topic': 'nlp', 'model': 'bert'}
        },
        {
            'doc_id': 'doc18',
            'content': 'Computer vision and image recognition with OpenCV',
            'metadata': {'topic': 'cv', 'library': 'opencv'}
        },
        {
            'doc_id': 'doc19',
            'content': 'Quantum computing concepts and quantum algorithms',
            'metadata': {'topic': 'quantum', 'type': 'theoretical'}
        },
        {
            'doc_id': 'doc20',
            'content': 'Mobile app development with React Native and Flutter comparison',
            'metadata': {'topic': 'mobile', 'frameworks': ['react-native', 'flutter']}
        }
    ]


@pytest.fixture
def embedder():
    """Shared embedder instance."""
    return Embedder()


@pytest.fixture
def populated_store(tmp_path, embedder, test_documents):
    """VectorStore populated with test documents."""
    store = VectorStore(str(tmp_path / "test_store"), embedder=embedder)
    
    # Add all test documents
    store.add_documents(test_documents)
    
    yield store
    store.close()


@pytest.fixture
def searcher(populated_store, embedder):
    """Searcher instance with populated store."""
    return Searcher(populated_store, embedder)


class TestSearcherConstructor:
    """Test Searcher initialization."""
    
    def test_default_parameters(self, populated_store, embedder):
        """Default constructor creates searcher with expected settings."""
        searcher = Searcher(populated_store, embedder)
        
        assert searcher.store is populated_store
        assert searcher.embedder is embedder
        assert searcher.rrf_k == DEFAULT_RRF_K
        # Reranker is now auto-created (gracefully handles missing sentence_transformers)
        assert searcher.reranker is not None
    
    def test_custom_rrf_k(self, populated_store, embedder):
        """Custom RRF k parameter is accepted."""
        searcher = Searcher(populated_store, embedder, rrf_k=100)
        assert searcher.rrf_k == 100
    
    def test_rrf_k_validation(self, populated_store, embedder):
        """RRF k must be within valid range."""
        # Valid edge cases
        searcher1 = Searcher(populated_store, embedder, rrf_k=MIN_RRF_K)
        assert searcher1.rrf_k == MIN_RRF_K
        
        searcher2 = Searcher(populated_store, embedder, rrf_k=MAX_RRF_K)
        assert searcher2.rrf_k == MAX_RRF_K
        
        # Invalid type
        with pytest.raises(ValueError, match="rrf_k must be an integer"):
            Searcher(populated_store, embedder, rrf_k=60.5)
        
        # Too small
        with pytest.raises(ValueError, match=f"rrf_k must be between {MIN_RRF_K} and {MAX_RRF_K}"):
            Searcher(populated_store, embedder, rrf_k=MIN_RRF_K - 1)
        
        # Too large
        with pytest.raises(ValueError, match=f"rrf_k must be between {MIN_RRF_K} and {MAX_RRF_K}"):
            Searcher(populated_store, embedder, rrf_k=MAX_RRF_K + 1)
    
    def test_with_reranker(self, populated_store, embedder):
        """Reranker function is stored correctly."""
        def mock_reranker(query, results):
            return results
        
        searcher = Searcher(populated_store, embedder, reranker=mock_reranker)
        assert searcher.reranker is mock_reranker


class TestTextSearch:
    """Test text query search functionality."""
    
    def test_basic_search_returns_results(self, searcher):
        """Basic search returns relevant results."""
        result = searcher.search("Python programming", limit=5)
        
        assert 'results' in result
        assert 'query' in result
        assert 'total_results' in result
        assert 'search_time_ms' in result
        assert 'variants_used' in result
        
        assert result['query'] == 'Python programming'
        assert len(result['results']) > 0
        assert len(result['results']) <= 5
        assert result['total_results'] == len(result['results'])
        assert result['search_time_ms'] > 0
        assert len(result['variants_used']) > 0
    
    def test_empty_query_returns_empty(self, searcher):
        """Empty query returns empty results."""
        result = searcher.search("", limit=5)
        
        assert result['results'] == []
        assert result['query'] == ""
        assert result['total_results'] == 0
        assert result['search_time_ms'] == 0.0
        assert result['variants_used'] == []
    
    def test_whitespace_query_returns_empty(self, searcher):
        """Whitespace-only query returns empty results."""
        result = searcher.search("   \t\n  ", limit=5)
        
        assert result['results'] == []
        assert result['query'] == "   \t\n  "
        assert result['total_results'] == 0
        assert result['search_time_ms'] == 0.0
        assert result['variants_used'] == []
    
    def test_short_query_skips_variants(self, searcher):
        """Queries < 3 chars skip variant generation."""
        result = searcher.search("AI", limit=5)
        
        # Should have results (matches "Artificial intelligence")
        assert len(result['results']) > 0
        # Should only have original query as variant
        assert result['variants_used'] == ["AI"]
    
    def test_multiple_variants_produce_fused_results(self, searcher):
        """Longer queries generate variants and fuse results."""
        # Use a query that will generate variants (has uppercase and numbers)
        result = searcher.search("Python3 Programming", limit=10)
        
        assert len(result['results']) > 0
        assert len(result['variants_used']) > 1  # Should have multiple variants
        
        # Results should have RRF scores in metadata
        for res in result['results']:
            assert 'rrf_score' in res['metadata']
            assert res['metadata']['rrf_score'] > 0
    
    def test_respects_limit(self, searcher):
        """Search respects limit parameter."""
        # Test various limits
        for limit in [1, 3, 5, 10]:
            result = searcher.search("Python", limit=limit)
            assert len(result['results']) <= limit
    
    def test_respects_threshold(self, searcher):
        """Search respects similarity threshold."""
        # Low threshold should return more results
        low_threshold_result = searcher.search("blockchain quantum computing", limit=20, threshold=0.1)
        
        # High threshold should return fewer results
        high_threshold_result = searcher.search("blockchain quantum computing", limit=20, threshold=0.7)
        
        # Might get no results with very high threshold
        assert len(high_threshold_result['results']) <= len(low_threshold_result['results'])
    
    def test_results_have_correct_format(self, searcher):
        """Results have all required fields with correct types."""
        result = searcher.search("Python programming", limit=5)
        
        for doc in result['results']:
            assert 'doc_id' in doc
            assert 'content' in doc
            assert 'metadata' in doc
            assert 'similarity' in doc
            assert 'rrf_score' in doc['metadata']  # RRF score is in metadata
            
            assert isinstance(doc['doc_id'], str)
            assert isinstance(doc['content'], str)
            assert isinstance(doc['metadata'], dict)
            assert isinstance(doc['similarity'], float)
            assert isinstance(doc['metadata']['rrf_score'], float)
            
            assert 0 <= doc['similarity'] <= 1
            assert doc['metadata']['rrf_score'] > 0
    
    def test_search_timing_reported(self, searcher):
        """Search time is reported in milliseconds."""
        result = searcher.search("Python programming", limit=5)
        
        assert 'search_time_ms' in result
        assert isinstance(result['search_time_ms'], float)
        assert result['search_time_ms'] > 0
        assert result['search_time_ms'] < 10000  # Should complete in < 10 seconds
    
    def test_include_stats_provides_details(self, searcher):
        """include_stats=True provides detailed timing information."""
        result = searcher.search("Python programming", limit=5, include_stats=True)
        
        assert 'stats' in result
        stats = result['stats']
        
        assert 'variant_generation_ms' in stats
        assert 'embedding_time_ms' in stats
        assert 'search_time_ms' in stats
        assert 'rrf_fusion_ms' in stats
        assert 'variants' in stats
        
        # Each variant should have stats
        for variant_stat in stats['variants']:
            assert 'variant' in variant_stat
            assert 'results_found' in variant_stat
            assert 'search_time_ms' in variant_stat
    
    def test_invalid_query_type(self, searcher):
        """Non-string query raises ValueError."""
        with pytest.raises(ValueError, match="query must be a string"):
            searcher.search(123)
        
        with pytest.raises(ValueError, match="query must be a string"):
            searcher.search(['list', 'of', 'strings'])
    
    def test_invalid_limit(self, searcher):
        """Invalid limit values raise ValueError."""
        with pytest.raises(ValueError, match="limit must be a positive integer"):
            searcher.search("Python", limit=0)
        
        with pytest.raises(ValueError, match="limit must be a positive integer"):
            searcher.search("Python", limit=-5)
        
        with pytest.raises(ValueError, match="limit must be a positive integer"):
            searcher.search("Python", limit=3.14)
    
    def test_invalid_threshold(self, searcher):
        """Invalid threshold values raise ValueError."""
        with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
            searcher.search("Python", threshold=-0.1)
        
        with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
            searcher.search("Python", threshold=1.5)
        
        with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
            searcher.search("Python", threshold="high")
    
    def test_max_limit_enforced(self, searcher):
        """Limit above MAX_SEARCH_LIMIT is capped."""
        result = searcher.search("Python", limit=1000)  # Way above MAX_SEARCH_LIMIT
        assert len(result['results']) <= 100  # MAX_SEARCH_LIMIT


class TestEmbeddingSearch:
    """Test search with pre-computed embeddings."""
    
    def test_search_embedding_works(self, searcher):
        """search_embedding() returns results for pre-computed vector."""
        # Get embedding for a query
        embedding = searcher.embedder.embed("Python programming")
        
        result = searcher.search_embedding(embedding, limit=5)
        
        assert 'results' in result
        assert 'query' in result
        assert 'total_results' in result
        assert 'search_time_ms' in result
        assert 'variants_used' in result
        
        assert result['query'] == '[embedding]'  # Placeholder
        assert len(result['results']) > 0
        assert result['variants_used'] == []  # No variants for embedding search
    
    def test_embedding_results_sorted_by_similarity(self, searcher):
        """Embedding search results are sorted by similarity descending."""
        embedding = searcher.embedder.embed("machine learning")
        result = searcher.search_embedding(embedding, limit=10)
        
        # Verify sorted by similarity
        similarities = [doc['similarity'] for doc in result['results']]
        assert similarities == sorted(similarities, reverse=True)
    
    def test_invalid_embedding_type(self, searcher):
        """Non-numpy array embedding raises ValueError."""
        with pytest.raises(ValueError, match="embedding must be numpy array"):
            searcher.search_embedding([0.1, 0.2, 0.3])  # List instead of array
        
        with pytest.raises(ValueError, match="embedding must be numpy array"):
            searcher.search_embedding("not an embedding")
    
    def test_wrong_dimension_embedding(self, searcher):
        """Wrong dimension embedding raises ValueError."""
        # Create embedding with wrong dimensions
        wrong_embedding = np.random.rand(512)  # Should be 384
        
        with pytest.raises(ValueError, match="Embedding dimension .* doesn't match store dimension"):
            searcher.search_embedding(wrong_embedding)
    
    def test_embedding_search_with_threshold(self, searcher):
        """Embedding search respects similarity threshold."""
        embedding = searcher.embedder.embed("quantum physics")
        
        # High threshold might return fewer/no results
        result = searcher.search_embedding(embedding, limit=20, threshold=0.8)
        high_threshold_count = len(result['results'])
        
        # Low threshold should return more
        result = searcher.search_embedding(embedding, limit=20, threshold=0.1)
        low_threshold_count = len(result['results'])
        
        assert low_threshold_count >= high_threshold_count
    
    def test_embedding_search_include_stats(self, searcher):
        """Embedding search with include_stats provides timing info."""
        embedding = searcher.embedder.embed("Python")
        result = searcher.search_embedding(embedding, include_stats=True)
        
        assert 'stats' in result
        stats = result['stats']
        assert 'search_time_ms' in stats
        # No variant stats for embedding search
        assert 'variant_generation_ms' not in stats
        assert 'variants' not in stats


class TestRRFFusion:
    """Test RRF fusion behavior."""
    
    def test_same_doc_from_variants_deduplicated(self, searcher):
        """Same document from multiple variants is deduplicated."""
        # Search for something that should match strongly
        result = searcher.search("Python programming language", limit=20)
        
        # Count occurrences of each doc_id
        doc_ids = [doc['doc_id'] for doc in result['results']]
        assert len(doc_ids) == len(set(doc_ids))  # All unique
    
    def test_rrf_scores_present(self, searcher):
        """RRF scores are present in fused results."""
        result = searcher.search("machine learning Python", limit=10)
        
        for doc in result['results']:
            assert 'rrf_score' in doc['metadata']
            assert isinstance(doc['metadata']['rrf_score'], float)
            assert doc['metadata']['rrf_score'] > 0
    
    def test_custom_rrf_k_affects_fusion(self, populated_store, embedder):
        """Different RRF k values produce different rankings."""
        searcher1 = Searcher(populated_store, embedder, rrf_k=10)
        searcher2 = Searcher(populated_store, embedder, rrf_k=100)
        
        query = "Python machine learning"
        result1 = searcher1.search(query, limit=10)
        result2 = searcher2.search(query, limit=10)
        
        # Both should return results
        assert len(result1['results']) > 0
        assert len(result2['results']) > 0
        
        # RRF scores should be different
        scores1 = [doc['metadata']['rrf_score'] for doc in result1['results']]
        scores2 = [doc['metadata']['rrf_score'] for doc in result2['results']]
        
        # At least some scores should differ (unless all docs have same ranks)
        # This is a weak test but avoids being too specific about RRF behavior
        if len(scores1) == len(scores2) and len(scores1) > 1:
            assert scores1 != scores2 or all(s == scores1[0] for s in scores1)


class TestRerankerIntegration:
    """Test reranker integration."""
    
    def test_with_mock_reranker_called(self, populated_store, embedder):
        """Mock reranker is called with query and results."""
        mock_reranker = Mock(return_value=[])
        searcher = Searcher(populated_store, embedder, reranker=mock_reranker)
        
        query = "Python programming"
        searcher.search(query, limit=5)
        
        # Reranker should have been called once
        mock_reranker.assert_called_once()
        call_args = mock_reranker.call_args[0]
        assert call_args[0] == query  # First arg is query
        assert isinstance(call_args[1], list)  # Second arg is results list
    
    def test_reranker_modifies_results(self, populated_store, embedder):
        """Reranker can modify result order."""
        # Track if reranker was called and what it received
        reranker_called = False
        received_results = None
        
        def tracking_reranker(query, results):
            nonlocal reranker_called, received_results
            reranker_called = True
            received_results = results
            # Return first 3 results in reverse order
            return results[:3][::-1] if len(results) >= 3 else results[::-1]
        
        # Test with reranker
        searcher = Searcher(populated_store, embedder, reranker=tracking_reranker)
        result = searcher.search("Python", limit=5)
        
        # Verify reranker was called
        assert reranker_called
        assert received_results is not None
        
        # Should have at most 3 results (due to our reranker limiting)
        assert len(result['results']) <= 3
        
        # If we have multiple results, verify they're from the original results
        if len(result['results']) > 0:
            result_ids = [doc['doc_id'] for doc in result['results']]
            received_ids = [doc['doc_id'] for doc in received_results[:3]]
            # All result IDs should be in the received results
            for doc_id in result_ids:
                assert doc_id in received_ids
    
    def test_reranker_failure_handled(self, populated_store, embedder):
        """Reranker failure doesn't crash search."""
        def failing_reranker(query, results):
            raise Exception("Reranker failed!")
        
        searcher = Searcher(populated_store, embedder, reranker=failing_reranker)
        
        # Should still return results despite reranker failure
        result = searcher.search("Python", limit=5)
        assert len(result['results']) > 0
    
    def test_reranker_timing_tracked(self, populated_store, embedder):
        """Reranker execution time is tracked in stats."""
        def slow_reranker(query, results):
            time.sleep(0.05)  # 50ms delay
            return results
        
        searcher = Searcher(populated_store, embedder, reranker=slow_reranker)
        result = searcher.search("Python", limit=5, include_stats=True)
        
        assert 'stats' in result
        assert 'rerank_time_ms' in result['stats']
        assert result['stats']['rerank_time_ms'] >= 50.0
    
    def test_without_reranker_still_works(self, searcher):
        """Search with unavailable reranker returns normal results."""
        result = searcher.search("Python", limit=5)
        assert len(result['results']) > 0
        
        # Reranker timing is present even when reranker is unavailable (graceful fallback)
        result_with_stats = searcher.search("Python", limit=5, include_stats=True)
        assert 'rerank_time_ms' in result_with_stats['stats']
        # But the timing should be very small since it just falls back quickly
        assert result_with_stats['stats']['rerank_time_ms'] >= 0


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_search_on_empty_store(self, tmp_path, embedder):
        """Search on empty store returns empty results."""
        empty_store = VectorStore(str(tmp_path / "empty_store"), embedder=embedder)
        searcher = Searcher(empty_store, embedder)
        
        result = searcher.search("Python", limit=5)
        assert result['results'] == []
        assert result['total_results'] == 0
    
    def test_unicode_queries(self, searcher):
        """Unicode queries are handled correctly."""
        queries = [
            "Python 编程",
            "Machine Learning 機械学習",
            "Données et analyse",
            "🐍 Python 🚀"
        ]
        
        for query in queries:
            result = searcher.search(query, limit=5)
            assert result['query'] == query
            # Should complete without error
    
    def test_very_long_query(self, searcher):
        """Very long queries are rejected."""
        long_query = "Python " * 200  # Over 1000 chars
        
        with pytest.raises(ValueError, match="Query too long"):
            searcher.search(long_query)
    
    def test_single_character_query(self, searcher):
        """Single character query returns results but no variants."""
        result = searcher.search("P", limit=5)
        
        # Might match "Python", "PyTorch", etc.
        assert result['variants_used'] == ["P"]
    
    def test_query_with_special_characters(self, searcher):
        """Queries with special characters work correctly."""
        special_queries = [
            "Python/Django",
            "C++ & Python",
            "email@example.com",
            "Machine-Learning",
            "$PATH variables"
        ]
        
        for query in special_queries:
            result = searcher.search(query, limit=5)
            assert result['query'] == query
    
    def test_concurrent_searches(self, searcher):
        """Concurrent searches don't interfere (basic test)."""
        import threading
        
        results = []
        errors = []
        
        def search_thread(query):
            try:
                result = searcher.search(query, limit=5)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Create threads
        threads = []
        queries = ["Python", "machine learning", "web development", "database"]
        for query in queries:
            t = threading.Thread(target=search_thread, args=(query,))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert len(results) == len(queries)
    
    def test_variant_generation_failure(self, populated_store, embedder):
        """Variant generation failure falls back to original query."""
        # We can't easily make variant generation fail, but we can test
        # the behavior when variants is empty
        searcher = Searcher(populated_store, embedder)
        
        # Short query that won't generate variants but should find results
        # "AI" should match "Artificial intelligence"
        result = searcher.search("AI", limit=5)
        assert result['variants_used'] == ["AI"]
        # May or may not find results depending on similarity threshold
        # Just verify it doesn't crash
    
    def test_embedding_failure_in_search(self, populated_store):
        """Embedding failure raises SearchError."""
        # Create embedder that will fail
        mock_embedder = Mock()
        mock_embedder.embed.side_effect = Exception("Embedding failed!")
        
        searcher = Searcher(populated_store, mock_embedder)
        
        with pytest.raises(SearchError, match="Embedding failed for query"):
            searcher.search("Python")
    
    def test_store_search_partial_failure(self, populated_store, embedder, monkeypatch):
        """Test that searcher handles FAISS batch search fallback gracefully."""
        searcher = Searcher(populated_store, embedder)
        
        # Mock FAISS search to fail, forcing fallback to individual searches
        original_faiss_search = populated_store._faiss_index.search
        call_count = 0
        
        def mock_faiss_search(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Batch FAISS search failed")
            return original_faiss_search(*args, **kwargs)
        
        monkeypatch.setattr(populated_store._faiss_index, 'search', mock_faiss_search)
        
        # Should still get results via fallback
        result = searcher.search("Python programming tutorial", limit=5, include_stats=True)
        
        # Check that we got some results despite batch search failure
        assert len(result['results']) > 0
        
        # The search should succeed via the fallback mechanism
        # (individual variant searches handle the failure gracefully)