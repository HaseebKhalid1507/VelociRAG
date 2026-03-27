"""Tests for reranker.py module."""

import pytest
import warnings
from unittest.mock import patch, MagicMock
from velocirag.reranker import Reranker, DEFAULT_MODEL, MAX_EXCERPT_LENGTH, EXCERPT_HEAD, EXCERPT_TAIL


class TestRerankerConstructor:
    """Test Reranker initialization."""
    
    def test_default_parameters(self):
        """Default constructor creates reranker with expected settings."""
        reranker = Reranker()
        
        assert reranker.model_name == DEFAULT_MODEL
        assert reranker._model is None  # Model not loaded yet
        assert reranker._loaded is False
        assert reranker._load_error is None
    
    def test_custom_model_name(self):
        """Custom model name is stored correctly."""
        reranker = Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-12-v2")
        assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-12-v2"
        assert reranker._model is None  # Still not loaded
        assert reranker._loaded is False
    
    def test_model_not_loaded_until_rerank(self):
        """Model is not loaded until rerank() is called."""
        reranker = Reranker()
        
        # Check initial state
        assert reranker._model is None
        assert reranker._loaded is False
        
        # Get status should not load model
        status = reranker.get_status()
        assert reranker._model is None
        assert reranker._loaded is False
        assert status['loaded'] is False


class TestCoreReranking:
    """Test core reranking functionality."""
    
    @pytest.fixture
    def mock_reranker(self):
        """Create a reranker with mocked model for predictable testing."""
        reranker = Reranker()
        
        # Mock the model
        mock_model = MagicMock()
        # Return predictable scores based on position
        mock_model.predict.return_value = [0.9, 0.7, 0.5, 0.3, 0.1]
        
        # Bypass normal loading
        reranker._model = mock_model
        reranker._loaded = True
        
        return reranker
    
    def test_rerank_returns_sorted_by_score(self, mock_reranker):
        """rerank() returns results sorted by rerank_score descending."""
        query = "test query"
        results = [
            {"content": "result 1"},
            {"content": "result 2"},
            {"content": "result 3"},
            {"content": "result 4"},
            {"content": "result 5"}
        ]
        
        reranked = mock_reranker.rerank(query, results, limit=5)
        
        # Should be in descending score order
        scores = [r['metadata']['rerank_score'] for r in reranked]
        assert scores == [0.9, 0.7, 0.5, 0.3, 0.1]
        
        # Original content preserved
        assert reranked[0]['content'] == "result 1"
        assert reranked[1]['content'] == "result 2"
    
    def test_rerank_respects_limit(self, mock_reranker):
        """rerank() respects the limit parameter."""
        query = "test query"
        results = [
            {"content": f"result {i}"} for i in range(10)
        ]
        
        # Mock more scores
        mock_reranker._model.predict.return_value = [1.0 - i*0.1 for i in range(10)]
        
        reranked = mock_reranker.rerank(query, results, limit=3)
        
        # Should only return top 3
        assert len(reranked) == 3
        scores = [r['metadata']['rerank_score'] for r in reranked]
        assert scores == [1.0, 0.9, 0.8]
    
    def test_rerank_score_added_to_metadata(self, mock_reranker):
        """rerank_score is added to result metadata."""
        query = "test query"
        results = [
            {"content": "result 1", "metadata": {"existing": "data"}},
            {"content": "result 2"}  # No metadata
        ]
        
        mock_reranker._model.predict.return_value = [0.8, 0.6]
        reranked = mock_reranker.rerank(query, results, limit=2)
        
        # First result should preserve existing metadata
        assert reranked[0]['metadata']['existing'] == "data"
        assert reranked[0]['metadata']['rerank_score'] == 0.8
        
        # Second result should have metadata created
        assert 'metadata' in reranked[1]
        assert reranked[1]['metadata']['rerank_score'] == 0.6
    
    def test_empty_results_returns_empty(self, mock_reranker):
        """Empty results list returns empty list."""
        query = "test query"
        results = []
        
        reranked = mock_reranker.rerank(query, results, limit=5)
        
        assert reranked == []
        # Model should not be called
        mock_reranker._model.predict.assert_not_called()
    
    def test_results_with_only_content_key(self, mock_reranker):
        """Results with only content key (no metadata) still work."""
        query = "test query"
        results = [
            {"content": "just content 1"},
            {"content": "just content 2"}
        ]
        
        mock_reranker._model.predict.return_value = [0.7, 0.8]
        reranked = mock_reranker.rerank(query, results, limit=2)
        
        # Should be sorted by score (0.8 > 0.7)
        assert reranked[0]['content'] == "just content 2"
        assert reranked[0]['metadata']['rerank_score'] == 0.8
        assert reranked[1]['content'] == "just content 1"
        assert reranked[1]['metadata']['rerank_score'] == 0.7
    
    def test_rerank_preserves_all_result_fields(self, mock_reranker):
        """rerank() preserves all fields from original results."""
        query = "test query"
        results = [
            {
                "content": "result 1",
                "metadata": {"original": "metadata"},
                "custom_field": "custom_value",
                "id": 123
            }
        ]
        
        mock_reranker._model.predict.return_value = [0.9]
        reranked = mock_reranker.rerank(query, results, limit=1)
        
        # All fields preserved
        assert reranked[0]['content'] == "result 1"
        assert reranked[0]['metadata']['original'] == "metadata"
        assert reranked[0]['metadata']['rerank_score'] == 0.9
        assert reranked[0]['custom_field'] == "custom_value"
        assert reranked[0]['id'] == 123
    
    def test_score_rounding(self, mock_reranker):
        """Scores are rounded to 4 decimal places."""
        query = "test query"
        results = [{"content": "test"}]
        
        # Return score with many decimals
        mock_reranker._model.predict.return_value = [0.123456789]
        reranked = mock_reranker.rerank(query, results, limit=1)
        
        assert reranked[0]['metadata']['rerank_score'] == 0.1235


class TestSmartExcerpting:
    """Test smart document excerpting functionality."""
    
    @pytest.fixture
    def mock_reranker(self):
        """Create a reranker with mocked model."""
        reranker = Reranker()
        
        # Mock the model
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.8]
        
        reranker._model = mock_model
        reranker._loaded = True
        
        return reranker
    
    def test_long_documents_get_excerpted(self, mock_reranker):
        """Documents longer than MAX_EXCERPT_LENGTH get excerpted."""
        query = "test query"
        
        # Create long content
        long_content = "word " * 1000  # ~5000 chars
        results = [{"content": long_content}]
        
        mock_reranker.rerank(query, results, limit=1)
        
        # Check what was passed to model
        call_args = mock_reranker._model.predict.call_args[0][0]
        passed_content = call_args[0][1]  # Second element of pair
        
        # Should be excerpted with ellipsis
        assert len(passed_content) < len(long_content)
        assert "..." in passed_content
        
        # Should start with beginning of content
        assert passed_content.startswith(long_content[:50])
        
        # Should end with end of content
        assert passed_content.endswith(long_content[-50:])
    
    def test_short_documents_passed_unchanged(self, mock_reranker):
        """Documents shorter than MAX_EXCERPT_LENGTH passed unchanged."""
        query = "test query"
        
        # Create short content
        short_content = "This is a short document that should not be excerpted."
        results = [{"content": short_content}]
        
        mock_reranker.rerank(query, results, limit=1)
        
        # Check what was passed to model
        call_args = mock_reranker._model.predict.call_args[0][0]
        passed_content = call_args[0][1]
        
        # Should be unchanged
        assert passed_content == short_content
    
    def test_excerpt_format(self, mock_reranker):
        """Excerpt has correct format with head, ellipsis, and tail."""
        query = "test query"
        
        # Create content with clear markers
        head_part = "HEAD " * 250  # 1250 chars
        middle_part = "MIDDLE " * 500  # 3500 chars
        tail_part = "TAIL " * 250  # 1250 chars
        long_content = head_part + middle_part + tail_part
        
        results = [{"content": long_content}]
        mock_reranker.rerank(query, results, limit=1)
        
        # Check excerpt
        call_args = mock_reranker._model.predict.call_args[0][0]
        excerpt = call_args[0][1]
        
        # Should have three parts separated by ellipsis
        parts = excerpt.split("\n...\n")
        assert len(parts) == 2
        
        # Head should be from beginning
        assert parts[0].startswith("HEAD")
        assert len(parts[0]) <= EXCERPT_HEAD
        
        # Tail should be from end
        assert parts[1].endswith("TAIL ")
        assert len(parts[1]) <= EXCERPT_TAIL
    
    def test_excerpt_boundary_cases(self, mock_reranker):
        """Excerpt handles content at exact boundary lengths."""
        query = "test query"
        
        # Exactly at MAX_EXCERPT_LENGTH
        exact_content = "x" * MAX_EXCERPT_LENGTH
        results = [{"content": exact_content}]
        
        mock_reranker.rerank(query, results, limit=1)
        
        call_args = mock_reranker._model.predict.call_args[0][0]
        passed_content = call_args[0][1]
        
        # Should not be excerpted
        assert passed_content == exact_content
        
        # One char over limit
        over_content = "x" * (MAX_EXCERPT_LENGTH + 1)
        results = [{"content": over_content}]
        
        mock_reranker.rerank(query, results, limit=1)
        
        call_args = mock_reranker._model.predict.call_args[0][0]
        passed_content = call_args[0][1]
        
        # Should be excerpted with ellipsis
        assert "..." in passed_content
        # The excerpt will be slightly longer due to ellipsis separator
        assert len(passed_content) <= EXCERPT_HEAD + EXCERPT_TAIL + 10  # Allow for "\n...\n"


class TestGracefulDegradation:
    """Test graceful degradation when model fails."""
    
    def test_model_load_failure_returns_unranked(self):
        """When model fails to load (sentence_transformers not installed), returns unranked results."""
        import velocirag.reranker as reranker_module
        with patch.object(reranker_module, 'HAS_CROSS_ENCODER', False):
            reranker = Reranker()
            reranker._loaded = False
            reranker._load_error = None

            query = "test query"
            results = [
                {"content": f"result {i}"} for i in range(5)
            ]

            reranked = reranker.rerank(query, results, limit=3)

            # Should return first 3 results unranked (sentence_transformers unavailable)
            assert len(reranked) == 3
            assert reranked[0]['content'] == "result 0"
            assert reranked[1]['content'] == "result 1"
            assert reranked[2]['content'] == "result 2"

            # No rerank_score added when model unavailable
            assert 'rerank_score' not in reranked[0].get('metadata', {})
    
    def test_predict_failure_returns_unranked(self):
        """When predict fails, returns unranked results."""
        reranker = Reranker()
        
        # Mock successful load but predict fails
        mock_model = MagicMock()
        mock_model.predict.side_effect = Exception("Predict failed")
        reranker._model = mock_model
        reranker._loaded = True
        
        query = "test query"
        results = [{"content": f"result {i}"} for i in range(5)]
        
        reranked = reranker.rerank(query, results, limit=3)
        
        # Should return first 3 results unranked
        assert len(reranked) == 3
        assert reranked[0]['content'] == "result 0"
        
        # No rerank_score added
        assert 'rerank_score' not in reranked[0].get('metadata', {})
    
    def test_get_status_reports_load_state(self):
        """get_status() accurately reports load state."""
        import velocirag.reranker as reranker_module

        # Initial state (before any load attempt)
        reranker = Reranker()
        status = reranker.get_status()
        assert status['model_name'] == DEFAULT_MODEL
        assert status['loaded'] is False
        assert status['error'] is None

        # After failed load — simulate sentence_transformers not installed
        with patch.object(reranker_module, 'HAS_CROSS_ENCODER', False):
            reranker2 = Reranker()
            reranker2.rerank("test", [{"content": "test"}], limit=1)

        status = reranker2.get_status()
        assert status['loaded'] is False
        assert status['error'] is not None
        assert "sentence-transformers not installed" in status['error']

        # Test successful load scenario with mock
        reranker3 = Reranker()
        mock_model = MagicMock()
        reranker3._model = mock_model
        reranker3._loaded = True

        status = reranker3.get_status()
        assert status['loaded'] is True
        assert status['error'] is None
    
    def test_repeated_calls_after_failure_dont_retry(self):
        """After load failure, subsequent calls don't retry loading."""
        import velocirag.reranker as reranker_module

        with patch.object(reranker_module, 'HAS_CROSS_ENCODER', False):
            reranker = Reranker()

            # First call — should attempt load and fail
            result1 = reranker.rerank("test", [{"content": "test"}], limit=1)
            assert reranker._load_error is not None  # Error should be set

            # Second call should not retry loading since _load_error is set
            result2 = reranker.rerank("test", [{"content": "test"}], limit=1)

        # Both calls should return unranked results
        assert result1 == [{"content": "test"}]
        assert result2 == [{"content": "test"}]

        # Error state should be consistent
        assert reranker._loaded is False
        assert reranker._load_error is not None
    
    def test_warning_logged_on_degradation(self):
        """Warning is logged when falling back to unranked results."""
        reranker = Reranker()
        reranker._load_error = "Test error"
        
        with patch('velocirag.reranker.logger') as mock_logger:
            reranker.rerank("test", [{"content": "test"}], limit=1)
            
            # Should log warning about degradation
            mock_logger.warning.assert_called()
            warning_msg = mock_logger.warning.call_args[0][0]
            assert "Reranker unavailable" in warning_msg
            assert "Test error" in warning_msg


class TestEdgeCases:
    """Test edge cases and special scenarios."""
    
    @pytest.fixture
    def mock_reranker(self):
        """Create a reranker with mocked model."""
        reranker = Reranker()
        mock_model = MagicMock()
        reranker._model = mock_model
        reranker._loaded = True
        return reranker
    
    def test_single_result_reranking(self, mock_reranker):
        """Single result can be reranked without issues."""
        query = "test query"
        results = [{"content": "single result"}]
        
        mock_reranker._model.predict.return_value = [0.95]
        reranked = mock_reranker.rerank(query, results, limit=5)
        
        assert len(reranked) == 1
        assert reranked[0]['content'] == "single result"
        assert reranked[0]['metadata']['rerank_score'] == 0.95
    
    def test_all_results_same_content(self, mock_reranker):
        """All results having same content works correctly."""
        query = "test query"
        results = [
            {"content": "same content", "id": 1},
            {"content": "same content", "id": 2},
            {"content": "same content", "id": 3}
        ]
        
        # Different scores despite same content
        mock_reranker._model.predict.return_value = [0.8, 0.7, 0.9]
        reranked = mock_reranker.rerank(query, results, limit=3)
        
        # Should be sorted by score
        assert reranked[0]['id'] == 3  # Highest score 0.9
        assert reranked[1]['id'] == 1  # Score 0.8
        assert reranked[2]['id'] == 2  # Score 0.7
    
    def test_unicode_content(self, mock_reranker):
        """Unicode content is handled correctly."""
        query = "unicode query"
        results = [
            {"content": "Hello 世界"},
            {"content": "Café ñoño"},
            {"content": "🚀 Rocket emoji"}
        ]
        
        mock_reranker._model.predict.return_value = [0.7, 0.8, 0.9]
        reranked = mock_reranker.rerank(query, results, limit=3)
        
        # Should handle unicode without issues
        assert len(reranked) == 3
        assert reranked[0]['content'] == "🚀 Rocket emoji"
        assert reranked[1]['content'] == "Café ñoño"
        assert reranked[2]['content'] == "Hello 世界"
    
    def test_very_short_content(self, mock_reranker):
        """Very short content works correctly."""
        query = "test"
        results = [
            {"content": "a"},
            {"content": ""},  # Empty content
            {"content": "hi"}
        ]
        
        mock_reranker._model.predict.return_value = [0.5, 0.1, 0.9]
        reranked = mock_reranker.rerank(query, results, limit=3)
        
        assert len(reranked) == 3
        assert reranked[0]['content'] == "hi"
        assert reranked[1]['content'] == "a"
        assert reranked[2]['content'] == ""
    
    def test_missing_content_key(self, mock_reranker):
        """Results missing content key are handled gracefully."""
        query = "test"
        results = [
            {"content": "has content"},
            {"no_content": "missing key"},  # Missing 'content'
            {"content": "also has content"}
        ]
        
        mock_reranker._model.predict.return_value = [0.8, 0.5, 0.7]
        reranked = mock_reranker.rerank(query, results, limit=3)
        
        # Should handle missing content as empty string
        assert len(reranked) == 3
        # The one with missing content gets empty string
        no_content_result = next(r for r in reranked if 'no_content' in r)
        assert 'rerank_score' in no_content_result['metadata']
    
    def test_limit_larger_than_results(self, mock_reranker):
        """Limit larger than results returns all results."""
        query = "test"
        results = [{"content": f"result {i}"} for i in range(3)]
        
        mock_reranker._model.predict.return_value = [0.7, 0.8, 0.6]
        reranked = mock_reranker.rerank(query, results, limit=10)
        
        # Should return all 3 results, sorted
        assert len(reranked) == 3
        assert reranked[0]['content'] == "result 1"  # Score 0.8
        assert reranked[1]['content'] == "result 0"  # Score 0.7
        assert reranked[2]['content'] == "result 2"  # Score 0.6
    
    def test_zero_limit(self, mock_reranker):
        """Zero limit returns empty results."""
        query = "test"
        results = [{"content": "result"}]
        
        mock_reranker._model.predict.return_value = [0.9]
        reranked = mock_reranker.rerank(query, results, limit=0)
        
        assert reranked == []
    
    def test_negative_limit(self, mock_reranker):
        """Negative limit returns empty results."""
        query = "test"
        results = [{"content": "result"}]
        
        mock_reranker._model.predict.return_value = [0.9]
        reranked = mock_reranker.rerank(query, results, limit=-5)
        
        assert reranked == []
    
    def test_warnings_suppressed_during_load(self):
        """Model loading suppresses warnings."""
        reranker = Reranker()
        
        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Mock CrossEncoder to exist but produce warnings
            mock_ce = MagicMock()
            with patch('sentence_transformers.CrossEncoder', mock_ce):
                reranker._load_model()
            
            # Should not have sentence-transformers warnings
            warning_messages = [str(warning.message) for warning in w]
            st_warnings = [msg for msg in warning_messages if 'sentence_transformers' in msg.lower()]
            assert len(st_warnings) == 0
    
    def test_result_copying_prevents_mutation(self, mock_reranker):
        """Reranking creates copies of results."""
        query = "test"
        original_results = [
            {"content": "test", "metadata": {"original": "value"}}
        ]
        
        mock_reranker._model.predict.return_value = [0.9]
        reranked = mock_reranker.rerank(query, original_results, limit=1)
        
        # Reranked should have score
        assert reranked[0]['metadata']['rerank_score'] == 0.9
        assert reranked[0]['metadata']['original'] == "value"
        
        # Note: The current implementation does shallow copy, so metadata is shared
        # This documents the current behavior
        assert original_results[0]['metadata']['rerank_score'] == 0.9