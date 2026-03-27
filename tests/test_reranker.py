"""Tests for reranker.py module."""

import pytest
import numpy as np
import warnings
from unittest.mock import patch, MagicMock
from velocirag.reranker import Reranker, DEFAULT_MODEL, MAX_EXCERPT_LENGTH, EXCERPT_HEAD, EXCERPT_TAIL


class TestRerankerConstructor:
    """Test Reranker initialization."""

    def test_default_parameters(self):
        """Default constructor creates reranker with expected settings."""
        reranker = Reranker()

        assert reranker.model_name == DEFAULT_MODEL
        assert reranker._model_session is None   # ONNX session not loaded yet
        assert reranker._loaded is False
        assert reranker._load_error is None

    def test_custom_model_name(self):
        """Custom model name is stored correctly."""
        reranker = Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-12-v2")
        assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-12-v2"
        assert reranker._model_session is None
        assert reranker._loaded is False

    def test_model_not_loaded_until_rerank(self):
        """Model is not loaded until rerank() is called."""
        reranker = Reranker()

        assert reranker._model_session is None
        assert reranker._loaded is False

        # get_status should not trigger loading
        status = reranker.get_status()
        assert reranker._model_session is None
        assert reranker._loaded is False
        assert status['loaded'] is False


class TestCoreReranking:
    """Test core reranking functionality."""

    @pytest.fixture
    def mock_reranker(self):
        """Create a reranker with mocked _predict for predictable testing."""
        reranker = Reranker()
        reranker._loaded = True
        # Patch _predict at the instance level so call_args is inspectable
        reranker._predict = MagicMock(return_value=np.array([0.9, 0.7, 0.5, 0.3, 0.1]))
        return reranker

    def test_rerank_returns_sorted_by_score(self, mock_reranker):
        """rerank() returns results sorted by rerank_score descending."""
        query = "test query"
        results = [{"content": f"result {i}"} for i in range(5)]

        reranked = mock_reranker.rerank(query, results, limit=5)

        scores = [r['metadata']['rerank_score'] for r in reranked]
        assert scores == [0.9, 0.7, 0.5, 0.3, 0.1]
        assert reranked[0]['content'] == "result 0"
        assert reranked[1]['content'] == "result 1"

    def test_rerank_respects_limit(self, mock_reranker):
        """rerank() respects the limit parameter."""
        results = [{"content": f"result {i}"} for i in range(10)]
        mock_reranker._predict.return_value = np.array([1.0 - i * 0.1 for i in range(10)])

        reranked = mock_reranker.rerank("test", results, limit=3)

        assert len(reranked) == 3
        scores = [r['metadata']['rerank_score'] for r in reranked]
        assert scores[0] == pytest.approx(1.0, abs=1e-4)
        assert scores[1] == pytest.approx(0.9, abs=1e-4)
        assert scores[2] == pytest.approx(0.8, abs=1e-4)

    def test_rerank_score_added_to_metadata(self, mock_reranker):
        """rerank_score is added to result metadata."""
        results = [
            {"content": "result 1", "metadata": {"existing": "data"}},
            {"content": "result 2"},
        ]
        mock_reranker._predict.return_value = np.array([0.8, 0.6])

        reranked = mock_reranker.rerank("test", results, limit=2)

        assert reranked[0]['metadata']['existing'] == "data"
        assert reranked[0]['metadata']['rerank_score'] == pytest.approx(0.8, abs=1e-4)
        assert reranked[1]['metadata']['rerank_score'] == pytest.approx(0.6, abs=1e-4)

    def test_empty_results_returns_empty(self, mock_reranker):
        """Empty results list returns empty list without calling _predict."""
        reranked = mock_reranker.rerank("test", [], limit=5)

        assert reranked == []
        mock_reranker._predict.assert_not_called()

    def test_results_with_only_content_key(self, mock_reranker):
        """Results with only content key work."""
        results = [{"content": "just content 1"}, {"content": "just content 2"}]
        mock_reranker._predict.return_value = np.array([0.7, 0.8])

        reranked = mock_reranker.rerank("test", results, limit=2)

        assert reranked[0]['content'] == "just content 2"
        assert reranked[0]['metadata']['rerank_score'] == pytest.approx(0.8, abs=1e-4)

    def test_rerank_preserves_all_result_fields(self, mock_reranker):
        """rerank() preserves all fields from original results."""
        results = [{
            "content": "result 1",
            "metadata": {"original": "metadata"},
            "custom_field": "custom_value",
            "id": 123
        }]
        mock_reranker._predict.return_value = np.array([0.9])

        reranked = mock_reranker.rerank("test", results, limit=1)

        assert reranked[0]['content'] == "result 1"
        assert reranked[0]['metadata']['original'] == "metadata"
        assert reranked[0]['metadata']['rerank_score'] == pytest.approx(0.9, abs=1e-4)
        assert reranked[0]['custom_field'] == "custom_value"
        assert reranked[0]['id'] == 123

    def test_score_rounding(self, mock_reranker):
        """Scores are rounded to 4 decimal places."""
        mock_reranker._predict.return_value = np.array([0.123456789])
        reranked = mock_reranker.rerank("test", [{"content": "test"}], limit=1)
        assert reranked[0]['metadata']['rerank_score'] == 0.1235


class TestSmartExcerpting:
    """Test smart document excerpting functionality."""

    @pytest.fixture
    def mock_reranker(self):
        reranker = Reranker()
        reranker._loaded = True
        reranker._predict = MagicMock(return_value=np.array([0.8]))
        return reranker

    def test_long_documents_get_excerpted(self, mock_reranker):
        """Documents longer than MAX_EXCERPT_LENGTH get excerpted."""
        long_content = "word " * 1000  # ~5000 chars
        mock_reranker.rerank("test", [{"content": long_content}], limit=1)

        call_args = mock_reranker._predict.call_args[0][0]
        passed_content = call_args[0][1]

        assert len(passed_content) < len(long_content)
        assert "..." in passed_content
        assert passed_content.startswith(long_content[:50])
        assert passed_content.endswith(long_content[-50:])

    def test_short_documents_passed_unchanged(self, mock_reranker):
        """Documents shorter than MAX_EXCERPT_LENGTH passed unchanged."""
        short_content = "This is a short document that should not be excerpted."
        mock_reranker.rerank("test", [{"content": short_content}], limit=1)

        call_args = mock_reranker._predict.call_args[0][0]
        passed_content = call_args[0][1]

        assert passed_content == short_content

    def test_excerpt_format(self, mock_reranker):
        """Excerpt has correct format: head + ellipsis + tail."""
        head_part = "HEAD " * 250    # 1250 chars
        middle_part = "MIDDLE " * 500  # 3500 chars
        tail_part = "TAIL " * 250    # 1250 chars
        long_content = head_part + middle_part + tail_part

        mock_reranker.rerank("test", [{"content": long_content}], limit=1)

        call_args = mock_reranker._predict.call_args[0][0]
        excerpt = call_args[0][1]

        parts = excerpt.split("\n...\n")
        assert len(parts) == 2
        assert parts[0].startswith("HEAD")
        assert len(parts[0]) <= EXCERPT_HEAD
        assert parts[1].endswith("TAIL ")
        assert len(parts[1]) <= EXCERPT_TAIL

    def test_excerpt_boundary_cases(self, mock_reranker):
        """Excerpt handles content at exact boundary lengths."""
        exact_content = "x" * MAX_EXCERPT_LENGTH
        mock_reranker.rerank("test", [{"content": exact_content}], limit=1)
        call_args = mock_reranker._predict.call_args[0][0]
        assert call_args[0][1] == exact_content

        over_content = "x" * (MAX_EXCERPT_LENGTH + 1)
        mock_reranker._predict.return_value = np.array([0.8])
        mock_reranker.rerank("test", [{"content": over_content}], limit=1)
        call_args = mock_reranker._predict.call_args[0][0]
        assert "..." in call_args[0][1]


class TestGracefulDegradation:
    """Test graceful degradation when model fails to load."""

    def test_model_load_failure_returns_unranked(self):
        """When model fails to load, returns unranked results."""
        reranker = Reranker()
        # Simulate a load failure by injecting _load_error directly
        with patch.object(reranker, '_load_model', side_effect=lambda: setattr(reranker, '_load_error', 'onnxruntime not available')):
            results = [{"content": f"result {i}"} for i in range(5)]
            reranked = reranker.rerank("test query", results, limit=3)

            assert len(reranked) == 3
            assert reranked[0]['content'] == "result 0"
            assert reranked[1]['content'] == "result 1"
            assert reranked[2]['content'] == "result 2"
            assert 'rerank_score' not in reranked[0].get('metadata', {})

    def test_predict_failure_returns_unranked(self):
        """When _predict fails, returns unranked results."""
        reranker = Reranker()
        reranker._loaded = True
        reranker._predict = MagicMock(side_effect=Exception("Predict failed"))

        results = [{"content": f"result {i}"} for i in range(5)]
        reranked = reranker.rerank("test", results, limit=3)

        assert len(reranked) == 3
        assert reranked[0]['content'] == "result 0"
        assert 'rerank_score' not in reranked[0].get('metadata', {})

    def test_get_status_reports_load_state(self):
        """get_status() accurately reports load state."""
        reranker = Reranker()

        status = reranker.get_status()
        assert status['model_name'] == DEFAULT_MODEL
        assert status['loaded'] is False
        assert status['error'] is None

        # Inject error to simulate failed load
        reranker._load_error = "onnxruntime unavailable"
        status = reranker.get_status()
        assert status['loaded'] is False
        assert status['error'] == "onnxruntime unavailable"

        # Successful load state
        reranker2 = Reranker()
        reranker2._loaded = True
        status = reranker2.get_status()
        assert status['loaded'] is True
        assert status['error'] is None

    def test_repeated_calls_after_failure_dont_retry(self):
        """After load failure, subsequent calls don't retry loading."""
        reranker = Reranker()
        reranker._load_error = "model unavailable"

        load_mock = MagicMock()
        with patch.object(reranker, '_load_model', load_mock):
            result1 = reranker.rerank("test", [{"content": "test"}], limit=1)
            result2 = reranker.rerank("test", [{"content": "test"}], limit=1)

        # _load_model should NOT be called — error already set
        load_mock.assert_not_called()

        assert result1 == [{"content": "test"}]
        assert result2 == [{"content": "test"}]
        assert reranker._loaded is False
        assert reranker._load_error is not None

    def test_warning_logged_on_degradation(self):
        """Warning is logged when falling back to unranked results."""
        reranker = Reranker()
        reranker._load_error = "Test error"

        with patch('velocirag.reranker.logger') as mock_logger:
            reranker.rerank("test", [{"content": "test"}], limit=1)

            mock_logger.warning.assert_called()
            warning_msg = mock_logger.warning.call_args[0][0]
            assert "Reranker unavailable" in warning_msg
            assert "Test error" in warning_msg


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    @pytest.fixture
    def mock_reranker(self):
        reranker = Reranker()
        reranker._loaded = True
        reranker._predict = MagicMock(return_value=np.array([0.9]))
        return reranker

    def test_single_result_reranking(self, mock_reranker):
        mock_reranker._predict.return_value = np.array([0.95])
        reranked = mock_reranker.rerank("test", [{"content": "single result"}], limit=5)
        assert len(reranked) == 1
        assert reranked[0]['content'] == "single result"
        assert reranked[0]['metadata']['rerank_score'] == pytest.approx(0.95, abs=1e-4)

    def test_all_results_same_content(self, mock_reranker):
        results = [
            {"content": "same", "id": 1},
            {"content": "same", "id": 2},
            {"content": "same", "id": 3},
        ]
        mock_reranker._predict.return_value = np.array([0.8, 0.7, 0.9])
        reranked = mock_reranker.rerank("test", results, limit=3)
        assert reranked[0]['id'] == 3
        assert reranked[1]['id'] == 1
        assert reranked[2]['id'] == 2

    def test_unicode_content(self, mock_reranker):
        results = [
            {"content": "Hello 世界"},
            {"content": "Café ñoño"},
            {"content": "🚀 Rocket"},
        ]
        mock_reranker._predict.return_value = np.array([0.7, 0.8, 0.9])
        reranked = mock_reranker.rerank("test", results, limit=3)
        assert len(reranked) == 3
        assert reranked[0]['content'] == "🚀 Rocket"

    def test_missing_content_key(self, mock_reranker):
        results = [
            {"content": "has content"},
            {"no_content": "missing key"},
            {"content": "also has content"},
        ]
        mock_reranker._predict.return_value = np.array([0.8, 0.5, 0.7])
        reranked = mock_reranker.rerank("test", results, limit=3)
        assert len(reranked) == 3

    def test_limit_larger_than_results(self, mock_reranker):
        results = [{"content": f"result {i}"} for i in range(3)]
        mock_reranker._predict.return_value = np.array([0.7, 0.8, 0.6])
        reranked = mock_reranker.rerank("test", results, limit=10)
        assert len(reranked) == 3
        assert reranked[0]['content'] == "result 1"
        assert reranked[1]['content'] == "result 0"
        assert reranked[2]['content'] == "result 2"

    def test_zero_limit(self, mock_reranker):
        mock_reranker._predict.return_value = np.array([0.9])
        reranked = mock_reranker.rerank("test", [{"content": "result"}], limit=0)
        assert reranked == []

    def test_negative_limit(self, mock_reranker):
        mock_reranker._predict.return_value = np.array([0.9])
        reranked = mock_reranker.rerank("test", [{"content": "result"}], limit=-5)
        assert reranked == []

    def test_result_copying_prevents_mutation(self, mock_reranker):
        """Reranking creates copies of results, not in-place mutation."""
        original_results = [{"content": "test", "metadata": {"original": "value"}}]
        mock_reranker._predict.return_value = np.array([0.9])

        reranked = mock_reranker.rerank("test", original_results, limit=1)

        assert reranked[0]['metadata']['rerank_score'] == pytest.approx(0.9, abs=1e-4)
        assert reranked[0]['metadata']['original'] == "value"

    def test_callable_interface(self, mock_reranker):
        """Reranker is callable — can be passed as function."""
        mock_reranker._predict.return_value = np.array([0.8, 0.9])
        results = [{"content": "a"}, {"content": "b"}]

        # Call directly like a function
        reranked = mock_reranker("test query", results, limit=2)

        assert len(reranked) == 2
        assert reranked[0]['content'] == "b"  # Higher score
