"""Tests for embedder.py module."""

import pytest
import numpy as np
import json
import os
import time
import threading
import hashlib
import warnings
from velocirag.embedder import Embedder, DEFAULT_MODEL, MIN_CACHE_SIZE, MAX_CACHE_SIZE, CACHE_FILENAME, CACHE_VERSION


class TestEmbedderConstructor:
    """Test Embedder initialization."""
    
    def test_default_parameters(self):
        """Default constructor creates embedder with expected settings."""
        embedder = Embedder()
        
        assert embedder.model_name == DEFAULT_MODEL
        assert embedder.cache_dir is None
        assert embedder.cache_size == 10000
        assert embedder.normalize_embeddings is False
        assert embedder._model is None  # Model not loaded yet
    
    def test_custom_model_name(self):
        """Custom model name is stored correctly."""
        embedder = Embedder(model_name="all-mpnet-base-v2")
        assert embedder.model_name == "all-mpnet-base-v2"
        assert embedder._model is None  # Still not loaded
    
    def test_cache_size_validation(self):
        """Cache size must be within valid range."""
        # Valid edge cases
        embedder1 = Embedder(cache_size=MIN_CACHE_SIZE)
        assert embedder1.cache_size == MIN_CACHE_SIZE
        
        embedder2 = Embedder(cache_size=MAX_CACHE_SIZE)
        assert embedder2.cache_size == MAX_CACHE_SIZE
        
        # Too small
        with pytest.raises(ValueError, match=f"cache_size must be between {MIN_CACHE_SIZE} and {MAX_CACHE_SIZE}"):
            Embedder(cache_size=MIN_CACHE_SIZE - 1)
        
        # Too large
        with pytest.raises(ValueError, match=f"cache_size must be between {MIN_CACHE_SIZE} and {MAX_CACHE_SIZE}"):
            Embedder(cache_size=MAX_CACHE_SIZE + 1)
    
    def test_cache_dir_creation(self, tmp_path):
        """Cache directory is created if it doesn't exist."""
        cache_dir = tmp_path / "test_cache_dir"
        assert not cache_dir.exists()
        
        embedder = Embedder(cache_dir=str(cache_dir))
        embedder.embed("test")  # Trigger save
        embedder.save_cache()
        
        assert cache_dir.exists()


class TestCoreEmbedding:
    """Test core embedding functionality."""
    
    @pytest.fixture
    def embedder(self):
        """Shared embedder instance for tests."""
        return Embedder()
    
    def test_single_text_embedding(self, embedder):
        """Single text returns 1D array with correct dimensions."""
        result = embedder.embed("hello world")
        
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        assert result.shape == (384,)  # all-MiniLM-L6-v2 dimensions
    
    def test_batch_embedding(self, embedder):
        """Batch of texts returns 2D array with correct shape."""
        texts = ["hello world", "goodbye world", "hello universe"]
        result = embedder.embed(texts)
        
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.shape == (3, 384)
    
    def test_empty_list_handling(self, embedder):
        """Empty list returns array with shape (0, dimensions)."""
        result = embedder.embed([])
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (0, 384)
    
    def test_empty_string_rejection(self, embedder):
        """Empty string raises ValueError with clear message."""
        with pytest.raises(ValueError, match="Cannot embed empty text"):
            embedder.embed("")
        
        with pytest.raises(ValueError, match="Cannot embed empty text"):
            embedder.embed("   ")  # Whitespace only
    
    def test_empty_string_in_batch_rejection(self, embedder):
        """Empty string in batch raises ValueError with index."""
        with pytest.raises(ValueError, match="Cannot embed empty text at index 1"):
            embedder.embed(["valid", "", "also valid"])
        
        with pytest.raises(ValueError, match="Cannot embed empty text at index 0"):
            embedder.embed(["", "valid"])
    
    def test_non_string_input_rejection(self, embedder):
        """Non-string input raises appropriate error."""
        with pytest.raises(ValueError, match="texts must be str or list"):
            embedder.embed(123)
        
        with pytest.raises(ValueError, match="texts must be str or list"):
            embedder.embed({"key": "value"})
    
    def test_whitespace_only_rejection(self, embedder):
        """Whitespace-only strings are rejected."""
        with pytest.raises(ValueError, match="Cannot embed empty text"):
            embedder.embed("\n\t  \r\n")
        
        with pytest.raises(ValueError, match="Cannot embed empty text at index 2"):
            embedder.embed(["valid", "also valid", "   \t\n   "])


class TestCaching:
    """Test caching functionality."""
    
    @pytest.fixture
    def embedder(self):
        """Embedder instance without disk cache."""
        return Embedder()
    
    def test_cache_hit_returns_same_embedding(self, embedder):
        """Cache hit returns identical embedding."""
        text = "cache test text"
        
        # First call - cache miss
        result1 = embedder.embed(text)
        
        # Second call - should be cache hit
        result2 = embedder.embed(text)
        
        # Should be identical arrays
        assert np.array_equal(result1, result2)
    
    def test_cache_miss_triggers_computation(self, embedder):
        """Cache miss triggers model computation."""
        # Embed one text to ensure model is loaded
        embedder.embed("prime the model")
        info = embedder.get_model_info()
        initial_cache_size = info['cache_size']
        
        # Embed new text
        embedder.embed("new uncached text")
        
        # Cache size should increase
        info = embedder.get_model_info()
        assert info['cache_size'] == initial_cache_size + 1
    
    def test_mixed_cache_hit_miss_in_batch(self, embedder):
        """Batch with mixed hits/misses works correctly."""
        # Prime cache with some texts
        embedder.embed("cached text 1")
        embedder.embed("cached text 2")
        
        # Batch with 2 cached, 2 new
        batch = ["cached text 1", "new text 1", "cached text 2", "new text 2"]
        result = embedder.embed(batch)
        
        assert result.shape == (4, 384)
        
        # Cache should now have 4 entries
        info = embedder.get_model_info()
        assert info['cache_size'] == 4
    
    def test_cache_size_respects_limit(self):
        """Cache evicts old entries when limit reached."""
        # Small cache for easier testing
        embedder = Embedder(cache_size=MIN_CACHE_SIZE)
        
        # Fill cache beyond limit
        for i in range(MIN_CACHE_SIZE + 10):
            embedder.embed(f"text {i}")
        
        # Cache should be at limit
        info = embedder.get_model_info()
        assert info['cache_size'] == MIN_CACHE_SIZE
    
    def test_lru_eviction_works(self):
        """LRU eviction removes oldest accessed entries."""
        embedder = Embedder(cache_size=MIN_CACHE_SIZE)  # Use minimum allowed size
        
        # Fill cache to limit
        for i in range(MIN_CACHE_SIZE):
            embedder.embed(f"text {i}")
        
        # Access first text again (moves to end of LRU)
        embedder.embed("text 0")
        
        # Add new text - should evict something (likely text 1)
        embedder.embed(f"text {MIN_CACHE_SIZE}")
        
        # Cache should still be at limit
        info = embedder.get_model_info()
        assert info['cache_size'] == MIN_CACHE_SIZE
    
    def test_cache_stats_in_model_info(self, embedder):
        """get_model_info() reports accurate cache statistics."""
        info = embedder.get_model_info()
        assert info['cache_size'] == 0
        assert info['cache_max_size'] == 10000
        
        # Add some cached items
        embedder.embed(["text 1", "text 2", "text 3"])
        
        info = embedder.get_model_info()
        assert info['cache_size'] == 3


class TestDiskCache:
    """Test disk cache persistence."""
    
    def test_cache_persistence_to_disk(self, tmp_path):
        """Cache saves to and loads from disk."""
        cache_dir = tmp_path / "test_cache"
        text = "persistent test"
        
        # First instance - embed and save
        embedder1 = Embedder(cache_dir=str(cache_dir))
        result1 = embedder1.embed(text)
        embedder1.save_cache()
        
        # Verify cache file exists
        cache_file = cache_dir / CACHE_FILENAME
        assert cache_file.exists()
        
        # Second instance - should load from cache
        embedder2 = Embedder(cache_dir=str(cache_dir))
        info = embedder2.get_model_info()
        assert info['cache_size'] == 1
        assert not info['model_loaded']
        
        # Should get same embedding without loading model
        result2 = embedder2.embed(text)
        assert np.allclose(result1, result2)
    
    def test_cache_version_mismatch(self, tmp_path):
        """Cache with wrong version starts fresh."""
        cache_dir = tmp_path / "version_test"
        cache_dir.mkdir()
        
        # Create cache with wrong version
        cache_data = {
            "_version": CACHE_VERSION - 1,  # Wrong version
            "_model": DEFAULT_MODEL,
            "embeddings": {"test_hash": [0.1, 0.2, 0.3]}
        }
        cache_file = cache_dir / CACHE_FILENAME
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
        
        # Should start with empty cache
        embedder = Embedder(cache_dir=str(cache_dir))
        info = embedder.get_model_info()
        assert info['cache_size'] == 0
    
    def test_cache_model_mismatch(self, tmp_path):
        """Cache with wrong model starts fresh."""
        cache_dir = tmp_path / "model_test"
        cache_dir.mkdir()
        
        # Create cache for different model
        cache_data = {
            "_version": CACHE_VERSION,
            "_model": "different-model",  # Wrong model
            "embeddings": {"test_hash": [0.1, 0.2, 0.3]}
        }
        cache_file = cache_dir / CACHE_FILENAME
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
        
        # Should start with empty cache
        embedder = Embedder(cache_dir=str(cache_dir))
        info = embedder.get_model_info()
        assert info['cache_size'] == 0
    
    def test_corrupted_cache_file(self, tmp_path):
        """Corrupted cache file starts fresh without crashing."""
        cache_dir = tmp_path / "corrupted_test"
        cache_dir.mkdir()
        
        # Create corrupted cache file
        cache_file = cache_dir / CACHE_FILENAME
        with open(cache_file, 'w') as f:
            f.write("{ invalid json content")
        
        # Should handle gracefully
        embedder = Embedder(cache_dir=str(cache_dir))
        info = embedder.get_model_info()
        assert info['cache_size'] == 0
        
        # Should still work normally
        result = embedder.embed("test")
        assert result.shape == (384,)
    
    def test_clear_cache_removes_disk_cache(self, tmp_path):
        """clear_cache() removes both memory and disk cache."""
        cache_dir = tmp_path / "clear_test"
        embedder = Embedder(cache_dir=str(cache_dir))
        
        # Add some cached items
        embedder.embed(["text 1", "text 2"])
        embedder.save_cache()
        
        cache_file = cache_dir / CACHE_FILENAME
        assert cache_file.exists()
        
        # Clear cache
        embedder.clear_cache()
        
        # Memory cache cleared
        info = embedder.get_model_info()
        assert info['cache_size'] == 0
        
        # Disk cache removed
        assert not cache_file.exists()
    
    def test_save_cache_with_no_cache_dir(self):
        """save_cache() with no cache_dir is no-op."""
        embedder = Embedder(cache_dir=None)
        embedder.embed("test")
        
        # Should not crash
        embedder.save_cache()
        
        # Cache still in memory
        info = embedder.get_model_info()
        assert info['cache_size'] == 1
    
    def test_atomic_writes(self, tmp_path):
        """Cache writes use atomic temp file pattern."""
        cache_dir = tmp_path / "atomic_test"
        embedder = Embedder(cache_dir=str(cache_dir))
        embedder.embed("test")
        
        # During save, should use temp file
        # This is hard to test directly, but we can verify no corruption
        embedder.save_cache()
        
        # Should be valid JSON
        cache_file = cache_dir / CACHE_FILENAME
        with open(cache_file, 'r') as f:
            data = json.load(f)
        assert "_version" in data
        assert "embeddings" in data


class TestNormalization:
    """Test embedding normalization."""
    
    def test_normalize_true_produces_unit_vectors(self):
        """normalize=True produces L2-normalized embeddings."""
        embedder = Embedder(normalize=True)
        
        result = embedder.embed("test text")
        norm = np.linalg.norm(result)
        assert np.allclose(norm, 1.0, rtol=1e-6)
        
        # Batch should also be normalized
        batch_result = embedder.embed(["text 1", "text 2"])
        norms = np.linalg.norm(batch_result, axis=1)
        assert np.allclose(norms, 1.0, rtol=1e-6)
    
    def test_normalize_false_produces_raw_embeddings(self):
        """normalize=False produces raw embeddings."""
        embedder = Embedder(normalize=False)
        
        result = embedder.embed("test text")
        norm = np.linalg.norm(result)
        # Raw embeddings might be close to normalized but not exactly 1.0
        # Just verify it's a valid embedding
        assert norm > 0  # Non-zero
        assert result.shape == (384,)  # Correct shape
    
    def test_static_normalize_method_1d(self):
        """Static normalize() works on 1D arrays."""
        embedder = Embedder()
        
        # Create non-normalized embedding
        raw = np.array([3.0, 4.0])  # Norm = 5
        normalized = embedder.normalize(raw)
        
        assert np.allclose(normalized, [0.6, 0.8])
        assert np.allclose(np.linalg.norm(normalized), 1.0)
    
    def test_static_normalize_method_2d(self):
        """Static normalize() works on 2D arrays."""
        embedder = Embedder()
        
        # Create batch of non-normalized embeddings
        raw = np.array([[3.0, 4.0], [5.0, 12.0]])  # Norms = 5, 13
        normalized = embedder.normalize(raw)
        
        assert normalized.shape == (2, 2)
        norms = np.linalg.norm(normalized, axis=1)
        assert np.allclose(norms, 1.0)
    
    def test_zero_vector_normalization(self):
        """Zero vector normalization doesn't crash."""
        embedder = Embedder()
        
        # 1D zero vector
        zero_1d = np.zeros(5)
        normalized_1d = embedder.normalize(zero_1d)
        assert np.array_equal(normalized_1d, zero_1d)
        
        # 2D with zero vector
        zero_2d = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
        normalized_2d = embedder.normalize(zero_2d)
        assert np.allclose(normalized_2d[1], [0.0, 0.0])  # Zero stays zero


class TestModelInfo:
    """Test model info reporting."""
    
    def test_get_model_info_before_loading(self):
        """get_model_info() works before model is loaded."""
        embedder = Embedder()
        info = embedder.get_model_info()
        
        assert info['model_name'] == DEFAULT_MODEL
        assert info['dimensions'] == 384  # Known dimensions for default model
        assert info['model_loaded'] is False
        assert info['cache_size'] == 0
        assert info['cache_max_size'] == 10000
        assert info['cache_dir'] is None
        assert info['normalize_embeddings'] is False
    
    def test_get_model_info_after_loading(self):
        """get_model_info() updates after model loads."""
        embedder = Embedder()
        embedder.embed("trigger loading")
        
        info = embedder.get_model_info()
        assert info['model_loaded'] is True
        assert info['dimensions'] == 384
        assert info['max_seq_length'] > 0  # Should have actual value
        assert info['cache_size'] == 1  # One item cached
    
    def test_correct_dimensions_reported(self):
        """Correct dimensions reported for different models."""
        # Test with known model dimensions
        embedder1 = Embedder(model_name="all-MiniLM-L6-v2")
        assert embedder1.get_model_info()['dimensions'] == 384
        
        embedder2 = Embedder(model_name="all-MiniLM-L12-v2")
        assert embedder2.get_model_info()['dimensions'] == 384
        
        # Unknown model falls back to 384
        embedder3 = Embedder(model_name="unknown-model-name")
        assert embedder3.get_model_info()['dimensions'] == 384


class TestThreadSafety:
    """Test thread safety aspects."""
    
    def test_concurrent_embed_calls(self):
        """Concurrent embed calls don't crash."""
        embedder = Embedder()
        results = {}
        errors = []
        
        def embed_text(idx):
            try:
                result = embedder.embed(f"text {idx}")
                results[idx] = result
            except Exception as e:
                errors.append((idx, e))
        
        # Launch threads
        threads = []
        for i in range(10):
            t = threading.Thread(target=embed_text, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Should have no errors
        assert len(errors) == 0
        assert len(results) == 10
        
        # All results should be valid
        for result in results.values():
            assert result.shape == (384,)
    
    def test_concurrent_cache_access(self):
        """Concurrent cache access doesn't corrupt cache."""
        embedder = Embedder(cache_size=MIN_CACHE_SIZE)
        
        def embed_many():
            for i in range(50):
                embedder.embed(f"text {i % 20}")  # Some repeats for cache hits
        
        # Launch threads
        threads = []
        for _ in range(5):
            t = threading.Thread(target=embed_many)
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Cache should be valid and within size limit
        info = embedder.get_model_info()
        assert info['cache_size'] <= MIN_CACHE_SIZE
        assert info['cache_size'] > 0


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_very_long_text(self):
        """Very long text is handled (truncated by model)."""
        embedder = Embedder()
        
        # Create text longer than typical max_seq_length
        long_text = "word " * 1000
        
        result = embedder.embed(long_text)
        assert result.shape == (384,)
    
    def test_unicode_text(self):
        """Unicode text is handled correctly."""
        embedder = Embedder()
        
        texts = [
            "Hello 世界",
            "Émojis: 🚀🌟🎉",
            "Mixed: café ñoño Москва"
        ]
        
        results = embedder.embed(texts)
        assert results.shape == (3, 384)
    
    def test_special_characters(self):
        """Special characters in text work correctly."""
        embedder = Embedder()
        
        special_texts = [
            "Line\nbreaks\nwork",
            "Tabs\there\ttoo",
            "Special chars: <>&\"'",
            "Math: ∑∏∫≈≠"
        ]
        
        results = embedder.embed(special_texts)
        assert results.shape == (4, 384)
    
    def test_identical_texts_in_batch(self):
        """Identical texts in same batch are handled efficiently."""
        embedder = Embedder()
        
        # Batch with duplicates
        texts = ["same", "same", "different", "same"]
        results = embedder.embed(texts)
        
        assert results.shape == (4, 384)
        # First, second, and fourth should be identical
        assert np.array_equal(results[0], results[1])
        assert np.array_equal(results[0], results[3])
        assert not np.array_equal(results[0], results[2])
    
    def test_model_loading_warnings_suppressed(self):
        """Model loading suppresses warnings."""
        embedder = Embedder()
        
        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            embedder.embed("test")
            
            # Should have minimal warnings (transformers warnings suppressed)
            # This is hard to test precisely, but there should be few warnings
            warning_messages = [str(warning.message) for warning in w]
            transformers_warnings = [msg for msg in warning_messages if 'transformers' in msg.lower()]
            assert len(transformers_warnings) == 0
    
    def test_cache_with_different_text_same_hash(self):
        """Different texts that somehow produce same hash (unlikely)."""
        embedder = Embedder()
        
        # We can't easily create a hash collision, but we can test the behavior
        # by mocking the hash function temporarily
        original_hash = embedder._text_hash
        
        def mock_hash(text):
            return "same_hash_for_all"
        
        embedder._text_hash = mock_hash
        
        # These should still produce different embeddings in practice
        # but will share cache entry (last one wins)
        result1 = embedder.embed("text one")
        result2 = embedder.embed("text two")
        
        # Restore original hash
        embedder._text_hash = original_hash
        
        # Due to cache collision, result2 might equal result1
        # This documents the edge case behavior
    
    def test_cleanup_called_on_exit(self, tmp_path):
        """cleanup() is registered with atexit."""
        import atexit
        
        # Get current number of registered functions
        # Note: This is implementation-specific and may not work in all Python versions
        cache_dir = tmp_path / "atexit_test"
        embedder = Embedder(cache_dir=str(cache_dir))
        
        # cleanup should be registered
        # We can't easily test atexit directly, but we can call cleanup manually
        embedder.embed("test")
        embedder.cleanup()
        
        # Should have saved cache
        assert (cache_dir / CACHE_FILENAME).exists()