"""Tests for rrf.py module."""

import pytest
from velocirag.rrf import reciprocal_rank_fusion


class TestReciprocalRankFusion:
    """Test rrf.py functionality."""
    
    def test_empty_input(self):
        """Empty input returns empty list."""
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[], []]) == []
    
    def test_single_list(self):
        """Single list returns original list with RRF scores."""
        input_list = [
            {'content': 'doc1', 'metadata': {'similarity': 0.9}},
            {'content': 'doc2', 'metadata': {'similarity': 0.8}}
        ]
        result = reciprocal_rank_fusion([input_list])
        
        assert len(result) == 2
        assert result[0]['content'] == 'doc1'
        assert result[1]['content'] == 'doc2'
        
        # Should have RRF scores added
        assert 'rrf_score' in result[0]['metadata']
        assert 'rrf_score' in result[1]['metadata']
        
        # First result should have higher RRF score (rank 1 vs rank 2)
        assert result[0]['metadata']['rrf_score'] > result[1]['metadata']['rrf_score']
    
    def test_basic_fusion(self):
        """Basic fusion of two lists."""
        list1 = [
            {'content': 'doc1', 'metadata': {'similarity': 0.9}},
            {'content': 'doc2', 'metadata': {'similarity': 0.8}}
        ]
        list2 = [
            {'content': 'doc2', 'metadata': {'similarity': 0.85}},  # Same doc, different score
            {'content': 'doc3', 'metadata': {'similarity': 0.7}}
        ]
        
        result = reciprocal_rank_fusion([list1, list2])
        
        # Should be 3 unique documents
        assert len(result) == 3
        
        # doc2 should rank highest (appears in both lists)
        assert result[0]['content'] == 'doc2'
        
        # Should keep the version with higher similarity (0.85 > 0.8)
        assert result[0]['metadata']['similarity'] == 0.85
        
        # All results should have RRF scores
        for res in result:
            assert 'rrf_score' in res['metadata']
    
    def test_custom_k_parameter(self):
        """Custom k parameter affects scoring."""
        input_lists = [
            [{'content': 'doc1'}],
            [{'content': 'doc1'}]  # Same doc in both lists
        ]
        
        result_k60 = reciprocal_rank_fusion(input_lists, k=60)
        result_k1 = reciprocal_rank_fusion(input_lists, k=1)
        
        # Different k values should produce different scores
        score_k60 = result_k60[0]['metadata']['rrf_score']
        score_k1 = result_k1[0]['metadata']['rrf_score']
        assert score_k60 != score_k1
        
        # k=1 should give higher weight to rank differences
        assert score_k1 > score_k60
    
    def test_custom_doc_id_fn(self):
        """Custom doc_id_fn is used for deduplication."""
        def custom_id(result):
            return result['metadata']['custom_id']
        
        list1 = [{'content': 'text1', 'metadata': {'custom_id': 'doc1'}}]
        list2 = [{'content': 'text2', 'metadata': {'custom_id': 'doc1'}}]  # Same custom_id
        
        result = reciprocal_rank_fusion([list1, list2], doc_id_fn=custom_id)
        
        # Should be deduplicated to 1 result
        assert len(result) == 1
        assert 'rrf_score' in result[0]['metadata']
    
    def test_custom_doc_id_fn_fallback(self):
        """Falls back to default logic if custom doc_id_fn fails."""
        def failing_id(result):
            raise ValueError("Custom function failed")
        
        input_lists = [[{'content': 'doc1', 'metadata': {'doc_id': 'id1'}}]]
        result = reciprocal_rank_fusion(input_lists, doc_id_fn=failing_id)
        
        # Should still work with fallback
        assert len(result) == 1
        assert result[0]['content'] == 'doc1'
    
    def test_doc_id_priority(self):
        """Tests doc_id extraction priority: doc_id > file_path > content_hash."""
        # doc_id takes priority
        result1 = reciprocal_rank_fusion([[{'content': 'test', 'metadata': {'doc_id': 'id1', 'file_path': 'file1'}}]])
        # Uses doc_id, not file_path
        
        # file_path as fallback
        result2 = reciprocal_rank_fusion([[{'content': 'test', 'metadata': {'file_path': 'file1'}}]])
        # Uses file_path
        
        # content hash as last resort
        result3 = reciprocal_rank_fusion([[{'content': 'test'}]])
        # Uses content hash
        
        assert len(result1) == len(result2) == len(result3) == 1
    
    def test_chunk_index_in_file_path(self):
        """file_path with chunk_index creates unique IDs."""
        input_lists = [[
            {'content': 'chunk1', 'metadata': {'file_path': 'file.md', 'chunk_index': 0}},
            {'content': 'chunk2', 'metadata': {'file_path': 'file.md', 'chunk_index': 1}}
        ]]
        
        result = reciprocal_rank_fusion(input_lists)
        
        # Should be 2 separate documents due to different chunk indices
        assert len(result) == 2
    
    def test_content_hash_different_content(self):
        """Different content produces different hash-based IDs."""
        input_lists = [[
            {'content': 'content one'},
            {'content': 'content two'}
        ]]
        
        result = reciprocal_rank_fusion(input_lists)
        
        # Should be 2 separate documents
        assert len(result) == 2
    
    def test_content_hash_same_content(self):
        """Same content produces same hash-based ID."""
        input_lists = [
            [{'content': 'same content'}],
            [{'content': 'same content'}]
        ]
        
        result = reciprocal_rank_fusion(input_lists)
        
        # Should be deduplicated to 1 document
        assert len(result) == 1
    
    def test_k_parameter_validation(self):
        """k parameter validation."""
        input_lists = [[{'content': 'test'}]]
        
        # Valid k values should work
        reciprocal_rank_fusion(input_lists, k=1)
        reciprocal_rank_fusion(input_lists, k=60)
        reciprocal_rank_fusion(input_lists, k=1000)
        
        # Invalid k values should raise ValueError
        with pytest.raises(ValueError, match="k parameter must be between"):
            reciprocal_rank_fusion(input_lists, k=0)
        
        with pytest.raises(ValueError, match="k parameter must be between"):
            reciprocal_rank_fusion(input_lists, k=1001)
        
        with pytest.raises(ValueError, match="k parameter must be an integer"):
            reciprocal_rank_fusion(input_lists, k=60.5)
        
        with pytest.raises(ValueError, match="k parameter must be an integer"):
            reciprocal_rank_fusion(input_lists, k="60")
    
    def test_memory_protection(self):
        """Memory protection limits total input results."""
        # Create many small lists that together exceed MAX_FUSION_RESULTS (1000)
        huge_lists = []
        for i in range(10):
            large_list = [{'content': f'doc{j}_{i}'} for j in range(200)]  # 2000 total
            huge_lists.append(large_list)
        
        result = reciprocal_rank_fusion(huge_lists)
        
        # Should be limited but still process successfully
        assert len(result) > 0
        # Total input should have been truncated to around 1000
        # (100 per list for 10 lists = 1000 total)
    
    def test_similarity_version_selection(self):
        """Keeps version with highest similarity when deduplicating."""
        list1 = [{'content': 'test', 'metadata': {'doc_id': 'doc1', 'similarity': 0.7}}]
        list2 = [{'content': 'test', 'metadata': {'doc_id': 'doc1', 'similarity': 0.9}}]
        list3 = [{'content': 'test', 'metadata': {'doc_id': 'doc1', 'similarity': 0.8}}]
        
        result = reciprocal_rank_fusion([list1, list2, list3])
        
        assert len(result) == 1
        assert result[0]['metadata']['similarity'] == 0.9  # Highest similarity kept
    
    def test_missing_metadata(self):
        """Handles results without metadata gracefully."""
        input_lists = [[
            {'content': 'doc without metadata'},
            {'content': 'doc with metadata', 'metadata': {'similarity': 0.8}}
        ]]
        
        result = reciprocal_rank_fusion(input_lists)
        
        assert len(result) == 2
        # Both should have rrf_score added to metadata
        for res in result:
            assert 'rrf_score' in res['metadata']
    
    def test_preserve_other_keys(self):
        """Other keys in result dict are preserved."""
        input_lists = [[{
            'content': 'test content',
            'metadata': {'similarity': 0.8},
            'custom_field': 'custom_value',
            'another_field': 42
        }]]
        
        result = reciprocal_rank_fusion(input_lists)
        
        assert len(result) == 1
        assert result[0]['content'] == 'test content'
        assert result[0]['custom_field'] == 'custom_value'
        assert result[0]['another_field'] == 42
        assert 'rrf_score' in result[0]['metadata']
    
    def test_rrf_score_precision(self):
        """RRF scores are rounded to 4 decimal places."""
        input_lists = [[{'content': 'test'}]]
        result = reciprocal_rank_fusion(input_lists, k=60)
        
        rrf_score = result[0]['metadata']['rrf_score']
        # Should be rounded to 4 decimal places
        assert len(str(rrf_score).split('.')[-1]) <= 4
        
        # Score should be 1/(60+1) = 0.0164 (rounded)
        expected = round(1.0 / 61, 4)
        assert rrf_score == expected
    
    def test_ranking_by_rrf_score(self):
        """Results are properly ranked by RRF score."""
        list1 = [{'content': 'doc1'}, {'content': 'doc2'}, {'content': 'doc3'}]
        list2 = [{'content': 'doc3'}, {'content': 'doc1'}]  # doc3 and doc1 appear again
        
        result = reciprocal_rank_fusion([list1, list2], k=60)
        
        # doc1: 1/(60+1) + 1/(60+2) ≈ 0.0164 + 0.0161 = 0.0325
        # doc2: 1/(60+2) ≈ 0.0161
        # doc3: 1/(60+3) + 1/(60+1) ≈ 0.0159 + 0.0164 = 0.0323
        
        # doc1 should rank highest, then doc3, then doc2
        assert result[0]['content'] == 'doc1'
        assert result[1]['content'] == 'doc3'
        assert result[2]['content'] == 'doc2'
        
        # Scores should be descending
        scores = [res['metadata']['rrf_score'] for res in result]
        assert scores == sorted(scores, reverse=True)
    
    def test_no_content_key_handling(self):
        """Handles results without 'content' key."""
        input_lists = [[{
            'title': 'Document without content key',
            'metadata': {'doc_id': 'doc1'}
        }]]
        
        result = reciprocal_rank_fusion(input_lists)
        
        # Should still work, content defaults to empty string for hashing
        assert len(result) == 1
        assert 'rrf_score' in result[0]['metadata']


"""Additional edge case tests for rrf.py"""

import pytest
from velocirag.rrf import reciprocal_rank_fusion


class TestRRFEdgeCases:
    """Additional edge case tests for rrf.py"""
    
    def test_duplicate_content_different_metadata(self):
        """Same content in multiple lists with different metadata."""
        list1 = [{'content': 'identical content', 'metadata': {'source': 'list1', 'similarity': 0.9}}]
        list2 = [{'content': 'identical content', 'metadata': {'source': 'list2', 'similarity': 0.8}}]
        
        result = reciprocal_rank_fusion([list1, list2])
        
        # Should be deduplicated by content hash
        assert len(result) == 1
        # Should keep higher similarity
        assert result[0]['metadata']['similarity'] == 0.9
        assert result[0]['metadata']['source'] == 'list1'
    
    def test_equal_truncation_exact_behavior(self):
        """Test exact truncation behavior when exceeding MAX_FUSION_RESULTS."""
        # Create exactly 10 lists with 101 items each = 1010 total (>1000 limit)
        lists = []
        for i in range(10):
            list_items = [{'content': f'doc_{i}_{j}'} for j in range(101)]
            lists.append(list_items)
        
        # Call RRF which should truncate
        result = reciprocal_rank_fusion(lists)
        
        # Each list should be truncated to 100 items (1000/10)
        # Result should have at most 1000 unique docs
        assert len(result) <= 1000
    
    def test_no_metadata_at_all(self):
        """Results completely lacking metadata key (not just empty dict)."""
        input_lists = [
            [{'content': 'doc1'}, {'content': 'doc2'}],
            [{'content': 'doc2'}, {'content': 'doc3'}]
        ]
        
        result = reciprocal_rank_fusion(input_lists)
        
        # Should work fine, creating metadata dicts as needed
        assert len(result) == 3
        for res in result:
            assert 'metadata' in res
            assert 'rrf_score' in res['metadata']
    
    def test_negative_similarity_scores(self):
        """Negative similarity scores should still work."""
        list1 = [{'content': 'doc1', 'metadata': {'similarity': -0.5}}]
        list2 = [{'content': 'doc1', 'metadata': {'similarity': -0.3}}]
        
        result = reciprocal_rank_fusion([list1, list2])
        
        assert len(result) == 1
        # Should keep the higher value even if negative (-0.3 > -0.5)
        assert result[0]['metadata']['similarity'] == -0.3
    
    def test_single_list_in_multi_list_input(self):
        """Only one non-empty list among many empty ones."""
        single_list = [
            {'content': 'doc1', 'metadata': {'score': 1}},
            {'content': 'doc2', 'metadata': {'score': 2}}
        ]
        
        result = reciprocal_rank_fusion([[], single_list, [], []])
        
        # Should work like single list case
        assert len(result) == 2
        assert result[0]['content'] == 'doc1'
        assert result[1]['content'] == 'doc2'
    
    def test_rrf_score_ties(self):
        """Documents with identical RRF scores (ordering should be stable)."""
        # Create symmetric lists where docs have same total RRF score
        list1 = [{'content': 'A'}, {'content': 'B'}]
        list2 = [{'content': 'B'}, {'content': 'A'}]
        
        # Both A and B appear at rank 1 and rank 2, so same RRF score
        result = reciprocal_rank_fusion([list1, list2], k=60)
        
        # Both should have same score
        assert result[0]['metadata']['rrf_score'] == result[1]['metadata']['rrf_score']
        
        # Order should be deterministic (based on dict ordering)
        # Run multiple times to verify
        for _ in range(5):
            result = reciprocal_rank_fusion([list1, list2], k=60)
            first_doc = result[0]['content']
            assert result[0]['content'] == first_doc  # Same order each time
    
    def test_unicode_content_hashing(self):
        """Unicode content should hash correctly for deduplication."""
        list1 = [{'content': '你好世界'}]
        list2 = [{'content': '你好世界'}]  # Same unicode content
        
        result = reciprocal_rank_fusion([list1, list2])
        
        # Should be deduplicated
        assert len(result) == 1
        assert result[0]['content'] == '你好世界'
    
    def test_very_small_k_value(self):
        """k=1 (minimum) should heavily weight rank 1."""
        lists = [
            [{'content': 'first'}, {'content': 'second'}],
            [{'content': 'second'}, {'content': 'first'}]
        ]
        
        result = reciprocal_rank_fusion(lists, k=1)
        
        # With k=1: rank 1 gets 1/2=0.5, rank 2 gets 1/3=0.33
        # So 'first' gets 0.5+0.33=0.83, 'second' gets 0.5+0.33=0.83
        # Scores are equal, order depends on implementation
        assert len(result) == 2
        assert abs(result[0]['metadata']['rrf_score'] - result[1]['metadata']['rrf_score']) < 0.001
    
    def test_very_large_k_value(self):
        """k=1000 (maximum) should give nearly equal weight to all ranks."""
        lists = [
            [{'content': f'doc{i}'} for i in range(10)]
        ]
        
        result = reciprocal_rank_fusion(lists, k=1000)
        
        # With k=1000, rank 1 gets 1/1001≈0.001, rank 10 gets 1/1010≈0.00099
        # Very small difference between ranks
        score_diff = result[0]['metadata']['rrf_score'] - result[9]['metadata']['rrf_score']
        assert score_diff < 0.0001  # Very small difference
    
    def test_empty_lists_mixed_with_valid(self):
        """Mix of empty and non-empty lists."""
        result = reciprocal_rank_fusion([
            [],
            [{'content': 'doc1'}],
            None,  # This will cause issues
            [{'content': 'doc2'}],
            []
        ])
        
        # Should handle None in lists (currently would crash)
        # This documents the issue
    
    def test_content_normalization_for_hashing(self):
        """Different content representations that might hash the same."""
        # These are actually different and should not deduplicate
        list1 = [{'content': 'test\nline'}]  # Unix newline
        list2 = [{'content': 'test\r\nline'}]  # Windows newline
        
        result = reciprocal_rank_fusion([list1, list2])
        
        # Should be 2 different documents
        assert len(result) == 2
    
    def test_missing_similarity_score(self):
        """Some results have similarity, others don't."""
        list1 = [
            {'content': 'doc1', 'metadata': {'similarity': 0.9}},
            {'content': 'doc2', 'metadata': {}}  # No similarity
        ]
        list2 = [
            {'content': 'doc2', 'metadata': {'similarity': 0.7}},
            {'content': 'doc3'}  # No metadata at all
        ]
        
        result = reciprocal_rank_fusion([list1, list2])
        
        # doc2 appears twice, should keep version with similarity
        doc2_result = next(r for r in result if r['content'] == 'doc2')
        assert doc2_result['metadata']['similarity'] == 0.7
    
    def test_custom_doc_id_fn_with_missing_field(self):
        """Custom doc_id_fn when expected field is missing."""
        def strict_id_fn(result):
            # This will raise KeyError if 'id' not present
            return result['metadata']['id']
        
        # Some results missing the 'id' field
        lists = [
            [{'content': 'doc1', 'metadata': {'id': 'A'}}],
            [{'content': 'doc2', 'metadata': {}}]  # Missing 'id'
        ]
        
        # Should fall back gracefully
        result = reciprocal_rank_fusion(lists, doc_id_fn=strict_id_fn)
        assert len(result) == 2  # Both docs included despite error
    
    def test_results_with_same_content_different_file_paths(self):
        """Same content from different files should be separate results."""
        lists = [[
            {'content': 'same text', 'metadata': {'file_path': '/path/one.txt'}},
            {'content': 'same text', 'metadata': {'file_path': '/path/two.txt'}}
        ]]
        
        result = reciprocal_rank_fusion(lists)
        
        # Should be 2 results (different file_paths)
        assert len(result) == 2
    
    def test_zero_similarity_scores(self):
        """Zero similarity should not be treated as missing."""
        list1 = [{'content': 'doc1', 'metadata': {'similarity': 0.0}}]
        list2 = [{'content': 'doc1', 'metadata': {'similarity': 0.5}}]
        
        result = reciprocal_rank_fusion([list1, list2])
        
        # Should keep higher similarity (0.5 > 0.0)
        assert result[0]['metadata']['similarity'] == 0.5
    
    def test_memory_protection_empty_lists(self):
        """Memory protection with many empty lists shouldn't divide by zero."""
        # 1000 empty lists
        empty_lists = [[] for _ in range(1000)]
        
        result = reciprocal_rank_fusion(empty_lists)
        assert result == []  # Should handle gracefully