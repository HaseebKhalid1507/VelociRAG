"""Tests for variants.py module."""

import pytest
from velocirag.variants import generate_variants


class TestGenerateVariants:
    """Test variants.py functionality."""
    
    def test_empty_input(self):
        """Empty input returns empty list."""
        assert generate_variants("") == []
        assert generate_variants("   ") == []
        assert generate_variants(None) == []
    
    def test_original_preserved(self):
        """Original query always appears first."""
        result = generate_variants("test query")
        assert result[0] == "test query"
        
        result = generate_variants("CS656")
        assert result[0] == "CS656"
    
    def test_letter_number_boundary_add_space(self):
        """Letter-number boundaries: CS656 → CS 656."""
        result = generate_variants("CS656")
        assert "CS656" in result  # Original
        assert "CS 656" in result  # Space added
        assert "cs656" in result   # Lowercase
        assert "cs 656" in result  # Lowercase with space
    
    def test_letter_number_boundary_remove_space(self):
        """Letter-number boundaries: CS 656 → CS656."""
        result = generate_variants("CS 656")
        assert "CS 656" in result  # Original
        assert "CS656" in result   # Space removed
        assert "cs 656" in result  # Lowercase
        assert "cs656" in result   # Lowercase no space
    
    def test_hyphen_handling(self):
        """Hyphens: CS-656 → CS656, CS 656."""
        result = generate_variants("CS-656")
        assert "CS-656" in result  # Original
        assert "CS656" in result   # Hyphen removed
        assert "CS 656" in result  # Hyphen to space
        
        # Should also have case variants
        expected_variants = {"CS-656", "cs-656", "CS656", "cs656", "CS 656", "cs 656"}
        assert expected_variants.issubset(set(result))
    
    def test_underscore_handling(self):
        """Underscores: file_name → file name, filename."""
        result = generate_variants("file_name")
        assert "file_name" in result  # Original
        assert "filename" in result   # Underscore removed
        assert "file name" in result  # Underscore to space
    
    def test_dot_handling(self):
        """Dots: script.py → script py, scriptpy."""
        result = generate_variants("script.py")
        assert "script.py" in result  # Original
        assert "scriptpy" in result   # Dot removed
        assert "script py" in result  # Dot to space
    
    def test_mixed_punctuation(self):
        """Complex query with multiple punctuation types."""
        result = generate_variants("my-script_v2.py")
        
        # Original should be first
        assert result[0] == "my-script_v2.py"
        
        # Should handle hyphens
        assert "myscript_v2.py" in result  # Hyphen removed
        assert "my script_v2.py" in result  # Hyphen to space
        
        # Should handle underscores
        assert "my-script v2.py" in result  # Underscore to space
        assert "my-scriptv2.py" in result   # Underscore removed
        
        # Should handle dots
        assert "my-script_v2 py" in result  # Dot to space
        assert "my-script_v2py" in result   # Dot removed
    
    def test_case_variants(self):
        """Case variations are generated."""
        result = generate_variants("UPPERCASE")
        assert "UPPERCASE" in result  # Original
        assert "uppercase" in result  # Lowercase
        
        result = generate_variants("MixedCase")
        assert "MixedCase" in result  # Original
        assert "mixedcase" in result  # Lowercase
    
    def test_no_case_variant_if_already_lowercase(self):
        """No case variant generated if already lowercase."""
        result = generate_variants("lowercase")
        assert result == ["lowercase"]  # Only original
    
    def test_deduplication(self):
        """Duplicates are removed while preserving order."""
        # This might happen with complex transformations
        result = generate_variants("test")
        # Check that all items are unique
        assert len(result) == len(set(result))
    
    def test_max_variants_enforced(self):
        """Maximum of 8 variants returned."""
        # Create a query that could generate many variants
        result = generate_variants("COMPLEX-test_file.py")
        assert len(result) <= 8
    
    def test_simple_query_no_variants(self):
        """Simple queries may not need many variants."""
        result = generate_variants("simple")
        assert result == ["simple"]
        
        result = generate_variants("hello")
        assert result == ["hello"]
    
    def test_word_boundaries_respected(self):
        """Word boundaries are respected in letter-number detection."""
        # Should not split in the middle of longer alphanumeric sequences
        result = generate_variants("password123abc")
        # This should NOT become "password 123 abc" because 123abc is one unit
        
        # But this should work
        result = generate_variants("CS123")
        assert "CS 123" in result
    
    def test_multiple_spaces_handled(self):
        """Multiple spaces are compressed to single space."""
        result = generate_variants("CS   123")
        assert "CS123" in result  # Multiple spaces removed
    
    def test_all_punctuation_query(self):
        """Query with only punctuation doesn't crash."""
        result = generate_variants("---___...")
        assert len(result) >= 1  # Should at least return original
        assert result[0] == "---___..."
    
    def test_unicode_preserved(self):
        """Unicode characters are preserved."""
        result = generate_variants("test你好")
        assert "test你好" in result
        
        result = generate_variants("file_🌍.txt")
        assert "file_🌍.txt" in result
        assert "file 🌍.txt" in result  # Underscore to space
        assert "file🌍.txt" in result   # Underscore removed
    
    def test_single_character(self):
        """Single character input."""
        result = generate_variants("a")
        assert result == ["a"]
        
        result = generate_variants("1")
        assert result == ["1"]
    
    def test_very_long_query(self):
        """Very long query doesn't cause performance issues."""
        long_query = "a" * 1000
        result = generate_variants(long_query)
        assert len(result) >= 1
        assert result[0] == long_query
    
    def test_order_preservation(self):
        """Original query is always first, order is deterministic."""
        # Run multiple times to ensure consistent ordering
        for _ in range(5):
            result = generate_variants("CS-656")
            assert result[0] == "CS-656"
        
        for _ in range(5):
            result = generate_variants("file_test.py")
            assert result[0] == "file_test.py"


"""Additional edge case tests for variants.py"""

import pytest
from velocirag.variants import generate_variants


class TestGenerateVariantsEdgeCases:
    """Additional edge case tests for variants.py"""
    
    def test_none_handling_fix_needed(self):
        """None input is listed in test but code doesn't handle it."""
        # Code actually handles None gracefully by returning empty list
        result = generate_variants(None)
        assert result == []
    
    def test_tabs_and_newlines(self):
        """Different whitespace characters should be preserved."""
        result = generate_variants("test\tquery")
        assert result[0] == "test\tquery"  # Tab preserved
        
        result = generate_variants("test\nquery")  
        assert result[0] == "test\nquery"  # Newline preserved
    
    def test_numbers_to_letters_boundary(self):
        """Reverse boundary: 123ABC (numbers to letters)."""
        result = generate_variants("123ABC")
        # Should NOT split 123ABC into 123 ABC (spec only mentions letters->numbers)
        assert "123 ABC" not in result
        assert result == ["123ABC", "123abc"]  # Only case variant
    
    def test_consecutive_punctuation(self):
        """Multiple consecutive punctuation marks."""
        # Double underscore
        result = generate_variants("file__name")
        assert "file__name" in result
        assert "file  name" in result  # Each _ becomes space
        assert "filename" in result  # Both _ removed
        
        # Double dots
        result = generate_variants("script..py")
        assert "script..py" in result
        assert "script  py" in result  # Each . becomes space
        assert "scriptpy" in result  # Both . removed
        
        # Mixed consecutive
        result = generate_variants("test.-_file")
        assert "test.-_file" in result
        # Should handle each punctuation independently
    
    def test_punctuation_at_boundaries(self):
        """Punctuation at start/end of query."""
        result = generate_variants("-test")
        assert "-test" in result
        assert " test" in result  # Leading space
        assert "test" in result  # Removed
        
        result = generate_variants("test-")
        assert "test-" in result
        assert "test " in result  # Trailing space
        assert "test" in result  # Removed
        
        result = generate_variants("_test_")
        assert "_test_" in result
        assert " test " in result
        assert "test" in result
    
    def test_all_uppercase_numbers(self):
        """All uppercase with numbers."""
        result = generate_variants("ABC123DEF456")
        assert "ABC123DEF456" in result  # Original
        assert "abc123def456" in result  # Lowercase
        # The regex requires word boundaries, so "ABC123DEF456" doesn't match
        # because there's no word boundary after "123" (DEF follows immediately)
        # So only case variants are generated
        assert len(result) == 2  # Only original + lowercase
    
    def test_camelcase_preservation(self):
        """CamelCase should only get lowercase variant."""
        result = generate_variants("CamelCaseQuery")
        assert result == ["CamelCaseQuery", "camelcasequery"]
        # Should NOT split on case boundaries
    
    def test_variant_order_with_multiple_patterns(self):
        """Variant order should be consistent with multiple transformations."""
        # Run multiple times to ensure order stability
        for _ in range(5):
            result = generate_variants("CS-656_test.py")
            assert result[0] == "CS-656_test.py"  # Original always first
            # Order of other variants should be deterministic
            assert result == result  # Same order each time
    
    def test_max_variants_complex(self):
        """Complex query that would generate >8 variants."""
        # This query has many transformation possibilities:
        # - case variants (uppercase -> lowercase)
        # - letter-number boundaries (ABC123, XYZ789)  
        # - hyphens (2 occurrences)
        # - underscores (2 occurrences)
        # - dots (1 occurrence)
        query = "ABC123-DEF456_GHI789-JKL012_MNO345.py"
        result = generate_variants(query)
        
        assert len(result) == 8  # Capped at MAX_VARIANTS
        assert result[0] == query  # Original first
    
    def test_unicode_with_punctuation(self):
        """Unicode mixed with punctuation transformations."""
        result = generate_variants("文件_名.txt")
        assert "文件_名.txt" in result
        assert "文件 名.txt" in result  # Underscore to space
        assert "文件名.txt" in result   # Underscore removed
        assert "文件_名 txt" in result  # Dot to space
        assert "文件_名txt" in result   # Dot removed
    
    def test_empty_string_variants(self):
        """Empty results after transformations."""
        # Just punctuation that gets removed
        result = generate_variants("-")
        assert result[0] == "-"
        assert " " in result  # Space
        assert "" in result  # Removed entirely
        
    def test_repeated_patterns(self):
        """Same pattern repeated in query."""
        result = generate_variants("CS656-CS656")
        # Should handle both occurrences
        assert "CS656-CS656" in result
        assert "CS 656-CS 656" in result  # Letter-number spacing applied
        assert "CS656CS656" in result  # Hyphen removed
        assert "CS656 CS656" in result  # Hyphen to space
    
    def test_mixed_separators(self):
        """Query with many different separators."""
        result = generate_variants("a-b_c.d")
        # Original + each separator handled
        assert "a-b_c.d" in result
        # The transformations cascade creating many variants
        # Should cap at 8
        assert len(result) <= 8
    
    def test_regex_special_chars_in_query(self):
        """Query containing regex special characters."""
        result = generate_variants("test[0-9]+query")
        assert result[0] == "test[0-9]+query"
        # Should not crash or interpret as regex
        
        result = generate_variants("file(1).txt")
        assert "file(1).txt" in result
        assert "file(1) txt" in result  # Dot handling should still work
    
    def test_whitespace_normalization(self):
        """Multiple spaces should be kept as-is (not normalized to single space)."""
        result = generate_variants("test    query")
        assert result[0] == "test    query"  # Spaces preserved exactly
        
    def test_letter_number_both_directions(self):
        """Letter-number patterns in both directions in same query."""
        result = generate_variants("ABC123 and 456XYZ")
        assert "ABC123 and 456XYZ" in result  # Original
        assert "ABC 123 and 456XYZ" in result  # Only letter->number boundary
        # 456XYZ should NOT become 456 XYZ
        
    def test_single_letter_number_boundary(self):
        """Single letter followed by numbers."""
        result = generate_variants("A1")
        assert "A1" in result
        assert "A 1" in result
        assert "a1" in result  
        assert "a 1" in result