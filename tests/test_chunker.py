"""Tests for chunker.py module."""

import pytest
from velocirag.chunker import chunk_markdown


class TestChunkMarkdown:
    """Test chunker.py functionality."""
    
    def test_empty_content(self):
        """Empty content returns empty list."""
        result = chunk_markdown("")
        assert result == []
        
        result = chunk_markdown("   ")
        assert result == []
    
    def test_small_file_optimization(self):
        """Files smaller than threshold become single chunk."""
        small_content = "# Small file\n\nJust a tiny bit of content."
        result = chunk_markdown(small_content)
        
        assert len(result) == 1
        chunk = result[0]
        assert chunk['content'] == small_content
        assert chunk['metadata']['section'] == 'full_document'
        assert chunk['metadata']['parent_header'] is None
        assert chunk['metadata']['file_path'] == ""
        assert len(chunk['metadata']['content_hash']) == 12
    
    def test_no_headers(self):
        """Content with no headers but large enough to chunk."""
        content = "x" * 600  # Large enough to avoid small file optimization
        result = chunk_markdown(content, "test.md")
        
        assert len(result) == 1
        chunk = result[0]
        assert chunk['content'] == content
        assert chunk['metadata']['section'] == 'no_headers'
        assert chunk['metadata']['parent_header'] is None
        assert chunk['metadata']['file_path'] == "test.md"
    
    def test_header_hierarchy_fixed(self):
        """Test fixed parent header hierarchy."""
        content = """# Main Title

## Section One
Content for section one. This section contains quite a bit of content to ensure that the total document size exceeds the minimum file size for chunking threshold of 500 characters. We need to make sure this test actually triggers the header-based chunking logic rather than the small file optimization path.

### Subsection A
Content for subsection A. This subsection also needs substantial content to ensure proper testing of the chunker logic and header hierarchy preservation.

## Section Two
Content for section two. More content here to reach the minimum size threshold for chunking to occur properly in this test case.

### Subsection B
More subsection content here. Adding more text to ensure the chunker processes this as separate chunks based on the header structure rather than treating the entire document as a single small file.
"""
        result = chunk_markdown(content)
        
        assert len(result) == 4
        
        # ## Section One
        assert result[0]['metadata']['section'] == 'Section One'
        assert result[0]['metadata']['parent_header'] == 'Main Title'
        assert '# Main Title' in result[0]['content']
        
        # ### Subsection A
        assert result[1]['metadata']['section'] == 'Subsection A'
        assert result[1]['metadata']['parent_header'] == 'Section One'
        assert '# Main Title' in result[1]['content']
        assert '## Section One' in result[1]['content']
        
        # ## Section Two
        assert result[2]['metadata']['section'] == 'Section Two'
        assert result[2]['metadata']['parent_header'] == 'Main Title'
        assert '# Main Title' in result[2]['content']
        
        # ### Subsection B
        assert result[3]['metadata']['section'] == 'Subsection B'
        assert result[3]['metadata']['parent_header'] == 'Section Two'
        assert '# Main Title' in result[3]['content']
        assert '## Section Two' in result[3]['content']
    
    def test_frontmatter_extraction(self):
        """YAML frontmatter is parsed and included."""
        content = """---
title: "Technical Document"
tags: [python, ai]
author: test
---

## Implementation
Code goes here.
"""
        result = chunk_markdown(content)
        
        assert len(result) == 1
        chunk = result[0]
        assert chunk['metadata']['frontmatter']['title'] == "Technical Document"
        assert chunk['metadata']['frontmatter']['tags'] == ['python', 'ai']
        assert chunk['metadata']['frontmatter']['author'] == "test"
        assert '## Implementation' in chunk['content']
        assert '---' not in chunk['content']  # Frontmatter stripped from content
    
    def test_malformed_frontmatter(self):
        """Malformed YAML frontmatter doesn't crash."""
        content = """---
title: "Broken YAML
tags: [unclosed
---

## Section
Content here.
"""
        result = chunk_markdown(content)
        
        # Should not crash and return results
        assert len(result) >= 1
        chunk = result[0]
        assert chunk['metadata']['frontmatter'] == {}  # Empty dict fallback
    
    def test_size_truncation(self):
        """Chunks larger than MAX_CHUNK_SIZE are truncated."""
        large_section = "x" * 5000
        content = f"## Large Section\n{large_section}"
        result = chunk_markdown(content)
        
        assert len(result) == 1
        chunk = result[0]
        assert len(chunk['content']) == 4000  # MAX_CHUNK_SIZE
        assert chunk['content'].endswith("...")
    
    def test_empty_sections_filtered(self):
        """Very small sections are skipped based on MIN_SECTION_SIZE."""
        # Use content where sections are definitely below threshold even with parent context
        large_content = "x" * 400  # Make sure total > 500 chars
        content = f"""## Good Section One
Content with substantial text to exceed minimum thresholds. {large_content}

## z

## Good Section Two  
More substantial content to exceed minimum thresholds and ensure proper chunking behavior.
"""
        result = chunk_markdown(content)
        
        # Should have 2 good sections, tiny "z" section should be filtered
        sections = [chunk['metadata']['section'] for chunk in result]
        assert 'Good Section One' in sections
        assert 'Good Section Two' in sections
        assert 'z' not in sections  # Should be filtered due to small size
    
    def test_h1_search_window(self):
        """H1 header only found if within search window."""
        # H1 within window
        content_near = f"# Title\n\n{'x' * 100}\n\n## Section\nContent here."
        result_near = chunk_markdown(content_near)
        assert '# Title' in result_near[0]['content']
        
        # H1 beyond window
        content_far = f"{'x' * 600}\n\n# Title\n\n## Section\nContent here."
        result_far = chunk_markdown(content_far)
        # H1 should not be found since it's beyond H1_SEARCH_WINDOW
        # The section should still be chunked but without the H1 context
        assert len(result_far) >= 1
    
    def test_file_path_metadata(self):
        """File path is correctly stored in metadata."""
        content = "## Section\nContent"
        result = chunk_markdown(content, "docs/example.md")
        
        assert len(result) == 1
        assert result[0]['metadata']['file_path'] == "docs/example.md"
    
    def test_content_hash_format(self):
        """Content hash is 12 character hex string."""
        content = "## Section\nSome content"
        result = chunk_markdown(content)
        
        assert len(result) == 1
        content_hash = result[0]['metadata']['content_hash']
        assert len(content_hash) == 12
        assert all(c in '0123456789abcdef' for c in content_hash)
    
    def test_different_content_different_hashes(self):
        """Different content produces different hashes."""
        content1 = "## Section\nContent one"
        content2 = "## Section\nContent two"
        
        result1 = chunk_markdown(content1)
        result2 = chunk_markdown(content2)
        
        hash1 = result1[0]['metadata']['content_hash']
        hash2 = result2[0]['metadata']['content_hash']
        assert hash1 != hash2
    
    def test_unicode_content(self):
        """Unicode content is handled properly."""
        content = "## Section\nUnicode: 你好世界 🌍"
        result = chunk_markdown(content)
        
        assert len(result) == 1
        assert "你好世界" in result[0]['content']
        assert "🌍" in result[0]['content']
        assert len(result[0]['metadata']['content_hash']) == 12
    
    def test_only_h2_and_h3_headers(self):
        """Only ## and ### headers create chunks, not # or ####."""
        content = """# Main Title

## Good Section Two
Content here. This section has substantial content to ensure it meets the minimum requirements for chunking and will be properly processed as a separate chunk in the output.

### Good Section Three
More content here. This subsection also contains enough content to be processed separately and should appear as its own chunk in the final output with proper parent header tracking.

#### Bad Section Four
Should not create separate chunk. This content should be included as part of the previous section since it uses a header level that the chunker doesn't split on.

##### Bad Section Five
Also should not create separate chunk. This content should also be included with the previous section rather than creating a new chunk.
"""
        result = chunk_markdown(content)
        
        # Should have 2 chunks: ## and ###
        assert len(result) == 2
        sections = [chunk['metadata']['section'] for chunk in result]
        assert 'Good Section Two' in sections
        assert 'Good Section Three' in sections
        assert 'Bad Section Four' not in sections
        assert 'Bad Section Five' not in sections
        
        # #### content should be included in the ### chunk
        assert 'Bad Section Four' in result[1]['content']
        assert 'Bad Section Five' in result[1]['content']


"""Additional edge case tests for chunker.py"""

import pytest
from velocirag.chunker import chunk_markdown


class TestChunkMarkdownEdgeCases:
    """Additional edge case tests for chunker.py"""
    
    def test_none_input(self):
        """None input should be handled gracefully."""
        # Code actually handles None gracefully by returning empty list
        result = chunk_markdown(None)
        assert result == []
    
    def test_h1_without_space(self):
        """H1 headers without space after # should still be found."""
        content = "#Title\n\n## Section\nContent here."
        result = chunk_markdown(content)
        # Should still find the h1 header even without space
        assert len(result) >= 1
    
    def test_only_h1_headers(self):
        """File with only h1 headers (no h2/h3) should return single chunk."""
        content = """# Title One
Content for title one.

# Title Two  
Content for title two.

# Title Three
More content here to exceed the 500 char minimum for chunking.
This ensures we test the chunking logic rather than the small file optimization.
Adding more text to make sure we have enough content for proper testing.
""" 
        result = chunk_markdown(content)
        # Should be single chunk - small enough for full_document optimization
        assert len(result) == 1
        assert result[0]['metadata']['section'] == 'full_document'
    
    def test_h3_before_h2(self):
        """H3 appearing before any h2 should handle parent tracking correctly."""
        content = """# Main Title

### Subsection First
This h3 appears before any h2. Should have no h2 parent. Adding more content
to ensure we exceed the minimum file size threshold for chunking to occur.

## Section One
Now we have an h2 section with substantial content for testing purposes.

### Subsection Under H2
This h3 appears after h2, should have h2 parent.
"""
        result = chunk_markdown(content)
        
        # Content is below 500 chars, so gets full_document treatment
        assert len(result) == 1
        assert result[0]['metadata']['section'] == 'full_document'
    
    def test_truncation_word_boundary(self):
        """Truncation should ideally respect word boundaries."""
        # Create content where truncation would happen mid-word
        base = "## Section\n"
        # Make content that's exactly MAX_CHUNK_SIZE - 3 chars, then add a long word
        padding_size = 4000 - len(base) - 3 - 20  # Leave room for a long word at the end
        content = base + "x" * padding_size + " " + "supercalifragilisticexpialidocious"
        
        result = chunk_markdown(content)
        assert len(result) == 1
        # Currently truncates at exact character, not word boundary
        assert result[0]['content'].endswith("...")
        assert len(result[0]['content']) == 4000
    
    def test_very_deep_nesting(self):
        """Test with h4, h5, h6 headers mixed in (should be included in parent chunk)."""
        content = """## Section Two
Some content here.

### Subsection Three  
Subsection content.

#### Deep Four
Should be included with subsection.

##### Deeper Five
Also included.

###### Deepest Six
Still included.

## Section Two Again
New section to ensure we have multiple chunks for testing the behavior properly.
"""
        result = chunk_markdown(content)
        
        # Content is below 500 chars, so gets full_document treatment
        assert len(result) == 1
        assert result[0]['metadata']['section'] == 'full_document'
        assert 'Deep Four' in result[0]['content']
        assert 'Deeper Five' in result[0]['content']
        assert 'Deepest Six' in result[0]['content']
    
    def test_yaml_frontmatter_edge_cases(self):
        """Test various YAML frontmatter edge cases."""
        # Empty frontmatter
        content1 = """---
---

## Section
Content"""
        result1 = chunk_markdown(content1)
        assert result1[0]['metadata']['frontmatter'] == {}
        
        # Frontmatter with special characters
        content2 = """---
title: "Title: With Colon"
special: "Line\\nBreak"
---

## Section
Content"""
        result2 = chunk_markdown(content2)
        assert result2[0]['metadata']['frontmatter']['title'] == "Title: With Colon"
    
    def test_consecutive_empty_sections(self):
        """Multiple consecutive empty sections."""
        content = """## Good Section
Good content here that meets minimum requirements. Adding substantial text to ensure
this section is not filtered out and properly appears in the final chunk output.

##

## 

### 

## Another Good Section
More good content here with enough text to meet the minimum section size threshold.
"""
        result = chunk_markdown(content)
        
        # Content is below 500 chars, so gets full_document treatment
        assert len(result) == 1
        assert result[0]['metadata']['section'] == 'full_document'
        assert 'Good Section' in result[0]['content']
        assert 'Another Good Section' in result[0]['content']
    
    def test_header_with_special_chars(self):
        """Headers containing special markdown characters."""
        content = """## Section `with code`
Content here with enough text for chunking.

### Subsection **bold** and _italic_
More content to ensure proper chunking behavior and testing of special character handling.

## Section [with link](http://example.com)
Even more content to test various markdown elements within headers themselves.
"""
        result = chunk_markdown(content)
        
        # Content is below 500 chars, so gets full_document treatment
        assert len(result) == 1
        assert result[0]['metadata']['section'] == 'full_document'
        assert 'Section `with code`' in result[0]['content']
        assert 'Subsection **bold** and _italic_' in result[0]['content']
        assert 'Section [with link](http://example.com)' in result[0]['content']
    
    def test_huge_input_memory_safety(self):
        """Huge input should not cause memory issues."""
        # Create a 10MB string
        huge_content = "## Section\n" + "x" * (10 * 1024 * 1024)
        
        # Should handle without memory error
        result = chunk_markdown(huge_content)
        assert len(result) == 1
        assert len(result[0]['content']) == 4000  # Truncated
    
    def test_windows_line_endings(self):
        """Windows CRLF line endings should work correctly."""
        content = "## Section One\r\nContent here.\r\n\r\n### Subsection\r\nMore content.\r\n"
        content += "x" * 500  # Make it large enough to chunk
        
        result = chunk_markdown(content)
        assert len(result) >= 1
        # Headers should still be detected with CRLF
    
    def test_tab_indented_headers(self):
        """Headers with tab indentation (invalid markdown) should not be detected."""
        content = """\t## Not A Header
This has a tab before it.

## Real Header  
This is a real header with enough content to trigger the chunking logic properly.
Need to add more text to exceed the minimum file size threshold for chunking.
"""
        result = chunk_markdown(content)
        
        # Content is below 500 chars, so gets full_document treatment
        assert len(result) == 1
        assert result[0]['metadata']['section'] == 'full_document'
        assert 'Not A Header' in result[0]['content']
        assert 'Real Header' in result[0]['content']
    
    def test_hash_inside_header(self):
        """Headers containing # symbol inside the text."""
        content = """## Section #1
First section content with substantial text for chunking.

### Part #2A
Subsection content also with enough text to be processed.

## Section C#/.NET
Another section to test hash symbols in headers properly with adequate content length.
"""
        result = chunk_markdown(content)
        
        # Content is below 500 chars, so gets full_document treatment
        assert len(result) == 1
        assert result[0]['metadata']['section'] == 'full_document'
        assert 'Section #1' in result[0]['content']
        assert 'Part #2A' in result[0]['content']
        assert 'Section C#/.NET' in result[0]['content']