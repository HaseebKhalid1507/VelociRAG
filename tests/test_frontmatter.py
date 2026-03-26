"""Tests for frontmatter.py module."""

import pytest
from datetime import datetime, date
from velocirag.frontmatter import (
    parse_frontmatter,
    extract_tags_from_content,
    extract_wiki_links,
    _normalize_frontmatter_values
)


class TestParseFrontmatter:
    """Test YAML frontmatter parsing."""
    
    def test_basic_frontmatter(self):
        """Basic frontmatter parsing works."""
        content = """---
title: Test Document
category: notes
tags:
  - python
  - programming
---

# Main Content

This is the body of the document.
"""
        
        frontmatter, body = parse_frontmatter(content)
        
        assert frontmatter['title'] == 'Test Document'
        assert frontmatter['category'] == 'notes'
        assert frontmatter['tags'] == ['python', 'programming']
        assert body.strip().startswith('# Main Content')
        assert 'This is the body' in body
    
    def test_no_frontmatter(self):
        """Content without frontmatter returns empty dict."""
        content = """# Just a regular markdown file

No frontmatter here.
"""
        
        frontmatter, body = parse_frontmatter(content)
        
        assert frontmatter == {}
        assert body == content
    
    def test_empty_frontmatter(self):
        """Empty frontmatter section."""
        content = """---
---

# Content after empty frontmatter
"""
        
        frontmatter, body = parse_frontmatter(content)
        
        assert frontmatter == {}
        assert body.strip().startswith('# Content')
    
    def test_malformed_yaml(self):
        """Malformed YAML returns empty dict without crashing."""
        content = """---
title: Test
invalid yaml: [unclosed
another: value
---

# Content
"""
        
        frontmatter, body = parse_frontmatter(content)
        
        # Should gracefully handle malformed YAML
        assert frontmatter == {}
        assert body.strip().startswith('# Content')
    
    def test_frontmatter_with_dates(self):
        """Date objects in frontmatter are converted to strings."""
        content = """---
title: Test Document
created: 2024-01-15
updated: 2024-03-26T14:30:00
---

# Content
"""
        
        frontmatter, body = parse_frontmatter(content)
        
        assert frontmatter['title'] == 'Test Document'
        assert isinstance(frontmatter['created'], str)
        assert isinstance(frontmatter['updated'], str)
        assert 'T' in frontmatter['updated']  # ISO format
    
    def test_frontmatter_non_dict_yaml(self):
        """YAML that parses to non-dict returns empty dict."""
        content = """---
- just
- a
- list
---

# Content
"""
        
        frontmatter, body = parse_frontmatter(content)
        
        assert frontmatter == {}
        assert body.strip().startswith('# Content')
    
    def test_frontmatter_null_value(self):
        """Null YAML value returns empty dict."""
        content = """---
null
---

# Content
"""
        
        frontmatter, body = parse_frontmatter(content)
        
        assert frontmatter == {}
        assert body.strip().startswith('# Content')
    
    def test_frontmatter_complex_structure(self):
        """Complex nested frontmatter structures."""
        content = """---
title: Complex Document
metadata:
  author: John Doe
  version: 1.0
  tags: [python, tutorial]
config:
  draft: false
  published: true
---

# Content
"""
        
        frontmatter, body = parse_frontmatter(content)
        
        assert frontmatter['title'] == 'Complex Document'
        assert frontmatter['metadata']['author'] == 'John Doe'
        assert frontmatter['metadata']['tags'] == ['python', 'tutorial']
        assert frontmatter['config']['draft'] is False
    
    def test_empty_content(self):
        """Empty content returns empty dict and empty string."""
        frontmatter, body = parse_frontmatter("")
        assert frontmatter == {}
        assert body == ""
        
        frontmatter, body = parse_frontmatter(None)
        assert frontmatter == {}
        assert body is None
    
    def test_frontmatter_with_special_chars(self):
        """Frontmatter with special characters."""
        content = """---
title: "Document with: special chars"
description: |
  Multi-line description
  with special characters: !@#$%
url: "https://example.com/path?param=value"
---

# Content
"""
        
        frontmatter, body = parse_frontmatter(content)
        
        assert frontmatter['title'] == 'Document with: special chars'
        assert 'Multi-line description' in frontmatter['description']
        assert frontmatter['url'] == 'https://example.com/path?param=value'
    
    def test_frontmatter_whitespace_handling(self):
        """Frontmatter with various whitespace patterns."""
        content = """---   
title: Test
category: notes  
---


# Content with extra newlines
"""
        
        frontmatter, body = parse_frontmatter(content)
        
        assert frontmatter['title'] == 'Test'
        assert frontmatter['category'] == 'notes'
        assert body.strip().startswith('# Content')


class TestExtractTags:
    """Test hashtag extraction."""
    
    def test_basic_hashtags(self):
        """Basic hashtag extraction works."""
        content = "This is about #python and #programming concepts."
        
        tags = extract_tags_from_content(content)
        
        assert set(tags) == {'python', 'programming'}
    
    def test_hashtags_with_underscores_and_dashes(self):
        """Hashtags with underscores and dashes."""
        content = "Topics: #web-development #data_science #machine-learning"
        
        tags = extract_tags_from_content(content)
        
        assert set(tags) == {'web-development', 'data_science', 'machine-learning'}
    
    def test_hashtags_case_normalization(self):
        """Hashtags are normalized to lowercase."""
        content = "Tags: #Python #PROGRAMMING #JavaScript"
        
        tags = extract_tags_from_content(content)
        
        assert set(tags) == {'python', 'programming', 'javascript'}
    
    def test_hashtags_deduplication(self):
        """Duplicate hashtags are deduplicated."""
        content = "About #python and more #python stuff. Also #Python."
        
        tags = extract_tags_from_content(content)
        
        assert tags == ['python']  # Only one instance
    
    def test_hashtags_exclude_headers(self):
        """Markdown headers are not treated as tags."""
        content = """
# Main Header
## Sub Header
### Another Header

This is about #python programming.
"""
        
        tags = extract_tags_from_content(content)
        
        # Should not extract header markers, only the actual tag
        assert tags == ['python']
    
    def test_hashtags_exclude_numbers_only(self):
        """Hashtags that are only numbers are excluded."""
        content = "Issue #123 and #456 but also #python and #bug123"
        
        tags = extract_tags_from_content(content)
        
        # Should exclude pure numbers but include alphanumeric
        assert set(tags) == {'python', 'bug123'}
    
    def test_hashtags_multiline(self):
        """Hashtags across multiple lines."""
        content = """
First paragraph has #python.

Second paragraph mentions #javascript
and #html.

Final line: #css
"""
        
        tags = extract_tags_from_content(content)
        
        assert set(tags) == {'python', 'javascript', 'html', 'css'}
    
    def test_hashtags_empty_content(self):
        """Empty content returns empty list."""
        assert extract_tags_from_content("") == []
        assert extract_tags_from_content(None) == []
    
    def test_hashtags_no_matches(self):
        """Content without hashtags returns empty list."""
        content = "This is just regular text without any special tags."
        
        tags = extract_tags_from_content(content)
        
        assert tags == []
    
    def test_hashtags_with_punctuation(self):
        """Hashtags followed by punctuation."""
        content = "Topics: #python, #javascript; #html. Also #css!"
        
        tags = extract_tags_from_content(content)
        
        assert set(tags) == {'python', 'javascript', 'html', 'css'}


class TestExtractWikiLinks:
    """Test wiki link extraction."""
    
    def test_basic_wiki_links(self):
        """Basic wiki link extraction works."""
        content = "See [[Other Document]] and [[Another Page]]."
        
        links = extract_wiki_links(content)
        
        assert set(links) == {'Other Document', 'Another Page'}
    
    def test_wiki_links_with_display_text(self):
        """Wiki links with display text."""
        content = "Check out [[display text|target page]] for more info."
        
        links = extract_wiki_links(content)
        
        # Should extract the target, not the display text
        assert links == ['target page']
    
    def test_wiki_links_mixed_formats(self):
        """Mixed wiki link formats."""
        content = """
Reference [[Simple Link]] and [[Display|Target]].
Also see [[Another Simple Link]].
"""
        
        links = extract_wiki_links(content)
        
        assert set(links) == {'Simple Link', 'Target', 'Another Simple Link'}
    
    def test_wiki_links_deduplication(self):
        """Duplicate wiki links are deduplicated."""
        content = "See [[Page]] and [[Page]] again."
        
        links = extract_wiki_links(content)
        
        assert links == ['Page']  # Only one instance
    
    def test_wiki_links_whitespace_handling(self):
        """Wiki links with extra whitespace."""
        content = "Links: [[  Page With Spaces  ]] and [[Display  |  Target  ]]"
        
        links = extract_wiki_links(content)
        
        assert set(links) == {'Page With Spaces', 'Target'}
    
    def test_wiki_links_multiline(self):
        """Wiki links across multiple lines."""
        content = """
First paragraph has [[Link One]].

Second mentions [[Link Two]]
and [[Link Three]].
"""
        
        links = extract_wiki_links(content)
        
        assert set(links) == {'Link One', 'Link Two', 'Link Three'}
    
    def test_wiki_links_empty_content(self):
        """Empty content returns empty list."""
        assert extract_wiki_links("") == []
        assert extract_wiki_links(None) == []
    
    def test_wiki_links_no_matches(self):
        """Content without wiki links returns empty list."""
        content = "This is regular text with no special links."
        
        links = extract_wiki_links(content)
        
        assert links == []
    
    def test_wiki_links_nested_brackets(self):
        """Nested or malformed brackets."""
        content = "Bad: [[[malformed]] but good: [[proper link]]"
        
        links = extract_wiki_links(content)
        
        # Should only extract the properly formed wiki link
        assert links == ['proper link']
    
    def test_wiki_links_with_paths(self):
        """Wiki links that look like file paths."""
        content = "See [[docs/technical.md]] and [[images/diagram.png]]."
        
        links = extract_wiki_links(content)
        
        assert set(links) == {'docs/technical.md', 'images/diagram.png'}


class TestNormalizeFrontmatterValues:
    """Test frontmatter value normalization."""
    
    def test_datetime_normalization(self):
        """Datetime objects are converted to ISO strings."""
        data = {
            'created': datetime(2024, 1, 15, 14, 30, 45),
            'updated': date(2024, 3, 26)
        }
        
        normalized = _normalize_frontmatter_values(data)
        
        assert isinstance(normalized['created'], str)
        assert isinstance(normalized['updated'], str)
        assert 'T' in normalized['created']  # ISO datetime format
        assert 'T' not in normalized['updated']  # Date format
    
    def test_nested_dict_normalization(self):
        """Nested dictionaries are recursively normalized."""
        data = {
            'metadata': {
                'created': datetime(2024, 1, 15),
                'author': 'John Doe'
            },
            'title': 'Test Document'
        }
        
        normalized = _normalize_frontmatter_values(data)
        
        assert isinstance(normalized['metadata']['created'], str)
        assert normalized['metadata']['author'] == 'John Doe'
        assert normalized['title'] == 'Test Document'
    
    def test_list_normalization(self):
        """Lists with datetime objects are normalized."""
        data = {
            'dates': [
                datetime(2024, 1, 15),
                'regular string',
                date(2024, 3, 26)
            ]
        }
        
        normalized = _normalize_frontmatter_values(data)
        
        assert isinstance(normalized['dates'][0], str)
        assert normalized['dates'][1] == 'regular string'
        assert isinstance(normalized['dates'][2], str)
    
    def test_primitive_values_unchanged(self):
        """Primitive values are left unchanged."""
        data = {
            'string': 'text',
            'number': 42,
            'boolean': True,
            'null': None
        }
        
        normalized = _normalize_frontmatter_values(data)
        
        assert normalized['string'] == 'text'
        assert normalized['number'] == 42
        assert normalized['boolean'] is True
        assert normalized['null'] is None
    
    def test_non_dict_input(self):
        """Non-dict input is returned as-is."""
        assert _normalize_frontmatter_values("string") == "string"
        assert _normalize_frontmatter_values(42) == 42
        assert _normalize_frontmatter_values(None) is None
    
    def test_datetime_conversion_error(self):
        """Handles datetime conversion errors gracefully."""
        # Create mock object with isoformat method that raises
        class MockDateTime:
            def isoformat(self):
                raise ValueError("Mock error")
            
            def __str__(self):
                return "mock datetime"
        
        data = {'bad_date': MockDateTime()}
        
        normalized = _normalize_frontmatter_values(data)
        
        # Should fallback to string conversion
        assert normalized['bad_date'] == 'mock datetime'


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    def test_complete_markdown_processing(self):
        """Process a complete markdown file with all features."""
        content = """---
title: Python Tutorial
category: programming
tags:
  - python
  - tutorial
created: 2024-01-15
author: John Doe
---

# Python Programming Tutorial

This tutorial covers #python basics and #advanced-topics.

See also [[Advanced Python]] and [[Data Structures]].

## Introduction

Python is great for #web-development and #data-science.

References:
- [[Python Documentation]]
- [[Best Practices|coding-standards]]

Tags in content: #beginner #intermediate
"""
        
        # Parse frontmatter
        frontmatter, body = parse_frontmatter(content)
        
        # Extract content features
        content_tags = extract_tags_from_content(body)
        wiki_links = extract_wiki_links(body)
        
        # Verify frontmatter
        assert frontmatter['title'] == 'Python Tutorial'
        assert frontmatter['category'] == 'programming'
        assert frontmatter['tags'] == ['python', 'tutorial']
        assert frontmatter['author'] == 'John Doe'
        assert isinstance(frontmatter['created'], str)
        
        # Verify content tags
        expected_tags = {
            'python', 'advanced-topics', 'web-development', 
            'data-science', 'beginner', 'intermediate'
        }
        assert set(content_tags) == expected_tags
        
        # Verify wiki links
        expected_links = {
            'Advanced Python', 'Data Structures', 
            'Python Documentation', 'coding-standards'
        }
        assert set(wiki_links) == expected_links
    
    def test_edge_case_combinations(self):
        """Test combinations of edge cases."""
        content = """---
title: Edge Case Document
malformed: [unclosed
valid: true
---

# Document with [[Malformed Link

But also [[Good Link]] and #valid-tag.

## Headers are not hashtags

More content with #123numbers and #valid_tag.
"""
        
        frontmatter, body = parse_frontmatter(content)
        content_tags = extract_tags_from_content(body)
        wiki_links = extract_wiki_links(body)
        
        # Frontmatter should handle malformed YAML gracefully
        assert frontmatter == {}  # Malformed YAML returns empty dict
        
        # Should extract valid tags only
        assert set(content_tags) == {'valid-tag', 'valid_tag'}
        
        # Should extract valid wiki links only
        assert wiki_links == ['Good Link']