"""Tests for Contextual Chunk Headers (CCH) functionality."""

import pytest
from velocirag.chunker import chunk_markdown, build_context_header


def test_context_header_with_full_metadata():
    """Chunk with title, tags, category gets full header."""
    content = """---
title: API Security Guide
tags: [security, api, authentication]
category: tutorial
---

## Authentication
### JWT Tokens

JWT tokens are signed with HMAC-SHA256...
"""
    
    chunks = chunk_markdown(content, "Notes/3. Learning/api-security.md")
    assert len(chunks) == 1
    
    chunk = chunks[0]
    assert chunk['metadata']['has_context_header'] is True
    
    lines = chunk['content'].split('\n')
    assert '[Document: API Security Guide]' in lines
    assert '[Source: Notes/3. Learning/api-security.md]' in lines
    assert '[Tags: security, api, authentication]' in lines
    assert '[Category: tutorial]' in lines
    assert '---' in lines


def test_context_header_minimal():
    """Chunk with no frontmatter gets filename-based header."""
    content = """
## Basic Usage

This is some content.
"""
    
    chunks = chunk_markdown(content, "docs/readme.md")
    assert len(chunks) == 1
    
    chunk = chunks[0]
    assert chunk['metadata']['has_context_header'] is True
    
    lines = chunk['content'].split('\n')
    assert '[Document: readme]' in lines
    assert '[Source: docs/readme.md]' in lines
    assert not any('[Tags:' in line for line in lines)
    assert not any('[Category:' in line for line in lines)


def test_context_header_no_empty_lines():
    """Empty tags/category fields are omitted, not shown as empty."""
    content = """---
title: Test Doc
tags: []
category: ""
---

## Section

Content here.
"""
    
    chunks = chunk_markdown(content, "test.md")
    chunk = chunks[0]
    
    lines = chunk['content'].split('\n')
    assert '[Document: Test Doc]' in lines
    assert not any('[Tags:' in line for line in lines)
    assert not any('[Category:' in line for line in lines)


def test_context_header_with_h1_title():
    """Uses H1 title when no frontmatter title."""
    content = """# My Great Document

## Section One

Some content here.
"""
    
    chunks = chunk_markdown(content, "document.md")
    chunk = chunks[0]
    
    lines = chunk['content'].split('\n')
    assert '[Document: My Great Document]' in lines


def test_small_file_gets_header():
    """Files under MIN_FILE_SIZE_FOR_CHUNKING still get context header."""
    content = """---
title: Short Note
tags: [quick]
---

Brief content."""
    
    chunks = chunk_markdown(content, "short.md")
    assert len(chunks) == 1
    
    chunk = chunks[0]
    assert chunk['metadata']['has_context_header'] is True
    assert chunk['metadata']['section'] == 'full_document'
    
    lines = chunk['content'].split('\n')
    assert '[Document: Short Note]' in lines
    assert '[Tags: quick]' in lines


def test_max_chunk_size_excludes_header():
    """MAX_CHUNK_SIZE applies to body content, not header."""
    # Create content that would exceed MAX_CHUNK_SIZE with header
    long_content = "x" * 4500  # Exceeds MAX_CHUNK_SIZE (4000)
    
    content = f"""---
title: Long Document
tags: [test]
---

## Long Section

{long_content}
"""
    
    chunks = chunk_markdown(content, "long.md")
    chunk = chunks[0]
    
    # Header should still be present
    lines = chunk['content'].split('\n')
    assert '[Document: Long Document]' in lines
    assert '[Tags: test]' in lines
    assert '---' in lines
    
    # Content should be truncated (ends with ...)
    assert chunk['content'].endswith('...')


def test_backward_compatible_metadata():
    """Existing metadata fields preserved, has_context_header added."""
    content = """---
title: Test
category: note
---

## Section

Content.
"""
    
    chunks = chunk_markdown(content, "test.md")
    chunk = chunks[0]
    
    metadata = chunk['metadata']
    # Original fields still present
    assert 'file_path' in metadata
    assert 'section' in metadata
    assert 'parent_header' in metadata
    assert 'frontmatter' in metadata
    assert 'content_hash' in metadata
    
    # New field added
    assert metadata['has_context_header'] is True


def test_build_context_header_function():
    """Test build_context_header function directly."""
    # Full metadata
    frontmatter = {
        'title': 'Test Document',
        'tags': ['tag1', 'tag2'],
        'category': 'guide'
    }
    
    header = build_context_header("path/file.md", frontmatter)
    lines = header.split('\n')
    
    assert '[Document: Test Document]' in lines
    assert '[Source: path/file.md]' in lines
    assert '[Tags: tag1, tag2]' in lines
    assert '[Category: guide]' in lines
    assert '---' in lines
    
    # Minimal metadata
    header = build_context_header("simple.md", {})
    lines = header.split('\n')
    
    assert '[Document: simple]' in lines
    assert '[Source: simple.md]' in lines
    assert not any('[Tags:' in line for line in lines)
    assert not any('[Category:' in line for line in lines)
    assert '---' in lines


def test_tags_as_string():
    """Handle tags field as string rather than list."""
    content = """---
title: String Tags
tags: "single-tag"
---

## Section

Content.
"""
    
    chunks = chunk_markdown(content, "test.md")
    chunk = chunks[0]
    
    lines = chunk['content'].split('\n')
    assert '[Tags: single-tag]' in lines


def test_status_as_category():
    """Status field used as category when category absent."""
    content = """---
title: Status Test
status: active
---

## Section

Content.
"""
    
    chunks = chunk_markdown(content, "test.md")
    chunk = chunks[0]
    
    lines = chunk['content'].split('\n')
    assert '[Category: active]' in lines