#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/haseeb/velocirag/src')

from velocirag.chunker import chunk_markdown
from velocirag.semantic_chunker import hybrid_chunk_markdown
from unittest.mock import Mock

content = """---
title: Test Document
---

# Main Title

## Small Section

This is a small section with only a few sentences. It should pass through unchanged because it's under the large_section_threshold.

## Another Small Section  

This is also small and should not be semantically split."""

print("=== Testing header-based chunking ===")
header_chunks = chunk_markdown(content, "test.md")
print(f"Header chunks: {len(header_chunks)}")
for i, chunk in enumerate(header_chunks):
    print(f"Chunk {i+1}:")
    print(f"  Section: {chunk['metadata']['section']}")
    print(f"  Content length: {len(chunk['content'])}")
    print(f"  Content preview: {chunk['content'][:100]}...")
    print()

print("=== Testing hybrid chunking ===")
mock_embedder = Mock()
chunks = hybrid_chunk_markdown(content, "test.md", mock_embedder, large_section_threshold=2000)
print(f"Hybrid chunks: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:")
    print(f"  Section: {chunk['metadata']['section']}")  
    print(f"  Content length: {len(chunk['content'])}")
    print(f"  Hybrid: {chunk['metadata'].get('hybrid_chunk', False)}")
    print()