#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/haseeb/velocirag/src')

from velocirag.chunker import chunk_markdown
from velocirag.semantic_chunker import hybrid_chunk_markdown
from unittest.mock import Mock
import numpy as np

# Test 1: CCH headers test
print("=== Test 1: CCH Headers Test ===")
content1 = """---
title: Hybrid Test
tags: [test, hybrid]
category: testing
---

# Document Title

## Large Mixed Section

First topic about programming languages. Python is a versatile programming language. JavaScript runs in browsers and servers. Each language has unique syntax and features.

Second topic about cooking recipes. Pasta requires boiling water and salt. Different sauces complement various pasta shapes. Italian cuisine emphasizes fresh ingredients."""

mock_embedder = Mock()
embeddings = np.array([
    [1.0, 0.0],  # Programming sentences
    [0.9, 0.1],
    [0.8, 0.2],
    [0.0, 1.0],  # Cooking sentences  
    [0.1, 0.9],
    [0.0, 0.8],
])
mock_embedder.embed.return_value = embeddings

print("Header-based chunking:")
header_chunks1 = chunk_markdown(content1, "test.md")
print(f"Chunks: {len(header_chunks1)}")
for i, chunk in enumerate(header_chunks1):
    print(f"  Chunk {i+1}: section='{chunk['metadata']['section']}', length={len(chunk['content'])}")

print(f"\nBody length check:")
for chunk in header_chunks1:
    content_text = chunk['content']
    if "---" in content_text:
        parts = content_text.split("---", 1)
        if len(parts) == 2:
            body = parts[1].strip()
            print(f"  Body length: {len(body)} chars")
        else:
            print(f"  No proper CCH split")
    else:
        print(f"  No CCH separator found")

print("\nHybrid chunking:")
chunks1 = hybrid_chunk_markdown(content1, "test.md", mock_embedder, large_section_threshold=100)
print(f"Chunks: {len(chunks1)}")
for i, chunk in enumerate(chunks1):
    print(f"  Chunk {i+1}: section='{chunk['metadata']['section']}', hybrid={chunk['metadata'].get('hybrid_chunk', False)}")

print("\n" + "="*60)

# Test 2: Preserving chunks test  
print("=== Test 2: Preserving Chunks Test ===")
content2 = """# Test Document

## Small Section One
Short content here.

## Small Section Two  
Also short content.

## Large Single Topic Section
This section is large enough to trigger semantic analysis but discusses only one coherent topic throughout. Every sentence relates to machine learning and artificial intelligence concepts. The content maintains thematic consistency across all sentences. Neural networks and deep learning continue to be the main focus. All discussion points relate to the same domain of AI research and development."""

print("Header-based chunking:")
header_chunks2 = chunk_markdown(content2, "test.md") 
print(f"Chunks: {len(header_chunks2)}")
for i, chunk in enumerate(header_chunks2):
    print(f"  Chunk {i+1}: section='{chunk['metadata']['section']}', length={len(chunk['content'])}")

print("\nHybrid chunking:")
mock_embedder2 = Mock()
embeddings2 = np.array([
    [1.0, 0.0],  # All sentences very similar
    [0.95, 0.05],
    [0.9, 0.1],
    [0.92, 0.08],
    [0.88, 0.12],
    [0.91, 0.09],
])
mock_embedder2.embed.return_value = embeddings2

chunks2 = hybrid_chunk_markdown(content2, "test.md", mock_embedder2, large_section_threshold=300)
print(f"Chunks: {len(chunks2)}")
for i, chunk in enumerate(chunks2):
    print(f"  Chunk {i+1}: section='{chunk['metadata']['section']}', hybrid={chunk['metadata'].get('hybrid_chunk', False)}, length={len(chunk['content'])}")