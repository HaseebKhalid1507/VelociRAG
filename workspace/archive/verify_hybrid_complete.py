#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/haseeb/velocirag/src')

from velocirag.semantic_chunker import hybrid_chunk_markdown
from unittest.mock import Mock
import numpy as np

def test_complete_hybrid_workflow():
    """Complete test of hybrid chunking workflow."""
    
    print("🔄 Testing complete hybrid chunking workflow...\n")
    
    # Test content with mixed section sizes and topics
    content = """---
title: Hybrid Test Document
tags: [python, cooking, ai]
category: tutorial
status: active
---

# Complete Hybrid Test

## Small Section One
This section is small and should not be split semantically.

## Small Section Two  
Another small section that should pass through unchanged.

## Large Section About Programming and AI

Programming languages have evolved significantly over the past decades. Python has become one of the most popular languages due to its simplicity and versatility. It's widely used in web development, data science, machine learning, and automation. The language's clean syntax makes it accessible to beginners while still being powerful enough for complex applications. Many Fortune 500 companies rely on Python for their critical systems.

Now let's discuss artificial intelligence and machine learning concepts which are quite different from basic programming. Machine learning algorithms require large datasets and computational power to train effective models. Neural networks have revolutionized computer vision, natural language processing, and predictive analytics. Deep learning architectures like transformers have enabled breakthrough applications in AI. The field continues to advance rapidly with new research published daily.

Finally, let's cover cooking and culinary arts which is completely unrelated to technology. Professional cooking requires understanding heat transfer, chemical reactions, and flavor combinations. Different cuisines around the world have developed unique techniques over centuries. French cuisine emphasizes technique and precision. Italian cooking focuses on high-quality ingredients. Asian cuisines bring complex flavor profiles through spice combinations and fermentation processes.

## Another Small Section

This is small and should not trigger semantic splitting.
"""
    
    # Mock embedder with realistic topic transitions
    mock_embedder = Mock()
    
    # Simulate embeddings for different topics:
    # Programming (high similarity), AI (medium similarity), Cooking (low similarity)
    embeddings = np.array([
        # Programming topic - high internal similarity
        [0.9, 0.1, 0.0],  # Python sentence
        [0.85, 0.15, 0.0],  # Versatility sentence
        [0.8, 0.2, 0.0],   # Web dev sentence  
        [0.9, 0.1, 0.0],   # Clean syntax sentence
        [0.85, 0.15, 0.0], # Fortune 500 sentence
        
        # AI topic - different from programming
        [0.3, 0.7, 0.0],   # ML algorithms sentence
        [0.2, 0.8, 0.0],   # Neural networks sentence
        [0.25, 0.75, 0.0], # Deep learning sentence
        [0.3, 0.7, 0.0],   # Transformers sentence
        [0.2, 0.8, 0.0],   # Field advancement sentence
        
        # Cooking topic - completely different
        [0.0, 0.1, 0.9],   # Professional cooking sentence
        [0.0, 0.05, 0.95], # Cuisines sentence
        [0.0, 0.1, 0.9],   # French cuisine sentence
        [0.0, 0.05, 0.95], # Italian cooking sentence
        [0.0, 0.1, 0.9],   # Asian cuisines sentence
    ])
    
    mock_embedder.embed.return_value = embeddings
    
    # Test hybrid chunking
    chunks = hybrid_chunk_markdown(
        content, 
        "test_hybrid.md", 
        mock_embedder,
        semantic_threshold=30.0,  # More aggressive splitting
        large_section_threshold=400,  # Lower threshold for testing
        min_chunk_size=50,
        max_chunk_size=2000
    )
    
    print(f"📊 Results: {len(chunks)} total chunks generated\n")
    
    # Analyze results
    hybrid_chunks = [c for c in chunks if c['metadata'].get('hybrid_chunk')]
    header_chunks = [c for c in chunks if not c['metadata'].get('hybrid_chunk')]
    
    print(f"📁 Header-based chunks: {len(header_chunks)}")
    for chunk in header_chunks:
        print(f"  • {chunk['metadata']['section']}")
    
    print(f"\n🔀 Hybrid chunks: {len(hybrid_chunks)}")
    for chunk in hybrid_chunks:
        metadata = chunk['metadata']
        print(f"  • {metadata['section']} (from '{metadata['parent_section']}')")
        print(f"    Part {metadata['part_number']}/{metadata['total_parts']}")
    
    # Verify expectations
    print("\n✅ Verification:")
    
    # Should have small sections unchanged
    small_sections = [c for c in chunks if c['metadata']['section'] in ['Small Section One', 'Small Section Two', 'Another Small Section']]
    print(f"  Small sections preserved: {len(small_sections)}/3 ✓" if len(small_sections) == 3 else f"  Small sections preserved: {len(small_sections)}/3 ❌")
    
    # Should have large section split
    large_section_parts = [c for c in chunks if 'Large Section About Programming and AI' in c['metadata']['section']]
    print(f"  Large section split: {len(large_section_parts)} parts ✓" if len(large_section_parts) > 1 else f"  Large section not split ❌")
    
    # Check CCH headers
    all_have_cch = all('[Document: Hybrid Test Document]' in c['content'] for c in chunks)
    print(f"  All chunks have CCH headers: {'✓' if all_have_cch else '❌'}")
    
    # Check metadata inheritance
    all_hybrid_have_metadata = all(
        c['metadata']['frontmatter']['title'] == 'Hybrid Test Document' and
        c['metadata']['frontmatter']['category'] == 'tutorial' and
        'part_number' in c['metadata'] and
        'parent_section' in c['metadata']
        for c in hybrid_chunks
    )
    print(f"  Hybrid chunks have complete metadata: {'✓' if all_hybrid_have_metadata else '❌'}")
    
    # Show content preview of hybrid chunks
    print(f"\n📄 Hybrid chunk content previews:")
    for i, chunk in enumerate(hybrid_chunks):
        # Extract body after CCH header
        content_lines = chunk['content'].split('\n')
        body_start = next((i for i, line in enumerate(content_lines) if line == '---'), 0) + 1
        body = '\n'.join(content_lines[body_start:]).strip()
        preview = body[:100].replace('\n', ' ')
        print(f"  Part {chunk['metadata']['part_number']}: {preview}...")
    
    print(f"\n🎉 Hybrid chunking test completed!")
    
    return len(chunks) > 3 and len(hybrid_chunks) > 0  # Success criteria

if __name__ == "__main__":
    success = test_complete_hybrid_workflow()
    print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}: Hybrid chunking working as expected!")