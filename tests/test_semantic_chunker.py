"""
Tests for semantic chunking functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from velocirag.semantic_chunker import (
    split_sentences, 
    calculate_boundary_scores,
    find_semantic_boundaries,
    semantic_chunk_markdown,
    hybrid_chunk_markdown,
    _merge_small_chunks,
    _split_large_chunks
)


def test_sentence_splitter():
    """Sentence splitter handles edge cases (abbreviations, URLs, numbers)."""
    
    # Basic sentences
    text = "This is sentence one. This is sentence two! Is this a question? Yes it is."
    sentences = split_sentences(text)
    assert len(sentences) == 4
    assert sentences[0] == "This is sentence one."
    assert sentences[1] == "This is sentence two!"
    assert sentences[2] == "Is this a question?"
    assert sentences[3] == "Yes it is."
    
    # Abbreviations should not split
    text = "Dr. Smith went to St. Mary's hospital. He met Mr. Johnson there."
    sentences = split_sentences(text)
    assert len(sentences) == 2
    assert "Dr. Smith" in sentences[0]
    assert "Mr. Johnson" in sentences[1]
    
    # Numbers should not split
    text = "The value is 3.14159. This is the next sentence."
    sentences = split_sentences(text)
    assert len(sentences) == 2
    assert "3.14159" in sentences[0]
    
    # Paragraph breaks
    text = "First paragraph.\n\nSecond paragraph with more content."
    sentences = split_sentences(text)
    assert len(sentences) == 2
    
    # Edge cases
    text = ""
    sentences = split_sentences(text)
    assert len(sentences) == 0
    
    # Very short fragments should be filtered
    text = "A. B. This is a proper sentence."
    sentences = split_sentences(text)
    assert len(sentences) == 1  # Only the proper sentence


def test_calculate_boundary_scores():
    """Test similarity calculation between sentences."""
    mock_embedder = Mock()
    
    # Mock embeddings: first two similar, third different
    embeddings = np.array([
        [1.0, 0.0, 0.0],  # sentence 0
        [0.9, 0.1, 0.0],  # sentence 1 (similar to 0)
        [0.0, 0.0, 1.0],  # sentence 2 (different)
    ])
    mock_embedder.embed.return_value = embeddings
    
    sentences = ["First sentence", "Similar sentence", "Different topic entirely"]
    similarities = calculate_boundary_scores(sentences, mock_embedder)
    
    assert len(similarities) == 2
    # First similarity should be high (similar sentences)
    assert similarities[0] > 0.8
    # Second similarity should be low (different topics)
    assert similarities[1] < 0.2


def test_find_semantic_boundaries():
    """Test boundary detection using percentile method."""
    
    # Similarities: high, high, low, high
    similarities = [0.9, 0.8, 0.2, 0.7]
    
    # With 25th percentile threshold, should split at the low similarity
    boundaries = find_semantic_boundaries(similarities, method='percentile', threshold=25.0)
    
    # Should include start (0), boundary after low similarity (3), and end (5)
    expected = [0, 3, 5]  # sentences 0-2, 3-4
    assert boundaries == expected
    
    # Test stddev method
    boundaries = find_semantic_boundaries(similarities, method='stddev', threshold=1.0)
    assert 0 in boundaries
    assert len(similarities) + 1 in boundaries  # End boundary


def test_semantic_chunking_basic():
    """Multiple topics get split into separate chunks."""
    
    # Create content with distinct topics (make it long enough to exceed MIN_FILE_SIZE_FOR_CHUNKING)
    content = """# Document Title

This is about machine learning and artificial intelligence. We discuss neural networks and deep learning architectures in great detail. This paragraph focuses on ML concepts and how they apply to real-world problems. Machine learning has revolutionized many industries and continues to grow rapidly. Deep learning models are particularly powerful for computer vision and natural language processing tasks. The field of AI is expanding at an unprecedented rate.

This section covers cooking recipes and culinary arts. We talk about ingredients and preparation methods in depth. Cooking is very different from machine learning and requires different skills. Professional chefs spend years mastering their craft. The art of cooking combines creativity with technical precision. Different cuisines from around the world offer unique flavors and techniques. Understanding ingredients and their interactions is crucial for successful cooking."""
    
    # Mock embedder to return different embeddings for different topics
    mock_embedder = Mock()
    # Create more embeddings to match the longer content
    embeddings = np.array([
        [0.5, 0.5],  # Header
        [1.0, 0.0],  # ML sentence 1
        [0.9, 0.1],  # ML sentence 2  
        [0.8, 0.2],  # ML sentence 3
        [0.85, 0.15],  # ML sentence 4
        [0.9, 0.1],  # ML sentence 5
        [0.8, 0.2],  # ML sentence 6
        [0.1, 0.9],  # Cooking sentence 1
        [0.0, 1.0],  # Cooking sentence 2
        [0.2, 0.8],  # Cooking sentence 3
        [0.1, 0.9],  # Cooking sentence 4
        [0.15, 0.85],  # Cooking sentence 5
        [0.05, 0.95],  # Cooking sentence 6
    ])
    mock_embedder.embed.return_value = embeddings
    
    chunks = semantic_chunk_markdown(content, "test.md", mock_embedder)
    
    # Should create multiple chunks
    assert len(chunks) >= 2
    
    # Each chunk should have required metadata
    for chunk in chunks:
        assert 'content' in chunk
        assert 'metadata' in chunk
        assert chunk['metadata']['semantic_chunk'] == True
        assert chunk['metadata']['has_context_header'] == True


def test_semantic_chunking_single_topic():
    """Coherent text about one topic stays as one chunk."""
    
    content = """# Machine Learning

This document is all about machine learning. We discuss neural networks in depth. 
Deep learning is a subset of machine learning. All sentences are related to AI and ML concepts."""
    
    # Mock embedder to return similar embeddings (high similarity)
    mock_embedder = Mock()
    embeddings = np.array([
        [1.0, 0.0],  # All sentences very similar
        [0.9, 0.1],
        [0.95, 0.05],
        [0.92, 0.08],
    ])
    mock_embedder.embed.return_value = embeddings
    
    chunks = semantic_chunk_markdown(content, "test.md", mock_embedder)
    
    # Should create only one chunk since all content is similar
    assert len(chunks) == 1
    assert "machine learning" in chunks[0]['content'].lower()


def test_semantic_chunking_fallback():
    """Short documents fall back to header-based chunking."""
    
    # Very short content
    content = """# Short Doc

Just one sentence."""
    
    mock_embedder = Mock()
    
    chunks = semantic_chunk_markdown(content, "test.md", mock_embedder)
    
    # Should fall back to regular chunking
    assert len(chunks) == 1
    # Should not have semantic_chunk flag
    assert chunks[0]['metadata'].get('semantic_chunk') != True


def test_semantic_chunking_has_cch_headers():
    """All chunks get CCH context headers."""
    
    content = """---
title: Test Document
tags: [test, example]
---

# Main Title

First section content. This talks about topic A.

Second section content. This talks about topic B which is different."""
    
    # Mock embedder
    mock_embedder = Mock()
    embeddings = np.array([
        [1.0, 0.0],  # Topic A
        [0.0, 1.0],  # Topic B 
    ])
    mock_embedder.embed.return_value = embeddings
    
    chunks = semantic_chunk_markdown(content, "test.md", mock_embedder)
    
    for chunk in chunks:
        content_text = chunk['content']
        # Should have CCH header
        assert '[Document: Test Document]' in content_text
        assert '[Source: test.md]' in content_text
        assert '[Tags: test, example]' in content_text
        assert '---' in content_text
        
        # Should have context header flag
        assert chunk['metadata']['has_context_header'] == True


def test_semantic_chunking_min_max_size():
    """Tiny chunks merge, huge chunks split."""
    
    # Test merging small chunks
    small_chunks = ["A", "B", "C"]  # All tiny
    merged = _merge_small_chunks(small_chunks, min_chunk_size=5)
    assert len(merged) < len(small_chunks)
    
    # Test splitting large chunks  
    sentences = [f"Sentence {i} with some content." for i in range(20)]
    large_chunk = " ".join(sentences)
    large_chunks = [large_chunk]
    
    split_chunks = _split_large_chunks(large_chunks, max_chunk_size=100, all_sentences=sentences)
    assert len(split_chunks) > 1
    
    # All chunks should be under max size
    for chunk in split_chunks:
        assert len(chunk) <= 100 or "..." in chunk  # Allow truncation marker


def test_semantic_chunking_output_format():
    """Output matches chunk_markdown() dict schema exactly."""
    
    content = """---
title: Test Document  
category: example
---

# Test

This is test content. It has multiple sentences. Each sentence talks about testing."""
    
    mock_embedder = Mock()
    embeddings = np.array([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]])
    mock_embedder.embed.return_value = embeddings
    
    chunks = semantic_chunk_markdown(content, "test.md", mock_embedder)
    
    # Should have at least one chunk
    assert len(chunks) >= 1
    
    # Check schema compliance
    for chunk in chunks:
        # Required top-level keys
        assert 'content' in chunk
        assert 'metadata' in chunk
        
        # Required metadata keys (same as chunk_markdown)
        metadata = chunk['metadata']
        required_keys = ['file_path', 'section', 'parent_header', 'frontmatter', 'content_hash', 'has_context_header']
        for key in required_keys:
            assert key in metadata, f"Missing metadata key: {key}"
        
        # Types should match
        assert isinstance(metadata['file_path'], str)
        assert isinstance(metadata['section'], str)
        assert metadata['parent_header'] is None or isinstance(metadata['parent_header'], str)
        assert isinstance(metadata['frontmatter'], dict)
        assert isinstance(metadata['content_hash'], str)
        assert isinstance(metadata['has_context_header'], bool)
        
        # Content should be string
        assert isinstance(chunk['content'], str)
        
        # Should have frontmatter data
        assert metadata['frontmatter']['title'] == 'Test Document'
        assert metadata['frontmatter']['category'] == 'example'


def test_semantic_chunking_embedder_failure():
    """Falls back to header-based chunking when embedder fails."""
    
    content = """# Test Document

This is content that should be chunked. We have multiple sentences here.
This talks about one topic. This sentence discusses another topic entirely."""
    
    # Mock embedder that raises exception
    mock_embedder = Mock()
    mock_embedder.embed.side_effect = Exception("Embedder failed")
    
    chunks = semantic_chunk_markdown(content, "test.md", mock_embedder)
    
    # Should still return chunks (fallback)
    assert len(chunks) >= 1
    
    # Should not have semantic_chunk flag (indicating fallback was used)
    for chunk in chunks:
        assert chunk['metadata'].get('semantic_chunk') != True


# Hybrid chunking tests

def test_hybrid_small_sections_unchanged():
    """Sections under threshold pass through without semantic splitting."""
    
    content = """---
title: Test Document
---

# Main Title

## Small Section

This is a small section with only a few sentences. It should pass through unchanged because it's under the large_section_threshold. We need to make this content longer so it exceeds the minimum file size for chunking which is 500 characters. This additional text ensures the document is large enough to be properly chunked by header-based splitting instead of being treated as a single full document.

## Another Small Section  

This is also small and should not be semantically split. Again we add more content here to make sure the overall document size is sufficient for header-based chunking to activate. The chunker needs enough content to recognize that this should be split into multiple sections rather than kept as one large chunk. Both sections should remain unchanged."""
    
    mock_embedder = Mock()
    
    # Set threshold high so these sections are considered small
    chunks = hybrid_chunk_markdown(content, "test.md", mock_embedder, 
                                 large_section_threshold=2000)
    
    # Should have original header-based chunks
    assert len(chunks) == 2  # Two ## sections
    
    # None should be marked as hybrid chunks
    for chunk in chunks:
        assert chunk['metadata'].get('hybrid_chunk') != True
        # Should not have part numbers
        assert 'part_number' not in chunk['metadata']


def test_hybrid_large_sections_split():
    """Large sections with topic changes get sub-split."""
    
    # Create content with one large section containing multiple topics
    content = """---
title: Test Document
---

# Main Title

## Large Section About Multiple Topics

This section discusses machine learning extensively. Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models. Deep learning networks have revolutionized computer vision and natural language processing. Neural networks consist of interconnected nodes that process information in layers. Training these models requires large datasets and significant computational resources. The field of AI continues to advance rapidly with new architectures and techniques being developed constantly.

Now let's completely shift topics to cooking and culinary arts. Cooking is fundamentally different from machine learning and requires entirely different skills. Professional chefs must master various techniques including knife skills, temperature control, and flavor balancing. Understanding ingredients and their chemical interactions is crucial for creating exceptional dishes. Different cuisines from around the world offer unique approaches to combining flavors and textures. The art of cooking combines creativity with precise technical execution in the kitchen environment.

This final paragraph returns to technology topics but focuses on databases and data storage. Relational databases use structured query language for data management. NoSQL databases offer flexibility for unstructured data storage solutions. Database optimization involves indexing strategies and query performance tuning. Modern applications often require distributed database architectures for scalability purposes."""
    
    mock_embedder = Mock()
    
    # Mock embeddings to show topic changes: ML, cooking, databases
    embeddings = np.array([
        [1.0, 0.0, 0.0],  # ML sentences (similar to each other)
        [0.9, 0.1, 0.0],
        [0.8, 0.2, 0.0],
        [0.85, 0.15, 0.0],
        [0.9, 0.1, 0.0],
        [0.8, 0.2, 0.0],
        [0.0, 1.0, 0.0],  # Cooking sentences (different topic)
        [0.1, 0.9, 0.0],
        [0.0, 0.8, 0.2],
        [0.1, 0.9, 0.0],
        [0.05, 0.95, 0.0],
        [0.0, 0.1, 0.9],  # Database sentences (third topic)
        [0.1, 0.0, 1.0],
        [0.0, 0.05, 0.95],
        [0.2, 0.0, 0.8],
    ])
    mock_embedder.embed.return_value = embeddings
    
    # Set threshold low so the section is considered large
    chunks = hybrid_chunk_markdown(content, "test.md", mock_embedder, 
                                 large_section_threshold=500)
    
    # Should create multiple sub-chunks from the large section
    assert len(chunks) > 1
    
    # At least some should be marked as hybrid chunks
    hybrid_chunks = [c for c in chunks if c['metadata'].get('hybrid_chunk')]
    assert len(hybrid_chunks) > 0
    
    # Check hybrid chunk metadata
    for chunk in hybrid_chunks:
        assert chunk['metadata']['hybrid_chunk'] == True
        assert 'parent_section' in chunk['metadata']
        assert 'part_number' in chunk['metadata']
        assert 'total_parts' in chunk['metadata']
        assert chunk['metadata']['parent_section'] == "Large Section About Multiple Topics"


def test_hybrid_preserves_cch_headers():
    """All sub-chunks get the same CCH header as parent."""
    
    content = """---
title: Hybrid Test
tags: [test, hybrid]
category: testing
---

# Document Title

## Large Mixed Section

First topic about programming languages and software development in general. Python is a versatile programming language that supports multiple paradigms. JavaScript runs in browsers and servers with excellent performance characteristics. Each language has unique syntax and features that make it suitable for different applications. Programming languages continue to evolve with new features and improvements.

Second topic about cooking recipes and culinary arts which is completely different. Pasta requires boiling water and salt for proper preparation techniques. Different sauces complement various pasta shapes in Italian cuisine. Italian cuisine emphasizes fresh ingredients and traditional preparation methods. Professional chefs spend years mastering these complex culinary techniques and flavor combinations."""
    
    mock_embedder = Mock()
    
    # Different topics should split
    embeddings = np.array([
        [1.0, 0.0],  # Programming sentences
        [0.9, 0.1],
        [0.8, 0.2],
        [0.0, 1.0],  # Cooking sentences  
        [0.1, 0.9],
        [0.0, 0.8],
    ])
    mock_embedder.embed.return_value = embeddings
    
    chunks = hybrid_chunk_markdown(content, "test.md", mock_embedder, 
                                 large_section_threshold=100)
    
    # Should have multiple chunks
    assert len(chunks) > 1
    
    # All chunks should have identical CCH headers
    for chunk in chunks:
        content_text = chunk['content']
        assert '[Document: Hybrid Test]' in content_text
        assert '[Source: test.md]' in content_text
        assert '[Tags: test, hybrid]' in content_text
        assert '[Category: testing]' in content_text
        assert '---' in content_text
        
        # Should preserve CCH flag
        assert chunk['metadata']['has_context_header'] == True


def test_hybrid_metadata_inheritance():
    """Sub-chunks inherit parent metadata with part numbering."""
    
    content = """---
title: Parent Doc
status: active
project: hybrid-test
---

# Main Title

## Section To Split

Topic one covers artificial intelligence and machine learning concepts in great detail. This includes neural networks, deep learning architectures, and various AI applications across industries. The field continues to evolve rapidly with new breakthroughs happening frequently.

Topic two discusses cooking techniques and culinary arts extensively. This covers knife skills, cooking methods, ingredient selection, and recipe development. Professional chefs require years of training to master these complex skills."""
    
    mock_embedder = Mock()
    
    # Two different topics
    embeddings = np.array([
        [1.0, 0.0],  # AI topic
        [0.9, 0.1],
        [0.8, 0.2],
        [0.0, 1.0],  # Cooking topic
        [0.1, 0.9],
        [0.0, 0.8],
    ])
    mock_embedder.embed.return_value = embeddings
    
    chunks = hybrid_chunk_markdown(content, "test.md", mock_embedder, 
                                 large_section_threshold=200)
    
    # Find the hybrid sub-chunks
    hybrid_chunks = [c for c in chunks if c['metadata'].get('hybrid_chunk')]
    assert len(hybrid_chunks) >= 2
    
    for chunk in hybrid_chunks:
        metadata = chunk['metadata']
        
        # Should inherit frontmatter
        assert metadata['frontmatter']['title'] == 'Parent Doc'
        assert metadata['frontmatter']['status'] == 'active'
        assert metadata['frontmatter']['project'] == 'hybrid-test'
        
        # Should have hybrid-specific metadata
        assert metadata['hybrid_chunk'] == True
        assert metadata['parent_section'] == 'Section To Split'
        assert 'part_number' in metadata
        assert 'total_parts' in metadata
        assert metadata['part_number'] >= 1
        assert metadata['total_parts'] >= 2
        
        # Section name should indicate part
        assert "(part" in metadata['section']
        assert metadata['section'].startswith('Section To Split (part')


def test_hybrid_no_embedder_fallback():
    """Without embedder, falls back to pure header-based chunking."""
    
    content = """# Test Document

## Large Section  

This is a very large section that would normally be considered for semantic splitting. It contains multiple sentences and topics. However, without an embedder, it should fall back to header-based chunking and remain as a single chunk."""
    
    # No embedder provided
    chunks = hybrid_chunk_markdown(content, "test.md", embedder=None)
    
    # Should get header-based chunks
    assert len(chunks) >= 1
    
    # Should not have hybrid chunk markers
    for chunk in chunks:
        assert chunk['metadata'].get('hybrid_chunk') != True


def test_hybrid_preserves_existing_chunks():
    """Small sections and single-topic sections pass through unchanged."""
    
    content = """# Test Document

## Small Section One
Short content here.

## Small Section Two  
Also short content.

## Large Single Topic Section
This section is large enough to trigger semantic analysis but discusses only one coherent topic throughout. Every sentence relates to machine learning and artificial intelligence concepts. The content maintains thematic consistency across all sentences. Neural networks and deep learning continue to be the main focus. All discussion points relate to the same domain of AI research and development."""
    
    mock_embedder = Mock()
    
    # All sentences about same topic (identical embeddings)
    embeddings = np.array([
        [1.0, 0.0],  # All sentences identical (perfect similarity)
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ])
    mock_embedder.embed.return_value = embeddings
    
    chunks = hybrid_chunk_markdown(content, "test.md", mock_embedder, 
                                 large_section_threshold=300)
    
    # Should have header-based chunks for all sections
    assert len(chunks) == 3  # Three ## sections
    
    # None should be marked as hybrid chunks (small sections + single topic)
    for chunk in chunks:
        assert chunk['metadata'].get('hybrid_chunk') != True


if __name__ == "__main__":
    pytest.main([__file__])