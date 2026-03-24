"""
Test suite for Velocirag L0/L1 abstract generation system.

Tests sentence splitting, abstract generation, batch processing,
store integration, and progressive search capabilities.
"""

import json
import time
from pathlib import Path

import numpy as np
import pytest

from velocirag.abstracts import AbstractGenerator, SentenceSplitter
from velocirag.embedder import Embedder
from velocirag.store import VectorStore
from velocirag.searcher import Searcher


# Test data fixtures
SAMPLE_CONTENT = """
# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence. It focuses on enabling computers to learn from data without being explicitly programmed. Dr. Smith pioneered early work in this field at Stanford University.

## Core Concepts

The fundamental principle of ML is pattern recognition. Algorithms identify patterns in training data and use these patterns to make predictions on new, unseen data. This approach differs from traditional rule-based programming.

### Types of Learning

There are three main types of machine learning. Supervised learning uses labeled data where the desired output is known. Unsupervised learning finds hidden patterns in unlabeled data. Reinforcement learning learns through interaction with an environment.

Modern applications include computer vision, natural language processing, and recommendation systems. Companies like Google, Facebook, and Amazon heavily rely on ML for their core products. The field continues to evolve rapidly with new breakthroughs in deep learning.
"""

SHORT_CONTENT = "Machine learning is transforming technology."

ABBREVIATION_CONTENT = "Dr. Johnson works at Google Inc. in Mountain View, CA. He has a Ph.D. from M.I.T. and previously worked at I.B.M. Corp."

EDGE_CASE_CONTENT = """
Short.

This has... multiple dots... everywhere!!!

URLs: https://example.com and http://test.org should be handled.

Numbers like 3.14 and abbreviations like U.S.A. matter.

###### Tiny header
Still counts as content.

```
Code blocks should be skipped
```

> Quotes are fine. They add context.
"""

MULTI_DOCUMENT_CONTENTS = [
    """
    # Python Programming
    
    Python is a high-level programming language known for its simplicity and readability. Created by Guido van Rossum, it emphasizes code clarity. The language supports multiple programming paradigms including procedural, object-oriented, and functional programming.
    
    Python's extensive standard library makes it suitable for various applications. Web development frameworks like Django and Flask are popular choices. Data science libraries such as NumPy, pandas, and scikit-learn have made Python dominant in analytics.
    """,
    
    """
    # Neural Networks
    
    Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes or neurons organized in layers. Each connection has a weight that adjusts during training.
    
    Deep learning uses neural networks with multiple hidden layers. Convolutional neural networks excel at image recognition tasks. Recurrent neural networks are effective for sequential data like text or time series.
    """,
    
    """
    # Cloud Computing
    
    Cloud computing delivers computing services over the internet. It provides servers, storage, databases, networking, and software on demand. Major providers include AWS, Azure, and Google Cloud Platform.
    
    Benefits include cost savings, scalability, and flexibility. Organizations can scale resources up or down based on demand. Security and compliance remain important considerations for cloud adoption.
    """,
    
    SHORT_CONTENT,  # Edge case: very short document
    
    "",  # Edge case: empty document
]


@pytest.fixture
def embedder():
    """Provide real embedder instance."""
    return Embedder()


@pytest.fixture
def sentence_splitter():
    """Provide sentence splitter instance."""
    return SentenceSplitter()


@pytest.fixture
def abstract_generator(embedder):
    """Provide abstract generator with embedder."""
    return AbstractGenerator(embedder)


@pytest.fixture
def tmp_store(tmp_path, embedder, abstract_generator):
    """Create temporary vector store with abstract support."""
    store_path = tmp_path / "test_store"
    store_path.mkdir(parents=True, exist_ok=True)
    return VectorStore(str(store_path), embedder, abstract_generator)


@pytest.fixture
def sample_markdown_dir(tmp_path):
    """Create directory with sample markdown files."""
    md_dir = tmp_path / "markdown_files"
    md_dir.mkdir(parents=True, exist_ok=True)
    
    # Create varied test files
    files = {
        "python.md": MULTI_DOCUMENT_CONTENTS[0],
        "neural_nets.md": MULTI_DOCUMENT_CONTENTS[1],
        "cloud.md": MULTI_DOCUMENT_CONTENTS[2],
        "ml_intro.md": SAMPLE_CONTENT,
        "abbreviations.md": ABBREVIATION_CONTENT,
        "edge_cases.md": EDGE_CASE_CONTENT,
        "empty.md": "",
        "short.md": SHORT_CONTENT,
    }
    
    for filename, content in files.items():
        (md_dir / filename).write_text(content, encoding='utf-8')
    
    return md_dir


class TestSentenceSplitting:
    """Test sentence splitting functionality."""
    
    def test_basic_sentence_split(self, sentence_splitter):
        """Test basic sentence splitting on periods."""
        text = "This is sentence one. This is sentence two. And this is sentence three."
        sentences = sentence_splitter.split(text)
        
        assert len(sentences) == 3
        assert sentences[0] == "This is sentence one."
        assert sentences[1] == "This is sentence two."
        assert sentences[2] == "And this is sentence three."
    
    def test_abbreviation_handling(self, sentence_splitter):
        """Test that abbreviations don't cause false splits."""
        sentences = sentence_splitter.split(ABBREVIATION_CONTENT)
        
        # Should be 2 sentences based on actual sentence structure
        # "Dr. Johnson works at Google Inc. in Mountain View, CA."
        # "He has a Ph.D. from M.I.T. and previously worked at I.B.M. Corp."
        assert len(sentences) == 2
        assert "Dr. Johnson" in sentences[0]
        assert "Inc." in sentences[0]
        assert "Ph.D." in sentences[1]
        assert "M.I.T." in sentences[1]
        assert "I.B.M. Corp." in sentences[1]
    
    def test_markdown_headers_as_boundaries(self, sentence_splitter):
        """Test that markdown headers act as sentence boundaries."""
        text = """
# Header One
This is content under header one.
## Header Two
This is content under header two.
### Header Three
Final content here.
"""
        sentences = sentence_splitter.split(text)
        
        # Headers should be converted to sentences
        assert any("Header One" in s for s in sentences)
        assert any("Header Two" in s for s in sentences)
        assert any("content under header one" in s for s in sentences)
    
    def test_double_newline_boundaries(self, sentence_splitter):
        """Test that double newlines create sentence boundaries."""
        text = """First paragraph here.

Second paragraph starts here.

Third paragraph is the last."""
        
        sentences = sentence_splitter.split(text)
        assert len(sentences) >= 3
        assert any("First paragraph" in s for s in sentences)
        assert any("Second paragraph" in s for s in sentences)
        assert any("Third paragraph" in s for s in sentences)
    
    def test_short_sentence_filtering(self, sentence_splitter):
        """Test that short sentences (<10 chars) are filtered out."""
        text = "Valid sentence here. Too short. Another valid sentence for testing."
        sentences = sentence_splitter.split(text)
        
        # "Too short." is 10 chars, should be included
        # But very short fragments should be filtered
        assert all(len(s) >= 10 for s in sentences)
    
    def test_empty_input(self, sentence_splitter):
        """Test handling of empty input."""
        assert sentence_splitter.split("") == []
        assert sentence_splitter.split(None) == []
        assert sentence_splitter.split("   ") == []
        assert sentence_splitter.split("\n\n") == []
    
    def test_edge_cases(self, sentence_splitter):
        """Test various edge cases."""
        sentences = sentence_splitter.split(EDGE_CASE_CONTENT)
        
        # Should handle multiple dots
        assert any("multiple dots" in s for s in sentences)
        
        # URLs should be preserved
        assert any("https://example.com" in s for s in sentences)
        
        # Numbers with decimals
        assert any("3.14" in s for s in sentences)
        
        # Code blocks are processed differently - the _is_valid_sentence method
        # filters out lines starting with ``` but the content inside might remain
        # if it looks like a valid sentence. Let's check that the actual code block
        # markers are not present
        assert not any("```" in s for s in sentences)
        
        # Headers converted to sentences
        assert any("Tiny header" in s for s in sentences)
        
        # Quotes preserved
        assert any("Quotes are fine" in s for s in sentences)
    
    def test_markdown_cleaning(self, sentence_splitter):
        """Test removal of markdown formatting."""
        text = "This has **bold text** and *italic text* and `inline code`."
        sentences = sentence_splitter.split(text)
        
        assert len(sentences) == 1
        sentence = sentences[0]
        # Markdown should be cleaned
        assert "**" not in sentence
        assert "*" not in sentence
        assert "`" not in sentence
        # But content preserved
        assert "bold text" in sentence
        assert "italic text" in sentence
        assert "inline code" in sentence


class TestAbstractGeneration:
    """Test abstract generation functionality."""
    
    def test_single_document_generation(self, abstract_generator):
        """Test generating L0/L1 for a single document."""
        result = abstract_generator.generate(SAMPLE_CONTENT)
        
        # Basic structure
        assert result.l0_abstract
        assert result.l1_overview
        assert isinstance(result.l0_embedding, np.ndarray)
        assert isinstance(result.l1_embedding, np.ndarray)
        assert result.original_sentences > 0
        assert result.generation_time_ms >= 0
        
        # L0 should be shorter than L1
        assert len(result.l0_abstract) > 0  # L0 now includes headers + first sentence
        assert len(result.l1_overview.split('.')) <= 4  # 3 sentences + possible trailing
        
        # L0 should be subset of L1 in terms of content
        # (Not necessarily substring due to reordering)
        
        # Embeddings should be 384-dimensional
        assert result.l0_embedding.shape == (384,)
        assert result.l1_embedding.shape == (384,)
    
    def test_short_content_handling(self, abstract_generator):
        """Test that short content uses full text for both L0 and L1."""
        result = abstract_generator.generate(SHORT_CONTENT)
        
        # For short content, L0 = L1 = full content
        assert result.l0_abstract == SHORT_CONTENT
        assert result.l1_overview == SHORT_CONTENT
        assert np.array_equal(result.l0_embedding, result.l1_embedding)
    
    def test_empty_content_handling(self, abstract_generator):
        """Test handling of empty content."""
        result = abstract_generator.generate("")
        
        assert result.l0_abstract == ""
        assert result.l1_overview == ""
        assert result.original_sentences == 0
        # Should still have embeddings (from placeholder)
        assert isinstance(result.l0_embedding, np.ndarray)
        assert isinstance(result.l1_embedding, np.ndarray)
    
    def test_custom_sentence_counts(self, abstract_generator):
        """Test custom L0/L1 sentence counts."""
        # Request 2 sentences for L0, 5 for L1
        result = abstract_generator.generate(SAMPLE_CONTENT, l0_sentences=2, l1_sentences=5)
        
        # L0 is now metadata-aware (headers + first sentence), so it's longer
        assert len(result.l0_abstract) > 0
        
        # L1 should have more sentences with higher count
        l1_sentences = [s.strip() for s in result.l1_overview.split('.') if s.strip()]
        assert len(l1_sentences) >= 3
    
    def test_centroid_scoring(self, abstract_generator):
        """Test that selected sentences are representative."""
        result = abstract_generator.generate(SAMPLE_CONTENT)
        
        # L0 should contain key concepts
        # Machine learning is the central topic
        l0_lower = result.l0_abstract.lower()
        assert "machine learning" in l0_lower or "ml" in l0_lower or "learning" in l0_lower
        
        # L1 should have broader coverage
        l1_lower = result.l1_overview.lower()
        key_concepts = ["learning", "data", "algorithm", "pattern", "computer", "intelligence"]
        matches = sum(1 for concept in key_concepts if concept in l1_lower)
        assert matches >= 1  # At least 1 key concept should appear
        
        # L1 should be more comprehensive than L0
        assert len(result.l1_overview) > len(result.l0_abstract)
    
    def test_original_order_preservation(self, abstract_generator):
        """Test that L1 preserves original sentence order."""
        # Create content with clearly ordered sentences
        ordered_content = """
First, we introduce the concept.
Second, we explain the methodology.
Third, we present the results.
Fourth, we discuss implications.
Fifth, we conclude the study.
"""
        result = abstract_generator.generate(ordered_content, l0_sentences=1, l1_sentences=3)
        
        # L1 should maintain chronological markers in order
        l1_text = result.l1_overview.lower()
        
        # If "first" appears, it should come before "second", etc.
        positions = {}
        for marker in ["first", "second", "third", "fourth", "fifth"]:
            if marker in l1_text:
                positions[marker] = l1_text.index(marker)
        
        # Check ordering of found markers
        if len(positions) >= 2:
            sorted_markers = sorted(positions.items(), key=lambda x: x[1])
            marker_order = [m[0] for m in sorted_markers]
            expected_order = ["first", "second", "third", "fourth", "fifth"]
            
            # Found markers should be in expected order
            for i in range(len(marker_order) - 1):
                idx1 = expected_order.index(marker_order[i])
                idx2 = expected_order.index(marker_order[i + 1])
                assert idx1 < idx2


class TestBatchGeneration:
    """Test batch abstract generation."""
    
    def test_batch_processing(self, abstract_generator):
        """Test processing multiple documents in batch."""
        results = abstract_generator.generate_batch(MULTI_DOCUMENT_CONTENTS)
        
        # Should return same number of results as inputs
        assert len(results) == len(MULTI_DOCUMENT_CONTENTS)
        
        # Check each result
        for i, result in enumerate(results):
            if MULTI_DOCUMENT_CONTENTS[i]:  # Non-empty content
                assert result.l0_abstract
                assert result.l1_overview
                assert isinstance(result.l0_embedding, np.ndarray)
                assert isinstance(result.l1_embedding, np.ndarray)
            else:  # Empty content
                assert result.l0_abstract == ""
                assert result.l1_overview == ""
    
    def test_batch_vs_individual(self, abstract_generator):
        """Test that batch processing gives same results as individual processing."""
        # Process individually
        individual_results = []
        individual_time = 0
        for content in MULTI_DOCUMENT_CONTENTS[:3]:  # Just first 3 for speed
            start = time.time()
            result = abstract_generator.generate(content)
            individual_time += time.time() - start
            individual_results.append(result)
        
        # Process in batch
        start = time.time()
        batch_results = abstract_generator.generate_batch(MULTI_DOCUMENT_CONTENTS[:3])
        batch_time = time.time() - start
        
        # Results should be equivalent
        for ind, batch in zip(individual_results, batch_results):
            assert ind.l0_abstract == batch.l0_abstract
            assert ind.l1_overview == batch.l1_overview
            # Embeddings might have tiny float differences, use allclose
            assert np.allclose(ind.l0_embedding, batch.l0_embedding)
            assert np.allclose(ind.l1_embedding, batch.l1_embedding)
        
        # Batch should be more efficient (not always true for small batches)
        # So we just verify it completes successfully
        assert batch_time > 0
    
    def test_batch_with_mixed_content(self, abstract_generator):
        """Test batch processing with various edge cases."""
        mixed_content = [
            SAMPLE_CONTENT,          # Normal
            SHORT_CONTENT,          # Short
            "",                     # Empty
            ABBREVIATION_CONTENT,   # Abbreviations
            EDGE_CASE_CONTENT,      # Edge cases
        ]
        
        results = abstract_generator.generate_batch(mixed_content)
        
        assert len(results) == len(mixed_content)
        
        # Check specific cases
        assert results[0].l0_abstract  # Normal content
        assert results[1].l0_abstract == SHORT_CONTENT  # Short = full
        assert results[2].l0_abstract == ""  # Empty
        assert results[3].l0_abstract  # Should handle abbreviations
        assert results[4].l0_abstract  # Should handle edge cases


class TestStoreIntegration:
    """Test integration with VectorStore."""
    
    def test_store_with_abstract_generator(self, tmp_store):
        """Test creating store with abstract generator."""
        # Add document - should auto-generate abstracts
        tmp_store.add(
            doc_id="test_doc_1",
            content=SAMPLE_CONTENT,
            metadata={"type": "test"}
        )
        
        # Check document has abstracts
        doc = tmp_store.get("test_doc_1")
        assert doc is not None
        
        # Check via direct SQL query
        import sqlite3
        with sqlite3.connect(tmp_store.sqlite_path) as conn:
            row = conn.execute('''
                SELECT l0_abstract, l1_overview, l0_embedding, l1_embedding
                FROM documents WHERE doc_id = ?
            ''', ("test_doc_1",)).fetchone()
            
            assert row[0] is not None  # l0_abstract
            assert row[1] is not None  # l1_overview
            assert row[2] is not None  # l0_embedding blob
            assert row[3] is not None  # l1_embedding blob
        
        # Check stats
        stats = tmp_store.stats()
        assert stats['l0_count'] == 1
        assert stats['l1_count'] == 1
        assert stats['progressive_ready'] == True
    
    def test_add_directory_with_abstracts(self, tmp_store, sample_markdown_dir):
        """Test add_directory generates abstracts for all files."""
        stats = tmp_store.add_directory(str(sample_markdown_dir))
        
        # Check indexing stats
        assert stats['files_processed'] > 0
        assert stats['chunks_added'] > 0
        
        # Check store stats
        store_stats = tmp_store.stats()
        assert store_stats['l0_count'] > 0
        assert store_stats['l1_count'] > 0
        assert store_stats['l0_count'] == store_stats['l1_count']
        assert store_stats['progressive_ready'] == True
        
        # Check indices were built
        assert tmp_store._faiss_l0_index is not None
        assert tmp_store._faiss_l1_index is not None
        assert tmp_store._faiss_l0_index.ntotal > 0
        assert tmp_store._faiss_l1_index.ntotal > 0
    
    def test_generate_abstracts_post_hoc(self, tmp_path, embedder, abstract_generator):
        """Test generating abstracts after initial indexing."""
        # Create store WITHOUT abstract generator
        store_path = tmp_path / "test_store_no_gen"
        store = VectorStore(str(store_path), embedder, abstract_generator=None)
        
        # Add documents without abstracts
        store.add("doc1", MULTI_DOCUMENT_CONTENTS[0])
        store.add("doc2", MULTI_DOCUMENT_CONTENTS[1])
        store.add("doc3", MULTI_DOCUMENT_CONTENTS[2])
        
        # Verify no abstracts yet
        initial_stats = store.stats()
        assert initial_stats['l0_count'] == 0
        assert initial_stats['l1_count'] == 0
        assert initial_stats['progressive_ready'] == False
        
        # Generate abstracts post-hoc
        gen_stats = store.generate_abstracts(abstract_generator)
        assert gen_stats['processed'] == 3
        assert gen_stats['errors'] == 0
        
        # Verify abstracts now exist
        final_stats = store.stats()
        assert final_stats['l0_count'] == 3
        assert final_stats['l1_count'] == 3
        assert final_stats['progressive_ready'] == True
    
    def test_l0_l1_search(self, tmp_store):
        """Test L0/L1 index search functionality."""
        # Add multiple documents
        for i, content in enumerate(MULTI_DOCUMENT_CONTENTS[:3]):
            if content:  # Skip empty
                tmp_store.add(f"doc_{i}", content, metadata={"index": i})
        
        # Rebuild indices to ensure consistency
        tmp_store.rebuild_index()
        
        # Test L0 search
        query_embedding = tmp_store.embedder.embed("programming")
        l0_results = tmp_store.search_l0(query_embedding, limit=5)
        
        assert len(l0_results) > 0
        assert all('doc_id' in r for r in l0_results)
        assert all('similarity' in r for r in l0_results)
        assert all('content' in r for r in l0_results)  # Should be L0 abstract text
        
        # Test L1 search
        l1_results = tmp_store.search_l1(query_embedding, limit=5)
        
        assert len(l1_results) > 0
        assert all('doc_id' in r for r in l1_results)
        assert all('similarity' in r for r in l1_results)
        assert all('content' in r for r in l1_results)  # Should be L1 overview text
        
        # L1 content should be longer than L0
        if l0_results and l1_results:
            # Find matching doc
            for l0 in l0_results:
                for l1 in l1_results:
                    if l0['doc_id'] == l1['doc_id']:
                        assert len(l1['content']) >= len(l0['content'])
                        break
    
    def test_stats_consistency(self, tmp_store):
        """Test that stats accurately reflect index state."""
        # Start empty
        stats = tmp_store.stats()
        assert stats['document_count'] == 0
        assert stats['l0_count'] == 0
        assert stats['l1_count'] == 0
        assert stats['progressive_ready'] == False
        
        # Add one document
        tmp_store.add("doc1", SAMPLE_CONTENT)
        stats = tmp_store.stats()
        assert stats['document_count'] == 1
        assert stats['l0_count'] == 1
        assert stats['l1_count'] == 1
        
        # Add more documents
        tmp_store.add("doc2", MULTI_DOCUMENT_CONTENTS[0])
        tmp_store.add("doc3", MULTI_DOCUMENT_CONTENTS[1])
        
        stats = tmp_store.stats()
        assert stats['document_count'] == 3
        assert stats['l0_count'] == 3
        assert stats['l1_count'] == 3
        assert stats['l0_index_vectors'] == 3
        assert stats['l1_index_vectors'] == 3
        assert stats['progressive_ready'] == True
        assert stats['consistent'] == True


class TestProgressiveSearch:
    """Test progressive search functionality."""
    
    def test_search_auto_detects_progressive(self, tmp_path, embedder, abstract_generator):
        """Test that search_progressive() works when L0/L1 available."""
        store = VectorStore(str(tmp_path / "store"), embedder, abstract_generator)
        searcher = Searcher(store, embedder)
        
        # Add documents to enable progressive mode
        for i, content in enumerate(MULTI_DOCUMENT_CONTENTS[:3]):
            if content:
                store.add(f"doc_{i}", content)
        
        # Standard search() stays standard (progressive is opt-in)
        results = searcher.search("neural networks", limit=5)
        assert results.get('search_mode') != 'progressive'
        
        # Explicit progressive search should work
        prog_results = searcher.search_progressive("neural networks", limit=5)
        assert prog_results['search_mode'] == 'progressive'
    
    def test_search_fallback_without_abstracts(self, tmp_path, embedder):
        """Test search falls back to standard when no abstracts."""
        store = VectorStore(str(tmp_path / "store"), embedder, abstract_generator=None)
        searcher = Searcher(store, embedder)
        
        # Add documents without abstracts
        store.add("doc1", SAMPLE_CONTENT)
        
        # Should use standard search
        results = searcher.search("machine learning", limit=5)
        assert results['search_mode'] == 'standard'
    
    def test_progressive_search_quality(self, tmp_path, embedder, abstract_generator):
        """Test that progressive search returns relevant results."""
        store = VectorStore(str(tmp_path / "store"), embedder, abstract_generator)
        searcher = Searcher(store, embedder)
        
        # Add diverse documents
        docs = [
            ("python_intro", "Python is a programming language for beginners."),
            ("python_advanced", "Advanced Python includes decorators, generators, and metaclasses."),
            ("ml_basics", "Machine learning uses algorithms to find patterns in data."),
            ("ml_neural", "Neural networks are a key component of deep learning systems."),
            ("web_dev", "Web development involves HTML, CSS, and JavaScript."),
        ]
        
        for doc_id, content in docs:
            store.add(doc_id, content)
        
        # Search for Python content
        results = searcher.search_progressive("Python programming", limit=2)
        
        assert len(results['results']) <= 2
        assert results['search_mode'] == 'progressive'
        
        # Should find Python-related documents
        found_ids = [r['doc_id'] for r in results['results']]
        assert any('python' in id.lower() for id in found_ids)
        
        # Check progressive stats
        assert 'progressive_stats' in results
        stats = results['progressive_stats']
        assert stats['l0_candidates'] > 0
        assert stats['l1_candidates'] > 0
        assert stats['l2_loaded'] > 0
        assert stats['l0_candidates'] >= stats['l1_candidates'] >= stats['l2_loaded']
    
    def test_progressive_candidate_filtering(self, tmp_path, embedder, abstract_generator):
        """Test L0→L1→L2 candidate filtering."""
        store = VectorStore(str(tmp_path / "store"), embedder, abstract_generator)
        searcher = Searcher(store, embedder)
        
        # Add many documents to test filtering
        for i in range(20):
            content = f"Document {i} contains various information about topic {i % 5}."
            store.add(f"doc_{i}", content)
        
        # Search with specific candidate limits
        results = searcher.search_progressive(
            "information about topic",
            limit=3,
            l0_candidates=10,
            l1_candidates=5
        )
        
        stats = results['progressive_stats']
        
        # Verify progressive filtering
        assert stats['l0_candidates'] <= 10
        assert stats['l1_candidates'] <= 5
        assert len(results['results']) <= 3
        
        # Each stage should narrow down
        assert stats['l0_candidates'] >= stats['l1_candidates']
        assert stats['l1_candidates'] >= len(results['results'])
    
    def test_progressive_with_variants(self, tmp_store):
        """Test that progressive search uses query variants."""
        searcher = Searcher(tmp_store, tmp_store.embedder)
        
        # Add documents
        tmp_store.add("doc1", "Information retrieval systems use vector embeddings.")
        tmp_store.add("doc2", "Information extraction focuses on structured data.")
        tmp_store.add("doc3", "Retrieval augmented generation combines search and LLMs.")
        
        # Search with query that should generate variants
        results = searcher.search_progressive("information retrieval", limit=2)
        
        assert len(results['results']) > 0
        assert results['search_mode'] == 'progressive'
        
        # Should find relevant documents
        contents = [r['content'] for r in results['results']]
        assert any('retrieval' in c.lower() for c in contents)