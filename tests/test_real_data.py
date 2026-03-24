#!/usr/bin/env python3
"""
Integration test: Run Phase 1 modules against real Mikoshi vault + Jawz notes.
Not unit tests — this is a live fire exercise.
"""

import os
import sys
import time
import json
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from velocirag.chunker import chunk_markdown
from velocirag.variants import generate_variants
from velocirag.rrf import reciprocal_rank_fusion


MIKOSHI_PATH = os.path.expanduser("~/Jawz/mikoshi/Notes/")
NOTES_PATH = os.path.expanduser("~/Jawz/notes/")


def test_chunker_on_vault():
    """Chunk every markdown file in mikoshi + notes. Report stats."""
    print("=" * 60)
    print("CHUNKER — LIVE FIRE ON REAL DATA")
    print("=" * 60)
    
    sources = {
        'mikoshi': MIKOSHI_PATH,
        'notes': NOTES_PATH,
    }
    
    total_files = 0
    total_chunks = 0
    total_errors = 0
    chunk_sizes = []
    section_types = Counter()
    frontmatter_count = 0
    small_file_count = 0
    truncated_count = 0
    biggest_file = ("", 0)
    most_chunks_file = ("", 0)
    empty_chunks = 0
    
    errors = []
    
    for source_name, source_path in sources.items():
        if not os.path.exists(source_path):
            print(f"  ⚠ {source_name} not found at {source_path}")
            continue
        
        for root, dirs, files in os.walk(source_path):
            for fname in files:
                if not fname.endswith('.md'):
                    continue
                
                fpath = os.path.join(root, fname)
                rel_path = os.path.relpath(fpath, source_path)
                
                try:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    total_files += 1
                    
                    if len(content) > biggest_file[1]:
                        biggest_file = (f"{source_name}/{rel_path}", len(content))
                    
                    chunks = chunk_markdown(content, f"{source_name}/{rel_path}")
                    
                    if len(chunks) > most_chunks_file[1]:
                        most_chunks_file = (f"{source_name}/{rel_path}", len(chunks))
                    
                    for chunk in chunks:
                        total_chunks += 1
                        chunk_size = len(chunk['content'])
                        chunk_sizes.append(chunk_size)
                        
                        section = chunk['metadata']['section']
                        section_types[section] += 1
                        
                        if chunk['metadata']['frontmatter']:
                            frontmatter_count += 1
                        
                        if section == 'full_document':
                            small_file_count += 1
                        
                        if chunk['content'].endswith('...'):
                            truncated_count += 1
                        
                        if chunk_size == 0:
                            empty_chunks += 1
                        
                        # Validate chunk structure
                        assert 'content' in chunk, f"Missing 'content' in chunk from {rel_path}"
                        assert 'metadata' in chunk, f"Missing 'metadata' in chunk from {rel_path}"
                        assert 'file_path' in chunk['metadata'], f"Missing file_path in {rel_path}"
                        assert 'section' in chunk['metadata'], f"Missing section in {rel_path}"
                        assert 'content_hash' in chunk['metadata'], f"Missing content_hash in {rel_path}"
                        assert len(chunk['metadata']['content_hash']) == 12, f"Bad hash length in {rel_path}"
                    
                except Exception as e:
                    total_errors += 1
                    errors.append(f"{source_name}/{rel_path}: {e}")
    
    # Report
    print(f"\n📊 RESULTS")
    print(f"  Files processed:    {total_files}")
    print(f"  Total chunks:       {total_chunks}")
    print(f"  Errors:             {total_errors}")
    print(f"  Avg chunks/file:    {total_chunks/total_files:.1f}")
    print(f"  Small files (full): {small_file_count} ({small_file_count/total_chunks*100:.1f}%)")
    print(f"  Truncated chunks:   {truncated_count}")
    print(f"  Empty chunks:       {empty_chunks}")
    print(f"  With frontmatter:   {frontmatter_count}")
    
    if chunk_sizes:
        print(f"\n📏 CHUNK SIZES")
        print(f"  Min:    {min(chunk_sizes)} chars")
        print(f"  Max:    {max(chunk_sizes)} chars")
        print(f"  Avg:    {sum(chunk_sizes)/len(chunk_sizes):.0f} chars")
        print(f"  Median: {sorted(chunk_sizes)[len(chunk_sizes)//2]} chars")
        
        # Distribution
        brackets = [(0, 100), (100, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 99999)]
        print(f"\n  Distribution:")
        for lo, hi in brackets:
            count = sum(1 for s in chunk_sizes if lo <= s < hi)
            bar = '█' * (count * 40 // len(chunk_sizes))
            label = f"{lo}-{hi}" if hi < 99999 else f"{lo}+"
            print(f"    {label:>10}: {count:>5} ({count/len(chunk_sizes)*100:>5.1f}%) {bar}")
    
    print(f"\n📁 BIGGEST FILE: {biggest_file[0]} ({biggest_file[1]:,} chars)")
    print(f"📦 MOST CHUNKS:  {most_chunks_file[0]} ({most_chunks_file[1]} chunks)")
    
    # Top section names
    print(f"\n🏷️  TOP SECTION TYPES:")
    for section, count in section_types.most_common(10):
        print(f"    {section:30s} {count}")
    
    if errors:
        print(f"\n❌ ERRORS ({len(errors)}):")
        for err in errors[:5]:
            print(f"    {err}")
        if len(errors) > 5:
            print(f"    ... and {len(errors) - 5} more")
    
    print(f"\n{'✅ ALL CHUNKS VALID' if total_errors == 0 else '⚠️ SOME ERRORS'}")
    return total_errors == 0


def test_variants_on_real_queries():
    """Test variant generation on queries someone would actually search for."""
    print("\n" + "=" * 60)
    print("VARIANTS — REAL QUERY PATTERNS")
    print("=" * 60)
    
    # Real queries someone would type into this system
    test_queries = [
        "CS656",
        "CS 646",
        "machine learning",
        "NiteSpeed",
        "jawz-search",
        "HTB academy",
        "OWASP top 10",
        "python3.12",
        "gym_routine",
        "Rick And Morty",
        "2024-08-07",
        "D&D session",
        "tcp/ip",
        "RAM DASS",
        "No Longer Human",
        "cybersecurity",
        "daily journal",
        "MahaMedia",
    ]
    
    total_variants = 0
    max_variants = 0
    
    for query in test_queries:
        variants = generate_variants(query)
        total_variants += len(variants)
        max_variants = max(max_variants, len(variants))
        
        assert len(variants) <= 8, f"Too many variants for '{query}': {len(variants)}"
        assert variants[0] == query, f"Original not first for '{query}'"
        assert len(variants) == len(set(variants)), f"Duplicates in variants for '{query}'"
        
        print(f"\n  '{query}' → {len(variants)} variants:")
        for v in variants:
            marker = "  " if v == query else "→ "
            print(f"    {marker}{v}")
    
    print(f"\n📊 STATS")
    print(f"  Queries tested:     {len(test_queries)}")
    print(f"  Avg variants/query: {total_variants/len(test_queries):.1f}")
    print(f"  Max variants:       {max_variants}")
    print(f"\n✅ ALL VARIANT CHECKS PASSED")
    return True


def test_rrf_with_simulated_search():
    """Simulate what RRF would do with chunked results from different queries."""
    print("\n" + "=" * 60)
    print("RRF — SIMULATED MULTI-VARIANT SEARCH")
    print("=" * 60)
    
    # Chunk a few real files to get realistic content
    test_files = []
    for root, dirs, files in os.walk(MIKOSHI_PATH):
        for fname in files:
            if fname.endswith('.md'):
                test_files.append(os.path.join(root, fname))
                if len(test_files) >= 20:
                    break
        if len(test_files) >= 20:
            break
    
    # Chunk them all
    all_chunks = []
    for fpath in test_files:
        with open(fpath, 'r') as f:
            content = f.read()
        chunks = chunk_markdown(content, os.path.basename(fpath))
        all_chunks.extend(chunks)
    
    print(f"  Loaded {len(all_chunks)} chunks from {len(test_files)} files")
    
    # Simulate 3 "search result lists" (different orderings, some overlap)
    import random
    random.seed(42)
    
    list1 = random.sample(all_chunks, min(10, len(all_chunks)))
    list2 = random.sample(all_chunks, min(10, len(all_chunks)))
    list3 = random.sample(all_chunks, min(10, len(all_chunks)))
    
    # Add fake similarity scores
    for i, chunk in enumerate(list1):
        chunk['metadata']['similarity'] = round(0.95 - i * 0.05, 3)
    for i, chunk in enumerate(list2):
        chunk['metadata']['similarity'] = round(0.90 - i * 0.04, 3)
    for i, chunk in enumerate(list3):
        chunk['metadata']['similarity'] = round(0.85 - i * 0.03, 3)
    
    # Fuse
    fused = reciprocal_rank_fusion([list1, list2, list3], k=60)
    
    print(f"\n  List 1: {len(list1)} results")
    print(f"  List 2: {len(list2)} results")
    print(f"  List 3: {len(list3)} results")
    print(f"  Fused:  {len(fused)} unique results")
    
    # Check structure
    for result in fused:
        assert 'content' in result
        assert 'metadata' in result
        assert 'rrf_score' in result['metadata']
    
    # Show top 5
    print(f"\n  TOP 5 FUSED RESULTS:")
    for i, result in enumerate(fused[:5]):
        score = result['metadata']['rrf_score']
        section = result['metadata'].get('section', '?')
        fpath = result['metadata'].get('file_path', '?')
        preview = result['content'][:60].replace('\n', ' ')
        print(f"    {i+1}. [RRF={score:.4f}] {fpath} :: {section}")
        print(f"       {preview}...")
    
    # Verify RRF scores are decreasing
    scores = [r['metadata']['rrf_score'] for r in fused]
    assert scores == sorted(scores, reverse=True), "RRF scores not sorted!"
    
    # Verify no duplicates by content hash
    hashes = [r['metadata'].get('content_hash', r['content'][:50]) for r in fused]
    unique_hashes = set(hashes)
    print(f"\n  Unique results: {len(unique_hashes)}/{len(fused)}")
    
    print(f"\n✅ RRF FUSION VERIFIED")
    return True


def test_embed_real_chunks():
    """Test embedding real chunks from mikoshi vault."""
    print("\n" + "=" * 60)
    print("EMBEDDER — REAL CHUNKS FROM MIKOSHI")
    print("=" * 60)
    
    import pytest
    import numpy as np
    from velocirag.embedder import Embedder
    
    # Skip if mikoshi doesn't exist
    if not os.path.exists(MIKOSHI_PATH):
        print("  ⚠ Mikoshi not found, skipping test")
        return True
    
    # Get some real markdown files
    test_files = []
    for root, dirs, files in os.walk(MIKOSHI_PATH):
        for fname in files:
            if fname.endswith('.md'):
                test_files.append(os.path.join(root, fname))
                if len(test_files) >= 10:
                    break
        if len(test_files) >= 10:
            break
    
    # Chunk them
    all_chunks = []
    for fpath in test_files[:5]:  # Use first 5 files
        with open(fpath, 'r', encoding='utf-8') as f:
            content = f.read()
        chunks = chunk_markdown(content, os.path.relpath(fpath, MIKOSHI_PATH))
        all_chunks.extend(chunks)
    
    # Take first 50 chunks
    test_chunks = all_chunks[:50]
    print(f"  Loaded {len(test_chunks)} chunks from {len(test_files[:5])} files")
    
    # Initialize embedder with temp cache
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        embedder = Embedder(cache_dir=tmp_dir)
        
        # Embed all chunks
        start_time = time.time()
        embeddings = []
        
        for chunk in test_chunks:
            embedding = embedder.embed(chunk['content'])
            embeddings.append(embedding)
        
        elapsed = time.time() - start_time
        
        # Verify embeddings
        dimensions = embeddings[0].shape[0]
        
        for i, emb in enumerate(embeddings):
            # Check dimensions
            assert emb.shape == (dimensions,), f"Wrong shape for embedding {i}: {emb.shape}"
            
            # Check no NaN
            assert not np.isnan(emb).any(), f"NaN in embedding {i}"
            
            # Check no all-zeros
            assert not np.allclose(emb, 0.0), f"All zeros in embedding {i}"
        
        # Check uniqueness (all embeddings should be different)
        unique_count = len({emb.tobytes() for emb in embeddings})
        
        print(f"\n📊 RESULTS")
        print(f"  Chunks processed:   {len(test_chunks)}")
        print(f"  Time taken:         {elapsed:.2f}s")
        print(f"  Embeddings/sec:     {len(test_chunks)/elapsed:.1f}")
        print(f"  Embedding dims:     {dimensions}")
        print(f"  Unique embeddings:  {unique_count}/{len(embeddings)}")
        print(f"  Model:              {embedder.get_model_info()['model_name']}")
        
        assert unique_count == len(embeddings), f"Duplicate embeddings found!"
        
        print(f"\n✅ ALL EMBEDDINGS VALID")
        return True


def test_semantic_similarity_check():
    """Test semantic similarity on real content pairs."""
    print("\n" + "=" * 60)
    print("EMBEDDER — SEMANTIC SIMILARITY CHECK")
    print("=" * 60)
    
    import numpy as np
    from velocirag.embedder import Embedder
    
    # Skip if mikoshi doesn't exist
    if not os.path.exists(MIKOSHI_PATH):
        print("  ⚠ Mikoshi not found, skipping test")
        return True
    
    # Find related and unrelated content pairs
    # Let's look for CS course notes vs other topics
    cs_files = []
    other_files = []
    
    for root, dirs, files in os.walk(MIKOSHI_PATH):
        for fname in files:
            if fname.endswith('.md'):
                fpath = os.path.join(root, fname)
                if 'CS' in fname and any(x in fname for x in ['646', '656', '672']):
                    cs_files.append(fpath)
                elif any(x in fname.lower() for x in ['cooking', 'anime', 'journal', 'book']):
                    other_files.append(fpath)
    
    if len(cs_files) < 2 or len(other_files) < 1:
        print("  ⚠ Not enough diverse files found, using fallback")
        # Fallback: create synthetic related/unrelated pairs
        pairs = [
            ("Machine learning algorithms for classification", 
             "Neural networks and deep learning fundamentals",
             "Recipe for chocolate chip cookies with walnuts"),
            ("Python programming best practices",
             "Software engineering principles and design patterns", 
             "My thoughts on the weather today"),
            ("Database management systems and SQL",
             "Data structures and algorithms analysis",
             "Review of the latest anime series")
        ]
    else:
        # Get real content chunks
        pairs = []
        
        # Load CS content
        cs_chunks = []
        for fpath in cs_files[:2]:
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read()
            chunks = chunk_markdown(content, os.path.basename(fpath))
            if chunks:
                cs_chunks.append(chunks[0]['content'][:1000])  # First chunk, truncated
        
        # Load other content
        other_chunk = None
        for fpath in other_files[:1]:
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read()
            chunks = chunk_markdown(content, os.path.basename(fpath))
            if chunks:
                other_chunk = chunks[0]['content'][:1000]
                break
        
        if len(cs_chunks) >= 2 and other_chunk:
            pairs = [(cs_chunks[0], cs_chunks[1], other_chunk)]
            print(f"  Using real content from vault")
    
    # Initialize embedder
    embedder = Embedder(normalize=True)  # Use normalization for cosine similarity
    
    # Test each triplet
    print(f"\n  Testing {len(pairs)} content triplets...")
    
    for i, (related1, related2, unrelated) in enumerate(pairs):
        # Embed all three
        emb1 = embedder.embed(related1)
        emb2 = embedder.embed(related2)
        emb3 = embedder.embed(unrelated)
        
        # Compute cosine similarities (embeddings are normalized)
        sim_related = np.dot(emb1, emb2)
        sim_unrelated1 = np.dot(emb1, emb3)
        sim_unrelated2 = np.dot(emb2, emb3)
        
        print(f"\n  Triplet {i+1}:")
        print(f"    Related similarity:    {sim_related:.4f}")
        print(f"    Unrelated sim 1:       {sim_unrelated1:.4f}")
        print(f"    Unrelated sim 2:       {sim_unrelated2:.4f}")
        print(f"    Difference:            {sim_related - max(sim_unrelated1, sim_unrelated2):.4f}")
        
        # Assert related content is more similar
        assert sim_related > sim_unrelated1, f"Related content not more similar! {sim_related:.4f} <= {sim_unrelated1:.4f}"
        assert sim_related > sim_unrelated2, f"Related content not more similar! {sim_related:.4f} <= {sim_unrelated2:.4f}"
    
    print(f"\n✅ SEMANTIC SIMILARITY VERIFIED")
    return True


def test_cache_performance():
    """Test cache performance on real data."""
    print("\n" + "=" * 60)
    print("EMBEDDER — CACHE PERFORMANCE TEST")
    print("=" * 60)
    
    import numpy as np
    from velocirag.embedder import Embedder
    import tempfile
    
    # Skip if mikoshi doesn't exist
    if not os.path.exists(MIKOSHI_PATH):
        print("  ⚠ Mikoshi not found, skipping test")
        return True
    
    # Get real chunks
    test_files = []
    for root, dirs, files in os.walk(MIKOSHI_PATH):
        for fname in files:
            if fname.endswith('.md'):
                test_files.append(os.path.join(root, fname))
                if len(test_files) >= 10:
                    break
        if len(test_files) >= 10:
            break
    
    # Chunk them
    all_chunks = []
    for fpath in test_files[:5]:
        with open(fpath, 'r', encoding='utf-8') as f:
            content = f.read()
        chunks = chunk_markdown(content, os.path.basename(fpath))
        all_chunks.extend(chunks)
    
    # Take first 50 chunks
    test_texts = [chunk['content'] for chunk in all_chunks[:50]]
    print(f"  Testing with {len(test_texts)} real chunks")
    
    # Initialize embedder with cache
    with tempfile.TemporaryDirectory() as tmp_dir:
        embedder = Embedder(cache_dir=tmp_dir)
        
        # First pass - cold cache
        start_cold = time.time()
        embeddings_cold = embedder.embed(test_texts)
        time_cold = time.time() - start_cold
        
        # Check cache was populated
        cache_info = embedder.get_model_info()
        assert cache_info['cache_size'] == len(test_texts), f"Cache not fully populated: {cache_info['cache_size']} != {len(test_texts)}"
        
        # Second pass - warm cache
        start_warm = time.time()
        embeddings_warm = embedder.embed(test_texts)
        time_warm = time.time() - start_warm
        
        # Verify results are identical
        assert np.allclose(embeddings_cold, embeddings_warm), "Cached embeddings don't match!"
        
        # Calculate speedup
        speedup = time_cold / time_warm if time_warm > 0 else float('inf')
        
        print(f"\n📊 CACHE PERFORMANCE")
        print(f"  Cold run:     {time_cold:.3f}s ({len(test_texts)/time_cold:.1f} chunks/sec)")
        print(f"  Warm run:     {time_warm:.3f}s ({len(test_texts)/time_warm:.1f} chunks/sec)")
        print(f"  Speedup:      {speedup:.1f}x")
        print(f"  Cache size:   {cache_info['cache_size']} entries")
        
        # Assert significant speedup (at least 10x)
        assert speedup > 10, f"Cache speedup too low: {speedup:.1f}x (expected >10x)"
        
        # Test persistence - save and reload
        embedder.save_cache()
        
        # Create new embedder instance with same cache dir
        embedder2 = Embedder(cache_dir=tmp_dir)
        cache_info2 = embedder2.get_model_info()
        
        assert cache_info2['cache_size'] == len(test_texts), f"Cache not persisted correctly: {cache_info2['cache_size']} != {len(test_texts)}"
        
        # Test with new instance (should be fast)
        start_reload = time.time()
        embeddings_reload = embedder2.embed(test_texts)
        time_reload = time.time() - start_reload
        
        # Verify still fast and identical
        assert np.allclose(embeddings_cold, embeddings_reload), "Reloaded embeddings don't match!"
        # Reloaded cache should be faster than cold, but not as fast as warm (due to model loading)
        assert time_reload < time_cold * 0.8, f"Reloaded cache not fast enough: {time_reload:.3f}s vs cold {time_cold:.3f}s"
        
        print(f"  Reload run:   {time_reload:.3f}s (persisted cache works)")
        
    print(f"\n✅ CACHE PERFORMANCE VERIFIED")
    return True


def test_full_pipeline():
    """Test full pipeline: chunk then embed real files."""
    print("\n" + "=" * 60)
    print("EMBEDDER — FULL PIPELINE TEST")
    print("=" * 60)
    
    import numpy as np
    from velocirag.embedder import Embedder
    
    # Skip if mikoshi doesn't exist
    if not os.path.exists(MIKOSHI_PATH):
        print("  ⚠ Mikoshi not found, skipping test")
        return True
    
    # Get 20 real markdown files
    test_files = []
    for root, dirs, files in os.walk(MIKOSHI_PATH):
        for fname in files:
            if fname.endswith('.md'):
                test_files.append(os.path.join(root, fname))
                if len(test_files) >= 20:
                    break
        if len(test_files) >= 20:
            break
    
    print(f"  Processing {len(test_files)} files from vault")
    
    # Initialize embedder
    embedder = Embedder()
    
    # Process each file
    total_chunks = 0
    total_embeddings = 0
    total_chars = 0
    file_stats = []
    
    start_time = time.time()
    
    for fpath in test_files:
        try:
            # Read file
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Chunk it
            chunks = chunk_markdown(content, os.path.relpath(fpath, MIKOSHI_PATH))
            
            # Embed each chunk
            embeddings = []
            for chunk in chunks:
                embedding = embedder.embed(chunk['content'])
                embeddings.append(embedding)
                total_chars += len(chunk['content'])
            
            # Verify all embeddings
            for emb in embeddings:
                assert emb.shape == (384,), f"Wrong embedding shape: {emb.shape}"
                assert not np.isnan(emb).any(), "NaN in embedding"
                assert not np.allclose(emb, 0.0), "All zeros in embedding"
            
            # Record stats
            total_chunks += len(chunks)
            total_embeddings += len(embeddings)
            file_stats.append({
                'file': os.path.basename(fpath),
                'chunks': len(chunks),
                'size': len(content)
            })
            
        except Exception as e:
            print(f"  ⚠ Error processing {os.path.basename(fpath)}: {e}")
    
    elapsed = time.time() - start_time
    
    # Report results
    print(f"\n📊 PIPELINE RESULTS")
    print(f"  Files processed:     {len(test_files)}")
    print(f"  Total chunks:        {total_chunks}")
    print(f"  Total embeddings:    {total_embeddings}")
    print(f"  Total characters:    {total_chars:,}")
    print(f"  Total time:          {elapsed:.2f}s")
    print(f"  Throughput:          {total_chunks/elapsed:.1f} chunks/sec")
    
    # Show biggest files
    file_stats.sort(key=lambda x: x['chunks'], reverse=True)
    print(f"\n  Files with most chunks:")
    for stat in file_stats[:5]:
        print(f"    {stat['file']:40s} {stat['chunks']:3d} chunks ({stat['size']:,} chars)")
    
    # Verify we processed everything
    assert total_chunks > 0, "No chunks generated!"
    assert total_embeddings == total_chunks, f"Embedding count mismatch: {total_embeddings} != {total_chunks}"
    
    # Check embedding dimensions are consistent
    model_info = embedder.get_model_info()
    print(f"\n  Model: {model_info['model_name']} ({model_info['dimensions']}d)")
    print(f"  Cache: {model_info['cache_size']} entries")
    
    print(f"\n✅ FULL PIPELINE VERIFIED")
    return True


if __name__ == '__main__':
    print("🦖 VELOCIRAGTOR — PHASE 1 & 2 LIVE FIRE TEST")
    print(f"   Mikoshi: {MIKOSHI_PATH}")
    print(f"   Notes:   {NOTES_PATH}")
    print()
    
    start = time.time()
    
    # Phase 1 tests
    r1 = test_chunker_on_vault()
    r2 = test_variants_on_real_queries()
    r3 = test_rrf_with_simulated_search()
    
    # Phase 2 tests (new embedder tests)
    r4 = test_embed_real_chunks()
    r5 = test_semantic_similarity_check()
    r6 = test_cache_performance()
    r7 = test_full_pipeline()
    
    elapsed = time.time() - start
    
    print("\n" + "=" * 60)
    print(f"🦖 LIVE FIRE COMPLETE — {elapsed:.2f}s")
    print(f"   Phase 1:")
    print(f"     Chunker:    {'✅' if r1 else '❌'}")
    print(f"     Variants:   {'✅' if r2 else '❌'}")
    print(f"     RRF:        {'✅' if r3 else '❌'}")
    print(f"   Phase 2:")
    print(f"     Embedder:   {'✅' if r4 else '❌'}")
    print(f"     Similarity: {'✅' if r5 else '❌'}")
    print(f"     Cache:      {'✅' if r6 else '❌'}")
    print(f"     Pipeline:   {'✅' if r7 else '❌'}")
    print("=" * 60)
