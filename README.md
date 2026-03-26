# 🦖 Velocirag

**Sub-second semantic search for markdown knowledge bases with multi-layer fusion.**

Point it at a folder of markdown files. Get lightning-fast semantic search with vector similarity, knowledge graph enrichment, metadata filtering, and cross-encoder reranking. All three layers fuse via RRF for precision that'll make your brain tingle.

## ✨ Features

- **🚀 Sub-second search** — 80ms average on warm daemon, 384-dim embeddings, runs on 8GB RAM
- **🧠 Triple fusion** — Vector similarity + knowledge graph + metadata filters via RRF
- **📝 Header-aware chunking** — Splits by `##`/`###` headers, preserves parent context
- **🎯 Cross-encoder reranking** — TinyBERT reranker for actual relevance (not just cosine similarity)
- **🔍 Query variants** — Handles `CS656` ↔ `CS 656`, acronyms, case variations automatically
- **📊 Knowledge graph** — Wiki-link parsing, 6 analyzers (entity, temporal, topic, semantic, centrality, explicit)
- **🏷️ Smart metadata** — Frontmatter extraction, tag filtering, project organization
- **💾 No GPU required** — CPU-only, scales to thousands of docs on modest hardware

## 🚀 Quick Start

```bash
# Install
pip install git+https://github.com/HaseebKhalid1507/VelociRAG.git

# Index your knowledge base
velocirag index ./my-notes/ --graph --metadata

# Search like a boss  
velocirag search "machine learning fundamentals"
```

**That's it.** Three commands to go from markdown folder to production search.

## 🐍 Python API

Use Velocirag as a library for programmatic access:

```python
from velocirag import (
    Embedder, VectorStore, Searcher, 
    GraphStore, MetadataStore, UnifiedSearch, 
    GraphPipeline, UsageTracker
)

# Basic vector search
embedder = Embedder()
store = VectorStore('./my-search-db', embedder)
store.add_directory('./my-notes')

searcher = Searcher(store, embedder)
results = searcher.search('machine learning', limit=5)

# Full 3-layer fusion (the good stuff)
graph_store = GraphStore('./my-search-db/graph.db')
metadata_store = MetadataStore('./my-search-db/metadata.db')

# Build knowledge graph  
pipeline = GraphPipeline(graph_store, embedder, metadata_store)
pipeline.build('./my-notes')

# Unified search with all layers
unified = UnifiedSearch(searcher, graph_store, metadata_store)
results = unified.search(
    'machine learning', 
    limit=5, 
    enrich_graph=True,
    filters={'tags': ['python'], 'status': 'active'}
)
```

## 🏗️ Architecture

Velocirag fuses three retrieval layers for maximum precision:

### 🎯 Vector Layer
- **all-MiniLM-L6-v2** embeddings (384-dim, balance of speed vs quality)
- **Query variants** — automatic expansion for better recall
- **Header-aware chunking** — maintains document structure
- **FAISS index** — sub-second similarity search

### 🕸️ Graph Layer  
Six specialized analyzers extract different relationship types:
- **Explicit** — `[[wiki links]]` and direct references
- **Entity** — Named entities and their connections  
- **Temporal** — Date relationships and sequences
- **Topic** — Thematic clustering via embeddings
- **Semantic** — Conceptual similarity networks
- **Centrality** — Document importance scoring

### 📋 Metadata Layer
- **Frontmatter parsing** — YAML metadata extraction
- **Auto-tagging** — Content analysis for tags
- **Cross-references** — Inter-document link tracking  
- **Usage analytics** — Search patterns and access frequency

### ⚡ Fusion Engine
**Reciprocal Rank Fusion (RRF)** combines all layers:
1. Each layer votes with ranked candidates
2. RRF merges votes with balanced weighting
3. **TinyBERT cross-encoder** reranks top candidates
4. Final results optimized for human relevance

## 💻 CLI Reference

### Index Documents
```bash
# Basic indexing
velocirag index ./docs/

# With knowledge graph and metadata
velocirag index ./docs/ --graph --metadata

# Force reindex everything
velocirag index ./docs/ --force --graph --metadata

# Custom database location  
velocirag index ./docs/ --db ~/my-search.db
```

### Search Content
```bash
# Basic search
velocirag search "neural networks"

# More results with lower threshold
velocirag search "python testing" --limit 10 --threshold 0.3

# Metadata filters
velocirag search "data structures" --tags python --status active

# JSON output for scripts
velocirag search "algorithms" --format json --stats
```

### Metadata Queries
```bash
# Find by tags
velocirag query --tags python --tags rust

# Project-specific docs
velocirag query --project myproject --status active

# Stale content audit
velocirag query --stale 90

# Recent additions
velocirag query --recent 7

# Metadata overview
velocirag query --stats
```

### Maintenance
```bash
# Check index health
velocirag status

# Rebuild corrupted index
velocirag reindex

# Detailed diagnostics
velocirag status
```

## ⚙️ Configuration

### Environment Variables
```bash
export VELOCIRAG_DB=/path/to/search.db         # Default database location
export NO_COLOR=1                            # Disable colored output
```

### Search Parameters  
- **Similarity threshold**: 0.0-1.0 (default: 0.3)
- **Result limit**: 1-50 (default: 5)
- **Graph enrichment**: Automatic when graph.db exists
- **Metadata filters**: Tags, status, category, project

### Index Settings
- **Embedding model**: all-MiniLM-L6-v2 (384 dimensions)
- **Chunk size**: Adaptive based on headers
- **Graph analyzers**: All 6 enabled by default
- **Cross-encoder**: TinyBERT for reranking

## 📊 Performance

Real numbers from a production knowledge base:

### Scale
- **3,354 documents** indexed from 831 markdown files
- **4,246 graph nodes**, 18,570 edges  
- **6 graph analyzers** running in parallel
- **384-dimensional** embeddings for optimal speed/quality

### Speed
- **~80ms** average query time (warm daemon)
- **~3s** cold start time  
- **Sub-second** indexing for typical markdown files
- **<8GB RAM** usage including models

### Quality
- **Cross-encoder reranking** for relevance beyond cosine similarity
- **Multi-layer fusion** captures semantic + structural + metadata signals
- **Query variant expansion** handles terminology variations
- **Header-aware chunking** preserves document context

## 📄 License

[PolyForm Noncommercial 1.0.0](LICENSE) — Free for personal, research, and non-commercial use.

---

Built by someone who actually uses it daily. Not a VC-funded startup. Just fast, precise search for people who think in markdown.

**Questions? Issues? Want to contribute?** Open an issue or submit a PR. Let's make knowledge search not suck.