# 🦖 Velocirag

**Sub-second semantic search for markdown knowledge bases with four-layer fusion.**

Point it at a folder of markdown files. Get lightning-fast search with vector similarity, BM25 keyword matching, knowledge graph traversal, and metadata filtering — all fused via RRF with cross-encoder reranking. 99/100 queries hit on a real 3,358-document knowledge base.

## ✨ Features

- **🚀 Sub-second search** — 80ms average (warm), 384-dim embeddings, runs on 8GB RAM
- **🧠 Four-layer fusion** — Vector + BM25 keyword + knowledge graph + metadata → RRF
- **🔑 BM25 keyword search** — SQLite FTS5 catches exact matches vectors miss (IPs, passwords, codes)
- **🎯 Cross-encoder reranking** — TinyBERT reranker for actual relevance, not just cosine similarity
- **🧬 GLiNER entity extraction** — Optional zero-hallucination NER for typed, scored knowledge graph entities
- **📝 Header-aware chunking** — Splits by `##`/`###` headers, preserves parent context
- **🔍 Query variants** — Handles `CS656` ↔ `CS 656`, acronyms, case variations automatically
- **📊 Knowledge graph** — 6 analyzers (entity, temporal, topic, semantic, centrality, explicit)
- **🏷️ Smart metadata** — Frontmatter extraction, tag filtering, project organization
- **💾 No GPU required** — CPU-only, scales to thousands of docs on modest hardware

## 🚀 Quick Start

```bash
# Install
pip install git+https://github.com/HaseebKhalid1507/VelociRAG.git

# Index your knowledge base
velocirag index ./my-notes/ --graph --metadata

# Search
velocirag search "machine learning fundamentals"

# Health check
velocirag health
```

## 🐍 Python API

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

# BM25 keyword search (exact matches)
store.rebuild_fts()  # Build FTS5 index
results = store.keyword_search('192.168.1.1', limit=5)

# Full 4-layer fusion
graph_store = GraphStore('./my-search-db/graph.db')
metadata_store = MetadataStore('./my-search-db/metadata.db')
pipeline = GraphPipeline(graph_store, embedder, metadata_store)
pipeline.build('./my-notes')

unified = UnifiedSearch(searcher, graph_store, metadata_store)
results = unified.search(
    'machine learning',
    limit=5,
    enrich_graph=True,
    filters={'tags': ['python'], 'status': 'active'}
)

# GLiNER entity extraction (optional)
# pip install velocirag[ner]
pipeline = GraphPipeline(graph_store, embedder, metadata_store, entity_extractor='gliner')
pipeline.build('./my-notes', force_rebuild=True)
```

## 🏗️ Architecture

Four retrieval layers, each catching what the others miss:

```
Query → [Vector] + [Keyword] + [Graph] + [Metadata] → RRF Fusion → Cross-Encoder Rerank → Results
```

### 🎯 Vector Layer
- **all-MiniLM-L6-v2** embeddings (384-dim)
- Query variant expansion for better recall
- Header-aware chunking preserves document structure
- FAISS IndexFlatIP for sub-second cosine similarity

### 🔑 Keyword Layer (BM25)
- SQLite FTS5 with Porter stemming
- Catches exact matches vectors miss — proper nouns, IPs, codes, specific phrases
- ~3ms per query

### 🕸️ Graph Layer
Six analyzers build the knowledge graph:
- **Explicit** — `[[wiki links]]` and references
- **Entity** — Named entities via regex or **GLiNER** encoder (zero hallucination)
- **Temporal** — Date relationships and sequences
- **Topic** — Thematic clustering
- **Semantic** — Conceptual similarity
- **Centrality** — Document importance (PageRank-style)

### 📋 Metadata Layer
- YAML frontmatter parsing
- Tag-based filtering
- Cross-reference tracking
- Usage analytics

### ⚡ Fusion Engine
1. Each layer contributes ranked candidates independently
2. **Reciprocal Rank Fusion (RRF)** merges results across all layers
3. **TinyBERT cross-encoder** reranks top candidates for relevance
4. Graph enrichment adds connections and related notes to final results

## 🧬 GLiNER Entity Extraction

Optional encoder-based NER that replaces regex pattern matching. GLiNER can only tag spans **actually present in the text** — zero hallucination by design.

```bash
pip install velocirag[ner]
velocirag index ./my-notes/ --graph --metadata --gliner
```

```
[person]       "Haseeb"       (0.979)
[technology]   "React"        (0.927)
[organization] "NJIT"         (0.964)
[concept]      "cybersecurity" (0.420)
```

Every entity gets a type and confidence score. Falls back to regex if GLiNER isn't installed.

## 💻 CLI Reference

```bash
# Indexing
velocirag index <path> [--graph] [--metadata] [--gliner] [--force] [--db PATH]

# Search
velocirag search <query> [--limit N] [--threshold F] [--format text|json] [--tags TAG]

# Metadata queries
velocirag query [--tags TAG] [--status S] [--category C] [--project P] [--stale N] [--recent N]

# Health check (all components)
velocirag health [--db PATH] [--format text|json]

# Status & maintenance
velocirag status [--db PATH]
velocirag reindex [--db PATH]
```

## ⚙️ Configuration

```bash
export VELOCIRAG_DB=/path/to/db   # Default database location
export NO_COLOR=1                 # Disable colored output
```

| Parameter | Default | Range |
|-----------|---------|-------|
| Similarity threshold | 0.3 | 0.0–1.0 |
| Result limit | 5 | 1–50 |
| Embedding model | all-MiniLM-L6-v2 | 384 dimensions |
| Cross-encoder | TinyBERT-L-2-v2 | ~17MB |
| GLiNER model | gliner_small-v2.1 | ~170MB, optional |

## 📊 Performance

Real numbers from a production deployment (3,358 documents, 831 markdown files):

| Metric | Value |
|--------|-------|
| Average query time (warm) | **80ms** |
| p50 / p95 / max | 90ms / 198ms / 328ms |
| Cold start | ~3s |
| BM25 keyword layer | ~3ms |
| Hit rate (100-query benchmark) | **99/100** |
| RAM usage (with models) | <8GB |
| Graph | 1,336 nodes, 16,818 edges |
| GLiNER entities | 617 typed + scored |

### What each layer catches

| Query type | Vector | Keyword | Graph | Metadata |
|-----------|--------|---------|-------|----------|
| Semantic ("improving study habits") | ✅ | — | — | — |
| Exact match ("192.168.1.167") | — | ✅ | — | — |
| Connected concepts | — | — | ✅ | — |
| Filtered ("tag:python status:active") | — | — | — | ✅ |
| Combined ("CS634 data mining") | ✅ | ✅ | ✅ | — |

## 🤖 AI Agent Integration

See [AGENTS.md](AGENTS.md) for machine-readable project context — module map, class signatures, response formats, and development patterns for AI coding assistants.

## 📄 License

[PolyForm Noncommercial 1.0.0](LICENSE) — Free for personal, research, and non-commercial use.

---

Built by someone who actually uses it daily. Not a VC-funded startup. Just fast, precise search for people who think in markdown.
