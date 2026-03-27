# 🦖 Velocirag

**Sub-second semantic search for markdown knowledge bases with four-layer fusion.**

Point it at a folder of markdown files. Get lightning-fast search with vector similarity, BM25 keyword matching, knowledge graph traversal, and metadata filtering — all fused via RRF with cross-encoder reranking. 99/100 queries hit on a real 3,357-document knowledge base.

## ✨ Features

- **🚀 Sub-second search** — 80ms average (warm), 384-dim embeddings, runs on 8GB RAM
- **🧠 Four-layer fusion** — Vector + BM25 keyword + knowledge graph + metadata → RRF
- **🔑 BM25 keyword search** — SQLite FTS5 catches exact matches vectors miss (IPs, passwords, codes)
- **🎯 Cross-encoder reranking** — Auto-initialized TinyBERT reranker, no manual wiring needed
- **🧬 GLiNER entity extraction** — Optional zero-hallucination NER for typed, scored knowledge graph entities
- **🔗 Relation extraction** — GLiNER Multitask finds semantic relationships between entities (uses, enables, evolved_from)
- **📝 Header-aware chunking** — Splits by `##`/`###` headers, preserves parent context
- **🔍 Smart query expansion** — 42-term acronym registry, question→statement rewrite, case/spacing variants
- **📊 Knowledge graph** — 7 analyzers (entity, relation, temporal, topic, semantic, centrality, explicit)
- **🏷️ Smart metadata** — Frontmatter extraction, tag filtering, project organization
- **🩺 Health checks** — Full stack diagnostics via CLI and daemon endpoint
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

# Index — one line
embedder = Embedder()
store = VectorStore('./my-search-db', embedder)
store.add_directory('./my-notes')

# Search — reranker auto-initializes, no manual setup
searcher = Searcher(store, embedder)
results = searcher.search('machine learning', limit=5)

# BM25 keyword search (exact matches)
store.rebuild_fts()
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
```

### GLiNER Entity + Relation Extraction

```bash
pip install velocirag[ner]
velocirag index ./my-notes/ --graph --metadata --gliner
```

```python
# Typed entities with confidence scores
pipeline = GraphPipeline(graph_store, embedder, metadata_store, entity_extractor='gliner')
pipeline.build('./my-notes')
# → [person] "Haseeb" (0.979)
# → [technology] "React" (0.927)
# → [organization] "NJIT" (0.964)

# Semantic relations between entities
# → NiteSpeed --[part_of]--> NJIT
# → Kubernetes --[evolved_from]--> Docker Swarm
```

### Custom Acronym Registry

```python
from velocirag.variants import register_acronyms

register_acronyms({
    'loar': 'law of accelerating returns',
    'njit': 'new jersey institute of technology'
})
# Now "loar" auto-expands in search queries
```

## 🏗️ Architecture

```
Query → expand (acronyms, variants, question rewrite)
      → [Vector]   FAISS cosine similarity (384d, MiniLM-L6-v2)
      → [Keyword]  BM25 via SQLite FTS5 (exact match)
      → [Graph]    Knowledge graph traversal (GLiNER entities + relations)
      → [Metadata] Structured SQL filters (tags, status, project)
      → RRF Fusion (merge all layers by rank)
      → Cross-encoder rerank (TinyBERT, auto-initialized)
      → Results with graph enrichment
```

### What each layer catches

| Query type | Vector | Keyword | Graph | Metadata |
|-----------|--------|---------|-------|----------|
| Semantic ("improving study habits") | ✅ | — | — | — |
| Exact match ("192.168.1.167") | — | ✅ | — | — |
| Connected concepts | — | — | ✅ | — |
| Filtered ("tag:python status:active") | — | — | — | ✅ |
| Combined ("CS634 data mining") | ✅ | ✅ | ✅ | — |

### Knowledge Graph

Seven analyzers build the graph from your markdown:

| Analyzer | What it finds |
|----------|-------------|
| **Explicit** | `[[wiki links]]` and references |
| **Entity** | Named entities via regex or GLiNER (person, technology, concept, organization) |
| **Relation** | Semantic relationships via GLiNER Multitask (uses, enables, evolved_from, created_by) |
| **Temporal** | Date relationships and sequences |
| **Topic** | Thematic clustering |
| **Semantic** | Conceptual similarity |
| **Centrality** | Document importance scoring |

## 💻 CLI Reference

```bash
# Indexing
velocirag index <path> [--graph] [--metadata] [--gliner] [--force] [--db PATH]

# Search — shows active layers, reranker scores, file paths
velocirag search <query> [--limit N] [--threshold F] [--format text|json|compact]

# Metadata queries
velocirag query [--tags TAG] [--status S] [--category C] [--project P] [--stale N] [--recent N]

# Health check (all components)
velocirag health [--db PATH] [--format text|json]

# Status & maintenance
velocirag status [--db PATH]
velocirag reindex [--db PATH]
```

### CLI Search Output

```
$ velocirag search "wireguard VPN setup" --db ./data

Query: "wireguard VPN setup"
Found 3 results in 293ms [vector + keyword + graph]

1. [0.686] projects/jade-homelab.md
   rerank=4.40 | via vector
   ### Wireguard VPN (wg-easy) - Container: wg-easy - Ports: 51820/udp...

2. [0.392] 3. Learning/Concepts/VPN.md
   rerank=-4.24 | via vector | → LDAP, OS Security
   # VPN - Virtual Private Networks extend private networks across...
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
| Cross-encoder | TinyBERT-L-2-v2 | ~17MB, auto-initialized |
| GLiNER entity model | gliner_small-v2.1 | ~170MB, optional |
| GLiNER relation model | gliner-multitask-large-v0.5 | ~1.5GB, optional |
| Acronym registry | 42 built-in terms | Extensible via `register_acronyms()` |

## 📊 Performance

Real numbers from a production deployment (3,357 documents, 831 markdown files):

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

## 🤖 AI Agent Integration

See [AGENTS.md](AGENTS.md) for machine-readable project context — module map, class signatures, response formats, and development patterns for AI coding assistants.

## 📄 License

[PolyForm Noncommercial 1.0.0](LICENSE) — Free for personal, research, and non-commercial use.

---

Built by someone who actually uses it daily. Not a VC-funded startup. Just fast, precise search for people who think in markdown.
