# AGENTS.md — Velocirag for AI Coding Agents

> Machine-readable project context for AI coding assistants (Claude Code, Cursor, Copilot, etc.)

## What Is This?

Velocirag is a **multi-layer RAG engine for markdown knowledge bases**. It indexes folders of `.md` files and provides semantic search with 4 retrieval layers fused via RRF.

- **Language:** Python 3.10+
- **Source:** `src/velocirag/` (17 modules, ~9K lines)
- **Tests:** `tests/` (18 test files)
- **CLI:** `velocirag` (click-based)
- **License:** PolyForm Noncommercial 1.0.0

## Architecture

```
markdown files → chunk → embed → store (SQLite + FAISS)
                                    ↓
                              4-layer search:
                                vector (cosine similarity)
                              + keyword (BM25 via FTS5)
                              + graph (knowledge graph traversal)
                              + metadata (structured SQL filters)
                                    ↓
                              RRF fusion → cross-encoder rerank → results
```

## Module Map

| Module | Lines | Purpose |
|--------|-------|---------|
| `store.py` | 1079 | Vector storage — SQLite + FAISS + FTS5. Core data layer. |
| `cli.py` | 1344 | Click CLI — index, search, query, status, health, reindex |
| `graph.py` | 909 | Knowledge graph — Node/Edge models, GraphStore (SQLite), GraphQuerier |
| `unified.py` | 839 | 4-layer fusion search orchestrator — vector + keyword + metadata + graph → RRF |
| `searcher.py` | 818 | High-level search — query variants, RRF fusion, caching |
| `analyzers.py` | 965 | 7 graph analyzers + GLiNER NER + relation extraction + GLiNER NER (optional) |
| `metadata.py` | 695 | Metadata store — frontmatter, tags, cross-refs, usage tracking |
| `pipeline.py` | 573 | 10-stage graph build pipeline |
| `abstracts.py` | 513 | L0/L1 abstract generation for progressive search |
| `embedder.py` | 461 | Sentence-transformer embeddings (all-MiniLM-L6-v2, 384d) |
| `tracker.py` | 308 | Usage tracking — search hits, reads, access patterns |
| `frontmatter.py` | 172 | YAML frontmatter parser, tag extraction, wiki-link extraction |
| `reranker.py` | 166 | Cross-encoder reranking (TinyBERT) |
| `chunker.py` | 158 | Markdown chunking by headers with parent context preservation |
| `rrf.py` | 143 | Reciprocal Rank Fusion implementation |
| `variants.py` | 250 | Query variant generation + acronym registry + question rewrite (casing, spacing, acronyms) |

## Key Classes

```python
# Index documents
embedder = Embedder()
store = VectorStore("./db", embedder)
store.add_directory("./my-notes", source="notes")

# Basic search
searcher = Searcher(store, embedder)
results = searcher.search("query", limit=5, threshold=0.3)

# Full unified search (4 layers)
graph_store = GraphStore("./db/graph.db")
metadata_store = MetadataStore("./db/metadata.db")
unified = UnifiedSearch(searcher, graph_store, metadata_store)
results = unified.search("query", limit=5, enrich_graph=True)

# Build knowledge graph
pipeline = GraphPipeline(graph_store, embedder, metadata_store, entity_extractor="gliner")
pipeline.build("./my-notes", force_rebuild=True)

# BM25 keyword search (direct)
results = store.keyword_search("exact phrase", limit=10)

# Graph queries
querier = GraphQuerier(graph_store)
querier.find_connections("node_title", depth=2)
querier.get_topic_web("topic")
querier.get_hub_nodes(limit=10)
```

## CLI Commands

```bash
velocirag index <path> [--db PATH] [--graph] [--metadata] [--gliner] [--force]
velocirag search <query> [--db PATH] [--limit N] [--threshold F] [--format text|json]
velocirag query [--tags TAG] [--status S] [--category C] [--project P]
velocirag health [--db PATH] [--format text|json]
velocirag status [--db PATH]
velocirag reindex [--db PATH]
```

## Data Files

When you index, Velocirag creates these in the `--db` directory:

| File | Format | Contains |
|------|--------|----------|
| `store.db` | SQLite | Document chunks, embeddings, file cache, FTS5 index |
| `index.faiss` | FAISS | Vector similarity index (384d, IndexFlatIP) |
| `graph.db` | SQLite | Knowledge graph nodes + edges |
| `metadata.db` | SQLite | Frontmatter metadata, tags, cross-references, usage log |

## Search Response Format

`UnifiedSearch.search()` returns:

```json
{
  "results": [
    {
      "doc_id": "source::path/to/file.md::chunk_N",
      "content": "chunk text...",
      "similarity": 0.85,
      "score": 0.85,
      "metadata": {
        "file_path": "path/to/file.md",
        "section": "section header",
        "source_name": "notes",
        "rrf_score": 0.016,
        "source_layers": "vector",
        "graph_connections": ["Related Note 1", "Related Note 2"],
        "found_in_graph": true
      }
    }
  ],
  "query": "search query",
  "total_results": 5,
  "search_time_ms": 85.3,
  "search_mode": "unified_full",
  "layer_stats": {
    "vector": {"candidates": 15, "time_ms": 10.2},
    "keyword": {"candidates": 8, "time_ms": 2.1},
    "metadata": {"candidates": 0, "time_ms": 0.5},
    "graph": {"candidates": 3, "time_ms": 5.8}
  },
  "fusion_stats": {"layers_fused": 3, "total_candidates": 26}
}
```

## Dependencies

**Required:**
- `numpy`, `faiss-cpu`, `sentence-transformers`, `python-frontmatter`, `pyyaml`, `click`

**Optional:**
- `pip install velocirag[ner]` — GLiNER for encoder-based entity extraction (zero hallucination NER)
- `pip install velocirag[graph]` — networkx, scikit-learn for advanced graph analysis
- `pip install velocirag[dev]` — pytest, ruff for development

## Testing

```bash
pytest tests/ -x -q                    # Full suite
pytest tests/test_store.py -x -q       # Just vector store
pytest tests/test_unified.py -x -q     # Just unified search
pytest tests/ -k "not incremental"     # Skip known flaky mtime tests
```

**Known flaky tests:** `test_add_directory_incremental`, `test_add_directory_modified_file`, `test_add_directory_deleted_file` — mtime-based file cache tests that fail on fast filesystems.

## Development Patterns

- **SQLite connections:** Always use `self._connect()` context manager (properly closes). Never `with sqlite3.connect() as conn:` (leaks FDs in long-running processes).
- **Embedding model:** Singleton via `Embedder()`. Lazy-loads on first `.embed()` call.
- **FTS5 queries:** Wrap search terms in quotes to escape special chars. Use try/except — MATCH can throw on syntax.
- **Graph analyzers:** Each implements `analyze(nodes) → (new_nodes, new_edges)`. Add new ones by subclassing the pattern in `analyzers.py`.
- **GLiNER:** Optional import — wrap in try/except ImportError. Model is ~170MB, lazy-loaded. Max ~512 tokens per input, chunk longer texts.

## File Layout

```
velocirag/
├── src/velocirag/       # Source modules
│   ├── __init__.py      # Public API exports
│   ├── store.py         # VectorStore (SQLite + FAISS + FTS5)
│   ├── searcher.py      # Search orchestration
│   ├── unified.py       # 4-layer fusion (the main search entry point)
│   ├── graph.py         # GraphStore + GraphQuerier
│   ├── analyzers.py     # 6 graph analyzers + GLiNERAnalyzer
│   ├── pipeline.py      # GraphPipeline (10-stage build)
│   ├── metadata.py      # MetadataStore
│   ├── embedder.py      # Sentence-transformer wrapper
│   ├── reranker.py      # Cross-encoder reranking
│   ├── chunker.py       # Markdown header-aware chunking
│   ├── rrf.py           # Reciprocal Rank Fusion
│   ├── variants.py      # Query variant generation
│   ├── frontmatter.py   # YAML frontmatter parsing
│   ├── tracker.py       # Usage tracking
│   ├── abstracts.py     # L0/L1 abstract generation
│   └── cli.py           # Click CLI
├── tests/               # 18 test files, pytest
├── examples/            # basic_search.py, unified_search.py
├── docs/internal/       # Design specs (internal dev docs)
├── pyproject.toml       # Package config
├── README.md            # User-facing documentation
└── LICENSE              # PolyForm Noncommercial 1.0.0
```
