# AGENTS.md — VelociRAG for AI Coding Agents

> Machine-readable project context for AI coding assistants (Claude Code, Cursor, Copilot, etc.)

## What Is This?

VelociRAG is **lightning-fast RAG for AI agents**. Pure retrieval engine powered by ONNX Runtime with 4-layer fusion, 3ms warm embeddings, MCP server, and Unix socket daemon.

- **Language:** Python 3.10+
- **Backend:** ONNX Runtime (no PyTorch)
- **Source:** `src/velocirag/` (18 modules, ~10K lines)
- **Tests:** `tests/` (18 test files)
- **CLI:** `velocirag` (click-based)
- **License:** MIT

## Architecture

```
markdown files → chunk → embed (ONNX) → store (SQLite + FAISS)
                                            ↓
                                      4-layer search:
                                        vector (FAISS cosine, 384d MiniLM-L6-v2)
                                      + keyword (BM25 via SQLite FTS5)
                                      + graph (knowledge graph traversal)
                                      + metadata (structured SQL filters)
                                            ↓
                                      RRF fusion → cross-encoder rerank → results
```

**Three ways to query:**
1. **MCP server** (`velocirag mcp`) — for AI agents via Model Context Protocol
2. **Daemon** (`velocirag serve`) — warm engine over Unix socket for CLI users
3. **Direct** (`velocirag search`) — cold search, no daemon needed

## Module Map

| Module | Lines | Purpose |
|--------|-------|---------|
| `cli.py` | 1524 | Click CLI — index, search, serve, stop, status, mcp, health, query, reindex |
| `analyzers.py` | 1063 | 7 graph analyzers + FAISS semantic (not O(n²)) + sampled centrality. GLiNER NER optional. |
| `graph.py` | 915 | Knowledge graph — Node/Edge models, GraphStore (SQLite), GraphQuerier |
| `unified.py` | 893 | 4-layer fusion search orchestrator — vector + keyword + metadata + graph → RRF. Filename cache. |
| `store.py` | 870 | Vector storage — SQLite + FAISS + FTS5. Batched rebuild for large corpora. |
| `metadata.py` | 695 | Metadata store — frontmatter, tags, cross-refs, usage tracking |
| `searcher.py` | 681 | High-level search — query variants, batch FAISS, RRF fusion, caching |
| `pipeline.py` | 638 | 10-stage graph build pipeline. Memory-safe: frees content + model after Stage 7. |
| `embedder.py` | 527 | ONNX Runtime embeddings (all-MiniLM-L6-v2, 384d). 3ms warm, 184ms cold. |
| `mcp_server.py` | 498 | FastMCP server — 5 tools (search, index, add_document, health, list_sources) |
| `daemon.py` | 437 | Unix socket search daemon — warm engine, auto-detected by CLI |
| `tracker.py` | 308 | Usage tracking — search hits, reads, access patterns |
| `variants.py` | 217 | Query variant generation + acronym registry + question rewrite |
| `reranker.py` | 186 | Cross-encoder reranking (TinyBERT). Optional: `pip install velocirag[reranker]` |
| `frontmatter.py` | 172 | YAML frontmatter parser, tag extraction, wiki-link extraction |
| `chunker.py` | 158 | Markdown chunking by headers with parent context preservation |
| `rrf.py` | 145 | Reciprocal Rank Fusion implementation |

## Key Classes

```python
# Index documents
embedder = Embedder()  # ONNX Runtime, lazy model download
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

# Daemon client
from velocirag.daemon import daemon_search, daemon_ping
if daemon_ping():
    results = daemon_search("query", limit=5)
```

## CLI Commands

```bash
velocirag index <path> [--db PATH] [--graph] [--metadata] [--gliner] [--force]
velocirag search <query> [--db PATH] [--limit N] [--threshold F] [--format text|json]
velocirag serve [--db PATH] [-f]       # start search daemon
velocirag stop                          # stop daemon
velocirag status                        # daemon health
velocirag mcp [--db PATH] [--transport stdio|sse]  # MCP server
velocirag query [--tags TAG] [--status S] [--category C] [--project P]
velocirag health [--db PATH] [--format text|json]
velocirag reindex [--db PATH]
```

## Data Files

When you index, VelociRAG creates these in the `--db` directory:

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

**Required (base install — ~54MB, no PyTorch):**
- `onnxruntime` — ONNX model inference
- `tokenizers` — HuggingFace fast tokenizers
- `huggingface-hub` — model downloads
- `faiss-cpu` — vector similarity search
- `numpy`, `python-frontmatter`, `pyyaml`, `click`

**Optional:**
- `pip install velocirag[reranker]` — cross-encoder reranking (adds sentence-transformers + torch)
- `pip install velocirag[mcp]` — MCP server (adds fastmcp)
- `pip install velocirag[ner]` — GLiNER entity extraction
- `pip install velocirag[graph]` — networkx, scikit-learn for advanced graph analysis
- `pip install velocirag[dev]` — pytest, ruff for development

## Testing

```bash
pytest tests/ -x -q                    # Full suite
pytest tests/test_mcp.py -x -q        # MCP server tests
pytest tests/test_store.py -x -q       # Vector store
pytest tests/test_unified.py -x -q     # Unified search
pytest tests/ -k "not incremental"     # Skip known flaky mtime tests
```

## Development Patterns

- **Embedding backend:** ONNX Runtime via `Embedder()`. Downloads `optimum/all-MiniLM-L6-v2` on first use to `~/.cache/velocirag/models/`. No PyTorch needed.
- **SQLite connections:** Always use `self._connect()` context manager (properly closes). Never `with sqlite3.connect() as conn:` (leaks FDs in long-running processes).
- **FTS5 queries:** Strip FTS5 operators, keep Unicode, wrap in quotes. Use try/except — MATCH can throw on syntax.
- **Graph OOM safety:** SemanticAnalyzer uses FAISS top-K (not O(n²) pairwise). TemporalAnalyzer caps at 50K edges. CentralityAnalyzer samples 500 BFS sources. Pipeline frees content + embedder after Stage 7. VectorStore freed before graph build in CLI.
- **Reranker:** Optional dependency. Falls back to unranked results if sentence-transformers not installed.
- **MCP server:** Thread-safe lazy init with double-checked locking. Logger warnings on component failures (no silent swallowing).
- **Daemon:** Single worker thread owns SQLite/FAISS. Connection threads submit to queue. Length-prefixed JSON over Unix socket.
- **GLiNER:** Optional import — wrap in try/except ImportError. Model is ~170MB, lazy-loaded. Max ~512 tokens per input, chunk longer texts.

## File Layout

```
velocirag/
├── src/velocirag/       # Source modules (18 files, ~10K lines)
│   ├── __init__.py      # Public API exports
│   ├── store.py         # VectorStore (SQLite + FAISS + FTS5)
│   ├── searcher.py      # Search orchestration
│   ├── unified.py       # 4-layer fusion (the main search entry point)
│   ├── graph.py         # GraphStore + GraphQuerier
│   ├── analyzers.py     # 7 graph analyzers + GLiNERAnalyzer
│   ├── pipeline.py      # GraphPipeline (10-stage build)
│   ├── metadata.py      # MetadataStore
│   ├── embedder.py      # ONNX Runtime embedding engine
│   ├── reranker.py      # Cross-encoder reranking (optional)
│   ├── daemon.py        # Unix socket search daemon
│   ├── mcp_server.py    # MCP server for AI agents
│   ├── chunker.py       # Markdown header-aware chunking
│   ├── rrf.py           # Reciprocal Rank Fusion
│   ├── variants.py      # Query variant generation
│   ├── frontmatter.py   # YAML frontmatter parsing
│   ├── tracker.py       # Usage tracking
│   └── cli.py           # Click CLI
├── tests/               # 18 test files, pytest
├── examples/            # basic_search.py, unified_search.py
├── pyproject.toml       # Package config
├── README.md            # User-facing documentation
└── LICENSE              # MIT
```
