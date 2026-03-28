# 🦖 VelociRAG

**Lightning-fast RAG for AI agents.**

_Four-layer retrieval fusion powered by ONNX Runtime. No PyTorch. Sub-200ms warm search. Incremental graph updates. MCP-ready._

---

Most RAG solutions either drag in 2GB+ of PyTorch or limit you to single-layer vector search. VelociRAG gives you four retrieval methods — vector similarity, BM25 keyword matching, knowledge graph traversal, and metadata filtering — fused through reciprocal rank fusion with cross-encoder reranking. All running on ONNX Runtime, no GPU, no API keys. Comes with an MCP server for agent integration, a Unix socket daemon for warm queries, and a CLI that just works.

## 🚀 Quick Start

### MCP Server (Claude, Cursor, Windsurf)

```bash
pip install "velocirag[mcp]"
velocirag index ./my-docs
velocirag mcp
```

**Claude Code** — add to `.mcp.json` in your project root:
```json
{
  "mcpServers": {
    "velocirag": {
      "command": "velocirag",
      "args": ["mcp"],
      "env": { "VELOCIRAG_DB": "/path/to/data" }
    }
  }
}
```
Then open `/mcp` in Claude Code and enable the `velocirag` server. If using a virtualenv, use the full path to the binary (e.g. `.venv/bin/velocirag`).

**Claude Desktop** — add to `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "velocirag": {
      "command": "velocirag",
      "args": ["mcp", "--db", "/path/to/data"]
    }
  }
}
```

**Cursor** — add to `.cursor/mcp.json`:
```json
{
  "mcpServers": {
    "velocirag": {
      "command": "velocirag",
      "args": ["mcp", "--db", "/path/to/data"]
    }
  }
}
```

### Python API

```python
from velocirag import Embedder, VectorStore, Searcher

embedder = Embedder()
store = VectorStore('./my-db', embedder)
store.add_directory('./my-docs')
searcher = Searcher(store, embedder)
results = searcher.search('query', limit=5)
```

### CLI

```bash
pip install velocirag
velocirag index ./my-docs
velocirag search "your query here"
```

### Search Daemon (warm engine for CLI users)

```bash
velocirag serve --db ./my-data        # start daemon (background)
velocirag search "query"              # auto-routes through daemon
velocirag status                      # check daemon health
velocirag stop                        # stop daemon
```

The daemon keeps the ONNX model + FAISS index warm over a Unix socket. First query loads the engine (~1s), subsequent queries return in ~180ms with full 4-layer fusion.

## 🎯 Why VelociRAG?

| | VelociRAG | LangChain | LlamaIndex | Chroma | mcp-local-rag |
|---|:---:|:---:|:---:|:---:|:---:|
| **Search layers** | 4 | 2 | 2 | 1 | 2 |
| **Cross-encoder reranking** | ✅ | ❌ | ✅ | ❌ | ❌ |
| **Knowledge graph** | ✅ | ❌ | ✅ | ❌ | ❌ |
| **Incremental updates** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **LLM required for search** | ❌ | ⚠️ | ⚠️ | ❌ | ❌ |
| **MCP server** | ✅ | ❌ | ❌ | ❌ | ✅ |
| **GPU required** | ❌ | ❌ | ❌ | ❌ | ❌ |
| **PyTorch required** | ❌ | ✅ | ✅ | ❌ | ❌ |
| **Install size** | ~80MB | ~750MB+ | ~750MB+ | ~50MB | ~30MB |
| **Warm search latency** | ~3ms | — | — | ~50ms | ~200ms |

## 🏗️ How It Works

**The 4-layer pipeline:**
```
Query → expand (acronyms, variants)
      → [Vector]   FAISS cosine similarity (384d, MiniLM-L6-v2 via ONNX)
      → [Keyword]  BM25 via SQLite FTS5
      → [Graph]    Knowledge graph traversal
      → [Metadata] Structured SQL filters (tags, status, project)
      → RRF Fusion → Cross-encoder rerank → Results
```

**What each layer catches:**

| Query type | Vector | Keyword | Graph | Metadata |
|-----------|:---:|:---:|:---:|:---:|
| Conceptual ("improve error handling") | ✅ | — | — | — |
| Exact match ("ERR_CONNECTION_REFUSED") | — | ✅ | — | — |
| Connected concepts | — | — | ✅ | — |
| Filtered ("#python status:active") | — | — | — | ✅ |
| Combined ("React state management") | ✅ | ✅ | ✅ | ✅ |

## ✨ Features

- **ONNX Runtime** — 184ms cold start, 3ms cached. No PyTorch, no GPU
- **Four-layer fusion** — FAISS vector similarity + SQLite FTS5 (BM25) + knowledge graph + metadata filtering, merged via reciprocal rank fusion
- **Cross-encoder reranking** — TinyBERT reranker via ONNX Runtime — included in base install, no PyTorch needed. Downloads ~17MB model on first use
- **Incremental graph updates** — file-centric provenance tracking detects what changed and only rebuilds affected nodes/edges. Cascading deletes maintain consistency across all stores (vector, graph, metadata). Multi-source support with isolated provenance per source
- **MCP server** — Five tools (search, index, add_document, health, list_sources) for Claude, Cursor, Windsurf
- **Search daemon** — Unix socket server keeps ONNX model + FAISS index warm between queries
- **Knowledge graph** — Analyzers build entity, temporal, topic, and explicit-link edges from markdown. Optional GLiNER NER. 418 files in 2.1s
- **Smart chunking** — Header-aware splitting preserves document structure and parent context
- **Query expansion** — Acronym registry, casing/spacing variants, underscore-aware tokenization
- **Runs anywhere** — CPU-only, 8GB RAM, no API keys, no external services

## 🤖 MCP Server

VelociRAG exposes a Model Context Protocol server for seamless agent integration:

**Available tools:**
- `search` — 4-layer fusion search with reranking
- `index` — Add documents to the knowledge base
- `add_document` — Insert single document
- `health` — System diagnostics
- `list_sources` — Show indexed document sources

The MCP server process stays alive between queries, so models load once and every subsequent search is warm. Works with any MCP-compatible client.

## 🐍 Python API

**Full 4-layer unified search:**
```python
from velocirag import (
    Embedder, VectorStore, Searcher,
    GraphStore, MetadataStore, UnifiedSearch,
    GraphPipeline
)

# Build the full stack
embedder = Embedder()
store = VectorStore('./search-db', embedder)
graph_store = GraphStore('./search-db/graph.db')
metadata_store = MetadataStore('./search-db/metadata.db')

# Index with graph + metadata
store.add_directory('./docs')
pipeline = GraphPipeline(graph_store, embedder, metadata_store)
pipeline.build('./docs', source_name='my-docs')

# Unified search across all layers
searcher = Searcher(store, embedder)
unified = UnifiedSearch(searcher, graph_store, metadata_store)
results = unified.search(
    'machine learning algorithms',
    limit=5,
    enrich_graph=True,
    filters={'tags': ['python'], 'status': 'active'}
)
```

**Quick semantic search:**
```python
from velocirag import Embedder, VectorStore, Searcher

embedder = Embedder()
store = VectorStore('./db', embedder)
store.add_directory('./docs')
searcher = Searcher(store, embedder)
results = searcher.search('neural networks', limit=10)
```

**Incremental graph updates:**
```python
from velocirag import Embedder, GraphStore, GraphPipeline

# First run — full build, populates provenance
gs = GraphStore('./db/graph.db')
pipeline = GraphPipeline(gs, embedder=Embedder())
pipeline.build('./docs', source_name='my-docs')  # full build

# Subsequent runs — only changed files get reprocessed
pipeline.build('./docs', source_name='my-docs')  # incremental (automatic)

# Force full rebuild
pipeline.build('./docs', source_name='my-docs', force_rebuild=True)

# Multi-source graphs
pipeline.build('./project-a', source_name='project-a')
pipeline.build('./project-b', source_name='project-b')  # isolated provenance

# Deleted files automatically cascade across all stores
# (vector, FTS5, graph, metadata) on next build
```

## 💻 CLI Reference

```bash
# Index documents (graph + metadata built by default)
velocirag index <path> [--no-graph] [--no-metadata] [--gliner] [--full-graph] [--force]
                       [--source NAME] [--db PATH]

# Search across all layers (auto-routes through daemon if running)
velocirag search <query> [--limit N] [--threshold F] [--format text|json]

# Search daemon
velocirag serve [--db PATH] [-f]         # start daemon (-f for foreground)
velocirag stop                            # stop daemon
velocirag status                          # check daemon health

# Metadata queries
velocirag query [--tags TAG] [--status S] [--project P] [--recent N]

# System health and status
velocirag health [--format text|json]

# Start MCP server
velocirag mcp [--db PATH] [--transport stdio|sse]
```

**Options:**
- `--no-graph` — Skip knowledge graph build
- `--no-metadata` — Skip metadata extraction
- `--full-graph` — Build graph WITH semantic similarity edges (~2GB extra RAM)
- `--source NAME` — Label for multi-source provenance isolation
- `--force` — Clear and rebuild from scratch
- `--gliner` — Use GLiNER for entity extraction (requires `pip install "velocirag[ner]"`)

## 📊 Performance

Real benchmarks on [ByteByteGo/system-design-101](https://github.com/ByteByteGoHq/system-design-101) (418 files, 1,001 chunks):

| Metric | Value |
|--------|-------|
| **Index (418 files)** | **13.6s** |
| **Search (warm, 5 results)** | **35–90ms** |
| **Graph build (light)** | **2.1s** → 2,397 nodes, 8,717 edges |
| **Incremental update (1 file)** | **1.3s** |
| **Reranker** | Cross-encoder TinyBERT via ONNX |
| **Install size** | ~80MB (no PyTorch) |
| **RAM usage** | <1GB with all models loaded |

Production deployment (3,400+ documents, 4 sources):

| Metric | Value |
|--------|-------|
| **Embedding (warm)** | **3ms** |
| **Embedding (cold)** | **184ms** |
| **Full 4-layer search (warm)** | **76–350ms** |
| **Hit rate (100-query benchmark)** | **99/100** |
| **Graph** | 3,740 nodes, 92,719 edges |
| **Provenance** | 870 files across 4 sources |

## ⚙️ Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `VELOCIRAG_DB` | `./.velocirag` | Database directory |
| `VELOCIRAG_SOCKET` | `/tmp/velocirag-daemon.sock` | Daemon socket path |
| `NO_COLOR` | — | Disable colored output |

**Dependencies (all included in base install):**
- `onnxruntime` — ONNX inference (embedder + reranker)
- `tokenizers` + `huggingface-hub` — model loading
- `faiss-cpu` — vector similarity search
- `networkx` + `scikit-learn` — knowledge graph + topic clustering
- `numpy`, `click`, `pyyaml`, `python-frontmatter`

**Optional extras:**
- `pip install "velocirag[mcp]"` — MCP server (adds `fastmcp`)
- `pip install "velocirag[ner]"` — GLiNER entity extraction (adds `gliner`, requires PyTorch)

## 📄 License

[MIT](LICENSE) — Use it anywhere, build anything.

**Need agent integration help?** Check [AGENTS.md](AGENTS.md) for machine-readable project context.

---

_Built for agents who think fast and remember faster._

<!-- mcp-name: io.github.HaseebKhalid1507/velocirag -->
