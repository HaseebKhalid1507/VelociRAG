# 🦖 VelociRAG

**Lightning-fast RAG for AI agents.**

_Four-layer retrieval fusion powered by ONNX Runtime. 54MB install, sub-200ms warm search, no PyTorch. MCP-ready._

---

Most RAG solutions either drag in 750MB of PyTorch or limit you to single-layer vector search. VelociRAG gives you four retrieval methods — vector similarity, BM25 keyword matching, knowledge graph traversal, and metadata filtering — fused through reciprocal rank fusion with optional cross-encoder reranking. All running on ONNX Runtime, no GPU, no API keys. Comes with an MCP server for agent integration, a Unix socket daemon for warm queries, and a CLI that just works.

## 🚀 Quick Start

### MCP Server (Claude, Cursor, Windsurf)

```bash
pip install "velocirag[mcp]"
velocirag index ./my-docs --graph --metadata
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
velocirag index ./my-docs --graph --metadata
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
| **LLM required for search** | ❌ | ⚠️ | ⚠️ | ❌ | ❌ |
| **MCP server** | ✅ | ❌ | ❌ | ❌ | ✅ |
| **GPU required** | ❌ | ❌ | ❌ | ❌ | ❌ |
| **PyTorch required** | ❌ | ✅ | ✅ | ❌ | ❌ |
| **Install size** | ~54MB | ~750MB+ | ~750MB+ | ~50MB | ~30MB |
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

- **ONNX Runtime** — 184ms cold start, 3ms cached. 54MB install — no PyTorch, no GPU
- **Four-layer fusion** — FAISS vector similarity + SQLite FTS5 (BM25) + knowledge graph + metadata filtering, merged via reciprocal rank fusion
- **Cross-encoder reranking** — Optional TinyBERT reranker with score blending (`pip install velocirag[reranker]`)
- **MCP server** — Five tools (search, index, add_document, health, list_sources) for Claude, Cursor, Windsurf
- **Search daemon** — Unix socket server keeps ONNX model + FAISS index warm between queries
- **Knowledge graph** — Seven analyzers build entity, temporal, topic, and explicit-link edges from markdown. Optional GLiNER NER. 680 files in 3.8s
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
pipeline.build('./docs')

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

## 💻 CLI Reference

```bash
# Index documents with all layers
velocirag index <path> [--graph] [--metadata] [--gliner] [--light-graph] [--force]

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

## 📊 Performance

Real benchmarks from production deployment (3,416 documents, ONNX Runtime, v0.5.0):

| Metric | Value |
|--------|-------|
| **Embedding (warm)** | **3ms** |
| **Embedding (cold)** | **184ms** |
| **Full 4-layer search (warm)** | **76–350ms** |
| **Graph build (680 files, --light-graph)** | **3.8s** |
| **Graph build (7K files, --light-graph)** | **~90s** (no OOM on 8GB) |
| **Hit rate (100-query benchmark)** | **99/100** |
| **Install size** | **~54MB** (no PyTorch) |
| **RAM usage** | **<1GB** with ONNX models |
| **Graph** | 4,837 nodes, 11,443 edges |

## ⚙️ Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `VELOCIRAG_DB` | `./.velocirag` | Database directory |
| `VELOCIRAG_SOCKET` | `/tmp/velocirag-daemon.sock` | Daemon socket path |
| `NO_COLOR` | — | Disable colored output |

**Dependencies:**
- **Base:** `onnxruntime`, `tokenizers`, `huggingface-hub`, `faiss-cpu`, `numpy`, `click`
- **Reranker:** `pip install velocirag[reranker]` (adds sentence-transformers)
- **MCP:** `pip install velocirag[mcp]` (adds fastmcp)
- **NER:** `pip install velocirag[ner]` (adds GLiNER)
- **Graph:** `pip install velocirag[graph]` (adds networkx, scikit-learn)

## 📄 License

[MIT](LICENSE) — Use it anywhere, build anything.

**Need agent integration help?** Check [AGENTS.md](AGENTS.md) for machine-readable project context.

---

_Built because every agent deserves a search engine that doesn't need a GPU._
