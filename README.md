# 🦖 VelociRAG

**Lightning-fast RAG for AI agents.**

_RAG engine, not a RAG framework. 4-layer retrieval fusion in under 3ms. No LLM in the search path. ONNX-powered. MCP-ready._

---

Every agent framework needs retrieval. Most RAG solutions are heavy orchestration frameworks (LangChain, LlamaIndex) that bundle their own agent loop and drag in 750MB of PyTorch. VelociRAG is the opposite — a **pure retrieval engine** powered by ONNX Runtime. Four-layer fusion (vector + BM25 + graph + metadata), cross-encoder reranking, 3ms warm queries, 54MB install. No torch. No API keys. Just fast search.

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

- **Sub-second search** — 3ms warm embeddings, 180ms full 4-layer fusion via daemon
- **ONNX Runtime** — No PyTorch, no GPU. 54MB install, 16x faster than sentence-transformers
- **Four-layer fusion** — Vector + BM25 + graph + metadata → RRF
- **Cross-encoder reranking** — TinyBERT, optional (`pip install velocirag[reranker]`)
- **MCP server** — Claude Desktop, Cursor, Windsurf, Claude Code
- **Search daemon** — Unix socket, warm engine, auto-detected by CLI
- **Knowledge graph** — 7 analyzers, optional GLiNER NER, 680 files in 3.8s, scales to 7K+
- **Header-aware markdown chunking** — Preserves document structure
- **Smart query expansion** — Acronyms, variants, question rewrite
- **No GPU, no API keys** — Pure CPU, zero external dependencies

## 🤖 MCP Server

VelociRAG exposes a Model Context Protocol server for seamless agent integration:

**Available tools:**
- `search` — 4-layer fusion search with reranking
- `index` — Add documents to the knowledge base
- `add_document` — Insert single document
- `health` — System diagnostics
- `list_sources` — Show indexed document sources

Models stay warm after first query. Thread-safe initialization for concurrent access. Compatible with any MCP-enabled agent framework.

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

Real benchmarks from production deployment (3,153 documents, ONNX Runtime):

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
| **Graph** | 3,478 nodes, 11,834 edges |

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

_Built for agents who think fast and search faster._
