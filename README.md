# 🦖 VelociRAG

**Lightning-fast RAG for AI agents.**

_RAG engine, not a RAG framework. 4-layer retrieval fusion in 80ms. No LLM in the search path. MCP-ready._

---

Every agent framework needs retrieval. Most RAG solutions are heavy orchestration frameworks (LangChain, LlamaIndex) that bundle their own agent loop. VelociRAG is the opposite — a **pure retrieval engine** that any agent can plug into. Four-layer fusion (vector + BM25 + graph + metadata), cross-encoder reranking, 80ms warm queries, and zero API dependencies. It's a search engine that speaks agent.

## 🚀 Quick Start

### MCP Server (Flagship)

```bash
pip install "velocirag[mcp]"
velocirag index ./my-docs --graph --metadata
velocirag mcp
```

**Claude Desktop config:**
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

**Cursor config:**
```json
{
  "mcp": {
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

## 🎯 Why VelociRAG?

| | VelociRAG | LangChain | LlamaIndex | Chroma | mcp-local-rag |
|---|:---:|:---:|:---:|:---:|:---:|
| **Search layers** | 4 | 2 | 2 | 1 | 2 |
| **Cross-encoder reranking** | ✅ | ❌ | ✅ | ❌ | ❌ |
| **Knowledge graph** | ✅ | ❌ | ✅ | ❌ | ❌ |
| **LLM required for search** | ❌ | ⚠️ | ⚠️ | ❌ | ❌ |
| **MCP server** | ✅ | ❌ | ❌ | ❌ | ✅ |
| **GPU required** | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Warm search latency** | ~80ms | — | — | ~50ms | ~200ms |

## 🏗️ How It Works

**The 4-layer pipeline:**
```
Query → expand (acronyms, variants)
      → [Vector]   FAISS cosine similarity (384d, MiniLM-L6-v2)
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

- **Sub-second search** — 80ms warm, CPU-only, runs on 8GB RAM
- **Four-layer fusion** — Vector + BM25 + graph + metadata → RRF
- **Cross-encoder reranking** — TinyBERT, auto-initialized
- **MCP server for AI agents** — Claude Desktop, Cursor, Windsurf, etc.
- **Knowledge graph** — 7 analyzers, optional GLiNER NER
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

Models stay warm after first query (80ms average response time). Compatible with any MCP-enabled agent framework.

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
velocirag index <path> [--graph] [--metadata] [--gliner] [--force]

# Search across all layers
velocirag search <query> [--limit N] [--threshold F] [--format text|json]

# Metadata queries
velocirag query [--tags TAG] [--status S] [--project P] [--recent N]

# System health and status
velocirag health [--format text|json]
velocirag status

# Start MCP server
velocirag mcp [--db PATH] [--transport stdio|sse]
```

## 📊 Performance

Real benchmarks from production deployment (3,357 documents):

| Metric | Value |
|--------|-------|
| **Average query time (warm)** | **80ms** |
| **p50 / p95 / max** | 90ms / 198ms / 328ms |
| **Cold start** | ~3s |
| **Hit rate (100-query benchmark)** | **99/100** |
| **RAM usage with all models** | <8GB |
| **Graph nodes/edges** | 1,336 nodes, 16,818 edges |

## ⚙️ Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `VELOCIRAG_DB` | `./data` | Database directory |
| `NO_COLOR` | `false` | Disable colored output |

**Model Configuration:**
- **Embedding:** all-MiniLM-L6-v2 (384d)
- **Cross-encoder:** TinyBERT-L-2-v2 (~17MB)
- **Entity extraction:** GLiNER-small-v2.1 (~170MB, optional)
- **Similarity threshold:** 0.3 (configurable)

## 📄 License

[MIT](LICENSE) — Use it anywhere, build anything.

**Need agent integration help?** Check [AGENTS.md](AGENTS.md) for machine-readable project context.

---

_Built for agents who think fast and search faster._