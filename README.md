# 🦖 Velocirag

**Progressive multi-layer RAG engine for markdown knowledge bases.**

Point it at a folder of markdown files. Get sub-second semantic search with cross-encoder reranking, progressive L0/L1/L2 retrieval, and optional knowledge graph enrichment.

## Features

- **Header-aware chunking** — splits markdown by `##`/`###` headers, preserves parent context
- **Progressive search** — L0 abstracts → L1 overviews → L2 full content → cross-encoder rerank
- **Query variant generation** — handles `CS656` ↔ `CS 656`, acronyms, case variations
- **RRF fusion** — combines results across query variants using Reciprocal Rank Fusion
- **Cross-encoder reranking** — TinyBERT reranker for actual relevance scoring (not just cosine)
- **Knowledge graph** *(optional)* — wiki-link parsing, entity extraction, temporal/semantic/topic analysis
- **Warm daemon** — keeps models loaded for <1s queries (~740MB)
- **8GB friendly** — runs on constrained hardware, no GPU required

## Quick Start

```bash
pip install velocirag

# Index a folder of markdown files
velocirag index ./my-notes/

# Search
velocirag search "machine learning fundamentals"
```

## License

[PolyForm Noncommercial 1.0.0](LICENSE) — free for personal, research, and non-commercial use.
