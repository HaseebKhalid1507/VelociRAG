# Contributing to VelociRAG

Thanks for your interest in contributing! Here's how to get started.

## Quick Start

```bash
git clone https://github.com/HaseebKhalid1507/VelociRAG.git
cd VelociRAG
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -q
```

## Development Setup

- **Python 3.11+** required
- **ONNX Runtime** is the only ML dependency — no PyTorch
- Run `ruff check src/` for linting
- Run `pytest tests/ -q` for tests

## How to Contribute

### Bug Reports
Use the [bug report template](https://github.com/HaseebKhalid1507/VelociRAG/issues/new?template=bug_report.md). Include:
- VelociRAG version (`velocirag --version`)
- Python version
- Steps to reproduce
- Expected vs actual behavior

### Feature Requests
Use the [feature request template](https://github.com/HaseebKhalid1507/VelociRAG/issues/new?template=feature_request.md) or start a [Discussion](https://github.com/HaseebKhalid1507/VelociRAG/discussions).

### Pull Requests
1. Fork the repo and create a branch from `main`
2. Write tests for new functionality
3. Run the full test suite: `pytest tests/ -q`
4. Run linting: `ruff check src/`
5. Keep PRs focused — one feature or fix per PR
6. Update documentation if needed

### Code Style
- Follow existing patterns in the codebase
- Use type hints
- Add docstrings to public functions
- `ruff` handles formatting — run it before committing

## Architecture Overview

```
src/velocirag/
├── store.py          # VectorStore — FAISS + SQLite storage
├── embedder.py       # ONNX embedding (MiniLM-L6-v2)
├── searcher.py       # Search pipeline + reranking
├── unified.py        # UnifiedSearch — 4-layer fusion + RRF
├── graph.py          # Knowledge graph (SQLite-backed)
├── analyzers.py      # GLiNER entity extraction + 6 analyzers
├── chunker.py        # Markdown chunking
├── metadata.py       # Metadata store + filtering
├── reranker.py       # Cross-encoder reranking (ONNX)
├── rrf.py            # Reciprocal Rank Fusion
├── variants.py       # Query expansion
├── cli.py            # Click CLI
├── daemon.py         # Unix socket search daemon
└── mcp_server.py     # MCP server
```

## Design Principles

- **No PyTorch.** All inference via ONNX Runtime.
- **SQLite everywhere.** No external databases.
- **Incremental.** Indexing only touches changed files.
- **Fast.** Sub-200ms warm search is the target.

## Questions?

Open a [Discussion](https://github.com/HaseebKhalid1507/VelociRAG/discussions) or check existing issues.
