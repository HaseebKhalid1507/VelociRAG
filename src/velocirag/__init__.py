"""Velocirag — Progressive multi-layer RAG engine for markdown knowledge bases."""

__version__ = "0.4.0"

from .chunker import chunk_markdown
from .variants import generate_variants
from .rrf import reciprocal_rank_fusion
from .embedder import Embedder
from .store import VectorStore
from .searcher import Searcher
from .reranker import Reranker
from .graph import GraphStore, GraphQuerier, Node, Edge, NodeType, RelationType
from .analyzers import (
    ExplicitAnalyzer, EntityAnalyzer, TemporalAnalyzer,
    TopicAnalyzer, SemanticAnalyzer, CentralityAnalyzer
)

# Optional GLiNER analyzers
try:
    from .analyzers import GLiNERAnalyzer, RelationAnalyzer
except ImportError:
    GLiNERAnalyzer = None
    RelationAnalyzer = None
from .pipeline import GraphPipeline
from .unified import UnifiedSearch
from .abstracts import AbstractGenerator, AbstractResult
from .metadata import MetadataStore
from .frontmatter import parse_frontmatter, extract_tags_from_content, extract_wiki_links
from .tracker import UsageTracker
