"""VelociRAG — Lightning-fast RAG for AI agents."""

__version__ = "0.5.0"

from .chunker import chunk_markdown as chunk_markdown
from .variants import generate_variants as generate_variants
from .rrf import reciprocal_rank_fusion as reciprocal_rank_fusion
from .embedder import Embedder as Embedder
from .store import VectorStore as VectorStore
from .searcher import Searcher as Searcher
from .reranker import Reranker as Reranker
from .graph import GraphStore as GraphStore
from .graph import GraphQuerier as GraphQuerier
from .graph import Node as Node
from .graph import Edge as Edge
from .graph import NodeType as NodeType
from .graph import RelationType as RelationType
from .analyzers import ExplicitAnalyzer as ExplicitAnalyzer
from .analyzers import EntityAnalyzer as EntityAnalyzer
from .analyzers import TemporalAnalyzer as TemporalAnalyzer
from .analyzers import TopicAnalyzer as TopicAnalyzer
from .analyzers import SemanticAnalyzer as SemanticAnalyzer
from .analyzers import CentralityAnalyzer as CentralityAnalyzer
from .pipeline import GraphPipeline as GraphPipeline
from .unified import UnifiedSearch as UnifiedSearch
from .metadata import MetadataStore as MetadataStore
from .frontmatter import parse_frontmatter as parse_frontmatter
from .frontmatter import extract_tags_from_content as extract_tags_from_content
from .frontmatter import extract_wiki_links as extract_wiki_links
from .tracker import UsageTracker as UsageTracker

# Optional GLiNER analyzers
try:
    from .analyzers import GLiNERAnalyzer as GLiNERAnalyzer
    from .analyzers import RelationAnalyzer as RelationAnalyzer
except ImportError:
    GLiNERAnalyzer = None  # type: ignore
    RelationAnalyzer = None  # type: ignore
