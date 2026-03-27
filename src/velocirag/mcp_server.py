"""
Velocirag MCP Server - Model Context Protocol integration for AI agents.

Wraps the Velocirag RAG engine as MCP tools for seamless AI agent integration.
Supports Claude Desktop, Cursor, Windsurf, pi, and any MCP-compatible client.

Five core tools:
- search: Four-layer fusion search (vector + BM25 + graph + metadata)
- index: Build knowledge base from markdown directories
- add_document: Index single files on-the-fly
- health: Engine status and component health
- list_sources: Show indexed document sources
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastmcp import FastMCP

# Suppress noisy output during imports
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)

# Global engine state for lazy initialization
_engine: Dict[str, Any] = {}

def _get_db_path() -> Path:
    """Get database path from environment or default."""
    env_path = os.environ.get('VELOCIRAG_DB')
    if env_path:
        return Path(env_path).expanduser().absolute()
    return Path('.velocirag').absolute()

def _init_engine() -> None:
    """Lazily initialize the Velocirag engine on first use."""
    if _engine.get('initialized'):
        return
    
    from .embedder import Embedder
    from .store import VectorStore
    from .searcher import Searcher
    from .unified import UnifiedSearch
    from .graph import GraphStore
    from .metadata import MetadataStore
    from .pipeline import GraphPipeline
    
    db_path = _get_db_path()
    db_path.mkdir(parents=True, exist_ok=True)
    
    # Core components
    _engine['db_path'] = db_path
    _engine['embedder'] = Embedder()
    _engine['store'] = VectorStore(str(db_path), _engine['embedder'])
    _engine['searcher'] = Searcher(_engine['store'], _engine['embedder'])
    
    # Optional components (check if databases exist)
    graph_db = db_path / "graph.db"
    metadata_db = db_path / "metadata.db"
    
    _engine['graph_store'] = None
    _engine['metadata_store'] = None
    
    if graph_db.exists():
        try:
            _engine['graph_store'] = GraphStore(str(graph_db))
        except Exception:
            pass  # Graceful degradation
    
    if metadata_db.exists():
        try:
            _engine['metadata_store'] = MetadataStore(str(metadata_db))
        except Exception:
            pass  # Graceful degradation
    
    # Initialize unified search
    _engine['unified_search'] = UnifiedSearch(
        searcher=_engine['searcher'],
        graph_store=_engine['graph_store'],
        metadata_store=_engine['metadata_store']
    )
    
    _engine['initialized'] = True

# FastMCP server instance
mcp = FastMCP(
    name="velocirag",
    instructions="VelociRAG: Lightning-fast RAG for AI agents. Four-layer retrieval fusion (vector + BM25 + knowledge graph + metadata) with cross-encoder reranking. Use 'search' to query indexed documents, 'index' to build the knowledge base, 'health' to check status."
)

@mcp.tool()
def search(query: str, limit: int = 5, threshold: float = 0.3) -> dict:
    """
    Search indexed content using four-layer fusion (vector + BM25 + graph + metadata).
    
    The main event. Combines vector similarity, keyword matching, graph connections,
    and metadata filtering through reciprocal rank fusion for optimal results.
    
    Args:
        query: Search query string
        limit: Maximum results to return (default: 5, max: 50)
        threshold: Minimum similarity threshold (default: 0.3)
    
    Returns:
        dict: Search results with metadata and timing information
    """
    if not query.strip():
        return {
            'error': 'Query cannot be empty',
            'results': [],
            'total_results': 0,
            'search_time_ms': 0
        }
    
    # Validate parameters
    limit = max(1, min(limit, 50))
    threshold = max(0.0, min(threshold, 1.0))
    
    try:
        _init_engine()
        
        # Check if index exists
        store_stats = _engine['store'].stats()
        if store_stats['document_count'] == 0:
            return {
                'error': 'No documents indexed. Use the index tool to add documents first.',
                'results': [],
                'total_results': 0,
                'search_time_ms': 0
            }
        
        # Execute search
        start_time = time.time()
        results = _engine['unified_search'].search(
            query=query,
            limit=limit,
            threshold=threshold,
            enrich_graph=True
        )
        search_time = (time.time() - start_time) * 1000
        
        # Format results for MCP
        formatted_results = []
        for result in results.get('results', []):
            # Handle both old and new result formats
            similarity = result.get('similarity', result.get('score', 0))
            content = result.get('content', result.get('chunk', ''))
            metadata = result.get('metadata', {})
            
            formatted_result = {
                'content': content,
                'score': similarity,
                'file_path': metadata.get('file_path', result.get('doc_id', '')),
                'graph_connections': metadata.get('graph_connections', [])
            }
            formatted_results.append(formatted_result)
        
        # Determine active layers
        layer_stats = results.get('layer_stats', {})
        active_layers = [layer for layer, stats in layer_stats.items() 
                        if stats.get('candidates', 0) > 0]
        
        return {
            'results': formatted_results,
            'search_time_ms': round(search_time, 1),
            'total_results': len(formatted_results),
            'layers_active': active_layers
        }
        
    except Exception as e:
        return {
            'error': f'Search failed: {str(e)}',
            'results': [],
            'total_results': 0,
            'search_time_ms': 0
        }

@mcp.tool()
def index(directory: str, build_graph: bool = True, extract_metadata: bool = True) -> dict:
    """
    Index a directory of markdown files with optional graph and metadata extraction.
    
    Recursively processes all .md files, building vector embeddings and optionally
    constructing knowledge graphs and extracting metadata for enhanced retrieval.
    
    Args:
        directory: Path to directory containing markdown files
        build_graph: Whether to build knowledge graph (default: True)
        extract_metadata: Whether to extract metadata from frontmatter (default: True)
    
    Returns:
        dict: Indexing results and statistics
    """
    try:
        _init_engine()
        
        dir_path = Path(directory).expanduser().absolute()
        if not dir_path.exists():
            return {
                'error': f'Directory does not exist: {directory}',
                'files_processed': 0,
                'chunks_added': 0
            }
        
        if not dir_path.is_dir():
            return {
                'error': f'Path is not a directory: {directory}',
                'files_processed': 0,
                'chunks_added': 0
            }
        
        # Check for markdown files
        md_files = list(dir_path.rglob('*.md'))
        if not md_files:
            return {
                'error': f'No markdown files found in {directory}',
                'files_processed': 0,
                'chunks_added': 0
            }
        
        start_time = time.time()
        
        # Index directory with vector store
        stats = _engine['store'].add_directory(str(dir_path))
        
        result = {
            'files_processed': stats['files_processed'],
            'chunks_added': stats['chunks_added'],
            'graph_nodes': 0,
            'graph_edges': 0,
            'time_seconds': round(time.time() - start_time, 2)
        }
        
        # Build graph if requested
        if build_graph:
            try:
                from .graph import GraphStore
                from .metadata import MetadataStore
                from .pipeline import GraphPipeline
                
                graph_db = _engine['db_path'] / "graph.db"
                graph_store = GraphStore(str(graph_db))
                
                metadata_store_obj = None
                if extract_metadata:
                    metadata_db = _engine['db_path'] / "metadata.db"
                    metadata_store_obj = MetadataStore(str(metadata_db))
                
                pipeline = GraphPipeline(
                    graph_store=graph_store,
                    embedder=_engine['embedder'],
                    metadata_store=metadata_store_obj,
                    entity_extractor='regex'  # Use regex by default, not GLiNER
                )
                
                graph_stats = pipeline.build(str(dir_path), force_rebuild=False)
                
                if graph_stats.get('success'):
                    result['graph_nodes'] = graph_stats['final_nodes']
                    result['graph_edges'] = graph_stats['final_edges']
                    
                    # Re-initialize engine to pick up new graph/metadata
                    _engine['initialized'] = False
                    _init_engine()
                    
            except Exception as e:
                result['graph_error'] = f'Graph build failed: {str(e)}'
        
        return result
        
    except Exception as e:
        return {
            'error': f'Indexing failed: {str(e)}',
            'files_processed': 0,
            'chunks_added': 0
        }

@mcp.tool()
def add_document(file_path: str) -> dict:
    """
    Index a single markdown file on the fly.
    
    Reads the specified file, chunks it, and adds to the vector store.
    Useful for incremental updates without full directory reindexing.
    
    Args:
        file_path: Path to the markdown file to index
    
    Returns:
        dict: File indexing results
    """
    try:
        _init_engine()
        
        path = Path(file_path).expanduser().absolute()
        if not path.exists():
            return {
                'error': f'File does not exist: {file_path}',
                'chunks_added': 0,
                'success': False
            }
        
        if not path.suffix.lower() == '.md':
            return {
                'error': f'File is not a markdown file: {file_path}',
                'chunks_added': 0,
                'success': False
            }
        
        # Read and index the file
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Add document to store
        from .chunker import chunk_markdown
        
        # Chunk the content
        chunks = chunk_markdown(content, file_path=str(path))
        chunks_added = 0
        
        for i, chunk_data in enumerate(chunks):
            doc_id = f"{path.name}#{i}"
            chunk_content = chunk_data['content']
            chunk_metadata = chunk_data['metadata'].copy()
            chunk_metadata.update({
                'file_path': str(path),
                'chunk_index': i,
                'total_chunks': len(chunks)
            })
            _engine['store'].add(doc_id, chunk_content, chunk_metadata)
            chunks_added += 1
        
        return {
            'chunks_added': chunks_added,
            'file_path': str(path),
            'success': True
        }
        
    except Exception as e:
        return {
            'error': f'Document indexing failed: {str(e)}',
            'chunks_added': 0,
            'file_path': file_path,
            'success': False
        }

@mcp.tool()
def health() -> dict:
    """
    Engine status check and component health overview.
    
    Returns comprehensive status information about all Velocirag components
    including document counts, index health, and available features.
    
    Returns:
        dict: Complete engine health status
    """
    try:
        _init_engine()
        
        store_stats = _engine['store'].stats()
        db_path = _engine['db_path']
        
        # Check component availability
        graph_available = _engine['graph_store'] is not None
        metadata_available = _engine['metadata_store'] is not None
        
        # Get component stats
        graph_nodes = 0
        graph_edges = 0
        if graph_available:
            try:
                import sqlite3
                conn = sqlite3.connect(str(db_path / "graph.db"))
                graph_nodes = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
                graph_edges = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
                conn.close()
            except Exception:
                pass
        
        # Active layers (what search will use)
        layers = []
        if store_stats['document_count'] > 0:
            layers.extend(['vector', 'bm25'])
        if graph_available and graph_nodes > 0:
            layers.append('graph')
        if metadata_available:
            layers.append('metadata')
        
        # Get embedder info
        embedder_info = 'sentence-transformers/all-MiniLM-L6-v2'  # Default model
        if hasattr(_engine['embedder'], 'model_name'):
            embedder_info = _engine['embedder'].model_name
        
        return {
            'total_documents': store_stats['document_count'],
            'total_chunks': store_stats['faiss_vectors'],
            'index_dimensions': store_stats['dimensions'],
            'graph_nodes': graph_nodes,
            'graph_edges': graph_edges,
            'layers': layers,
            'model_name': embedder_info,
            'db_path': str(db_path),
            'components': {
                'vector_store': store_stats['consistent'],
                'graph_store': graph_available,
                'metadata_store': metadata_available,
                'unified_search': True
            }
        }
        
    except Exception as e:
        return {
            'error': f'Health check failed: {str(e)}',
            'total_documents': 0,
            'total_chunks': 0
        }

@mcp.tool()
def list_sources(limit: int = 50) -> dict:
    """
    List all indexed document sources.
    
    Shows which files have been indexed, useful for understanding
    the knowledge base contents and debugging missing documents.
    
    Args:
        limit: Maximum sources to return (default: 50)
    
    Returns:
        dict: List of indexed file paths
    """
    try:
        _init_engine()
        
        import sqlite3
        
        db_file = _engine['db_path'] / "store.db"
        if not db_file.exists():
            return {
                'error': 'Database not found',
                'sources': [],
                'total_sources': 0
            }
        
        conn = sqlite3.connect(str(db_file))
        
        # Get distinct file paths from documents table
        cursor = conn.execute("""
            SELECT DISTINCT json_extract(metadata, '$.file_path') as file_path
            FROM documents 
            WHERE json_extract(metadata, '$.file_path') IS NOT NULL
            LIMIT ?
        """, (limit,))
        
        sources = [row[0] for row in cursor.fetchall() if row[0]]
        
        # Get total count
        total = conn.execute("""
            SELECT COUNT(DISTINCT json_extract(metadata, '$.file_path'))
            FROM documents 
            WHERE json_extract(metadata, '$.file_path') IS NOT NULL
        """).fetchone()[0]
        
        conn.close()
        
        return {
            'sources': sources,
            'total_sources': total
        }
        
    except Exception as e:
        return {
            'error': f'Failed to list sources: {str(e)}',
            'sources': [],
            'total_sources': 0
        }

if __name__ == "__main__":
    mcp.run()