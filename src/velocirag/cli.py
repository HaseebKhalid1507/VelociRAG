"""
Velocirag Phase 8 CLI - Command line dominance.

Click-based command line interface providing complete access to Velocirag's
indexing and search capabilities. Designed for both interactive exploration
and automated pipeline integration.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

import click

# Import internal modules with graceful error handling
try:
    from .store import VectorStore
    from .embedder import Embedder
    from .searcher import Searcher
    from .metadata import MetadataStore
    from .tracker import UsageTracker
    from .graph import GraphStore
    from .pipeline import GraphPipeline
    from .unified import UnifiedSearch
except ImportError as e:
    click.echo(f"Error: Missing dependencies. Please run 'pip install velocirag'", err=True)
    sys.exit(1)


# Color support
def use_colors() -> bool:
    """Check if colors should be used based on environment."""
    return not os.environ.get('NO_COLOR') and sys.stdout.isatty()


def success(text: str) -> str:
    """Format success message with color."""
    if use_colors():
        return click.style(text, fg='green')
    return text


def warning(text: str) -> str:
    """Format warning message with color."""
    if use_colors():
        return click.style(text, fg='yellow')
    return text


def error(text: str) -> str:
    """Format error message with color."""
    if use_colors():
        return click.style(text, fg='red')
    return text


def info(text: str) -> str:
    """Format info message with color."""
    if use_colors():
        return click.style(text, fg='blue')
    return text


def resolve_db_path(db_param: Optional[str]) -> Path:
    """
    Intelligent database path resolution with clear precedence:
    
    1. Explicit --db parameter (highest priority)
    2. VELOCIRAG_DB environment variable  
    3. ./.velocirag/ in current working directory (default)
    
    Creates directory structure if needed.
    """
    if db_param:
        path = Path(db_param).expanduser().absolute()
    else:
        env_path = os.environ.get('VELOCIRAG_DB')
        if env_path:
            path = Path(env_path).expanduser().absolute()
        else:
            path = Path('.velocirag').absolute()
    
    # Ensure directory exists
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


@click.group()
@click.version_option()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx, verbose: bool):
    """
    Velocirag — Production vector search for markdown documents.
    
    Built for speed, precision, and relentless knowledge extraction.
    """
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose


@cli.command()
@click.argument('path', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--db', type=click.Path(), default=None, 
              help='Database path (default: ./.velocirag/)')
@click.option('--source', '-s', default='', 
              help='Source identifier for this content')
@click.option('--force', '-f', is_flag=True,
              help='Force reindex all files (ignore mtime cache)')
@click.option('--graph', '-g', is_flag=True,
              help='Build knowledge graph during indexing')
@click.option('--metadata', '-m', is_flag=True,
              help='Extract metadata from frontmatter and content')
@click.option('--gliner', is_flag=True,
              help='Use GLiNER encoder model for entity extraction (requires pip install velocirag[ner])')
@click.pass_context
def index(ctx, path: str, db: Optional[str], source: str, force: bool, graph: bool, metadata: bool, gliner: bool):
    """
    Index a directory of markdown files.
    
    Recursively processes all .md files, extracting content and building
    vector representations for semantic search. Uses mtime-based incremental
    indexing to avoid redundant processing.
    
    Examples:
        velocirag index ~/notes
        velocirag index ~/docs --db ~/search.db --source "documentation"
        velocirag index ~/blog --force
    """
    verbose = ctx.obj.get('verbose', False)
    db_path = resolve_db_path(db)
    path = Path(path).absolute()
    
    if verbose:
        click.echo(f"Database path: {db_path}")
        click.echo(f"Source directory: {path}")
        click.echo(f"Source name: {source or '(default)'}")
        click.echo(f"Force reindex: {force}")
        click.echo()
    
    # Check if directory contains markdown files
    md_files = list(path.rglob('*.md'))
    if not md_files:
        click.echo(warning(f"No markdown files found in {path}"))
        return
    
    click.echo(f"Found {len(md_files)} markdown files")
    
    try:
        # Initialize components
        if verbose:
            click.echo("Initializing embedder...")
        embedder = Embedder()
        
        if verbose:
            click.echo("Initializing vector store...")
        store = VectorStore(str(db_path), embedder)
        
        # Set force flag for mtime override
        if force:
            if verbose:
                click.echo("Force mode: ignoring file cache")
        
        # Index directory
        start_time = time.time()
        click.echo(f"Indexing {path}...")
        
        if verbose:
            click.echo("Processing files...")
        
        stats = store.add_directory(str(path), source)
        
        duration = time.time() - start_time
        
        # Display results
        click.echo()
        click.echo(success("Results:"))
        click.echo(f"  Files processed: {stats['files_processed']}")
        click.echo(f"  Files skipped: {stats['files_skipped']} (unchanged)")
        click.echo(f"  Chunks added: {stats['chunks_added']}")
        
        if stats.get('files_deleted', 0) > 0:
            click.echo(f"  Files cleaned up: {stats['files_deleted']}")
        
        # Get total stats
        store_stats = store.stats()
        click.echo(f"  Total documents: {store_stats['document_count']}")
        click.echo(f"  Processing time: {duration:.1f}s")
        
        if stats.get('errors'):
            click.echo()
            click.echo(warning(f"{len(stats['errors'])} errors encountered:"))
            for error_msg in stats['errors'][:5]:  # Show first 5 errors
                click.echo(f"  • {error_msg}")
            if len(stats['errors']) > 5:
                click.echo(f"  ... and {len(stats['errors']) - 5} more")
        
        # Build graph if requested
        if graph:
            click.echo()
            click.echo(f"Building knowledge graph...")
            try:
                graph_db = db_path / "graph.db"
                graph_store = GraphStore(str(graph_db))
                
                metadata_store_obj = None
                if metadata:
                    metadata_db = db_path / "metadata.db"
                    metadata_store_obj = MetadataStore(str(metadata_db))
                
                entity_ext = 'gliner' if gliner else 'regex'
                pipeline = GraphPipeline(
                    graph_store=graph_store,
                    embedder=embedder,
                    metadata_store=metadata_store_obj,
                    entity_extractor=entity_ext
                )
                
                graph_stats = pipeline.build(str(path), force_rebuild=force)
                
                if graph_stats.get('success'):
                    click.echo(success("Graph build complete:"))
                    click.echo(f"  Nodes: {graph_stats['final_nodes']}")
                    click.echo(f"  Edges: {graph_stats['final_edges']}")
                    click.echo(f"  Time: {graph_stats['duration_seconds']:.1f}s")
                    
                    if metadata and 'metadata' in graph_stats.get('stages', {}):
                        meta_stats = graph_stats['stages']['metadata']
                        click.echo()
                        click.echo(success("Metadata extraction:"))
                        click.echo(f"  Documents: {meta_stats['documents_processed']}")
                        click.echo(f"  Tags extracted: {meta_stats['tags_extracted']}")
                        click.echo(f"  Cross-refs: {meta_stats['cross_refs_extracted']}")
                else:
                    click.echo(error(f"Graph build failed: {graph_stats.get('error', 'Unknown error')}"))
                    
            except Exception as e:
                click.echo(error(f"Graph build failed: {e}"))
                if verbose:
                    import traceback
                    traceback.print_exc()
        
        # Extract metadata without graph if requested
        elif metadata:
            click.echo()
            click.echo(f"Extracting metadata...")
            try:
                metadata_db = db_path / "metadata.db"
                metadata_store_obj = MetadataStore(str(metadata_db))
                
                # Create a minimal pipeline just for metadata extraction
                graph_store = GraphStore(":memory:")  # Temporary in-memory graph
                pipeline = GraphPipeline(
                    graph_store=graph_store,
                    embedder=None,  # No embedder needed for metadata only
                    metadata_store=metadata_store_obj
                )
                
                # Just run the first two stages
                pipeline._stage_1_scan_files(str(path))
                pipeline._stage_2_metadata_extraction()
                
                if 'metadata' in pipeline.stats.get('stages', {}):
                    meta_stats = pipeline.stats['stages']['metadata']
                    click.echo(success("Metadata extraction complete:"))
                    click.echo(f"  Documents: {meta_stats['documents_processed']}")
                    click.echo(f"  Tags extracted: {meta_stats['tags_extracted']}")
                    click.echo(f"  Cross-refs: {meta_stats['cross_refs_extracted']}")
                    
            except Exception as e:
                click.echo(error(f"Metadata extraction failed: {e}"))
                if verbose:
                    import traceback
                    traceback.print_exc()
        
        click.echo()
        click.echo(success("Index ready for search."))
        
    except Exception as e:
        click.echo(error(f"Indexing failed: {e}"), err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument('query')
@click.option('--db', type=click.Path(), default=None,
              help='Database path (default: ./.velocirag/)')
@click.option('--limit', '-l', default=5, type=int, 
              help='Maximum results to return (default: 5, max: 50)')
@click.option('--threshold', '-t', default=0.3, type=float,
              help='Minimum similarity threshold (default: 0.3)')
@click.option('--format', 'output_format', default='text',
              type=click.Choice(['text', 'json', 'compact']),
              help='Output format (default: text)')
@click.option('--stats', is_flag=True,
              help='Include search performance statistics')
@click.option('--tags', multiple=True,
              help='Filter results by tags (metadata filter)')
@click.option('--status', default=None,
              help='Filter results by status (metadata filter)')
@click.option('--category', default=None,
              help='Filter results by category (metadata filter)')
@click.option('--project', default=None,
              help='Filter results by project (metadata filter)')
@click.pass_context  
def search(ctx, query: str, db: Optional[str], limit: int, 
          threshold: float, output_format: str, stats: bool,
          tags: tuple, status: str, category: str, project: str):
    """
    Search indexed content using semantic similarity.
    
    Processes query through embedding model and returns most relevant
    content chunks ranked by cosine similarity. Uses query variants
    and reciprocal rank fusion for enhanced recall.
    
    Examples:
        velocirag search "machine learning algorithms"
        velocirag search "python testing" --limit 10 --threshold 0.5
        velocirag search "data structures" --format json --stats
    """
    verbose = ctx.obj.get('verbose', False)
    db_path = resolve_db_path(db)
    
    # Validate parameters
    if limit <= 0 or limit > 50:
        click.echo(error(f"Limit must be between 1 and 50, got {limit}"), err=True)
        sys.exit(1)
    
    if not 0.0 <= threshold <= 1.0:
        click.echo(error(f"Threshold must be between 0.0 and 1.0, got {threshold}"), err=True)
        sys.exit(1)
    
    if not query.strip():
        click.echo(error("Query cannot be empty"), err=True)
        sys.exit(1)
    
    try:
        # Check if database exists
        if not (db_path / "store.db").exists():
            click.echo(error("Database not found. Run 'velocirag index <path>' to create an index."), err=True)
            sys.exit(1)
        
        # Initialize components
        if verbose:
            click.echo("Initializing components...")
        
        embedder = Embedder()
        store = VectorStore(str(db_path), embedder)
        searcher = Searcher(store, embedder)
        
        # Check if index has content
        store_stats = store.stats()
        if store_stats['document_count'] == 0:
            click.echo(warning("No documents indexed. Run 'velocirag index <path>' first."))
            return
        
        # Initialize optional components
        graph_store = None
        metadata_store = None
        tracker = None
        
        # Check for graph database
        graph_db = db_path / "graph.db"
        if graph_db.exists():
            try:
                graph_store = GraphStore(str(graph_db))
                if verbose:
                    click.echo("Graph database found and loaded")
            except Exception as e:
                if verbose:
                    click.echo(warning(f"Could not load graph database: {e}"))
        
        # Check for metadata database
        metadata_db = db_path / "metadata.db"
        if metadata_db.exists():
            try:
                metadata_store = MetadataStore(str(metadata_db))
                tracker = UsageTracker(metadata_store)
                if verbose:
                    click.echo("Metadata database found and loaded")
            except Exception as e:
                if verbose:
                    click.echo(warning(f"Could not load metadata database: {e}"))
        
        # Build metadata filters if provided
        filters = None
        if any([tags, status, category, project]):
            filters = {}
            if tags:
                filters['tags'] = list(tags)
            if status:
                filters['status'] = status
            if category:
                filters['category'] = category
            if project:
                filters['project'] = project
            
            if verbose:
                click.echo(f"Applying metadata filters: {filters}")
        
        # Create unified search
        unified_search = UnifiedSearch(
            searcher=searcher,
            graph_store=graph_store,
            metadata_store=metadata_store,
            tracker=tracker
        )
        
        # Execute search
        if verbose:
            click.echo(f"Searching for: '{query}'")
        
        results = unified_search.search(
            query=query,
            limit=limit,
            threshold=threshold,
            enrich_graph=True,
            filters=filters
        )
        
        # Format output
        if output_format == 'json':
            click.echo(json.dumps(results, indent=2))
        elif output_format == 'compact':
            # Compact format for scripting
            for i, result in enumerate(results['results']):
                click.echo(f"{result['similarity']:.3f}\t{result['doc_id']}")
        else:
            # Human-readable text format
            search_time = results['search_time_ms']
            total_results = results['total_results']
            
            if total_results == 0:
                click.echo(f"Query: \"{query}\"")
                click.echo(warning(f"No results found in {search_time:.0f}ms"))
                click.echo()
                click.echo("Try:")
                click.echo("• Lowering the similarity threshold with --threshold")
                click.echo("• Using different keywords")
                click.echo("• Checking if your content was indexed correctly")
                return
            
            # Header
            click.echo(f"Query: \"{query}\"")
            mode_str = f" ({results.get('search_mode', 'vector')})" if verbose else ""
            click.echo(info(f"Found {total_results} result{'s' if total_results != 1 else ''} in {search_time:.0f}ms{mode_str}"))
            click.echo()
            
            # Results
            for i, result in enumerate(results['results'], 1):
                # Handle both old format (similarity) and new format (score)
                similarity = result.get('similarity', result.get('score', 0))
                doc_id = result.get('doc_id', '')
                content = result.get('content', result.get('chunk', ''))
                metadata = result.get('metadata', {})
                
                # Similarity score with color coding
                if similarity >= 0.8:
                    score_str = success(f"[{similarity:.3f}]")
                elif similarity >= 0.6:
                    score_str = warning(f"[{similarity:.3f}]")
                else:
                    score_str = f"[{similarity:.3f}]"
                
                # Add metadata match indicator
                if metadata.get('_metadata_match'):
                    score_str += " " + success("✓")
                
                click.echo(f"{i}. {score_str} {doc_id}")
                
                # Content preview (truncated)
                preview = content[:200].replace('\n', ' ').strip()
                if len(content) > 200:
                    preview += "..."
                click.echo(f"   {preview}")
                
                # Show graph connections if available
                if metadata.get('graph_connections'):
                    connections = metadata['graph_connections'][:3]
                    click.echo(f"   Connections: {', '.join(connections)}")
                
                # Show metadata if filters were used
                if filters and metadata.get('found_in_graph'):
                    file_path = metadata.get('file_path', '')
                    if file_path:
                        click.echo(f"   File: {Path(file_path).name}")
                
                if i < len(results['results']):  # Don't add newline after last result
                    click.echo()
            
            # Optional stats
            if stats:
                click.echo()
                click.echo(info("Search Statistics:"))
                
                # Enrichment stats
                if 'enrichment_stats' in results:
                    enrich = results['enrichment_stats']
                    click.echo(f"  Vector results: {enrich['vector_results']}")
                    if enrich.get('metadata_available'):
                        click.echo(f"  Metadata matches: {enrich.get('metadata_matches', 0)}")
                    if enrich.get('graph_available'):
                        click.echo(f"  Graph enriched: {enrich['graph_enriched']}")
                
                # Timing breakdown
                click.echo()
                click.echo("  Timing:")
                if 'vector_time_ms' in results:
                    click.echo(f"    Vector search: {results['vector_time_ms']:.1f}ms")
                if 'metadata_time_ms' in results:
                    click.echo(f"    Metadata query: {results['metadata_time_ms']:.1f}ms")
                if 'fusion_time_ms' in results:
                    click.echo(f"    RRF fusion: {results['fusion_time_ms']:.1f}ms")
                click.echo(f"    Total: {search_time:.1f}ms")
        
    except Exception as e:
        click.echo(error(f"Search failed: {e}"), err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option('--db', type=click.Path(), default=None,
              help='Database path (default: ./.velocirag/)')
@click.option('--format', 'output_format', default='text',
              type=click.Choice(['text', 'json']),
              help='Output format (default: text)')
@click.pass_context
def status(ctx, db: Optional[str], output_format: str):
    """
    Show index statistics and health information.
    
    Displays document count, storage size, index consistency,
    and other diagnostic information for troubleshooting.
    
    Examples:
        velocirag status
        velocirag status --format json
    """
    verbose = ctx.obj.get('verbose', False)
    db_path = resolve_db_path(db)
    
    try:
        # Check if database exists
        if not (db_path / "store.db").exists():
            if output_format == 'json':
                click.echo(json.dumps({
                    'error': 'Database not found',
                    'db_path': str(db_path),
                    'exists': False
                }))
            else:
                click.echo(error("Database not found. Run 'velocirag index <path>' to create an index."))
            return
        
        # Initialize store (without embedder since we're just reading stats)
        store = VectorStore(str(db_path))
        stats = store.stats()
        
        # Get additional info
        db_file = db_path / "store.db"
        faiss_file = db_path / "index.faiss"
        
        # Calculate last modified time
        last_modified = None
        if db_file.exists():
            last_modified = db_file.stat().st_mtime
            last_modified_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last_modified))
        
        if output_format == 'json':
            # JSON output for machine consumption
            output = {
                'database': str(db_path),
                'document_count': stats['document_count'],
                'faiss_vectors': stats['faiss_vectors'],
                'consistent': stats['consistent'],
                'dimensions': stats['dimensions'],
                'schema_version': stats['schema_version'],
                'index_dirty': stats['index_dirty'],
                'db_size_bytes': stats.get('db_size_bytes', 0),
                'db_size_mb': stats.get('db_size_mb', 0.0),
                'files_exist': {
                    'db': db_file.exists(),
                    'faiss': faiss_file.exists()
                },
                'last_modified': last_modified_str if last_modified else None
            }
            click.echo(json.dumps(output, indent=2))
        else:
            # Human-readable text format
            click.echo(info("Velocirag Index Status"))
            click.echo()
            
            # Basic info
            click.echo(f"Database: {db_path}")
            click.echo(f"Documents: {stats['document_count']} chunks")
            
            if stats['faiss_vectors'] > 0:
                click.echo(f"Vector Index: {stats['faiss_vectors']} embeddings ({stats['dimensions']} dimensions)")
            else:
                click.echo(warning("Vector Index: No vectors found"))
            
            # Consistency check
            if stats['consistent']:
                click.echo(success("Index Status: Consistent ✓"))
            else:
                click.echo(warning(f"Index Status: Inconsistent ({stats['document_count']} docs vs {stats['faiss_vectors']} vectors)"))
            
            # File sizes
            if 'db_size_mb' in stats:
                click.echo(f"Database Size: {format_file_size(stats.get('db_size_bytes', 0))}")
            
            if last_modified:
                click.echo(f"Last Modified: {last_modified_str}")
            
            # File existence
            click.echo()
            click.echo(info("Files:"))
            click.echo(f"  SQLite DB: {success('✓') if db_file.exists() else error('✗')} {db_file}")
            click.echo(f"  FAISS Index: {success('✓') if faiss_file.exists() else error('✗')} {faiss_file}")
            
            # Health check
            click.echo()
            if stats['document_count'] > 0 and stats['consistent'] and not stats.get('index_dirty', False):
                click.echo(success("Health Check: All systems operational ✓"))
            else:
                click.echo(warning("Health Check: Issues detected"))
                if stats['document_count'] == 0:
                    click.echo("  • No documents indexed")
                if not stats['consistent']:
                    click.echo("  • Index inconsistency detected")
                if stats.get('index_dirty', False):
                    click.echo("  • Index needs rebuilding")
                    click.echo("  • Run 'velocirag reindex' to fix")
        
    except Exception as e:
        if output_format == 'json':
            click.echo(json.dumps({'error': str(e)}))
        else:
            click.echo(error(f"Status check failed: {e}"), err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option('--db', type=click.Path(), default=None,
              help='Database path (default: ./.velocirag/)')
@click.option('--format', 'output_format', type=click.Choice(['text', 'json']),
              default='text', help='Output format')
@click.option('--daemon', type=str, default=None,
              help='Daemon socket path to also check daemon health')
@click.pass_context
def health(ctx, db, output_format, daemon):
    """Full stack health check — every component, sync status, live test."""
    import sqlite3
    import socket
    import struct
    
    verbose = ctx.obj.get('verbose', False)
    db_path = resolve_db_path(db)
    
    def _query_daemon(socket_path, request):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(5.0)
        s.connect(socket_path)
        data = json.dumps(request).encode()
        s.sendall(struct.pack('>I', len(data)) + data)
        raw = s.recv(4)
        msg_len = struct.unpack('>I', raw)[0]
        buf = b""
        while len(buf) < msg_len:
            buf += s.recv(min(msg_len - len(buf), 65536))
        s.close()
        return json.loads(buf)

    issues = []
    health_stats = {}
    healthy_count = 0
    total_count = 0

    # Check daemon socket path
    daemon_socket = daemon or "/tmp/jawz-search.sock"
    check_daemon = daemon is not None or os.path.exists(daemon_socket)

    # 1. Vector Store
    total_count += 1
    try:
        if (db_path / "store.db").exists():
            store = VectorStore(str(db_path))
            store_stats = store.stats()
            doc_count = store_stats['document_count']
            faiss_count = store_stats['faiss_vectors']
            consistent = store_stats['consistent']
            
            if doc_count > 0 and consistent:
                status = "✅"
                healthy_count += 1
            elif doc_count > 0:
                status = "⚠️"
                issues.append("Vector store inconsistent")
            else:
                status = "❌"
                issues.append("No documents in vector store")
                
            health_stats['vector_store'] = {
                "status": status,
                "document_count": doc_count,
                "faiss_vectors": faiss_count,
                "consistent": consistent
            }
        else:
            status = "❌"
            issues.append("Vector store database missing")
            health_stats['vector_store'] = {
                "status": status,
                "error": "store.db not found"
            }
    except Exception as e:
        status = "❌"
        issues.append(f"Vector store error: {str(e)}")
        health_stats['vector_store'] = {
            "status": status,
            "error": str(e)
        }

    # 2. FTS5 Index
    total_count += 1
    try:
        if (db_path / "store.db").exists():
            conn = sqlite3.connect(str(db_path / "store.db"))
            try:
                fts_count = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
                doc_count = health_stats.get('vector_store', {}).get('document_count', 0)
                
                if fts_count > 0 and fts_count == doc_count:
                    status = "✅"
                    healthy_count += 1
                elif fts_count > 0:
                    status = "⚠️"
                    issues.append(f"FTS5 sync issue ({fts_count} vs {doc_count} docs)")
                else:
                    status = "❌"
                    issues.append("No FTS5 entries")
                    
                health_stats['fts5_index'] = {
                    "status": status,
                    "entry_count": fts_count,
                    "synced_with_docs": fts_count == doc_count
                }
                
                # Test query
                test_result = conn.execute("SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'test'").fetchone()[0]
                health_stats['fts5_index']['test_query_results'] = test_result
            except Exception as e:
                status = "❌"
                issues.append(f"FTS5 error: {str(e)}")
                health_stats['fts5_index'] = {
                    "status": status,
                    "error": str(e)
                }
            finally:
                conn.close()
        else:
            status = "❌"
            issues.append("FTS5 database missing")
            health_stats['fts5_index'] = {
                "status": status,
                "error": "store.db not found"
            }
    except Exception as e:
        status = "❌"
        issues.append(f"FTS5 check failed: {str(e)}")
        health_stats['fts5_index'] = {
            "status": status,
            "error": str(e)
        }

    # 3. Knowledge Graph
    total_count += 1
    try:
        if (db_path / "graph.db").exists():
            conn = sqlite3.connect(str(db_path / "graph.db"))
            try:
                node_count = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
                edge_count = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
                gliner_count = conn.execute("SELECT COUNT(*) FROM nodes WHERE metadata LIKE '%gliner%'").fetchone()[0]
                
                # Get type breakdown
                node_types = dict(conn.execute("SELECT type, COUNT(*) FROM nodes GROUP BY type").fetchall())
                edge_types = dict(conn.execute("SELECT type, COUNT(*) FROM edges GROUP BY type").fetchall())
                
                if node_count > 0 and edge_count > 0:
                    status = "✅"
                    healthy_count += 1
                elif node_count > 0:
                    status = "⚠️"
                    issues.append("Graph has nodes but no edges")
                else:
                    status = "❌"
                    issues.append("Empty knowledge graph")
                    
                health_stats['knowledge_graph'] = {
                    "status": status,
                    "node_count": node_count,
                    "edge_count": edge_count,
                    "gliner_entities": gliner_count,
                    "node_types": node_types,
                    "edge_types": edge_types
                }
            except Exception as e:
                status = "❌"
                issues.append(f"Graph error: {str(e)}")
                health_stats['knowledge_graph'] = {
                    "status": status,
                    "error": str(e)
                }
            finally:
                conn.close()
        else:
            status = "⚠️"
            health_stats['knowledge_graph'] = {
                "status": status,
                "note": "Graph database not present (optional)"
            }
    except Exception as e:
        status = "❌"
        issues.append(f"Graph check failed: {str(e)}")
        health_stats['knowledge_graph'] = {
            "status": status,
            "error": str(e)
        }

    # 4. Metadata Store
    total_count += 1
    try:
        if (db_path / "metadata.db").exists():
            conn = sqlite3.connect(str(db_path / "metadata.db"))
            try:
                doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
                tag_count = conn.execute("SELECT COUNT(*) FROM tags").fetchone()[0]
                
                # Get top tags
                top_tags = dict(conn.execute("""
                    SELECT t.name, COUNT(*) as count 
                    FROM tags t 
                    JOIN document_tags dt ON t.id = dt.tag_id 
                    GROUP BY t.name 
                    ORDER BY count DESC 
                    LIMIT 5
                """).fetchall())
                
                if doc_count > 0:
                    status = "✅"
                    healthy_count += 1
                else:
                    status = "❌"
                    issues.append("No documents in metadata store")
                    
                health_stats['metadata_store'] = {
                    "status": status,
                    "document_count": doc_count,
                    "tag_count": tag_count,
                    "top_tags": top_tags
                }
            except Exception as e:
                status = "❌"
                issues.append(f"Metadata error: {str(e)}")
                health_stats['metadata_store'] = {
                    "status": status,
                    "error": str(e)
                }
            finally:
                conn.close()
        else:
            status = "⚠️"
            health_stats['metadata_store'] = {
                "status": status,
                "note": "Metadata database not present (optional)"
            }
    except Exception as e:
        status = "❌"
        issues.append(f"Metadata check failed: {str(e)}")
        health_stats['metadata_store'] = {
            "status": status,
            "error": str(e)
        }

    # 5. Sources
    total_count += 1
    try:
        if (db_path / "store.db").exists():
            conn = sqlite3.connect(str(db_path / "store.db"))
            try:
                sources = dict(conn.execute("""
                    SELECT json_extract(metadata, '$.source_name') as source, COUNT(*) as count
                    FROM documents 
                    WHERE json_extract(metadata, '$.source_name') IS NOT NULL
                    GROUP BY source
                """).fetchall())
                
                if sources:
                    status = "✅"
                    healthy_count += 1
                else:
                    status = "⚠️"
                    issues.append("No source breakdown available")
                    
                health_stats['sources'] = {
                    "status": status,
                    "breakdown": sources
                }
            except Exception as e:
                status = "❌"
                issues.append(f"Sources error: {str(e)}")
                health_stats['sources'] = {
                    "status": status,
                    "error": str(e)
                }
            finally:
                conn.close()
        else:
            status = "❌"
            issues.append("Sources database missing")
            health_stats['sources'] = {
                "status": status,
                "error": "store.db not found"
            }
    except Exception as e:
        status = "❌"
        issues.append(f"Sources check failed: {str(e)}")
        health_stats['sources'] = {
            "status": status,
            "error": str(e)
        }

    # 6. Daemon (if socket exists or path given)
    if check_daemon:
        total_count += 1
        try:
            daemon_health = _query_daemon(daemon_socket, {"cmd": "health"})
            
            if daemon_health.get('status') == 'ok':
                status = "✅"
                healthy_count += 1
                health_stats['daemon'] = {
                    "status": status,
                    "uptime_seconds": daemon_health.get('uptime_seconds', 0),
                    "requests_served": daemon_health.get('requests_served', 0),
                    "components": daemon_health.get('components', {}),
                    "fd_count": daemon_health.get('fd_count', 0),
                    "search_test": daemon_health.get('search_test', False),
                    "store": daemon_health.get('store', {}),
                    "graph": daemon_health.get('graph', {}),
                    "fts5_count": daemon_health.get('fts5_count', 0)
                }
            else:
                status = "❌"
                issues.append(f"Daemon unhealthy: {daemon_health}")
                health_stats['daemon'] = {
                    "status": status,
                    "error": daemon_health
                }
        except Exception as e:
            status = "❌"
            issues.append(f"Daemon check failed: {str(e)}")
            health_stats['daemon'] = {
                "status": status,
                "error": str(e)
            }

    # Output results
    if output_format == 'json':
        result = {
            "summary": {
                "healthy_components": healthy_count,
                "total_components": total_count,
                "overall_health": healthy_count == total_count
            },
            "components": health_stats,
            "issues": issues
        }
        click.echo(json.dumps(result, indent=2))
    else:
        # Text format
        click.echo(info("Velocirag Health Check"))
        click.echo()
        
        # Vector Store
        vs = health_stats.get('vector_store', {})
        click.echo(f"{vs.get('status', '❌')} Vector Store")
        if 'document_count' in vs:
            click.echo(f"   Documents: {vs['document_count']}")
            click.echo(f"   FAISS vectors: {vs['faiss_vectors']}")
            click.echo(f"   Consistent: {vs['consistent']}")
        elif 'error' in vs:
            click.echo(f"   Error: {vs['error']}")
        
        # FTS5 Index  
        fts = health_stats.get('fts5_index', {})
        click.echo(f"{fts.get('status', '❌')} FTS5 Index")
        if 'entry_count' in fts:
            click.echo(f"   Entries: {fts['entry_count']}")
            click.echo(f"   Synced: {fts['synced_with_docs']}")
            click.echo(f"   Test query results: {fts.get('test_query_results', 0)}")
        elif 'error' in fts:
            click.echo(f"   Error: {fts['error']}")
        
        # Knowledge Graph
        kg = health_stats.get('knowledge_graph', {})
        click.echo(f"{kg.get('status', '❌')} Knowledge Graph")
        if 'node_count' in kg:
            click.echo(f"   Nodes: {kg['node_count']}")
            click.echo(f"   Edges: {kg['edge_count']}")
            click.echo(f"   GLiNER entities: {kg['gliner_entities']}")
            if kg['node_types']:
                types_str = ', '.join(f"{k}:{v}" for k, v in list(kg['node_types'].items())[:3])
                click.echo(f"   Node types: {types_str}")
        elif 'error' in kg:
            click.echo(f"   Error: {kg['error']}")
        elif 'note' in kg:
            click.echo(f"   {kg['note']}")
        
        # Metadata Store
        ms = health_stats.get('metadata_store', {})
        click.echo(f"{ms.get('status', '❌')} Metadata Store")
        if 'document_count' in ms:
            click.echo(f"   Documents: {ms['document_count']}")
            click.echo(f"   Tags: {ms['tag_count']}")
            if ms['top_tags']:
                tags_str = ', '.join(f"{k}({v})" for k, v in list(ms['top_tags'].items())[:3])
                click.echo(f"   Top tags: {tags_str}")
        elif 'error' in ms:
            click.echo(f"   Error: {ms['error']}")
        elif 'note' in ms:
            click.echo(f"   {ms['note']}")
        
        # Sources
        src = health_stats.get('sources', {})
        click.echo(f"{src.get('status', '❌')} Sources")
        if 'breakdown' in src:
            if src['breakdown']:
                breakdown_str = ', '.join(f"{k}({v})" for k, v in list(src['breakdown'].items())[:3])
                click.echo(f"   Breakdown: {breakdown_str}")
            else:
                click.echo("   No source metadata")
        elif 'error' in src:
            click.echo(f"   Error: {src['error']}")
        
        # Daemon (if checked)
        if check_daemon:
            daemon = health_stats.get('daemon', {})
            click.echo(f"{daemon.get('status', '❌')} Daemon")
            if 'uptime_seconds' in daemon:
                click.echo(f"   Uptime: {daemon['uptime_seconds']}s")
                click.echo(f"   Requests: {daemon['requests_served']}")
                click.echo(f"   FD count: {daemon['fd_count']}")
                click.echo(f"   Search test: {daemon['search_test']}")
                components = daemon['components']
                comp_str = ', '.join(k for k, v in components.items() if v)
                click.echo(f"   Components: {comp_str}")
            elif 'error' in daemon:
                click.echo(f"   Error: {daemon['error']}")
        
        # Summary
        click.echo()
        if healthy_count == total_count:
            click.echo(success(f"✅ {healthy_count}/{total_count} components healthy"))
        else:
            click.echo(warning(f"⚠️ {healthy_count}/{total_count} components healthy"))
            if issues:
                click.echo()
                click.echo(error("Issues found:"))
                for issue in issues:
                    click.echo(f"  • {issue}")


@cli.command()
@click.option('--db', type=click.Path(), default=None,
              help='Database path (default: ./.velocirag/)')
@click.option('--yes', '-y', is_flag=True,
              help='Skip confirmation prompt')
@click.pass_context
def reindex(ctx, db: Optional[str], yes: bool):
    """
    Rebuild FAISS index from stored SQLite embeddings.
    
    Forces complete reconstruction of the vector search index.
    Use when index is corrupted or after significant schema changes.
    
    Examples:
        velocirag reindex
        velocirag reindex --yes
    """
    verbose = ctx.obj.get('verbose', False)
    db_path = resolve_db_path(db)
    
    try:
        # Check if database exists
        if not (db_path / "store.db").exists():
            click.echo(error("Database not found. Run 'velocirag index <path>' to create an index."), err=True)
            sys.exit(1)
        
        # Initialize store
        store = VectorStore(str(db_path))
        stats = store.stats()
        
        if stats['document_count'] == 0:
            click.echo(warning("No documents in database. Nothing to reindex."))
            return
        
        # Confirmation prompt
        if not yes:
            click.echo(f"This will rebuild the entire FAISS index from {stats['document_count']} stored embeddings.")
            click.confirm("Continue?", abort=True)
        
        # Rebuild index
        click.echo(f"Rebuilding FAISS index from {stats['document_count']} stored embeddings...")
        
        start_time = time.time()
        store.rebuild_index()
        rebuild_time = time.time() - start_time
        
        # Verify rebuild
        new_stats = store.stats()
        
        click.echo()
        click.echo(success("Index rebuilt successfully."))
        click.echo(f"Documents: {new_stats['document_count']}")
        click.echo(f"Vectors: {new_stats['faiss_vectors']}")
        click.echo(f"Rebuild time: {rebuild_time:.1f}s")
        
        if new_stats['consistent']:
            click.echo(success("Index is now consistent ✓"))
        else:
            click.echo(error("Warning: Index still inconsistent"))
        
    except KeyboardInterrupt:
        click.echo(warning("\nReindex cancelled"))
        sys.exit(1)
    except Exception as e:
        click.echo(error(f"Reindex failed: {e}"), err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option('--db', type=click.Path(), default=None,
              help='Database path (default: ./.velocirag/)')
@click.option('--tags', '-t', multiple=True,
              help='Filter by tags (can specify multiple)')
@click.option('--status', '-s', default=None,
              help='Filter by status (active, archived, etc.)')
@click.option('--category', '-c', default=None,
              help='Filter by category')
@click.option('--project', '-p', default=None,
              help='Filter by project')
@click.option('--stale', type=int, default=None,
              help='Find documents not accessed in N days')
@click.option('--recent', type=int, default=None,
              help='Find documents created in last N days')
@click.option('--limit', '-l', default=20, type=int,
              help='Maximum results to return (default: 20)')
@click.option('--format', 'output_format', default='text',
              type=click.Choice(['text', 'json', 'compact']),
              help='Output format (default: text)')
@click.option('--stats', is_flag=True,
              help='Show metadata statistics instead of query results')
@click.pass_context
def query(ctx, db: Optional[str], tags: tuple, status: str, category: str, 
         project: str, stale: Optional[int], recent: Optional[int], 
         limit: int, output_format: str, stats: bool):
    """
    Query documents using metadata filters.
    
    Structured queries based on tags, categories, status, and other metadata.
    Requires metadata to be extracted during indexing.
    
    Examples:
        velocirag query --tags python --tags rust
        velocirag query --status active --project myproject
        velocirag query --stale 90
        velocirag query --recent 7
        velocirag query --stats
    """
    verbose = ctx.obj.get('verbose', False)
    db_path = resolve_db_path(db)
    
    try:
        # Check if metadata database exists
        metadata_db = db_path / "metadata.db"
        if not metadata_db.exists():
            if output_format == 'json':
                click.echo(json.dumps({
                    'error': 'Metadata database not found',
                    'message': 'Run indexing with metadata extraction first'
                }))
            else:
                click.echo(error("Metadata database not found. Metadata extraction may not be configured."))
            sys.exit(1)
        
        # Initialize metadata store
        metadata_store = MetadataStore(str(metadata_db))
        
        # Handle stats request
        if stats:
            stats_data = metadata_store.stats()
            
            if output_format == 'json':
                click.echo(json.dumps(stats_data, indent=2))
            else:
                click.echo(info("Metadata Statistics"))
                click.echo()
                click.echo(f"Total documents: {stats_data['total_documents']}")
                click.echo(f"Total tags: {stats_data['total_tags']}")
                click.echo(f"Total cross-references: {stats_data['total_cross_refs']}")
                click.echo(f"Total usage events: {stats_data['total_usage_events']}")
                click.echo(f"Database size: {stats_data['db_size_mb']} MB")
                
                if stats_data['categories']:
                    click.echo()
                    click.echo(info("Categories:"))
                    for cat, count in stats_data['categories'].items():
                        click.echo(f"  {cat}: {count}")
                
                if stats_data['top_tags']:
                    click.echo()
                    click.echo(info("Top tags:"))
                    for tag, count in list(stats_data['top_tags'].items())[:10]:
                        click.echo(f"  {tag}: {count}")
                
                if stats_data['projects']:
                    click.echo()
                    click.echo(info("Projects:"))
                    for proj, count in stats_data['projects'].items():
                        click.echo(f"  {proj}: {count}")
            return
        
        # Build query filters
        filters = {}
        
        if tags:
            filters['tags'] = list(tags)
        if status:
            filters['status'] = status
        if category:
            filters['category'] = category
        if project:
            filters['project'] = project
        if stale is not None:
            filters['stale_days'] = stale
        if recent is not None:
            # Use get_recent method for recent documents
            results = metadata_store.get_recent(days=recent)
        elif stale is not None and not filters.get('tags'):
            # Use get_stale method for stale documents  
            results = metadata_store.get_stale(days=stale)
        else:
            # Regular query
            filters['limit'] = limit
            results = metadata_store.query(**filters)
        
        # Format output
        if output_format == 'json':
            click.echo(json.dumps(results, indent=2, default=str))
        elif output_format == 'compact':
            for doc in results:
                click.echo(f"{doc['filename']}\t{doc['title']}")
        else:
            # Human-readable format
            if not results:
                click.echo(warning("No documents found matching the criteria"))
                return
            
            click.echo(f"Found {len(results)} document{'s' if len(results) != 1 else ''}")
            click.echo()
            
            for i, doc in enumerate(results, 1):
                # Title and filename
                click.echo(f"{i}. {info(doc['title'])}")
                click.echo(f"   File: {doc['filename']}")
                
                # Metadata fields
                if doc.get('category'):
                    click.echo(f"   Category: {doc['category']}")
                if doc.get('status'):
                    status_color = success if doc['status'] == 'active' else warning
                    click.echo(f"   Status: {status_color(doc['status'])}")
                if doc.get('project'):
                    click.echo(f"   Project: {doc['project']}")
                
                # Tags (from separate query if needed)
                if 'tags' in doc and doc['tags']:
                    tags_str = ', '.join(f"#{tag}" for tag in doc['tags'])
                    click.echo(f"   Tags: {tags_str}")
                
                # Dates
                if doc.get('created_date'):
                    click.echo(f"   Created: {doc['created_date']}")
                if doc.get('updated_at'):
                    click.echo(f"   Updated: {doc['updated_at']}")
                
                # Usage info if available
                if 'usage_stats' in doc:
                    usage = doc['usage_stats']
                    if usage['total_usage'] > 0:
                        click.echo(f"   Usage: {usage['search_hits']} searches, {usage['reads']} reads")
                
                if i < len(results):
                    click.echo()
        
    except Exception as e:
        if output_format == 'json':
            click.echo(json.dumps({'error': str(e)}))
        else:
            click.echo(error(f"Query failed: {e}"), err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    cli()