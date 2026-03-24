"""
Velociragtor Phase 8 CLI - Command line dominance.

Click-based command line interface providing complete access to Velociragtor's
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
    2. VELOCIRAGTOR_DB environment variable  
    3. ./.velocirag/ in current working directory (default)
    
    Creates directory structure if needed.
    """
    if db_param:
        path = Path(db_param).expanduser().absolute()
    else:
        env_path = os.environ.get('VELOCIRAGTOR_DB')
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
    Velociragtor — Production vector search for markdown documents.
    
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
@click.pass_context
def index(ctx, path: str, db: Optional[str], source: str, force: bool):
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
@click.pass_context  
def search(ctx, query: str, db: Optional[str], limit: int, 
          threshold: float, output_format: str, stats: bool):
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
        
        # Execute search
        if verbose:
            click.echo(f"Searching for: '{query}'")
        
        results = searcher.search(
            query=query,
            limit=limit,
            threshold=threshold,
            include_stats=stats or verbose
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
            click.echo(info(f"Found {total_results} result{'s' if total_results != 1 else ''} in {search_time:.0f}ms"))
            click.echo()
            
            # Results
            for i, result in enumerate(results['results'], 1):
                similarity = result['similarity']
                doc_id = result['doc_id']
                content = result['content']
                
                # Similarity score with color coding
                if similarity >= 0.8:
                    score_str = success(f"[{similarity:.3f}]")
                elif similarity >= 0.6:
                    score_str = warning(f"[{similarity:.3f}]")
                else:
                    score_str = f"[{similarity:.3f}]"
                
                click.echo(f"{i}. {score_str} {doc_id}")
                
                # Content preview (truncated)
                preview = content[:200].replace('\n', ' ').strip()
                if len(content) > 200:
                    preview += "..."
                click.echo(f"   {preview}")
                
                if i < len(results['results']):  # Don't add newline after last result
                    click.echo()
            
            # Optional stats
            if stats and 'stats' in results:
                click.echo()
                click.echo(info("Search Statistics:"))
                search_stats = results['stats']
                
                if 'variants' in search_stats:
                    click.echo(f"  Query variants: {len(search_stats['variants'])}")
                    for variant_stat in search_stats['variants'][:3]:  # Show first 3
                        click.echo(f"    • '{variant_stat['variant']}' → {variant_stat['results_found']} results")
                
                if 'embedding_time_ms' in search_stats:
                    click.echo(f"  Embedding time: {search_stats['embedding_time_ms']:.1f}ms")
                if 'search_time_ms' in search_stats:
                    click.echo(f"  Vector search: {search_stats['search_time_ms']:.1f}ms")
                if 'rrf_fusion_ms' in search_stats:
                    click.echo(f"  RRF fusion: {search_stats['rrf_fusion_ms']:.1f}ms")
        
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
            click.echo(info("Velociragtor Index Status"))
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


if __name__ == '__main__':
    cli()