#!/usr/bin/env python3
"""Full 3-layer unified search with graph + metadata."""
import sys
from pathlib import Path
from velocirag import (
    Embedder, VectorStore, Searcher,
    GraphStore, MetadataStore, GraphPipeline,
    UnifiedSearch, UsageTracker
)

def main():
    if len(sys.argv) < 3:
        print('Usage: python unified_search.py <markdown_directory> <query>')
        sys.exit(1)
    
    docs_path = sys.argv[1]
    query = sys.argv[2]
    db_path = './velocirag-data'
    
    # Initialize components
    embedder = Embedder()
    store = VectorStore(db_path, embedder)
    
    # Index documents
    print(f'Indexing {docs_path}...')
    stats = store.add_directory(docs_path)
    print(f'Indexed {stats["files_processed"]} files')
    
    # Build knowledge graph + metadata
    print('Building knowledge graph...')
    graph_store = GraphStore(f'{db_path}/graph.db')
    metadata_store = MetadataStore(f'{db_path}/metadata.db')
    pipeline = GraphPipeline(graph_store, embedder, metadata_store)
    graph_stats = pipeline.build(docs_path, force_rebuild=True)
    print(f'Graph: {graph_stats["final_nodes"]} nodes, {graph_stats["final_edges"]} edges')
    
    # Unified search (vector + graph + metadata fusion)
    searcher = Searcher(store, embedder)
    tracker = UsageTracker(metadata_store)
    unified = UnifiedSearch(searcher, graph_store, metadata_store, tracker)
    
    print(f'\nSearching: "{query}"')
    results = unified.search(query, limit=5, enrich_graph=True)
    
    print(f'Mode: {results["search_mode"]}')
    print(f'Time: {results["search_time_ms"]}ms')
    print(f'Layers: {list(results.get("layer_stats", {}).keys())}')
    
    for i, r in enumerate(results.get('results', []), 1):
        score = r.get('similarity', 0)
        path = r.get('metadata', {}).get('file_path', '?')
        graph = r.get('metadata', {}).get('found_in_graph', False)
        conns = len(r.get('metadata', {}).get('graph_connections', []))
        content = r.get('content', '')[:150]
        print(f'\n--- Result {i} [{score:.3f}] {path} (graph: {graph}, connections: {conns}) ---')
        print(content)

if __name__ == '__main__':
    main()