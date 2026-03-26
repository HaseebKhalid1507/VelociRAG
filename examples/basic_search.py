#!/usr/bin/env python3
"""Basic Velocirag search example — index markdown files and search."""
import sys
from pathlib import Path
from velocirag import Embedder, VectorStore, Searcher

def main():
    if len(sys.argv) < 2:
        print('Usage: python basic_search.py <markdown_directory> [query]')
        print('Example: python basic_search.py ./my-notes "machine learning"')
        sys.exit(1)
    
    docs_path = sys.argv[1]
    query = sys.argv[2] if len(sys.argv) > 2 else 'test'
    db_path = './velocirag-data'
    
    # Initialize
    print(f'Indexing {docs_path}...')
    embedder = Embedder()
    store = VectorStore(db_path, embedder)
    stats = store.add_directory(docs_path)
    print(f'Indexed {stats["files_processed"]} files, {stats["chunks_added"]} chunks')
    
    # Search
    print(f'\nSearching: "{query}"')
    searcher = Searcher(store, embedder)
    results = searcher.search(query, limit=5)
    
    for i, r in enumerate(results.get('results', []), 1):
        score = r.get('similarity', 0)
        path = r.get('metadata', {}).get('file_path', '?')
        content = r.get('content', '')[:200]
        print(f'\n--- Result {i} [{score:.3f}] {path} ---')
        print(content)

if __name__ == '__main__':
    main()