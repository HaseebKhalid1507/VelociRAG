#!/usr/bin/env python3
"""
Velociragtor Phase 6 Knowledge Graph Demo

Shows how to build and query a knowledge graph from markdown files.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from velociragtor.graph import GraphStore, GraphQuerier, NodeType
from velociragtor.pipeline import GraphPipeline
from velociragtor.embedder import Embedder

def main():
    print("🧠 Velociragtor Phase 6 Knowledge Graph Demo")
    print("=" * 50)
    
    # Setup
    db_path = "knowledge_graph.db"
    graph_store = GraphStore(db_path)
    embedder = Embedder()  # Optional: remove for faster builds without semantic analysis
    pipeline = GraphPipeline(graph_store, embedder)
    
    # Build graph from test data
    test_data = Path("test_data")
    if not test_data.exists():
        print("❌ test_data directory not found. Please run from the project root.")
        return
    
    print(f"📁 Building knowledge graph from: {test_data}")
    results = pipeline.build(str(test_data), force_rebuild=True)
    
    print(f"\n✅ Graph built in {results['duration_seconds']:.1f}s")
    print(f"   📊 {results['final_nodes']} nodes, {results['final_edges']} edges")
    
    # Initialize querier
    querier = GraphQuerier(graph_store)
    
    # Demo queries
    print("\n🔍 Demo Queries")
    print("-" * 30)
    
    # 1. Find connections
    print("1. Connections to 'Project Alpha':")
    connections = querier.find_connections("Project Alpha", depth=2)
    if 'error' not in connections:
        print(f"   Found {connections['total_connections']} connections")
        for rel_type, items in connections['connections_by_type'].items():
            print(f"   {rel_type}: {len(items)} connections")
            for item in items[:3]:  # Show first 3
                print(f"     - {item['node']} (weight: {item['weight']:.2f})")
    
    # 2. Hub nodes
    print("\n2. Most connected nodes:")
    hubs = querier.get_hub_nodes(limit=5)
    for i, hub in enumerate(hubs, 1):
        print(f"   {i}. {hub['title']} ({hub['connection_count']} connections)")
    
    # 3. Find path between nodes
    print("\n3. Path finding:")
    # Get some node IDs
    stats = graph_store.stats()
    if stats['node_count'] > 1:
        # Find path between first two note nodes
        note_nodes = graph_store.get_nodes_by_type(NodeType.NOTE)
        if len(note_nodes) >= 2:
            path = querier.find_path(note_nodes[0].id, note_nodes[1].id)
            if path:
                print(f"   Path found: {len(path['path'])} nodes, distance: {path['distance']}")
            else:
                print("   No path found between the nodes")
    
    # 4. Database stats
    print("\n📊 Database Statistics:")
    for stat_name, value in stats.items():
        if isinstance(value, dict):
            print(f"   {stat_name}:")
            for k, v in value.items():
                print(f"     {k}: {v}")
        else:
            print(f"   {stat_name}: {value}")
    
    print(f"\n💾 Graph saved to: {db_path}")
    print("\n🎯 Usage Examples:")
    print("   from velociragtor.graph import GraphStore, GraphQuerier")
    print("   from velociragtor.pipeline import GraphPipeline")
    print("   from velociragtor.embedder import Embedder")
    print("")
    print("   # Load existing graph")
    print(f"   store = GraphStore('{db_path}')")
    print("   querier = GraphQuerier(store)")
    print("")
    print("   # Build new graph")
    print("   pipeline = GraphPipeline(store, Embedder())")
    print("   results = pipeline.build('/path/to/markdown/files')")
    print("")
    print("   # Query the graph")
    print("   connections = querier.find_connections('topic name')")
    print("   similar = querier.find_similar('node_id')")
    print("   hubs = querier.get_hub_nodes()")

if __name__ == "__main__":
    main()