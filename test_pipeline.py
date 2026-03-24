#!/usr/bin/env python3
"""
Test script for Velociragtor Phase 6 knowledge graph pipeline.
"""

import tempfile
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from velociragtor.graph import GraphStore, GraphQuerier
from velociragtor.pipeline import GraphPipeline
from velociragtor.embedder import Embedder

def main():
    print("🚀 Testing Velociragtor Phase 6 Knowledge Graph Pipeline")
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
        db_path = tmp_db.name
    
    print(f"📊 Using database: {db_path}")
    
    # Initialize components
    graph_store = GraphStore(db_path)
    embedder = Embedder()  # Use default embedder
    pipeline = GraphPipeline(graph_store, embedder)
    
    # Test data path
    test_data_path = Path(__file__).parent / "test_data"
    
    print(f"📁 Building graph from: {test_data_path}")
    
    # Run pipeline
    try:
        results = pipeline.build(str(test_data_path), force_rebuild=True)
        
        print("\n✅ Pipeline Results:")
        print(f"   Duration: {results.get('duration_seconds', 0):.1f}s")
        print(f"   Nodes: {results.get('final_nodes', 0)}")
        print(f"   Edges: {results.get('final_edges', 0)}")
        
        # Show stage breakdown
        print("\n📈 Stage Breakdown:")
        for stage, stats in results.get('stages', {}).items():
            duration = stats.get('duration_seconds', 0)
            print(f"   {stage}: {duration:.1f}s")
        
        # Test querier
        print("\n🔍 Testing Graph Querier:")
        querier = GraphQuerier(graph_store)
        
        # Test connections
        connections = querier.find_connections("Project Alpha", depth=2)
        print(f"   Connections for 'Project Alpha': {connections.get('total_connections', 0)}")
        
        # Test hub nodes
        hubs = querier.get_hub_nodes(limit=5)
        print(f"   Hub nodes found: {len(hubs)}")
        for hub in hubs:
            print(f"     - {hub['title']} ({hub['connection_count']} connections)")
        
        # Get final stats
        final_stats = graph_store.stats()
        print(f"\n📊 Final Database Stats:")
        print(f"   Nodes: {final_stats['node_count']}")
        print(f"   Edges: {final_stats['edge_count']}")
        print(f"   Node types: {final_stats['node_types']}")
        print(f"   Edge types: {final_stats['edge_types']}")
        print(f"   DB size: {final_stats['db_size_mb']} MB")
        
        print("\n🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        try:
            Path(db_path).unlink()
        except:
            pass

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)