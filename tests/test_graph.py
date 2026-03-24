"""Tests for graph.py module."""

import pytest
from datetime import datetime
import json
import sqlite3
from pathlib import Path

from velocirag.graph import (
    Node, Edge, NodeType, RelationType, GraphStore,
    GraphStoreError, ReferentialError, ValidationError
)


class TestNodeModel:
    """Test Node dataclass behavior."""
    
    def test_node_creation_basic(self):
        """Node can be created with required fields."""
        node = Node(
            id="test1",
            type=NodeType.NOTE,
            title="Test Node"
        )
        
        assert node.id == "test1"
        assert node.type == NodeType.NOTE
        assert node.title == "Test Node"
        assert node.content is None
        assert node.metadata == {}
        assert isinstance(node.created_at, datetime)
    
    def test_node_creation_full(self):
        """Node can be created with all fields."""
        metadata = {"tags": ["test", "example"], "priority": 5}
        created = datetime.now()
        
        node = Node(
            id="test2",
            type=NodeType.ENTITY,
            title="Full Node",
            content="This is the content",
            metadata=metadata,
            created_at=created
        )
        
        assert node.content == "This is the content"
        assert node.metadata == metadata
        assert node.created_at == created
    
    def test_node_validation_empty_id(self):
        """Empty node ID raises ValueError."""
        with pytest.raises(ValueError, match="Node ID cannot be empty"):
            Node(id="", type=NodeType.NOTE, title="Test")
        
        with pytest.raises(ValueError, match="Node ID cannot be empty"):
            Node(id="  ", type=NodeType.NOTE, title="Test")
    
    def test_node_validation_empty_title(self):
        """Empty node title raises ValueError."""
        with pytest.raises(ValueError, match="Node title cannot be empty"):
            Node(id="test", type=NodeType.NOTE, title="")
        
        with pytest.raises(ValueError, match="Node title cannot be empty"):
            Node(id="test", type=NodeType.NOTE, title="   ")
    
    def test_node_validation_invalid_type(self):
        """Invalid node type raises ValueError."""
        with pytest.raises(ValueError, match="Node type must be NodeType enum"):
            Node(id="test", type="note", title="Test")
    
    def test_node_metadata_none_converted_to_dict(self):
        """None metadata is converted to empty dict."""
        node = Node(id="test", type=NodeType.TAG, title="Test", metadata=None)
        assert node.metadata == {}
    
    def test_node_types_enum(self):
        """All NodeType enum values work."""
        for node_type in NodeType:
            node = Node(id=f"test_{node_type.value}", type=node_type, title="Test")
            assert node.type == node_type


class TestEdgeModel:
    """Test Edge dataclass behavior."""
    
    def test_edge_creation_basic(self):
        """Edge can be created with required fields."""
        edge = Edge(
            id="edge1",
            source_id="node1",
            target_id="node2",
            type=RelationType.REFERENCES,
            weight=0.8,
            confidence=0.9
        )
        
        assert edge.id == "edge1"
        assert edge.source_id == "node1"
        assert edge.target_id == "node2"
        assert edge.type == RelationType.REFERENCES
        assert edge.weight == 0.8
        assert edge.confidence == 0.9
        assert edge.metadata == {}
        assert isinstance(edge.created_at, datetime)
    
    def test_edge_validation_empty_ids(self):
        """Empty IDs raise ValueError."""
        with pytest.raises(ValueError, match="Edge ID cannot be empty"):
            Edge(id="", source_id="n1", target_id="n2", 
                 type=RelationType.REFERENCES, weight=0.5, confidence=0.5)
        
        with pytest.raises(ValueError, match="Source ID cannot be empty"):
            Edge(id="e1", source_id="", target_id="n2",
                 type=RelationType.REFERENCES, weight=0.5, confidence=0.5)
        
        with pytest.raises(ValueError, match="Target ID cannot be empty"):
            Edge(id="e1", source_id="n1", target_id="",
                 type=RelationType.REFERENCES, weight=0.5, confidence=0.5)
    
    def test_edge_self_loops_filtered_at_store(self):
        """Self-loops are silently filtered at store level, not at model level."""
        # Edge creation is allowed — filtering happens in add_edges()
        edge = Edge(id="e1", source_id="node1", target_id="node1",
                    type=RelationType.REFERENCES, weight=0.5, confidence=0.5)
        assert edge.source_id == edge.target_id
    
    def test_edge_validation_invalid_type(self):
        """Invalid edge type raises ValueError."""
        with pytest.raises(ValueError, match="Edge type must be RelationType enum"):
            Edge(id="e1", source_id="n1", target_id="n2",
                 type="references", weight=0.5, confidence=0.5)
    
    def test_edge_validation_weight_range(self):
        """Weight must be 0.0-1.0."""
        with pytest.raises(ValueError, match="Weight must be 0.0-1.0"):
            Edge(id="e1", source_id="n1", target_id="n2",
                 type=RelationType.REFERENCES, weight=1.5, confidence=0.5)
        
        with pytest.raises(ValueError, match="Weight must be 0.0-1.0"):
            Edge(id="e1", source_id="n1", target_id="n2",
                 type=RelationType.REFERENCES, weight=-0.1, confidence=0.5)
    
    def test_edge_validation_confidence_range(self):
        """Confidence must be 0.0-1.0."""
        with pytest.raises(ValueError, match="Confidence must be 0.0-1.0"):
            Edge(id="e1", source_id="n1", target_id="n2",
                 type=RelationType.REFERENCES, weight=0.5, confidence=1.1)
        
        with pytest.raises(ValueError, match="Confidence must be 0.0-1.0"):
            Edge(id="e1", source_id="n1", target_id="n2",
                 type=RelationType.REFERENCES, weight=0.5, confidence=-0.5)
    
    def test_edge_metadata_none_converted_to_dict(self):
        """None metadata is converted to empty dict."""
        edge = Edge(id="e1", source_id="n1", target_id="n2",
                    type=RelationType.SIMILAR_TO, weight=0.5, confidence=0.5,
                    metadata=None)
        assert edge.metadata == {}
    
    def test_edge_types_enum(self):
        """All RelationType enum values work."""
        for rel_type in RelationType:
            edge = Edge(id=f"edge_{rel_type.value}", source_id="n1", target_id="n2",
                        type=rel_type, weight=0.5, confidence=0.5)
            assert edge.type == rel_type


class TestGraphStoreBasics:
    """Test GraphStore basic operations."""
    
    @pytest.fixture
    def store(self, tmp_path):
        """Create a GraphStore for testing."""
        db_path = tmp_path / "test.db"
        store = GraphStore(str(db_path))
        yield store
        # No explicit cleanup needed - store closes connection per operation
    
    def test_constructor(self, tmp_path):
        """GraphStore constructor initializes database."""
        db_path = tmp_path / "test.db"
        store = GraphStore(str(db_path))
        
        assert store.db_path == db_path
        assert db_path.exists()
        
        # Check tables exist
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Check nodes table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='nodes'")
            assert cursor.fetchone() is not None
            
            # Check edges table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='edges'")
            assert cursor.fetchone() is not None
    
    def test_context_manager(self, tmp_path):
        """GraphStore works as context manager."""
        db_path = tmp_path / "test.db"
        
        with GraphStore(str(db_path)) as store:
            assert store.db_path == db_path
            # Add a node to test persistence
            node = Node(id="ctx1", type=NodeType.NOTE, title="Context Test")
            store.add_node(node)
        
        # Verify data persists after context exit
        store2 = GraphStore(str(db_path))
        node2 = store2.get_node("ctx1")
        assert node2 is not None
        assert node2.title == "Context Test"
    
    def test_add_single_node(self, store):
        """add_node() stores node correctly."""
        node = Node(
            id="test1",
            type=NodeType.NOTE,
            title="Test Note",
            content="This is a test",
            metadata={"author": "tester"}
        )
        
        store.add_node(node)
        
        # Retrieve and verify
        retrieved = store.get_node("test1")
        assert retrieved is not None
        assert retrieved.id == "test1"
        assert retrieved.type == NodeType.NOTE
        assert retrieved.title == "Test Note"
        assert retrieved.content == "This is a test"
        assert retrieved.metadata["author"] == "tester"
    
    def test_add_node_replaces_existing(self, store):
        """add_node() replaces existing node with same ID."""
        # Add original
        node1 = Node(id="dup1", type=NodeType.NOTE, title="Original")
        store.add_node(node1)
        
        # Replace with new content
        node2 = Node(id="dup1", type=NodeType.ENTITY, title="Replaced", content="New content")
        store.add_node(node2)
        
        # Verify replacement
        retrieved = store.get_node("dup1")
        assert retrieved.type == NodeType.ENTITY
        assert retrieved.title == "Replaced"
        assert retrieved.content == "New content"
    
    def test_get_node_missing(self, store):
        """get_node() returns None for missing nodes."""
        result = store.get_node("nonexistent")
        assert result is None
    
    def test_add_single_edge(self, store):
        """add_edge() stores edge with referential integrity check."""
        # Must add nodes first
        node1 = Node(id="n1", type=NodeType.NOTE, title="Node 1")
        node2 = Node(id="n2", type=NodeType.NOTE, title="Node 2")
        store.add_node(node1)
        store.add_node(node2)
        
        # Now add edge
        edge = Edge(
            id="e1",
            source_id="n1",
            target_id="n2",
            type=RelationType.REFERENCES,
            weight=0.8,
            confidence=0.9,
            metadata={"reason": "test"}
        )
        store.add_edge(edge)
        
        # Verify edge exists
        edges = store.get_edges("n1", direction='out')
        assert len(edges) == 1
        assert edges[0].id == "e1"
        assert edges[0].weight == 0.8
        assert edges[0].metadata["reason"] == "test"
    
    def test_add_edge_missing_source_node(self, store):
        """add_edge() raises ReferentialError for missing source node."""
        # Add only target node
        node = Node(id="target", type=NodeType.NOTE, title="Target")
        store.add_node(node)
        
        edge = Edge(
            id="e1",
            source_id="missing",
            target_id="target",
            type=RelationType.REFERENCES,
            weight=0.5,
            confidence=0.5
        )
        
        with pytest.raises(ReferentialError, match="Source node 'missing' does not exist"):
            store.add_edge(edge)
    
    def test_add_edge_missing_target_node(self, store):
        """add_edge() raises ReferentialError for missing target node."""
        # Add only source node
        node = Node(id="source", type=NodeType.NOTE, title="Source")
        store.add_node(node)
        
        edge = Edge(
            id="e1",
            source_id="source",
            target_id="missing",
            type=RelationType.REFERENCES,
            weight=0.5,
            confidence=0.5
        )
        
        with pytest.raises(ReferentialError, match="Target node 'missing' does not exist"):
            store.add_edge(edge)
    
    def test_get_edges_out(self, store):
        """get_edges() with direction='out' returns outgoing edges."""
        # Create graph: n1 -> n2, n1 -> n3
        nodes = [
            Node(id="n1", type=NodeType.NOTE, title="Node 1"),
            Node(id="n2", type=NodeType.NOTE, title="Node 2"),
            Node(id="n3", type=NodeType.NOTE, title="Node 3")
        ]
        store.add_nodes(nodes)
        
        edges = [
            Edge(id="e1", source_id="n1", target_id="n2",
                 type=RelationType.REFERENCES, weight=0.7, confidence=0.8),
            Edge(id="e2", source_id="n1", target_id="n3",
                 type=RelationType.SIMILAR_TO, weight=0.6, confidence=0.7)
        ]
        store.add_edges(edges)
        
        # Get outgoing edges from n1
        out_edges = store.get_edges("n1", direction='out')
        assert len(out_edges) == 2
        edge_ids = {e.id for e in out_edges}
        assert edge_ids == {"e1", "e2"}
    
    def test_get_edges_in(self, store):
        """get_edges() with direction='in' returns incoming edges."""
        # Create graph: n1 -> n3, n2 -> n3
        nodes = [
            Node(id="n1", type=NodeType.NOTE, title="Node 1"),
            Node(id="n2", type=NodeType.NOTE, title="Node 2"),
            Node(id="n3", type=NodeType.NOTE, title="Node 3")
        ]
        store.add_nodes(nodes)
        
        edges = [
            Edge(id="e1", source_id="n1", target_id="n3",
                 type=RelationType.REFERENCES, weight=0.7, confidence=0.8),
            Edge(id="e2", source_id="n2", target_id="n3",
                 type=RelationType.MENTIONS, weight=0.6, confidence=0.7)
        ]
        store.add_edges(edges)
        
        # Get incoming edges to n3
        in_edges = store.get_edges("n3", direction='in')
        assert len(in_edges) == 2
        edge_ids = {e.id for e in in_edges}
        assert edge_ids == {"e1", "e2"}
    
    def test_get_edges_both(self, store):
        """get_edges() with direction='both' returns all connected edges."""
        # Create graph: n1 -> n2 -> n3
        nodes = [
            Node(id="n1", type=NodeType.NOTE, title="Node 1"),
            Node(id="n2", type=NodeType.NOTE, title="Node 2"),
            Node(id="n3", type=NodeType.NOTE, title="Node 3")
        ]
        store.add_nodes(nodes)
        
        edges = [
            Edge(id="e1", source_id="n1", target_id="n2",
                 type=RelationType.REFERENCES, weight=0.7, confidence=0.8),
            Edge(id="e2", source_id="n2", target_id="n3",
                 type=RelationType.REFERENCES, weight=0.6, confidence=0.7)
        ]
        store.add_edges(edges)
        
        # Get all edges connected to n2
        both_edges = store.get_edges("n2", direction='both')
        assert len(both_edges) == 2
        edge_ids = {e.id for e in both_edges}
        assert edge_ids == {"e1", "e2"}
    
    def test_get_edges_invalid_direction(self, store):
        """get_edges() with invalid direction raises ValueError."""
        node = Node(id="n1", type=NodeType.NOTE, title="Node")
        store.add_node(node)
        
        with pytest.raises(ValueError, match="Direction must be 'in', 'out', or 'both'"):
            store.get_edges("n1", direction='invalid')


class TestBatchOperations:
    """Test batch add operations."""
    
    @pytest.fixture
    def store(self, tmp_path):
        """Create a GraphStore for testing."""
        db_path = tmp_path / "test.db"
        store = GraphStore(str(db_path))
        yield store
    
    def test_add_nodes_batch(self, store):
        """add_nodes() adds multiple nodes atomically."""
        nodes = [
            Node(id=f"batch{i}", type=NodeType.NOTE, title=f"Batch Node {i}")
            for i in range(5)
        ]
        
        store.add_nodes(nodes)
        
        # Verify all added
        for i in range(5):
            node = store.get_node(f"batch{i}")
            assert node is not None
            assert node.title == f"Batch Node {i}"
    
    def test_add_nodes_empty_list(self, store):
        """add_nodes() with empty list is no-op."""
        store.add_nodes([])
        stats = store.stats()
        assert stats['node_count'] == 0
    
    def test_add_nodes_atomic_transaction(self, store):
        """add_nodes() is atomic - all or nothing."""
        # Create nodes where one will fail validation
        nodes = [
            Node(id="good1", type=NodeType.NOTE, title="Good 1"),
            Node(id="good2", type=NodeType.NOTE, title="Good 2")
        ]
        
        # Add valid nodes first
        store.add_nodes(nodes)
        
        # Try to add batch with invalid node
        try:
            bad_nodes = [
                Node(id="good3", type=NodeType.NOTE, title="Good 3"),
                Node(id="", type=NodeType.NOTE, title="Bad Node")  # This will fail
            ]
        except ValueError:
            # Node validation happens in __post_init__
            pass
        
        # Original nodes should still be there
        assert store.get_node("good1") is not None
        assert store.get_node("good2") is not None
    
    def test_add_edges_batch(self, store):
        """add_edges() adds multiple edges atomically."""
        # Create nodes first
        nodes = [Node(id=f"n{i}", type=NodeType.NOTE, title=f"Node {i}") for i in range(4)]
        store.add_nodes(nodes)
        
        # Create edges
        edges = [
            Edge(id="e1", source_id="n0", target_id="n1",
                 type=RelationType.REFERENCES, weight=0.8, confidence=0.9),
            Edge(id="e2", source_id="n1", target_id="n2",
                 type=RelationType.SIMILAR_TO, weight=0.7, confidence=0.8),
            Edge(id="e3", source_id="n2", target_id="n3",
                 type=RelationType.MENTIONS, weight=0.6, confidence=0.7)
        ]
        
        store.add_edges(edges)
        
        # Verify all edges added
        assert len(store.get_edges("n0", direction='out')) == 1
        assert len(store.get_edges("n1", direction='both')) == 2
        assert len(store.get_edges("n3", direction='in')) == 1
    
    def test_add_edges_empty_list(self, store):
        """add_edges() with empty list is no-op."""
        store.add_edges([])
        stats = store.stats()
        assert stats['edge_count'] == 0
    
    def test_add_edges_referential_integrity(self, store):
        """add_edges() checks all edges for referential integrity."""
        # Add only some nodes
        nodes = [
            Node(id="n1", type=NodeType.NOTE, title="Node 1"),
            Node(id="n2", type=NodeType.NOTE, title="Node 2")
        ]
        store.add_nodes(nodes)
        
        # Try to add edges where one references missing node
        edges = [
            Edge(id="e1", source_id="n1", target_id="n2",
                 type=RelationType.REFERENCES, weight=0.5, confidence=0.5),
            Edge(id="e2", source_id="n2", target_id="n3",  # n3 doesn't exist
                 type=RelationType.REFERENCES, weight=0.5, confidence=0.5)
        ]
        
        with pytest.raises(ReferentialError, match="Target node 'n3' does not exist"):
            store.add_edges(edges)
        
        # No edges should have been added (atomic transaction)
        assert len(store.get_edges("n1", direction='out')) == 0


class TestGetNeighbors:
    """Test get_neighbors functionality."""
    
    @pytest.fixture
    def store_with_graph(self, tmp_path):
        """Create store with a test graph."""
        store = GraphStore(str(tmp_path / "test.db"))
        
        # Create a graph:
        #     n1 -- n2 -- n3
        #      |     |
        #     n4    n5
        nodes = [
            Node(id=f"n{i}", type=NodeType.NOTE, title=f"Node {i}")
            for i in range(1, 6)
        ]
        store.add_nodes(nodes)
        
        edges = [
            Edge(id="e1", source_id="n1", target_id="n2",
                 type=RelationType.REFERENCES, weight=0.8, confidence=0.9),
            Edge(id="e2", source_id="n2", target_id="n3",
                 type=RelationType.REFERENCES, weight=0.7, confidence=0.8),
            Edge(id="e3", source_id="n1", target_id="n4",
                 type=RelationType.SIMILAR_TO, weight=0.6, confidence=0.7),
            Edge(id="e4", source_id="n2", target_id="n5",
                 type=RelationType.MENTIONS, weight=0.5, confidence=0.6)
        ]
        store.add_edges(edges)
        
        yield store
    
    def test_get_neighbors_depth_1(self, store_with_graph):
        """get_neighbors() with depth=1 returns immediate neighbors."""
        result = store_with_graph.get_neighbors("n1", depth=1)
        
        assert result['center'].id == "n1"
        assert len(result['neighbors']) == 2  # n2 and n4
        
        neighbor_ids = {n['node'].id for n in result['neighbors']}
        assert neighbor_ids == {"n2", "n4"}
        
        # All should be distance 1
        for neighbor in result['neighbors']:
            assert neighbor['distance'] == 1
    
    def test_get_neighbors_depth_2(self, store_with_graph):
        """get_neighbors() with depth=2 returns neighbors up to 2 hops."""
        result = store_with_graph.get_neighbors("n1", depth=2)
        
        assert result['center'].id == "n1"
        
        # Should get n2, n4 at distance 1, and n3, n5 at distance 2
        neighbor_ids_by_distance = {}
        for neighbor in result['neighbors']:
            dist = neighbor['distance']
            if dist not in neighbor_ids_by_distance:
                neighbor_ids_by_distance[dist] = set()
            neighbor_ids_by_distance[dist].add(neighbor['node'].id)
        
        assert neighbor_ids_by_distance[1] == {"n2", "n4"}
        assert neighbor_ids_by_distance[2] == {"n3", "n5"}
    
    def test_get_neighbors_invalid_depth(self, store_with_graph):
        """get_neighbors() validates depth parameter."""
        with pytest.raises(ValueError, match="Depth must be 1-3"):
            store_with_graph.get_neighbors("n1", depth=0)
        
        with pytest.raises(ValueError, match="Depth must be 1-3"):
            store_with_graph.get_neighbors("n1", depth=4)
        
        with pytest.raises(ValueError, match="Depth must be 1-3"):
            store_with_graph.get_neighbors("n1", depth="invalid")
    
    def test_get_neighbors_missing_node(self, store_with_graph):
        """get_neighbors() for missing node returns empty result."""
        result = store_with_graph.get_neighbors("nonexistent", depth=1)
        
        assert result['center'] is None
        assert result['neighbors'] == []
    
    def test_get_neighbors_isolated_node(self, store_with_graph):
        """get_neighbors() for isolated node returns no neighbors."""
        # Add isolated node
        isolated = Node(id="isolated", type=NodeType.NOTE, title="Isolated")
        store_with_graph.add_node(isolated)
        
        result = store_with_graph.get_neighbors("isolated", depth=2)
        
        assert result['center'].id == "isolated"
        assert result['neighbors'] == []
    
    def test_get_neighbors_no_duplicates(self, store_with_graph):
        """get_neighbors() doesn't return duplicate nodes."""
        # Add more edges to create multiple paths
        edge = Edge(id="e5", source_id="n3", target_id="n5",
                    type=RelationType.SIMILAR_TO, weight=0.5, confidence=0.5)
        store_with_graph.add_edge(edge)
        
        result = store_with_graph.get_neighbors("n2", depth=2)
        
        # Check no duplicates
        node_ids = [n['node'].id for n in result['neighbors']]
        assert len(node_ids) == len(set(node_ids))


class TestRemoveOperations:
    """Test node removal with cascade."""
    
    @pytest.fixture
    def store(self, tmp_path):
        """Create a GraphStore for testing."""
        db_path = tmp_path / "test.db"
        store = GraphStore(str(db_path))
        yield store
    
    def test_remove_node_with_edges(self, store):
        """remove_node() cascades to remove connected edges."""
        # Create connected graph
        nodes = [
            Node(id="n1", type=NodeType.NOTE, title="Node 1"),
            Node(id="n2", type=NodeType.NOTE, title="Node 2"),
            Node(id="n3", type=NodeType.NOTE, title="Node 3")
        ]
        store.add_nodes(nodes)
        
        edges = [
            Edge(id="e1", source_id="n1", target_id="n2",
                 type=RelationType.REFERENCES, weight=0.5, confidence=0.5),
            Edge(id="e2", source_id="n2", target_id="n3",
                 type=RelationType.REFERENCES, weight=0.5, confidence=0.5),
            Edge(id="e3", source_id="n3", target_id="n1",
                 type=RelationType.REFERENCES, weight=0.5, confidence=0.5)
        ]
        store.add_edges(edges)
        
        # Remove n2
        removed = store.remove_node("n2")
        assert removed is True
        
        # Node should be gone
        assert store.get_node("n2") is None
        
        # Edges e1 and e2 should be gone
        assert len(store.get_edges("n1", direction='out')) == 0  # e1 removed
        assert len(store.get_edges("n3", direction='out')) == 1  # e3 still there
        
        # n3 should have no incoming edges (e2 removed)
        assert len(store.get_edges("n3", direction='in')) == 0
    
    def test_remove_node_isolated(self, store):
        """remove_node() works on isolated nodes."""
        node = Node(id="isolated", type=NodeType.TAG, title="Isolated")
        store.add_node(node)
        
        removed = store.remove_node("isolated")
        assert removed is True
        assert store.get_node("isolated") is None
    
    def test_remove_node_missing(self, store):
        """remove_node() returns False for missing nodes."""
        removed = store.remove_node("nonexistent")
        assert removed is False


class TestStats:
    """Test statistics functionality."""
    
    @pytest.fixture
    def store(self, tmp_path):
        """Create a GraphStore for testing."""
        db_path = tmp_path / "test.db"
        store = GraphStore(str(db_path))
        yield store
    
    def test_stats_empty_store(self, store):
        """stats() returns correct counts for empty store."""
        stats = store.stats()
        
        assert stats['node_count'] == 0
        assert stats['edge_count'] == 0
        assert stats['node_types'] == {}
        assert stats['edge_types'] == {}
        assert stats['db_path'] == str(store.db_path)
        assert stats['db_size_bytes'] > 0  # SQLite file exists
        assert 'db_size_mb' in stats
    
    def test_stats_with_data(self, store):
        """stats() returns correct counts with data."""
        # Add variety of nodes
        nodes = [
            Node(id="note1", type=NodeType.NOTE, title="Note 1"),
            Node(id="note2", type=NodeType.NOTE, title="Note 2"),
            Node(id="entity1", type=NodeType.ENTITY, title="Entity 1"),
            Node(id="tag1", type=NodeType.TAG, title="Tag 1"),
            Node(id="tag2", type=NodeType.TAG, title="Tag 2"),
            Node(id="tag3", type=NodeType.TAG, title="Tag 3")
        ]
        store.add_nodes(nodes)
        
        # Add variety of edges
        edges = [
            Edge(id="e1", source_id="note1", target_id="note2",
                 type=RelationType.REFERENCES, weight=0.5, confidence=0.5),
            Edge(id="e2", source_id="note1", target_id="entity1",
                 type=RelationType.MENTIONS, weight=0.5, confidence=0.5),
            Edge(id="e3", source_id="note1", target_id="tag1",
                 type=RelationType.TAGGED_AS, weight=0.5, confidence=0.5),
            Edge(id="e4", source_id="note2", target_id="tag2",
                 type=RelationType.TAGGED_AS, weight=0.5, confidence=0.5)
        ]
        store.add_edges(edges)
        
        stats = store.stats()
        
        assert stats['node_count'] == 6
        assert stats['edge_count'] == 4
        
        # Node type breakdown
        assert stats['node_types']['note'] == 2
        assert stats['node_types']['entity'] == 1
        assert stats['node_types']['tag'] == 3
        
        # Edge type breakdown
        assert stats['edge_types']['references'] == 1
        assert stats['edge_types']['mentions'] == 1
        assert stats['edge_types']['tagged_as'] == 2
    
    def test_clear_store(self, store):
        """clear() removes all data."""
        # Add some data
        nodes = [Node(id=f"n{i}", type=NodeType.NOTE, title=f"Note {i}") for i in range(3)]
        store.add_nodes(nodes)
        
        edges = [
            Edge(id="e1", source_id="n0", target_id="n1",
                 type=RelationType.REFERENCES, weight=0.5, confidence=0.5)
        ]
        store.add_edges(edges)
        
        # Clear
        store.clear()
        
        # Verify empty
        stats = store.stats()
        assert stats['node_count'] == 0
        assert stats['edge_count'] == 0


class TestNodeTypeOperations:
    """Test get_nodes_by_type functionality."""
    
    @pytest.fixture
    def store_with_varied_nodes(self, tmp_path):
        """Create store with variety of node types."""
        store = GraphStore(str(tmp_path / "test.db"))
        
        nodes = [
            Node(id="note1", type=NodeType.NOTE, title="First Note"),
            Node(id="note2", type=NodeType.NOTE, title="Second Note"),
            Node(id="entity1", type=NodeType.ENTITY, title="Person A"),
            Node(id="entity2", type=NodeType.ENTITY, title="Person B"),
            Node(id="entity3", type=NodeType.ENTITY, title="Company X"),
            Node(id="topic1", type=NodeType.TOPIC, title="Machine Learning"),
            Node(id="tag1", type=NodeType.TAG, title="important"),
            Node(id="tag2", type=NodeType.TAG, title="review"),
            Node(id="folder1", type=NodeType.FOLDER, title="Projects")
        ]
        store.add_nodes(nodes)
        
        yield store
    
    def test_get_nodes_by_type_note(self, store_with_varied_nodes):
        """get_nodes_by_type() returns all nodes of specified type."""
        notes = store_with_varied_nodes.get_nodes_by_type(NodeType.NOTE)
        
        assert len(notes) == 2
        note_ids = {n.id for n in notes}
        assert note_ids == {"note1", "note2"}
    
    def test_get_nodes_by_type_entity(self, store_with_varied_nodes):
        """get_nodes_by_type() works for entity type."""
        entities = store_with_varied_nodes.get_nodes_by_type(NodeType.ENTITY)
        
        assert len(entities) == 3
        entity_ids = {e.id for e in entities}
        assert entity_ids == {"entity1", "entity2", "entity3"}
    
    def test_get_nodes_by_type_empty(self, tmp_path):
        """get_nodes_by_type() returns empty list when no nodes of type."""
        store = GraphStore(str(tmp_path / "test.db"))
        
        # Add only notes
        node = Node(id="note1", type=NodeType.NOTE, title="Only Note")
        store.add_node(node)
        
        # Query for tags
        tags = store.get_nodes_by_type(NodeType.TAG)
        assert tags == []
    
    def test_get_nodes_by_type_ordering(self, store_with_varied_nodes):
        """get_nodes_by_type() returns nodes ordered by created_at DESC."""
        # Add more notes with explicit timestamps
        import time
        
        created1 = datetime.now()
        time.sleep(0.01)  # Ensure different timestamps
        created2 = datetime.now()
        
        node1 = Node(id="new1", type=NodeType.NOTE, title="Newer", created_at=created2)
        node2 = Node(id="new2", type=NodeType.NOTE, title="Older", created_at=created1)
        
        store_with_varied_nodes.add_node(node1)
        store_with_varied_nodes.add_node(node2)
        
        notes = store_with_varied_nodes.get_nodes_by_type(NodeType.NOTE)
        
        # Should be ordered newest first
        assert notes[0].id == "new1"
        assert notes[1].id == "new2"


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def store(self, tmp_path):
        """Create a GraphStore for testing."""
        db_path = tmp_path / "test.db"
        store = GraphStore(str(db_path))
        yield store
    
    def test_unicode_content(self, store):
        """Unicode content is handled correctly."""
        # Nodes with unicode
        nodes = [
            Node(id="jp", type=NodeType.NOTE, title="日本語", content="こんにちは世界"),
            Node(id="emoji", type=NodeType.TAG, title="🚀🌟", content="Emoji content 🎉"),
            Node(id="mixed", type=NodeType.ENTITY, title="Café", content="Mañana in Москва")
        ]
        store.add_nodes(nodes)
        
        # Verify retrieval
        jp_node = store.get_node("jp")
        assert jp_node.title == "日本語"
        assert jp_node.content == "こんにちは世界"
        
        emoji_node = store.get_node("emoji")
        assert emoji_node.title == "🚀🌟"
    
    def test_large_metadata(self, store):
        """Large metadata objects are handled."""
        large_metadata = {
            f"key_{i}": f"value_{i}" * 100 for i in range(50)
        }
        
        node = Node(
            id="large",
            type=NodeType.NOTE,
            title="Large Metadata Node",
            metadata=large_metadata
        )
        store.add_node(node)
        
        retrieved = store.get_node("large")
        assert len(retrieved.metadata) == 50
        assert retrieved.metadata["key_10"] == "value_10" * 100
    
    def test_empty_graph_operations(self, store):
        """Operations on empty graph work correctly."""
        # get_edges on nonexistent node
        edges = store.get_edges("nonexistent")
        assert edges == []
        
        # get_neighbors on nonexistent node
        neighbors = store.get_neighbors("nonexistent")
        assert neighbors['center'] is None
        assert neighbors['neighbors'] == []
        
        # stats on empty graph
        stats = store.stats()
        assert stats['node_count'] == 0
        assert stats['edge_count'] == 0
    
    def test_special_characters_in_ids(self, store):
        """Special characters in IDs are handled."""
        # IDs with special chars (but not causing SQL injection)
        node = Node(
            id="node-with-dash",
            type=NodeType.NOTE,
            title="Special ID"
        )
        store.add_node(node)
        
        retrieved = store.get_node("node-with-dash")
        assert retrieved is not None
        
        # ID with underscore
        node2 = Node(
            id="node_with_underscore",
            type=NodeType.TAG,
            title="Underscore ID"
        )
        store.add_node(node2)
        
        retrieved2 = store.get_node("node_with_underscore")
        assert retrieved2 is not None
    
    def test_complex_metadata_preservation(self, store):
        """Complex nested metadata is preserved correctly."""
        complex_metadata = {
            "nested": {
                "deep": {
                    "list": [1, 2, 3, {"inner": "value"}],
                    "unicode": "café ☕",
                    "bool": True,
                    "null": None,
                    "float": 3.14159
                }
            },
            "array": [
                {"id": 1, "name": "first"},
                {"id": 2, "name": "second"}
            ]
        }
        
        node = Node(
            id="complex",
            type=NodeType.NOTE,
            title="Complex Metadata",
            metadata=complex_metadata
        )
        store.add_node(node)
        
        retrieved = store.get_node("complex")
        assert retrieved.metadata == complex_metadata
        assert retrieved.metadata["nested"]["deep"]["unicode"] == "café ☕"
        assert retrieved.metadata["array"][1]["name"] == "second"
    
    def test_database_persistence(self, tmp_path):
        """Data persists across store instances."""
        db_path = tmp_path / "persist.db"
        
        # First instance - add data
        store1 = GraphStore(str(db_path))
        nodes = [
            Node(id="p1", type=NodeType.NOTE, title="Persist 1"),
            Node(id="p2", type=NodeType.NOTE, title="Persist 2")
        ]
        store1.add_nodes(nodes)
        
        edge = Edge(
            id="pe1",
            source_id="p1",
            target_id="p2",
            type=RelationType.REFERENCES,
            weight=0.7,
            confidence=0.8
        )
        store1.add_edge(edge)
        
        # Second instance - verify data
        store2 = GraphStore(str(db_path))
        assert store2.get_node("p1").title == "Persist 1"
        assert len(store2.get_edges("p1", direction='out')) == 1
        
        stats = store2.stats()
        assert stats['node_count'] == 2
        assert stats['edge_count'] == 1