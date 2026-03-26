"""
Velociragtor Phase 6 Graph Foundation - Graph models and storage.

Production-grade graph data models and SQLite storage for knowledge
representation. Designed for high-throughput node/edge operations with
bulletproof consistency guarantees and transaction safety.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("velocirag.graph")


class NodeType(Enum):
    """Classification taxonomy for graph nodes."""
    NOTE = "note"
    ENTITY = "entity"
    TOPIC = "topic"
    TAG = "tag"
    FOLDER = "folder"


class RelationType(Enum):
    """Relationship taxonomy for graph edges."""
    REFERENCES = "references"
    TAGGED_AS = "tagged_as"
    SIMILAR_TO = "similar_to"
    DISCUSSES = "discusses"
    MENTIONS = "mentions"
    PART_OF = "part_of"
    TEMPORAL = "temporal"


class GraphStoreError(Exception):
    """Base exception for storage failures."""
    pass


class ReferentialError(GraphStoreError):
    """Subclass for foreign key violations."""
    pass


class ValidationError(GraphStoreError):
    """Subclass for data validation failures."""
    pass


@dataclass
class Node:
    """
    Immutable node representation in the knowledge graph.
    
    Every node is a truth anchor — a fixed point in the knowledge space
    that other entities can reference and depend upon.
    """
    id: str
    type: NodeType
    title: str
    content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate and normalize on construction."""
        if not self.id or not self.id.strip():
            raise ValueError("Node ID cannot be empty")
        if not self.title or not self.title.strip():
            raise ValueError("Node title cannot be empty")
        if not isinstance(self.type, NodeType):
            raise ValueError(f"Node type must be NodeType enum, got {type(self.type)}")
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})
        if self.created_at is None:
            object.__setattr__(self, 'created_at', datetime.now())


@dataclass
class Edge:
    """
    Weighted relationship between nodes with confidence scoring.
    
    Each edge represents a calculated truth — evidence of relationship
    strength backed by algorithmic analysis and human validation.
    """
    id: str
    source_id: str
    target_id: str
    type: RelationType
    weight: float
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate and normalize on construction."""
        if not self.id or not self.id.strip():
            raise ValueError("Edge ID cannot be empty")
        if not self.source_id or not self.source_id.strip():
            raise ValueError("Source ID cannot be empty")
        if not self.target_id or not self.target_id.strip():
            raise ValueError("Target ID cannot be empty")
        # Self-loops are filtered at the store level, not here
        if not isinstance(self.type, RelationType):
            raise ValueError(f"Edge type must be RelationType enum, got {type(self.type)}")
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"Weight must be 0.0-1.0, got {self.weight}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be 0.0-1.0, got {self.confidence}")
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})
        if self.created_at is None:
            object.__setattr__(self, 'created_at', datetime.now())


class GraphQuerier:
    """
    Query engine for the knowledge graph.
    
    Path finding, similarity search, and relationship exploration.
    Built for speed and insight discovery.
    """
    
    def __init__(self, store: 'GraphStore'):
        """Initialize querier with graph store."""
        self.store = store
    
    @contextmanager
    def _connect(self):
        """Get a SQLite connection that's properly closed after use."""
        conn = sqlite3.connect(self.store.db_path)
        try:
            yield conn
        finally:
            conn.close()
    
    def find_connections(self, node_title: str, depth: int = 2) -> Dict[str, Any]:
        """
        Find all connections to a node by title within specified depth.
        
        Args:
            node_title: Title of the node to explore
            depth: Maximum connection depth (1-3)
            
        Returns:
            Dictionary with connection map and statistics
        """
        # First find the node by title
        target_node = None
        try:
            with self._connect() as conn:
                row = conn.execute('''
                    SELECT id FROM nodes WHERE title LIKE ? LIMIT 1
                ''', (f'%{node_title}%',)).fetchone()
                
                if row:
                    target_node = self.store.get_node(row[0])
        except sqlite3.Error:
            return {'error': 'Database query failed'}
        
        if not target_node:
            return {'error': f"Node with title '{node_title}' not found"}
        
        # Get neighborhood
        neighborhood = self.store.get_neighbors(target_node.id, depth=depth)
        
        # Organize by relationship type
        connections_by_type = {}
        for neighbor_data in neighborhood['neighbors']:
            edge_type = neighbor_data['edge'].type.value
            if edge_type not in connections_by_type:
                connections_by_type[edge_type] = []
            connections_by_type[edge_type].append({
                'node': neighbor_data['node'].title,
                'distance': neighbor_data['distance'],
                'weight': neighbor_data['edge'].weight,
                'confidence': neighbor_data['edge'].confidence
            })
        
        return {
            'center_node': target_node.title,
            'total_connections': len(neighborhood['neighbors']),
            'connections_by_type': connections_by_type,
            'max_depth': depth
        }
    
    def find_similar(self, node_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find nodes similar to the given node.
        
        Args:
            node_id: ID of the source node
            limit: Maximum number of similar nodes to return
            
        Returns:
            List of similar nodes with similarity scores
        """
        try:
            similar_nodes = []
            with self._connect() as conn:
                # Find edges of type SIMILAR_TO
                rows = conn.execute('''
                    SELECT target_id, weight, confidence 
                    FROM edges 
                    WHERE source_id = ? AND type = ? 
                    ORDER BY weight DESC 
                    LIMIT ?
                ''', (node_id, RelationType.SIMILAR_TO.value, limit)).fetchall()
                
                for target_id, weight, confidence in rows:
                    target_node = self.store.get_node(target_id)
                    if target_node:
                        similar_nodes.append({
                            'node': target_node,
                            'similarity': weight,
                            'confidence': confidence
                        })
            
            return similar_nodes
            
        except sqlite3.Error:
            return []
    
    def find_path(self, source_id: str, target_id: str) -> Optional[Dict[str, Any]]:
        """
        Find shortest path between two nodes.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            
        Returns:
            Path information or None if no path exists
        """
        if source_id == target_id:
            return {'path': [source_id], 'distance': 0, 'edges': []}
        
        # BFS for shortest path
        from collections import deque
        queue = deque([(source_id, [source_id], [])])
        visited = {source_id}
        
        try:
            for _ in range(50):  # Limit search depth
                if not queue:
                    break
                
                current_id, path, edges = queue.popleft()
                
                # Get neighbors
                current_edges = self.store.get_edges(current_id, direction='both')
                
                for edge in current_edges:
                    neighbor_id = edge.target_id if edge.source_id == current_id else edge.source_id
                    
                    if neighbor_id == target_id:
                        # Found path
                        return {
                            'path': path + [neighbor_id],
                            'distance': len(path),
                            'edges': edges + [edge]
                        }
                    
                    if neighbor_id not in visited and len(path) < 6:  # Max path length
                        visited.add(neighbor_id)
                        queue.append((neighbor_id, path + [neighbor_id], edges + [edge]))
            
            return None  # No path found
            
        except Exception:
            return None
    
    def get_topic_web(self, topic: str) -> Dict[str, Any]:
        """
        Get all nodes related to a specific topic.
        
        Args:
            topic: Topic name to explore
            
        Returns:
            Topic web with related nodes and connections
        """
        try:
            topic_nodes = []
            with self._connect() as conn:
                # Find topic node
                topic_row = conn.execute('''
                    SELECT id FROM nodes WHERE type = ? AND title LIKE ?
                ''', (NodeType.TOPIC.value, f'%{topic}%')).fetchone()
                
                if not topic_row:
                    return {'error': f"Topic '{topic}' not found"}
                
                topic_id = topic_row[0]
                
                # Find all nodes connected to this topic
                rows = conn.execute('''
                    SELECT DISTINCT n.id, n.title, n.type, e.weight
                    FROM nodes n
                    JOIN edges e ON (n.id = e.source_id OR n.id = e.target_id)
                    WHERE (e.source_id = ? OR e.target_id = ?) AND n.id != ?
                    ORDER BY e.weight DESC
                ''', (topic_id, topic_id, topic_id)).fetchall()
                
                for node_id, title, node_type, weight in rows:
                    topic_nodes.append({
                        'id': node_id,
                        'title': title,
                        'type': node_type,
                        'connection_strength': weight
                    })
            
            return {
                'topic': topic,
                'related_nodes': topic_nodes,
                'node_count': len(topic_nodes)
            }
            
        except sqlite3.Error:
            return {'error': 'Database query failed'}
    
    def get_hub_nodes(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most connected nodes in the graph.
        
        Args:
            limit: Maximum number of hub nodes to return
            
        Returns:
            List of hub nodes with connection counts
        """
        try:
            hub_nodes = []
            with self._connect() as conn:
                # Count connections for each node
                rows = conn.execute('''
                    SELECT n.id, n.title, n.type,
                           COUNT(DISTINCT e.id) as connection_count
                    FROM nodes n
                    LEFT JOIN edges e ON (n.id = e.source_id OR n.id = e.target_id)
                    GROUP BY n.id, n.title, n.type
                    ORDER BY connection_count DESC
                    LIMIT ?
                ''', (limit,)).fetchall()
                
                for node_id, title, node_type, count in rows:
                    node = self.store.get_node(node_id)
                    importance = node.metadata.get('importance_score', 0.0) if node and node.metadata else 0.0
                    
                    hub_nodes.append({
                        'node_id': node_id,
                        'title': title,
                        'type': node_type,
                        'connection_count': count,
                        'importance_score': importance
                    })
            
            return hub_nodes
            
        except sqlite3.Error:
            return []


class GraphStore:
    """
    SQLite-backed graph storage with transaction safety and consistency guarantees.
    
    This is the single source of truth for all graph data. Every node and edge
    passes through this checkpoint. Every query draws from this well.
    """
    
    def __init__(self, db_path: str):
        """
        Initialize graph storage.
        
        Args:
            db_path: Path to SQLite database file (created if not exists)
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        logger.info(f"GraphStore initialized: {self.db_path}")
    
    def __enter__(self) -> 'GraphStore':
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with cleanup."""
        # No explicit cleanup needed - connections are per-operation
        pass
    
    @contextmanager
    def _connect(self):
        """Get a SQLite connection that's properly closed after use."""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()
    
    def add_node(self, node: Node) -> None:
        """
        Add single node to graph with validation.
        
        Args:
            node: Node instance to store
            
        Raises:
            ValueError: Invalid node data
            GraphStoreError: Database operation failed
        """
        self.add_nodes([node])
    
    def add_nodes(self, nodes: List[Node]) -> None:
        """
        Batch add multiple nodes with atomic transaction.
        
        All nodes added or none added. No partial failures.
        
        Args:
            nodes: List of Node instances
            
        Raises:
            ValueError: Invalid node data in batch
            GraphStoreError: Database operation failed
        """
        if not nodes:
            return
        
        try:
            with self._transaction() as conn:
                for node in nodes:
                    # Node validation happens in __post_init__
                    metadata_json = json.dumps(node.metadata) if node.metadata else "{}"
                    created_at_iso = node.created_at.isoformat() if node.created_at else datetime.now().isoformat()
                    
                    conn.execute('''
                        INSERT OR REPLACE INTO nodes (id, type, title, content, metadata, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        node.id,
                        node.type.value,
                        node.title,
                        node.content,
                        metadata_json,
                        created_at_iso
                    ))
                    
        except sqlite3.Error as e:
            raise GraphStoreError(f"Failed to add nodes: {e}")
    
    def add_edge(self, edge: Edge) -> None:
        """
        Add single edge to graph with referential integrity check.
        
        Args:
            edge: Edge instance to store
            
        Raises:
            ValueError: Invalid edge data
            ReferentialError: Source or target node does not exist
            GraphStoreError: Database operation failed
        """
        self.add_edges([edge])
    
    def add_edges(self, edges: List[Edge]) -> None:
        """
        Batch add multiple edges with atomic transaction and integrity check.
        
        Args:
            edges: List of Edge instances
            
        Raises:
            ValueError: Invalid edge data in batch
            ReferentialError: Referenced node does not exist
            GraphStoreError: Database operation failed
        """
        if not edges:
            return
        
        # Filter out self-loops silently
        edges = [e for e in edges if e.source_id != e.target_id]
        if not edges:
            return
        
        try:
            with self._transaction() as conn:
                # Check referential integrity for all edges first
                for edge in edges:
                    # Check source node exists
                    source_exists = conn.execute(
                        'SELECT 1 FROM nodes WHERE id = ?',
                        (edge.source_id,)
                    ).fetchone()
                    if not source_exists:
                        raise ReferentialError(f"Source node '{edge.source_id}' does not exist")
                    
                    # Check target node exists
                    target_exists = conn.execute(
                        'SELECT 1 FROM nodes WHERE id = ?',
                        (edge.target_id,)
                    ).fetchone()
                    if not target_exists:
                        raise ReferentialError(f"Target node '{edge.target_id}' does not exist")
                
                # All checks passed, insert edges
                for edge in edges:
                    metadata_json = json.dumps(edge.metadata) if edge.metadata else "{}"
                    created_at_iso = edge.created_at.isoformat() if edge.created_at else datetime.now().isoformat()
                    
                    conn.execute('''
                        INSERT OR REPLACE INTO edges 
                        (id, source_id, target_id, type, weight, confidence, metadata, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        edge.id,
                        edge.source_id,
                        edge.target_id,
                        edge.type.value,
                        edge.weight,
                        edge.confidence,
                        metadata_json,
                        created_at_iso
                    ))
                    
        except sqlite3.Error as e:
            raise GraphStoreError(f"Failed to add edges: {e}")
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """
        Retrieve node by ID.
        
        Args:
            node_id: Unique node identifier
            
        Returns:
            Node instance or None if not found
        """
        try:
            with self._connect() as conn:
                row = conn.execute('''
                    SELECT id, type, title, content, metadata, created_at
                    FROM nodes WHERE id = ?
                ''', (node_id,)).fetchone()
                
                if not row:
                    return None
                
                id_val, type_val, title, content, metadata_json, created_at_iso = row
                
                # Parse values
                node_type = NodeType(type_val)
                metadata = json.loads(metadata_json) if metadata_json else {}
                created_at = datetime.fromisoformat(created_at_iso) if created_at_iso else None
                
                return Node(
                    id=id_val,
                    type=node_type,
                    title=title,
                    content=content,
                    metadata=metadata,
                    created_at=created_at
                )
                
        except sqlite3.Error as e:
            raise GraphStoreError(f"Failed to get node '{node_id}': {e}")
    
    def get_nodes_by_type(self, node_type: NodeType) -> List[Node]:
        """
        Get all nodes of specific type.
        
        Args:
            node_type: NodeType to filter by
            
        Returns:
            List of Node instances (may be empty)
        """
        try:
            nodes = []
            with self._connect() as conn:
                rows = conn.execute('''
                    SELECT id, type, title, content, metadata, created_at
                    FROM nodes WHERE type = ? ORDER BY created_at DESC
                ''', (node_type.value,)).fetchall()
                
                for row in rows:
                    id_val, type_val, title, content, metadata_json, created_at_iso = row
                    metadata = json.loads(metadata_json) if metadata_json else {}
                    created_at = datetime.fromisoformat(created_at_iso) if created_at_iso else None
                    
                    nodes.append(Node(
                        id=id_val,
                        type=NodeType(type_val),
                        title=title,
                        content=content,
                        metadata=metadata,
                        created_at=created_at
                    ))
                    
            return nodes
            
        except sqlite3.Error as e:
            raise GraphStoreError(f"Failed to get nodes by type '{node_type.value}': {e}")
    
    def get_edges(self, node_id: str, direction: str = 'both') -> List[Edge]:
        """
        Get edges connected to node.
        
        Args:
            node_id: Node to query edges for
            direction: 'in', 'out', or 'both'
            
        Returns:
            List of Edge instances (may be empty)
        """
        if direction not in ('in', 'out', 'both'):
            raise ValueError(f"Direction must be 'in', 'out', or 'both', got '{direction}'")
        
        try:
            edges = []
            with self._connect() as conn:
                if direction == 'out':
                    query = '''
                        SELECT id, source_id, target_id, type, weight, confidence, metadata, created_at
                        FROM edges WHERE source_id = ?
                    '''
                    params = (node_id,)
                elif direction == 'in':
                    query = '''
                        SELECT id, source_id, target_id, type, weight, confidence, metadata, created_at
                        FROM edges WHERE target_id = ?
                    '''
                    params = (node_id,)
                else:  # both
                    query = '''
                        SELECT id, source_id, target_id, type, weight, confidence, metadata, created_at
                        FROM edges WHERE source_id = ? OR target_id = ?
                    '''
                    params = (node_id, node_id)
                
                rows = conn.execute(query, params).fetchall()
                
                for row in rows:
                    id_val, source_id, target_id, type_val, weight, confidence, metadata_json, created_at_iso = row
                    metadata = json.loads(metadata_json) if metadata_json else {}
                    created_at = datetime.fromisoformat(created_at_iso) if created_at_iso else None
                    
                    edges.append(Edge(
                        id=id_val,
                        source_id=source_id,
                        target_id=target_id,
                        type=RelationType(type_val),
                        weight=weight,
                        confidence=confidence,
                        metadata=metadata,
                        created_at=created_at
                    ))
                    
            return edges
            
        except sqlite3.Error as e:
            raise GraphStoreError(f"Failed to get edges for node '{node_id}': {e}")
    
    def get_neighbors(self, node_id: str, depth: int = 1) -> Dict[str, Any]:
        """
        Get neighboring nodes within specified depth.
        
        Args:
            node_id: Starting node
            depth: Maximum traversal depth (1-3)
            
        Returns:
            Dictionary with structure:
            {
                'center': Node,
                'neighbors': [
                    {
                        'node': Node,
                        'edge': Edge,
                        'distance': int
                    }
                ]
            }
        """
        if not isinstance(depth, int) or depth < 1 or depth > 3:
            raise ValueError(f"Depth must be 1-3, got {depth}")
        
        center_node = self.get_node(node_id)
        if not center_node:
            return {'center': None, 'neighbors': []}
        
        try:
            visited = {node_id}
            neighbors = []
            current_level = [(node_id, 0)]
            
            for _ in range(depth):
                next_level = []
                
                for current_id, current_distance in current_level:
                    # Get all edges for current node
                    edges = self.get_edges(current_id, direction='both')
                    
                    for edge in edges:
                        # Determine neighbor node ID
                        if edge.source_id == current_id:
                            neighbor_id = edge.target_id
                        else:
                            neighbor_id = edge.source_id
                        
                        # Skip if already visited
                        if neighbor_id in visited:
                            continue
                        
                        # Get neighbor node
                        neighbor_node = self.get_node(neighbor_id)
                        if neighbor_node:
                            neighbors.append({
                                'node': neighbor_node,
                                'edge': edge,
                                'distance': current_distance + 1
                            })
                            
                            visited.add(neighbor_id)
                            next_level.append((neighbor_id, current_distance + 1))
                
                current_level = next_level
                if not current_level:  # No more neighbors
                    break
            
            return {
                'center': center_node,
                'neighbors': neighbors
            }
            
        except Exception as e:
            raise GraphStoreError(f"Failed to get neighbors for '{node_id}': {e}")
    
    def remove_node(self, node_id: str) -> bool:
        """
        Remove node and all connected edges.
        
        Cascading deletion maintains referential integrity.
        
        Args:
            node_id: Node to remove
            
        Returns:
            True if node existed and was removed, False if not found
        """
        try:
            with self._transaction() as conn:
                # Check if node exists
                node_exists = conn.execute(
                    'SELECT 1 FROM nodes WHERE id = ?',
                    (node_id,)
                ).fetchone()
                
                if not node_exists:
                    return False
                
                # Remove connected edges (foreign key constraints handle this automatically
                # but we do it explicitly for clarity)
                conn.execute(
                    'DELETE FROM edges WHERE source_id = ? OR target_id = ?',
                    (node_id, node_id)
                )
                
                # Remove the node
                conn.execute('DELETE FROM nodes WHERE id = ?', (node_id,))
                
                return True
                
        except sqlite3.Error as e:
            raise GraphStoreError(f"Failed to remove node '{node_id}': {e}")
    
    def stats(self) -> Dict[str, Any]:
        """
        Get comprehensive storage statistics.
        
        Returns:
            Dictionary with counts, types, database metrics
        """
        try:
            with self._connect() as conn:
                # Node counts
                node_count = conn.execute('SELECT COUNT(*) FROM nodes').fetchone()[0]
                
                # Node type breakdown
                node_types = {}
                node_type_rows = conn.execute('''
                    SELECT type, COUNT(*) FROM nodes GROUP BY type
                ''').fetchall()
                for type_val, count in node_type_rows:
                    node_types[type_val] = count
                
                # Edge counts
                edge_count = conn.execute('SELECT COUNT(*) FROM edges').fetchone()[0]
                
                # Edge type breakdown
                edge_types = {}
                edge_type_rows = conn.execute('''
                    SELECT type, COUNT(*) FROM edges GROUP BY type
                ''').fetchall()
                for type_val, count in edge_type_rows:
                    edge_types[type_val] = count
                
                # Database file size
                db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
                
                return {
                    'node_count': node_count,
                    'edge_count': edge_count,
                    'node_types': node_types,
                    'edge_types': edge_types,
                    'db_path': str(self.db_path),
                    'db_size_bytes': db_size,
                    'db_size_mb': round(db_size / (1024 * 1024), 2)
                }
                
        except sqlite3.Error as e:
            raise GraphStoreError(f"Failed to get stats: {e}")
    
    def clear(self) -> None:
        """
        Delete all nodes and edges.
        
        Nuclear option. Use with appropriate caution.
        """
        try:
            with self._transaction() as conn:
                conn.execute('DELETE FROM edges')
                conn.execute('DELETE FROM nodes')
                logger.warning("GraphStore cleared - all data deleted")
                
        except sqlite3.Error as e:
            raise GraphStoreError(f"Failed to clear database: {e}")
    
    def _init_database(self) -> None:
        """Initialize SQLite database and tables."""
        try:
            with self._connect() as conn:
                # Enable foreign key constraints
                conn.execute('PRAGMA foreign_keys = ON')
                
                # Create nodes table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS nodes (
                        id TEXT PRIMARY KEY,
                        type TEXT NOT NULL,
                        title TEXT NOT NULL,
                        content TEXT,
                        metadata TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL
                    )
                ''')
                
                # Create edges table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS edges (
                        id TEXT PRIMARY KEY,
                        source_id TEXT NOT NULL,
                        target_id TEXT NOT NULL,
                        type TEXT NOT NULL,
                        weight REAL NOT NULL,
                        confidence REAL NOT NULL,
                        metadata TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (source_id) REFERENCES nodes(id) ON DELETE CASCADE,
                        FOREIGN KEY (target_id) REFERENCES nodes(id) ON DELETE CASCADE
                    )
                ''')
                
                # Create performance indexes
                conn.execute('CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(type)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_edges_weight ON edges(weight DESC)')
                
                conn.commit()
                
        except sqlite3.Error as e:
            raise GraphStoreError(f"Failed to initialize database: {e}")
    
    @contextmanager
    def _transaction(self):
        """Context manager for SQLite transactions."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute('PRAGMA foreign_keys = ON')
            conn.execute('BEGIN')
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()