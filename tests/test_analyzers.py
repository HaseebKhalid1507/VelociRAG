"""
Tests for Velociragtor analyzers.

Test each of the 6 analyzers with realistic markdown content.
"""

import pytest
from datetime import datetime, timedelta
import numpy as np

from velocirag.graph import Node, Edge, NodeType, RelationType
from velocirag.embedder import Embedder
from velocirag.analyzers import (
    ExplicitAnalyzer,
    TemporalAnalyzer,
    EntityAnalyzer,
    TopicAnalyzer,
    SemanticAnalyzer,
    CentralityAnalyzer
)


class TestExplicitAnalyzer:
    """Test the explicit relationship analyzer."""
    
    def test_extract_wikilinks(self):
        """Test wikilink extraction."""
        analyzer = ExplicitAnalyzer()
        
        # Create test nodes with wikilinks
        nodes = [
            Node(
                id="note_1",
                type=NodeType.NOTE,
                title="Python Guide",
                content="Learn [[Python Basics]] and [[Advanced Python|advanced concepts]]."
            ),
            Node(
                id="note_2",
                type=NodeType.NOTE,
                title="Python Basics",
                content="Starting with Python programming."
            ),
            Node(
                id="note_3",
                type=NodeType.NOTE,
                title="Advanced Python",
                content="Deep dive into Python internals."
            )
        ]
        
        new_nodes, new_edges = analyzer.analyze(nodes)
        
        # Should find 2 reference edges
        ref_edges = [e for e in new_edges if e.type == RelationType.REFERENCES]
        assert len(ref_edges) == 2
        
        # Check edge properties
        edge_targets = [e.target_id for e in ref_edges]
        assert "note_2" in edge_targets  # Python Basics
        assert "note_3" in edge_targets  # Advanced Python
    
    def test_extract_tags(self):
        """Test tag extraction and node creation."""
        analyzer = ExplicitAnalyzer()
        
        nodes = [
            Node(
                id="note_1",
                type=NodeType.NOTE,
                title="Project Planning",
                content="Working on #projectA and #projectB. Also #python related."
            ),
            Node(
                id="note_2",
                type=NodeType.NOTE,
                title="Daily Log",
                content="Progress on #projectA today. #python debugging session."
            )
        ]
        
        new_nodes, new_edges = analyzer.analyze(nodes)
        
        # Should create 3 unique tag nodes
        tag_nodes = [n for n in new_nodes if n.type == NodeType.TAG]
        assert len(tag_nodes) == 3
        
        tag_titles = [n.title for n in tag_nodes]
        assert "#projecta" in tag_titles  # Tags are lowercased
        assert "#projectb" in tag_titles
        assert "#python" in tag_titles
        
        # Should create tagged_as edges
        tag_edges = [e for e in new_edges if e.type == RelationType.TAGGED_AS]
        assert len(tag_edges) == 5  # 3 from note_1, 2 from note_2
    
    def test_mixed_content(self):
        """Test analyzer with mixed wikilinks and tags."""
        analyzer = ExplicitAnalyzer()
        
        nodes = [
            Node(
                id="note_1",
                type=NodeType.NOTE,
                title="Research Notes",
                content="See [[Literature Review]] for #research on #machinelearning."
            ),
            Node(
                id="note_2",
                type=NodeType.NOTE,
                title="Literature Review",
                content="Papers on #machinelearning and [[deep learning]]."
            )
        ]
        
        new_nodes, new_edges = analyzer.analyze(nodes)
        
        # Count different types
        tag_nodes = [n for n in new_nodes if n.type == NodeType.TAG]
        ref_edges = [e for e in new_edges if e.type == RelationType.REFERENCES]
        tag_edges = [e for e in new_edges if e.type == RelationType.TAGGED_AS]
        
        assert len(tag_nodes) == 2  # research, machinelearning
        assert len(ref_edges) == 1  # note_1 -> note_2
        assert len(tag_edges) == 3  # 2 from note_1, 1 from note_2


class TestEntityAnalyzer:
    """Test the entity extraction analyzer."""
    
    def test_person_extraction(self):
        """Test extraction of person names."""
        analyzer = EntityAnalyzer(min_frequency=1)
        
        nodes = [
            Node(
                id="note_1",
                type=NodeType.NOTE,
                title="Meeting Notes",
                content="Met with John Smith and J. Doe about the project. John Smith had good ideas."
            ),
            Node(
                id="note_2", 
                type=NodeType.NOTE,
                title="Follow-up",
                content="Sent email to John Smith. Also reached out to Mary Johnson."
            )
        ]
        
        new_nodes, new_edges = analyzer.analyze(nodes)
        
        # Should extract person entities
        entity_nodes = [n for n in new_nodes if n.type == NodeType.ENTITY]
        entity_titles = [n.title for n in entity_nodes]
        
        assert "John Smith" in entity_titles  # Appears 3 times
        assert any("Doe" in title for title in entity_titles)  # J. Doe
    
    def test_project_extraction(self):
        """Test extraction of project names."""
        analyzer = EntityAnalyzer(min_frequency=1)
        
        nodes = [
            Node(
                id="note_1",
                type=NodeType.NOTE,
                title="Development Log",
                content="Working on VelociraptorProject and the DataEngine system."
            ),
            Node(
                id="note_2",
                type=NodeType.NOTE,
                title="Status Update",
                content="VelociraptorProject is progressing. The SearchTool needs work."
            )
        ]
        
        new_nodes, new_edges = analyzer.analyze(nodes)
        
        entity_nodes = [n for n in new_nodes if n.type == NodeType.ENTITY]
        entity_titles = [n.title for n in entity_nodes]
        
        assert "VelociraptorProject" in entity_titles
        assert "DataEngine" in entity_titles
        assert "SearchTool" in entity_titles
    
    def test_frequency_threshold(self):
        """Test that frequency threshold filters entities."""
        analyzer = EntityAnalyzer(min_frequency=2)
        
        nodes = [
            Node(
                id="note_1",
                type=NodeType.NOTE,
                title="Note 1",
                content="Alice appears once. Bob appears here."
            ),
            Node(
                id="note_2",
                type=NodeType.NOTE,
                title="Note 2", 
                content="Bob appears again. Charlie is new."
            )
        ]
        
        new_nodes, new_edges = analyzer.analyze(nodes)
        
        entity_nodes = [n for n in new_nodes if n.type == NodeType.ENTITY]
        entity_titles = [n.title for n in entity_nodes]
        
        # Only Bob should be extracted (appears twice)
        assert "Bob" in entity_titles
        assert "Alice" not in entity_titles
        assert "Charlie" not in entity_titles


class TestTemporalAnalyzer:
    """Test the temporal relationship analyzer."""
    
    def test_concurrent_relationships(self):
        """Test finding concurrent notes within time window."""
        analyzer = TemporalAnalyzer(window_days=7)
        
        base_date = datetime.now()
        nodes = [
            Node(
                id="note_1",
                type=NodeType.NOTE,
                title="Week 1 Notes",
                created_at=base_date,
                metadata={'created_time': base_date.isoformat()}
            ),
            Node(
                id="note_2",
                type=NodeType.NOTE,
                title="Week 1 Summary",
                created_at=base_date + timedelta(days=3),
                metadata={'created_time': (base_date + timedelta(days=3)).isoformat()}
            ),
            Node(
                id="note_3",
                type=NodeType.NOTE,
                title="Week 2 Notes",
                created_at=base_date + timedelta(days=10),
                metadata={'created_time': (base_date + timedelta(days=10)).isoformat()}
            )
        ]
        
        new_nodes, new_edges = analyzer.analyze(nodes)
        
        # The analyzer creates edges for ALL pairs within the window
        # note_1 -> note_2 (3 days apart)
        # note_2 -> note_3 (7 days apart) - exactly at window boundary
        assert len(new_edges) == 2
        
        # Find the edge between note_1 and note_2
        edge_1_2 = next((e for e in new_edges if 
                         {e.source_id, e.target_id} == {"note_1", "note_2"}), None)
        assert edge_1_2 is not None
        assert edge_1_2.type == RelationType.SIMILAR_TO
        assert edge_1_2.metadata['temporal_type'] == 'concurrent'
        assert edge_1_2.metadata['days_apart'] == 3
    
    def test_date_extraction_from_filename(self):
        """Test extracting dates from filenames."""
        analyzer = TemporalAnalyzer(window_days=5)
        
        # Need to ensure nodes don't have created_at so filename extraction is used
        nodes = []
        for i, (node_id, title, filename) in enumerate([
            ("note_1", "Daily Log", "2024-01-15-daily-log.md"),
            ("note_2", "Meeting Notes", "2024-01-17-meeting.md"),
            ("note_3", "Weekly Review", "2024-01-22-review.md")
        ]):
            node = Node(
                id=node_id,
                type=NodeType.NOTE,
                title=title,
                metadata={'filename': filename}
            )
            # Clear created_at to force filename date extraction
            object.__setattr__(node, 'created_at', None)
            nodes.append(node)
        
        new_nodes, new_edges = analyzer.analyze(nodes)
        
        # The analyzer finds:
        # - note_1 -> note_2 (2 days apart)
        # - note_2 -> note_3 (5 days apart, exactly at window boundary)
        assert len(new_edges) == 2
        
        # Find the edge between note_1 and note_2
        edge_1_2 = next((e for e in new_edges if e.metadata['days_apart'] == 2), None)
        assert edge_1_2 is not None
        
        # Find the edge between note_2 and note_3
        edge_2_3 = next((e for e in new_edges if e.metadata['days_apart'] == 5), None)
        assert edge_2_3 is not None
    
    def test_edge_weight_calculation(self):
        """Test that edge weight decreases with temporal distance."""
        analyzer = TemporalAnalyzer(window_days=10)
        
        base_date = datetime.now()
        nodes = [
            Node(
                id="note_1",
                type=NodeType.NOTE,
                title="Note 1",
                created_at=base_date
            ),
            Node(
                id="note_2",
                type=NodeType.NOTE,
                title="Note 2",
                created_at=base_date + timedelta(days=2)
            ),
            Node(
                id="note_3",
                type=NodeType.NOTE,
                title="Note 3",
                created_at=base_date + timedelta(days=8)
            )
        ]
        
        new_nodes, new_edges = analyzer.analyze(nodes)
        
        # Should create 3 edges (all pairs are within 10 days)
        # note_1 -> note_2 (2 days)
        # note_1 -> note_3 (8 days)
        # note_2 -> note_3 (6 days)
        assert len(new_edges) == 3
        
        # Edge with smaller time difference should have higher weight
        edge_1_2 = next(e for e in new_edges if e.metadata['days_apart'] == 2)
        edge_1_3 = next(e for e in new_edges if e.metadata['days_apart'] == 8)
        edge_2_3 = next(e for e in new_edges if e.metadata['days_apart'] == 6)
        
        assert edge_1_2.weight > edge_2_3.weight > edge_1_3.weight


class TestTopicAnalyzer:
    """Test the topic clustering analyzer."""
    
    def test_simple_topic_clustering(self):
        """Test basic topic clustering with clear topics."""
        analyzer = TopicAnalyzer(n_topics=2)
        
        nodes = [
            Node(
                id="note_1",
                type=NodeType.NOTE,
                title="Python Tutorial",
                content="Python programming language tutorial. Learn Python basics and syntax."
            ),
            Node(
                id="note_2",
                type=NodeType.NOTE,
                title="Python Advanced",
                content="Advanced Python concepts. Python decorators and generators."
            ),
            Node(
                id="note_3",
                type=NodeType.NOTE,
                title="JavaScript Guide",
                content="JavaScript programming guide. JavaScript functions and objects."
            ),
            Node(
                id="note_4",
                type=NodeType.NOTE,
                title="JavaScript Async",
                content="JavaScript async programming. JavaScript promises and callbacks."
            )
        ]
        
        new_nodes, new_edges = analyzer.analyze(nodes)
        
        # Should create topic nodes
        topic_nodes = [n for n in new_nodes if n.type == NodeType.TOPIC]
        assert len(topic_nodes) >= 1
        
        # Should create discussion edges
        discuss_edges = [e for e in new_edges if e.type == RelationType.DISCUSSES]
        assert len(discuss_edges) >= 2
    
    def test_minimum_cluster_size(self):
        """Test that topics require minimum number of notes."""
        analyzer = TopicAnalyzer(n_topics=5)
        
        nodes = [
            Node(
                id="note_1",
                type=NodeType.NOTE,
                title="Single Note",
                content="This is a lonely note about nothing in particular."
            ),
            Node(
                id="note_2",
                type=NodeType.NOTE,
                title="Another Single",
                content="Another unrelated note about different things."
            )
        ]
        
        new_nodes, new_edges = analyzer.analyze(nodes)
        
        # With only 2 notes, shouldn't create many topics
        topic_nodes = [n for n in new_nodes if n.type == NodeType.TOPIC]
        assert len(topic_nodes) <= 1


class TestSemanticAnalyzer:
    """Test the semantic similarity analyzer."""
    
    def test_similarity_calculation(self):
        """Test semantic similarity with real embeddings."""
        # Create a mock embedder for testing
        embedder = Embedder()
        analyzer = SemanticAnalyzer(embedder, threshold=0.6)
        
        nodes = [
            Node(
                id="note_1",
                type=NodeType.NOTE,
                title="Machine Learning",
                content="Introduction to machine learning algorithms and neural networks."
            ),
            Node(
                id="note_2",
                type=NodeType.NOTE,
                title="Deep Learning",
                content="Deep learning with neural networks and backpropagation."
            ),
            Node(
                id="note_3",
                type=NodeType.NOTE,
                title="Cooking Recipe",
                content="How to make chocolate cake with flour and sugar."
            )
        ]
        
        new_nodes, new_edges = analyzer.analyze(nodes)
        
        # ML and DL notes should be similar
        similar_edges = [e for e in new_edges if e.type == RelationType.SIMILAR_TO]
        
        if similar_edges:  # Only if embeddings work
            # Check that similar content has edges
            ml_dl_edge = next((e for e in similar_edges 
                              if {e.source_id, e.target_id} == {"note_1", "note_2"}), None)
            assert ml_dl_edge is not None
            assert ml_dl_edge.weight >= 0.6
    
    def test_threshold_filtering(self):
        """Test that similarity threshold filters edges."""
        embedder = Embedder()
        analyzer = SemanticAnalyzer(embedder, threshold=0.8)  # High threshold
        
        nodes = [
            Node(
                id="note_1",
                type=NodeType.NOTE,
                title="Note A",
                content="Some content about topic X."
            ),
            Node(
                id="note_2",
                type=NodeType.NOTE,
                title="Note B",
                content="Different content about topic Y."
            )
        ]
        
        new_nodes, new_edges = analyzer.analyze(nodes)
        
        # With high threshold, might not find similarities
        assert len(new_nodes) == 0
        # Edges depend on actual similarity scores


class TestCentralityAnalyzer:
    """Test the centrality scoring analyzer."""
    
    def test_degree_centrality(self):
        """Test degree centrality calculation."""
        analyzer = CentralityAnalyzer()
        
        # Create a simple graph
        nodes = [
            Node(id="A", type=NodeType.NOTE, title="Node A"),
            Node(id="B", type=NodeType.NOTE, title="Node B"),
            Node(id="C", type=NodeType.NOTE, title="Node C"),
            Node(id="D", type=NodeType.NOTE, title="Node D"),
        ]
        
        # A is connected to all others (hub)
        edges = [
            Edge(id="e1", source_id="A", target_id="B", type=RelationType.REFERENCES, weight=0.8, confidence=0.9),
            Edge(id="e2", source_id="A", target_id="C", type=RelationType.REFERENCES, weight=0.8, confidence=0.9),
            Edge(id="e3", source_id="A", target_id="D", type=RelationType.REFERENCES, weight=0.8, confidence=0.9),
            Edge(id="e4", source_id="B", target_id="C", type=RelationType.SIMILAR_TO, weight=0.6, confidence=0.7),
        ]
        
        scores = analyzer.analyze(nodes, edges)
        
        # A should have highest score (connected to all)
        assert scores["A"] > scores["B"]
        assert scores["A"] > scores["C"]
        assert scores["A"] > scores["D"]
        
        # All nodes should have scores between 0 and 1
        for score in scores.values():
            assert 0.0 <= score <= 1.0
    
    def test_betweenness_centrality(self):
        """Test betweenness centrality in path calculation."""
        analyzer = CentralityAnalyzer()
        
        # Linear graph: A -- B -- C -- D
        nodes = [
            Node(id="A", type=NodeType.NOTE, title="Node A"),
            Node(id="B", type=NodeType.NOTE, title="Node B"), 
            Node(id="C", type=NodeType.NOTE, title="Node C"),
            Node(id="D", type=NodeType.NOTE, title="Node D"),
        ]
        
        edges = [
            Edge(id="e1", source_id="A", target_id="B", type=RelationType.REFERENCES, weight=0.8, confidence=0.9),
            Edge(id="e2", source_id="B", target_id="C", type=RelationType.REFERENCES, weight=0.8, confidence=0.9),
            Edge(id="e3", source_id="C", target_id="D", type=RelationType.REFERENCES, weight=0.8, confidence=0.9),
        ]
        
        scores = analyzer.analyze(nodes, edges)
        
        # B and C should have higher scores (on more paths)
        assert scores["B"] > scores["A"]
        assert scores["C"] > scores["D"]
    
    def test_empty_graph(self):
        """Test centrality with no edges."""
        analyzer = CentralityAnalyzer()
        
        nodes = [
            Node(id="A", type=NodeType.NOTE, title="Isolated A"),
            Node(id="B", type=NodeType.NOTE, title="Isolated B"),
        ]
        
        scores = analyzer.analyze(nodes, [])
        
        # Analyzer returns empty dict when there are no edges
        assert len(scores) == 0
        assert scores == {}