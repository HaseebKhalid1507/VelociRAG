"""
Tests for Velociragtor graph build pipeline.

Test the full pipeline with temporary markdown files and verify graph construction.
"""

import pytest
import json
from pathlib import Path
from datetime import datetime

from velocirag.graph import GraphStore, NodeType, RelationType
from velocirag.embedder import Embedder
from velocirag.pipeline import GraphPipeline


class TestGraphPipeline:
    """Test the graph build pipeline."""
    
    @pytest.fixture
    def sample_markdown_files(self, tmp_path):
        """Create sample markdown files for testing."""
        # Create directory structure
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        
        projects_dir = docs_dir / "projects"
        projects_dir.mkdir()
        
        # Create markdown files with various content
        files = {
            "python-guide.md": """# Python Programming Guide

Learn [[Python Basics]] and explore #python #programming concepts.

This guide covers Python fundamentals and advanced topics.

Last updated: 2024-01-15
""",
            "python-basics.md": """# Python Basics

Introduction to Python programming language.

Topics covered:
- Variables and data types
- Control structures  
- Functions and modules

See also [[Python Guide]] for more information.

#python #tutorial #beginner
""",
            "machine-learning.md": """# Machine Learning with Python

Implementing ML algorithms using Python and scikit-learn.

Related: [[Deep Learning]] and [[Data Science]]

Key concepts:
- Supervised learning
- Neural networks
- Model evaluation

#machinelearning #python #datascience
""",
            "deep-learning.md": """# Deep Learning

Advanced neural network architectures and training techniques.

Building on [[Machine Learning]] concepts with TensorFlow and PyTorch.

#deeplearning #neuralnetworks #ai
""",
            "projects/velociraptor.md": """# Velociraptor Project

Building a knowledge graph system with vector search.

Components:
- [[Vector Search]]
- [[Knowledge Graph]]
- Graph algorithms

Lead: John Smith
Status: Active

#project #search #graph
""",
            "daily-log-2024-01-20.md": """# Daily Log - 2024-01-20

Worked on VelociraptorProject today. Met with John Smith about architecture.

Tasks completed:
- Implemented [[graph storage]]
- Fixed search bugs
- Updated documentation

#daily #progress
""",
            "daily-log-2024-01-22.md": """# Daily Log - 2024-01-22  

Continued work on VelociraptorProject. Good progress on search features.

Alice Johnson joined the team. She'll work on the DataEngine component.

#daily #progress
""",
            "meeting-notes.md": """# Team Meeting Notes

Participants: John Smith, Alice Johnson, Bob Williams

Discussed:
- Project timeline
- Technical challenges
- Resource allocation

Action items assigned to team members.

#meeting #planning
"""
        }
        
        # Write files
        for filename, content in files.items():
            if "/" in filename:
                # Handle subdirectory
                parts = filename.split("/")
                file_path = docs_dir / Path(*parts)
            else:
                file_path = docs_dir / filename
            
            file_path.write_text(content)
        
        return docs_dir
    
    @pytest.fixture
    def graph_store(self, tmp_path):
        """Create a temporary graph store."""
        db_path = tmp_path / "test_graph.db"
        return GraphStore(str(db_path))
    
    def test_basic_pipeline_build(self, sample_markdown_files, graph_store):
        """Test basic pipeline execution."""
        pipeline = GraphPipeline(graph_store)
        
        # Run pipeline
        stats = pipeline.build(str(sample_markdown_files))
        
        # Verify completion
        assert stats['success'] is True
        assert stats['final_nodes'] > 0
        assert stats['final_edges'] > 0
        
        # Check stage completion
        assert 'scan' in stats['stages']
        assert 'explicit' in stats['stages']
        assert 'entity' in stats['stages']
        assert 'temporal' in stats['stages']
        assert 'topic' in stats['stages']
        assert 'centrality' in stats['stages']
        assert 'storage' in stats['stages']
        
        # Verify files were scanned
        assert stats['stages']['scan']['files_found'] == 8
        assert stats['stages']['scan']['notes_created'] == 8
    
    def test_explicit_analysis_stage(self, sample_markdown_files, graph_store):
        """Test that explicit analysis finds wikilinks and tags."""
        pipeline = GraphPipeline(graph_store)
        stats = pipeline.build(str(sample_markdown_files))
        
        # Check explicit analysis found relationships
        explicit_stats = stats['stages']['explicit']
        assert explicit_stats['nodes_added'] > 0  # Tag nodes
        assert explicit_stats['edges_added'] > 0  # Reference and tag edges
        
        # Verify in storage
        stored_stats = graph_store.stats()
        
        # Should have tag nodes
        tag_nodes = graph_store.get_nodes_by_type(NodeType.TAG)
        tag_titles = [n.title for n in tag_nodes]
        assert "#python" in tag_titles
        assert "#project" in tag_titles
        assert "#machinelearning" in tag_titles
    
    def test_entity_extraction_stage(self, sample_markdown_files, graph_store):
        """Test that entity extraction finds people and projects."""
        pipeline = GraphPipeline(graph_store)
        stats = pipeline.build(str(sample_markdown_files))
        
        # Check entity analysis results
        entity_stats = stats['stages']['entity']
        assert entity_stats['nodes_added'] > 0
        assert entity_stats['edges_added'] > 0
        
        # Verify entities in storage
        entity_nodes = graph_store.get_nodes_by_type(NodeType.ENTITY)
        entity_titles = [n.title for n in entity_nodes]
        
        # Should find people mentioned multiple times
        assert "John Smith" in entity_titles
        assert "VelociraptorProject" in entity_titles
    
    def test_temporal_analysis_stage(self, sample_markdown_files, graph_store):
        """Test temporal relationship detection."""
        pipeline = GraphPipeline(graph_store)
        stats = pipeline.build(str(sample_markdown_files))
        
        # Check temporal analysis
        temporal_stats = stats['stages']['temporal']
        assert temporal_stats['edges_added'] >= 1  # Daily logs are close in time
    
    def test_topic_clustering_stage(self, sample_markdown_files, graph_store):
        """Test topic analysis and clustering."""
        pipeline = GraphPipeline(graph_store)
        stats = pipeline.build(str(sample_markdown_files))
        
        # Check topic analysis
        topic_stats = stats['stages']['topic']
        assert topic_stats['nodes_added'] > 0  # Topic nodes created
        assert topic_stats['edges_added'] > 0  # Discussion edges
        
        # Verify topic nodes
        topic_nodes = graph_store.get_nodes_by_type(NodeType.TOPIC)
        assert len(topic_nodes) > 0
    
    def test_centrality_calculation(self, sample_markdown_files, graph_store):
        """Test that centrality scores are calculated."""
        pipeline = GraphPipeline(graph_store)
        stats = pipeline.build(str(sample_markdown_files))
        
        # Check centrality stage
        centrality_stats = stats['stages']['centrality']
        assert centrality_stats['nodes_scored'] > 0
        assert 'top_nodes' in centrality_stats
        
        # Verify importance scores in metadata
        all_nodes = graph_store.get_nodes_by_type(NodeType.NOTE)
        scored_nodes = [n for n in all_nodes if 'importance_score' in n.metadata]
        assert len(scored_nodes) > 0
    
    def test_force_rebuild(self, sample_markdown_files, graph_store):
        """Test force rebuild clears existing data."""
        pipeline = GraphPipeline(graph_store)
        
        # First build
        stats1 = pipeline.build(str(sample_markdown_files))
        nodes_before = stats1['final_nodes']
        
        # Add an extra node manually
        from velocirag.graph import Node
        extra_node = Node(
            id="manual_node",
            type=NodeType.NOTE,
            title="Manual Node"
        )
        graph_store.add_node(extra_node)
        
        # Force rebuild
        stats2 = pipeline.build(str(sample_markdown_files), force_rebuild=True)
        
        # Should have same count as first build (manual node removed)
        assert stats2['final_nodes'] == nodes_before
        
        # Verify manual node is gone
        assert graph_store.get_node("manual_node") is None
    
    def test_semantic_analysis_with_embedder(self, sample_markdown_files, graph_store):
        """Test semantic analysis when embedder is available."""
        embedder = Embedder()
        pipeline = GraphPipeline(graph_store, embedder=embedder)
        
        stats = pipeline.build(str(sample_markdown_files))
        
        # Check if semantic analysis ran
        if 'semantic' in stats['stages']:
            semantic_stats = stats['stages']['semantic']
            # May or may not find similarities depending on embeddings
            assert 'edges_added' in semantic_stats
    
    def test_pipeline_error_handling(self, tmp_path, graph_store):
        """Test pipeline handles missing directory gracefully."""
        pipeline = GraphPipeline(graph_store)
        
        # Try to build from non-existent directory
        stats = pipeline.build(str(tmp_path / "non_existent"))
        
        # Should fail and return stats with error
        assert 'error' in stats
        assert 'Source path does not exist' in stats['error']
    
    def test_graph_optimization(self, sample_markdown_files, graph_store):
        """Test graph processing and optimization stage."""
        pipeline = GraphPipeline(graph_store)
        stats = pipeline.build(str(sample_markdown_files))
        
        # Check processing stage  
        processing_stats = stats['stages']['processing']
        assert 'nodes_before' in processing_stats
        assert 'nodes_after' in processing_stats
        assert 'edges_before' in processing_stats
        assert 'edges_after' in processing_stats
        
        # Some optimization should occur
        assert processing_stats['nodes_after'] <= processing_stats['nodes_before']
        assert processing_stats['edges_after'] <= processing_stats['edges_before']
    
    def test_final_storage_verification(self, sample_markdown_files, graph_store):
        """Test that final graph is properly stored."""
        pipeline = GraphPipeline(graph_store)
        stats = pipeline.build(str(sample_markdown_files))
        
        # Verify storage stats
        storage_stats = stats['stages']['storage']
        assert storage_stats['nodes_stored'] > 0
        assert storage_stats['edges_stored'] > 0
        assert storage_stats['db_size_mb'] > 0
        
        # Verify we can query the stored graph
        final_stats = graph_store.stats()
        assert final_stats['node_count'] == storage_stats['nodes_stored']
        assert final_stats['edge_count'] == storage_stats['edges_stored']
        
        # Check different node types exist
        assert len(graph_store.get_nodes_by_type(NodeType.NOTE)) > 0
        assert len(graph_store.get_nodes_by_type(NodeType.TAG)) > 0
        assert len(graph_store.get_nodes_by_type(NodeType.ENTITY)) > 0