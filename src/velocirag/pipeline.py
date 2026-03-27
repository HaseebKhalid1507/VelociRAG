"""
Graph build pipeline for Velocirag.

10-stage pipeline: scan → metadata → explicit → entity → temporal → topic → semantic → processing → centrality → store
Clean orchestration with proper error handling and progress reporting.
"""

import gc
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import hashlib

from .graph import GraphStore, Node, Edge, NodeType, RelationType
from .embedder import Embedder, MIN_CACHE_SIZE
from .analyzers import (
    ExplicitAnalyzer,
    TemporalAnalyzer, 
    EntityAnalyzer,
    TopicAnalyzer,
    SemanticAnalyzer,
    CentralityAnalyzer
)
from .metadata import MetadataStore
from .frontmatter import parse_frontmatter, extract_tags_from_content, extract_wiki_links

logger = logging.getLogger("velocirag.pipeline")


class GraphPipeline:
    """
    Graph build pipeline orchestrator.
    
    Scans markdown, runs analyzers, merges results, stores final graph.
    Built for reliability and transparency.
    """
    
    def __init__(self, graph_store: GraphStore, embedder: Optional[Embedder] = None, 
                 metadata_store: Optional[MetadataStore] = None,
                 entity_extractor: str = 'regex'):
        """
        Initialize pipeline.
        
        Args:
            graph_store: Storage backend for the graph
            embedder: Optional embedder for semantic analysis
            metadata_store: Optional metadata store for structured queries
        """
        self.graph_store = graph_store
        self.embedder = embedder
        self.metadata_store = metadata_store
        
        # Initialize analyzers
        self.explicit_analyzer = ExplicitAnalyzer()
        self.temporal_analyzer = TemporalAnalyzer(window_days=7)
        
        # Entity extraction — regex (default) or gliner (optional)
        if entity_extractor == 'gliner':
            try:
                from .analyzers import GLiNERAnalyzer
                self.entity_analyzer = GLiNERAnalyzer(min_frequency=2)
                logger.info("Using GLiNER entity extraction")
            except ImportError:
                logger.warning("GLiNER not available, falling back to regex EntityAnalyzer")
                self.entity_analyzer = EntityAnalyzer(min_frequency=2)
        else:
            self.entity_analyzer = EntityAnalyzer(min_frequency=2)
        
        # Relation extraction — uses GLiNER multitask (optional, only with gliner entity extractor)
        self.relation_analyzer = None
        if entity_extractor == 'gliner':
            try:
                from .analyzers import RelationAnalyzer
                self.relation_analyzer = RelationAnalyzer()
            except ImportError:
                pass
        
        self.topic_analyzer = TopicAnalyzer(n_topics=10)
        # Minimize embedder cache for graph builds — 7k cached embeddings as Python lists
        # consume ~2GB of RAM.  The graph pipeline only embeds each doc once anyway.
        if embedder is not None:
            embedder._cache.clear()
            embedder.cache_size = MIN_CACHE_SIZE
        self.semantic_analyzer = SemanticAnalyzer(embedder, threshold=0.7) if embedder else None
        self.centrality_analyzer = CentralityAnalyzer()
        
        # Pipeline state
        self.nodes: List[Node] = []
        self.edges: List[Edge] = []
        self.stats = {}
        self.file_to_doc_id: Dict[str, int] = {}  # Track filename to doc_id mapping
    
    def build(self, source_path: str, force_rebuild: bool = False, skip_semantic: bool = False) -> Dict[str, Any]:
        """
        Execute the complete graph build pipeline.
        
        Args:
            source_path: Path to markdown directory
            force_rebuild: Clear existing graph before building
            skip_semantic: Skip Stage 7 (semantic similarity). Saves ~2GB RAM.
                          Explicit links, entities, temporal, topics still built.
            
        Returns:
            Build statistics and results
        """
        start_time = datetime.now()
        
        logger.info(f"Starting graph build from: {source_path}")
        if force_rebuild:
            logger.info("Force rebuild: clearing existing graph")
            self.graph_store.clear()
        
        # Reset state
        self.nodes = []
        self.edges = []
        self.stats = {
            'start_time': start_time.isoformat(),
            'source_path': source_path,
            'force_rebuild': force_rebuild,
            'stages': {}
        }
        
        try:
            # Stage 1: Scan and create note nodes
            self._stage_1_scan_files(source_path)
            
            # Stage 2: Metadata extraction (if metadata store available)
            if self.metadata_store:
                self._stage_2_metadata_extraction()
            
            # Stage 3: Explicit relationships
            self._stage_3_explicit_analysis()
            
            # Stage 4: Entity extraction
            self._stage_4_entity_analysis()
            
            # Free entity model before loading relation model (memory safety on 8GB)
            if hasattr(self.entity_analyzer, '_model') and self.entity_analyzer._model is not None:
                self.entity_analyzer._model = None
            gc.collect()

            # Stage 4.5: Relation extraction (if available)
            if self.relation_analyzer:
                self._stage_4_5_relation_analysis()
                # Free relation model too
                if hasattr(self.relation_analyzer, '_model') and self.relation_analyzer._model is not None:
                    self.relation_analyzer._model = None
            gc.collect()
            
            # Stage 5: Temporal analysis
            self._stage_5_temporal_analysis()
            
            # Stage 6: Topic analysis
            self._stage_6_topic_analysis()
            
            # Stage 7: Semantic analysis (if embedder available and not skipped)
            if self.semantic_analyzer and not skip_semantic:
                self._stage_7_semantic_analysis()
            elif skip_semantic:
                logger.info("Stage 7: Skipped (--light-graph / skip_semantic=True)")

            # Free memory after semantic analysis — content and embedder no longer needed
            content_freed = 0
            for node in self.nodes:
                if node.content:
                    node.content = None
                    content_freed += 1
            if self.semantic_analyzer and self.semantic_analyzer.embedder:
                self.semantic_analyzer.embedder._model = None
                self.semantic_analyzer.embedder._cache.clear()
            self.semantic_analyzer = None
            if self.embedder:
                self.embedder._model = None
                self.embedder._cache.clear()
                self.embedder = None
            self.topic_analyzer = None
            gc.collect()
            logger.info(f"Freed content from {content_freed} nodes + embedder model")
            
            # Stage 8: Graph processing and optimization
            self._stage_8_graph_processing()
            
            # Stage 9: Centrality calculation
            self._stage_9_centrality_analysis()
            
            # Stage 10: Storage
            self._stage_10_storage()
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            self.stats['error'] = str(e)
            return self.stats
        
        # Final statistics
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        self.stats.update({
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'final_nodes': len(self.nodes),
            'final_edges': len(self.edges),
            'success': True
        })
        
        logger.info(f"Graph build completed in {duration:.1f}s: {len(self.nodes)} nodes, {len(self.edges)} edges")
        return self.stats
    
    def _stage_1_scan_files(self, source_path: str) -> None:
        """Stage 1: Scan markdown files and create note nodes."""
        logger.info("Stage 1: Scanning markdown files...")
        start_time = datetime.now()
        
        source_path = Path(source_path).resolve()
        if not source_path.exists():
            raise ValueError(f"Source path does not exist: {source_path}")

        # Find all markdown files, skipping symlinks that escape the source directory
        md_files = [
            f for f in source_path.rglob("*.md")
            if not f.is_symlink() or f.resolve().is_relative_to(source_path)
        ]
        logger.info(f"Found {len(md_files)} markdown files")
        
        # Create note nodes
        notes_created = 0
        for md_file in md_files:
            try:
                node = self._create_note_node(md_file, source_path)
                if node:
                    self.nodes.append(node)
                    notes_created += 1
            except Exception as e:
                logger.warning(f"Failed to process {md_file}: {e}")
        
        duration = (datetime.now() - start_time).total_seconds()
        self.stats['stages']['scan'] = {
            'duration_seconds': duration,
            'files_found': len(md_files),
            'notes_created': notes_created
        }
        
        logger.info(f"Stage 1 complete: {notes_created} notes created in {duration:.1f}s")
    
    def _stage_2_metadata_extraction(self) -> None:
        """Stage 2: Extract metadata from markdown files.
        
        Uses a single transaction for all upserts instead of per-document
        transactions. 680 files: 4+ min → ~2s.
        """
        logger.info("Stage 2: Extracting metadata from markdown files...")
        start_time = datetime.now()
        
        metadata_added = 0
        tags_extracted = 0
        cross_refs_extracted = 0
        errors = []
        
        import sqlite3
        conn = sqlite3.connect(self.metadata_store.db_path)
        conn.execute('PRAGMA foreign_keys = ON')
        conn.execute('BEGIN')
        
        # Process each note node in a single transaction
        for node in self.nodes:
            if node.type != NodeType.NOTE:
                continue
                
            try:
                file_path = node.metadata.get('file_path')
                if not file_path:
                    continue
                
                # Parse frontmatter and extract metadata
                frontmatter, body = parse_frontmatter(node.content or '')
                
                # Extract tags from both frontmatter and content
                frontmatter_tags = []
                if 'tags' in frontmatter:
                    # Handle tags in frontmatter (can be list or string)
                    if isinstance(frontmatter['tags'], list):
                        frontmatter_tags = [str(tag).strip().lower() for tag in frontmatter['tags']]
                    elif isinstance(frontmatter['tags'], str):
                        frontmatter_tags = [tag.strip().lower() for tag in frontmatter['tags'].split(',')]
                
                content_tags = extract_tags_from_content(body)
                all_tags = list(set(frontmatter_tags + content_tags))
                
                # Extract wiki links as cross-references
                wiki_links = extract_wiki_links(body)
                
                # Build metadata dict
                metadata = {
                    'category': frontmatter.get('category', frontmatter.get('type')),
                    'status': frontmatter.get('status', 'active'),
                    'project': frontmatter.get('project'),
                    'created_date': frontmatter.get('created_date', frontmatter.get('date')),
                    'updated_date': frontmatter.get('updated_date', frontmatter.get('modified')),
                }
                
                # Add any custom frontmatter fields
                for key, value in frontmatter.items():
                    if key not in metadata and key not in ['tags']:
                        metadata[key] = value
                
                # Upsert document directly via shared connection (not per-doc transaction)
                filename = node.metadata.get('relative_path', node.metadata.get('filename', node.id))
                category = metadata.get('category')
                status = metadata.get('status', 'active')
                project = metadata.get('project')
                created_date = metadata.get('created_date')
                updated_date = metadata.get('updated_date')
                meta_fields = {k: v for k, v in metadata.items()
                              if k not in ('category', 'status', 'project', 'created_date', 'updated_date')}
                meta_json = json.dumps(meta_fields) if meta_fields else None
                
                existing = conn.execute(
                    'SELECT id FROM documents WHERE filename = ?', (filename,)
                ).fetchone()
                
                if existing:
                    conn.execute('''
                        UPDATE documents SET title=?, category=?, status=?, project=?,
                            created_date=?, updated_date=?, meta=?, updated_at=CURRENT_TIMESTAMP
                        WHERE filename=?
                    ''', (node.title, category, status, project, created_date, updated_date, meta_json, filename))
                    doc_id = existing[0]
                else:
                    cursor = conn.execute('''
                        INSERT INTO documents (filename, title, category, status, project,
                            created_date, updated_date, meta) VALUES (?,?,?,?,?,?,?,?)
                    ''', (filename, node.title, category, status, project, created_date, updated_date, meta_json))
                    doc_id = cursor.lastrowid
                
                self.file_to_doc_id[file_path] = doc_id
                metadata_added += 1
                
                # Add tags via shared connection
                if all_tags:
                    for tag in all_tags:
                        tag_row = conn.execute('SELECT id FROM tags WHERE name = ?', (tag,)).fetchone()
                        if tag_row:
                            tag_id = tag_row[0]
                        else:
                            tag_id = conn.execute('INSERT INTO tags (name) VALUES (?)', (tag,)).lastrowid
                        conn.execute('INSERT OR IGNORE INTO document_tags (document_id, tag_id) VALUES (?,?)', (doc_id, tag_id))
                    tags_extracted += len(all_tags)
                
                # Add cross-references via shared connection
                for wiki_link in wiki_links:
                    conn.execute('''
                        INSERT OR IGNORE INTO cross_refs (source_doc_id, target_ref, ref_type)
                        VALUES (?,?,?)
                    ''', (doc_id, wiki_link, 'wiki_link'))
                    cross_refs_extracted += 1
                    
            except Exception as e:
                logger.warning(f"Failed to extract metadata from {node.id}: {e}")
                errors.append(str(e))
        
        # Commit the single transaction for all documents
        conn.commit()
        conn.close()
        
        duration = (datetime.now() - start_time).total_seconds()
        self.stats['stages']['metadata'] = {
            'duration_seconds': duration,
            'documents_processed': metadata_added,
            'tags_extracted': tags_extracted,
            'cross_refs_extracted': cross_refs_extracted,
            'errors': len(errors)
        }
        
        logger.info(f"Stage 2 complete: {metadata_added} documents, {tags_extracted} tags, {cross_refs_extracted} cross-refs in {duration:.1f}s")
    
    def _create_note_node(self, file_path: Path, base_path: Path) -> Optional[Node]:
        """Create a note node from a markdown file."""
        try:
            # Read file content
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            
            # Extract title from filename
            title = file_path.stem.replace('-', ' ').replace('_', ' ').title()
            
            # Create unique ID
            relative_path = file_path.relative_to(base_path)
            node_id = f"note_{hashlib.md5(str(relative_path).encode()).hexdigest()[:12]}"
            
            # Get file stats
            stat = file_path.stat()
            
            # Create node
            node = Node(
                id=node_id,
                type=NodeType.NOTE,
                title=title,
                content=content,
                metadata={
                    'file_path': str(file_path),
                    'relative_path': str(relative_path),
                    'filename': file_path.name,
                    'word_count': len(content.split()),
                    'char_count': len(content),
                    'created_time': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat()
                }
            )
            
            return node
            
        except Exception as e:
            logger.warning(f"Failed to create node for {file_path}: {e}")
            return None
    
    def _stage_3_explicit_analysis(self) -> None:
        """Stage 3: Extract explicit relationships (wikilinks, tags)."""
        logger.info("Stage 3: Analyzing explicit relationships...")
        start_time = datetime.now()
        
        new_nodes, new_edges = self.explicit_analyzer.analyze(self.nodes)
        
        self.nodes.extend(new_nodes)
        self.edges.extend(new_edges)
        
        duration = (datetime.now() - start_time).total_seconds()
        self.stats['stages']['explicit'] = {
            'duration_seconds': duration,
            'nodes_added': len(new_nodes),
            'edges_added': len(new_edges)
        }
        
        logger.info(f"Stage 3 complete: +{len(new_nodes)} nodes, +{len(new_edges)} edges in {duration:.1f}s")
    
    def _stage_4_entity_analysis(self) -> None:
        """Stage 4: Extract entities and entity relationships."""
        logger.info("Stage 4: Analyzing entities...")
        start_time = datetime.now()
        
        new_nodes, new_edges = self.entity_analyzer.analyze(self.nodes)
        
        self.nodes.extend(new_nodes)
        self.edges.extend(new_edges)
        
        duration = (datetime.now() - start_time).total_seconds()
        self.stats['stages']['entity'] = {
            'duration_seconds': duration,
            'nodes_added': len(new_nodes),
            'edges_added': len(new_edges)
        }
        
        logger.info(f"Stage 4 complete: +{len(new_nodes)} entities, +{len(new_edges)} edges in {duration:.1f}s")
    
    def _stage_4_5_relation_analysis(self) -> None:
        """Stage 4.5: Extract semantic relations between entities."""
        logger.info("Stage 4.5: Extracting semantic relations...")
        start_time = datetime.now()
        
        # Get mention edges from stage 4 (connects notes to entities)
        mention_edges = [e for e in self.edges if e.type == RelationType.MENTIONS]
        
        new_edges = self.relation_analyzer.analyze(self.nodes, mention_edges)
        self.edges.extend(new_edges)
        
        duration = (datetime.now() - start_time).total_seconds()
        self.stats['stages']['relations'] = {
            'relation_edges': len(new_edges),
            'duration_seconds': duration
        }
        logger.info(f"Stage 4.5 complete: +{len(new_edges)} relation edges in {duration:.1f}s")
    
    def _stage_5_temporal_analysis(self) -> None:
        """Stage 5: Analyze temporal relationships."""
        logger.info("Stage 5: Analyzing temporal patterns...")
        start_time = datetime.now()
        
        new_nodes, new_edges = self.temporal_analyzer.analyze(self.nodes)
        
        self.nodes.extend(new_nodes)
        self.edges.extend(new_edges)
        
        duration = (datetime.now() - start_time).total_seconds()
        self.stats['stages']['temporal'] = {
            'duration_seconds': duration,
            'nodes_added': len(new_nodes),
            'edges_added': len(new_edges)
        }
        
        logger.info(f"Stage 5 complete: +{len(new_edges)} temporal edges in {duration:.1f}s")
    
    def _stage_6_topic_analysis(self) -> None:
        """Stage 6: Analyze topics and themes."""
        logger.info("Stage 6: Analyzing topics...")
        start_time = datetime.now()
        
        new_nodes, new_edges = self.topic_analyzer.analyze(self.nodes)
        
        self.nodes.extend(new_nodes)
        self.edges.extend(new_edges)
        
        duration = (datetime.now() - start_time).total_seconds()
        self.stats['stages']['topic'] = {
            'duration_seconds': duration,
            'nodes_added': len(new_nodes),
            'edges_added': len(new_edges)
        }
        
        logger.info(f"Stage 6 complete: +{len(new_nodes)} topics, +{len(new_edges)} edges in {duration:.1f}s")
    
    def _stage_7_semantic_analysis(self) -> None:
        """Stage 7: Semantic similarity analysis."""
        logger.info("Stage 7: Analyzing semantic relationships...")
        start_time = datetime.now()
        
        new_nodes, new_edges = self.semantic_analyzer.analyze(self.nodes)
        
        self.nodes.extend(new_nodes)
        self.edges.extend(new_edges)
        
        duration = (datetime.now() - start_time).total_seconds()
        self.stats['stages']['semantic'] = {
            'duration_seconds': duration,
            'nodes_added': len(new_nodes),
            'edges_added': len(new_edges)
        }
        
        logger.info(f"Stage 7 complete: +{len(new_edges)} semantic edges in {duration:.1f}s")
    
    def _stage_8_graph_processing(self) -> None:
        """Stage 8: Process and optimize the graph."""
        logger.info("Stage 8: Processing graph...")
        start_time = datetime.now()
        
        original_node_count = len(self.nodes)
        original_edge_count = len(self.edges)
        
        # Merge duplicate nodes
        self.nodes, self.edges = self._merge_duplicates(self.nodes, self.edges)
        
        # Prune weak edges
        self.edges = self._prune_weak_edges(self.edges)
        
        duration = (datetime.now() - start_time).total_seconds()
        self.stats['stages']['processing'] = {
            'duration_seconds': duration,
            'nodes_before': original_node_count,
            'nodes_after': len(self.nodes),
            'edges_before': original_edge_count,
            'edges_after': len(self.edges),
            'nodes_merged': original_node_count - len(self.nodes),
            'edges_pruned': original_edge_count - len(self.edges)
        }
        
        logger.info(f"Stage 8 complete: {len(self.nodes)} nodes, {len(self.edges)} edges in {duration:.1f}s")
    
    def _stage_9_centrality_analysis(self) -> None:
        """Stage 9: Calculate centrality and importance scores."""
        logger.info("Stage 9: Calculating centrality...")
        start_time = datetime.now()
        
        importance_scores = self.centrality_analyzer.analyze(self.nodes, self.edges)
        
        # Store importance in node metadata
        nodes_updated = 0
        for node in self.nodes:
            if node.id in importance_scores:
                if not hasattr(node, 'metadata') or node.metadata is None:
                    node.metadata = {}
                node.metadata['importance_score'] = importance_scores[node.id]
                nodes_updated += 1
        
        duration = (datetime.now() - start_time).total_seconds()
        self.stats['stages']['centrality'] = {
            'duration_seconds': duration,
            'nodes_scored': nodes_updated,
            'top_nodes': self._get_top_nodes(importance_scores, 5)
        }
        
        logger.info(f"Stage 9 complete: scored {nodes_updated} nodes in {duration:.1f}s")
    
    def _stage_10_storage(self) -> None:
        """Stage 10: Store the final graph."""
        logger.info("Stage 10: Storing graph...")
        start_time = datetime.now()
        
        # Store in batches for better performance
        self.graph_store.add_nodes(self.nodes)
        self.graph_store.add_edges(self.edges)
        
        duration = (datetime.now() - start_time).total_seconds()
        final_stats = self.graph_store.stats()
        
        self.stats['stages']['storage'] = {
            'duration_seconds': duration,
            'nodes_stored': final_stats['node_count'],
            'edges_stored': final_stats['edge_count'],
            'db_size_mb': final_stats['db_size_mb']
        }
        
        logger.info(f"Stage 10 complete: stored in {duration:.1f}s")
    
    def _merge_duplicates(self, nodes: List[Node], edges: List[Edge]) -> Tuple[List[Node], List[Edge]]:
        """Merge duplicate nodes and update edge references."""
        # Simple deduplication by title for entities and tags
        unique_nodes = []
        node_mapping = {}  # old_id -> new_id
        
        seen_titles = {}  # (type, title) -> node_id
        
        for node in nodes:
            key = (node.type.value, node.title.lower().strip())
            
            if key in seen_titles:
                # Duplicate found, map to existing node
                existing_id = seen_titles[key]
                node_mapping[node.id] = existing_id
            else:
                # New unique node
                seen_titles[key] = node.id
                unique_nodes.append(node)
                node_mapping[node.id] = node.id
        
        # Update edges with new node IDs
        updated_edges = []
        for edge in edges:
            new_source = node_mapping.get(edge.source_id, edge.source_id)
            new_target = node_mapping.get(edge.target_id, edge.target_id)
            
            # Skip self-loops
            if new_source != new_target:
                # Update edge IDs if needed
                if new_source != edge.source_id or new_target != edge.target_id:
                    edge.source_id = new_source
                    edge.target_id = new_target
                    edge.id = f"{edge.type.value}_{new_source}_{new_target}"
                
                updated_edges.append(edge)
        
        logger.info(f"Merged {len(nodes) - len(unique_nodes)} duplicate nodes")
        return unique_nodes, updated_edges
    
    def _prune_weak_edges(self, edges: List[Edge], min_weight: float = 0.3, max_per_node: int = 50) -> List[Edge]:
        """Remove weak edges and limit edges per node."""
        # Filter by minimum weight
        strong_edges = [edge for edge in edges if edge.weight >= min_weight]
        
        # Count edges per node
        node_edge_count = defaultdict(int)
        pruned_edges = []
        
        # Sort by weight descending to keep strongest edges
        sorted_edges = sorted(strong_edges, key=lambda e: e.weight, reverse=True)
        
        for edge in sorted_edges:
            source_count = node_edge_count[edge.source_id]
            target_count = node_edge_count[edge.target_id]
            
            # Keep edge if both nodes are under limit
            if source_count < max_per_node and target_count < max_per_node:
                pruned_edges.append(edge)
                node_edge_count[edge.source_id] += 1
                node_edge_count[edge.target_id] += 1
        
        logger.info(f"Pruned {len(edges) - len(pruned_edges)} weak/excess edges")
        return pruned_edges
    
    def _get_top_nodes(self, importance_scores: Dict[str, float], limit: int = 5) -> List[Dict[str, Any]]:
        """Get top N most important nodes for reporting."""
        if not importance_scores:
            return []
        
        sorted_scores = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        
        top_nodes = []
        for node_id, score in sorted_scores:
            # Find the node
            node = next((n for n in self.nodes if n.id == node_id), None)
            if node:
                top_nodes.append({
                    'id': node_id,
                    'title': node.title,
                    'type': node.type.value,
                    'importance_score': round(score, 4)
                })
        
        return top_nodes