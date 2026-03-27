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
    
    def build(self, source_path: str, force_rebuild: bool = False, skip_semantic: bool = False, 
              incremental: bool = True) -> Dict[str, Any]:
        """
        Execute the complete graph build pipeline.
        
        Args:
            source_path: Path to markdown directory
            force_rebuild: Clear existing graph before building
            skip_semantic: Skip Stage 7 (semantic similarity). Saves ~2GB RAM.
                          Explicit links, entities, temporal, topics still built.
            incremental: Enable incremental updates if provenance exists
            
        Returns:
            Build statistics and results
        """
        start_time = datetime.now()
        
        logger.info(f"Starting graph build from: {source_path}")
        
        # Check for incremental update if enabled
        if incremental and not force_rebuild and self.graph_store._provenance_exists():
            logger.info("Incremental mode: detecting changes...")
            changed_files, deleted_files = self._detect_changes(source_path, source_name="")
            if changed_files or deleted_files:
                logger.info(f"Incremental update: {len(changed_files)} changed, {len(deleted_files)} deleted")
                return self.update_incremental(changed_files, deleted_files, source_name="")
            else:
                logger.info("No changes detected, skipping build")
                return {'success': True, 'incremental': True, 'changes': False}
        
        if force_rebuild:
            logger.info("Force rebuild: clearing existing graph")
            self.graph_store.clear()
            # Clear provenance table too
            with self.graph_store._connect() as conn:
                conn.execute('DELETE FROM file_provenance')
                conn.commit()
        
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
            # Don't nullify self.semantic_analyzer - it breaks subsequent builds
            # self.semantic_analyzer = None
            if self.embedder:
                self.embedder._model = None
                self.embedder._cache.clear()
                # Don't nullify self.embedder - it breaks subsequent builds
                # self.embedder = None
            # Don't nullify self.topic_analyzer - it breaks subsequent builds  
            # self.topic_analyzer = None
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
        
        # Write file provenance for full build (enables future incremental updates)
        self._write_file_provenance(source_path, "")
        
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
                        conn.execute('INSERT OR IGNORE INTO document_tags (doc_id, tag_id) VALUES (?,?)', (doc_id, tag_id))
                    tags_extracted += len(all_tags)
                
                # Add cross-references via shared connection
                for wiki_link in wiki_links:
                    conn.execute('''
                        INSERT OR IGNORE INTO cross_refs (doc_id, ref_target, ref_type)
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
            
            # Create node with source file tracking
            abs_file_path = str(Path(file_path).resolve())
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
                    'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'source_file': abs_file_path,
                    'source_name': ""
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
        
        # Set source_file and source_name on all nodes and edges during full build
        for node in self.nodes:
            if not hasattr(node, 'metadata') or node.metadata is None:
                node.metadata = {}
            # Note nodes already have source_file set, others use empty string
            if 'source_file' not in node.metadata:
                node.metadata['source_file'] = ""
            if 'source_name' not in node.metadata:
                node.metadata['source_name'] = ""
        
        for edge in self.edges:
            if not hasattr(edge, 'metadata') or edge.metadata is None:
                edge.metadata = {}
            if 'source_file' not in edge.metadata:
                edge.metadata['source_file'] = ""
            if 'source_name' not in edge.metadata:
                edge.metadata['source_name'] = ""
        
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
    
    def _detect_changes(self, source_path: str, source_name: str) -> Tuple[List[str], List[str]]:
        """
        Compare filesystem mtimes vs file_provenance table.
        Returns (changed_files, deleted_files) as absolute path strings.
        A file is 'changed' if: not in provenance, OR mtime > last_modified, OR content_hash differs.
        A file is 'deleted' if: in provenance but not on filesystem.
        """
        source_path = Path(source_path).resolve()
        
        # Scan filesystem for current markdown files
        current_files = set()
        for md_file in source_path.rglob("*.md"):
            if not md_file.is_symlink() or md_file.resolve().is_relative_to(source_path):
                current_files.add(str(md_file.resolve()))
        
        # Get tracked files from provenance
        tracked_files = {}
        try:
            with self.graph_store._connect() as conn:
                rows = conn.execute('''
                    SELECT file_path, last_modified, content_hash 
                    FROM file_provenance 
                    WHERE source_name = ?
                ''', (source_name,)).fetchall()
                
                for file_path, last_modified, content_hash in rows:
                    tracked_files[file_path] = {
                        'last_modified': last_modified,
                        'content_hash': content_hash
                    }
        except Exception as e:
            logger.warning(f"Failed to query file provenance: {e}")
            # If we can't read provenance, treat all current files as changed
            return list(current_files), []
        
        changed_files = []
        deleted_files = []
        
        # Check for changes in current files
        for file_path in current_files:
            if file_path not in tracked_files:
                # New file
                changed_files.append(file_path)
                continue
            
            try:
                file_stat = Path(file_path).stat()
                current_mtime = file_stat.st_mtime
                
                # Check mtime first (faster)
                if current_mtime > tracked_files[file_path]['last_modified']:
                    # File modified - also check content hash to be sure
                    current_hash = self._compute_file_hash(file_path)
                    if current_hash != tracked_files[file_path]['content_hash']:
                        changed_files.append(file_path)
                elif tracked_files[file_path]['content_hash']:
                    # Mtime same, but double-check content hash
                    current_hash = self._compute_file_hash(file_path)
                    if current_hash != tracked_files[file_path]['content_hash']:
                        changed_files.append(file_path)
            except Exception as e:
                logger.warning(f"Failed to check {file_path}: {e}")
                changed_files.append(file_path)  # Safe default
        
        # Check for deleted files
        for tracked_path in tracked_files:
            if tracked_path not in current_files:
                deleted_files.append(tracked_path)
        
        logger.info(f"Change detection: {len(changed_files)} changed, {len(deleted_files)} deleted")
        return changed_files, deleted_files
    
    def _delete_file(self, file_path: str, source_name: str, conn) -> Tuple[List[str], int, int]:
        """
        CRITICAL ORDER: Find dependent edges BEFORE deleting nodes.
        1. Get owned node IDs
        2. Find cross-file edges pointing TO owned nodes (collect source_files as dependents)
        3. Delete edges (owned + pointing-to-owned)
        4. Delete owned nodes
        5. Delete from file_provenance
        Returns (dependent_files, nodes_removed, edges_removed)
        """
        abs_path = str(Path(file_path).resolve())
        
        # Step 1: Get owned node IDs FIRST (before any deletion)
        owned_node_ids = [
            row[0] for row in conn.execute(
                'SELECT id FROM nodes WHERE source_file = ? AND source_name = ?',
                (abs_path, source_name)
            ).fetchall()
        ]
        if not owned_node_ids:
            return [], 0, 0
        
        placeholders = ','.join('?' * len(owned_node_ids))
        
        # Step 2: Find cross-file edges pointing TO owned nodes (while nodes still exist)
        dependent_rows = conn.execute(f'''
            SELECT DISTINCT e.source_file
            FROM edges e
            JOIN nodes n ON e.source_id = n.id
            WHERE e.target_id IN ({placeholders})
              AND (e.source_file != ? OR e.source_file IS NULL)
              AND e.source_file IS NOT NULL
        ''', owned_node_ids + [abs_path]).fetchall()
        dependent_files = [row[0] for row in dependent_rows if row[0]]
        
        # Step 3: Now safe to delete edges (both owned and pointing-to-owned)
        conn.execute(f'''
            DELETE FROM edges
            WHERE source_file = ? OR target_id IN ({placeholders})
        ''', [abs_path] + owned_node_ids)
        
        edges_removed = conn.execute('SELECT changes()').fetchone()[0]
        
        # Step 4: Delete owned nodes
        conn.execute(
            'DELETE FROM nodes WHERE source_file = ? AND source_name = ?',
            (abs_path, source_name)
        )
        nodes_removed = conn.execute('SELECT changes()').fetchone()[0]
        
        # Step 5: Remove from provenance
        conn.execute(
            'DELETE FROM file_provenance WHERE file_path = ? AND source_name = ?',
            (abs_path, source_name)
        )
        
        return dependent_files, nodes_removed, edges_removed
    
    def update_incremental(
        self,
        changed_files: List[str],
        deleted_files: List[str],
        source_name: str = "",
        embedder=None
    ) -> Dict[str, Any]:
        """
        Atomic incremental update.
        Flow:
        1. Normalize all paths to absolute
        2. Process deletions via _delete_file(), collect dependent_files
        3. Merge dependent_files into changed_files (dedup)
        4. For each changed file: read content, build Node objects (same as Stage 1 in build())
        5. Remove old nodes/edges owned by changed files from graph DB
        6. Run each analyzer's analyze_incremental(changed_nodes, all_nodes, existing_edges, changed_files_set)
        7. Add new nodes/edges to graph with source_file set
        8. Update file_provenance for changed files
        Everything in a single _connect() context (auto-commit/rollback on exception)
        Returns stats dict: {files_updated, files_deleted, nodes_added, edges_added, duration_s}
        """
        start_time = datetime.now()
        
        # Normalize all paths to absolute
        changed_files = [str(Path(f).resolve()) for f in changed_files]
        deleted_files = [str(Path(f).resolve()) for f in deleted_files]
        
        stats = {
            'start_time': start_time.isoformat(),
            'incremental': True,
            'files_changed': len(changed_files),
            'files_deleted': len(deleted_files),
            'nodes_added': 0,
            'edges_added': 0,
            'nodes_removed': 0,
            'edges_removed': 0
        }
        
        try:
            with self.graph_store._transaction() as conn:
                # Process deletions and collect dependent files
                all_dependent_files = set()
                total_nodes_removed = 0
                total_edges_removed = 0
                
                for deleted_file in deleted_files:
                    dependent_files, nodes_removed, edges_removed = self._delete_file(
                        deleted_file, source_name, conn
                    )
                    all_dependent_files.update(dependent_files)
                    total_nodes_removed += nodes_removed
                    total_edges_removed += edges_removed
                
                stats['nodes_removed'] = total_nodes_removed
                stats['edges_removed'] = total_edges_removed
                
                # Merge dependent files into changed files (dedup)
                all_changed_files = set(changed_files) | all_dependent_files
                all_changed_files = list(all_changed_files)
                
                if not all_changed_files:
                    stats['files_updated'] = 0
                    stats['success'] = True
                    return stats
                
                # Read content and build nodes for changed files
                changed_nodes = []
                for file_path in all_changed_files:
                    try:
                        file_obj = Path(file_path)
                        if not file_obj.exists():
                            continue
                        
                        content = file_obj.read_text(encoding='utf-8', errors='ignore')
                        title = file_obj.stem.replace('-', ' ').replace('_', ' ').title()
                        
                        node_id = f"note_{hashlib.md5(file_path.encode()).hexdigest()[:12]}"
                        stat = file_obj.stat()
                        
                        node = Node(
                            id=node_id,
                            type=NodeType.NOTE,
                            title=title,
                            content=content,
                            metadata={
                                'file_path': file_path,
                                'relative_path': file_path,  # Simplified for incremental
                                'filename': file_obj.name,
                                'word_count': len(content.split()),
                                'char_count': len(content),
                                'created_time': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                                'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                                'source_file': file_path,
                                'source_name': source_name
                            }
                        )
                        changed_nodes.append(node)
                        
                    except Exception as e:
                        logger.warning(f"Failed to process changed file {file_path}: {e}")
                
                # Remove old nodes/edges owned by changed files
                for file_path in all_changed_files:
                    conn.execute('DELETE FROM edges WHERE source_file = ?', (file_path,))
                    conn.execute('DELETE FROM nodes WHERE source_file = ?', (file_path,))
                
                # If embedder not provided, lazy-create one (ONNX loads on first use)
                _embedder = embedder or Embedder()

                # Reload all current nodes/edges for full-corpus lookups
                all_nodes = self.graph_store.get_all_nodes()
                existing_edges = self.graph_store.get_all_edges()
                changed_files_set = set(all_changed_files)

                new_nodes: List[Node] = []
                new_edges: List[Edge] = []

                # --- Run all 6 analyzers in incremental mode ---
                # Each returns (nodes, edges) for changed files only.
                # Process one changed file at a time so source_file attribution is correct.

                for file_path in all_changed_files:
                    file_changed_nodes = [n for n in changed_nodes
                                          if n.metadata.get('file_path') == file_path
                                          or n.metadata.get('source_file') == file_path]
                    if not file_changed_nodes:
                        continue

                    per_file_nodes: List[Node] = list(file_changed_nodes)
                    per_file_edges: List[Edge] = []

                    # 1. Explicit (wikilinks, tags)
                    try:
                        en, ee = self.explicit_analyzer.analyze_incremental(
                            file_changed_nodes, all_nodes, existing_edges, {file_path})
                        per_file_nodes.extend(en)
                        per_file_edges.extend(ee)
                    except Exception as e:
                        logger.warning(f"ExplicitAnalyzer incremental failed for {file_path}: {e}")

                    # 2. Entity
                    if self.entity_analyzer:
                        try:
                            en, ee = self.entity_analyzer.analyze_incremental(
                                file_changed_nodes, all_nodes, existing_edges, {file_path})
                            per_file_nodes.extend(en)
                            per_file_edges.extend(ee)
                        except Exception as e:
                            logger.warning(f"EntityAnalyzer incremental failed for {file_path}: {e}")

                    # 3. Temporal
                    try:
                        en, ee = self.temporal_analyzer.analyze_incremental(
                            file_changed_nodes, all_nodes, existing_edges, {file_path})
                        per_file_nodes.extend(en)
                        per_file_edges.extend(ee)
                    except Exception as e:
                        logger.warning(f"TemporalAnalyzer incremental failed for {file_path}: {e}")

                    # 4. Semantic (only if available)
                    if self.semantic_analyzer:
                        try:
                            self.semantic_analyzer.embedder = _embedder
                            en, ee = self.semantic_analyzer.analyze_incremental(
                                file_changed_nodes, all_nodes, existing_edges, {file_path})
                            per_file_nodes.extend(en)
                            per_file_edges.extend(ee)
                        except Exception as e:
                            logger.warning(f"SemanticAnalyzer incremental failed for {file_path}: {e}")

                    # Stamp source_file on all per-file nodes/edges before collecting
                    for node in per_file_nodes:
                        if node.metadata is None:
                            node.metadata = {}
                        node.metadata.setdefault('source_file', file_path)
                        node.metadata.setdefault('source_name', source_name)
                    for edge in per_file_edges:
                        if edge.metadata is None:
                            edge.metadata = {}
                        edge.metadata.setdefault('source_file', file_path)
                        edge.metadata.setdefault('source_name', source_name)

                    new_nodes.extend(per_file_nodes)
                    new_edges.extend(per_file_edges)

                # 5. TopicAnalyzer — full rebuild, runs once over all nodes
                if self.topic_analyzer:
                    try:
                        final_all_nodes = all_nodes + new_nodes
                        en, ee = self.topic_analyzer.analyze_incremental(
                            changed_nodes, final_all_nodes, existing_edges + new_edges, changed_files_set)
                        # Topic nodes/edges aren't file-specific — leave source_file unset
                        new_nodes.extend(en)
                        new_edges.extend(ee)
                    except Exception as e:
                        logger.warning(f"TopicAnalyzer incremental failed: {e}")

                # 6. CentralityAnalyzer — runs last over final edge state
                try:
                    final_all_nodes = all_nodes + new_nodes
                    final_all_edges = existing_edges + new_edges
                    en, ee = self.centrality_analyzer.analyze_incremental(
                        changed_nodes, final_all_nodes, final_all_edges, changed_files_set)
                    new_nodes.extend(en)
                    new_edges.extend(ee)
                except Exception as e:
                    logger.warning(f"CentralityAnalyzer incremental failed: {e}")
                
                # Store nodes and edges in database
                self._store_nodes_edges_with_provenance(new_nodes, new_edges, conn)
                
                # Update file provenance for changed files
                for file_path in all_changed_files:
                    try:
                        file_obj = Path(file_path)
                        if file_obj.exists():
                            file_hash = self._compute_file_hash(file_path)
                            file_mtime = file_obj.stat().st_mtime
                            
                            # Count nodes/edges for this file
                            node_count = len([n for n in new_nodes if n.metadata.get('source_file') == file_path])
                            edge_count = len([e for e in new_edges if e.metadata.get('source_file') == file_path])
                            
                            conn.execute('''
                                INSERT OR REPLACE INTO file_provenance 
                                (file_path, source_name, last_modified, content_hash, node_count, edge_count, indexed_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                            ''', (file_path, source_name, file_mtime, file_hash, node_count, edge_count, 
                                  datetime.now().isoformat()))
                    except Exception as e:
                        logger.warning(f"Failed to update provenance for {file_path}: {e}")

                stats.update({
                    'files_updated': len(all_changed_files),
                    'files_deleted': len(deleted_files),
                    'nodes_added': len(new_nodes),
                    'edges_added': len(new_edges),
                    'success': True
                })
                
        except Exception as e:
            logger.error(f"Incremental update failed: {e}")
            stats['error'] = str(e)
            stats['success'] = False
            return stats
        
        duration = (datetime.now() - start_time).total_seconds()
        stats['duration_seconds'] = duration
        
        logger.info(f"Incremental update completed in {duration:.1f}s: "
                   f"{stats['nodes_added']} nodes, {stats['edges_added']} edges")
        return stats
    
    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of file contents."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception:
            return ""
    
    def _write_file_provenance(self, source_path: str, source_name: str) -> None:
        """Write file provenance after full build."""
        source_path = Path(source_path).resolve()
        
        try:
            with self.graph_store._transaction() as conn:
                # Clear existing provenance for this source
                conn.execute('DELETE FROM file_provenance WHERE source_name = ?', (source_name,))
                
                # Scan and record all markdown files
                for md_file in source_path.rglob("*.md"):
                    if not md_file.is_symlink() or md_file.resolve().is_relative_to(source_path):
                        abs_path = str(md_file.resolve())
                        file_hash = self._compute_file_hash(abs_path)
                        file_mtime = md_file.stat().st_mtime
                        
                        # Count nodes/edges owned by this file
                        node_count = conn.execute(
                            'SELECT COUNT(*) FROM nodes WHERE source_file = ?', (abs_path,)
                        ).fetchone()[0]
                        edge_count = conn.execute(
                            'SELECT COUNT(*) FROM edges WHERE source_file = ?', (abs_path,)
                        ).fetchone()[0]
                        
                        conn.execute('''
                            INSERT INTO file_provenance 
                            (file_path, source_name, last_modified, content_hash, node_count, edge_count, indexed_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (abs_path, source_name, file_mtime, file_hash, node_count, edge_count,
                              datetime.now().isoformat()))
                
                logger.info(f"File provenance written for {source_name}")
                
        except Exception as e:
            logger.warning(f"Failed to write file provenance: {e}")
    
    def _store_nodes_edges_with_provenance(self, nodes: List[Node], edges: List[Edge], conn) -> None:
        """Store nodes and edges with proper source_file attribution."""
        # Store nodes
        for node in nodes:
            metadata_json = json.dumps(node.metadata) if node.metadata else "{}"
            created_at_iso = node.created_at.isoformat() if node.created_at else datetime.now().isoformat()
            source_file = node.metadata.get('source_file') if node.metadata else None
            source_name = node.metadata.get('source_name') if node.metadata else None
            
            conn.execute('''
                INSERT OR REPLACE INTO nodes (id, type, title, content, metadata, created_at, source_file, source_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                node.id,
                node.type.value,
                node.title,
                node.content,
                metadata_json,
                created_at_iso,
                source_file,
                source_name
            ))
        
        # Store edges
        for edge in edges:
            # Skip self-loops
            if edge.source_id == edge.target_id:
                continue
            
            # Check that both nodes exist
            source_exists = conn.execute('SELECT 1 FROM nodes WHERE id = ?', (edge.source_id,)).fetchone()
            target_exists = conn.execute('SELECT 1 FROM nodes WHERE id = ?', (edge.target_id,)).fetchone()
            
            if not source_exists or not target_exists:
                continue
            
            metadata_json = json.dumps(edge.metadata) if edge.metadata else "{}"
            created_at_iso = edge.created_at.isoformat() if edge.created_at else datetime.now().isoformat()
            source_file = edge.metadata.get('source_file') if edge.metadata else None
            source_name = edge.metadata.get('source_name') if edge.metadata else None
            
            conn.execute('''
                INSERT OR REPLACE INTO edges 
                (id, source_id, target_id, type, weight, confidence, metadata, created_at, source_file, source_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                edge.id,
                edge.source_id,
                edge.target_id,
                edge.type.value,
                edge.weight,
                edge.confidence,
                metadata_json,
                created_at_iso,
                source_file,
                source_name
            ))