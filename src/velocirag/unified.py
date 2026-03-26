"""
Velocirag Phase 7 Unified Search - True 3-Layer Fusion.

Revolutionary search orchestration that fuses vector similarity, metadata filtering, 
and graph discovery into a single ranked result set. Each layer contributes candidates
independently, then RRF fusion determines the final ranking.

No layer dominates. All layers contribute. True fusion.
"""

import logging
import time
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .searcher import Searcher
from .graph import GraphStore, GraphQuerier
from .metadata import MetadataStore
from .tracker import UsageTracker
from .rrf import reciprocal_rank_fusion

logger = logging.getLogger("velocirag.unified")


class UnifiedSearchError(Exception):
    """Base exception for unified search failures."""
    pass


class UnifiedSearch:
    """
    Unified search orchestrator with true 3-layer fusion.
    
    The apex predator of search. Combines vector precision, metadata structure,
    and graph connections through battle-tested RRF fusion. Each layer hunts
    independently, then they converge on the truth.
    """
    
    def __init__(self, searcher: Searcher, graph_store: Optional[GraphStore] = None,
                 metadata_store: Optional[MetadataStore] = None, tracker: Optional[UsageTracker] = None):
        """
        Initialize unified search orchestrator.
        
        Args:
            searcher: Vector searcher (required)
            graph_store: Knowledge graph (optional — degrades gracefully without it)
            metadata_store: Metadata store for structured queries (optional)
            tracker: Usage tracker for logging search hits (optional)
        """
        if not searcher:
            raise ValueError("Searcher is required")
        
        self.searcher = searcher
        self.graph_store = graph_store
        self.graph_querier = GraphQuerier(graph_store) if graph_store else None
        self.metadata_store = metadata_store
        self.tracker = tracker
        
        logger.info(f"UnifiedSearch initialized: vector={True}, graph={bool(graph_store)}, "
                   f"metadata={bool(metadata_store)}, tracker={bool(tracker)}")
    
    def search(self, query: str, limit: int = 5, threshold: float = 0.3, 
               enrich_graph: bool = True, filters: Optional[Dict] = None) -> Dict[str, Any]:
        """
        True 3-layer fusion search: vector + metadata + graph → RRF → final results.
        
        Revolutionary pipeline:
        1. Layer 1 — Vector: embed query → FAISS → top candidates by cosine similarity
        2. Layer 2 — Metadata: query SQL → docs matching filters or title text search  
        3. Layer 3 — Graph: find query in graph → traverse edges → discover connected docs
        4. RRF Fusion: merge all 3 candidate lists using reciprocal rank fusion
        5. Load full content for top fused results + graph enrichment
        
        Args:
            query: Search query string
            limit: Maximum results to return
            threshold: Minimum similarity threshold (for vector layer)
            enrich_graph: Enable graph enrichment of final results
            filters: Optional metadata filters (tags, status, category, etc.)
            
        Returns:
            Dictionary with fused search results and layer statistics
        """
        start_time = time.time()
        
        if not query or not query.strip():
            return {
                'results': [],
                'query': query,
                'total_results': 0,
                'search_mode': 'empty_query',
                'search_time_ms': 0.0,
                'layer_stats': {}
            }
        
        query = query.strip()
        layer_stats = {}
        
        # === LAYER 1: VECTOR SEARCH ===
        # Always runs — this is the foundation
        vector_start = time.time()
        try:
            vector_result = self.searcher.search(
                query=query,
                limit=limit * 3,  # Get more candidates for fusion
                threshold=threshold,
                include_stats=False
            )
            vector_results = vector_result.get('results', [])
            layer_stats['vector'] = {
                'candidates': len(vector_results),
                'time_ms': round((time.time() - vector_start) * 1000, 2)
            }
        except Exception as e:
            raise UnifiedSearchError(f"Vector search failed: {e}")
        
        # === LAYER 2: METADATA SEARCH ===
        # Runs if metadata_store exists
        metadata_results = []
        if self.metadata_store:
            metadata_start = time.time()
            try:
                if filters:
                    # Filtered query — use provided filters
                    metadata_docs = self.metadata_store.query(**filters, limit=limit * 3)
                else:
                    # Unfiltered query — text search against titles
                    metadata_docs = self._search_metadata_titles(query, limit=limit * 3)
                
                metadata_results = metadata_docs
                layer_stats['metadata'] = {
                    'candidates': len(metadata_results),
                    'time_ms': round((time.time() - metadata_start) * 1000, 2),
                    'search_type': 'filtered' if filters else 'title_search'
                }
            except Exception as e:
                logger.warning(f"Metadata search failed: {e}")
                layer_stats['metadata'] = {
                    'candidates': 0,
                    'time_ms': 0,
                    'error': str(e)
                }
        
        # === LAYER 3: GRAPH SEARCH ===
        # Runs if graph_querier exists and enrich_graph=True
        graph_results = []
        if self.graph_querier and enrich_graph:
            graph_start = time.time()
            try:
                # Try multiple graph search strategies
                graph_candidates = []
                
                # Strategy 1: Find connections by full query
                try:
                    connections = self.graph_querier.find_connections(query, depth=2)
                    if 'connections_by_type' in connections:
                        for rel_type, conns in connections['connections_by_type'].items():
                            for conn in conns[:3]:  # Limit per relation type
                                graph_candidates.append({
                                    'title': conn['node'],
                                    'score': conn.get('weight', 0.5),
                                    'source': f'connection_{rel_type}'
                                })
                except Exception:
                    pass
                
                # Strategy 2: Try individual words from query
                query_words = query.lower().split()
                for word in query_words:
                    if len(word) > 2:  # Skip very short words
                        try:
                            connections = self.graph_querier.find_connections(word, depth=1)
                            if 'connections_by_type' in connections:
                                for rel_type, conns in connections['connections_by_type'].items():
                                    for conn in conns[:2]:  # Limit per word
                                        graph_candidates.append({
                                            'title': conn['node'],
                                            'score': conn.get('weight', 0.3),
                                            'source': f'word_{word}_{rel_type}'
                                        })
                        except Exception:
                            pass
                
                # Strategy 3: Topic web if query looks topical
                if len(query.split()) <= 3:  # Short queries might be topics
                    try:
                        topic_web = self.graph_querier.get_topic_web(query)
                        if 'related_nodes' in topic_web:
                            for node in topic_web['related_nodes'][:5]:
                                graph_candidates.append({
                                    'title': node['title'],
                                    'score': node.get('connection_strength', 0.3),
                                    'source': 'topic_web'
                                })
                    except Exception:
                        pass
                
                # Remove duplicates
                seen_titles = set()
                graph_results = []
                for candidate in graph_candidates:
                    title = candidate['title']
                    if title not in seen_titles:
                        seen_titles.add(title)
                        graph_results.append(candidate)
                
                layer_stats['graph'] = {
                    'candidates': len(graph_results),
                    'time_ms': round((time.time() - graph_start) * 1000, 2),
                    'strategies_used': ['connections', 'topic_web']
                }
            except Exception as e:
                logger.warning(f"Graph search failed: {e}")
                layer_stats['graph'] = {
                    'candidates': 0,
                    'time_ms': 0,
                    'error': str(e)
                }
        
        # === STEP 4: NORMALIZE ALL RESULTS FOR RRF FUSION ===
        fusion_start = time.time()
        
        # Normalize vector results: extract filename and create ranked list
        vector_ranked = []
        for i, result in enumerate(vector_results):
            filename = self._extract_filename(result)
            if filename:
                vector_ranked.append({
                    'doc_id': filename,
                    'score': result.get('similarity', result.get('score', 0)),
                    'content': result.get('content', result.get('chunk', '')),
                    'metadata': result.get('metadata', {}),
                    '_source': 'vector',
                    '_rank': i + 1
                })
        
        # Normalize metadata results: extract filename and create ranked list
        metadata_ranked = []
        for i, result in enumerate(metadata_results):
            filename = result.get('filename', '')
            if filename:
                metadata_ranked.append({
                    'doc_id': filename,
                    'score': 1.0,  # Metadata matches get uniform high score
                    'content': result.get('title', ''),
                    'metadata': {'source_doc': result},
                    '_source': 'metadata',
                    '_rank': i + 1
                })
        
        # Normalize graph results: title to filename mapping
        graph_ranked = []
        for i, result in enumerate(graph_results):
            # Convert graph node title to filename format
            title = result['title']
            filename = self._title_to_filename(title)
            if filename:
                # Cap graph scores at 0.5 — graph candidates should SUPPORT vector results, not override them
                graph_score = min(result.get('score', 0.3), 0.5)
                graph_ranked.append({
                    'doc_id': filename,
                    'score': graph_score,
                    'content': title,
                    'metadata': {'graph_source': result['source']},
                    '_source': 'graph',
                    '_rank': i + 1
                })
        
        # === STEP 5: RRF FUSION ACROSS ALL 3 LAYERS ===
        available_lists = []
        if vector_ranked:
            available_lists.append(vector_ranked)
        if metadata_ranked:
            available_lists.append(metadata_ranked)
        if graph_ranked:
            available_lists.append(graph_ranked)
        
        try:
            # Use custom doc_id function for filename-based deduplication
            def filename_doc_id(result):
                return result.get('doc_id', '')
            
            fused_results = reciprocal_rank_fusion(
                available_lists,
                k=60,
                doc_id_fn=filename_doc_id
            )
        except Exception as e:
            logger.error(f"RRF fusion failed: {e}")
            # Fallback to vector results only
            fused_results = vector_ranked
        
        fusion_time_ms = round((time.time() - fusion_start) * 1000, 2)
        
        # === VECTOR CONFIRMATION FILTER ===
        # Filter: graph-only candidates must have minimum vector similarity to survive
        # This prevents random graph connections from polluting results
        confirmed_results = []
        vector_filenames = {self._extract_filename(r) for r in vector_results if self._extract_filename(r)}
        
        # Only apply filtering if we have valid vector results to compare against
        if vector_filenames or not graph_ranked:
            for fused_result in fused_results:
                filename = fused_result.get('doc_id', '')
                source = fused_result.get('_source', '')
                
                if filename in vector_filenames:
                    # Found by vector search — always keep
                    confirmed_results.append(fused_result)
                elif source == 'metadata':
                    # Found by metadata — keep (structured match is trustworthy)
                    confirmed_results.append(fused_result)
                elif source == 'vector' or not vector_filenames:
                    # Vector result or no valid vector filenames to compare — always keep
                    confirmed_results.append(fused_result)
                else:
                    # Graph-only candidate — verify it has some vector relevance
                    # Search the vector results for any similarity to this doc
                    has_vector_support = False
                    for vr in vector_results:
                        if self._extract_filename(vr) == filename:
                            has_vector_support = True
                            break
                    
                    if has_vector_support:
                        confirmed_results.append(fused_result)
                    else:
                        # Check: does this doc appear ANYWHERE in a wider vector search?
                        # Skip it — graph-only with no vector confirmation is noise
                        logger.debug(f"Filtered graph-only candidate: {filename}")
        else:
            # No filtering if we have no vector filenames to validate against
            confirmed_results = fused_results
        
        fused_results = confirmed_results
        
        # === STEP 6: LOAD FULL CONTENT FOR TOP FUSED RESULTS ===
        final_results = []
        for fused_result in fused_results[:limit]:
            # Determine which original result to use for full content
            filename = fused_result.get('doc_id', '')
            rrf_score = fused_result.get('metadata', {}).get('rrf_score', 0)
            source_layers = fused_result.get('_source', 'unknown')
            
            # Find the best original result (prefer vector for full content)
            best_original = None
            for result in vector_results:
                if self._extract_filename(result) == filename:
                    best_original = result
                    break
            
            # If not found in vector results, create from fused result
            if not best_original:
                best_original = {
                    'content': fused_result.get('content', ''),
                    'score': fused_result.get('score', 0),
                    'similarity': fused_result.get('score', 0),
                    'metadata': {
                        'file_path': filename,
                        'fusion_only': True
                    }
                }
            
            # Create enriched result (preserve original field structure)
            enriched_result = best_original.copy()
            
            # Ensure consistent score/similarity fields
            if 'similarity' not in enriched_result and 'score' in enriched_result:
                enriched_result['similarity'] = enriched_result['score']
            elif 'score' not in enriched_result and 'similarity' in enriched_result:
                enriched_result['score'] = enriched_result['similarity']
            
            # Ensure metadata dict exists
            if 'metadata' not in enriched_result:
                enriched_result['metadata'] = {}
            else:
                enriched_result['metadata'] = enriched_result['metadata'].copy()
            
            # Add fusion metadata
            enriched_result['metadata']['rrf_score'] = rrf_score
            enriched_result['metadata']['source_layers'] = source_layers
            
            # Check if this result was found in metadata layer (for backward compatibility)
            if metadata_results:
                for meta_result in metadata_results:
                    if meta_result.get('filename', '') == filename:
                        enriched_result['_metadata_match'] = True
                        break
            
            # Initialize graph metadata
            enriched_result['metadata']['graph_connections'] = []
            enriched_result['metadata']['related_notes'] = []
            enriched_result['metadata']['found_in_graph'] = False
            
            final_results.append(enriched_result)
        
        # === STEP 7: GRAPH ENRICHMENT OF FINAL RESULTS ===
        enrichment_errors = []
        graph_enriched_count = 0
        
        if self.graph_querier and enrich_graph:
            for result in final_results:
                try:
                    graph_info = self._enrich_with_graph(result)
                    if graph_info['success']:
                        result['metadata']['graph_connections'] = graph_info['connections']
                        result['metadata']['related_notes'] = graph_info['related_notes']
                        result['metadata']['found_in_graph'] = graph_info['found_in_graph']
                        graph_enriched_count += 1
                    elif graph_info.get('error'):
                        enrichment_errors.append(graph_info['error'])
                except Exception as e:
                    enrichment_errors.append(f"Enrichment error: {e}")
        
        # === STEP 8: USAGE TRACKING ===
        if self.tracker:
            try:
                for result in final_results:
                    filename = self._extract_filename(result)
                    if filename:
                        self.tracker.log_search_hit(filename, query)
            except Exception as e:
                logger.warning(f"Usage tracking failed: {e}")
        
        # === STEP 9: DETERMINE SEARCH MODE ===
        # Based on what layers were attempted, not what returned results
        layers_attempted = ['vector']  # Vector always runs
        
        if self.metadata_store:
            layers_attempted.append('metadata')
        
        if self.graph_querier and enrich_graph:
            layers_attempted.append('graph')
        
        if len(layers_attempted) == 3:
            search_mode = 'unified_full'
        elif len(layers_attempted) == 2:
            if 'metadata' in layers_attempted and 'graph' in layers_attempted:
                search_mode = 'vector_metadata_graph'  # This case shouldn't happen
            elif 'metadata' in layers_attempted:
                search_mode = 'vector_metadata'
            elif 'graph' in layers_attempted:
                search_mode = 'vector_graph'
            else:
                search_mode = 'vector_only'
        else:
            search_mode = 'vector_only'
        
        # === FINAL RESPONSE ===
        total_time_ms = round((time.time() - start_time) * 1000, 2)
        
        response = {
            'results': final_results,
            'query': query,
            'total_results': len(final_results),
            'search_time_ms': total_time_ms,
            'search_mode': search_mode,
            'layer_stats': layer_stats,
            'fusion_stats': {
                'fusion_time_ms': fusion_time_ms,
                'layers_fused': len(available_lists),
                'total_candidates': sum(len(lst) for lst in available_lists),
                'graph_enriched': graph_enriched_count
            },
            # Backward compatibility - keep enrichment_stats for tests
            'enrichment_stats': {
                'vector_results': len(vector_results),
                'graph_enriched': graph_enriched_count,
                'graph_available': bool(self.graph_store and enrich_graph),
                'metadata_available': bool(self.metadata_store),
                'metadata_matches': len(metadata_results) if self.metadata_store else 0
            }
        }
        
        # Add layer-specific timing for backward compatibility
        if 'search_time_ms' in vector_result:
            response['vector_time_ms'] = vector_result['search_time_ms']
        if metadata_results and 'metadata' in layer_stats:
            response['metadata_time_ms'] = layer_stats['metadata']['time_ms']
        if len(available_lists) > 1:
            response['fusion_time_ms'] = fusion_time_ms
            
        if enrichment_errors:
            response['graph_errors'] = enrichment_errors[:3]
        
        return response
    
    def _search_metadata_titles(self, query: str, limit: int) -> List[Dict]:
        """
        Search metadata store by matching query against document titles.
        
        Args:
            query: Search query
            limit: Max results
            
        Returns:
            List of matching documents
        """
        if not self.metadata_store:
            return []
        
        try:
            import sqlite3
            conn = sqlite3.connect(self.metadata_store.db_path)
            try:
                # Simple title search using LIKE
                query_pattern = f"%{query}%"
                cursor = conn.execute('''
                    SELECT * FROM documents 
                    WHERE title LIKE ? OR filename LIKE ?
                    ORDER BY 
                        CASE WHEN title LIKE ? THEN 0 ELSE 1 END,
                        updated_at DESC
                    LIMIT ?
                ''', (query_pattern, query_pattern, query_pattern, limit))
                
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                
                results = []
                for row in rows:
                    doc_dict = dict(zip(columns, row))
                    # Parse JSON meta field if present
                    if doc_dict.get('meta'):
                        try:
                            import json
                            doc_dict['meta'] = json.loads(doc_dict['meta'])
                        except json.JSONDecodeError:
                            doc_dict['meta'] = {}
                    results.append(doc_dict)
                
                return results
            finally:
                conn.close()
        except Exception as e:
            logger.warning(f"Metadata title search failed: {e}")
            return []
    
    def _extract_filename(self, result: Dict) -> str:
        """
        Extract filename from search result for deduplication.
        
        Args:
            result: Search result dictionary
            
        Returns:
            Filename string or empty if not found
        """
        metadata = result.get('metadata', {})
        file_path = metadata.get('file_path', '')
        
        if file_path:
            # Handle various file path formats
            if '::' in file_path:
                # Chunk format: "prefix::file_path::suffix"
                parts = file_path.split('::')
                if len(parts) >= 2:
                    file_path = parts[1] if len(parts) > 2 else parts[0]
            
            # Extract just the filename
            return Path(file_path).name
        
        # Fallback to doc_id or other identifiers
        return result.get('doc_id', '')
    
    def _title_to_filename(self, title: str) -> str:
        """
        Convert graph node title to likely filename format.
        
        Args:
            title: Node title from graph
            
        Returns:
            Probable filename
        """
        if not title:
            return ''
        
        # Clean title for filename matching
        # Common patterns: "python-guide" -> "python-guide.md"
        cleaned = title.lower().strip()
        
        # If it already looks like a filename, return as-is
        if '.' in cleaned and cleaned.count('.') == 1:
            return cleaned
        
        # Common extensions to try
        common_extensions = ['.md', '.txt', '.rst', '.py', '.js', '.html']
        
        # Return with .md as most common for notes
        return f"{cleaned}.md"
    
    def _enrich_with_graph(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich a single search result with graph connections.
        
        Args:
            result: Vector search result to enrich
            
        Returns:
            Dictionary with enrichment data and success status
        """
        try:
            # Extract note title from file path
            note_title = self._extract_note_title(result)
            if not note_title:
                return {
                    'success': False,
                    'error': 'Could not extract note title from result',
                    'connections': [],
                    'related_notes': [],
                    'found_in_graph': False
                }
            
            # Find connections using graph querier
            connections_data = self.graph_querier.find_connections(note_title, depth=2)
            
            if 'error' in connections_data:
                return {
                    'success': False,
                    'error': connections_data['error'],
                    'connections': [],
                    'related_notes': [],
                    'found_in_graph': False
                }
            
            # Extract connection information
            total_connections = connections_data.get('total_connections', 0)
            found_in_graph = total_connections > 0
            
            # Get connected node titles
            connections = []
            related_notes = []
            
            connections_by_type = connections_data.get('connections_by_type', {})
            for relation_type, type_connections in connections_by_type.items():
                for conn in type_connections[:3]:  # Limit per type
                    node_title = conn['node']
                    connections.append(node_title)
                    related_notes.append(node_title)
            
            # Deduplicate and limit
            unique_connections = list(dict.fromkeys(connections))[:5]
            unique_related = list(dict.fromkeys(related_notes))[:5]
            
            return {
                'success': True,
                'connections': unique_connections,
                'related_notes': unique_related,
                'found_in_graph': found_in_graph,
                'total_graph_connections': total_connections
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'connections': [],
                'related_notes': [],
                'found_in_graph': False
            }
    
    def _extract_note_title(self, result: Dict[str, Any]) -> Optional[str]:
        """
        Extract note title from search result for graph lookup.
        
        Args:
            result: Vector search result with metadata
            
        Returns:
            Clean note title or None if extraction fails
        """
        metadata = result.get('metadata', {})
        file_path = metadata.get('file_path', '')
        
        if not file_path or not file_path.strip():
            return None
        
        try:
            # Convert to Path for easier manipulation
            path = Path(file_path)
            
            # Get filename without extension
            title = path.stem
            
            return title if title and title.strip() else None
            
        except Exception:
            return None
    
    def query(self, **filters) -> List[Dict]:
        """
        Direct metadata query without vector search.
        
        Convenience method that goes straight to MetadataStore for structured queries.
        
        Args:
            **filters: Metadata filters (tags, status, category, project, etc.)
            
        Returns:
            List of matching documents from metadata store with tags included
        """
        if not self.metadata_store:
            logger.warning("No metadata store configured for direct queries")
            return []
        
        try:
            results = self.metadata_store.query(**filters)
            
            # Enrich results with tags for each document
            enriched_results = []
            for doc in results:
                doc_copy = doc.copy()
                
                # Add tags to the document
                doc_id = doc.get('id')
                if doc_id:
                    try:
                        # Get tags for this document
                        import sqlite3
                        conn = sqlite3.connect(self.metadata_store.db_path)
                        try:
                            tag_rows = conn.execute('''
                                SELECT t.name FROM tags t
                                JOIN document_tags dt ON t.id = dt.tag_id
                                WHERE dt.doc_id = ?
                                ORDER BY t.name
                            ''', (doc_id,)).fetchall()
                            doc_copy['tags'] = [row[0] for row in tag_rows]
                        finally:
                            conn.close()
                    except Exception as e:
                        logger.warning(f"Failed to get tags for doc {doc_id}: {e}")
                        doc_copy['tags'] = []
                else:
                    doc_copy['tags'] = []
                
                enriched_results.append(doc_copy)
            
            # Log usage if tracker available
            if self.tracker:
                for doc in enriched_results[:10]:  # Limit logging to first 10
                    try:
                        filename = doc.get('filename')
                        if filename:
                            self.tracker.log_search_hit(filename, f"metadata_query: {filters}")
                    except Exception as e:
                        logger.warning(f"Failed to log metadata query hit: {e}")
            
            return enriched_results
            
        except Exception as e:
            logger.error(f"Metadata query failed: {e}")
            return []
    
    def stats(self) -> Dict[str, Any]:
        """
        Get comprehensive unified search statistics.
        
        Returns:
            Dictionary with searcher stats and graph store info
        """
        stats = {
            'searcher_available': bool(self.searcher),
            'graph_available': bool(self.graph_store),
            'metadata_available': bool(self.metadata_store),
            'components': {
                'searcher': 'available' if self.searcher else 'missing',
                'graph_store': 'available' if self.graph_store else 'missing',
                'graph_querier': 'available' if self.graph_querier else 'missing',
                'metadata_store': 'available' if self.metadata_store else 'missing'
            }
        }
        
        # Add searcher stats if available
        if self.searcher and hasattr(self.searcher, 'store'):
            try:
                store_stats = self.searcher.store.stats()
                stats['vector_stats'] = store_stats
            except Exception as e:
                stats['vector_stats'] = {'error': str(e)}
        
        # Add graph stats if available
        if self.graph_store:
            try:
                graph_stats = self.graph_store.stats()
                stats['graph_stats'] = graph_stats
            except Exception as e:
                stats['graph_stats'] = {'error': str(e)}
        
        # Add metadata stats if available
        if self.metadata_store:
            try:
                metadata_stats = self.metadata_store.stats()
                stats['metadata_stats'] = metadata_stats
            except Exception as e:
                stats['metadata_stats'] = {'error': str(e)}
        
        return stats