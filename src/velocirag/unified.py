"""
Velociragtor Phase 7 Unified Search - Orchestration layer for vector + graph search.

Clean orchestration that combines vector search results with optional graph enrichment.
Designed for graceful degradation when graph components are unavailable.
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
    Unified search orchestrator combining vector search with graph enrichment.
    
    The conductor of the Velociragtor symphony — brings together vector precision
    with graph context for richer, more meaningful search results.
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
        Unified search: vector search + metadata filtering + graph enrichment.
        
        Pipeline:
        1. If filters provided, query MetadataStore for matching filenames
        2. Run vector search (searcher.search)
        3. If filters, apply RRF fusion to boost filtered results
        4. For each result, find graph connections (if graph_store provided)
        5. Add graph metadata to results (connected nodes, related notes)
        6. Log search hits via tracker (if provided)
        7. Return enriched results
        
        Args:
            query: Search query string
            limit: Maximum results to return
            threshold: Minimum similarity threshold
            enrich_graph: Enable graph enrichment (requires graph_store)
            filters: Optional metadata filters (tags, status, category, etc.)
            
        Returns:
            Dictionary with enriched search results and metadata
        """
        start_time = time.time()
        
        # Step 1: Query metadata store if filters provided
        metadata_results = []
        metadata_filenames = set()
        metadata_time_ms = 0
        
        if self.metadata_store and filters:
            try:
                metadata_start = time.time()
                metadata_docs = self.metadata_store.query(**filters, limit=limit * 3)
                metadata_time_ms = (time.time() - metadata_start) * 1000
                
                # Extract filenames for filtering/boosting
                for doc in metadata_docs:
                    metadata_filenames.add(doc.get('filename', ''))
                    metadata_results.append({
                        'filename': doc.get('filename', ''),
                        'title': doc.get('title', ''),
                        'score': 1.0  # Metadata matches get full score
                    })
                    
                logger.info(f"Metadata query found {len(metadata_docs)} matching documents")
            except Exception as e:
                logger.warning(f"Metadata query failed: {e}")
                # Continue without metadata filtering
        
        # Step 2: Execute vector search
        try:
            vector_start = time.time()
            vector_result = self.searcher.search(
                query=query,
                limit=limit * 2 if filters else limit * 2,  # Get more for fusion/enrichment
                threshold=threshold,
                include_stats=True
            )
            vector_time = (time.time() - vector_start) * 1000
            
        except Exception as e:
            raise UnifiedSearchError(f"Vector search failed: {e}")
        
        vector_results = vector_result.get('results', [])
        
        # Step 3: Apply RRF fusion if metadata filtering was used
        fusion_time_ms = 0
        if metadata_results and filters:
            try:
                fusion_start = time.time()
                
                # Convert vector results to RRF format
                vector_rrf_results = []
                for i, result in enumerate(vector_results):
                    # Try to extract filename from metadata
                    file_path = result.get('metadata', {}).get('file_path', '')
                    filename = Path(file_path).name if file_path else result.get('doc_id', '')
                    
                    vector_rrf_results.append({
                        'doc_id': result.get('doc_id', f'vec_{i}'),
                        'score': result.get('similarity', result.get('score', 0)),
                        'metadata': result.get('metadata', {}),
                        'content': result.get('content', ''),
                        '_source': 'vector',
                        '_filename': filename
                    })
                
                # Convert metadata results to RRF format
                metadata_rrf_results = []
                for i, result in enumerate(metadata_results):
                    metadata_rrf_results.append({
                        'doc_id': result.get('filename', f'meta_{i}'),
                        'score': result.get('score', 1.0),
                        'metadata': {'filename': result.get('filename')},
                        'content': '',
                        '_source': 'metadata',
                        '_filename': result.get('filename', '')
                    })
                
                # Apply RRF fusion
                fused_results_list = reciprocal_rank_fusion(
                    [vector_rrf_results, metadata_rrf_results],
                    k=60
                )
                
                # Map fused results back to original vector results
                fused_results = []
                seen_docs = set()
                
                for fused_result in fused_results_list[:limit * 2]:
                    # Extract document identification
                    fused_filename = fused_result.get('_filename', '')
                    fused_doc_id = fused_result.get('doc_id', '')
                    rrf_score = fused_result.get('metadata', {}).get('rrf_score', 0.0)
                    
                    # Find the original vector result
                    for result in vector_results:
                        result_doc_id = result.get('doc_id', '')
                        result_filename = Path(result.get('metadata', {}).get('file_path', '')).name
                        
                        match_key = result_filename or result_doc_id
                        if (match_key == fused_filename or match_key == fused_doc_id) and match_key not in seen_docs:
                            # Boost score if in metadata results
                            if result_filename in metadata_filenames:
                                result['_metadata_match'] = True
                            result['_rrf_score'] = rrf_score
                            fused_results.append(result)
                            seen_docs.add(match_key)
                            break
                
                vector_results = fused_results
                fusion_time_ms = (time.time() - fusion_start) * 1000
                
            except Exception as e:
                logger.warning(f"RRF fusion failed: {e}")
                # Continue with original vector results
        
        # Determine search mode
        graph_available = bool(self.graph_store and enrich_graph)
        metadata_available = bool(self.metadata_store and filters)
        
        if metadata_available and graph_available:
            search_mode = 'unified_full'  # All three layers
        elif metadata_available:
            search_mode = 'vector_metadata'  # Vector + metadata
        elif graph_available:
            search_mode = 'vector_graph'  # Vector + graph
        else:
            search_mode = 'vector_only'  # Vector only
        
        # Initialize enrichment stats
        enrichment_stats = {
            'vector_results': len(vector_results),
            'graph_enriched': 0,
            'graph_available': graph_available,
            'metadata_available': metadata_available,
            'metadata_matches': len(metadata_filenames) if filters else 0
        }
        
        # Step 2: Enrich results with graph connections
        enriched_results = []
        graph_errors = []
        
        for result in vector_results[:limit]:  # Only enrich up to final limit
            # Deep copy to avoid modifying original results
            enriched_result = result.copy()
            enriched_result['metadata'] = result['metadata'].copy()
            
            # Initialize graph metadata fields
            enriched_result['metadata']['graph_connections'] = []
            enriched_result['metadata']['related_notes'] = []
            enriched_result['metadata']['found_in_graph'] = False
            
            # Enrich with graph data if available
            if graph_available:
                try:
                    graph_info = self._enrich_with_graph(result)
                    if graph_info['success']:
                        enriched_result['metadata']['graph_connections'] = graph_info['connections']
                        enriched_result['metadata']['related_notes'] = graph_info['related_notes']
                        enriched_result['metadata']['found_in_graph'] = graph_info['found_in_graph']
                        enrichment_stats['graph_enriched'] += 1
                    elif graph_info.get('error'):
                        graph_errors.append(graph_info['error'])
                        
                except Exception as e:
                    graph_errors.append(f"Graph enrichment error: {e}")
                    logger.warning(f"Graph enrichment failed for result: {e}")
            
            enriched_results.append(enriched_result)
        
        # Log search hits via tracker if available
        if self.tracker:
            try:
                for result in enriched_results:
                    file_path = result.get('metadata', {}).get('file_path')
                    if file_path:
                        filename = Path(file_path).name
                        self.tracker.log_search_hit(filename, query)
            except Exception as e:
                logger.warning(f"Failed to log search hits: {e}")
        
        # Calculate timing
        total_time = time.time() - start_time
        
        # Build response
        response = {
            'results': enriched_results,
            'query': query,
            'total_results': len(enriched_results),
            'search_time_ms': round(total_time * 1000, 2),
            'search_mode': search_mode,
            'enrichment_stats': enrichment_stats
        }
        
        # Add timing metadata
        if 'search_time_ms' in vector_result:
            response['vector_time_ms'] = vector_result['search_time_ms']
        if 'variants_used' in vector_result:
            response['variants_used'] = vector_result['variants_used']
        if metadata_time_ms > 0:
            response['metadata_time_ms'] = round(metadata_time_ms, 2)
        if fusion_time_ms > 0:
            response['fusion_time_ms'] = round(fusion_time_ms, 2)
        
        # Log any graph errors (but don't fail the search)
        if graph_errors:
            unique_errors = list(set(graph_errors))
            logger.warning(f"Graph enrichment errors: {unique_errors}")
            response['graph_errors'] = unique_errors[:3]  # Limit error spam
        
        return response
    
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
        
        if not file_path:
            return None
        
        try:
            # Convert to Path for easier manipulation
            path = Path(file_path)
            
            # Get filename without extension
            title = path.stem
            
            # Clean common prefixes/suffixes if needed
            # (Add project-specific cleaning logic here)
            
            return title if title else None
            
        except Exception:
            return None
    
    def stats(self) -> Dict[str, Any]:
        """
        Get comprehensive unified search statistics.
        
        Returns:
            Dictionary with searcher stats and graph store info
        """
        stats = {
            'searcher_available': bool(self.searcher),
            'graph_available': bool(self.graph_store),
            'components': {
                'searcher': 'available' if self.searcher else 'missing',
                'graph_store': 'available' if self.graph_store else 'missing',
                'graph_querier': 'available' if self.graph_querier else 'missing'
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
        
        return stats
    
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
                        with sqlite3.connect(self.metadata_store.db_path) as conn:
                            tag_rows = conn.execute('''
                                SELECT t.name FROM tags t
                                JOIN document_tags dt ON t.id = dt.tag_id
                                WHERE dt.doc_id = ?
                                ORDER BY t.name
                            ''', (doc_id,)).fetchall()
                            doc_copy['tags'] = [row[0] for row in tag_rows]
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