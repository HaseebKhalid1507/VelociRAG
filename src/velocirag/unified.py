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
    
    def __init__(self, searcher: Searcher, graph_store: Optional[GraphStore] = None):
        """
        Initialize unified search orchestrator.
        
        Args:
            searcher: Vector searcher (required)
            graph_store: Knowledge graph (optional — degrades gracefully without it)
        """
        if not searcher:
            raise ValueError("Searcher is required")
        
        self.searcher = searcher
        self.graph_store = graph_store
        self.graph_querier = GraphQuerier(graph_store) if graph_store else None
        
        logger.info(f"UnifiedSearch initialized: vector={True}, graph={bool(graph_store)}")
    
    def search(self, query: str, limit: int = 5, threshold: float = 0.3, 
               enrich_graph: bool = True) -> Dict[str, Any]:
        """
        Unified search: vector search + graph enrichment.
        
        Pipeline:
        1. Run vector search (searcher.search)
        2. For each result, find graph connections (if graph_store provided)
        3. Add graph metadata to results (connected nodes, related notes)
        4. Return enriched results
        
        Args:
            query: Search query string
            limit: Maximum results to return
            threshold: Minimum similarity threshold
            enrich_graph: Enable graph enrichment (requires graph_store)
            
        Returns:
            Dictionary with enriched search results and metadata
        """
        start_time = time.time()
        
        # Step 1: Execute vector search
        try:
            vector_start = time.time()
            vector_result = self.searcher.search(
                query=query,
                limit=limit * 2,  # Get more for enrichment filtering
                threshold=threshold,
                include_stats=True
            )
            vector_time = (time.time() - vector_start) * 1000
            
        except Exception as e:
            raise UnifiedSearchError(f"Vector search failed: {e}")
        
        vector_results = vector_result.get('results', [])
        
        # Determine search mode
        graph_available = bool(self.graph_store and enrich_graph)
        search_mode = 'unified' if graph_available else 'vector_only'
        
        # Initialize enrichment stats
        enrichment_stats = {
            'vector_results': len(vector_results),
            'graph_enriched': 0,
            'graph_available': graph_available
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
        
        # Add vector-specific metadata
        if 'search_time_ms' in vector_result:
            response['vector_time_ms'] = vector_result['search_time_ms']
        if 'variants_used' in vector_result:
            response['variants_used'] = vector_result['variants_used']
        
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