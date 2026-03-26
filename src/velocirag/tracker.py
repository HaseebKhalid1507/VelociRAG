"""
Velocirag Usage Tracker.

Track document usage events and access patterns through the metadata store.
Provides analytics and insights into document usage and staleness.
"""

import logging
from typing import Dict, List, Optional

from .metadata import MetadataStore

logger = logging.getLogger("velocirag.tracker")


class UsageTracker:
    """
    Document usage tracker that logs events through MetadataStore.
    
    Tracks search hits, reads, updates and provides access analytics.
    Works as a facade over MetadataStore for usage-specific operations.
    """
    
    def __init__(self, metadata_store: MetadataStore):
        """
        Initialize usage tracker.
        
        Args:
            metadata_store: MetadataStore instance for logging events
        """
        self.metadata_store = metadata_store
        logger.info("UsageTracker initialized")
    
    def log_search_hit(self, filename: str, query: str) -> None:
        """
        Log when a document appears in search results.
        
        Args:
            filename: Document filename
            query: Search query that returned this document
        """
        doc = self.metadata_store.get_document(filename)
        if doc:
            self.metadata_store.log_usage(
                doc['id'], 
                'search_hit', 
                source=f'query: {query[:100]}'  # Truncate long queries
            )
        else:
            logger.warning(f"Cannot log search hit for unknown document: {filename}")
    
    def log_read(self, filename: str, source: str = None) -> None:
        """
        Log when a document is read/accessed.
        
        Args:
            filename: Document filename
            source: Optional source identifier (user, system, etc.)
        """
        doc = self.metadata_store.get_document(filename)
        if doc:
            self.metadata_store.log_usage(
                doc['id'], 
                'read', 
                source=source
            )
        else:
            logger.warning(f"Cannot log read for unknown document: {filename}")
    
    def log_update(self, filename: str) -> None:
        """
        Log when a document is updated/modified.
        
        Args:
            filename: Document filename
        """
        doc = self.metadata_store.get_document(filename)
        if doc:
            self.metadata_store.log_usage(
                doc['id'], 
                'update', 
                source='file_system'
            )
        else:
            logger.warning(f"Cannot log update for unknown document: {filename}")
    
    def get_access_history(self, filename: str, limit: int = 20) -> List[Dict]:
        """
        Get access history for a specific document.
        
        Args:
            filename: Document filename
            limit: Maximum number of events to return
            
        Returns:
            List of usage event dictionaries
        """
        doc = self.metadata_store.get_document(filename)
        if not doc:
            return []
        
        try:
            import sqlite3
            with sqlite3.connect(self.metadata_store.db_path) as conn:
                rows = conn.execute('''
                    SELECT action, source, timestamp FROM usage_log
                    WHERE doc_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (doc['id'], limit)).fetchall()
                
                return [
                    {
                        'action': row[0],
                        'source': row[1],
                        'timestamp': row[2]
                    }
                    for row in rows
                ]
                
        except sqlite3.Error as e:
            logger.error(f"Failed to get access history for {filename}: {e}")
            return []
    
    def get_never_accessed(self) -> List[Dict]:
        """
        Get documents with zero usage log entries.
        
        Returns:
            List of document dictionaries that have never been accessed
        """
        try:
            import sqlite3
            with sqlite3.connect(self.metadata_store.db_path) as conn:
                cursor = conn.execute('''
                    SELECT d.* FROM documents d
                    WHERE d.id NOT IN (
                        SELECT DISTINCT doc_id FROM usage_log
                    )
                    ORDER BY d.created_at DESC
                ''')
                rows = cursor.fetchall()
                
                # Convert to dictionaries
                columns = [desc[0] for desc in cursor.description]
                results = []
                
                for row in rows:
                    doc_dict = dict(zip(columns, row))
                    
                    # Parse JSON meta field
                    if doc_dict.get('meta'):
                        try:
                            import json
                            doc_dict['meta'] = json.loads(doc_dict['meta'])
                        except json.JSONDecodeError:
                            doc_dict['meta'] = {}
                    else:
                        doc_dict['meta'] = {}
                    
                    results.append(doc_dict)
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to get never accessed documents: {e}")
            return []
    
    def get_most_accessed(self, limit: int = 10) -> List[Dict]:
        """
        Get documents with the most access events.
        
        Args:
            limit: Maximum number of documents to return
            
        Returns:
            List of documents with access counts, sorted by access count descending
        """
        try:
            import sqlite3
            with sqlite3.connect(self.metadata_store.db_path) as conn:
                rows = conn.execute('''
                    SELECT 
                        d.filename,
                        d.title,
                        d.category,
                        d.project,
                        COUNT(ul.id) as access_count,
                        MAX(ul.timestamp) as last_accessed,
                        COUNT(CASE WHEN ul.action = 'search_hit' THEN 1 END) as search_hits,
                        COUNT(CASE WHEN ul.action = 'read' THEN 1 END) as reads
                    FROM documents d
                    JOIN usage_log ul ON d.id = ul.doc_id
                    GROUP BY d.id, d.filename, d.title, d.category, d.project
                    ORDER BY access_count DESC
                    LIMIT ?
                ''', (limit,)).fetchall()
                
                return [
                    {
                        'filename': row[0],
                        'title': row[1],
                        'category': row[2],
                        'project': row[3],
                        'access_count': row[4],
                        'last_accessed': row[5],
                        'search_hits': row[6],
                        'reads': row[7]
                    }
                    for row in rows
                ]
                
        except Exception as e:
            logger.error(f"Failed to get most accessed documents: {e}")
            return []
    
    def get_usage_trends(self, days: int = 30) -> Dict:
        """
        Get usage trends over the specified time period.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary with trend statistics
        """
        try:
            import sqlite3
            from datetime import datetime, timedelta
            
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            with sqlite3.connect(self.metadata_store.db_path) as conn:
                # Total events in period
                total_events = conn.execute('''
                    SELECT COUNT(*) FROM usage_log WHERE timestamp >= ?
                ''', (cutoff_date,)).fetchone()[0]
                
                # Events by action type
                action_counts = {}
                action_rows = conn.execute('''
                    SELECT action, COUNT(*) FROM usage_log 
                    WHERE timestamp >= ?
                    GROUP BY action
                    ORDER BY COUNT(*) DESC
                ''', (cutoff_date,)).fetchall()
                
                for action, count in action_rows:
                    action_counts[action] = count
                
                # Daily activity
                daily_activity = {}
                daily_rows = conn.execute('''
                    SELECT DATE(timestamp) as day, COUNT(*) as events 
                    FROM usage_log 
                    WHERE timestamp >= ?
                    GROUP BY DATE(timestamp)
                    ORDER BY day
                ''', (cutoff_date,)).fetchall()
                
                for day, events in daily_rows:
                    daily_activity[day] = events
                
                # Most active documents in period
                active_docs = conn.execute('''
                    SELECT d.filename, d.title, COUNT(*) as events
                    FROM usage_log ul
                    JOIN documents d ON ul.doc_id = d.id
                    WHERE ul.timestamp >= ?
                    GROUP BY d.id, d.filename, d.title
                    ORDER BY events DESC
                    LIMIT 5
                ''', (cutoff_date,)).fetchall()
                
                top_docs = [
                    {'filename': row[0], 'title': row[1], 'events': row[2]}
                    for row in active_docs
                ]
                
                return {
                    'period_days': days,
                    'total_events': total_events,
                    'events_by_action': action_counts,
                    'daily_activity': daily_activity,
                    'top_documents': top_docs,
                    'avg_events_per_day': round(total_events / days, 2) if days > 0 else 0
                }
                
        except Exception as e:
            logger.error(f"Failed to get usage trends: {e}")
            return {
                'period_days': days,
                'total_events': 0,
                'events_by_action': {},
                'daily_activity': {},
                'top_documents': [],
                'avg_events_per_day': 0
            }