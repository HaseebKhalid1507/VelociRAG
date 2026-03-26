"""
Knowledge graph analyzers for Velocirag.

All 6 analyzers in one clean file. Extract relationships from markdown content
using explicit links, entity extraction, temporal analysis, topic clustering,
semantic similarity, and centrality scoring.
"""

import os
import re
import logging
from collections import defaultdict, Counter, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import hashlib

# Conditional imports
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from .graph import Node, Edge, NodeType, RelationType
from .embedder import Embedder

logger = logging.getLogger("velocirag.analyzers")


class ExplicitAnalyzer:
    """Parse [[wiki-links]] and #tags from markdown."""
    
    def analyze(self, nodes: List[Node]) -> Tuple[List[Node], List[Edge]]:
        """Extract explicit relationships from markdown content."""
        new_nodes = []
        new_edges = []
        all_tags = set()
        
        # First pass: collect all tags
        for node in nodes:
            if node.type == NodeType.NOTE and node.content:
                tags = self._extract_tags(node.content)
                all_tags.update(tags)
        
        # Create tag nodes
        tag_nodes = {}
        for tag in all_tags:
            tag_id = f"tag_{tag.lower()}"
            tag_node = Node(
                id=tag_id,
                type=NodeType.TAG,
                title=f"#{tag}",
                metadata={'tag_name': tag}
            )
            new_nodes.append(tag_node)
            tag_nodes[tag] = tag_id
        
        # Second pass: create relationships
        for node in nodes:
            if node.type != NodeType.NOTE or not node.content:
                continue
            
            # Wiki-link relationships
            wikilinks = self._extract_wikilinks(node.content)
            for link in wikilinks:
                # Find target node by title
                target_node = self._find_node_by_title(nodes + new_nodes, link)
                if target_node:
                    edge_id = f"ref_{node.id}_{target_node.id}"
                    edge = Edge(
                        id=edge_id,
                        source_id=node.id,
                        target_id=target_node.id,
                        type=RelationType.REFERENCES,
                        weight=0.8,
                        confidence=0.9,
                        metadata={'link_text': link}
                    )
                    new_edges.append(edge)
            
            # Tag relationships
            tags = self._extract_tags(node.content)
            for tag in tags:
                if tag in tag_nodes:
                    edge_id = f"tagged_{node.id}_{tag_nodes[tag]}"
                    edge = Edge(
                        id=edge_id,
                        source_id=node.id,
                        target_id=tag_nodes[tag],
                        type=RelationType.TAGGED_AS,
                        weight=0.7,
                        confidence=1.0,
                        metadata={'tag_name': tag}
                    )
                    new_edges.append(edge)
        
        logger.info(f"ExplicitAnalyzer: {len(new_nodes)} nodes, {len(new_edges)} edges")
        return new_nodes, new_edges
    
    def _extract_wikilinks(self, content: str) -> List[str]:
        """Extract [[wiki-links]] from content."""
        links = re.findall(r'\[\[([^\]]+)\]\]', content)
        clean_links = []
        for link in links:
            # Remove display text if present (e.g., [[Link|Display]])
            if '|' in link:
                link = link.split('|')[0]
            clean_links.append(link.strip())
        return clean_links
    
    def _extract_tags(self, content: str) -> List[str]:
        """Extract #tags from content."""
        tags = re.findall(r'#([a-zA-Z0-9_-]+)', content)
        return [tag.lower() for tag in tags]
    
    def _find_node_by_title(self, nodes: List[Node], title: str) -> Optional[Node]:
        """Find node by title (case-insensitive partial match)."""
        title_lower = title.lower()
        for node in nodes:
            if title_lower in node.title.lower() or node.title.lower() in title_lower:
                return node
        return None


class TemporalAnalyzer:
    """Find time-based relationships between dated documents."""
    
    def __init__(self, window_days: int = 7):
        self.window_days = window_days
    
    def analyze(self, nodes: List[Node]) -> Tuple[List[Node], List[Edge]]:
        """Extract temporal relationships."""
        new_nodes = []
        new_edges = []
        
        # Group nodes by date
        dated_nodes = self._extract_dates_from_nodes(nodes)
        if len(dated_nodes) < 2:
            return new_nodes, new_edges
        
        # Sort by date
        sorted_nodes = sorted(dated_nodes, key=lambda x: x['date'])
        
        # Create temporal edges
        for i, current in enumerate(sorted_nodes):
            for j, other in enumerate(sorted_nodes[i+1:], i+1):
                time_diff = abs((other['date'] - current['date']).days)
                
                if time_diff <= self.window_days:
                    # Concurrent relationship
                    edge_id = f"temporal_{current['node'].id}_{other['node'].id}"
                    weight = max(0.3, 1.0 - (time_diff / self.window_days) * 0.5)
                    
                    edge = Edge(
                        id=edge_id,
                        source_id=current['node'].id,
                        target_id=other['node'].id,
                        type=RelationType.TEMPORAL,  # Temporal proximity, NOT semantic similarity
                        weight=weight,
                        confidence=0.7,
                        metadata={
                            'temporal_type': 'concurrent',
                            'days_apart': time_diff,
                            'source_date': current['date'].isoformat(),
                            'target_date': other['date'].isoformat()
                        }
                    )
                    new_edges.append(edge)
        
        logger.info(f"TemporalAnalyzer: {len(new_edges)} temporal edges")
        return new_nodes, new_edges
    
    def _extract_dates_from_nodes(self, nodes: List[Node]) -> List[Dict]:
        """Extract dates from node metadata or content."""
        dated_nodes = []
        
        for node in nodes:
            if node.type != NodeType.NOTE:
                continue
            
            # Try to get date from metadata
            node_date = None
            if node.metadata:
                # Check for various date fields
                for date_field in ['created_time', 'modified_time', 'file_date', 'date']:
                    if date_field in node.metadata:
                        try:
                            if isinstance(node.metadata[date_field], str):
                                node_date = datetime.fromisoformat(node.metadata[date_field].replace('Z', '+00:00'))
                            else:
                                node_date = node.metadata[date_field]
                            break
                        except (ValueError, TypeError):
                            continue
            
            # Fallback to created_at
            if not node_date and node.created_at:
                node_date = node.created_at
            
            # Try to extract date from filename
            if not node_date and 'filename' in (node.metadata or {}):
                filename = node.metadata['filename']
                date_match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
                if date_match:
                    try:
                        node_date = datetime.strptime(date_match.group(1), '%Y-%m-%d')
                    except ValueError:
                        pass
            
            if node_date:
                dated_nodes.append({
                    'node': node,
                    'date': node_date
                })
        
        return dated_nodes


class EntityAnalyzer:
    """Extract entities (people, projects, concepts) from content."""
    
    def __init__(self, min_frequency: int = 2):
        self.min_frequency = min_frequency
        
        # Simple entity patterns
        self.entity_patterns = {
            'person': [
                r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # FirstName LastName
                r'\b[A-Z]\. [A-Z][a-z]+\b',      # J. Smith
            ],
            'project': [
                r'\b[A-Z][a-zA-Z]*(?:Project|System|Engine|Tool|App)\b',
                r'\b(?:Project|System) [A-Z][a-zA-Z]+\b',
            ],
            'technology': [
                r'\b(?:Python|JavaScript|React|Vue|Angular|Node\.js|Django|Flask|Docker|Kubernetes|AWS|Azure|GCP)\b',
            ],
            'concept': [
                r'\b[A-Z][a-zA-Z]{2,}\b',  # General capitalized words
            ]
        }
    
    def analyze(self, nodes: List[Node]) -> Tuple[List[Node], List[Edge]]:
        """Extract entities and relationships."""
        new_nodes = []
        new_edges = []
        
        # Extract entities from all notes
        all_entities = self._extract_all_entities(nodes)
        
        # Filter by frequency
        entity_counts = Counter(entity for entities in all_entities.values() for entity in entities)
        frequent_entities = {entity for entity, count in entity_counts.items() 
                           if count >= self.min_frequency}
        
        # Create entity nodes
        entity_node_map = {}
        for entity in frequent_entities:
            entity_type = self._classify_entity(entity)
            entity_id = f"entity_{hashlib.md5(entity.lower().encode()).hexdigest()[:8]}"
            
            entity_node = Node(
                id=entity_id,
                type=NodeType.ENTITY,
                title=entity,
                metadata={
                    'entity_type': entity_type,
                    'frequency': entity_counts[entity]
                }
            )
            new_nodes.append(entity_node)
            entity_node_map[entity] = entity_id
        
        # Create entity-note relationships
        for node in nodes:
            if node.type != NodeType.NOTE or node.id not in all_entities:
                continue
            
            for entity in all_entities[node.id]:
                if entity in entity_node_map:
                    edge_id = f"mentions_{node.id}_{entity_node_map[entity]}"
                    edge = Edge(
                        id=edge_id,
                        source_id=node.id,
                        target_id=entity_node_map[entity],
                        type=RelationType.MENTIONS,
                        weight=0.6,
                        confidence=0.8,
                        metadata={'entity_name': entity}
                    )
                    new_edges.append(edge)
        
        logger.info(f"EntityAnalyzer: {len(new_nodes)} entities, {len(new_edges)} mention edges")
        return new_nodes, new_edges
    
    def _extract_all_entities(self, nodes: List[Node]) -> Dict[str, Set[str]]:
        """Extract entities from all note content."""
        all_entities = {}
        
        for node in nodes:
            if node.type != NodeType.NOTE or not node.content:
                continue
            
            entities = set()
            
            # Apply entity patterns
            for entity_type, patterns in self.entity_patterns.items():
                for pattern in patterns:
                    matches = re.findall(pattern, node.content)
                    entities.update(match.strip() for match in matches if len(match.strip()) > 2)
            
            # Extract from wikilinks (often entities)
            wikilinks = re.findall(r'\[\[([^\]]+)\]\]', node.content)
            for link in wikilinks:
                if '|' in link:
                    link = link.split('|')[0]
                entities.add(link.strip())
            
            # Filter out common words
            entities = {e for e in entities if not self._is_common_word(e)}
            all_entities[node.id] = entities
        
        return all_entities
    
    def _classify_entity(self, entity: str) -> str:
        """Classify entity type."""
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                if re.match(pattern, entity):
                    return entity_type
        return 'concept'
    
    def _is_common_word(self, word: str) -> bool:
        """Check if word is too common to be an entity."""
        common_words = {
            'The', 'This', 'That', 'These', 'Those', 'And', 'But', 'Or', 'So',
            'Then', 'Now', 'Here', 'There', 'When', 'Where', 'Why', 'How',
            'What', 'Which', 'Who', 'Whom', 'Whose', 'Will', 'Would', 'Could',
            'Should', 'Might', 'May', 'Can', 'Must', 'Shall'
        }
        return word in common_words


class GLiNERAnalyzer:
    """Extract typed entities using GLiNER encoder model. Zero hallucination — can only tag spans present in text."""
    
    DEFAULT_LABELS = ['person', 'technology', 'organization', 'concept', 'project', 'course']
    
    def __init__(self, labels: list = None, model_name: str = 'urchade/gliner_small-v2.1',
                 threshold: float = 0.35, min_frequency: int = 2):
        self.labels = labels or self.DEFAULT_LABELS
        self.threshold = threshold
        self.min_frequency = min_frequency
        self.model_name = model_name
        self._model = None  # Lazy load
    
    def _load_model(self):
        """Lazy-load GLiNER model."""
        if self._model is None:
            try:
                from gliner import GLiNER
                self._model = GLiNER.from_pretrained(self.model_name)
                logger.info(f"GLiNER model loaded: {self.model_name}")
            except ImportError:
                raise ImportError(
                    "GLiNER not installed. Install with: pip install velocirag[ner]"
                )
        return self._model
    
    @staticmethod
    def _chunk_for_gliner(text: str, max_chars: int = 1500, overlap: int = 200) -> List[str]:
        """Chunk text for GLiNER processing. Model max is ~512 tokens (~1500 chars)."""
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + max_chars
            chunks.append(text[start:end])
            start = end - overlap  # Overlap to catch entities at boundaries
        return chunks
    
    def analyze(self, nodes: List[Node]) -> Tuple[List[Node], List[Edge]]:
        """Extract entities using GLiNER and create graph nodes + edges."""
        model = self._load_model()
        
        new_nodes = []
        new_edges = []
        
        # Extract entities from all notes
        all_entities: Dict[str, List[Dict]] = {}  # node_id -> list of {text, label, score}
        entity_counts = Counter()
        
        for node in nodes:
            if node.type != NodeType.NOTE or not node.content or not node.content.strip():
                continue
            
            # GLiNER max sequence ~512 tokens. ~1500 chars is safe.
            # Chunk longer texts with overlap to catch entities at boundaries.
            chunks = self._chunk_for_gliner(node.content)
            
            try:
                node_entities = []
                seen_spans = set()  # dedupe across chunks
                
                for chunk in chunks:
                    entities = model.predict_entities(chunk, self.labels, threshold=self.threshold)
                    for e in entities:
                        entity_text = e['text'].strip()
                        if len(entity_text) < 2:
                            continue
                        key = (entity_text.lower(), e['label'])
                        if key not in seen_spans:
                            seen_spans.add(key)
                            node_entities.append({
                                'text': entity_text,
                                'label': e['label'],
                                'score': e['score']
                            })
                            entity_counts[entity_text.lower()] += 1
                
                all_entities[node.id] = node_entities
            except Exception as ex:
                logger.warning(f"GLiNER extraction failed for {node.id}: {ex}")
                continue
        
        # Filter by frequency, keep best label/score per entity
        entity_info: Dict[str, Dict] = {}  # lowercase -> {text, label, score, count}
        for node_id, entities in all_entities.items():
            for e in entities:
                key = e['text'].lower()
                if entity_counts[key] < self.min_frequency:
                    continue
                if key not in entity_info or e['score'] > entity_info[key]['score']:
                    entity_info[key] = {
                        'text': e['text'],
                        'label': e['label'],
                        'score': e['score'],
                        'count': entity_counts[key]
                    }
        
        # Create entity nodes
        entity_node_map = {}  # lowercase entity text -> node id
        for key, info in entity_info.items():
            entity_id = f"entity_{hashlib.md5(key.encode()).hexdigest()[:8]}"
            entity_node = Node(
                id=entity_id,
                type=NodeType.ENTITY,
                title=info['text'],
                metadata={
                    'entity_type': info['label'],
                    'gliner_score': round(info['score'], 3),
                    'frequency': info['count'],
                    'extractor': 'gliner'
                }
            )
            new_nodes.append(entity_node)
            entity_node_map[key] = entity_id
        
        # Create mention edges (note → entity)
        for node in nodes:
            if node.id not in all_entities:
                continue
            
            seen = set()  # avoid duplicate edges per note
            for e in all_entities[node.id]:
                key = e['text'].lower()
                if key in entity_node_map and key not in seen:
                    seen.add(key)
                    edge_id = f"mentions_{node.id}_{entity_node_map[key]}"
                    edge = Edge(
                        id=edge_id,
                        source_id=node.id,
                        target_id=entity_node_map[key],
                        type=RelationType.MENTIONS,
                        weight=min(e['score'], 0.9),  # Score-weighted edges
                        confidence=e['score'],
                        metadata={
                            'entity_name': e['text'],
                            'entity_type': e['label'],
                            'extractor': 'gliner'
                        }
                    )
                    new_edges.append(edge)
        
        logger.info(f"GLiNERAnalyzer: {len(new_nodes)} entities, {len(new_edges)} mention edges")
        return new_nodes, new_edges


class RelationAnalyzer:
    """Extract semantic relations between entities using GLiNER multitask model.
    
    Two-pass approach (from JR's architecture):
    1. GLiNERAnalyzer already extracted entity positions
    2. This analyzer finds relation SPANS using relation-type labels
    3. Matches spans to nearby entities to create typed edges
    
    O(r) complexity — scans for relation spans once, not entity pairs.
    """
    
    RELATION_LABELS = ['uses', 'built by', 'enables', 'evolved from', 'part of', 'requires', 'influences']
    
    # Map GLiNER labels → RelationType enum values
    LABEL_TO_TYPE = {
        'uses': RelationType.USES,
        'built by': RelationType.CREATED_BY,
        'enables': RelationType.ENABLES,
        'evolved from': RelationType.EVOLVED_FROM,
        'part of': RelationType.PART_OF,
        'requires': RelationType.REQUIRES,
        'influences': RelationType.INFLUENCES,
    }
    
    def __init__(self, relation_labels: list = None, 
                 model_name: str = 'knowledgator/gliner-multitask-large-v0.5',
                 threshold: float = 0.4):
        self.relation_labels = relation_labels or self.RELATION_LABELS
        self.model_name = model_name
        self.threshold = threshold
        self._model = None
    
    def _load_model(self):
        """Lazy-load GLiNER multitask model."""
        if self._model is None:
            try:
                from gliner import GLiNER
                self._model = GLiNER.from_pretrained(self.model_name)
                logger.info(f"GLiNER multitask model loaded: {self.model_name}")
            except ImportError:
                raise ImportError("GLiNER not installed. pip install velocirag[ner]")
        return self._model
    
    def analyze(self, nodes: List[Node], entity_edges: List[Edge]) -> List[Edge]:
        """Extract relations between entities.
        
        Args:
            nodes: Note nodes (need their content)
            entity_edges: MENTIONS edges from GLiNERAnalyzer (maps notes → entities)
        
        Returns:
            New edges representing entity-to-entity relations
        """
        model = self._load_model()
        new_edges = []
        
        # Build lookup: note_id → list of (entity_id, entity_name, position)
        # We need entity positions, but we don't have them from edges alone.
        # So we re-extract entities from text to get positions.
        note_entities = {}  # note_id → [(entity_name, entity_id)]
        for edge in entity_edges:
            if edge.type == RelationType.MENTIONS:
                note_id = edge.source_id
                entity_id = edge.target_id
                entity_name = edge.metadata.get('entity_name', '')
                if note_id not in note_entities:
                    note_entities[note_id] = []
                note_entities[note_id].append((entity_name, entity_id))
        
        for node in nodes:
            if node.type != NodeType.NOTE or not node.content:
                continue
            if node.id not in note_entities:
                continue
            
            entities_in_note = note_entities[node.id]
            if len(entities_in_note) < 2:
                continue  # Need at least 2 entities for a relation
            
            # Chunk text for GLiNER (max ~1500 chars)
            chunks = GLiNERAnalyzer._chunk_for_gliner(node.content)
            
            for chunk in chunks:
                # Find entity positions in this chunk
                entity_positions = []
                for ent_name, ent_id in entities_in_note:
                    # Find all occurrences of this entity in the chunk
                    start = 0
                    while True:
                        idx = chunk.lower().find(ent_name.lower(), start)
                        if idx == -1:
                            break
                        entity_positions.append({
                            'name': ent_name,
                            'id': ent_id,
                            'start': idx,
                            'end': idx + len(ent_name)
                        })
                        start = idx + 1
                
                if len(entity_positions) < 2:
                    continue
                
                # Run relation extraction
                try:
                    rel_spans = model.predict_entities(chunk, self.relation_labels, threshold=self.threshold)
                except Exception as e:
                    logger.warning(f"Relation extraction failed for {node.id}: {e}")
                    continue
                
                # Match relation spans to entity pairs
                for rel in rel_spans:
                    rel_type = self.LABEL_TO_TYPE.get(rel['label'])
                    if not rel_type:
                        continue
                    
                    # Find which entity overlaps with this relation span
                    obj_entity = None
                    for ep in entity_positions:
                        # Check overlap
                        if (ep['start'] <= rel['start'] < ep['end'] or 
                            rel['start'] <= ep['start'] < rel['end']):
                            obj_entity = ep
                            break
                        # Exact text match
                        if ep['name'].lower() == rel['text'].lower():
                            obj_entity = ep
                            break
                    
                    if not obj_entity:
                        continue
                    
                    # Find nearest different entity (subject)
                    best_subj = None
                    best_dist = float('inf')
                    for ep in entity_positions:
                        if ep['id'] == obj_entity['id']:
                            continue
                        dist = abs(ep['start'] - rel['start'])
                        if dist < best_dist and dist < 200:  # Max 200 chars apart
                            best_dist = dist
                            best_subj = ep
                    
                    if not best_subj:
                        continue
                    
                    # Create relation edge
                    edge_id = f"rel_{best_subj['id']}_{rel_type.value}_{obj_entity['id']}"
                    edge = Edge(
                        id=edge_id,
                        source_id=best_subj['id'],
                        target_id=obj_entity['id'],
                        type=rel_type,
                        weight=min(rel['score'], 0.9),
                        confidence=rel['score'],
                        metadata={
                            'relation_label': rel['label'],
                            'source_text': rel['text'],
                            'source_note': node.id,
                            'extractor': 'gliner_multitask'
                        }
                    )
                    new_edges.append(edge)
        
        # Deduplicate edges (same source→target→type, keep highest confidence)
        edge_map = {}
        for edge in new_edges:
            key = (edge.source_id, edge.target_id, edge.type)
            if key not in edge_map or edge.confidence > edge_map[key].confidence:
                edge_map[key] = edge
        
        new_edges = list(edge_map.values())
        logger.info(f"RelationAnalyzer: {len(new_edges)} relation edges")
        return new_edges


class TopicAnalyzer:
    """Cluster documents by shared topics using TF-IDF."""
    
    def __init__(self, n_topics: int = 10):
        self.n_topics = n_topics
    
    def analyze(self, nodes: List[Node]) -> Tuple[List[Node], List[Edge]]:
        """Extract topic clusters."""
        new_nodes = []
        new_edges = []
        
        # Get note content
        note_nodes = [node for node in nodes if node.type == NodeType.NOTE and node.content]
        if len(note_nodes) < 3:
            logger.info("TopicAnalyzer: Too few notes for topic analysis")
            return new_nodes, new_edges
        
        # Use sklearn if available, otherwise fallback
        if SKLEARN_AVAILABLE:
            topics = self._sklearn_topic_analysis(note_nodes)
        else:
            topics = self._simple_topic_analysis(note_nodes)
        
        # Create topic nodes and edges
        for topic_name, note_ids in topics.items():
            if len(note_ids) >= 2:  # Only create topic if it has multiple notes
                topic_id = f"topic_{hashlib.md5(topic_name.encode()).hexdigest()[:8]}"
                topic_node = Node(
                    id=topic_id,
                    type=NodeType.TOPIC,
                    title=topic_name,
                    metadata={
                        'note_count': len(note_ids),
                        'topic_type': 'cluster'
                    }
                )
                new_nodes.append(topic_node)
                
                # Connect notes to topic
                for note_id in note_ids:
                    edge_id = f"discusses_{note_id}_{topic_id}"
                    edge = Edge(
                        id=edge_id,
                        source_id=note_id,
                        target_id=topic_id,
                        type=RelationType.DISCUSSES,
                        weight=0.7,
                        confidence=0.6,
                        metadata={'topic_name': topic_name}
                    )
                    new_edges.append(edge)
        
        logger.info(f"TopicAnalyzer: {len(new_nodes)} topics, {len(new_edges)} discussion edges")
        return new_nodes, new_edges
    
    def _sklearn_topic_analysis(self, note_nodes: List[Node]) -> Dict[str, List[str]]:
        """TF-IDF + KMeans clustering."""
        documents = [node.content for node in note_nodes]
        node_ids = [node.id for node in note_nodes]
        
        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(documents)
            
            # K-means clustering
            n_clusters = min(self.n_topics, len(note_nodes) // 2)
            if n_clusters < 2:
                return {}
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            
            # Get feature names for topic naming
            feature_names = vectorizer.get_feature_names_out()
            
            # Create topic clusters
            topics = {}
            for cluster_id in range(n_clusters):
                # Get top terms for this cluster
                cluster_center = kmeans.cluster_centers_[cluster_id]
                top_indices = cluster_center.argsort()[-3:][::-1]
                top_terms = [feature_names[i] for i in top_indices]
                topic_name = f"Topic: {' '.join(top_terms)}"
                
                # Get notes in this cluster
                cluster_note_ids = [node_ids[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
                if cluster_note_ids:
                    topics[topic_name] = cluster_note_ids
            
            return topics
            
        except Exception as e:
            logger.warning(f"sklearn topic analysis failed: {e}")
            return self._simple_topic_analysis(note_nodes)
    
    def _simple_topic_analysis(self, note_nodes: List[Node]) -> Dict[str, List[str]]:
        """Fallback: word frequency-based clustering."""
        # Extract significant words from all notes
        all_words = Counter()
        note_words = {}
        
        for node in note_nodes:
            words = re.findall(r'\b[a-zA-Z]{4,}\b', node.content.lower())
            words = [w for w in words if w not in {'that', 'this', 'with', 'have', 'will', 'from', 'they', 'been', 'were'}]
            note_words[node.id] = Counter(words)
            all_words.update(words)
        
        # Get most common words as potential topics
        common_words = [word for word, count in all_words.most_common(20) if count >= 2]
        
        # Group notes by shared significant words
        topics = {}
        for word in common_words:
            related_notes = []
            for note_id, word_counts in note_words.items():
                if word_counts.get(word, 0) >= 2:  # Word appears at least twice in note
                    related_notes.append(note_id)
            
            if len(related_notes) >= 2:
                topics[f"Topic: {word}"] = related_notes
        
        return topics


class SemanticAnalyzer:
    """Create similarity edges using embedding cosine similarity."""
    
    def __init__(self, embedder: Embedder, threshold: float = 0.7, max_edges_per_node: int = 20):
        self.embedder = embedder
        self.threshold = threshold
        self.max_edges_per_node = max_edges_per_node
    
    def analyze(self, nodes: List[Node]) -> Tuple[List[Node], List[Edge]]:
        """Find semantic similarity relationships."""
        new_nodes = []
        new_edges = []
        
        if not NUMPY_AVAILABLE:
            logger.warning("NumPy not available, skipping semantic analysis")
            return new_nodes, new_edges
        
        # Get note nodes with content
        note_nodes = [node for node in nodes if node.type == NodeType.NOTE and node.content]
        if len(note_nodes) < 2:
            return new_nodes, new_edges
        
        # Generate embeddings
        embeddings = {}
        for node in note_nodes:
            try:
                embedding = self.embedder.embed(node.content)
                if embedding is not None:
                    embeddings[node.id] = np.array(embedding)
            except Exception as e:
                logger.warning(f"Failed to embed {node.id}: {e}")
        
        if len(embeddings) < 2:
            logger.warning("Not enough embeddings for similarity analysis")
            return new_nodes, new_edges
        
        # Calculate pairwise similarities with per-node edge limits
        node_ids = list(embeddings.keys())
        node_edge_counts = defaultdict(int)
        similarity_pairs = []
        
        # First pass: calculate all similarities and sort by strength
        for i, source_id in enumerate(node_ids):
            for j, target_id in enumerate(node_ids):
                if i >= j:  # Skip self and duplicates
                    continue
                
                similarity = self._cosine_similarity(embeddings[source_id], embeddings[target_id])
                
                if similarity >= self.threshold:
                    similarity_pairs.append((similarity, source_id, target_id))
        
        # Sort by similarity (strongest first) to prioritize best connections
        similarity_pairs.sort(reverse=True)
        
        # Second pass: create edges respecting per-node limits
        for similarity, source_id, target_id in similarity_pairs:
            # Check if either node has reached its edge limit
            if (node_edge_counts[source_id] >= self.max_edges_per_node or 
                node_edge_counts[target_id] >= self.max_edges_per_node):
                continue
            
            edge_id = f"semantic_{source_id}_{target_id}"
            edge = Edge(
                id=edge_id,
                source_id=source_id,
                target_id=target_id,
                type=RelationType.SIMILAR_TO,
                weight=similarity,
                confidence=similarity,
                metadata={
                    'similarity_score': float(similarity),
                    'analysis_method': 'cosine_similarity'
                }
            )
            new_edges.append(edge)
            
            # Update edge counts for both nodes
            node_edge_counts[source_id] += 1
            node_edge_counts[target_id] += 1
        
        logger.info(f"SemanticAnalyzer: {len(new_edges)} similarity edges")
        return new_nodes, new_edges
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class CentralityAnalyzer:
    """Calculate node importance scores."""
    
    def analyze(self, nodes: List[Node], edges: List[Edge]) -> Dict[str, float]:
        """Calculate importance scores for all nodes."""
        if not nodes or not edges:
            return {}
        
        # Build adjacency graph
        graph = defaultdict(set)
        node_ids = {node.id for node in nodes}
        
        for edge in edges:
            if edge.source_id in node_ids and edge.target_id in node_ids:
                graph[edge.source_id].add(edge.target_id)
                graph[edge.target_id].add(edge.source_id)
        
        # Calculate degree centrality
        degree_scores = {}
        for node_id in node_ids:
            degree_scores[node_id] = len(graph[node_id])
        
        # Calculate betweenness centrality (simplified)
        betweenness_scores = self._simple_betweenness(graph, node_ids)
        
        # Combine scores
        max_degree = max(degree_scores.values()) if degree_scores else 1
        max_betweenness = max(betweenness_scores.values()) if betweenness_scores else 1
        
        importance_scores = {}
        for node_id in node_ids:
            degree_norm = degree_scores[node_id] / max_degree if max_degree > 0 else 0
            betweenness_norm = betweenness_scores[node_id] / max_betweenness if max_betweenness > 0 else 0
            
            # Weighted combination
            importance_scores[node_id] = 0.7 * degree_norm + 0.3 * betweenness_norm
        
        logger.info(f"CentralityAnalyzer: Calculated importance for {len(importance_scores)} nodes")
        return importance_scores
    
    def _simple_betweenness(self, graph: Dict, node_ids: Set[str]) -> Dict[str, float]:
        """Simplified betweenness centrality calculation."""
        betweenness = {node_id: 0.0 for node_id in node_ids}
        
        # For each pair of nodes, find shortest paths
        for source in node_ids:
            # BFS from source
            paths = self._bfs_paths(graph, source)
            
            # Count paths through each node
            for target, path in paths.items():
                if len(path) > 2:  # Path has intermediate nodes
                    for intermediate in path[1:-1]:
                        betweenness[intermediate] += 1.0
        
        return betweenness
    
    def _bfs_paths(self, graph: Dict, start: str) -> Dict[str, List[str]]:
        """BFS to find shortest paths from start node."""
        paths = {start: [start]}
        queue = deque([start])
        visited = {start}
        
        while queue:
            current = queue.popleft()
            
            for neighbor in graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    paths[neighbor] = paths[current] + [neighbor]
                    queue.append(neighbor)
        
        return paths