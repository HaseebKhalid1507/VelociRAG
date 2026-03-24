"""
Centroid-based extractive summarization for L0/L1 abstracts.

Identifies the most representative sentences in documents using
the existing MiniLM embedder. No LLM, no GPU, no new models.
"""

import logging
import re
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .embedder import Embedder

# Constants
DEFAULT_L0_SENTENCES = 1
DEFAULT_L1_SENTENCES = 3
MIN_SENTENCE_LENGTH = 10
MIN_SENTENCES_FOR_EXTRACTION = 3

# Setup logging
logger = logging.getLogger("velocirag.abstracts")


@dataclass
class AbstractResult:
    """Container for abstract generation results."""
    l0_abstract: str          # Single most representative sentence
    l1_overview: str          # 3-5 sentences preserving original order
    l0_embedding: np.ndarray  # Embedding of L0 abstract
    l1_embedding: np.ndarray  # Embedding of L1 overview
    original_sentences: int   # Count of sentences in source
    generation_time_ms: float # Performance tracking


class SentenceSplitter:
    """Robust regex-based sentence segmentation with abbreviation handling."""
    
    # Common abbreviations that shouldn't trigger sentence breaks
    ABBREVIATIONS = {
        'Mr.', 'Mrs.', 'Ms.', 'Dr.', 'Prof.', 'Sr.', 'Jr.',
        'vs.', 'etc.', 'e.g.', 'i.e.', 'Fig.', 'Vol.',
        'No.', 'pp.', 'Ph.D.', 'M.D.', 'B.A.', 'M.A.',
        'U.S.', 'U.K.', 'N.Y.', 'L.A.', 'St.', 'Ave.',
        'Inc.', 'Ltd.', 'Corp.', 'Co.'
    }
    
    def __init__(self):
        # Pattern for sentence endings followed by whitespace + capital letter
        self.sentence_pattern = re.compile(r'([.!?]+)\s+([A-Z])')
        # Pattern for markdown headers
        self.header_pattern = re.compile(r'^#{1,6}\s+(.+)$', re.MULTILINE)
        # Pattern for abbreviations
        self.abbrev_pattern = re.compile(r'\b(' + '|'.join(
            re.escape(abbr.rstrip('.')) for abbr in self.ABBREVIATIONS
        ) + r')\.')
    
    def split(self, text: str) -> List[str]:
        """
        Split text into sentences, handling abbreviations, URLs, numbers.
        
        Args:
            text: Input text to split
            
        Returns:
            List of cleaned sentences
        """
        if not text or not text.strip():
            return []
        
        # Clean up the text - order matters!
        
        # First replace markdown headers with sentence breaks
        text = re.sub(r'\n#{1,6}\s*([^\n]+)', r'. \1', text)
        
        # Replace double newlines with sentence breaks
        text = re.sub(r'\n\n+', '. ', text)
        
        # Replace single newlines with spaces
        text = re.sub(r'\n', ' ', text)
        
        # Finally, replace multiple whitespace with single spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Clean up formatting issues
        text = re.sub(r'\.+', '.', text)  # Multiple periods to single
        text = re.sub(r'\s+\.', '.', text)  # Space before period
        
        # Temporarily replace abbreviations to prevent false splits
        abbrev_replacements = {}
        abbrev_counter = 0
        
        # Find and replace abbreviations
        for abbrev in self.ABBREVIATIONS:
            if abbrev in text:
                placeholder = f"__ABBREV_{abbrev_counter}__"
                text = text.replace(abbrev, placeholder)
                abbrev_replacements[placeholder] = abbrev
                abbrev_counter += 1
        
        # Split by sentence patterns
        sentences = self._split_by_punctuation(text)
        
        # Restore abbreviations and clean up
        final_sentences = []
        for sentence in sentences:
            # Restore abbreviations
            for placeholder, original in abbrev_replacements.items():
                sentence = sentence.replace(placeholder, original)
            
            # Clean and validate
            sentence = sentence.strip()
            if self._is_valid_sentence(sentence):
                # Remove markdown formatting
                sentence = self._clean_markdown(sentence)
                final_sentences.append(sentence)
        
        return final_sentences
    
    def _split_by_punctuation(self, text: str) -> List[str]:
        """Split text by punctuation marks."""
        # Use regex to find sentence boundaries
        # Pattern: sentence ending + optional quotes/parens + space + capital letter
        pattern = r'([.!?]+[\'")\]]?)\s+([A-Z])'
        
        # Find all split points
        splits = list(re.finditer(pattern, text))
        
        if not splits:
            # No sentence boundaries found, return as single sentence
            return [text.strip()] if text.strip() else []
        
        sentences = []
        start = 0
        
        for match in splits:
            # Add sentence up to the punctuation
            end = match.start() + len(match.group(1))
            sentence = text[start:end].strip()
            if sentence:
                sentences.append(sentence)
            
            # Next sentence starts with the capital letter
            start = match.start() + len(match.group(1))
            # Skip whitespace
            while start < len(text) and text[start].isspace():
                start += 1
        
        # Add remaining text
        if start < len(text):
            remaining = text[start:].strip()
            if remaining:
                sentences.append(remaining)
        
        return sentences
    
    def _is_valid_sentence(self, sentence: str) -> bool:
        """Filter out non-content sentences."""
        if not sentence or len(sentence) < MIN_SENTENCE_LENGTH:
            return False
        
        # Skip markdown headers (already processed)
        if sentence.startswith('#'):
            return False
        
        # Skip metadata/code blocks
        if sentence.startswith('---') or sentence.startswith('```'):
            return False
        
        # Skip URL-only lines
        if sentence.startswith(('http://', 'https://')) and ' ' not in sentence:
            return False
        
        # Must contain alphabetic characters
        if not any(c.isalpha() for c in sentence):
            return False
        
        return True
    
    def _clean_markdown(self, sentence: str) -> str:
        """Remove basic markdown formatting."""
        # Remove bold/italic markers
        sentence = re.sub(r'\*\*(.*?)\*\*', r'\1', sentence)
        sentence = re.sub(r'\*(.*?)\*', r'\1', sentence)
        sentence = re.sub(r'__(.*?)__', r'\1', sentence)
        sentence = re.sub(r'_(.*?)_', r'\1', sentence)
        
        # Remove inline code
        sentence = re.sub(r'`([^`]+)`', r'\1', sentence)
        
        # Remove links
        sentence = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', sentence)
        
        return sentence.strip()


class AbstractGenerator:
    """Generate L0/L1 abstracts via hybrid extractive summarization.
    
    Strategy:
    - L0: metadata-aware (section header + first sentence + key terms)
    - L1: centroid-based (top representative sentences in original order)
    
    This hybrid approach preserves searchable terms in L0 (headers, titles)
    while using semantic centrality for L1 overview.
    """
    
    def __init__(self, embedder: Embedder):
        """
        Initialize with existing embedder instance.
        
        Args:
            embedder: Configured embedder for generating sentence embeddings
        """
        self.embedder = embedder
        self.splitter = SentenceSplitter()
    
    def generate(self, content: str, metadata: dict | None = None,
                l0_sentences: int = DEFAULT_L0_SENTENCES, 
                l1_sentences: int = DEFAULT_L1_SENTENCES) -> AbstractResult:
        """
        Generate L0 and L1 abstracts from content.
        
        Hybrid algorithm:
        - L0: metadata-aware — uses headers, title, first sentence (preserves search terms)
        - L1: centroid-based — top representative sentences in original order
        
        Args:
            content: Text content to summarize
            metadata: Optional chunk metadata (section, parent_header, file_path)
            l0_sentences: Number of sentences for L0 abstract
            l1_sentences: Number of sentences for L1 overview
            
        Returns:
            AbstractResult with generated abstracts and embeddings
        """
        start_time = time.time()
        metadata = metadata or {}
        
        # Handle empty content
        if not content or not content.strip():
            placeholder_text = "."
            empty_embedding = self.embedder.embed(placeholder_text)
            return AbstractResult(
                l0_abstract="",
                l1_overview="",
                l0_embedding=empty_embedding,
                l1_embedding=empty_embedding,
                original_sentences=0,
                generation_time_ms=0.0
            )
        
        # Split into sentences
        sentences = self.splitter.split(content)
        
        # Handle edge cases
        if len(sentences) < MIN_SENTENCES_FOR_EXTRACTION:
            full_content = content.strip()
            full_embedding = self.embedder.embed(full_content)
            return AbstractResult(
                l0_abstract=full_content,
                l1_overview=full_content,
                l0_embedding=full_embedding,
                l1_embedding=full_embedding,
                original_sentences=len(sentences),
                generation_time_ms=(time.time() - start_time) * 1000
            )
        
        # Embed all sentences
        sentence_embeddings = self.embedder.embed(sentences)
        
        # L0: metadata-aware (headers + first sentence = best for search)
        l0_abstract = self._build_metadata_l0(content, sentences, metadata)
        
        # L1: centroid-based (representative sentences in order)
        scores = self._compute_centroid_scores(sentence_embeddings)
        _, l1_overview = self._select_representatives(
            sentences, scores, l0_sentences, l1_sentences
        )
        
        # Generate final embeddings
        l0_embedding = self.embedder.embed(l0_abstract)
        l1_embedding = self.embedder.embed(l1_overview)
        
        generation_time = (time.time() - start_time) * 1000
        
        return AbstractResult(
            l0_abstract=l0_abstract,
            l1_overview=l1_overview,
            l0_embedding=l0_embedding,
            l1_embedding=l1_embedding,
            original_sentences=len(sentences),
            generation_time_ms=generation_time
        )
    
    def _build_metadata_l0(self, content: str, sentences: list, metadata: dict) -> str:
        """
        Build L0 abstract from metadata + first sentence.
        
        This preserves the terms users actually search for:
        - Section headers (## Network Security → "network security")
        - Parent headers (# CS656 → "CS656")
        - File path terms (projects/jade-homelab.md → "jade homelab")
        - First meaningful sentence (topic sentence)
        """
        parts = []
        
        # Extract headers from content directly
        import re
        headers = re.findall(r'^#{1,3}\s+(.+)$', content, re.MULTILINE)
        if headers:
            parts.extend(headers[:3])  # Top 3 headers
        
        # Add metadata section/parent if available and not already in headers
        section = metadata.get('section', '')
        parent = metadata.get('parent_header', '')
        if parent and parent not in ' '.join(parts):
            parts.append(parent)
        if section and section not in ' '.join(parts) and section not in ('full_document', 'no_headers'):
            parts.append(section)
        
        # Add file path terms (clean up path → searchable terms)
        file_path = metadata.get('file_path', '')
        if file_path:
            # Extract filename without extension
            fname = file_path.split('/')[-1].replace('.md', '').replace('-', ' ').replace('_', ' ')
            if fname and fname not in ' '.join(parts):
                parts.append(fname)
        
        # Add first meaningful sentence
        if sentences:
            parts.append(sentences[0])
        
        # Join and deduplicate
        l0 = '. '.join(parts)
        
        # Cap at reasonable length
        if len(l0) > 500:
            l0 = l0[:497] + '...'
        
        return l0
    
    def generate_batch(self, contents: List[str], metadatas: List[dict] | None = None,
                      l0_sentences: int = DEFAULT_L0_SENTENCES,
                      l1_sentences: int = DEFAULT_L1_SENTENCES) -> List[AbstractResult]:
        """
        Batch generation — processes multiple documents efficiently.
        
        Collects all sentences from all documents and embeds them in one batch
        for maximum efficiency.
        
        Args:
            contents: List of text contents to process
            l0_sentences: Number of sentences for L0 abstracts
            l1_sentences: Number of sentences for L1 overviews
            
        Returns:
            List of AbstractResult objects corresponding to input contents
        """
        if not contents:
            return []
        
        if metadatas is None:
            metadatas = [{} for _ in contents]
        
        start_time = time.time()
        
        # Collect all sentences from all documents
        all_sentences = []
        doc_sentence_map = []  # (doc_idx, sentence_start_idx, sentence_count)
        
        for doc_idx, content in enumerate(contents):
            if not content or not content.strip():
                doc_sentence_map.append((doc_idx, len(all_sentences), 0))
                continue
            
            sentences = self.splitter.split(content)
            sentence_start = len(all_sentences)
            all_sentences.extend(sentences)
            doc_sentence_map.append((doc_idx, sentence_start, len(sentences)))
        
        # Single batch embedding call for all sentences
        if all_sentences:
            all_embeddings = self.embedder.embed(all_sentences)
        else:
            all_embeddings = np.array([])
        
        # Process each document
        results = []
        
        for doc_idx, content in enumerate(contents):
            # Find this document's sentence data
            _, sentence_start, sentence_count = doc_sentence_map[doc_idx]
            
            if sentence_count == 0:
                # Empty document - use a placeholder embedding
                placeholder_text = "."
                empty_embedding = self.embedder.embed(placeholder_text)
                results.append(AbstractResult(
                    l0_abstract="",
                    l1_overview="",
                    l0_embedding=empty_embedding,
                    l1_embedding=empty_embedding,
                    original_sentences=0,
                    generation_time_ms=0.0
                ))
                continue
            
            # Extract this document's sentences and embeddings
            doc_sentences = all_sentences[sentence_start:sentence_start + sentence_count]
            doc_embeddings = all_embeddings[sentence_start:sentence_start + sentence_count]
            
            # Process this document
            if sentence_count < MIN_SENTENCES_FOR_EXTRACTION:
                # Use full content
                full_content = content.strip()
                full_embedding = self.embedder.embed(full_content)
                
                results.append(AbstractResult(
                    l0_abstract=full_content,
                    l1_overview=full_content,
                    l0_embedding=full_embedding,
                    l1_embedding=full_embedding,
                    original_sentences=sentence_count,
                    generation_time_ms=0.0  # Individual timing not tracked in batch
                ))
            else:
                # Normal processing — hybrid L0 (metadata) + centroid L1
                doc_metadata = metadatas[doc_idx] if doc_idx < len(metadatas) else {}
                l0_abstract = self._build_metadata_l0(content, doc_sentences, doc_metadata)
                
                scores = self._compute_centroid_scores(doc_embeddings)
                _, l1_overview = self._select_representatives(
                    doc_sentences, scores, l0_sentences, l1_sentences
                )
                
                # Generate final embeddings
                l0_embedding = self.embedder.embed(l0_abstract)
                l1_embedding = self.embedder.embed(l1_overview)
                
                results.append(AbstractResult(
                    l0_abstract=l0_abstract,
                    l1_overview=l1_overview,
                    l0_embedding=l0_embedding,
                    l1_embedding=l1_embedding,
                    original_sentences=sentence_count,
                    generation_time_ms=0.0  # Individual timing not tracked in batch
                ))
        
        total_time = (time.time() - start_time) * 1000
        logger.info(f"Batch processed {len(contents)} documents in {total_time:.1f}ms")
        
        return results
    
    def _compute_centroid_scores(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity scores against centroid.
        
        Args:
            embeddings: 2D array of sentence embeddings
            
        Returns:
            1D array of similarity scores
        """
        # Compute centroid
        centroid = np.mean(embeddings, axis=0)
        
        # Normalize embeddings and centroid
        embedding_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embedding_norms = np.where(embedding_norms > 0, embedding_norms, 1.0)
        normalized_embeddings = embeddings / embedding_norms
        
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            normalized_centroid = centroid / centroid_norm
        else:
            normalized_centroid = centroid
        
        # Compute cosine similarities
        scores = np.dot(normalized_embeddings, normalized_centroid)
        
        return scores
    
    def _select_representatives(self, sentences: List[str], scores: np.ndarray,
                              l0_count: int, l1_count: int) -> Tuple[str, str]:
        """
        Select and format L0/L1 abstracts maintaining original order.
        
        Args:
            sentences: List of sentence strings
            scores: Array of similarity scores for each sentence
            l0_count: Number of sentences for L0
            l1_count: Number of sentences for L1
            
        Returns:
            Tuple of (L0 abstract, L1 overview)
        """
        # Get indices of top-scoring sentences
        top_l0_indices = np.argsort(scores)[-l0_count:]
        top_l1_indices = np.argsort(scores)[-l1_count:]
        
        # L0: Join top sentences (no reordering)
        l0_sentences = [sentences[i] for i in sorted(top_l0_indices)]
        l0_abstract = ' '.join(l0_sentences)
        
        # L1: Reorder by original position to preserve narrative flow
        l1_indices_ordered = sorted(top_l1_indices)
        l1_sentences = [sentences[i] for i in l1_indices_ordered]
        l1_overview = ' '.join(l1_sentences)
        
        return l0_abstract, l1_overview