"""
Semantic chunking for VelociRAG.

Splits documents at topic boundaries using embedding similarity,
instead of arbitrary header positions. Falls back to header-based
chunking when embedder is unavailable.
"""

import gc
import re
import numpy as np
from typing import Optional
import logging

from .chunker import chunk_markdown, build_context_header, _content_hash, _sanitize_frontmatter
from .chunker import MIN_FILE_SIZE_FOR_CHUNKING, H1_SEARCH_WINDOW

# Semantic chunking constants
MIN_SENTENCES_FOR_SEMANTIC = 5  # Minimum sentences to attempt semantic chunking
MIN_CHUNK_SIZE = 100            # Merge chunks smaller than this
MAX_CHUNK_SIZE_SEMANTIC = 4000  # Split chunks larger than this

logger = logging.getLogger("velocirag.semantic_chunker")


def split_sentences(text: str) -> list[str]:
    """
    Split text into sentences. Use regex — no NLTK/spacy deps.
    Handle: periods, question marks, exclamation marks, newlines.
    Don't split on: abbreviations (Mr., Dr., etc.), URLs, numbers (3.14).
    """
    # Common abbreviations that shouldn't be split on
    abbreviations = {
        'Mr', 'Mrs', 'Ms', 'Dr', 'Prof', 'Sr', 'Jr', 'Ph.D', 'M.D',
        'B.A', 'M.A', 'B.S', 'M.S', 'Ph.D', 'etc', 'vs', 'Inc', 'Corp',
        'Ltd', 'Co', 'St', 'Ave', 'Blvd', 'Rd', 'i.e', 'e.g', 'cf'
    }
    
    # Replace abbreviations with placeholders
    protected_text = text
    replacements = {}
    for abbr in abbreviations:
        placeholder = f"__ABBR_{len(replacements)}__"
        protected_text = protected_text.replace(f"{abbr}.", placeholder)
        replacements[placeholder] = f"{abbr}."
    
    # Split on sentence-ending punctuation followed by whitespace or end of string
    # Also split on double newlines (paragraph breaks)
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?:\n\s*\n)'
    sentences = re.split(sentence_pattern, protected_text)
    
    # Restore abbreviations and clean up
    cleaned_sentences = []
    for sentence in sentences:
        # Restore abbreviations
        for placeholder, original in replacements.items():
            sentence = sentence.replace(placeholder, original)
        
        # Clean up whitespace
        sentence = sentence.strip()
        if sentence and len(sentence) >= 5:  # Skip very short fragments
            cleaned_sentences.append(sentence)
    
    return cleaned_sentences


MAX_SENTENCES_FOR_SEMANTIC = 200  # Cap sentences per section to prevent memory explosion


def calculate_boundary_scores(sentences: list[str], embedder) -> list[float]:
    """
    Embed each sentence, return cosine similarity between consecutive sentences.
    Returns list of N-1 similarity scores.
    Caps at MAX_SENTENCES_FOR_SEMANTIC to prevent memory explosion on huge sections.
    """
    if len(sentences) < 2:
        return []
    
    # Cap sentences to prevent memory explosion on huge files
    if len(sentences) > MAX_SENTENCES_FOR_SEMANTIC:
        logger.info(f"Capping semantic analysis from {len(sentences)} to {MAX_SENTENCES_FOR_SEMANTIC} sentences")
        sentences = sentences[:MAX_SENTENCES_FOR_SEMANTIC]
    
    try:
        # Embed all sentences at once for efficiency
        embeddings = embedder.embed(sentences)
        if embeddings.ndim == 1:  # Single sentence case
            return []
        
        # Calculate cosine similarity between consecutive sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            # Normalize embeddings for cosine similarity
            emb1 = embeddings[i]
            emb2 = embeddings[i + 1]
            
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                similarity = 0.0
            else:
                similarity = np.dot(emb1, emb2) / (norm1 * norm2)
            
            similarities.append(similarity)
        
        return similarities
    
    except Exception as e:
        logger.warning(f"Failed to calculate sentence similarities: {e}")
        return []


def find_semantic_boundaries(similarities: list[float], method: str = 'percentile', threshold: float = 25.0) -> list[int]:
    """
    Find indices where topic changes.
    method='percentile': split where similarity is below the Nth percentile (lower = fewer splits)
    method='stddev': split where similarity is more than threshold*stddev below mean
    Returns list of sentence indices that are boundaries.
    """
    if not similarities:
        return []
    
    similarities_array = np.array(similarities)
    boundaries = []
    
    if method == 'percentile':
        # Split at points below the given percentile
        cutoff = np.percentile(similarities_array, threshold)
        boundaries = [i + 1 for i, sim in enumerate(similarities) if sim < cutoff]
    
    elif method == 'stddev':
        # Split at points more than threshold standard deviations below mean
        mean_sim = np.mean(similarities_array)
        std_sim = np.std(similarities_array)
        cutoff = mean_sim - (threshold * std_sim)
        boundaries = [i + 1 for i, sim in enumerate(similarities) if sim < cutoff]
    
    else:
        raise ValueError(f"Unknown boundary detection method: {method}")
    
    # Always include the start and end
    # If we have N similarities, we have N+1 sentences
    boundaries = [0] + sorted(set(boundaries)) + [len(similarities) + 1]
    
    return boundaries


def semantic_chunk_markdown(content: str, file_path: str, embedder, 
                           method: str = 'percentile', threshold: float = 25.0,
                           min_chunk_size: int = MIN_CHUNK_SIZE, 
                           max_chunk_size: int = MAX_CHUNK_SIZE_SEMANTIC) -> list[dict]:
    """
    Semantic chunking with CCH headers.
    
    1. Parse frontmatter (same as chunk_markdown)
    2. Split body into sentences
    3. If < 5 sentences, fall back to regular chunk_markdown()
    4. Embed sentences and find boundaries
    5. Group sentences between boundaries
    6. Merge tiny chunks (< min_chunk_size) with neighbors
    7. Split huge chunks (> max_chunk_size) at sentence boundaries
    8. Apply CCH context header to each chunk
    9. Return same format as chunk_markdown() — drop-in compatible
    
    Returns: list of chunk dicts with same schema as chunk_markdown()
    """
    
    # Handle empty content
    if not content or not content.strip():
        return []
    
    # Parse frontmatter (copied from chunker.py)
    import frontmatter
    import yaml
    
    try:
        post = frontmatter.loads(content)
        body = post.content
        metadata = _sanitize_frontmatter(post.metadata)
    except (yaml.YAMLError, ValueError, TypeError, KeyError, AttributeError):
        body = content
        metadata = {}
    
    # Extract h1 header if exists at start (for context header)
    h1_pattern = re.compile(r'^#\s+(.+)$', re.MULTILINE)
    h1_match = h1_pattern.search(body[:H1_SEARCH_WINDOW])
    h1_title = h1_match.group(1).strip() if h1_match else None
    
    # Build context header
    context_header = build_context_header(file_path, metadata, h1_title)
    
    # Small file optimization - don't chunk files under threshold
    if len(body.strip()) < MIN_FILE_SIZE_FOR_CHUNKING:
        content_with_header = context_header + "\n" + body.strip()
        return [{
            'content': content_with_header,
            'metadata': {
                'file_path': file_path,
                'section': 'full_document',
                'parent_header': None,
                'frontmatter': metadata,
                'content_hash': _content_hash(body.strip()),
                'has_context_header': True
            }
        }]
    
    # Split into sentences
    sentences = split_sentences(body)
    
    # Fallback to header-based chunking if too few sentences
    if len(sentences) < MIN_SENTENCES_FOR_SEMANTIC:
        logger.debug(f"Too few sentences ({len(sentences)}) for semantic chunking, falling back to header-based")
        return chunk_markdown(content, file_path)
    
    # Try semantic chunking with embedder
    try:
        # Calculate similarity scores
        similarities = calculate_boundary_scores(sentences, embedder)
        if not similarities:
            logger.debug("Failed to calculate similarities, falling back to header-based chunking")
            return chunk_markdown(content, file_path)
        
        # Find boundaries
        boundaries = find_semantic_boundaries(similarities, method, threshold)
        
        # Group sentences into chunks
        chunks = []
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]
            chunk_sentences = sentences[start_idx:end_idx]
            
            if not chunk_sentences:
                continue
            
            chunk_text = ' '.join(chunk_sentences)
            
            # Skip very small chunks (will be merged later)
            if len(chunk_text.strip()) < 20:
                continue
            
            chunks.append(chunk_text)
        
        # If no valid chunks were created, fall back
        if not chunks:
            logger.debug("No valid semantic chunks created, falling back to header-based chunking")
            return chunk_markdown(content, file_path)
        
        # Merge small chunks with neighbors
        chunks = _merge_small_chunks(chunks, min_chunk_size)
        
        # Split overly large chunks
        chunks = _split_large_chunks(chunks, max_chunk_size, sentences)
        
        # Build final chunk objects with CCH headers
        final_chunks = []
        for i, chunk_text in enumerate(chunks):
            # Use first sentence (truncated) as section name
            first_sentence = chunk_text.split('.')[0].strip()
            section_name = first_sentence[:50] + "..." if len(first_sentence) > 50 else first_sentence
            
            # Extract parent header from content if headers present
            parent_header = _extract_parent_header(chunk_text, h1_title)
            
            # Prepend context header
            full_content = context_header + "\n" + chunk_text.strip()
            
            chunk_obj = {
                'content': full_content,
                'metadata': {
                    'file_path': file_path,
                    'section': section_name,
                    'parent_header': parent_header,
                    'frontmatter': metadata,
                    'content_hash': _content_hash(chunk_text.strip()),
                    'has_context_header': True,
                    'semantic_chunk': True  # Flag to indicate semantic chunking
                }
            }
            final_chunks.append(chunk_obj)
        
        return final_chunks
    
    except Exception as e:
        logger.warning(f"Semantic chunking failed: {e}, falling back to header-based")
        return chunk_markdown(content, file_path)


def _merge_small_chunks(chunks: list[str], min_chunk_size: int) -> list[str]:
    """Merge chunks smaller than min_chunk_size with their neighbors."""
    if not chunks:
        return chunks
    
    merged = []
    i = 0
    
    while i < len(chunks):
        current_chunk = chunks[i]
        
        # If current chunk is too small, try to merge with next
        if len(current_chunk) < min_chunk_size and i + 1 < len(chunks):
            merged_chunk = current_chunk + " " + chunks[i + 1]
            merged.append(merged_chunk)
            i += 2  # Skip the next chunk since we merged it
        else:
            merged.append(current_chunk)
            i += 1
    
    return merged


def _split_large_chunks(chunks: list[str], max_chunk_size: int, all_sentences: list[str]) -> list[str]:
    """Split chunks larger than max_chunk_size at sentence boundaries."""
    split_chunks = []
    
    for chunk in chunks:
        if len(chunk) <= max_chunk_size:
            split_chunks.append(chunk)
            continue
        
        # Find sentences within this chunk
        chunk_sentences = []
        chunk_text = chunk
        for sentence in all_sentences:
            if sentence in chunk_text:
                chunk_sentences.append(sentence)
                chunk_text = chunk_text.replace(sentence, "", 1)
        
        if not chunk_sentences:
            # Fallback: just truncate
            split_chunks.append(chunk[:max_chunk_size] + "...")
            continue
        
        # Split at sentence boundaries
        current_chunk = ""
        for sentence in chunk_sentences:
            if len(current_chunk + sentence) > max_chunk_size:
                if current_chunk:
                    split_chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        if current_chunk:
            split_chunks.append(current_chunk.strip())
    
    return split_chunks


def _extract_parent_header(chunk_text: str, h1_title: Optional[str]) -> Optional[str]:
    """Extract parent header from chunk content, if headers are present."""
    # Look for headers in the chunk text
    header_pattern = re.compile(r'^(#{1,3})\s+(.+)$', re.MULTILINE)
    headers = list(header_pattern.finditer(chunk_text))
    
    if not headers:
        return h1_title  # Return document-level h1 if no headers in chunk
    
    # Find the highest-level header as parent context
    min_level = min(len(match.group(1)) for match in headers)
    
    for match in headers:
        if len(match.group(1)) == min_level:
            return match.group(2).strip()
    
    return h1_title


def hybrid_chunk_markdown(content: str, file_path: str, embedder,
                          semantic_threshold: float = 25.0,
                          large_section_threshold: int = 1500,
                          min_chunk_size: int = 100, 
                          max_chunk_size: int = 4000) -> list[dict]:
    """
    Hybrid chunking combining header-based and semantic splitting.
    
    1. Run chunk_markdown() first — get header-based chunks with CCH
    2. For each chunk, check body size (content after the `---` CCH separator)
    3. If body size > large_section_threshold:
       - Extract the body content (after CCH header)
       - Run semantic boundary detection on it
       - If semantic splitting produces 2+ sub-chunks, split it
       - Re-apply the SAME CCH header to each sub-chunk
       - Preserve the original section name in metadata, add sub-index
    4. If body size <= threshold: keep the chunk as-is
    5. Return all chunks in the same dict format as chunk_markdown()
    
    Args:
        content: Raw markdown content (may include YAML frontmatter)
        file_path: File path for metadata
        embedder: Embedder instance for semantic boundary detection
        semantic_threshold: Boundary threshold for semantic splitting (lower = fewer splits)
        large_section_threshold: Min body size in chars to consider for semantic splitting
        min_chunk_size: Merge semantic sub-chunks smaller than this
        max_chunk_size: Split semantic sub-chunks larger than this
        
    Returns:
        List of chunk dictionaries with same schema as chunk_markdown()
    """
    # Handle empty content
    if not content or not content.strip():
        return []
    
    # Handle missing embedder - fall back to pure header-based
    if embedder is None:
        logger.debug("No embedder provided, falling back to header-based chunking")
        return chunk_markdown(content, file_path)
    
    # Step 1: Get header-based chunks first
    header_chunks = chunk_markdown(content, file_path)
    
    # If no chunks or chunking failed, return as-is
    if not header_chunks:
        return header_chunks
    
    hybrid_chunks = []
    
    # Step 2: Process each header chunk
    for chunk in header_chunks:
        chunk_content = chunk['content']
        
        # Extract body content after CCH header (after the "---" separator)
        cch_separator = "---"
        if cch_separator in chunk_content:
            parts = chunk_content.split(cch_separator, 1)
            if len(parts) == 2:
                cch_header = parts[0] + cch_separator
                body_content = parts[1].strip()
            else:
                # No proper CCH header structure
                cch_header = ""
                body_content = chunk_content.strip()
        else:
            # No CCH header (shouldn't happen with current chunker, but handle gracefully)
            cch_header = ""
            body_content = chunk_content.strip()
        
        # Step 3: Check if body is large enough for semantic splitting
        if len(body_content) <= large_section_threshold:
            # Small section - keep as-is
            hybrid_chunks.append(chunk)
            continue
        
        # Step 4: Try semantic splitting on the body content
        try:
            # Split body into sentences
            sentences = split_sentences(body_content)
            
            # Need at least 5 sentences for meaningful semantic splitting
            if len(sentences) < MIN_SENTENCES_FOR_SEMANTIC:
                logger.debug(f"Too few sentences ({len(sentences)}) in large section, keeping as-is")
                hybrid_chunks.append(chunk)
                continue
            
            # Calculate similarity scores between sentences
            similarities = calculate_boundary_scores(sentences, embedder)
            if not similarities:
                logger.debug("Failed to calculate similarities, keeping section as-is")
                hybrid_chunks.append(chunk)
                continue
            
            # Find semantic boundaries
            boundaries = find_semantic_boundaries(similarities, method='percentile', threshold=semantic_threshold)
            
            # If only 2 boundaries (start and end), no meaningful split was found
            if len(boundaries) <= 2:
                logger.debug("No meaningful semantic boundaries found, keeping as-is")
                hybrid_chunks.append(chunk)
                continue
            
            # Step 5: Create sub-chunks from boundaries
            sub_chunks = []
            for i in range(len(boundaries) - 1):
                start_idx = boundaries[i]
                end_idx = boundaries[i + 1]
                sub_sentences = sentences[start_idx:end_idx]
                
                if not sub_sentences:
                    continue
                
                sub_body_text = ' '.join(sub_sentences)
                sub_chunks.append(sub_body_text)
            
            # Merge small sub-chunks and split large ones
            if sub_chunks:
                sub_chunks = _merge_small_chunks(sub_chunks, min_chunk_size)
                sub_chunks = _split_large_chunks(sub_chunks, max_chunk_size, sentences)
            
            # If semantic splitting didn't improve things (still 1 chunk), keep original
            if len(sub_chunks) <= 1:
                logger.debug("Semantic splitting didn't create meaningful divisions")
                hybrid_chunks.append(chunk)
                continue
            
            # Step 6: Create final chunk objects with CCH headers
            original_section = chunk['metadata']['section']
            
            for idx, sub_body in enumerate(sub_chunks, 1):
                # Re-apply the same CCH header to each sub-chunk
                sub_content = cch_header + "\n" + sub_body.strip() if cch_header else sub_body.strip()
                
                # Create section name for sub-chunk
                sub_section_name = f"{original_section} (part {idx})"
                
                # Use first sentence of sub-chunk (truncated) as fallback section name
                first_sentence = sub_body.split('.')[0].strip()
                if len(first_sentence) > 50:
                    first_sentence = first_sentence[:47] + "..."
                if not first_sentence:
                    first_sentence = sub_section_name
                
                # Copy metadata from parent chunk
                sub_metadata = chunk['metadata'].copy()
                sub_metadata['section'] = sub_section_name
                sub_metadata['content_hash'] = _content_hash(sub_body.strip())
                # Add hybrid chunking indicator
                sub_metadata['hybrid_chunk'] = True
                sub_metadata['parent_section'] = original_section
                sub_metadata['part_number'] = idx
                sub_metadata['total_parts'] = len(sub_chunks)
                
                sub_chunk = {
                    'content': sub_content,
                    'metadata': sub_metadata
                }
                
                hybrid_chunks.append(sub_chunk)
            
            # Free numpy arrays from this section's boundary analysis
            del similarities
            del sentences
            
        except Exception as e:
            logger.warning(f"Semantic splitting failed for section '{chunk['metadata']['section']}': {e}")
            # Keep original chunk on error
            hybrid_chunks.append(chunk)
    
    # Force garbage collection after processing all chunks
    gc.collect()
    
    return hybrid_chunks