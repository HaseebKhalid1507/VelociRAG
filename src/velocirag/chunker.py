"""
Markdown chunking for semantic vector search.

Split markdown documents by semantic sections (## and ### headers).
Preserves hierarchical context while maintaining manageable chunk sizes.
Extracted from production Jawz vector search system.
"""

import re
import hashlib
import frontmatter
import yaml

# Chunking constants
MIN_FILE_SIZE_FOR_CHUNKING = 500  # Files smaller than this become single chunk
MAX_CHUNK_SIZE = 4000            # Truncate chunks longer than this (~1000 tokens)
MIN_SECTION_SIZE = 10            # Skip empty sections smaller than this
H1_SEARCH_WINDOW = 500           # How far to look for h1 headers at start


def _sanitize_frontmatter(metadata: dict) -> dict:
    """Convert non-JSON-serializable frontmatter values (date, datetime) to strings."""
    import datetime as dt
    sanitized = {}
    for key, value in metadata.items():
        if isinstance(value, (dt.date, dt.datetime)):
            sanitized[key] = value.isoformat()
        elif isinstance(value, dict):
            sanitized[key] = _sanitize_frontmatter(value)
        elif isinstance(value, list):
            sanitized[key] = [
                v.isoformat() if isinstance(v, (dt.date, dt.datetime)) else v
                for v in value
            ]
        else:
            sanitized[key] = value
    return sanitized


def chunk_markdown(content: str | None, file_path: str = "") -> list[dict]:
    """
    Split markdown content by semantic sections (## and ### headers).
    Preserves parent context for nested sections.
    
    Args:
        content: Raw markdown content (may include YAML frontmatter). None returns empty list.
        file_path: Optional file path for metadata (used in chunk metadata)
    
    Returns:
        List of chunk dictionaries:
        {
            'content': str,           # Chunk text with parent headers prepended
            'metadata': {
                'file_path': str,     # Original file path
                'section': str,       # Header text of this section
                'parent_header': str | None,  # Parent header (## parent for ###, # parent for ##)
                'frontmatter': dict,  # Parsed YAML frontmatter
                'content_hash': str,  # MD5 hash (first 12 chars) for deduplication
            }
        }
    """
    
    # Handle empty content
    if not content or not content.strip():
        return []
    
    # Parse frontmatter
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
    
    # Split by headers (## and ###)
    header_pattern = re.compile(r'^(#{2,3})\s+(.+)$', re.MULTILINE)
    chunks = []
    
    # Find all headers
    headers = list(header_pattern.finditer(body))
    
    if not headers:
        # No headers found - return as single chunk
        content_with_header = context_header + "\n" + body.strip()
        return [{
            'content': content_with_header,
            'metadata': {
                'file_path': file_path,
                'section': 'no_headers',
                'parent_header': None,
                'frontmatter': metadata,
                'content_hash': _content_hash(body.strip()),
                'has_context_header': True
            }
        }]
    
    # Track parent context (h1/h2 hierarchy)
    current_h1 = h1_title  # Use already extracted h1_title
    current_h2 = None
    
    # Process each header section
    for i, header_match in enumerate(headers):
        start_pos = header_match.start()
        end_pos = headers[i + 1].start() if i + 1 < len(headers) else len(body)
        
        header_level = len(header_match.group(1))
        header_text = header_match.group(2).strip()
        section_content = body[start_pos:end_pos].strip()
        
        # Update parent tracking
        if header_level == 2:  # ##
            current_h2 = header_text
        elif header_level == 3:  # ###
            pass  # Keep current h2
        
        # Build parent context - FIXED: consistent hierarchy
        parent_context = []
        if current_h1:
            parent_context.append(f"# {current_h1}")
        if current_h2 and header_level == 3:
            parent_context.append(f"## {current_h2}")
        
        # Combine parent context with section content
        if parent_context:
            body_content = "\n".join(parent_context) + "\n\n" + section_content
        else:
            body_content = section_content
        
        # Skip empty sections
        if len(body_content.strip()) < MIN_SECTION_SIZE:
            continue
        
        # Truncate body content if too long (header is "free" context)
        if len(body_content) > MAX_CHUNK_SIZE:
            body_content = body_content[:MAX_CHUNK_SIZE - 3] + "..."
        
        # Prepend context header to body content
        full_content = context_header + "\n" + body_content
        
        # FIXED: Consistent parent_header - ## sections get # parent, ### sections get ## parent
        parent_header = None
        if header_level == 2:
            parent_header = current_h1
        elif header_level == 3:
            parent_header = current_h2
        
        chunks.append({
            'content': full_content.strip(),
            'metadata': {
                'file_path': file_path,
                'section': header_text,
                'parent_header': parent_header,
                'frontmatter': metadata,
                'content_hash': _content_hash(body_content.strip()),
                'has_context_header': True
            }
        })
    
    return chunks


def build_context_header(file_path: str, frontmatter: dict, h1_title: str | None = None) -> str:
    """Build document context header from available metadata."""
    import os
    
    # Title extraction priority: frontmatter title -> h1_title -> filename
    title = None
    if 'title' in frontmatter and frontmatter['title']:
        title = frontmatter['title']
    elif h1_title:
        title = h1_title
    else:
        # Use filename without extension
        filename = os.path.basename(file_path)
        title = os.path.splitext(filename)[0] if filename else "Unknown Document"
    
    lines = []
    lines.append(f"[Document: {title}]")
    lines.append(f"[Source: {file_path}]")
    
    # Tags from frontmatter
    if 'tags' in frontmatter and frontmatter['tags']:
        if isinstance(frontmatter['tags'], list):
            tags_str = ", ".join(str(tag) for tag in frontmatter['tags'])
        else:
            tags_str = str(frontmatter['tags'])
        lines.append(f"[Tags: {tags_str}]")
    
    # Category from frontmatter (category or status field)
    category = None
    if 'category' in frontmatter and frontmatter['category']:
        category = frontmatter['category']
    elif 'status' in frontmatter and frontmatter['status']:
        category = frontmatter['status']
    
    if category:
        lines.append(f"[Category: {category}]")
    
    lines.append("---")
    return "\n".join(lines)


def _content_hash(content: str) -> str:
    """Generate 12-character MD5 hash for content deduplication."""
    return hashlib.md5(content.encode()).hexdigest()[:12]