"""
Velocirag YAML Frontmatter Parser.

Parse YAML frontmatter from markdown content, extract hashtags and wiki links.
Handles edge cases gracefully with robust error handling.
"""

import re
import logging
from typing import Dict, List, Tuple, Any, Union

try:
    import yaml
except ImportError:
    raise ImportError("pyyaml is required for frontmatter parsing. Install with: pip install pyyaml")

logger = logging.getLogger("velocirag.frontmatter")


def parse_frontmatter(content: str) -> Tuple[Dict, str]:
    """
    Parse YAML frontmatter from markdown content.
    
    Args:
        content: Raw markdown content
        
    Returns:
        Tuple of (frontmatter_dict, body_without_frontmatter)
        If no frontmatter found, returns ({}, original_content)
    """
    if not content or not content.strip():
        return {}, content
    
    # Check for frontmatter delimiters
    frontmatter_pattern = re.compile(
        r'^---\s*\n(.*?)^---\s*\n',
        re.DOTALL | re.MULTILINE
    )
    
    match = frontmatter_pattern.match(content)
    if not match:
        # No frontmatter found
        return {}, content
    
    yaml_content = match.group(1)
    body = content[match.end():]
    
    # Parse YAML content
    try:
        frontmatter_dict = yaml.safe_load(yaml_content)
        
        # Handle edge cases
        if frontmatter_dict is None:
            frontmatter_dict = {}
        elif not isinstance(frontmatter_dict, dict):
            # YAML parsed to a non-dict (e.g., just a string or list)
            logger.warning(f"Frontmatter parsed to non-dict type: {type(frontmatter_dict)}")
            frontmatter_dict = {}
        else:
            # Convert date objects to strings for JSON compatibility
            frontmatter_dict = _normalize_frontmatter_values(frontmatter_dict)
            
    except yaml.YAMLError as e:
        logger.warning(f"Failed to parse YAML frontmatter: {e}")
        # Return empty dict for malformed YAML, don't crash
        frontmatter_dict = {}
    
    return frontmatter_dict, body


def extract_tags_from_content(content: str) -> List[str]:
    """
    Extract #hashtags from markdown content.
    
    Args:
        content: Markdown content
        
    Returns:
        List of tag names (without the # prefix)
    """
    if not content:
        return []
    
    # Pattern to match hashtags
    # Matches: #tag, #tag-name, #tag_name
    # Does not match: ##header, # header, #123 (numbers only)
    # Only match hashtags at start of line, after whitespace, or after punctuation (but not in middle of words)
    hashtag_pattern = re.compile(r'(?:^|\s)#([a-zA-Z][a-zA-Z0-9_-]*)', re.MULTILINE)
    
    matches = hashtag_pattern.findall(content)
    
    # Normalize and deduplicate
    tags = []
    seen = set()
    for match in matches:
        normalized = match.lower().strip()
        if normalized and normalized not in seen:
            tags.append(normalized)
            seen.add(normalized)
    
    return tags


def extract_wiki_links(content: str) -> List[str]:
    """
    Extract [[wiki links]] from markdown content.
    
    Args:
        content: Markdown content
        
    Returns:
        List of wiki link targets
    """
    if not content:
        return []
    
    # Pattern to match wiki links: [[target]], [[display|target]]
    # Use negative lookbehind and lookahead to avoid nested brackets
    wiki_pattern = re.compile(r'(?<!\[)\[\[([^\[\]]+)\]\](?!\])', re.MULTILINE)
    
    matches = wiki_pattern.findall(content)
    
    # Process matches
    links = []
    seen = set()
    for match in matches:
        # Handle display|target format
        if '|' in match:
            # Take the target part (after the |)
            target = match.split('|', 1)[1].strip()
        else:
            target = match.strip()
        
        # Normalize
        target = target.strip()
        if target and target not in seen:
            links.append(target)
            seen.add(target)
    
    return links


def _normalize_frontmatter_values(data: Dict) -> Dict:
    """
    Normalize frontmatter values for JSON compatibility.
    
    Converts datetime objects to ISO strings, handles other edge cases.
    """
    if not isinstance(data, dict):
        return data
    
    normalized = {}
    for key, value in data.items():
        if hasattr(value, 'isoformat'):
            # datetime/date object
            try:
                normalized[key] = value.isoformat()
            except:
                normalized[key] = str(value)
        elif isinstance(value, dict):
            # Recursive normalization for nested dicts
            normalized[key] = _normalize_frontmatter_values(value)
        elif isinstance(value, list):
            # Normalize list items
            normalized[key] = [
                item.isoformat() if hasattr(item, 'isoformat') else item
                for item in value
            ]
        else:
            # Keep as-is for strings, numbers, booleans, None
            normalized[key] = value
    
    return normalized