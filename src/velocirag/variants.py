"""
Query variant generation for improved search recall.

Generate normalized query variants to handle common text variations 
(casing, spacing, punctuation) that users might search for differently.
Extracted from production Jawz vector search system.
"""

import re

# Variant generation constants
MAX_VARIANTS = 8    # Maximum variants to return (performance constraint)


def generate_variants(query: str | None) -> list[str]:
    """
    Generate normalized query variants for improved recall.
    Original query always appears first in the returned list.
    
    Args:
        query: Original search query string. None returns empty list.
        
    Returns:
        Deduplicated list of query variants (max 8 variants)
        
    Patterns handled:
    - Case variants: "CS656" → "cs656"  
    - Spacing: "CS656" ↔ "CS 656"
    - Hyphens: "CS-656" → "CS656", "CS 656"
    - Underscores: "file_name" → "file name", "filename" 
    - Dots: "script.py" → "script py", "scriptpy"
    """
    if not query or not query.strip():
        return []
        
    variants = [query]  # Always include original
    
    # Handle case variations
    if query.lower() != query:
        variants.append(query.lower())
    
    # Pattern 1: Letters followed by numbers with no space -> add space
    # e.g., CS656 -> CS 656, HTB123 -> HTB 123
    # Use word boundaries to avoid mangling longer codes
    normalized = re.sub(r'\b([A-Za-z]+)(\d+)\b', r'\1 \2', query)
    if normalized != query:
        variants.append(normalized)
        # Also add lowercase version
        if normalized.lower() != normalized:
            variants.append(normalized.lower())
    
    # Pattern 2: Letters followed by space(s) and numbers -> remove space
    # e.g., CS 656 -> CS656, HTB 123 -> HTB123
    # Handle multiple spaces
    compressed = re.sub(r'\b([A-Za-z]+)\s+(\d+)\b', r'\1\2', query)
    if compressed != query:
        variants.append(compressed)
        # Also add lowercase version
        if compressed.lower() != compressed:
            variants.append(compressed.lower())
    
    # Pattern 3: Handle hyphens
    # e.g., CS-656 -> CS656, CS 656
    if '-' in query:
        no_hyphen = query.replace('-', '')
        if no_hyphen not in variants:
            variants.append(no_hyphen)
            # Add lowercase version
            if no_hyphen.lower() != no_hyphen and no_hyphen.lower() not in variants:
                variants.append(no_hyphen.lower())
        space_hyphen = query.replace('-', ' ')
        if space_hyphen not in variants:
            variants.append(space_hyphen)
            # Add lowercase version
            if space_hyphen.lower() != space_hyphen and space_hyphen.lower() not in variants:
                variants.append(space_hyphen.lower())
    
    # Pattern 4: Handle underscores
    # e.g., file_name -> file name, filename
    if '_' in query:
        no_underscore = query.replace('_', '')
        if no_underscore not in variants:
            variants.append(no_underscore)
        space_underscore = query.replace('_', ' ')
        if space_underscore not in variants:
            variants.append(space_underscore)
    
    # Pattern 5: Handle dots
    # e.g., script.py -> script py, scriptpy
    if '.' in query:
        no_dot = query.replace('.', '')
        if no_dot not in variants:
            variants.append(no_dot)
        space_dot = query.replace('.', ' ')
        if space_dot not in variants:
            variants.append(space_dot)
    
    # Pattern 6: Multi-word query expansions
    # For queries with 4+ words, add word pairs (not individual words — too noisy)
    # For 3-word queries, only add first-two and last-two pairs
    words = query.split()
    if len(words) >= 3:
        # Add first two words as phrase
        first_two = ' '.join(words[:2])
        if first_two not in variants:
            variants.append(first_two)
            if first_two.lower() != first_two and first_two.lower() not in variants:
                variants.append(first_two.lower())
        
        # Add last two words
        last_two = ' '.join(words[-2:])
        if last_two not in variants:
            variants.append(last_two)
            # Also add lowercase version
            if last_two.lower() != last_two and last_two.lower() not in variants:
                variants.append(last_two.lower())
    
    # Deduplicate variants while preserving order
    seen = set()
    deduplicated = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            deduplicated.append(v)
    
    # Enforce MAX_VARIANTS limit
    return deduplicated[:MAX_VARIANTS]