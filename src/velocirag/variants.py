"""
Query variant generation for improved search recall.

Generate normalized query variants to handle common text variations 
(casing, spacing, punctuation) that users might search for differently.
Extracted from production Jawz vector search system.
"""

import re

# Pattern 7: Acronym/alias expansion (bidirectional)
_ACRONYM_MAP = {
    'ml': 'machine learning',
    'ai': 'artificial intelligence',
    'nlp': 'natural language processing',
    'llm': 'large language model',
    'rag': 'retrieval augmented generation',
    'nn': 'neural network',
    'cnn': 'convolutional neural network',
    'rnn': 'recurrent neural network',
    'gpu': 'graphics processing unit',
    'cpu': 'central processing unit',
    'api': 'application programming interface',
    'cli': 'command line interface',
    'db': 'database',
    'sql': 'structured query language',
    'ssh': 'secure shell',
    'tls': 'transport layer security',
    'ssl': 'secure sockets layer',
    'dns': 'domain name system',
    'dhcp': 'dynamic host configuration protocol',
    'vpn': 'virtual private network',
    'vm': 'virtual machine',
    'os': 'operating system',
    'ci': 'continuous integration',
    'cd': 'continuous deployment',
    'k8s': 'kubernetes',
    'tf': 'tensorflow',
    'ner': 'named entity recognition',
    'rrf': 'reciprocal rank fusion',
    'bm25': 'best match 25',
    'fts': 'full text search',
    'orm': 'object relational mapping',
    'jwt': 'json web token',
    'oauth': 'open authorization',
    'cors': 'cross origin resource sharing',
    'xss': 'cross site scripting',
    'csrf': 'cross site request forgery',
    'sqli': 'sql injection',
    'mitm': 'man in the middle',
    'ddos': 'distributed denial of service',
    'ids': 'intrusion detection system',
    'ips': 'intrusion prevention system',
    'siem': 'security information event management',
    'osint': 'open source intelligence',
    'ctf': 'capture the flag',
}

# Build reverse map
_REVERSE_ACRONYM_MAP = {v: k for k, v in _ACRONYM_MAP.items()}

# Variant generation constants
MAX_VARIANTS = 12    # Maximum variants to return (performance constraint)


def generate_variants(query: str | None) -> list[str]:
    """
    Generate normalized query variants for improved recall.
    Original query always appears first in the returned list.
    
    Args:
        query: Original search query string. None returns empty list.
        
    Returns:
        Deduplicated list of query variants (max 12 variants)
        
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
    
    # Pattern 7: Acronym expansion (bidirectional)
    query_lower = query.lower().strip()
    if query_lower in _ACRONYM_MAP:
        expansion = _ACRONYM_MAP[query_lower]
        if expansion not in variants:
            variants.append(expansion)
    elif query_lower in _REVERSE_ACRONYM_MAP:
        acronym = _REVERSE_ACRONYM_MAP[query_lower]
        if acronym not in variants:
            variants.append(acronym)
    # Also check individual words
    for word in query.lower().split():
        if word in _ACRONYM_MAP:
            expanded = query.lower().replace(word, _ACRONYM_MAP[word])
            if expanded not in variants:
                variants.append(expanded)
    
    # Pattern 8: Question → statement rewrite
    q_lower = query.lower().strip()
    question_prefixes = ['what is ', 'what are ', 'how to ', 'how do ', 'how does ',
                         'why is ', 'why are ', 'why does ', 'when did ', 'when does ',
                         'where is ', 'where are ', 'who is ', 'who are ']
    for prefix in question_prefixes:
        if q_lower.startswith(prefix):
            statement = query[len(prefix):].strip().rstrip('?')
            if statement and statement not in variants:
                variants.append(statement)
            break
    
    # Deduplicate variants while preserving order
    seen = set()
    deduplicated = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            deduplicated.append(v)
    
    # Enforce MAX_VARIANTS limit
    return deduplicated[:MAX_VARIANTS]


def register_acronyms(custom_map: dict):
    """Register custom acronym mappings for query expansion.
    
    Args:
        custom_map: Dict of acronym → expansion (e.g., {'loar': 'law of accelerating returns'})
    """
    global _ACRONYM_MAP, _REVERSE_ACRONYM_MAP
    _ACRONYM_MAP.update({k.lower(): v.lower() for k, v in custom_map.items()})
    _REVERSE_ACRONYM_MAP = {v: k for k, v in _ACRONYM_MAP.items()}