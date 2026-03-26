# Velocirag Phase 1 Technical Specification

*Architecture by Zero — Cold precision for RAG supremacy*

---

## Overview

Phase 1 extracts three pure function modules from the production Jawz vector search system (10K+ lines) into clean, dependency-free building blocks for progressive RAG architecture.

**Design Principle:** Each module is a surgical instrument — no external dependencies, no I/O, no model loading. Pure functions that transform input to output with mathematical precision.

---

## Module 1: `chunker.py`

### Purpose
Split markdown documents into semantically meaningful chunks optimized for embedding and retrieval. Preserves hierarchical context while maintaining manageable chunk sizes.

### Public API

```python
def chunk_markdown(content: str, file_path: str = "") -> list[dict]:
    """
    Split markdown content by semantic sections (## and ### headers).
    Preserves parent context for nested sections.
    
    Args:
        content: Raw markdown content (may include YAML frontmatter)
        file_path: Optional file path for metadata (used in chunk metadata)
    
    Returns:
        List of chunk dictionaries:
        {
            'content': str,           # Chunk text with parent headers prepended
            'metadata': {
                'file_path': str,     # Original file path
                'section': str,       # Header text of this section
                'parent_header': str | None,  # Parent ## header (for ### sections)
                'frontmatter': dict,  # Parsed YAML frontmatter
                'content_hash': str,  # MD5 hash (first 12 chars) for deduplication
            }
        }
    """
```

### Constants

```python
MIN_FILE_SIZE_FOR_CHUNKING = 500   # Files smaller than this become single chunk
MAX_CHUNK_SIZE = 4000              # Truncate chunks longer than this (~1000 tokens)
MIN_SECTION_SIZE = 10              # Skip empty sections smaller than this
H1_SEARCH_WINDOW = 500             # How far to look for h1 headers at start
```

### Core Behavior

1. **Frontmatter Parsing:** Extract YAML frontmatter using `python-frontmatter` library
2. **Small File Optimization:** Files < 500 chars return as single chunk with `section: 'full_document'`
3. **Header Splitting:** Split only on `##` and `###` headers (not `#` or `####+`)
4. **Context Preservation:** 
   - `###` sections get parent `##` header prepended
   - Both get grandparent `#` header if found in first 500 chars
5. **Size Management:**
   - Chunks > 4000 chars truncated with "..." suffix
   - Sections < 10 chars skipped entirely
6. **Deduplication Hash:** MD5 of content, truncated to 12 characters

### Edge Cases

- **Empty Content:** Returns empty list
- **No Headers:** Returns single chunk with `section: 'no_headers'`
- **Unicode Content:** Handles UTF-8 properly via MD5 encoding
- **Malformed YAML:** Falls back to empty frontmatter dict
- **Huge Input:** Memory-safe via streaming regex processing

### Test Cases

```python
# Test 1: Small file optimization
input_content = "# Small file\n\nJust a tiny bit of content."
expected = [{
    'content': '# Small file\n\nJust a tiny bit of content.',
    'metadata': {
        'section': 'full_document',
        'parent_header': None,
        'frontmatter': {},
        'content_hash': 'a1b2c3d4e5f6'  # Example hash
    }
}]

# Test 2: Header hierarchy with context preservation
input_content = """# Main Title

## Section One
Content for section one.

### Subsection A
Content for subsection A.

## Section Two
Content for section two.
"""
expected = [
    # ## Section One
    {
        'content': '# Main Title\n\n## Section One\nContent for section one.',
        'metadata': {
            'section': 'Section One',
            'parent_header': 'Main Title',
            'frontmatter': {},
            'content_hash': 'xyz123abc456'
        }
    },
    # ### Subsection A  
    {
        'content': '# Main Title\n## Section One\n\n### Subsection A\nContent for subsection A.',
        'metadata': {
            'section': 'Subsection A',
            'parent_header': 'Section One',
            'frontmatter': {},
            'content_hash': 'def789ghi012'
        }
    },
    # ## Section Two
    {
        'content': '# Main Title\n\n## Section Two\nContent for section two.',
        'metadata': {
            'section': 'Section Two', 
            'parent_header': 'Main Title',
            'frontmatter': {},
            'content_hash': 'jkl345mno678'
        }
    }
]

# Test 3: Frontmatter extraction
input_content = """---
title: "Technical Document"
tags: [python, ai]
---

## Implementation
Code goes here.
"""
expected = [{
    'content': '## Implementation\nCode goes here.',
    'metadata': {
        'section': 'Implementation',
        'parent_header': None,
        'frontmatter': {'title': 'Technical Document', 'tags': ['python', 'ai']},
        'content_hash': 'frontmatter123'
    }
}]

# Test 4: Size truncation
input_content = f"## Large Section\n{'x' * 5000}"
expected = [{
    'content': f"## Large Section\n{'x' * 3987}...",  # Truncated to 4000 chars
    'metadata': {
        'section': 'Large Section',
        'parent_header': None,
        'frontmatter': {},
        'content_hash': 'truncated456'
    }
}]

# Test 5: Empty sections filtered
input_content = """## Good Section
Real content here.

## Empty Section

### Also Empty

## Another Good Section
More content.
"""
expected = [
    # Empty sections skipped, only "Good Section" and "Another Good Section" returned
]
```

---

## Module 2: `variants.py`

### Purpose
Generate normalized query variants to improve search recall by handling common text variations (casing, spacing, punctuation) that users might search for differently.

### Public API

```python
def generate_variants(query: str) -> list[str]:
    """
    Generate normalized query variants for improved recall.
    Original query always appears first in the returned list.
    
    Args:
        query: Original search query string
        
    Returns:
        Deduplicated list of query variants (max 8 variants)
        
    Patterns handled:
    - Case variants: "CS656" → "cs656"  
    - Spacing: "CS656" ↔ "CS 656"
    - Hyphens: "CS-656" → "CS656", "CS 656"
    - Underscores: "file_name" → "file name", "filename" 
    - Dots: "script.py" → "script py", "scriptpy"
    """
```

### Constants

```python
MAX_VARIANTS = 8    # Maximum variants to return (performance constraint)
```

### Core Behavior

1. **Input Validation:** Return empty list for None/empty input
2. **Original Preservation:** First variant is always the original query
3. **Letter-Number Boundaries:**
   - `CS656` → `CS 656` (add space)
   - `CS 656` → `CS656` (remove space)
4. **Case Variants:** Generate lowercase if different from original
5. **Punctuation Normalization:**
   - Hyphens: `CS-656` → `CS656`, `CS 656`
   - Underscores: `file_name` → `file name`, `filename`
   - Dots: `script.py` → `script py`, `scriptpy`
6. **Deduplication:** Preserve order, no duplicates
7. **Performance Cap:** Limit to 8 variants maximum

### Edge Cases

- **Empty Input:** Returns `[]`
- **Single Character:** Returns `[original]`
- **All Punctuation:** Handles gracefully without crashing
- **Unicode:** Preserves non-ASCII characters
- **Very Long Query:** Processes efficiently without exponential explosion

### Test Cases

```python
# Test 1: Letter-number boundary detection
assert generate_variants("CS656") == ["CS656", "cs656", "CS 656", "cs 656"]

# Test 2: Hyphen normalization
assert generate_variants("CS-656") == [
    "CS-656", "cs-656", "CS656", "cs656", "CS 656", "cs 656"
]

# Test 3: Underscore handling
assert generate_variants("file_name") == [
    "file_name", "file name", "filename"
]

# Test 4: Mixed punctuation
assert generate_variants("my-script_v2.py") == [
    "my-script_v2.py",
    "my script_v2.py",       # hyphen → space
    "myscript_v2.py",        # hyphen → removed  
    "my-script v2.py",       # underscore → space
    "my-scriptv2.py",        # underscore → removed
    "my-script_v2 py",       # dot → space
    "my-script_v2py",        # dot → removed
    # + lowercase variants (truncated to 8 total)
]

# Test 5: No variants needed
assert generate_variants("simple") == ["simple"]

# Test 6: Case-only variant  
assert generate_variants("UPPERCASE") == ["UPPERCASE", "uppercase"]

# Test 7: Empty/invalid input
assert generate_variants("") == []
assert generate_variants("   ") == []
```

---

## Module 3: `rrf.py` 

### Purpose
Reciprocal Rank Fusion — merge multiple ranked result lists into a single ranked list using proven RRF algorithm. Essential for combining results from different query variants or search strategies.

### Public API

```python
def reciprocal_rank_fusion(
    results_lists: list[list[dict]], 
    k: int = 60,
    doc_id_fn: callable = None
) -> list[dict]:
    """
    Combine multiple result lists using RRF scoring algorithm.
    
    RRF Score = Σ 1/(k + rank) across all lists where document appears
    
    Args:
        results_lists: List of ranked result lists to merge
        k: RRF parameter (default 60, empirically optimal)
        doc_id_fn: Optional function to extract document ID from result.
                   Signature: (result: dict) -> str
                   Default: tries metadata.doc_id → metadata.file_path → content hash
                   
    Returns:
        Merged results sorted by RRF score (highest first).
        Each result gets metadata.rrf_score added.
        
    Requirements for result dicts:
        - Must have 'content' key (string)
        - May have 'metadata' key (dict)
        - Other keys preserved as-is
    """
```

### Constants

```python
DEFAULT_RRF_K = 60          # Empirically proven optimal k value
MIN_RRF_K = 1               # Minimum allowed k value  
MAX_RRF_K = 1000            # Maximum allowed k value
MAX_FUSION_RESULTS = 1000   # Memory protection limit
```

### Core Behavior

1. **RRF Algorithm:** For each document, sum `1/(k + rank)` across all lists
2. **Document Deduplication:**
   - Try `metadata.doc_id` first
   - Fall back to `metadata.file_path`  
   - Last resort: hash of content
3. **Version Selection:** When same document appears multiple times, keep version with highest `metadata.similarity`
4. **Memory Protection:** Cap total input results at 1000 to prevent exhaustion
5. **Score Injection:** Add `rrf_score` to result metadata
6. **Parameter Validation:** Ensure k is integer in valid range

### Edge Cases

- **Empty Input:** Returns `[]`
- **Single List:** Returns original list with RRF scores added
- **No Common Documents:** All documents ranked by individual list performance
- **Missing Metadata:** Creates empty metadata dict if needed
- **Huge Input:** Memory protection via proportional truncation
- **Invalid k:** Raises ValueError with clear message

### Test Cases

```python
# Test 1: Basic RRF fusion
list1 = [
    {'content': 'doc1', 'metadata': {'similarity': 0.9}},
    {'content': 'doc2', 'metadata': {'similarity': 0.8}}
]
list2 = [
    {'content': 'doc2', 'metadata': {'similarity': 0.85}}, # Same doc, different score
    {'content': 'doc3', 'metadata': {'similarity': 0.7}}
]
expected_order = ['doc2', 'doc1', 'doc3']  # doc2 appears in both lists
result = reciprocal_rank_fusion([list1, list2], k=60)
actual_order = [r['content'] for r in result]
assert actual_order == expected_order
assert 'rrf_score' in result[0]['metadata']

# Test 2: Custom doc_id_fn
def custom_id(result):
    return result['metadata']['custom_id']
    
list1 = [{'content': 'text', 'metadata': {'custom_id': 'doc1'}}]
list2 = [{'content': 'different_text', 'metadata': {'custom_id': 'doc1'}}]
result = reciprocal_rank_fusion([list1, list2], doc_id_fn=custom_id)
assert len(result) == 1  # Deduplicated by custom_id

# Test 3: k parameter effect
list1 = [{'content': 'doc1'}, {'content': 'doc2'}]
list2 = [{'content': 'doc2'}, {'content': 'doc1'}]

# With k=60, doc2 should win (1/61 + 1/62 vs 1/62 + 1/61 - same, but doc2 first)
result_k60 = reciprocal_rank_fusion([list1, list2], k=60)

# With k=1, first list dominance should be stronger
result_k1 = reciprocal_rank_fusion([list1, list2], k=1)

# Verify scores are different
assert result_k60[0]['metadata']['rrf_score'] != result_k1[0]['metadata']['rrf_score']

# Test 4: Memory protection
huge_lists = []
for i in range(10):
    huge_list = [{'content': f'doc{j}'} for j in range(200)]  # 2000 total results
    huge_lists.append(huge_list)

result = reciprocal_rank_fusion(huge_lists)
# Should be truncated to ~1000 results total across all lists
total_input = sum(len(lst) for lst in huge_lists)
assert total_input <= 1000

# Test 5: Invalid parameters
try:
    reciprocal_rank_fusion([], k=0)  # k too small
    assert False, "Should raise ValueError"
except ValueError as e:
    assert "k parameter" in str(e)

try:
    reciprocal_rank_fusion([], k=2000)  # k too large  
    assert False, "Should raise ValueError"
except ValueError as e:
    assert "k parameter" in str(e)
```

---

## Implementation Notes

### Dependencies

**chunker.py:**
- `python-frontmatter` — YAML frontmatter parsing
- `hashlib` — MD5 hashing (stdlib)
- `re` — Regex for header detection (stdlib)

**variants.py:**
- `re` — Pattern matching (stdlib)
- `unicodedata` — Unicode normalization (stdlib)

**rrf.py:**  
- No external dependencies (pure stdlib)

### Performance Characteristics

**chunker.py:**
- O(n) where n = content length
- Memory usage: ~2x content size (original + chunks)
- Regex compilation cached for repeated calls

**variants.py:**
- O(1) — fixed number of operations regardless of query length
- Memory: O(8) — max 8 variants generated
- Extremely fast for real-time query processing

**rrf.py:**
- O(n log n) where n = total unique documents across all lists  
- Memory: O(n) for deduplication tracking
- Memory protection prevents pathological cases

### Error Handling Philosophy

- **Fail Fast:** Invalid parameters raise ValueError/TypeError immediately
- **Graceful Degradation:** Malformed content returns partial results rather than crashing
- **Defensive Coding:** Handle None/empty inputs gracefully
- **Clear Messages:** Error messages include context and suggested fixes

---

*"Architecture is the art of elimination. We remove everything that doesn't serve the core purpose... and what remains is **perfection**."*

— Zero, The Architect

