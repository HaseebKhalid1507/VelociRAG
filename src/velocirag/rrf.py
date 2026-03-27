"""
Reciprocal Rank Fusion (RRF) for combining multiple ranked result lists.

Merge multiple ranked result lists into a single ranked list using 
proven RRF algorithm. Essential for combining results from different 
query variants or search strategies.
Extracted from production Jawz vector search system.
"""

from typing import Any, Callable
import hashlib

# RRF constants
DEFAULT_RRF_K = 60          # Empirically proven optimal k value
MIN_RRF_K = 1               # Minimum allowed k value  
MAX_RRF_K = 1000            # Maximum allowed k value
MAX_FUSION_RESULTS = 1000   # Memory protection limit


def reciprocal_rank_fusion(
    results_lists: list[list[dict[str, Any]]], 
    k: int = DEFAULT_RRF_K,
    doc_id_fn: Callable[[dict[str, Any]], str] | None = None
) -> list[dict[str, Any]]:
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
    
    # Validate parameters
    if not isinstance(k, int):
        raise ValueError(f"k parameter must be an integer, got {type(k).__name__}")
    if not MIN_RRF_K <= k <= MAX_RRF_K:
        raise ValueError(f"k parameter must be between {MIN_RRF_K} and {MAX_RRF_K}, got {k}")
    
    if not results_lists:
        return []
    
    # Memory protection: Limit total results to prevent exhaustion
    total_results = sum(len(results) for results in results_lists if results)
    if total_results > MAX_FUSION_RESULTS:
        # Equal truncation (not proportional) as per orchestrator decision
        valid_sets = [r for r in results_lists if r]
        max_per_set = MAX_FUSION_RESULTS // len(valid_sets) if valid_sets else 0
        results_lists = [results[:max_per_set] for results in results_lists if results]
    
    doc_scores = {}  # doc_id -> cumulative RRF score
    doc_map = {}     # doc_id -> result dict (keeps highest similarity version)
    
    for query_results in results_lists:
        if not query_results:
            continue
            
        for rank, result in enumerate(query_results, start=1):
            # Generate document ID for deduplication
            doc_id = _generate_doc_id(result, doc_id_fn)
            
            # Store document (keep version with highest similarity if seen multiple times)
            if doc_id not in doc_map:
                doc_map[doc_id] = result
            else:
                # Keep the version with higher similarity score
                current_sim = result.get('metadata', {}).get('similarity', 0.0)
                stored_sim = doc_map[doc_id].get('metadata', {}).get('similarity', 0.0)
                if current_sim > stored_sim:
                    doc_map[doc_id] = result
            
            # Accumulate RRF score: sum of 1/(k + rank) across all queries
            rrf_score = 1.0 / (k + rank)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + rrf_score
    
    # Sort by RRF score (descending) and return documents
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Build final result list with RRF scores added to metadata
    fused_results = []
    for doc_id, rrf_score in sorted_docs:
        result = doc_map[doc_id].copy()
        if 'metadata' in result:
            result['metadata'] = result['metadata'].copy()
        
        # Add RRF score to metadata
        if 'metadata' not in result:
            result['metadata'] = {}
        result['metadata']['rrf_score'] = round(rrf_score, 4)
        
        fused_results.append(result)
    
    return fused_results


def _generate_doc_id(result: dict[str, Any], doc_id_fn: Callable[[dict], str] | None = None) -> str:
    """
    Generate a unique document ID for deduplication.
    
    Args:
        result: Result dictionary
        doc_id_fn: Optional custom function to extract document ID
        
    Returns:
        Document ID string for deduplication
    """
    
    # Use custom doc_id_fn if provided
    if doc_id_fn is not None:
        try:
            return doc_id_fn(result)
        except Exception:
            # Fall back to default logic if custom function fails
            pass
    
    metadata = result.get('metadata', {})
    
    # Try to use existing doc_id from metadata
    if 'doc_id' in metadata:
        return str(metadata['doc_id'])
    
    # Fall back to file path + chunk info if available
    if 'file_path' in metadata:
        file_id = metadata['file_path']
        if 'chunk_index' in metadata:
            file_id += f"_chunk_{metadata['chunk_index']}"
        return file_id
    
    # Last resort: hash the content (using MD5 like chunker)
    content = result.get('content', '')
    content_hash = hashlib.md5(content.encode()).hexdigest()[:12]
    return f"content_hash_{content_hash}"