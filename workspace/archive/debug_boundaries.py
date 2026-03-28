#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/haseeb/velocirag/src')

from velocirag.semantic_chunker import (
    split_sentences,
    calculate_boundary_scores, 
    find_semantic_boundaries
)
from unittest.mock import Mock
import numpy as np

# Test the boundary detection with the same embeddings as the failing test
embeddings = np.array([
    [1.0, 0.0],  # All sentences very similar
    [0.95, 0.05],
    [0.9, 0.1],
    [0.92, 0.08],
    [0.88, 0.12],
    [0.91, 0.09],
])

mock_embedder = Mock()
mock_embedder.embed.return_value = embeddings

# Dummy sentences (doesn't matter for the calculation)
sentences = [f"Sentence {i}" for i in range(len(embeddings))]

print("=== Testing boundary detection ===")
print(f"Input embeddings shape: {embeddings.shape}")
print("Embeddings:")
for i, emb in enumerate(embeddings):
    print(f"  {i}: {emb}")

print("\n=== Calculate similarities ===")
similarities = calculate_boundary_scores(sentences, mock_embedder)
print(f"Similarities: {similarities}")
print(f"Similarities length: {len(similarities)}")

if similarities:
    print(f"Min similarity: {min(similarities):.3f}")
    print(f"Max similarity: {max(similarities):.3f}")
    print(f"Mean similarity: {np.mean(similarities):.3f}")
    print(f"Std similarity: {np.std(similarities):.3f}")

    print("\n=== Find boundaries ===")
    boundaries = find_semantic_boundaries(similarities, method='percentile', threshold=25.0)
    print(f"Boundaries: {boundaries}")
    print(f"Number of splits: {len(boundaries) - 2}")  # Subtract start and end
    
    # Check what percentile threshold would be
    threshold_value = np.percentile(similarities, 25.0)
    print(f"25th percentile threshold: {threshold_value:.3f}")
    
    # Show which similarities are below threshold
    for i, sim in enumerate(similarities):
        below = sim < threshold_value
        print(f"  Similarity {i}->{i+1}: {sim:.3f} {'← SPLIT' if below else ''}")