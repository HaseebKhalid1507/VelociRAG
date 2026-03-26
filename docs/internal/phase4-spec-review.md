# Phase 4 Searcher Spec Review

*by Shady — keeping it real since always*

---

## 💀 Critical Issues That'll Break Your Shit

### 1. **API Mismatch with Store**
Your spec claims `store.search()` takes `(embedding, limit, threshold)`. Reality check:

```python
# What you spec'd:
store.search(embedding, limit, threshold)

# What store.py actually has:
store.search(query: str | np.ndarray, limit: int, min_similarity: float)
```

Not only are the parameter names different (`threshold` vs `min_similarity`), but the store accepts BOTH strings and embeddings as the first arg. Your searcher is gonna crash on line 1. **This is a 0/10 — can't even call the dependency correctly.**

### 2. **SearchError Doesn't Exist**
You're throwing `SearchError` all over but never defined it. What, you think Python's just gonna manifest exception classes for you? Add this to your spec or watch your error handling fail to... handle errors.

### 3. **Variant Generation is Vaporware**
```python
from .variants import generate_variants
```

Cool import bro. What does `generate_variants()` return? List of strings? Can it return duplicates? Empty list? None? The spec acts like this is solved but it's a black box. **Undefined behavior = broken code.**

### 4. **Embedder Batch Processing Assumption**
Your spec assumes `embedder.embed(variants)` magically handles batch processing. But looking at how store.py uses it:

```python
# Single text
embedding = self.embedder.embed(content)
if embedding.ndim == 1:
    pass  # Single text
else:
    embedding = embedding[0]  # Batch with single item
```

The embedder might return different shapes! Your searcher needs to handle both cases or it'll crash when someone passes a single variant. **Y'all gonna have shape errors.**

### 5. **Memory Explosion Waiting to Happen**
```python
VARIANT_SEARCH_MULTIPLIER = 2    # Search limit multiplier per variant
MAX_VARIANT_LIMIT = 20          # Cap results per variant
```

So with 8 variants and `limit=50`, each variant searches for 100 results? But then you cap at 20? Make up your mind! Either:
- Remove the multiplier and just use the cap
- Make the cap actually cap (like `min(limit * 2, 20)`)

Right now it's confusing AF and the math doesn't add up.

---

## 🤡 Design Decisions That Make No Sense

### 6. **RRF Score in Two Places**
```python
'similarity': float,      # Original FAISS similarity  
'rrf_score': float       # RRF fusion score
'metadata': {
    'rrf_score': ...     # Also here???
}
```

Why is `rrf_score` both a top-level field AND in metadata? Pick one. This redundancy is begging for inconsistency bugs.

### 7. **Deduplication Strategy MIA**
"Deduplicate by document ID" — what's the document ID? Is it:
- `doc_id` from the store?
- Some hash of content?
- First 50 chars?

The store has `doc_id` but your spec never mentions using it. How you gonna dedupe without knowing what to dedupe by?

### 8. **Stats Structure Handwaving**
You show example stats but never define the schema. What exactly is in `stats['variants']`? This matters for anyone trying to consume these stats. Define your data structures or don't include them.

---

## 🔥 What's Actually Good

- **Single-tier philosophy**: Not overengineering with L0/L1/L2 nonsense. Respect.
- **RRF choice**: Solid fusion algorithm that actually works
- **Reranker slot**: Good separation of concerns for Phase 5
- **Error test cases**: At least you're thinking about errors

---

## Missing Edge Cases

1. **Empty variant generation**: What if `generate_variants()` returns `[]`?
2. **Duplicate variants**: What if variants = ["CS656", "cs656", "CS656"]?
3. **Store rebuild during search**: Store can trigger rebuilds. Your search might get inconsistent results
4. **Partial embedder failure**: What if embedding works for 3/5 variants?
5. **Unicode normalization**: Query "café" vs "café" (different Unicode forms)

---

## The Verdict

**Rating: 4/10** — The architecture is solid but the implementation details are half-baked. This spec reads like you designed the ideal system in your head but didn't check if your dependencies actually work that way.

**Fix these before you write a single line of code:**
1. Match the actual store.py API
2. Define SearchError
3. Specify variant generation contract
4. Handle embedder shape variations
5. Fix the memory math
6. Pick ONE place for rrf_score
7. Define deduplication clearly

You got the vision but fumbled the details. The best architecture in the world doesn't mean shit if you can't call your dependencies correctly. Will the real API please stand up?

---

*"I'm not afraid to take a stand / Your spec's broken, here's my hand"*  
— Shady
