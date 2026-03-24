# Phase 1 Spec Review: Velocirag

*Review by Shady — No filter, just facts*

## Overall Verdict: 💀 4/10 — "Y'all playing telephone with the production code"

This spec reads like someone skimmed the production code while texting. Major behavioral gaps, wrong APIs, test cases that would fail against the actual implementation. Let me break it down issue by issue.

---

## 1. **CRITICAL** — RRF API is Completely Wrong

**Severity:** Critical  
**Issue:** The spec shows a functional API (`reciprocal_rank_fusion()`) but production uses a class-based approach (`RRFEngine`).

```python
# Spec says:
def reciprocal_rank_fusion(results_lists, k=60, doc_id_fn=None)

# Production has:
class RRFEngine:
    def __init__(self, k=60)
    def reciprocal_rank_fusion(self, results_list, k=None)
```

**Why this matters:** Developers will implement the wrong API. The `doc_id_fn` parameter doesn't exist. The k parameter works differently.

**Fix:** Update spec to match production's class-based API. Remove the fictional `doc_id_fn` parameter.

---

## 2. **CRITICAL** — Chunker Parent Header Bug Not Documented

**Severity:** Critical  
**Issue:** Production code has a bug in parent header tracking. For `###` sections, it stores the parent `##` header as `parent_header`, but for `##` sections it stores the `#` header. This is inconsistent and confusing.

```python
# Production line 107:
'parent_header': current_h2 if header_level == 3 else current_h1,
```

**Why this matters:** The spec examples show expected behavior that won't match actual output.

**Fix:** Either document the bug as-is, or fix the production code first. Don't spec fantasy behavior.

---

## 3. **MAJOR** — Variants Missing Core Features

**Severity:** Major  
**Issue:** The spec completely misses underscore and dot handling that exists in production.

Production handles:
- Underscores: `file_name` → `file name`, `filename`
- Dots: `script.py` → `script py`, `scriptpy`

But spec only mentions hyphens.

**Fix:** Add underscore and dot normalization to the spec. Update test case 4 to actually work.

---

## 4. **MAJOR** — No MAX_VARIANTS Enforcement

**Severity:** Major  
**Issue:** Spec claims "max 8 variants" but production code has no such limit. The `MAX_VARIANTS = 8` constant isn't used anywhere.

**Why this matters:** Production could generate way more than 8 variants for complex queries, causing performance issues.

**Fix:** Either remove the limit from spec or implement it in production. Don't lie about what the code does.

---

## 5. **MAJOR** — RRF Missing Core Production Features

**Severity:** Major  
**Issue:** Production RRFEngine has methods not in spec:
- `get_fusion_stats()` — debugging/analytics
- `explain_scores()` — score transparency

**Why this matters:** These are useful features that developers won't know exist.

**Fix:** Document all public methods or explicitly mark them as internal.

---

## 6. **MAJOR** — Exception Handling Too Vague

**Severity:** Major  
**Issue:** Chunker production catches all exceptions for frontmatter (`except Exception:`), but spec just says "falls back to empty dict". What about other exceptions?

**Fix:** Be explicit about exception handling strategy. Either catch specific exceptions or document the catch-all behavior.

---

## 7. **MINOR** — Test Cases Would Fail

**Severity:** Minor  
**Issue:** Multiple test cases show expected output that doesn't match production behavior:

1. Chunker test 2 shows wrong parent_header values
2. Variants test 4 shows more than 8 variants
3. RRF test 2 uses non-existent `doc_id_fn`

**Fix:** Run the actual test cases against production code. Use real output, not imagined output.

---

## 8. **MINOR** — Memory Protection Details Wrong

**Severity:** Minor  
**Issue:** RRF spec says "proportional truncation" but production does equal truncation (`max_per_set = MAX_FUSION_RESULTS // len(valid_sets)`).

**Fix:** Document actual behavior, not ideal behavior.

---

## 9. **MINOR** — Config Import Missing

**Severity:** Minor  
**Issue:** Production RRF imports from `.config` module, but spec doesn't mention this dependency.

**Fix:** List all imports, including internal ones.

---

## 10. **MINOR** — Hash Length Ambiguity

**Severity:** Minor  
**Issue:** Spec says "MD5 hash (first 12 chars)" but MD5 produces hex strings. Do you mean 12 hex chars or 12 bytes? Production uses 12 hex chars.

**Fix:** Be specific: "First 12 hexadecimal characters of MD5 hash"

---

## Behavioral Gaps Summary

Things production does that spec misses:
1. Class-based RRF with instance state
2. Underscore and dot normalization in variants
3. Analytics methods (get_fusion_stats, explain_scores)
4. Equal-split memory truncation, not proportional
5. Catch-all exception handling in chunker
6. No actual MAX_VARIANTS enforcement
7. Content hash fallback uses Python hash(), not MD5

---

## Recommendations

1. **Run the code first** — Half these issues come from not testing the examples
2. **Stop inventing APIs** — Document what exists, not what you wish existed
3. **Be honest about bugs** — The parent_header inconsistency needs addressing
4. **Version the spec** — "Extracted from production on 2024-XX-XX"
5. **Add integration examples** — Show how these modules work together

Look, the core ideas are solid. But this spec is like a cover band that learned the song from someone humming it. You need to actually read the production code, run it, test it, then document what it ACTUALLY does.

The production code is your source of truth. The spec is just the messenger. And right now, the messenger is drunk.

---

*"Mom's spaghetti code needs better documentation" — Shady*

---

## Update: Found More Issues After Digging

Just checked the config module. Get this:

**11. MAJOR — Unused Constants**
The config.py has `MAX_VARIANTS_GENERATED = 8` but the variants generator never imports or uses it. So the limit exists... in a different file... unused. 🤡

**12. MAJOR — Wrong Constant Names**
Spec says `MAX_VARIANTS = 8` but config says `MAX_VARIANTS_GENERATED = 8`. Can't even get the constant names consistent.

This is what happens when you spec from memory instead of reading the code.
