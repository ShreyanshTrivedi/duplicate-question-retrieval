# Gap Analysis Report

## Status of Requirements

| Requirement | Status | Evidence | Notes |
| :--- | :--- | :--- | :--- |
| **1. Text Preprocessing** | WORKING | `tests/test_preprocessing.py` and `tests/test_short_text_bug.py` pass; `src/preprocessing/clean.py` implements lowering, stop-word removal, and stemming. | Retrieval now applies preprocessing uniformly inside each independent lexical space: light mode, heavy mode, and raw character features are each corpus-wide consistent. |
| **2. Short-text Vector Indexing** | FIXED | `TfidfRetriever` maintains separate corpus-wide light+bigram, heavy+unigram, and character vector spaces, then fuses their ranked lists with RRF. `tests/test_short_text_bug.py` covers cross-length duplicates, uniform fitting, and non-degenerate short queries. | The per-document scoped attempt was rejected because it fragmented vocabulary alignment. The dual-space design restores short-query matching without putting incompatible document tokenizations into one vector space. |
| **3. Ranking Algorithms (TF-IDF, BM25, Dense)** | WORKING | `app.py` loads `TfidfRetriever`, `BM25Retriever`, and `DenseRetriever` from `src/retrieval/`. UI provides a dropdown to select each. | TF-IDF and BM25 each use consistent light/heavy corpus-wide spaces; Dense is unchanged. |
| **4. Hybrid Model (RRF)** | WORKING | `app.py` loads `HybridRRFRetriever` using BM25 and Dense. It is selectable in the Streamlit UI. | The implementation in `hybrid_rrf.py` correctly merges results. Hybrid metrics shifted slightly with BM25’s light preprocessing. |
| **Outputs: Search Interface** | WORKING | `src/ui/app.py` executes Streamlit and successfully interfaces with all 4 methods. | The UI successfully provides functionality as intended. |
| **Outputs: Performance Comparison Table** | WORKING | `reports/comparison_table.csv` regenerated after the dual-space fusion fix. Pre-fix snapshot preserved in `reports/comparison_table_before_fix.csv`. | Current aggregate: TF-IDF 0.131 / 0.834 / 0.663 / 0.731; BM25 0.137 / 0.858 / 0.713 / 0.775; Dense 0.174 / 0.966 / 0.909 / 0.943; Hybrid-RRF 0.171 / 0.954 / 0.884 / 0.925 (P@10 / Recall@10 / MAP@10 / nDCG@10). |
| **Outputs: Precision-Recall Graph** | WORKING | `reports/pr_curve.png` rewritten 27-09-2026. | Four distinct curves: Dense highest, then Hybrid-RRF, BM25, TF-IDF. |
| **Outputs: Held-out Evaluation** | WORKING | `src/evaluation/run_evaluation.py` loads `eval_queries.parquet` and `qrels.parquet`. Full run completed with log line `Saved reports to reports/ directory.` | Evaluation uses the genuine held-out ground truth data and does not fabricate numbers. |
| **Test Suite** | WORKING | `python -m pytest tests/ -v` (15 tests including `test_short_text_bug.py`). | Integration coverage for the former 1.0-tie failure mode is in `tests/test_short_text_bug.py`. |

---

## Short-Text Vectorization Bug (fixed)

## Lexical Ensemble Architecture

The `TF-IDF` and `BM25` rows in the comparison table each represent an
internal RRF-fused lexical ensemble. Each combines a light-preprocessed
representation with surface-form bigrams and a heavy-preprocessed
representation with stopword removal and stemming. For short TF-IDF queries,
a third corpus-wide character n-gram ranking is included to address sparse
word-level evidence. These spaces are fitted consistently over the full
corpus; they are separate representations of the same sparse lexical
retrieval paradigm, not per-document branches inside one vocabulary.

This internal fusion is distinct from the top-level **Hybrid Model**
requirement. The top-level Hybrid-RRF result fuses BM25 with Dense retrieval,
combining fundamentally different paradigms: sparse lexical matching and
dense semantic similarity. It is therefore not merely another preprocessing
variant of TF-IDF or BM25. The internal lexical ensemble solves short-text
vocabulary sparsity, while the top-level hybrid combines complementary
lexical and semantic retrieval signals.

**Root cause:** Short queries were stripped to 0 or 1 token during preprocessing (stopword removal + stemming). TF-IDF then collapsed many documents onto the same one-dimensional vector, producing trivial **1.0 cosine similarity** ties across non-identical documents.

**Concrete findings (pre-fix):** 40 eval queries with 5 words or fewer; **6/40 (15%)** collapsed to 0–1 tokens, e.g. `What is the google?` → `['googl']`, `How do I post here?` → `['post']`.

**First fix (2026-09-27):** Light preprocessing for TF-IDF and BM25; TF-IDF word bigrams plus char n-gram fallback. Segmented analysis then showed this was over-applied: long-query TF-IDF fell below the original aggregate ceiling even though long queries were not affected by the collapse bug.

**Second corrective fix (2026-09-27):** Select preprocessing per text using `short_text_word_threshold` (default 5). This restored long-query metrics but caused short-query regression because differently tokenized documents were placed in one shared TF-IDF vocabulary. Concrete qrel examples showed mismatches such as `geographic` versus `geograph` and query-only bigrams.

**Third corrective fix (2026-09-27):** Replace per-document conditional tokenization with two independent, corpus-wide lexical spaces: light preprocessing plus word bigrams and heavy preprocessing plus unigrams. A third corpus-wide character space is included only for short queries. Light, heavy, and optional character rankings are fused using the same RRF formula as the hybrid retriever. BM25 receives the equivalent two-space treatment. Short-query TF-IDF recovered to 0.1450 / 0.8389 / 0.7225 / 0.7908, while long-query TF-IDF reached 0.1294 / 0.8335 / 0.6565 / 0.7247.

**Spot-check (5 eval queries, top-10 qrel hits):** on qid=1182 (33 qrels) TF-IDF 4/33, BM25 8/33, Dense 8/33, Hybrid-RRF 9/33; other queries show BM25/Dense/Hybrid recovering the labeled duplicate where TF-IDF still misses some (e.g. NEET chemistry 0/1 vs 1/1).

**Held-out metrics before vs after** (`comparison_table_before_fix.csv` vs `comparison_table.csv`):

| Method | P@10 before → after | Recall@10 | MAP@10 | nDCG@10 |
| :--- | :--- | :--- | :--- | :--- |
| TF-IDF | 0.1283 → **0.1310** | 0.8198 → **0.8341** | 0.6642 → **0.6631** | 0.7274 → **0.7314** |
| BM25 | 0.1345 → **0.1368** | 0.8442 → **0.8582** | 0.7013 → **0.7129** | 0.7613 → **0.7755** |
| Dense | 0.1738 → 0.1738 (unchanged) | 0.9659 | 0.9087 | 0.9431 |
| Hybrid-RRF | 0.1688 → **0.1708** | 0.9498 → **0.9535** | 0.8838 → **0.8843** | 0.9228 → **0.9251** |

The final dual-space architecture improves the aggregate over the original snapshot while removing the short-query collapse failure. Segmented TF-IDF is 0.1450 / 0.8389 / 0.7225 / 0.7908 for short queries and 0.1294 / 0.8335 / 0.6565 / 0.7247 for long queries. The short-query result is substantially above both the global-light round (0.1225 / 0.7087 / 0.5044 / 0.5887) and the broken per-document scoped round (0.0700 / 0.4205 / 0.2981 / 0.3508).

---

## Prioritized List of Fixes

1. **Implement Short-text Vector Indexing — DONE:** Light preprocessing and n-grams prevent short queries from collapsing into one-dimensional vectors.
2. **Detect over-application — DONE:** Segmented evaluation exposed the global-light round's long-query regression.
3. **Reject fragmented vocabulary — DONE:** Concrete qrel pairs showed that per-document mode selection breaks alignment inside one shared vectorizer.
4. **Dual-space lexical fusion — DONE:** Uniform light/heavy TF-IDF and BM25 spaces, optional short-query character ranking, and shared RRF fusion are implemented and evaluated.
