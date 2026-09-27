# Project Context (read this first, every session)

## What this is
Duplicate Question Retrieval on Quora Question Pairs. Reframed from pairwise
classification into a retrieval task: corpus of unique questions + qrels
built from is_duplicate=1 pairs. Methods: TF-IDF, BM25, dense (sentence
transformers), RRF hybrid fusion. Evaluated with P@10, Recall, MAP, nDCG.

## Current state
- Phase: 8 (final report)
- Last completed: Third corrective fix using dual corpus-wide light/heavy
  lexical spaces with RRF fusion, plus a corpus-wide character space for short
  TF-IDF queries. The prior per-document scoped attempt fragmented vocabulary
  alignment and badly regressed short-query retrieval. Fresh full-evaluation
  numbers are in `reports/comparison_table.csv`; segmented results are in
  `reports/segmented_metrics.csv`; the original snapshot remains in
  `comparison_table_before_fix.csv`. Final segmented TF-IDF is 0.1450 /
  0.8389 / 0.7225 / 0.7908 for short queries and 0.1294 / 0.8335 / 0.6565 /
  0.7247 for long queries.
- The Streamlit UI now exposes all required outputs in-app: Search,
  Performance Comparison, and Precision-Recall Curve tabs. All four retrieval
  methods were live-tested after the dual-space rework.
- Next task: Phase 8 — README + final report

## Key decisions
(see DECISIONS.md; history is original short-text bug -> global light-mode
fix -> segmentation-detected overcorrection -> per-document scoped false
start -> concrete vocabulary diagnosis -> dual-space RRF fusion)
