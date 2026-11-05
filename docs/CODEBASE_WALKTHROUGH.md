# Codebase Walkthrough — Duplicate Question Retrieval (QQP)

Personal study notes for the viva/demo. Prioritises correctness over polish. Every claim is grounded in the code or in the existing docs; where the code and `DECISIONS.md` / `reports/gap_analysis.md` disagree, the disagreement is flagged in §3.7 instead of resolved silently.

**What I did and did not do**
- Read every file under `src/` (data, preprocessing, retrieval, evaluation, ui in full; reporting: the methodology section, the references and `main()` of `build_report.py`, and the header and `main()` of `build_slides.py`, but not their experiment/conclusion/slide-builder bodies; `theme.css`: the top of the file and its selector list only; empty `__init__.py` files confirmed empty).
- Read `DECISIONS.md`, `CONTEXT.md`, `CONVENTIONS.md`, `TASKS.md`, `config.yaml`, `reports/gap_analysis.md`, `requirements.txt`, and the tests that exercise retrieval.
- Inspected the data artifacts (row counts, column names, embedding shape) with a read-only pandas/numpy check.
- **Did not** launch Streamlit, run `run_evaluation.py`, run `pytest`, or sleep/poll. Timings in §5 come from the timestamped `.err.log` of the most recent full run (`reports/ui_redesign_evaluation.err.log`, 2026-09-27), not from new runs.

---

## 1. Project structure

```
duplicate-question-retrieval/
├── config.yaml                  Single source of truth: paths, preprocessing flags, top_k, rrf_k, dense model, corpus_subsample
├── requirements.txt             Python deps (python-pptx/python-docx were added, uncommitted)
├── CONTEXT.md                   Project state summary ("read first"); current phase, last completed work
├── DECISIONS.md                 Chronological decision log, incl. the short-text bug and its three corrective fixes
├── CONVENTIONS.md               Short coding conventions (3 lines)
├── TASKS.md                     Phase checklist (Phase 8 = README + final report)
├── .streamlit/config.toml       Dark theme colours + minimal toolbar for Streamlit
│
├── src/
│   ├── data/
│   │   ├── download.py          Fetches the QQP TSV from Quora's CDN, saves it as data/raw/questions.csv (skips if present)
│   │   └── build_corpus.py      raw CSV -> corpus.parquet, qrels.parquet, eval_queries.parquet (dedupe, bidirectional qrels, seeded 400-query hold-out)
│   ├── preprocessing/
│   │   └── clean.py             Lowercase -> strip punctuation -> word_tokenize -> (heavy only) stopword removal + Porter stem; is_short_text()
│   ├── retrieval/
│   │   ├── base.py              RetrieverBase ABC: rank(query, top_k) -> [(doc_id, score)]
│   │   ├── tfidf.py             TfidfRetriever: 3 corpus-wide spaces (light+bigrams, heavy stems, char_wb 3-5) fused by RRF
│   │   ├── bm25.py              BM25Retriever: 2 corpus-wide spaces (light, heavy) fused by RRF
│   │   ├── dense.py             DenseRetriever: all-MiniLM-L6-v2 embeddings + FAISS IndexFlatIP; optional corpus subsample + cache
│   │   ├── hybrid_rrf.py        reciprocal_rank_fusion() (shared by all) + HybridRRFRetriever (BM25 + Dense)
│   │   ├── inverted_index.py    From-scratch TF-IDF-style inverted index. NOT used by the app or evaluation (only by one test)
│   │   └── __init__.py          empty
│   ├── evaluation/
│   │   ├── metrics.py           precision_at_k, recall_at_k, average_precision, ndcg_at_k (pure functions)
│   │   ├── run_evaluation.py    Main 400-query eval: 4 methods -> reports/comparison_table.csv, PR curve data/png, spot-check dump
│   │   ├── run_segmented_evaluation.py   TF-IDF and BM25 split into short (<=5 words) / long queries -> reports/segmented_metrics.csv
│   │   └── __init__.py          empty
│   ├── ui/
│   │   ├── app.py               Streamlit app: Search / Performance Comparison / Precision-Recall tabs
│   │   └── theme.css            Dark theme CSS injected via st.markdown (hero, cards, pill tabs, fonts from Google Fonts)
│   └── reporting/               (UNTRACKED in git) Builds the course deliverables from reports/
│       ├── build_report.py      reports/IR_project8_report.docx, using docs/IR_report formatt.docx as the style template
│       └── build_slides.py      reports/project8_presentation.pptx (13.33x7.5in deck, no invented figures)
│
├── data/
│   ├── raw/questions.csv        (gitignored) 58 MB — the QQP file, saved as CSV
│   └── processed/               (gitignored) corpus.parquet (537,929 rows), qrels.parquet (298,526 rows),
│                                eval_queries.parquet (400 rows), embeddings.npy (100,000 x 384), embeddings_ids.npy
│
├── reports/                     Outputs. Tracked: comparison_table(.csv, _before_fix.csv), segmented_metrics.csv,
│                                pr_curve.png, pr_curve_data.json, gap_analysis.md, spot_check_rankings.md, *.log
│                                UNTRACKED: IR_project8_report.docx, project8_presentation.pptx, project_summary_for_writeup.md
│                                Also: _e2e_spot5.py ("temporary e2e spot-check; not part of the product")
├── tests/                       pytest suite (27 test functions by grep; see §3.7 D8)
├── docs/                        IR_report formatt.docx (report template); this file
├── check_bm25.py, check_bug.py, verify_fix.py   Ad-hoc diagnostic scripts from the bug hunt. Not part of the pipeline.
├── notebooks/                   Empty directory
├── .qodo/                       Tool config (agents/workflows). Not project code.
└── .venv/                       Local Windows venv, built from Python 3.13.7 (per pyvenv.cfg)
```

Notes:
- `src/` has no `__init__.py` in `data/`, `preprocessing/`, `ui/`, or top-level `src/`. Imports like `from src.retrieval.tfidf import ...` still work because Python 3 treats them as namespace packages. Run everything from the repo root.
- Every path in `config.yaml` and in the scripts is **relative** (`config.yaml`, `reports/...`). Running from any other directory will fail.

---

## 2. Data flow, end to end

### 2a. A query: from the search box to ranked results

There are two phases: a one-off **startup** (cached for the server process) and a **per-query** path.

**Startup (once per server process)**

1. `streamlit run src/ui/app.py` → `main()` at [app.py:218](../src/ui/app.py#L218). Streamlit re-runs the whole script on every interaction; the expensive part is guarded by `@st.cache_resource`.
2. `load_retrievers()` at [app.py:188](../src/ui/app.py#L188):
   - `load_config()` reads `config.yaml`.
   - `pd.read_parquet(corpus.parquet)` → 537,929 rows of `(qid, question)`.
   - `TfidfRetriever(config).fit(corpus_df)` → [tfidf.py:54](../src/retrieval/tfidf.py#L54). Builds three sparse matrices (light, heavy, char).
   - `BM25Retriever(config).fit(corpus_df)` → [bm25.py:21](../src/retrieval/bm25.py#L21). Two `BM25Okapi` indexes.
   - `DenseRetriever(config)` → [dense.py:13](../src/retrieval/dense.py#L13) loads the SentenceTransformer model. Then `.fit(corpus_df)` at [dense.py:86](../src/retrieval/dense.py#L86): `_select_corpus` subsamples to 100k (see §3.4), `_load_cached_embeddings` loads `embeddings.npy` if the cached ids match the current subsample exactly, otherwise it encodes and saves. Then a `faiss.IndexFlatIP` is built.
   - `HybridRRFRetriever([bm25, dense], config)` → [hybrid_rrf.py:18](../src/retrieval/hybrid_rrf.py#L18). It does **not** refit; it reuses the already-fitted BM25 and Dense objects.
3. Returns `{"TF-IDF", "BM25", "Dense", "Hybrid-RRF"}` plus the corpus DataFrame.

**Per query**

4. The user presses Enter in `st.text_input` at [app.py:255](../src/ui/app.py#L255) (Streamlit commits text on Enter/blur, not per keystroke). `st.selectbox` at [app.py:257](../src/ui/app.py#L257) picks the method. `top_k = 10` comes from `config.retrieval.top_k`.
5. `retriever.rank(query, top_k)` at [app.py:262](../src/ui/app.py#L262). The branch depends on the method:

   - **TF-IDF** — [`TfidfRetriever.rank`](../src/retrieval/tfidf.py#L71):
     1. `pool_size = max(100, top_k)` = 100.
     2. `_rank_scores(light_vectorizer, light_matrix, …)` → `vectorizer.transform([query])` calls the callable analyzer `_light_features` ([tfidf.py:40](../src/retrieval/tfidf.py#L40)) → `_preprocess_mode(text, True)` → [`clean.preprocess`](../src/preprocessing/clean.py#L43) (no stopwords, no stem) → tokens + `a__b` bigrams.
     3. `cosine_similarity(query_vec, matrix)` against all 537,929 rows, `argsort` descending, keep the first 100 with score > 0 and text ≠ query text.
     4. Same for the heavy space (`_heavy_features` → `preprocess` with stopwords removed and stemming on).
     5. If `is_short_text(query)` (≤5 whitespace words, [clean.py:25](../src/preprocessing/clean.py#L25)), a third list from the char_wb(3,5) space.
     6. `reciprocal_rank_fusion(lists, top_k, rrf_k=60)` ([hybrid_rrf.py:5](../src/retrieval/hybrid_rrf.py#L5)) → top 10 `(qid, rrf_score)`.

   - **BM25** — [`BM25Retriever.rank`](../src/retrieval/bm25.py#L38): query tokenised twice (light, heavy), `BM25Okapi.get_scores(tokens)` on each index, argsort, keep top 100 with score > 0 and text ≠ query, RRF → top 10.

   - **Dense** — [`DenseRetriever.rank`](../src/retrieval/dense.py#L107): `model.encode([query], normalize_embeddings=True)` → `index.search(q, top_k+5)` → skip -1 indices and same-text docs → first 10 `(qid, cosine)`.

   - **Hybrid-RRF** — [`HybridRRFRetriever.rank`](../src/retrieval/hybrid_rrf.py#L33): `bm25.rank(q, 100)` and `dense.rank(q, 100)` (each returns its own already-fused list), then RRF over those two lists → top 10.

6. `build_results_df(results, id_to_question)` at [app.py:36](../src/ui/app.py#L36). `id_to_question` is rebuilt from the full corpus on every rerun ([app.py:248](../src/ui/app.py#L248)); it is not cached.
7. `style_results` ([app.py:64](../src/ui/app.py#L64)) tints rank 1; `st.dataframe` shows `Score` as a `ProgressColumn` scaled min→max within that result set ([app.py:274](../src/ui/app.py#L274)).

### 2b. Corpus CSV → `corpus.parquet` + `qrels.parquet` (+ `eval_queries.parquet`)

1. [`download.download_data()`](../src/data/download.py#L25) reads `config.data.raw_path`. If the file exists it returns immediately. Otherwise it does `pd.read_csv(QQP_URL, sep="\t")` and writes `data/raw/questions.csv`. Despite the `.csv` name, the source is the QQP TSV. Columns: `id, qid1, qid2, question1, question2, is_duplicate`.
2. [`build_corpus.main()`](../src/data/build_corpus.py#L59) reads that CSV, then calls `process_data(df, eval_query_count=400, seed=42)` at [build_corpus.py:24](../src/data/build_corpus.py#L24):
   - `dropna` on the four required columns.
   - **Corpus:** stack `(qid1, question1)` and `(qid2, question2)`, then `drop_duplicates(subset=qid)`. → 537,929 unique questions.
   - **Qrels:** rows with `is_duplicate == 1`, emitted in both directions (`q1→q2` and `q2→q1`), then `drop_duplicates`. → 298,526 rows.
   - **Hold-out:** `np.random.seed(42)`, then `np.random.choice(unique query_ids that have qrels, size=400, replace=False)`. Eval queries are the corpus rows for those qids. Note: the eval queries **stay in the corpus** (see §3.7, D2).
3. `to_parquet` for corpus, qrels, and eval_queries ([build_corpus.py:78-80](../src/data/build_corpus.py#L78)).

### 2c. Evaluation (for completeness)

[`run_evaluation.run_evaluation()`](../src/evaluation/run_evaluation.py#L21) loads the three parquets and builds `qrels_dict` (qid → set of doc ids). It fits TF-IDF, BM25, and Dense, creates Hybrid from the fitted BM25 and Dense, and writes the spot-check file. Then, for each of the 400 queries and each method, it calls `rank(q, top_k=50)` and computes P/R/MAP/nDCG@10 plus P/R at k=1..50 for the PR curve. Outputs: `reports/comparison_table.csv`, `reports/pr_curve_data.json`, `reports/pr_curve.png`, `reports/spot_check_rankings.md`.

---

## 3. Per-module deep dive

### 3.1 `src/data/`

**[download.py](../src/data/download.py)**
- `QQP_URL` is hardcoded to `qim.fs.quoracdn.net`. I did not test that URL, so I can't confirm it still resolves.
- `download_data()` is idempotent (skips if the raw file exists). It sets the User-Agent through `storage_options`.

**[build_corpus.py](../src/data/build_corpus.py)**
- `process_data(df, eval_query_count, seed)` is pure (DataFrame in, three DataFrames out), so it's unit-tested in isolation ([tests/test_build_corpus.py](../tests/test_build_corpus.py)).
- **Design decision: reframing.** QQP is a pairwise classification dataset. This project turns it into retrieval: the corpus is every unique question, and the relevant set for a question is its labelled duplicates. Only `is_duplicate == 1` pairs make qrels. Non-duplicate pairs are not negatives; they simply don't appear. This means a retriever that returns a true-but-unlabelled duplicate is penalised.
- **Design decision: bidirectional qrels.** Duplication is symmetric, so both directions are added. A query can therefore hit its partner from either side.
- The 400 hold-out queries are drawn only from questions that have at least one qrel, so every eval query is scoreable.

### 3.2 `src/preprocessing/clean.py`

This is the single tokenizer for everything lexical.

- **Import-time side effects:** downloads NLTK `punkt`, `punkt_tab`, and `stopwords` on first import if missing. Needs internet on first run.
- `is_short_text(text, config)` ([clean.py:25](../src/preprocessing/clean.py#L25)): `len(text.split()) <= short_text_word_threshold` (default 5). Whitespace split, so punctuation counts toward length.
- `preprocess(text, config)` ([clean.py:43](../src/preprocessing/clean.py#L43)), in order:
  1. Lowercase (if `lowercase`).
  2. Replace every `string.punctuation` character with a space. This is done **before** tokenising, so `don't` becomes `don t`, and `word_tokenize` only sees words.
  3. `word_tokenize` (NLTK, sentence-aware Punkt-based).
  4. If `remove_stopwords` and not `short_text_mode`: drop NLTK English stopwords.
  5. If `stem` and not `short_text_mode`: Porter stemmer.
- The `short_text_mode` flag is passed in via `config["preprocessing"]["short_text_mode"]`. Each retriever sets that flag itself (see §3.3); `preprocess` never decides it.
- Non-obvious consequence: heavy preprocessing can produce an **empty** token list for a query made only of stopwords (e.g. "Who is this?"). The heavy space then returns nothing, and only the light space contributes.
- Dead code: `preprocess_for_retrieval()` ([clean.py:33](../src/preprocessing/clean.py#L33)) chooses light or heavy per text by length. Nothing in the app or evaluation calls it. It is the per-text design that DECISIONS.md rejected.

### 3.3 `src/retrieval/`

**Common interface — [base.py](../src/retrieval/base.py)**
`RetrieverBase.rank(query, top_k) -> list[tuple[str, float]]`. Every retriever returns `(qid as str, score)` best-first. Scores are **not comparable across methods**: RRF sums for TF-IDF/BM25/Hybrid, cosine for Dense.

**Shared fusion — [hybrid_rrf.py:5 `reciprocal_rank_fusion`](../src/retrieval/hybrid_rrf.py#L5)**
- `score(d) = Σ over lists m of 1 / (rrf_k + rank_m(d))`, with 1-based ranks and `rrf_k = 60` from config. A doc absent from a list contributes nothing.
- Only **positions** matter. The original scores are discarded. That is the point: BM25 is unbounded, cosine is in [-1, 1], and the two can't be summed meaningfully.
- The same function is used at three places: inside TF-IDF ([tfidf.py:80](../src/retrieval/tfidf.py#L80)), inside BM25 ([bm25.py:45](../src/retrieval/bm25.py#L45)), and at the top level in `HybridRRFRetriever` ([hybrid_rrf.py:47](../src/retrieval/hybrid_rrf.py#L47)).

**[tfidf.py](../src/retrieval/tfidf.py) — `TfidfRetriever`** (three corpus-wide spaces)
- Construction ([tfidf.py:9](../src/retrieval/tfidf.py#L9)):
  - `light_vectorizer = TfidfVectorizer(analyzer=self._light_features, token_pattern=None)`.
  - `heavy_vectorizer = TfidfVectorizer(analyzer=self._heavy_features, token_pattern=None)`.
  - `char_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5))`.
  - Because `analyzer` is a callable, sklearn skips its own lowercase, token pattern, and n-gram logic. Everything comes from `preprocess`.
- `_light_features` ([tfidf.py:40](../src/retrieval/tfidf.py#L40)): `preprocess` with `short_text_mode=True` (no stopwords, no stemming), plus adjacent bigrams joined by `__`. This is the "surface forms plus phrase" space.
- `_heavy_features` ([tfidf.py:46](../src/retrieval/tfidf.py#L46)): `preprocess` with stopwords removed and stemming on. Unigrams only.
- `fit(corpus_df)` ([tfidf.py:54](../src/retrieval/tfidf.py#L54)): fits all three vectorizers on the **full** corpus. Each space is one matrix, so every document is tokenised the same way.
- `_rank_scores` ([tfidf.py:61](../src/retrieval/tfidf.py#L61)): cosine of the query vector against the matrix, descending argsort, top `pool_size`, filtered to `score > 0` and text ≠ query text.
- `rank` ([tfidf.py:71](../src/retrieval/tfidf.py#L71)): light and heavy lists always; char list only when `is_short_text(query)`. RRF over however many lists there are.
- **Why three spaces** (the viva question): see §4, Q2. Short version: light keeps the surface form that short queries need, heavy keeps the normalisation that long queries benefit from, and char n-grams give sub-word evidence when a 3–5-word query has very few word matches. One shared vocabulary cannot serve all three because per-text mode selection fragments it.
- `preprocess_text(text)` ([tfidf.py:29](../src/retrieval/tfidf.py#L29)) and `_char_text` ([tfidf.py:50](../src/retrieval/tfidf.py#L50)): `preprocess_text` is only called by tests; `_char_text` is never called.

**[bm25.py](../src/retrieval/bm25.py) — `BM25Retriever`** (two corpus-wide spaces)
- `fit` ([bm25.py:21](../src/retrieval/bm25.py#L21)): preprocesses every document twice (light and heavy) and builds two `rank_bm25.BM25Okapi` objects.
- `_rank_space` ([bm25.py:29](../src/retrieval/bm25.py#L29)): `model.get_scores(tokens)`, descending argsort, top `pool_size`, filtered to `score > 0` and text ≠ query.
- `rank` ([bm25.py:38](../src/retrieval/bm25.py#L38)): light and heavy lists, always RRF-fused. It has **no** char space and **no** length gating; both spaces are used for every query.
- Note: `BM25Okapi` is a Python-level loop over every document for each query token (rank_bm25 implementation detail). That is the reason for the per-query latency in §5.

**[dense.py](../src/retrieval/dense.py) — `DenseRetriever`** (`all-MiniLM-L6-v2` + FAISS)
- `__init__` ([dense.py:13](../src/retrieval/dense.py#L13)): loads the SentenceTransformer by name from config. No lexical preprocessing; the raw question is encoded.
- `_select_corpus` ([dense.py:50](../src/retrieval/dense.py#L50)): if `corpus_subsample` (100,000) is less than the corpus, it keeps every qid in `eval_queries` plus every qrel doc of those queries (`_must_keep_qids`, [dense.py:31](../src/retrieval/dense.py#L31)), and fills the rest with `sample(random_state=42)`. Logged: "1276 forced-in" in the latest run (400 queries + 876 qrel docs). **Consequence: the dense index is a 100k subset, the lexical indexes are 538k. See §3.7, D1.**
- `_load_cached_embeddings` ([dense.py:74](../src/retrieval/dense.py#L74)): loads `embeddings.npy` and `embeddings_ids.npy` only if the cached id list is **exactly** equal to the current subsample's id list and the row count matches. Otherwise it recomputes and overwrites. This is a safe cache key.
- `fit` ([dense.py:86](../src/retrieval/dense.py#L86)): encodes with `normalize_embeddings=True` (unit vectors), then `faiss.IndexFlatIP(384)`. For unit vectors, inner product equals cosine similarity. `IndexFlatIP` is exact brute force, not approximate.
- `rank` ([dense.py:107](../src/retrieval/dense.py#L107)): encodes the query the same way, fetches `top_k + 5` so the self-match can be dropped without going short, skips `-1` (FAISS padding) and same-text docs.

**[hybrid_rrf.py](../src/retrieval/hybrid_rrf.py) — `HybridRRFRetriever`**
- Takes a list of already-constructed retrievers. Its `fit` calls each one's `fit`, but the app and evaluation fit the BM25 and Dense instances themselves and do not call `hybrid.fit`, so nothing is fitted twice.
- `rank` ([hybrid_rrf.py:33](../src/retrieval/hybrid_rrf.py#L33)): asks each child for 100 results, then RRF over those lists.
- **Important:** the BM25 child is the **internal** BM25 ensemble (two spaces, already RRF-fused). So the top-level Hybrid is RRF over RRF output: BM25-ensemble ranks plus Dense ranks.

**[inverted_index.py](../src/retrieval/inverted_index.py) — `InvertedIndexRetriever`** (dead code for the product)
- A from-scratch postings index with a `log(N/df)` IDF weighting. Only [tests/test_retrievers.py](../tests/test_retrievers.py) imports it, as a smoke-test participant. It is not used by `TfidfRetriever`, BM25, the UI, or the evaluation. Don't present it as part of the system.

### 3.4 `src/evaluation/`

**[metrics.py](../src/evaluation/metrics.py)** (pure functions; each takes `ranked_list: list[str]` and `relevant_docs: set[str]`)
- `precision_at_k`: hits in top k divided by k.
- `recall_at_k`: hits in top k divided by `|relevant|`.
- `average_precision(ranked, relevant)`: sum of precision at each hit position, divided by **`|relevant|` (all relevant docs)**, not by `min(|relevant|, k)`. The evaluation passes `ranked_list[:10]`, so this is **truncated AP@10 normalised by the full qrel count**. That is a choice, and it is lower than the `min(k, |R|)` convention for queries with more than 10 relevant docs. The eval set averages 2.19 relevant docs per query, so the difference is small here.
- `ndcg_at_k`: binary-relevance DCG divided by the ideal DCG (`min(k, |relevant|)` ones at the top).

**[run_evaluation.py](../src/evaluation/run_evaluation.py)**
- Builds `qrels_dict` from the qrels parquet and fits TF-IDF, BM25, and Dense on the **same** corpus DataFrame. Hybrid reuses the fitted BM25 and Dense.
- `_write_spot_check` ([run_evaluation.py:147](../src/evaluation/run_evaluation.py#L147)): runs five hard-coded qids (`1182, 1353, 4119, 5569, 9495`) through every method and writes human-readable rankings to `reports/spot_check_rankings.md`. It asserts no duplicate ids and that the query's own qid is never returned.
- Main loop ([run_evaluation.py:67](../src/evaluation/run_evaluation.py#L67)): for each method and each eval query, `rank(q_text, top_k=50)`. The first 10 feed P@10/R@10/MAP@10/nDCG@10. The first 50 feed the PR curve (k = 1…50, averaged over queries). Writes `comparison_table.csv`, `pr_curve_data.json`, `pr_curve.png`.
- Note: the eval loop uses `valid_queries` rather than `len(queries)` for averaging. Every eval query has qrels, so the two are equal here.

**[run_segmented_evaluation.py](../src/evaluation/run_segmented_evaluation.py)**
- Splits the 400 queries by `str.split().str.len() <= 5` (same rule as `is_short_text`), then runs TF-IDF and BM25 on each group. Writes `reports/segmented_metrics.csv`. Dense and Hybrid are **not** segmented.
- This is the instrument that exposed the over-application in the first fix (DECISIONS.md, 2026-09-27).

### 3.5 `src/ui/app.py` and `theme.css`

- `MODEL_COLORS` and `METHOD_DESCRIPTIONS` are display constants. The palette is hard-coded here and repeated in `theme.css` and `.streamlit/config.toml`.
- Pure helpers, so they are unit-tested in [tests/test_ui_helpers.py](../tests/test_ui_helpers.py):
  - `build_results_df` → DataFrame with `Rank, Question, Score (float), QID`. Missing text becomes `"N/A"`.
  - `style_results` → highlight rank 1 with the method's accent at 12% alpha.
  - `best_per_metric` → `(method, value)` argmax per metric column.
  - `summary_sentence` → names the method with the most metric wins, or reports a tie.
  - `build_pr_figure` → Plotly figure, one trace per method, k=10 point has a larger marker (index 9 of k=1…50).
- Three tabs:
  1. **Search**: described in §2a. The Hybrid path depends on the Dense index, so Dense's 100k subsample affects Hybrid results too.
  2. **Performance Comparison**: reads `reports/comparison_table.csv` and shows the best per metric. If the file is missing, it shows a warning.
  3. **Precision-Recall Curve**: reads `reports/pr_curve_data.json` and draws the interactive Plotly chart. Falls back to `pr_curve.png` if the JSON is missing.
- Caching: `load_retrievers` uses `@st.cache_resource`, so fitting happens once per Streamlit server process. Restarting the server refits everything.
- `theme.css`: CSS variables on `:root` (page, card, plot backgrounds, text, accent, three font stacks), then component rules for the hero header, chip row, cards, and pill-style tabs. Injected with `st.markdown(<style>…)`. It `@import`s Google Fonts (Fraunces, Inter, JetBrains Mono), which needs internet at page load.

### 3.6 `src/reporting/` (untracked in git)

- `build_report.py`: opens the course template (`docs/IR_report formatt.docx`), clears its body, keeps its styles and page setup, and writes the course report (the script itself is 831 lines), with tables, an RRF equation as a native Word equation, and a monospace architecture diagram. The numbers come from `comparison_table.csv` and `segmented_metrics.csv`. The methodology text (§3.1–3.4) matches the code; I checked it against `tfidf.py`, `bm25.py`, and `dense.py`.
- `build_slides.py`: builds a 9-slide 16:9 deck. Its docstring says it reads only real numbers from `reports/`. Its `TEAM_PLACEHOLDER` is literal text ("Your Name / Team Name") and needs filling in.
- I did not run either script in this pass.

---

### 3.7 Flagged discrepancies and caveats

These are the things I'd want to know before a viva. Items are ordered by how much they matter.

**D1 — The Dense index is smaller than the lexical indexes, and the logs and docs don't say what that does to the comparison.**
- `embeddings.npy` has **100,000** rows. `corpus.parquet` has **537,929**. TF-IDF and BM25 search all 537,929; Dense searches 100,000 (the 1,276 must-keep docs plus a random fill).
- Hybrid-RRF inherits this: its BM25 side has the full pool, its Dense side has the small pool.
- The effect is a smaller distractor set for Dense, so **Dense and Hybrid numbers are inflated relative to the lexical methods** by an unmeasured amount. The must-keep set guarantees every relevant document is in the Dense pool, so Dense is not handicapped on recall; it just faces fewer competitors.
- `DECISIONS.md` (lines 8–10) presents subsampling as a speed/memory measure and says nothing about the pool asymmetry. `src/reporting/build_report.py` line ~485 does state it ("100,000-question subsample"), and the report's own section 4.1 says the lexical models search 537,929. So the final report discloses it; the decision log and `gap_analysis.md` do not.
- The cleanest fix is to compare on equal pools: subsample the lexical indexes to the same 100k, or encode the full corpus for Dense (about five times the encoding cost).

**D2 — "Held-out" needs a qualifier.**
- The 400 eval queries are a random sample (`seed=42`) from questions that have qrels. Their **own questions and their duplicates stay in the corpus**. The only thing excluded is the query's own document, by exact text match (`.strip() ==`), not by qid.
- The DECISIONS line "a fixed number of eval queries are held out from the pool of questions" is true only in the narrow sense that these queries are not sampled as documents to retrieve. The retrievers are never trained, so there is no training-set leakage. The leakage that does exist is in **design selection** (D3).

**D3 — The design was tuned on the same 400 queries it's reported on.**
- Every step in DECISIONS.md (light mode, per-text threshold, per-document scoping, dual-space RRF) was accepted or rejected by looking at metrics from `run_evaluation.py` or `run_segmented_evaluation.py` on these 400 queries, and the spot-check qids come from the same set. There is no separate dev split. The reported numbers are therefore development-set numbers and will be optimistic.
- The final design was selected on the segmented analysis, which uses the same queries. That is a common practice in course projects, but it should be stated, not left implied.

**D4 — DECISIONS.md's first-fix paragraph describes a fallback that the code doesn't have.**
- DECISIONS lines 14–15: "TF-IDF word ngrams (1,2) plus char_wb 3–5 fallback **when the word vector is still degenerate**."
- The code ([tfidf.py:77-78](../src/retrieval/tfidf.py#L77)) adds the char list for **every** query with ≤5 words. There is no degeneracy check. This paragraph is superseded by the final entry (lines 38–39), which matches the code. Still, it reads as if a conditional fallback exists.

**D5 — DECISIONS.md's second-fix paragraph describes per-text selection that is no longer in the rank path.**
- Lines 27–28: "select preprocessing per text using `short_text_word_threshold` … in both TF-IDF and BM25, at fit and rank time."
- Superseded by the per-document rejection (lines 33–37) and the final dual-space entry (lines 38–42). In current code, the threshold only decides whether TF-IDF adds the char list and how the segmented evaluation splits queries. `preprocess_for_retrieval` ([clean.py:33](../src/preprocessing/clean.py#L33)) implements the rejected per-text selection and is unused.

**D6 — DECISIONS.md's numbers are the first-fix numbers, not the current ones.**
- Lines 16–20 give Hybrid-RRF P@10 after the first fix as 0.1673 and TF-IDF as 0.1158.
- `reports/comparison_table.csv` (the current run) gives TF-IDF 0.1310 and Hybrid-RRF 0.1708, which match `gap_analysis.md`'s table. DECISIONS.md does not say its numbers are from an earlier run.

**D7 — The spot-check hit count in DECISIONS.md and gap_analysis.md doesn't match the current spot-check file.**
- DECISIONS line 22 and gap_analysis line 50: TF-IDF **4/33** on qid 1182.
- Current `reports/spot_check_rankings.md` (written 2026-09-27 23:22, the latest run): TF-IDF **7/33**. BM25 8/33, Dense 8/33, Hybrid 9/33 all match.
- I'm reporting the file's value. The docs are probably from an earlier run of the same query.

**D8 — `gap_analysis.md` says the suite has 15 tests. Grep finds 27 test functions** (`tests/*.py`). I did not run pytest, so I can't say whether all pass.

**D9 — The DECISIONS log header lines contain an unexpanded shell placeholder.** Lines 3 and 8 read `$(date +%Y-%m-%d)`, not a date.

**D10 — There is no `README.md`.** `CONTEXT.md` and `TASKS.md` list "README + final report" as the remaining Phase 8 task. The run instructions in §5 are the only setup documentation in the repo right now.

**D11 — Unused code**, listed so you can say so before someone asks:
- `inverted_index.py` (only used by one smoke test).
- `clean.preprocess_for_retrieval` (per-text mode; rejected design).
- `TfidfRetriever._char_text` (never called).
- `TfidfRetriever.preprocess_text` (only used by tests).
- `config.preprocessing.min_tokens_fallback = 2` (no code reads it).

**D12 — UI scores aren't comparable across methods.** The Score column shows RRF sums for TF-IDF, BM25, and Hybrid, and cosine for Dense. The progress bar is scaled within each result list, so the bars look comparable when they aren't.

**D13 — Minor code-level observations**, not bugs I could confirm by running:
- `id_to_question` is rebuilt from 538k rows on every Streamlit rerun ([app.py:248](../src/ui/app.py#L248)).
- TF-IDF sorts all 538k similarities per query (`np.argsort`, [tfidf.py:64](../src/retrieval/tfidf.py#L64)); `argpartition` would be cheaper.
- Self-exclusion is by text, so a *different* question with identical text is also removed. Probably rare in QQP; not measured.

---

## 4. Likely viva questions, answered from the code

**Q1. Why RRF and not a learned re-ranker?**
- Scores aren't comparable across the things being fused. BM25 is unbounded, cosine is bounded, and TF-IDF cosine has a different scale again. RRF uses only ranks, so no calibration is needed ([hybrid_rrf.py:5](../src/retrieval/hybrid_rrf.py#L5), [tfidf.py:80](../src/retrieval/tfidf.py#L80)).
- RRF has one parameter (`rrf_k = 60`, from config). A learned re-ranker would need training pairs, and this repo has no training split. The qrels are the only labels, and they're also the evaluation labels. Training on them would contaminate the evaluation.
- Be honest about the gap: I found no sweep over `rrf_k`, and no experiment with a re-ranker. Say "not evaluated" if asked.

**Q2. Why does TF-IDF have three internal sub-spaces?**
- **Light** (`_light_features`): lowercased tokens with stopwords and stems kept, plus adjacent bigrams. Keeps the surface form and phrases. Needed because stemming collapsed "What is the google?" to `['googl']` in the original bug.
- **Heavy** (`_heavy_features`): stopwords removed, Porter-stemmed unigrams. The normalisation that helped long queries.
- **Char** (`char_wb` 3–5, raw text): sub-word overlap, used only when `is_short_text(query)`. Short queries have few word matches, so character overlap gives a second signal.
- **Why separate matrices and not one vocabulary**: DECISIONS.md (2026-09-27) rejected per-document mode selection in one shared vectorizer. Concrete example from the qrels: a short query's `geographic` could not match its longer duplicate's `geograph`, and query-only bigrams had no counterparts. Separate corpus-wide matrices keep each space internally consistent.
- Caveat: the char list is added only for short queries, so short queries get three lists in the fusion and long queries get two. That is intended, but it means a document that the char space likes gets extra RRF mass only on short queries.

**Q3. How is Hybrid-RRF different from the RRF inside TF-IDF and BM25?**
- Same function (`reciprocal_rank_fusion`), same `k = 60`. The difference is what gets fused.
- **Internal** fusion (TF-IDF, BM25): several representations of the same sparse lexical model, over the same corpus, tokenised from the same text. It fixes short-text vocabulary sparsity.
- **Top-level** Hybrid: BM25's already-fused ranking plus Dense's semantic ranking. These are two different paradigms (exact lexical match and learned embeddings). It captures complementary signal.
- Mechanically, Hybrid's BM25 input is the internal ensemble output, so the top-level fusion is RRF over RRF results. TF-IDF is not in Hybrid.
- The Dense side of Hybrid searches the 100k subset (D1).

**Q4. How do you know the evaluation is fair (held out, not leaked)?**
Answer in three parts, because the honest answer has a yes and a no:
- **No training, so no parameter leakage.** All four methods are unsupervised. Qrels are used only to score.
- **Eval queries are a seeded random sample of 400 questions** (`seed=42`, [build_corpus.py:50-55](../src/data/build_corpus.py#L50)). Each eval query is excluded from its own result list by exact text ([tfidf.py:68](../src/retrieval/tfidf.py#L68), [bm25.py:35](../src/retrieval/bm25.py#L35), [dense.py:120](../src/retrieval/dense.py#L120)). Query-as-document leakage is prevented.
- **But the same 400 queries were used to choose the design** (D3), and the two indexes have different sizes (D1). So the numbers are optimistic for the lexical design choices, and Dense/Hybrid are advantaged by the smaller pool. I'd present the numbers with that caveat. The fix is a separate dev split and equal pools.

**Q5. Why is Dense recall so high (0.966) and P@10 so low (0.174)?**
- P@10 is capped by the qrels. The average eval query has 2.19 relevant docs ([build_corpus.py:44-47](../src/data/build_corpus.py#L44) builds them; I measured the average from `qrels.parquet`). The ceiling for P@10 is therefore about 2.19/10 = 0.219, so 0.174 is close to it. Recall is the more informative number here.
- Part of Dense's recall comes from the smaller pool (D1). Not measured.

**Q6. Why exact search (`IndexFlatIP`) and not an approximate index?**
- A 100k × 384 float32 matrix is about 150 MB and brute-force search over it is fast enough for interactive use. Normalised vectors make inner product equal cosine. Exactness keeps the evaluation free of ANN recall noise. The code does not contain an approximate index.

**Q7. Why subsample the corpus for Dense?**
- DECISIONS.md (first entry) says encoding and FAISS building need to fit in memory and time during iteration. The subsample keeps every eval query and its qrels, so each query remains answerable. Consequence: D1.

**Q8. Why deduplicate the corpus by qid, and why bidirectional qrels?**
- The corpus is a set of questions, so each qid appears once ([build_corpus.py:41](../src/data/build_corpus.py#L41)). QQP's duplicate relation is symmetric, so `q1→q2` and `q2→q1` are both added. A query can then find its partner from either side.

**Q9. What does MAP@10 mean in this code?**
- Truncated AP over the top 10, divided by the full number of relevant docs (not `min(10, |R|)`). See §3.4. This is lower than the standard `min` convention when |R| > 10. With an average of 2.19 relevant per query, the effect is small.

**Q10. Why does the lexical pipeline lowercase and strip punctuation before tokenising?**
- Punctuation goes to spaces first, so `word_tokenize` sees clean words and doesn't split on apostrophes inconsistently. Lowercasing is applied first so stopword matching and stemming are consistent. Both are config flags (`lowercase`, `remove_punct`).

**Q11. Why is the short-text threshold 5 words?**
- `short_text_word_threshold` in config. The DECISIONS log says the 40 short eval queries (≤5 words) included 6 that collapsed to 0–1 tokens. The threshold is a whitespace count, so `"What is the google?"` has 4 words and counts as short. I didn't find a sweep over the threshold in the repo, so say it was chosen by inspecting that data.

**Q12. Why doesn't BM25 have a char space, and why does it not gate on length?**
- DECISIONS.md: "BM25 uses equivalent uniform light/heavy spaces." The documented short-text failure was a TF-IDF cosine-tie problem (1.0 ties), and cosine is what produces an exact 1.0. BM25 scores aren't bounded that way, and I found no BM25-specific tie analysis in the repo, so don't claim BM25 is immune. The code treats both spaces uniformly for every query.

---

## 5. How to run everything (clean clone → running UI)

Times are approximate. Where I could measure them I say so; otherwise they are estimates and labelled that way.

**Prerequisites:** Python 3.13 (the venv was built with 3.13.7). Internet on first run (NLTK data, Hugging Face model, QQP download).

1. **Clone and enter the repo** (~1 min). `git clone <repo>`, then `cd duplicate-question-retrieval`. Run every command below from this root directory.

2. **Create the venv and install deps** (estimate: 5–15 min; `sentence-transformers` pulls in PyTorch, which is large).
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
   `requirements.txt` does not pin torch; `sentence-transformers` installs it.

3. **Download the QQP dataset** (estimate: under 1 min on a good connection; not measured here). Writes `data/raw/questions.csv` (58 MB). Skips if the file exists.
   ```
   python -m src.data.download
   ```

4. **Build the corpus, qrels, and eval set** (estimate: well under 1 min; not measured here). Writes three parquets to `data/processed/`.
   ```
   python -m src.data.build_corpus
   ```
   Expected sizes: corpus 537,929; qrels 298,526; eval 400.

5. **NLTK data**: fetched automatically the first time `clean.py` is imported. Nothing to run.

6. **Dense embeddings**: no separate command. The first time `DenseRetriever.fit` runs without a cache, it encodes 100,000 questions with `all-MiniLM-L6-v2` (CPU or GPU, auto-detected) and saves `embeddings.npy` and `embeddings_ids.npy`. **Estimate 5–15 min on CPU; not measured here** (the latest logs were cache hits). The cache is reused only if the subsample ids match exactly.

7. **Run the test suite** (optional; not run in this pass). From the repo root, because tests open `config.yaml` by relative path:
   ```
   python -m pytest tests/ -v
   ```
   `test_retrievers.py` loads the Hugging Face model, so it needs network on the first run.

8. **Launch the UI** (the main deliverable):
   ```
   streamlit run src/ui/app.py
   ```
   - Startup, measured from the 2026-09-27 run's log (cache hit, warm NLTK): **about 7 min 20 s** before the Search tab is ready. Breakdown: parquet + qrels read ~10 s; TF-IDF fit **2 min 6 s**; BM25 fit **4 min 57 s**; Dense load from cache under 1 s, but it first contacts Hugging Face to check the model (about 15 s of HEAD requests). Set `HF_HUB_OFFLINE=1` to avoid those requests once the model is cached.
   - Without the embeddings cache, Dense would encode 100k questions during startup, adding the step-6 time.
   - Everything is in memory for the life of the server process (`st.cache_resource`). Restarting refits.
   - Per-query latency, measured inside the evaluation loop (same process, same code path): TF-IDF **~0.6 s**, BM25 **~2.0 s**, Dense **~0.04 s**, Hybrid-RRF **~2.1 s**. These are averages over 400 queries; the first query after startup may be slower.
   - The Performance Comparison and Precision-Recall tabs read the committed `reports/` files; they don't need a rerun.

9. **Re-run the full evaluation** (optional; regenerates `reports/` and overwrites the committed comparison table). Measured total in the 2026-09-27 run: **about 38 min** (fits ~7 min + evaluation ~31 min), with the embeddings cached.
   ```
   python -m src.evaluation.run_evaluation
   ```
   Per-method evaluation times from that run: TF-IDF 3 min 50 s, BM25 13 min 16 s, Dense 14 s, Hybrid-RRF 13 min 56 s. Outputs: `reports/comparison_table.csv`, `reports/pr_curve_data.json`, `reports/pr_curve.png`, `reports/spot_check_rankings.md`.

10. **Segmented evaluation** (optional; TF-IDF and BM25 only). Expect roughly the TF-IDF and BM25 fit time plus about 17 minutes of query time. Not measured in this pass.
    ```
    python -m src.evaluation.run_segmented_evaluation
    ```

11. **Report and slides** (optional; not run in this pass). Build from `reports/` into `reports/`:
    ```
    python -m src.reporting.build_report
    python -m src.reporting.build_slides
    ```
    `build_report.py` needs `docs/IR_report formatt.docx` as its template. `build_slides.py` has the literal placeholder `Your Name / Team Name` to replace.

**Things that trip people up**
- Run from the repo root. All paths are relative.
- `data/processed/` and `*.npy` are gitignored, so a clean clone has no corpus, no parquets, and no embeddings. Steps 3–4 must run first, and step 6 happens on the first UI launch.
- `reports/comparison_table.csv` and `pr_curve_data.json` are committed, so the UI's comparison and curve tabs work immediately even without rerunning evaluation. Running step 9 changes them.
- The UI's Search tab depends on `config.yaml`'s `eval_query_count` and `top_k`, and on `reports/` existing at the repo root (resolved relative to `app.py`).
