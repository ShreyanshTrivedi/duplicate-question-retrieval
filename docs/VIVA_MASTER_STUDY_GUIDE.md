# Viva Master Study Guide — Project 8: Duplicate Question Retrieval (QQP)

Companion to `reports/IR_project8_report.docx` (the final report, now 9 pages with a roomier layout). Everything here is checked against the code in `src/`, `config.yaml`, `DECISIONS.md`, `docs/CODEBASE_WALKTHROUGH.md`, `reports/*.csv`, `reports/spot_check_rankings.md`, the installed libraries in `.venv`, and the cached model files of `all-MiniLM-L6-v2`.

**Tags used throughout**
- **[assignment]**: the project sheet specified it (TF-IDF, BM25, Sentence Transformer, Hybrid RRF with k≈60, P@10 / Recall / MAP / nDCG@10, preprocessing steps, the three outputs, the QQP dataset).
- **[project]**: a real decision recorded in `DECISIONS.md` or visible in the code.
- **[general]**: textbook reasoning, not something this project tested.
- **[CHECK]**: not verified from a local file. Verify it before relying on it, or don't say it.

**Verified environment facts** (from `.venv`, read during the writing of this guide):
rank-bm25 0.2.2 · scikit-learn 1.9.1 · faiss-cpu 1.15.1 · sentence-transformers 6.1.0 · nltk 3.10.3 · streamlit 1.64.0 · torch 2.14.0 · transformers 5.17.0 · local Python 3.13.7 (Docker image uses python:3.11-slim).

**Report revision note (current version, 9 pages after a spacing pass: more generous line, paragraph, heading, table-cell and figure spacing, larger reference text, table headers kept with their rows; content unchanged).** This guide follows the revised layout of `reports/IR_project8_report.docx`. Changes from the previous 7-page draft: Table 8 (run time) was removed and its measured values now sit in the "Speed" limitation; Table 1 is a 4-column table; Tables 4 and 5 are full width with their explanations above or below instead of beside them; Table 7 now precedes Figure 4; Table 3 has an explicit Origin column and an "Alternatives" column; figures were redrawn at their inserted width so the text is a true 8 pt or larger; the P@10 ceiling leads with the exact 0.184; Figure 4 uses a neutral palette (not the per-method colours). The old draft is kept as `reports/IR_project8_report_v2_7pages_cramped.docx`, and the 10-page version as `reports/IR_project8_report_v1_10pages.docx`.

**Final layout (file `reports/IR_project8_report_v3_spacious.docx`, 9 pages).** Body 11 pt (the template default), tables and captions 9 pt, references 10 pt, line spacing 1.15. Block order changed only to avoid blank gaps: in 2.1 the "Questions are short" paragraph and Table 2 come before Figure 1; Figure 2 follows the Section 3 introduction (before 3.1); in Section 4 the "Precision ceiling" paragraph comes before Table 5, and Figure 5 comes after the "Segmented analysis" paragraph. No wording or number changed.

**The encoder-training-data point is now in the report.** The Limitations include an "Encoder training data" bullet: the cached model card (`README.md` of `all-MiniLM-L6-v2`) lists "Quora Question Triplets — 103,663" among its fine-tuning datasets, so Dense may have seen related pairs; the overlap with our queries was not measured. The old sentence "was not checked" is gone (see Part F, item 3).

---

## Part A — The project in 30 seconds, 2 minutes and 5 minutes

### A.1 Thirty seconds

"Quora has hundreds of thousands of questions, and many are the same question asked differently. We turned the Quora Question Pairs dataset into a search problem: given a new question, find the earlier questions that mean the same thing. We built four rankers: TF-IDF, BM25, a sentence-embedding model, and a hybrid that merges BM25 and the embedding model with Reciprocal Rank Fusion. On 400 evaluation queries the embedding model was best, with nDCG@10 of 0.94. The hybrid was second at 0.93, and BM25 beat TF-IDF. Along the way we found and fixed a bug where very short questions collapsed to a single word and every match tied at a score of 1.0."

### A.2 Two minutes

"The task is duplicate question detection on Quora Question Pairs. The original dataset is 404,290 question pairs, each labelled duplicate or not. That's a classification format, so we rebuilt it as retrieval. Every unique question became a document, 537,929 in total. Every duplicate pair became a relevance judgement in both directions, 298,526 qrel rows. Then we sampled 400 questions that have at least one duplicate and used them as queries.

We compared the four models the assignment asks for. TF-IDF and BM25 are sparse lexical models: they score by shared words. The dense model is all-MiniLM-L6-v2, a sentence transformer that turns each question into a 384-dimensional unit vector; we search those vectors exactly with FAISS. The hybrid fuses the BM25 ranking and the dense ranking with RRF at k equals 60, using only ranks, so we never have to compare a BM25 score with a cosine score.

The engineering story is short-text indexing. Questions are short, median 10 words. Stop-word removal plus stemming turned 'What is the google?' into the single token 'googl', and documents that shrank to the same token all scored a cosine of exactly 1.0, so the ranking among them was arbitrary. Our fix keeps separate light and heavy indexes over the whole corpus, plus character n-grams for short TF-IDF queries, and fuses their rankings with the same RRF function.

Results: Dense nDCG@10 0.9431, Hybrid 0.9251, BM25 0.7755, TF-IDF 0.7314. Two honest caveats. The dense index holds 100,000 questions while the lexical ones hold all 537,929. And we chose the design on the same 400 queries we report on, so these are development-set figures."

### A.3 Five minutes

Use A.2, then add these four blocks in this order.

1. **Why the problem is hard.** "Relevance here means same intent, not same topic. 'Which smartphone should I buy under INR 15k?' and the same sentence with 'INR 40000' share almost every word but are not duplicates. 'How do I speak English fluently?' and 'How can I take my spoken English to a higher level…' share few words but are duplicates. So pure word overlap fails both ways, which is why we compare lexical, semantic and fused rankers."

2. **How the short-text fix was found.** "Our first fix turned off stop-word removal and stemming for everything. It removed the ties, but when we split the evaluation into short and long queries we saw long-query TF-IDF fall below its original level, even though long queries never had the bug. The second attempt picked light or heavy processing per text inside one vocabulary. That broke matching between a short query and its long duplicate: the query kept 'geographic' while the document had the stem 'geograph'. The third design, which we kept, gives every document both representations in separate spaces and fuses rankings. On the 40 short queries TF-IDF nDCG@10 went from 0.3508 under the per-text design to 0.7908."

3. **How to read the numbers.** "Precision@10 looks low, 0.17 at best, because most queries have only one labelled duplicate. 294 of the 400 have exactly one, and the average is 2.19. A perfect ranker would score 0.184, so Dense's 0.1738 is 94% of the ceiling. The rank-sensitive metrics carry the comparison: Dense puts a duplicate at rank 1 for 90.75% of queries."

4. **What we'd do next.** "Give all models the same pool, hold out a separate development split, try weighted fusion and a sweep of the RRF constant, add a cross-encoder re-ranker, and use an encoder that hasn't seen Quora data. The model card lists Quora triplets in its training data."

<!-- END PART A -->

---

## Part B — Report companion (section by section)

Format for every item: **Means** (what it says), **Why** (why it is in the report), **Defend** (how to back it up), **Q→A** (one likely follow-up with an answer).

### B.0 Title, abstract, keywords

**"Duplicate Question Retrieval … Sparse, Dense and Rank-Fusion Retrieval with Short-Text Vector Indexing"**
- Means: the three families of model (sparse lexical, dense, fused) plus the project-specific engineering topic.
- Why: "Short-text vector indexing" is Project 8's distinct task on the sheet [assignment]; the dual-space design is our answer to it [project].
- Q→A: *"What exactly is 'short-text vector indexing'?"* → Building vector representations that still carry enough signal when a text has only a few words. In our case, keeping a surface-form space and a character space so a 4-word question doesn't shrink to one stem (Section 3.4).

**Abstract numbers: 537,929 / 298,526 / 400 / 0.9431 / 0.9087 / 0.9251 / 0.7755 / 0.7314**
- All defined in Part E. Every one comes from `data/processed/*.parquet` or `reports/comparison_table.csv`.
- Q→A: *"Why say 'development-set figures' in an abstract?"* → Because the short-text design was chosen by looking at results on these same 400 queries; there was no separate dev split. Saying so up front is more honest than calling them held-out results [project].

**"Standard preprocessing collapsed some short questions to one token and produced tied cosine scores of 1.0"**
- Defend: `tests/test_short_text_bug.py` reproduces the six example queries; `DECISIONS.md` (2026-09-27) records it; verified in the installed NLTK: heavy mode gives `['googl']` for "What is the google?".
- Q→A: *"Why exactly 1.0?"* → TF-IDF vectors are L2-normalised. A query reduced to one stem is a unit vector on that one axis. Any document also reduced to only that stem is the same unit vector, so their dot product (= cosine) is exactly 1.

**Keywords**: chosen to match the topics; nothing to defend.

### B.1 Section 1 — Introduction

**"Detecting a repeat is a retrieval problem: the new question is the query, earlier questions are the collection, and a document is relevant if annotators labelled it a duplicate."**
- Means: we redefine "relevance" as "labelled duplicate".
- Defend: `src/data/build_corpus.py` builds qrels only from `is_duplicate == 1` rows [project].
- Q→A: *"Is a non-duplicate pair a negative judgement?"* → No. Non-duplicate pairs add no qrels. Those questions simply sit in the collection, and anything not in the qrels counts as non-relevant. That includes true duplicates nobody labelled (Part F, item 10).

**"Relevance means the same intent, not the same topic" + the INR 15k / INR 40000 example**
- Defend: the pair is a real row of `data/raw/questions.csv` with `is_duplicate = 0`.
- Q→A: *"Which of your models would be fooled by it?"* → Any lexical model, because the two questions share every word except the number. Dense might also rank it high: we didn't measure this specific pair, so don't claim a model handles it.

**"Sparse lexical retrieval scores documents by weighted term overlap … TF-IDF [5] and BM25 [1]"** [general]
- Q→A: *"What does 'sparse' mean?"* → A vector with one dimension per vocabulary term, almost all zeros. A 10-word question touches about 10 of hundreds of thousands of dimensions.

**"It cannot see a paraphrase that shares few words, and normalisation can erase most of a very short question."** [general] + [project] (the second half is our bug).

**"Dense retrieval embeds each question as a vector with a bi-encoder such as Sentence-BERT [2] … can miss rare exact tokens"** [general]
- Q→A: *"Why would it miss exact tokens?"* → The model compresses a sentence into 384 numbers learned to capture meaning. A rare model number or name may hardly change the vector, while a lexical model matches it exactly. This is the general argument for hybrids; we didn't test it on QQP.

**"RRF does so from ranks alone, so no score calibration is needed"** [assignment] + [general]

**Contributions (i)–(iv)**: retrieval benchmark from QQP; four rankers compared on four metrics; dual-space short-text indexing; Streamlit interface. Each one maps to code: `build_corpus.py`; `src/retrieval/*`; `tfidf.py` / `bm25.py`; `src/ui/app.py`.

### B.2 Section 2 — About Dataset

**"404,290 pairs, 149,263 of them duplicates"**
- Means: rows in `data/raw/questions.csv` (the QQP TSV, saved as CSV by `src/data/download.py` from `qim.fs.quoracdn.net`); 36.9% are duplicates (derived: 149,263 / 404,290).
- Q→A: *"Is the label reliable?"* → It's human-assigned, and Quora's release described it as not guaranteed to be perfect [CHECK the exact wording of the release]. Treat it as noisy ground truth.

**"Corpus: every distinct question, 537,929"**
- How: `pd.concat([q1, q2]).drop_duplicates(subset=['qid'])` after `dropna` on the four required columns [project].
- Subtle point: deduplication is **by qid, not text**. Two different qids with the same wording stay as two documents. Example: 77805 "Is Time travel possible?" and 26235 "Is time travel possible?" both appear in `spot_check_rankings.md`.
- Q→A: *"Why did you deduplicate by ID rather than text?"* → The qrels refer to qids, so the qid is the document identity. Merging by text would need remapping the qrels. We didn't do that, and this is also why self-exclusion is by text (Part F, item 8).

**"Each duplicate pair is recorded in both directions, giving 298,526 rows over 149,650 query IDs"**
- Means: q1→q2 and q2→q1. 298,526 = 2 × 149,263 exactly, so no pair was duplicated within one direction.
- Why: duplication is symmetric, so a question can find its partner whichever side of the pair it was on [project].
- Q→A: *"Did you add transitive duplicates (A~B, B~C ⇒ A~C)?"* → No. Only directly labelled pairs are qrels, so A→C counts as non-relevant even if it's a true duplicate. That makes our scores conservative.

**"400 questions drawn at random (seed 42) from those with a qrel; 40 short, 360 long"**
- How: `np.random.seed(42); np.random.choice(queries_with_qrels, 400, replace=False)`; `config.yaml: eval_query_count: 400, seed: 42`.
- Q→A: *"Why 400?"* → A config choice that keeps a full evaluation run under an hour on CPU (the last full run took about 39 minutes, mostly BM25 and Hybrid). No power analysis was done [project]. *"Why only questions with a qrel?"* → A query with no relevant documents has undefined recall and AP. The metric functions return 0 for empty relevant sets, so sampling only scoreable queries avoids meaningless zeros.

**"They remain in the collection: only a document whose text equals the query is removed from its own results."**
- How: each retriever filters `doc_text.strip() != query.strip()` (`tfidf.py`, `bm25.py`, `dense.py`). Dense fetches `top_k + 5` to make room for the drop.
- Q→A: see Part F, item 8 (it also removes a *different* qid with identical text).

**Figure 1(a)**: question length; median 10 words; 8.2% (44,244 of 537,929) have ≤5 words. Computed from `corpus.parquet` with `str.split()`.
**Figure 1(b)**: relevant documents per evaluation query; 294 of 400 have exactly one; mean 2.19, median 1, max 33; 13 queries have 11 or more. Computed from `qrels.parquet` restricted to the 400 query IDs.
- Q→A: *"Why log scale in (b)?"* → 294 queries are in one bar and the tail has ones and twos. On a linear scale the tail is invisible.

**Table 1**: as above. "Dense index 100,000 × 384" is the shape of `data/processed/embeddings.npy`.
**Table 2**: three real rows of the raw file. Pair 1: paraphrase with little overlap (duplicate). Pair 2: a 3-word question (duplicate). Pair 3: high overlap but not a duplicate.
- Q→A: *"Did your system retrieve pair 1 correctly?"* → Not checked individually. Only the five spot-check queries in `reports/spot_check_rankings.md` were inspected.

### B.3 Section 3 — Methodology (intro and Figure 2)

**"All four rankers share one interface and are unsupervised"**
- Means: every retriever implements `rank(query, top_k) -> [(doc_id, score)]` (`src/retrieval/base.py`), and nothing is trained on our qrels (the report's wording is "nothing is trained on the qrels").
- Careful: the dense encoder **was** trained by its authors, on data that includes Quora question triplets. "Unsupervised" means "not trained on our labels", not "never saw Quora". See Part F, item 3.

**Figure 2 (pipeline)**: TF-IDF and BM25 each fuse their own spaces with RRF. Hybrid fuses the BM25 list and the Dense list, and TF-IDF is not in it. The dashed "live search" arrow means the Search tab calls `rank()` directly, while the Comparison and PR tabs read `reports/comparison_table.csv` and `reports/pr_curve_data.json`.

### B.4 Section 3.1 — Preprocessing and Indexing

**Pipeline order** (`src/preprocessing/clean.py::preprocess`): lowercase → replace every `string.punctuation` character with a space → `nltk.word_tokenize` → (heavy only) drop NLTK English stop-words → (heavy only) Porter stem.
- Verified: `word_tokenize` first runs Punkt sentence splitting, then the Treebank-style word tokenizer. The installed NLTK English stop-word list has **198** words, and it includes what/is/the/how/do/i/why/which/can/not/no. `PorterStemmer().mode == 'NLTK_EXTENSIONS'`.
- Q→A: *"Why replace punctuation before tokenising?"* → So the tokenizer only sees words. Side effect: "don't" becomes "don t" [project].

**"A text is short if it has at most five whitespace-separated words"** — `is_short_text`: `len(text.split()) <= 5` on the raw text (`config.yaml: short_text_word_threshold: 5`).
- Q→A: *"Why 5?"* → It was chosen by inspecting the data and was never swept [project]; the report says so.

**"TF-IDF stores sparse document-term matrices and scores a query against every row; BM25 … scans every document's term counts for each query term. Neither uses an inverted index at query time."**
- Verified: TF-IDF calls `cosine_similarity(query_vector, matrix)` over all 537,929 rows, then `np.argsort`. rank_bm25's `get_scores` loops over query tokens and builds `[doc.get(q) or 0 for doc in self.doc_freqs]` across every document.
- Q→A: *"Then why is BM25 so slow?"* → That Python-level scan over 537,929 dictionaries per query token, done twice (light and heavy spaces): 1.99 s per query (Limitations, "Speed").

**"384-dimensional unit vector held in a FAISS IndexFlatIP"** — verified: `config.json` hidden_size 384; `normalize_embeddings=True`; `IndexFlatIP` metric = inner product, no training step.

**"The dense index holds 100,000 questions: the 400 queries and their relevant documents (1,276 questions in the run log) plus a seeded random fill."**
- How: `dense.py::_select_corpus` keeps `_must_keep_qids` (eval qids + their qrel docs) and fills with `remaining.sample(n, random_state=42)`. The log line is "Subsampled corpus to 100000 documents (1276 forced-in for eval coverage)". 1,276 = 400 queries + 876 relevant documents (876 = 2.19 × 400).
- Q→A: *"Doesn't forcing the relevant docs in help Dense?"* → It guarantees every relevant document is findable, as it is in the full lexical index. The advantage is the smaller set of distractors: 100,000 against 537,929 (Part F, item 2).

### B.5 Section 3.2 — Ranking Models and Table 3

**Table 3 columns**: Component | Origin (Spec = assignment-specified; Ours = our choice; Spec + ours = family specified, checkpoint or mode ours) | Chosen | Alternatives | Why. The caption says "No alternative listed was evaluated", which is true: the alternatives are named only so you can discuss them (Part C), never as things we compared.

**Table 3 rows** (defend each in one sentence; the report's own "Why" text is shown in quotes where it is short):
- TF-IDF (spec): scikit-learn `TfidfVectorizer` with our callable analysers. The from-scratch `inverted_index.py` exists but is used only by a test [project].
- BM25 (spec): `rank_bm25.BM25Okapi` with library defaults k1 = 1.5, b = 0.75, ε = 0.25, untuned [project].
- Dense encoder (Spec + ours): the family is specified; the checkpoint `all-MiniLM-L6-v2` is ours. Report: "Encoder type specified; this checkpoint is small, free and CPU-friendly." It's small (6 layers, 384-d), free and runs on CPU [project]. We do not train it, but the model card shows its authors fine-tuned it on Quora triplets (Part F, item 3).
- Vector index (ours): exact flat inner product. 100,000 × 384 × 4 bytes = 153.6 MB, "about 150 MB".
- Fusion (spec): RRF at k = 60. A learned re-ranker would need training labels, and the only labels are the evaluation qrels [project].
- Preprocessing (Spec + ours): the steps are specified; the light/heavy modes are ours. Report: "heavy mode alone collapsed short questions (Section 3.4)".
- Interface (ours): Streamlit.
- Metrics (Spec): P@10, Recall, MAP, nDCG, all reported at cutoff 10. Report: "P@10 is capped (Section 4), so the other three carry the comparison."

**TF-IDF equation**: w(t,d) = tf(t,d) · (ln((1+N)/(1+df_t)) + 1), then L2 normalisation, then cos(q,d) = q·d / (‖q‖‖d‖).
- Verified in sklearn 1.9.1 source: `smooth_idf=True` adds 1 to df and N, uses `np.log`, adds 1.0; `sublinear_tf=False` (raw counts); `norm='l2'`.
- Q→A: *"Why +1 inside and outside?"* → Inside: smoothing, as if one extra document contained every term, so no division by zero. Outside: so a term in every document still has weight 1 rather than 0.

**BM25 equation**: score(q,d) = Σ_{t∈q} idf(t) · f(t,d)(k1+1) / (f(t,d) + k1(1 − b + b·|d|/avgdl)), with idf(t) = ln((N − n_t + 0.5)/(n_t + 0.5)).
- Verified in rank_bm25 0.2.2 source, including: if idf < 0 (term in more than half the documents) it is replaced by ε × mean idf, with ε = 0.25. A repeated query term is counted again because the loop is over the token list.
- Q→A: *"What do k1 and b do?"* → k1 controls term-frequency saturation: the second occurrence adds less than the first, and k1 = 0 means presence only. b controls length normalisation: b = 1 is full normalisation, b = 0 none. We used 1.5 and 0.75 without tuning.

**Dense**: sim(q,d) = f(q)·f(d) with f the normalised MiniLM embedding.
**Hybrid**: "RRF to two lists of 100 documents: the BM25 ensemble's output and the dense list. TF-IDF is not included … RRF over an RRF-fused list … dense side searches the same 100,000-question index."
- Verified: `HybridRRFRetriever([bm25, dense])`, `pool_size = max(100, top_k)`. The BM25 child returns its own fused top-100. A list can be shorter than 100 only if fewer documents score above zero.

### B.6 Section 3.3 — RRF Formulation and Table 4

**RRF(d) = Σ_{m∈M} 1/(k + rank_m(d)), k = 60; a list that omits d contributes nothing.** Verified in `hybrid_rrf.py::reciprocal_rank_fusion`: 1-based `enumerate(results, start=1)`, sums, sorts descending.
**Table 4** (illustrative ranks, computed by the project's own function): B (2,1) 0.0325 > A (1,3) 0.0323 > E (5,5) 0.0308 > D (–,2) 0.0161 > C (3,–) 0.0159.
- The point: E, ranked fifth by both lists, beats D, ranked second by only one. Agreement beats a single high rank.
- Q→A: *"What does k do?"* → Larger k flattens the curve, so the gap between rank 1 and rank 10 shrinks and agreement matters more. Smaller k lets top ranks dominate. k = 60 is the value from Cormack et al. [3] and the assignment; we didn't sweep it.

### B.7 Section 3.4 — Short-Text Vector Indexing and Figure 3

**"'What is the google?' becomes [googl] … six of the 40 short evaluation queries collapsed to zero or one token"** — `DECISIONS.md`, `reports/gap_analysis.md`, `tests/test_short_text_bug.py`. Verified token outputs in Part D.3.
**"Many documents were reduced to the same one-dimensional vector, so non-identical questions tied at cosine 1.0"** — see B.0.
**"Global light mode … pushed long-query TF-IDF below its original level"** — `DECISIONS.md` (2026-09-27, second entry).
**"Per-text mode selection … fragmented the vocabulary: 'geographic' vs 'geograph', bigrams only on the query side"** — `DECISIONS.md` (third entry). The dead function `clean.preprocess_for_retrieval` is that rejected design.
**"Dual-space indexing with rank fusion"** — `tfidf.py`: `light_vectorizer` (light tokens plus `a__b` bigrams), `heavy_vectorizer` (heavy unigrams), `char_vectorizer` (`analyzer='char_wb', ngram_range=(3,5)`, lowercase by default). The char list is added only when `is_short_text(query)`. Lists are fused with `reciprocal_rank_fusion`, k from config. `bm25.py`: light and heavy only, for every query.
- Q→A: *"Why not one vocabulary with both forms?"* → Then a short query's surface form and a long document's stem live as different dimensions in the same space and never match. Separate spaces mean each space tokenises queries and documents the same way.
**Figure 3**: the three lanes for the real query; tokens verified (Part D.3).

### B.8 Section 4 — Setup, metrics and Table 5

**"cutoff 10 (top 50 for the PR curves)"** — `run_evaluation.py` calls `rank(q, top_k=50)`, uses the first 10 for the four metrics and k = 1…50 for the curves.
**"Nothing is trained on the qrels; the queries stay in the collection …; design chosen on these same queries"** — Part F, item 1. (The earlier wording "no training data leaks" was replaced because the encoder was pre-trained on Quora data; Part F, item 3.)
**Table 5** — exactly `src/evaluation/metrics.py`:
- `precision_at_k`: hits in top k / k.
- `recall_at_k`: hits in top k / |R|.
- `average_precision(ranked[:10], R)`: Σ (precision at each hit) / |R|. The divisor is **|R|, not min(10, |R|)**.
- `ndcg_at_k`: binary gains; DCG = Σ 1/log2(i+1); IDCG uses min(k, |R|) ones.
- Q→A: *"Why is your MAP lower than standard for some queries?"* → For the 13 queries with more than 10 relevant documents, even a perfect top 10 gets AP = 10/|R| < 1. With min(10, |R|) it would be 1.

**P@10 ceiling paragraph** (below Table 5, "Precision ceiling"): a perfect ranker scores min(|R|,10)/10 per query, whose mean over the 400 queries is 0.184; the looser bound 2.19/10 ≈ 0.22 ignores that 13 queries have more than ten relevant documents; Dense 0.1738 / 0.184 = 94% of the exact ceiling. Hand check from the Figure 1(b) counts: (294×1 + 41×2 + 22×3 + 10×4 + 7×5 + 8×6 + 2×7 + 1×8 + 1×9 + 1×10 + 13×10)/400 = 736/400 = 1.84, and 1.84/10 = 0.184.
- Q→A: *"So is P@10 useless here?"* → It isn't useless, but it's capped. Recall@10, MAP and nDCG are more informative when most queries have one relevant document. P@1 is effectively Success@1 here.

### B.9 Section 4 — Results (Table 6, Figure 5)

**Table 6**: TF-IDF 0.1310 / 0.8341 / 0.6631 / 0.7314; BM25 0.1368 / 0.8582 / 0.7129 / 0.7755; Dense 0.1738 / 0.9659 / 0.9087 / 0.9431; Hybrid 0.1708 / 0.9535 / 0.8843 / 0.9251 (P@10 / R@10 / MAP@10 / nDCG@10), from `reports/comparison_table.csv`.
**P@1**: 0.9075 / 0.8925 / 0.7000 / 0.6450 (Dense / Hybrid / BM25 / TF-IDF), from `pr_curve_data.json` index 0.
**"order holds at every k from 1 to 50"**: checked for both precision and recall from `pr_curve_data.json`.
**Recall@50**: 0.999 / 0.996 / 0.945 / 0.929 (Dense / Hybrid / BM25 / TF-IDF). The report text now quotes only Dense 0.999, BM25 0.945 and TF-IDF 0.929 ("only 0.945 (BM25) and 0.929 (TF-IDF)").
**"plausible reason … no ablation"; "5.4 times smaller" (537,929 / 100,000 = 5.38); "Hybrid below Dense … not tested"** — causal claims are flagged as untested on purpose.
- Q→A: *"Then why have a hybrid at all?"* → It's a required output [assignment]. It keeps an exact-match signal and is second on every metric. On these data, equal-weight fusion with a weaker BM25 list didn't beat Dense alone.

**Figure 5(b) reading**: the x-axis is recall, so curves start at the recall reached at k = 1. That is 0.53 for TF-IDF and 0.74 for Dense [from JSON], not at 0. Big markers are k = 10.

### B.10 Section 4 — Segmented analysis (Table 7, Figure 4)

Layout note: Table 7 follows the "Segmented analysis" paragraph, and Figure 4 follows Table 7 at the top of page 7. Section 3.4 cites both ("Table 7 and Figure 4, Section 4"). Figure 4 uses neutral greys plus a magenta bar for the final design on purpose: it is TF-IDF only, so it must not reuse the per-method colours of Figure 5.

**Table 7** from `reports/segmented_metrics.csv` (`run_segmented_evaluation.py`, TF-IDF and BM25 only). Short = 40 queries, long = 360. The aggregate rows equal the query-weighted means, e.g. (40 × 0.1450 + 360 × 0.1294)/400 = 0.1310.
**"TF-IDF stronger on short (0.7908) than long (0.7247); BM25 the opposite (0.7314 / 0.7804) … not tested … noisy"**.
- Q→A: *"Any guess why?"* → TF-IDF's short-query path adds the character space and BM25's doesn't. That is a hypothesis, not a result, and with n = 40 the difference could be noise.
**Figure 4** (TF-IDF nDCG@10): Original All 0.7274; Global-light All 0.6184 and Short 0.5887; Per-text All 0.6784, Long 0.7148, Short 0.3508; Final All 0.7314, Short 0.7908, Long 0.7247. Missing bars were never recorded.
- Sources: `comparison_table_before_fix.csv`, `DECISIONS.md`, `scoped_full_evaluation.log`, `segmented_metrics.csv`. The two short-segment intermediate values (0.5887, 0.3508) appear only in `reports/gap_analysis.md` and the summary file. Say "from our development log" if asked.
- Report wording: "The final design beats the per-text design on both segments; its aggregate 0.7314 is within 0.004 of the original 0.7274, but with no 1.0 ties behind it." (0.7314 − 0.7274 = 0.0040.)
- Q→A: *"Why is the original 0.7274 'inflated'?"* → Part of it came from ties at 1.0 that happened to put relevant documents on top by arbitrary ordering. The final 0.7314 doesn't depend on ties. We cannot say how much of the original was luck, only that it was not a clean measurement.

### B.11 Run time (now the "Speed" limitation; Table 8 was removed)

The report no longer has a run-time table. The numbers are in the last Limitations bullet: "Approximate, from the log of one CPU run (2026-09-27; Dense used cached embeddings; per-query times include top-50 ranking): fitting took 126 s for TF-IDF, 297 s for BM25 and 0.8 s for Dense, and the mean time per query was 0.57 s (TF-IDF), 1.99 s (BM25), 0.035 s (Dense) and 2.09 s (Hybrid-RRF); start-up took 7.3 minutes, slow for interactive use."
- Source: timestamps in `reports/ui_redesign_evaluation.err.log` (2026-09-27, CPU), computed by `latency_from_log()` in `build_report.py`. Start-up 7.3 min = from "Loading datasets" to "Fitting Hybrid RRF". It was a single run, so treat the figures as approximate.
- Q→A: *"Why is Hybrid slower than BM25 plus Dense?"* → It's about the sum: 1.99 + 0.035 ≈ 2.03, plus fusion and overhead ≈ 2.09 [project]. *"Are these latencies from the UI?"* → No, they're from the evaluation script, which uses the same code path. The UI wasn't timed. *"Why did you drop the latency table?"* → To keep the report at seven pages; the values are still stated with their provenance.

### B.12 Section 5 — Conclusion, limitations, future work

- "within 0.0180 nDCG@10" = 0.9431 − 0.9251.
- The conclusion paragraph now ends with the short-query result "(0.3508 under the per-text design)". The earlier sentence about "two regressions that aggregate metrics had hidden" was cut for space; it is still true and you can say it: (1) global-light hurt long queries; (2) per-text hid the short-query collapse (aggregate 0.6784, short 0.3508).
- Limitations in the report (six bullets): Development-set figures; Unequal pools; Untuned and unswept; Encoder training data; Labels and cutoff; Speed. Each is expanded in Part F.
- Future work (one line): "a development split, equal-sized pools, a k sweep, a re-ranker and tuned BM25." Weighted fusion, a BM25 character space and significance tests were cut for space but remain sensible. None of these were tried; say "future work", never "we tried".

### B.13 References — what each is for

[1] BM25 (Robertson & Zaragoza; cited as "Found. Trends Inf. Retr.") · [2] Sentence-BERT · [3] RRF and k = 60 (Cormack et al.) · [4] the QQP release · [5] TF-IDF weighting (Salton & Buckley) · [6] Porter stemmer · [7] FAISS (Johnson et al.) · [8] MiniLM (the base architecture of all-MiniLM-L6-v2; cited as NeurIPS 33) · [9] nDCG (Järvelin & Kekäläinen; cited as "TOIS"). Bibliographic details (pages, volumes) of [5]–[9] were written from memory [CHECK].

<!-- END PART B -->

---

## Part C — Technology choices and alternatives

Each block lists five fields: **Chosen**, **How it works**, **Alternatives**, **When an alternative is better**, **Why not here**. Facts about alternatives are textbook knowledge [general] unless marked otherwise. None of the alternatives were tried in this project. Never say "we tested X".

### C.1 Dataset: QQP vs other collections
- **Chosen:** Quora Question Pairs, 404,290 labelled pairs [assignment: Project 8 dataset].
- **How it works:** pairs of questions with a binary human label. We converted it to corpus + qrels + queries (C.2).
- **Alternatives:** BEIR's "Quora" retrieval split, a pre-built version of the same idea with its own corpus/qrels/test queries [CHECK sizes]; CQADupStack (StackExchange duplicates); AskUbuntu duplicates; SemEval community-QA tasks; MS MARCO / Natural Questions (passage retrieval, not duplicates).
- **When better:** BEIR Quora gives standard, comparable splits and published baselines. CQADupStack tests domain variety. MS MARCO suits question→answer retrieval.
- **Why not here:** the sheet assigns QQP [assignment]. Building our own qrels from the raw file was also part of the work [project]. Trade-off: our numbers are not directly comparable with published BEIR numbers [general].

### C.2 Qrels construction
- **Chosen:** every `is_duplicate == 1` pair becomes qrels in both directions, binary relevance; corpus = unique qids [project] (`build_corpus.py`).
- **How it works:** 149,263 positive pairs × 2 directions = 298,526 rows. A query's relevant set is its directly labelled partners.
- **Alternatives:** (a) transitive closure: put duplicates into clusters, so A~B and B~C ⇒ A~C; (b) graded relevance; (c) one direction only; (d) use non-duplicate pairs as explicit negatives.
- **When better:** (a) recovers unlabelled duplicates and reduces false "non-relevant" judgements. On qid 1182 many relevant documents already exist, but obvious unlabelled duplicates like 77805 "Is Time travel possible?" still rank top and count as misses. (b) needs graded labels, which QQP doesn't have.
- **Why not here:** the simplest faithful mapping of the labels. Closure would add relevance that no annotator stated, and could chain errors through noisy labels [general]. Not tried.

### C.3 Tokenisation: NLTK vs spaCy vs regex vs subword
- **Chosen:** NLTK `word_tokenize`: Punkt sentence split, then Treebank-style word tokenizer (verified in NLTK 3.10.3), after replacing punctuation with spaces [project].
- **Alternatives:** spaCy tokenizer; a plain regex (`\w+`, scikit-learn's default `(?u)\b\w\w+\b`); subword tokenizers (WordPiece, as used inside MiniLM).
- **When better:** spaCy if you also want lemmas, POS tags or entities in one fast pipeline. A regex if speed matters and text is clean. Subwords for neural models and rare-word robustness.
- **Why not here:** NLTK is a standard, free teaching library that also supplies the stop-word list and Porter stemmer the sheet implies [assignment + project]. Since punctuation is already replaced by spaces, the tokenizer choice matters little here [general].

### C.4 Normalisation: Porter vs Snowball vs lemmatisation
- **Chosen:** NLTK `PorterStemmer` (mode `NLTK_EXTENSIONS`, verified) in heavy mode only [assignment: stemming] [project: light mode skips it].
- **How it works:** rule-based suffix stripping in steps (Porter 1980 [6]): plurals and -ed/-ing, then -ational→-ate and similar, then -ic/-ful/-ness, then final -e and -ll cleanup. Verified outputs: google→googl, geographic→geograph, geography→geographi, organize→organ, quickly→quickli, improve→improv.
- **Alternatives:** Snowball/"Porter2" (verified: quickly→quick, otherwise the same on our examples); Lancaster (more aggressive); lemmatisation (WordNet / spaCy), which maps to dictionary forms (studies→study).
- **When better:** lemmatisation gives readable, less over-merged forms. "organize"→"organ" is an over-stem that merges with "organ" (the body part). Snowball is generally preferred over the original Porter for English [general].
- **Why not here:** Porter is the classic, specified step [assignment]. The real problem was not stemmer quality but over-normalising short texts, which we fixed with the light space [project].

### C.5 Stop-word lists
- **Chosen:** NLTK English list, 198 words in the installed version, applied only in heavy mode [project].
- **Alternatives:** scikit-learn `ENGLISH_STOP_WORDS` (318 words, verified); a corpus-derived list (most frequent terms); no stop-word removal (our light mode).
- **When better:** no removal is better for short questions and phrases, where "what/how/why" carry intent: "how is X" vs "what is X". A corpus list adapts to the domain.
- **Why not here:** removal is a specified step [assignment]. Our finding is that it must not be the *only* view for short text, so the light space keeps all words [project].

### C.6 TF-IDF variants
- **Chosen:** scikit-learn defaults: raw tf, smoothed idf ln((1+N)/(1+df)) + 1, L2 normalisation, cosine. In SMART notation this is roughly "ntc" [general, check notation if asked].
- **Alternatives:** sublinear tf (1 + ln tf); unsmoothed idf ln(N/df) (as in our unused `inverted_index.py`); probabilistic idf ln((N−df)/df); pivoted length normalisation; binary tf.
- **When better:** sublinear tf when terms repeat a lot (long documents). Pivoted normalisation when document lengths vary widely.
- **Why not here:** questions rarely repeat a term, so raw vs sublinear tf barely matters [general]. Defaults keep it reproducible [project]. Not compared.

### C.7 BM25 vs BM25+/BM25L vs query-likelihood language models
- **Chosen:** Okapi BM25 via `rank_bm25.BM25Okapi`, k1 = 1.5, b = 0.75, ε = 0.25 (negative-idf floor), untuned [assignment: BM25] [project: defaults].
- **How it works:** a probabilistic relevance model [1]. Each matching term adds idf × a saturating tf term normalised by document length (formula in Part H).
- **Alternatives:** BM25L (`delta = 0.5` in rank_bm25, verified) and BM25+ (`delta = 1`, verified) fix over-penalising long documents; query-likelihood LM with Dirichlet or Jelinek–Mercer smoothing; divergence-from-randomness models.
- **When better:** BM25L/BM25+ when documents are long and varied. QL-Dirichlet is often comparable to BM25 and principled for short queries [general].
- **Why not here:** BM25 is specified [assignment]. Our documents are short questions, where the long-document fix matters little [general]. Parameters were not tuned; say so.

### C.8 Libraries: scikit-learn + rank_bm25 vs Lucene / Elasticsearch / Pyserini
- **Chosen:** scikit-learn sparse matrices for TF-IDF; rank_bm25 for BM25 [project].
- **How it works:** brute-force scoring of the query against every document (no postings traversal). TF-IDF: a sparse matrix-vector product over 537,929 rows. rank_bm25: a Python loop over all documents per query term.
- **Alternatives:** Lucene (inverted index, posting lists, skip pointers, WAND/block-max pruning), Elasticsearch/OpenSearch (Lucene behind a server), Pyserini (Lucene from Python, with standard BM25 settings for IR experiments).
- **When better:** at scale and for latency. An inverted index touches only documents containing query terms, so BM25 would take milliseconds, not 1.99 s per query. Pyserini also gives reproducible, standard BM25 baselines.
- **Why not here:** pure-Python, pip-installable and easy to change for the dual-space experiments [project]. Cost: slow BM25 and no inverted index at query time, which the report admits.

### C.9 Character n-grams (char_wb 3–5)
- **Chosen:** scikit-learn `analyzer='char_wb', ngram_range=(3,5)`, lowercased by default (verified), corpus-wide matrix, consulted only for queries of at most 5 words [project].
- **How it works:** each word, padded with spaces, is cut into overlapping 3-, 4- and 5-character pieces (" wh", "wha", "hat", " what", …). These are TF-IDF weighted, and cosine is taken in this space.
- **Alternatives:** word n-grams only; a k-gram index for wildcard/spelling (the IIR tolerant-retrieval chapter); edit-distance spelling correction; subword embeddings.
- **When better:** spelling correction when queries have typos and you want exact intent; subword embeddings for morphology.
- **Why not here:** char n-grams give sub-word overlap cheaply when a short query has few word matches [project]. They weren't added to BM25, and that wasn't tested.

### C.10 Dense encoder: bi-encoder MiniLM vs larger encoders vs cross-encoder vs ColBERT vs DPR
- **Chosen:** `sentence-transformers/all-MiniLM-L6-v2` bi-encoder [assignment: Sentence Transformer] [project: checkpoint]. Verified from the cached model files: `BertModel`, 6 layers, hidden size 384, 12 attention heads, FFN 1536, WordPiece vocab 30,522, uncased; max_seq_length 256 word pieces (position embeddings allow 512); mean pooling; a Normalize module; base `nreimers/MiniLM-L6-H384-uncased`; fine-tuned with a contrastive (in-batch) objective on over 1B pairs, including Quora Question Triplets (103,663) and StackExchange duplicate questions.
- **How it works:** query and document are encoded **separately**, so document vectors are pre-computed. Relevance = cosine of the two vectors.
- **Alternatives:** larger bi-encoders (e.g. all-mpnet-base-v2, 768-d [general]); cross-encoders (query and document read together; scores pairs, so used to re-rank a top-k); ColBERT (per-token vectors, late interaction "MaxSim"); DPR (dual BERT encoders trained for open-domain QA passages).
- **When better:** a cross-encoder is usually more accurate at re-ranking but too slow to score 537,929 documents per query. ColBERT is more accurate than single vectors but uses far more storage. Larger encoders cost more CPU time.
- **Why not here:** CPU-only, free, small and fast. 0.035 s per query, versus 1.99 s for BM25 [project]. Caveat: its training data includes Quora triplets (Part F, item 3).

### C.11 Vector index: FAISS flat vs IVF / HNSW / PQ vs Annoy / ScaNN
- **Chosen:** FAISS `IndexFlatIP` (exact; verified metric = inner product, no training) [project].
- **How it works:** computes the inner product of the query with all 100,000 stored vectors and returns the top ones. With unit vectors, inner product = cosine.
- **Alternatives:** IVF (k-means coarse quantiser; search only the nearest `nprobe` lists), HNSW (graph-based approximate nearest neighbour), PQ (compressed vectors), IVF-PQ; Annoy (random-projection trees); ScaNN (Google; anisotropic quantisation).
- **When better:** millions of vectors, or tight latency or memory budgets. Approximate search trades a little recall for big speed and memory gains.
- **Why not here:** 100,000 × 384 float32 ≈ 150 MB, and brute force takes 0.035 s per query including encoding. Exact search keeps approximation error out of the evaluation [project/general].

### C.12 Fusion: RRF vs CombSUM / CombMNZ vs weighted linear vs learning-to-rank
- **Chosen:** RRF, k = 60, equal weights, rank-only [assignment]; used at three points: inside TF-IDF, inside BM25, and in the hybrid [project].
- **Alternatives:** CombSUM (sum of normalised scores), CombMNZ (CombSUM × number of lists containing the doc), weighted linear interpolation α·s_dense + (1−α)·s_bm25 (needs score normalisation), learning-to-rank (LambdaMART and similar, trained on labelled features).
- **When better:** weighted fusion when one input is clearly stronger. Our Dense beats BM25 by a wide margin, so a tuned weight might keep Dense's advantage. LTR when there are training labels.
- **Why not here:** RRF is specified and needs no calibration [assignment]. Weights or LTR would need tuning data; our only labels are the evaluation qrels, and tuning on them would make the problem in Part F, item 1 worse [project]. Not tried.

### C.13 Metrics: P@10 / Recall / MAP / nDCG vs MRR / Success@k / F1
- **Chosen:** P@10, Recall@10, MAP@10 (|R|-normalised), nDCG@10 (binary gains) [assignment]; PR curves for k = 1…50 [assignment: PR graph].
- **Alternatives:** MRR (mean of 1/rank of the first relevant document); Success@k / Hit@k; F1@k; R-precision.
- **When better:** MRR and Success@k fit "find one duplicate" tasks, where most queries have exactly one relevant document (294/400 here). Our P@1 values (0.9075 Dense, …) are Success@1. MRR was not computed.
- **Why not here:** the four are specified [assignment]. Mention MRR as an obvious extra, and say we didn't compute it.

### C.14 Interface: Streamlit vs Flask / FastAPI / Gradio
- **Chosen:** Streamlit 1.64, three tabs (Search, Performance Comparison, Precision-Recall Curve); `@st.cache_resource` fits the retrievers once per server process [project].
- **Alternatives:** Flask/FastAPI plus an HTML front end (a real API, more control); Gradio (quick ML demos).
- **When better:** FastAPI when other programs call the search as a service, or when you need concurrency control. Gradio for single-function demos.
- **Why not here:** the sheet requires a search interface, a comparison table and a PR graph [assignment]. Streamlit shows all three from Python with tables and Plotly charts [project]. Downside: start-up fits everything, about 7.3 minutes measured for the evaluation script's identical fitting.

### C.15 Docker
- **Chosen:** `Dockerfile` (python:3.11-slim, CPU-only torch installed first, NLTK data and the MiniLM model baked in, `HF_HUB_OFFLINE=1`, port 8501). `data/` and `reports/` are mounted at runtime (`docs/DEPLOYMENT.md`) [project].
- **Alternatives:** plain venv instructions; conda; a cloud service (e.g. Streamlit Community Cloud).
- **When better:** a venv is simpler for one developer. A cloud service if a public URL is needed.
- **Why not here:** reproducible, offline-capable environment that is the same on any machine [project]. Note the image uses Python 3.11 while local development used 3.13.7 [project]. Whether the container was run end-to-end is not recorded in this guide's sources [CHECK].

<!-- END PART C -->

---

## Part D — Worked examples with real numbers

Every number below was computed with the installed libraries or the project's own functions (`src/evaluation/metrics.py`, `src/retrieval/hybrid_rrf.py`) while writing this guide. **Toy** examples are labelled toy; everything else is real repo data.

### D.1 TF-IDF and cosine on a 3-document corpus (toy)

Corpus: d1 = "learn python fast", d2 = "learn java", d3 = "python book". Query q = "learn python". N = 3. scikit-learn defaults: raw tf, idf = ln((1+N)/(1+df)) + 1, L2 normalisation.

| term | df | idf = ln(4/(1+df)) + 1 |
|---|---|---|
| learn | 2 | ln(4/3) + 1 = **1.2877** |
| python | 2 | **1.2877** |
| fast, java, book | 1 | ln(4/2) + 1 = **1.6931** |

- d1 raw = (fast 1.6931, learn 1.2877, python 1.2877); length = √(1.6931² + 2·1.2877²) = 2.4866; normalised = (0.6809, 0.5179, 0.5179).
- d2 raw = (java 1.6931, learn 1.2877); length = 2.1271; normalised = (0.7960, 0.6053).
- q normalised = (learn 0.7071, python 0.7071).
- **cos(q,d1)** = 0.7071·0.5179 + 0.7071·0.5179 = **0.7324**; **cos(q,d2)** = 0.7071·0.6053 = **0.4280**; **cos(q,d3)** = **0.4280**.
- Lesson: d1 wins because it matches both query terms. "fast" lowers d1's normalised weights, but it is still clearly ahead.

### D.2 BM25 with k1 and b (toy, rank_bm25 BM25Okapi defaults)

Same corpus. doc lengths 3, 2, 2 → avgdl = 7/3 = 2.3333. k1 = 1.5, b = 0.75, ε = 0.25.

1. **idf.** For learn/python, n_t = 2: idf = ln((3 − 2 + 0.5)/(2 + 0.5)) = ln(0.6) = **−0.5108**, which is negative. For fast/java/book: ln(2.5/1.5) = **+0.5108**.
2. **Negative-idf floor (rank_bm25).** Mean idf over all five terms = (2·(−0.5108) + 3·0.5108)/5 = 0.1022. Negative idfs become ε · mean = 0.25 · 0.1022 = **0.0255**.
3. **d1** (|d| = 3, f = 1 for each query term): denominator = 1 + 1.5·(1 − 0.75 + 0.75·3/2.3333) = 1 + 1.5·1.2143 = 2.8214. Per-term = 0.0255 · 2.5/2.8214 = 0.0226. Two terms → **0.0453**.
4. **d2** (|d| = 2, only "learn"): denominator = 1 + 1.5·(0.25 + 0.6429) = 2.3393 → 0.0255 · 2.5/2.3393 = **0.0273**. d3 is the same for "python": **0.0273**.

rank_bm25 printed exactly 0.0453 / 0.0273 / 0.0273.

**k1 = tf saturation** (|d| = avgdl, so the length term is 1). The contribution factor f(k1+1)/(f+k1) is:
- f = 1 → **1.0**
- f = 2 → **1.4286**
- f = 3 → **1.6667**
- f = 10 → **2.1739**
- upper limit k1 + 1 = **2.5**

**b = length normalisation** (f = 1):
- |d| = 2·avgdl → 2.5/(1 + 1.5·(0.25 + 1.5)) = **0.6897**
- |d| = avgdl → **1.0**
- |d| = 0.5·avgdl → 2.5/(1 + 1.5·(0.25 + 0.375)) = **1.2903**

Lesson: the toy corpus shows why the ε floor exists. With only 3 documents, common terms get a negative idf. In our 537,929-question corpus the floor applies only to terms in more than half the documents [general].

### D.3 The "What is the google?" collapse (real)

Verified outputs of `src/preprocessing/clean.py` and scikit-learn:
- **Heavy** (stop-words removed, Porter): `['googl']`. "what", "is", "the" are NLTK stop-words; "google" stems to "googl".
- **Light**: `['what', 'is', 'the', 'google']`, plus bigrams `what__is`, `is__the`, `the__google` (`tfidf.py::_light_features`).
- **char_wb 3–5** (first pieces): `' wh', 'wha', 'hat', 'at ', ' wha', 'what', 'hat ', ' what', 'what ', ' is', 'is ', ' is ', ' th', 'the', …`
- Same mechanism, verified: "How do I post here?" → `['post']`. Toy documents "Google?" and "What is Google?" both → `['googl']`.

**Why the tie is exactly 1.0.** In the heavy space the query vector is a unit vector on the single axis "googl". Any document whose heavy tokens are only "googl" (even repeated) is the same unit vector after L2 normalisation, so cos = 1.0. So "Google?" and "What is Google?" (toy) tie with each other and with every other such document. The order among them is arbitrary. Six of the 40 short evaluation queries collapsed to 0 or 1 token (`DECISIONS.md`, `gap_analysis.md`).

**After the fix**, the heavy list is only one of three ranked lists. The light list separates documents by "what/is/the" and the bigrams, the char list adds sub-word overlap, and RRF combines the three. `tests/test_short_text_bug.py` checks that no query among the six has more than one exact 1.0 score in its top 5.

### D.4 The geographic / geograph mismatch (mechanism real, sentences toy)

`DECISIONS.md` records the token pair from real qrel inspection, but not the exact query. So the sentences here are **toy**; the token outputs are verified with the project's code.
- Toy short query (4 words → short): "What is geographic information?" → light `['what','is','geographic','information']`; heavy `['geograph','inform']`.
- Toy long duplicate (12 words → long): "What are the main uses of geographic information systems in urban planning?" → heavy `['main','use','geograph','inform','system','urban','plan']`.
- **Rejected per-text design (one shared vocabulary):** the query is indexed light (`geographic`, `information`), the document heavy (`geograph`, `inform`). The content words are different dimensions, so their overlap is zero, and the query's bigrams exist on no document. A duplicate becomes invisible.
- **Final dual-space design:** in the heavy space both are processed heavy, so `geograph` and `inform` match. In the light space both are processed light, so `geographic` and `information` match. Each space is internally consistent; RRF merges the two lists.

### D.5 RRF with k = 60 (real and illustrative)

- **Table 4 (illustrative ranks, project function):** B = 1/62 + 1/61 = **0.0325**; A = 1/61 + 1/63 = **0.0323**; E = 1/65 + 1/65 = **0.0308**; D = 1/62 = **0.0161**; C = 1/63 = **0.0159**.
- **Real, qid 1182 ("How is time travel possible?")**, from `spot_check_rankings.md`:
  - Hybrid #1 is doc 14297 "Time Travel Is It Possible?". It is rank 2 in BM25's list and rank 5 in Dense's list, so 1/62 + 1/65 = 0.016129 + 0.015385 = **0.0315**, exactly the Hybrid score printed.
  - Doc 18194: BM25 #4, Dense #8 → 1/64 + 1/68 = **0.0303** (printed 0.0303).
  - Doc 2580: BM25 #9, Dense #4 → 1/69 + 1/64 = **0.0301** (printed 0.0301).
- **Real, qid 1353 ("Which book is best for Neet chemistry?"):** doc 1354 is rank 1 in both BM25 and Dense, so 2/61 = **0.0328**, the Hybrid score printed. That is the maximum possible score with two lists.
- **Bounds:** one list, rank 1 → 1/61 = 0.0164. So a document found by only one list can never beat a document found at rank 1 by both.

### D.6 P@10, Recall@10, AP@10, nDCG@10 by hand (real, qid 9495)

Query "How come ISIS has never attacked Israel?", |R| = 5 relevant: 119068, 119069, 153250, 64158, 9496.

**TF-IDF top 10** has hits at ranks **2, 4, 8, 9** (4 of 5):
- P@10 = 4/10 = **0.40**
- Recall@10 = 4/5 = **0.80**
- AP@10 = (1/2 + 2/4 + 3/8 + 4/9) / |R| = (0.5 + 0.5 + 0.375 + 0.4444)/5 = 1.8194/5 = **0.3639**
- DCG@10 = 1/log2(3) + 1/log2(5) + 1/log2(9) + 1/log2(10) = 0.6309 + 0.4307 + 0.3155 + 0.3010 = 1.6781
- IDCG@10 (5 relevant at the top) = 1 + 0.6309 + 0.5 + 0.4307 + 0.3869 = 2.9485
- nDCG@10 = 1.6781/2.9485 = **0.5691**

The project's metric functions return 0.4 / 0.8 / 0.3639 / 0.5691.

**Dense top 10** has hits at ranks 1–5: P@10 **0.5** (the maximum possible with |R| = 5), Recall **1.0**, AP **1.0**, nDCG **1.0**.

**MAP normalisation on qid 1182 (|R| = 33).** Hybrid has hits at ranks 1–9:
- P@10 = **0.9**
- Recall@10 = 9/33 = **0.2727** (the maximum possible is 10/33 = 0.3030)
- AP@10 = 9/33 = **0.2727** with our |R| divisor. With the min(10, |R|) convention it would be 9/10 = 0.9. This is the effect described in Table 5's note.

### D.7 Cosine = inner product for unit vectors (toy and real)

- **Toy:** a = (3, 4), c = (4, 3). cos = (12 + 12)/(5·5) = **0.96**. As unit vectors, (0.6, 0.8)·(0.8, 0.6) = 0.48 + 0.48 = **0.96**. Also ‖u − v‖² = 2 − 2cos = **0.08**, so for unit vectors the nearest by Euclidean distance is the most similar by cosine.
- **Real:** `dense.py` encodes with `normalize_embeddings=True` and the model ends in a Normalize module. `faiss.IndexFlatIP` returns inner products, so the Dense scores in `spot_check_rankings.md` (e.g. **0.9887** for "Which book is best for Neet chemistry?" vs doc 1354) are cosine similarities.

<!-- END PART D -->

---

## Part E — Numbers to memorise

### E.1 Data
| Number | Meaning | Source |
|---|---|---|
| 404,290 | raw QQP question pairs | `data/raw/questions.csv` |
| 149,263 | pairs labelled duplicate (36.9%) | same |
| 537,929 | unique questions (by qid) = corpus | `corpus.parquet` |
| 298,526 | qrel rows (= 2 × 149,263, both directions) | `qrels.parquet` |
| 149,650 | distinct query IDs with at least one qrel | same |
| 400 | evaluation queries, seed 42 | `config.yaml`, `eval_queries.parquet` |
| 40 / 360 | short (≤5 words) / long queries | `segmented_metrics.csv` |
| 2.19 / 1 / 33 | mean / median / max relevant docs per evaluation query | `qrels.parquet` |
| 294 | evaluation queries with exactly 1 relevant doc | same |
| 13 | evaluation queries with more than 10 relevant docs | same |
| 10 / 11.4 | median / mean words per corpus question | `corpus.parquet` |
| 8.2% (44,244) | corpus questions with ≤5 words | same |

### E.2 Pools, index and configuration
| Number | Meaning |
|---|---|
| 100,000 × 384 | dense index (vectors × dimensions), `embeddings.npy` |
| 1,276 | forced-in docs for the dense subsample = 400 queries + 876 relevant docs (run log) |
| 537,929 / 100,000 ≈ 5.4 | lexical pool is about 5.4× the dense pool |
| ≈150 MB | 100,000 × 384 × 4 bytes = 153.6 MB of float32 vectors |
| top_k = 10 | cutoff for metrics and UI (`config.yaml`) |
| rrf_k = 60 | RRF constant (`config.yaml`) |
| 5 | short-text word threshold (`short_text_word_threshold`) |
| 100 | per-list candidate pool for every RRF fusion (`max(100, top_k)`) |
| 50 | depth ranked in evaluation (PR curve k = 1…50) |
| top_k + 5 | FAISS fetch size, to allow dropping the self-match |
| 42 | seed for query sampling and dense fill |

### E.3 Main results (400 queries, cutoff 10) — `reports/comparison_table.csv`
| Method | P@10 | Recall@10 | MAP@10 | nDCG@10 | P@1 | Recall@50 |
|---|---|---|---|---|---|---|
| TF-IDF | 0.1310 | 0.8341 | 0.6631 | 0.7314 | 0.6450 | 0.929 |
| BM25 | 0.1368 | 0.8582 | 0.7129 | 0.7755 | 0.7000 | 0.945 |
| Dense | **0.1738** | **0.9659** | **0.9087** | **0.9431** | **0.9075** | **0.999** |
| Hybrid-RRF | 0.1708 | 0.9535 | 0.8843 | 0.9251 | 0.8925 | 0.996 |

- Gap Dense − Hybrid nDCG@10 = **0.0180**.
- P@10 ceiling: loose 2.19/10 = **0.219**; exact mean of min(|R|,10)/10 = **0.184**; Dense reaches **94%** of it.
- P@1 and Recall@50 come from `pr_curve_data.json`. The method ordering holds at every k from 1 to 50.

### E.4 Segmented (TF-IDF and BM25 only) — `reports/segmented_metrics.csv`
| Model | Group | n | P@10 | R@10 | MAP@10 | nDCG@10 |
|---|---|---|---|---|---|---|
| TF-IDF | short | 40 | 0.1450 | 0.8389 | 0.7225 | 0.7908 |
| TF-IDF | long | 360 | 0.1294 | 0.8335 | 0.6565 | 0.7247 |
| BM25 | short | 40 | 0.1375 | 0.8198 | 0.6550 | 0.7314 |
| BM25 | long | 360 | 0.1367 | 0.8625 | 0.7193 | 0.7804 |

Check: (40 × 0.1450 + 360 × 0.1294)/400 = 0.1310, which is the aggregate.

### E.5 Design stages (development history)
TF-IDF, P@10 / R@10 / MAP@10 / nDCG@10:
| Stage | Scope | Values | Source |
|---|---|---|---|
| Original (heavy only; inflated by 1.0 ties) | All | 0.1283 / 0.8198 / 0.6642 / 0.7274 | `comparison_table_before_fix.csv` |
| Global light mode | All | 0.1158 / 0.7301 / 0.5517 / 0.6184 | `DECISIONS.md` |
| Global light mode | Short | 0.1225 / 0.7087 / 0.5044 / 0.5887 | `gap_analysis.md` only |
| Per-text mode, shared vocabulary | All | 0.1200 / 0.7574 / 0.6206 / 0.6784 | `scoped_full_evaluation.log` |
| Per-text mode | Long | 0.1256 / 0.7949 / 0.6564 / 0.7148 | `DECISIONS.md` |
| Per-text mode | Short | 0.0700 / 0.4205 / 0.2981 / 0.3508 | `gap_analysis.md` only |
| Dual-space RRF (final) | All / Short / Long | E.3 / E.4 | csv files |

- BM25 nDCG@10 across the same stages: 0.7613 → 0.7526 → 0.7137 → **0.7755**.
- Hybrid nDCG@10 across the same stages: 0.9228 → 0.9081 → 0.8937 → **0.9251**.
- Original BM25 full row: 0.1345 / 0.8442 / 0.7013 / 0.7613. Original Hybrid: 0.1688 / 0.9498 / 0.8838 / 0.9228. Dense was unchanged in all stages: 0.1738 / 0.9659 / 0.9087 / 0.9431.
- Collapse: 6 of 40 short queries reduced to 0–1 tokens.

### E.6 Run time (one CPU run, 2026-09-27, `reports/ui_redesign_evaluation.err.log`; in the report only as the "Speed" limitation)
| | TF-IDF | BM25 | Dense | Hybrid-RRF |
|---|---|---|---|---|
| Fit | 126 s | 297 s | 0.8 s (cached embeddings) | reuses |
| Mean per query | 0.57 s | 1.99 s | 0.035 s | 2.09 s |
| 400-query eval | 3 min 50 s | 13 min 16 s | 14 s | 13 min 56 s |

- Loading data → all fitted: **7.3 min** (440 s).
- Whole evaluation run ≈ **39 min**.
- Hugging Face model checks at start-up: about 6 s.

### E.7 Model and library facts (verified in `.venv` and the model cache)
- **all-MiniLM-L6-v2:** BertModel; 6 layers; hidden size 384; 12 heads; FFN 1536; WordPiece vocab 30,522; uncased; max_seq_length 256 (position limit 512); mean pooling; Normalize module.
- **MiniLM training:** base `nreimers/MiniLM-L6-H384-uncased`; contrastive fine-tuning on more than 1B pairs, including **Quora Question Triplets 103,663**.
- **rank_bm25 0.2.2 BM25Okapi:** k1 = **1.5**, b = **0.75**, ε = **0.25**; idf = ln((N − n + 0.5)/(n + 0.5)); a negative idf is replaced by ε × mean idf. BM25L delta = 0.5; BM25Plus delta = 1.
- **scikit-learn 1.9.1 TfidfVectorizer:** idf = ln((1+N)/(1+df)) + 1 (`smooth_idf=True`); raw tf (`sublinear_tf=False`); `norm='l2'`; `char_wb` lowercases by default.
- **NLTK 3.10.3:** 198 English stop-words; Porter mode NLTK_EXTENSIONS. scikit-learn's own stop list has 318 words.
- **FAISS 1.15.1:** `IndexFlatIP` = exact inner product, no training.

### E.8 Spot-check (qid 1182, |R| = 33) — `reports/spot_check_rankings.md`
- Qrel hits in top 10: TF-IDF **7/33**, BM25 **8/33**, Dense **8/33**, Hybrid **9/33**.
- `DECISIONS.md` records TF-IDF 4/33 from the earlier first-fix run. The current file says 7/33; quote the file.

<!-- END PART E -->

---

## Part F — Honest concessions (say these before the examiner finds them)

Rule: concede quickly and precisely, give the size or direction of the effect if known, say "not measured" if not, then say what you'd do instead. Never improvise an experiment you didn't run.

**1. Development-set tuning, and the word "held-out".**
- Truth: the 400 queries are a seeded random sample. The rankers are never trained on qrels, so nothing leaks from our labels into the rankers (the encoder's own pre-training on Quora data is a separate point, item 3). But every design decision (global light, per-text, dual-space) was accepted or rejected by looking at metrics on these same 400 queries. The spot-check qids also come from this set [project: `DECISIONS.md`]. Earlier documents (`DECISIONS.md`, earlier report drafts) call them "held out"; the final report deliberately says "evaluation queries" and "development-set figures".
- Say: "They're held out from training, which doesn't exist, but not from design selection. So the TF-IDF and BM25 numbers are optimistic. Dense didn't change across the design stages (0.9431 throughout), so its number wasn't tuned on them."
- Instead: sample a separate development set, choose designs on it, then evaluate once on an untouched test set.

**2. Dense searches 100,000 questions; TF-IDF and BM25 search 537,929.**
- Truth: `corpus_subsample: 100000` in `config.yaml`. The subsample keeps all 400 queries and their 876 relevant docs (1,276 forced in), so Dense's recall isn't handicapped. It just faces about 5.4× fewer distractors. The hybrid's dense side shares this. The size of the effect wasn't measured [project].
- Why it was done: `DECISIONS.md` says it was to keep encoding and FAISS building tractable during iteration.
- Instead: give every model the same pool. Either encode all 537,929 questions, or restrict TF-IDF and BM25 to the same 100,000, and re-run.

**3. MiniLM was fine-tuned on Quora duplicates.**
- Truth, verified in the cached model card (`README.md` of all-MiniLM-L6-v2): its fine-tuning data includes "Quora Question Triplets — 103,663" and StackExchange duplicate-question pairs.
- So some of our evaluation pairs may have been seen, or near-seen, during the model's training. Which ones is unknown, and it was not measured.
- The report now says this under "Encoder training data": "The all-MiniLM-L6-v2 model card lists Quora Question Triplets (103,663) among its fine-tuning data, so Dense may have seen related pairs; the overlap with our queries was not measured." Add if asked: part of Dense's lead may come from training overlap.
- Instead: use an encoder whose training data excludes Quora [CHECK candidates on their model cards], or fine-tune on a QQP train split that is disjoint from the evaluation queries.

**4. No inverted index at query time.**
- Truth: TF-IDF uses a sparse matrix product against all rows. BM25 (rank_bm25) loops over every document per query term.
- `src/retrieval/inverted_index.py` is a from-scratch postings index with log(N/df) idf, but only one smoke test uses it (`tests/test_retrievers.py`). The app and evaluation never call it [project].
- Consequence: BM25 takes 1.99 s per query.
- Say: "We have an inverted index implementation, but the reported results don't use it."
- Instead: Lucene/Pyserini, or use the matrix's column (term) access as postings.

**5. RRF over RRF.**
- Truth: Hybrid fuses two lists. The first is the BM25 ensemble's output, which is itself an RRF of the light and heavy BM25 spaces; the second is Dense. TF-IDF is not in the hybrid [project: `app.py`, `run_evaluation.py`].
- Effect: the two BM25 spaces together count as one vote against Dense's one vote. Fusing all spaces flat would give the lexical side two votes.
- Not wrong, but say it plainly. Weighted or flat fusion was never tried.

**6. MAP normalisation.**
- Truth: `average_precision(ranked[:10], R)` divides by |R|, not min(10, |R|). This only matters for the 13 queries with |R| > 10, where even a perfect top 10 gets AP < 1.
- Example: qid 1182 Hybrid has 9 hits in ranks 1–9, so AP@10 = 9/33 = 0.27 rather than 0.9.
- Dividing by |R| is the classic TREC-style AP over a full ranking, here truncated at 10. Some libraries use min(k, |R|) for "AP@k" [general]. The report's phrase "lower than the usual convention" means lower than the min(k, |R|) convention.

**7. P@10 ceiling.**
- Truth: with mean |R| = 2.19 and 294/400 queries having one relevant doc, a perfect ranker gets P@10 = 0.184.
- So P@10 differences of 0.006 (TF-IDF vs BM25) are small in absolute terms; don't over-interpret them. Lean on nDCG@10, MAP@10 and Recall@10. No significance tests were run.

**8. Self-exclusion is by text, not by qid.**
- Truth: every retriever drops documents with `doc_text.strip() == query.strip()` (case-sensitive, exact).
- This removes the query's own document, but also any *different* qid with identical text. Such a document would be a perfect duplicate, possibly a relevant one, and would count as a miss. The frequency wasn't measured.
- Corpus deduplication is by qid, so identical texts with different qids do exist. 77805 "Is Time travel possible?" and 26235 "Is time travel possible?" differ only in case, so they aren't removed for each other.
- Instead: exclude by qid.

**9. Things that were never done.**
- No rrf_k sweep.
- No weighted fusion.
- No cross-encoder or other re-ranker.
- No BM25 parameter tuning.
- No other encoder compared.
- No ANN index.
- No stemmer or lemmatiser comparison.
- No significance tests.
- No sweep of the five-word threshold.
- No segmentation of Dense and Hybrid.
- No timing of the Streamlit UI itself.

Answer: "Not evaluated; that's future work." Never: "we tried it and it was worse."

**10. Incomplete qrels.**
- Truth: only labelled pairs are relevant. Real evidence from `spot_check_rankings.md`:
  - qid 1182: TF-IDF and BM25 rank 77805 "Is Time travel possible?" and 26235 "Is time travel possible?" in their top 3. Neither is in the qrels.
  - qid 1182: Dense's #1 is 380676 "How can Time Travel?", also unlabelled.
  - qid 1353: 2598 "Which book to use for preparing chemistry for neet?" is unlabelled.
- So absolute scores understate quality, and a model that finds unlabelled duplicates is penalised. Which model this hurts most was not measured.
- Instead: pool the top results of all systems and judge them (TREC-style pooling), or use transitive closure of the duplicate labels.

**11. Small or partial evidence.**
- The short segment is 40 queries.
- Segmentation covered only TF-IDF and BM25.
- The two intermediate short-segment numbers (0.5887 and 0.3508) exist only in `reports/gap_analysis.md`, not in a raw log.
- Latency comes from one run.

**12. Documentation inconsistencies an examiner might spot** (from `docs/CODEBASE_WALKTHROUGH.md`, section 3.7):
- `DECISIONS.md`'s first-fix paragraph mentions a char fallback "when the word vector is still degenerate". The code adds the char list for **every** short query.
- Its second-fix paragraph describes per-text selection, which is superseded (`preprocess_for_retrieval` is now unused).
- Its numbers are first-fix numbers, not final ones.
- It says TF-IDF got 4/33 on qid 1182; the current spot-check file says 7/33.
- Two header lines contain an unexpanded `$(date +%Y-%m-%d)`.
- Say: "The decision log is chronological; the final entry and the code are authoritative."

**13. Unused config and code** — know them so you aren't surprised:
- `config.preprocessing.min_tokens_fallback = 2` (read by nothing)
- `TfidfRetriever._char_text` (never called)
- `TfidfRetriever.preprocess_text` (tests only)
- `clean.preprocess_for_retrieval` (rejected design)
- `inverted_index.py` (one smoke test)

**14. UI scores are not comparable across methods.**
- The Score column shows RRF sums for TF-IDF, BM25 and Hybrid, and cosine for Dense. The progress bars are scaled within each result list [project: `app.py`].

**What we would do differently (one answer to rehearse)**
"Split the queries into dev and test before any design work. Index the same pool for every model. Use an encoder without Quora in its training data. Put BM25 on an inverted index (Pyserini). Sweep k and try weighted fusion. Add a cross-encoder re-ranker on the top 100. Exclude self-matches by qid. Report MRR and significance tests alongside the four metrics."

<!-- END PART F -->

---

## Part G — Question bank (98 questions, 15 traps)

⚠ TRAP marks a question whose obvious answer is wrong or overclaims; the **Danger** line says what not to say.

### G.1 IR fundamentals

**Q1. What is information retrieval, in one sentence for this project?**
Finding, from a large collection, the documents that satisfy an information need expressed as a query, ranked by estimated relevance. Here the collection is 537,929 Quora questions, the query is a new question, and "relevant" means a labelled duplicate. Output is a ranked list, judged at a cutoff of 10.

**Q2. Boolean retrieval vs ranked retrieval — which is yours?**
Ranked. Boolean retrieval returns the exact set matching a logical expression (AND/OR/NOT) with no order. Ranked retrieval scores every candidate and sorts. All four of our models produce scores and a ranking [project]. Boolean matching would be useless for paraphrases, since most duplicates don't share every term.

**Q3. ⚠ TRAP — Your system uses an inverted index, right?**
Not at query time. TF-IDF stores a sparse document-term matrix and scores the query against every row with a sparse matrix product. rank_bm25 loops over every document's term-count dictionary for each query term. We wrote a from-scratch postings index (`src/retrieval/inverted_index.py`), but only one smoke test uses it; the reported results don't. An inverted index would make BM25 far faster than 1.99 s per query.
**Danger:** saying "yes" — the examiner can open `tfidf.py` and `bm25.py`.

**Q4. What is the vector space model?**
Documents and queries become vectors in a space with one dimension per term, weighted (e.g. TF-IDF). Similarity is the angle between vectors, measured by cosine. It gives graded scores and partial matching, unlike Boolean retrieval. Our TF-IDF ranker is a vector space model in three separate spaces [project].

**Q5. Why cosine and not Euclidean distance?**
Cosine ignores vector length, so a long document isn't favoured just for having more terms. For unit vectors the two are equivalent for ranking, since ‖u − v‖² = 2 − 2cos(u, v). Both TF-IDF (L2-normalised) and Dense (normalised embeddings) use unit vectors, so inner product = cosine.

**Q6. What is a "document" and a "query" in your setup?**
A document is one unique question (by qid) from the 537,929-question corpus. A query is one of 400 sampled questions that has at least one labelled duplicate. The query's own document stays in the collection, and only identical text is filtered from its results.

**Q7. Sparse vs dense representations?**
Sparse: one dimension per vocabulary term or n-gram, mostly zeros, interpretable, exact matching. Dense: a fixed low dimension (384 here), every entry non-zero, learned so that similar meanings are close. Sparse misses paraphrases; dense can blur exact rare tokens [general]. The hybrid combines the two rankings.

**Q8. What is relevance here — binary or graded?**
Binary: a document is relevant if the pair was labelled `is_duplicate = 1`, in either direction. nDCG therefore uses gains of 0/1. QQP has no graded labels, so graded relevance wasn't possible.

**Q9. What is a test collection, and what are qrels?**
The Cranfield paradigm: a fixed document collection, a set of queries, and relevance judgements (qrels: query-document pairs judged relevant). Systems are compared by metrics computed from these. We built all three from QQP: 537,929 documents, 400 queries, 298,526 qrel rows.

**Q10. Explain the precision–recall trade-off with your data.**
Retrieving deeper (larger k) finds more relevant documents but adds more non-relevant ones. Figure 5(b): Dense's precision falls from 0.9075 at k = 1 to about 0.04 at k = 50 while recall rises to 0.999. Because most queries have one relevant document, precision must fall quickly once that document is found.

### G.2 Dataset and qrels

**Q11. What is Quora Question Pairs?**
A 2017 Quora release of question pairs with a human-assigned binary `is_duplicate` label. Our file has 404,290 pairs, 149,263 of them duplicates. It's designed as a pair-classification task, which is why we rebuilt it for retrieval [assignment: dataset].

**Q12. How did you turn it into a retrieval benchmark?**
`build_corpus.py`:
- Drop rows with missing questions or IDs.
- Corpus = union of both question columns, deduplicated by qid (537,929).
- Qrels = every duplicate pair in both directions (298,526).
- Queries = 400 qids sampled with seed 42 from the 149,650 that have qrels.

**Q13. Why record each pair in both directions?**
Duplication is symmetric. If A duplicates B, a user searching with B should find A too. Without both directions, half the questions in duplicate pairs would have no relevant documents. 298,526 is exactly 2 × 149,263.

**Q14. ⚠ TRAP — So your 400 queries are a held-out test set?**
They're held out from training only in a trivial sense: nothing is trained. They remain in the collection; only identical-text documents are filtered from their results. Every short-text design decision was evaluated on these same 400 queries. So the figures are development-set figures and probably optimistic for TF-IDF and BM25. Dense didn't change across stages.
**Danger:** "held-out test set" implies untouched-until-final evaluation, which isn't what happened. The final report avoids the term on purpose.

**Q15. How were the 400 queries chosen, and why 400?**
`np.random.seed(42)` then `np.random.choice` without replacement, from query IDs with at least one qrel. Requiring a qrel keeps recall and AP defined. 400 is a config value (`eval_query_count`) that keeps a full CPU evaluation around 39 minutes. No statistical power analysis was done.

**Q16. Are the QQP labels reliable and complete?**
Neither is guaranteed. Labels are human judgements and can be noisy [general; CHECK Quora's own wording]. They're incomplete for retrieval: only pairs that were shown to annotators are labelled. `spot_check_rankings.md` shows obvious unlabelled duplicates, such as 77805 "Is Time travel possible?" for "How is time travel possible?". So absolute scores are conservative.

**Q17. Why deduplicate the corpus by qid and not by text?**
The qrels refer to qids, so the qid is the document's identity. Text deduplication would need remapping the judgements. Consequence: different qids with identical text can coexist, and our text-based self-exclusion can remove such a document (Part F, item 8).

**Q18. What happens to the 255,027 non-duplicate pairs?**
Their questions are in the corpus as ordinary documents, but the pairs create no judgements. A non-duplicate is not a negative label; anything outside the qrels simply counts as non-relevant. They act as realistic hard distractors, since they were often paired because they looked similar.

**Q19. What does the question-length distribution show, and why does it matter?**
Median 10 words, mean 11.4. 8.2% of questions (44,244) have at most 5 words (Figure 1a). Those short questions are where stop-word removal plus stemming leaves almost nothing. That motivated the short-text work and the 5-word threshold.

### G.3 Preprocessing

**Q20. Walk through your preprocessing exactly.**
`clean.preprocess`, in order:
1. Lowercase.
2. Replace every `string.punctuation` character with a space.
3. `nltk.word_tokenize`.
4. Heavy mode only: drop NLTK English stop-words, then apply the Porter stemmer.

Light mode stops after step 3. Each space applies its own mode to both documents and queries.

**Q21. Why replace punctuation before tokenising?**
So the tokenizer sees only words and you don't get tokens like "?" or "n't". A side effect is that "don't" becomes "don t". Because it happens before tokenisation, the results are consistent for queries and documents [project].

**Q22. What does the Porter stemmer do? Give examples from your system.**
It's a rule-based suffix stripper with ordered steps that removes plural, tense and derivational endings to reach a stem, not a dictionary word [6]. Verified outputs: google → googl, geographic → geograph, organize → organ, quickly → quickli, improve → improv. It raises recall by merging word forms, but can over-merge ("organize" and "organ") or under-merge ("geographic" → geograph vs "geography" → geographi).

**Q23. Why not lemmatisation?**
Lemmatisation maps to dictionary forms using vocabulary and POS, so it's more precise and slower. Stemming was the specified step [assignment]. Our real problem was over-normalising short texts, which lemmatisation wouldn't fix (stop-words would still be removed). We didn't compare them.

**Q24. Why not always remove stop-words?**
For short questions they carry intent: "how is X possible" vs "what is X". Removing them collapsed six of the 40 short queries to 0–1 tokens. So stop-word removal is used only in the heavy space, and the light space keeps every word [project].

**Q25. ⚠ TRAP — How many stop-words does NLTK's English list have?**
In the installed NLTK 3.10.3, 198 (verified). The figure 179 is often quoted for older versions [general; CHECK]. scikit-learn's built-in list has 318.
**Danger:** quoting 179 as fact for our system.

**Q26. ⚠ TRAP — Do you stem and remove stop-words before the dense encoder?**
No. `DenseRetriever` encodes the raw question text. The model's own WordPiece tokenizer lowercases (it's uncased) and splits into subwords. Lexical preprocessing would remove information the encoder was trained to use.
**Danger:** saying "the same preprocessing is applied to all models".

**Q27. What exactly counts as "short"?**
`len(text.split()) <= 5` on the raw text: a whitespace split, so punctuation attached to words doesn't add tokens. "What is the google?" has 4 words, so it's short. The threshold (`short_text_word_threshold: 5`) was chosen by inspecting the data, not swept. It decides only two things: whether TF-IDF adds the char list, and how the segmented evaluation splits queries.

### G.4 TF-IDF

**Q28. Write the TF-IDF weight you actually use.**
w(t,d) = tf(t,d) · (ln((1+N)/(1+df_t)) + 1), with raw counts for tf. Vectors are then L2-normalised and compared by cosine. This is scikit-learn 1.9.1's default (`smooth_idf=True`, `sublinear_tf=False`, `norm='l2'`), verified in its source.

**Q29. Why idf?**
Terms that appear in many documents ("what", "best") discriminate poorly, so they're down-weighted. Rare terms ("NEET", "Trump") are strong evidence of a match. The log dampens the effect. The smoothed "+1" variant keeps every term's weight positive.

**Q30. Why L2 normalisation?**
So document length doesn't dominate the score. Cosine of normalised vectors equals their dot product, which makes scoring a single sparse matrix product. It's also exactly why a one-token query ties at 1.0 with every one-token document on the same axis.

**Q31. What are the three TF-IDF spaces?**
1. Light: surface tokens, stop-words kept, plus adjacent bigrams joined with `__`.
2. Heavy: stop-word-free Porter stems, unigrams only.
3. Char: `char_wb` 3–5-grams of the raw text, used only for queries of ≤5 words.

Each is fitted once over all 537,929 questions. The per-space top-100 lists are fused with RRF (k = 60) [project: `tfidf.py`].

**Q32. Why bigrams in the light space?**
They keep some word order and phrase evidence: "is__the", "the__google", "time__travel". With short questions, a matching phrase is strong evidence. They're built from light tokens, so stop-words take part in them [project].

**Q33. Why `char_wb` and not plain `char` n-grams?**
`char_wb` only builds n-grams inside word boundaries (words padded with spaces), so pieces don't span two words. That keeps the features about sub-word similarity: shared stems and prefixes. Plain `char` would mix in cross-word fragments. It was not compared.

**Q34. Does scikit-learn's own lowercasing or stop-word list apply in your light and heavy spaces?**
No. When `analyzer` is a callable, scikit-learn skips its own preprocessing, tokenisation and n-gram logic. Everything comes from our `preprocess`. The char space uses the built-in `char_wb` analyzer, which lowercases by default.

**Q35. What is the cost of a TF-IDF query?**
Transforming the query, then a sparse product with each 537,929-row matrix, then `np.argsort` of all scores, for two or three spaces. Measured: 0.57 s per query on CPU, including top-50 ranking and scoring. `argpartition` would be cheaper than a full sort, and an inverted index cheaper still.

### G.5 BM25

**Q36. Write the BM25 formula you use.**
score(q,d) = Σ_{t∈q} idf(t) · f(t,d)(k1+1) / (f(t,d) + k1(1 − b + b·|d|/avgdl)), with idf(t) = ln((N − n_t + 0.5)/(n_t + 0.5)). This is rank_bm25 0.2.2's BM25Okapi. If idf is negative (the term is in more than half the documents) it's replaced by ε × mean idf, with ε = 0.25.

**Q37. What do k1 and b do, and what values did you use?**
k1 controls saturation: extra occurrences of a term add less and less, bounded by k1 + 1. b controls length normalisation: 0 means none, 1 means full. We used the library defaults, k1 = 1.5 and b = 0.75, untuned. Worked numbers are in Part D.2.

**Q38. Can BM25 idf be negative? What does your library do?**
With this idf form, yes: when n_t > N/2. rank_bm25 replaces negative values with 0.25 × the mean idf. The toy example in D.2 shows it: in 3 documents, a term in 2 of them gets ln(0.6) = −0.51, floored to 0.0255.

**Q39. How does BM25 differ from TF-IDF?**
1. BM25's tf saturates, while TF-IDF's raw tf grows linearly.
2. BM25 normalises length explicitly via b and avgdl, instead of cosine normalisation.
3. Its idf comes from a probabilistic (Robertson–Spärck Jones) model.

Scores are unbounded sums, not cosines. Here BM25 beat TF-IDF on all four metrics (nDCG@10 0.7755 vs 0.7314).

**Q40. Why is BM25 so slow here?**
rank_bm25's `get_scores` builds a Python list over all 537,929 documents for each query term, and we do it for two spaces. That gives 1.99 s per query and a 297 s fit. An inverted index (e.g. Lucene) would only touch documents containing the term.

**Q41. ⚠ TRAP — How did you tune k1 and b?**
We didn't. They're the library defaults (1.5 and 0.75). Tuning them on the evaluation queries would worsen the development-set problem; a separate dev split would be needed. It's listed as future work.
**Danger:** inventing a grid search.

**Q42. Why doesn't BM25 have a character space?**
`DECISIONS.md` gives BM25 "equivalent uniform light/heavy spaces". The tie bug was diagnosed in TF-IDF's cosine. A char space for BM25 wasn't tried. BM25's weaker short-query results (nDCG@10 0.7314 vs 0.7804 long) make it a candidate, but that's untested.

**Q43. ⚠ TRAP — So BM25 is immune to the 1.0-tie problem?**
BM25 scores aren't capped at 1, so the *exact* tie at the maximum is a cosine phenomenon. But a query reduced to one stem would still give many documents near-identical BM25 scores, differing only by length. We didn't analyse BM25 ties separately. Both lexical models got the light and heavy spaces.
**Danger:** claiming BM25 is immune; we have no analysis showing that.

### G.6 Dense retrieval, MiniLM and FAISS

**Q44. What is a sentence transformer / bi-encoder?**
A transformer that maps a whole sentence to one fixed-size vector, trained so that paraphrases end up close (Sentence-BERT [2]). "Bi-encoder" means the query and document are encoded separately, so all document vectors can be pre-computed and searched with nearest-neighbour search.

**Q45. Describe all-MiniLM-L6-v2 precisely.**
Verified from the cached files:
- BERT architecture with 6 layers, hidden size 384, 12 attention heads, feed-forward size 1536.
- Uncased WordPiece vocabulary of 30,522.
- Starts from `nreimers/MiniLM-L6-H384-uncased` (a distilled MiniLM [8]).
- Fine-tuned with a contrastive objective on over 1B sentence pairs.

**Q46. How is one vector made from token outputs?**
Mean pooling over the token embeddings (`pooling_mode_mean_tokens: true`), then a Normalize module to unit length. We also pass `normalize_embeddings=True`, so vectors are unit length.

**Q47. ⚠ TRAP — What's the model's maximum input length?**
256 WordPiece tokens: `max_seq_length` in `sentence_bert_config.json`; longer input is truncated. The underlying BERT config allows 512 position embeddings, but sentence-transformers truncates at 256. Quora questions are far shorter (median 10 words), so truncation doesn't matter here.
**Danger:** answering "512" (the BERT position limit) or "128".

**Q48. Why normalise, and what index do you use?**
FAISS `IndexFlatIP`: exact inner-product search over all stored vectors, no training (verified metric = inner product). With unit vectors, inner product = cosine, so the scores are cosine similarities, e.g. 0.9887 for the NEET pair.

**Q49. Why exact search rather than HNSW or IVF?**
100,000 × 384 float32 is about 150 MB, and exact search with encoding takes 0.035 s per query on CPU. Exact search also keeps approximation error out of the evaluation. Approximate indexes matter at millions of vectors; we didn't build one.

**Q50. Why only 100,000 questions in the dense index?**
To keep encoding and index building tractable during iteration (`DECISIONS.md`). The subsample keeps the 400 queries and all 876 of their relevant documents (1,276 forced in), plus a seeded random fill. The cost is unequal pools (Q80).

**Q51. ⚠ TRAP — Was the encoder trained on your data?**
Not by us. Nothing is trained on our qrels. But the model card in the local cache lists "Quora Question Triplets (103,663)" in its fine-tuning data, so it has seen Quora duplicate questions and possibly some of our evaluation pairs. The report states this in its Limitations ("Encoder training data"). So part of Dense's lead may come from training overlap, by an unmeasured amount.
**Danger:** "No, it's completely unsupervised and never saw QQP."

**Q52. Bi-encoder vs cross-encoder?**
A cross-encoder reads query and document together through the transformer and outputs one relevance score. It's usually more accurate, because of token-level interaction, but needs a forward pass per pair, which is impossible over 537,929 documents per query. The standard design is a bi-encoder (or BM25) for the top 100, then a cross-encoder to re-rank. We didn't do that.

**Q53. Why did Dense win?**
A plausible reason: duplicates are paraphrases, and the encoder represents meaning rather than surface words (qid 9495: Dense found all 5 duplicates in ranks 1–5; TF-IDF got 4 of 5 at ranks 2, 4, 8, 9). Two confounds favour it: a 5.4× smaller pool, and Quora data in its training set. No ablation separated these.

### G.7 Fusion and RRF

**Q54. State RRF and explain k.**
RRF(d) = Σ_{m∈M} 1/(k + rank_m(d)), with 1-based ranks and k = 60; a list that misses d adds 0. k flattens the reciprocal curve: rank 1 gives 1/61 and rank 10 gives 1/70, so agreement across lists matters more than a single first place.

**Q55. Why fuse ranks, not scores?**
Scores live on different scales: BM25 is unbounded, cosine is in [−1, 1], and the three TF-IDF spaces have different distributions. Rank fusion needs no normalisation or calibration and has one parameter [assignment][general].

**Q56. ⚠ TRAP — Why is the hybrid worse than Dense?**
The observation is that Hybrid (nDCG@10 0.9251) is below Dense (0.9431). One consistent explanation: equal-weight RRF lets the weaker BM25 list pull some lexically similar but non-duplicate questions above Dense's correct ones. This was not tested: no weighted fusion or k sweep was run.
**Danger:** presenting the explanation as a measured finding.

**Q57. ⚠ TRAP — Which models does Hybrid-RRF combine?**
BM25 and Dense only; TF-IDF is not included (`HybridRRFRetriever([bm25, dense])`). And the BM25 input is itself the RRF fusion of BM25's light and heavy spaces, so it's RRF over an RRF-fused list. The dense side searches the 100,000-question index.
**Danger:** "all three models".

**Q58. Where else is RRF used in your system?**
The same `reciprocal_rank_fusion` function, with k = 60 from config, is used three times:
- inside TF-IDF, for the light, heavy and (short queries only) char lists;
- inside BM25, for the light and heavy lists;
- at the top level, for BM25 + Dense.

**Q59. How do CombSUM and CombMNZ compare?**
CombSUM adds normalised scores; CombMNZ multiplies that sum by the number of lists that retrieved the document. Both need score normalisation (min-max or z-score), which is fragile across different scorers. RRF avoids that [general]. Not tried.

**Q60. What happens to a document found by only one list?**
It gets one term: at best 1/61 = 0.0164. A document found at rank 1 by both lists gets 2/61 = 0.0328 (real: doc 1354 for qid 1353). A document at rank 5 in both (0.0308) beats one at rank 2 in a single list (0.0161), as Table 4 shows.

**Q61. ⚠ TRAP — Why k = 60? Did you try other values?**
60 is the value from Cormack et al. [3] and is what the assignment specifies. We didn't sweep it, so we can't claim it's optimal for QQP.
**Danger:** claiming a sweep or "we found 60 worked best".

### G.8 Short-text vector indexing

**Q62. Describe the bug.**
Heavy preprocessing reduced short questions to 0–1 tokens: "What is the google?" → [googl], "How do I post here?" → [post]. Six of the 40 short evaluation queries collapsed. Many documents collapsed to the same single stem too, and after L2 normalisation identical one-axis vectors have cosine 1.0, so unrelated questions tied at the top and their order was arbitrary.

**Q63. Why exactly 1.0 and not 0.99?**
A one-term vector, normalised, has value 1 on that axis and 0 elsewhere. Two such vectors on the same axis have dot product 1 × 1 = 1. Repetition doesn't matter: tf 2 still normalises to 1.

**Q64. What was the first fix, and why was it rejected?**
Global light mode: no stop-word removal or stemming for any text, plus bigrams and char n-grams. It removed the ties, but segmenting by query length showed long-query TF-IDF falling below its original level, even though long queries never had the bug. Aggregate TF-IDF nDCG@10 dropped from 0.7274 to 0.6184 [project: DECISIONS.md].

**Q65. What was the second attempt, and why was it rejected?**
Per-text mode selection: light processing for texts of ≤5 words, heavy for longer, inside one shared vocabulary. It restored long queries (nDCG@10 0.7148), but short queries collapsed to 0.3508. A short query kept surface forms such as "geographic" while its longer duplicate was stemmed to "geograph", and query-side bigrams had no document counterparts, so they never matched.

**Q66. What is the final design, and why does it work?**
Every question is indexed in each space independently and corpus-wide: light, heavy, and (for TF-IDF) char 3–5-grams for short queries. Queries are processed per space exactly like that space's documents, and the per-space rankings are fused with RRF. Within a space, tokenisation is consistent so matches can happen; across spaces, fusing ranks avoids comparing incompatible scores. TF-IDF short-query nDCG@10 is 0.7908 and long is 0.7247.

**Q67. How did you discover the regressions?**
Length-segmented evaluation (`run_segmented_evaluation.py`): the same metrics, split at 5 words. The aggregate number hid both problems. For example, the per-text design's aggregate 0.6784 contained a short-query 0.3508 [project].

**Q68. What evidence is there that the bug is fixed?**
1. `tests/test_short_text_bug.py` runs the six collapsing queries and asserts more than one token in the light view and at most one exact 1.0 score in the top 5.
2. Further tests check that heavy mode still stems long text and that the dual vectorizers are each fitted once.
3. Short-query TF-IDF nDCG@10 is 0.7908.

### G.9 Evaluation metrics

**Q69. Define P@10. Why is it so low?**
(Relevant documents in the top 10) / 10. Most queries have one relevant document (mean 2.19), so a perfect system scores only 0.184 on average. Dense's 0.1738 is 94% of that ceiling. It's low by construction, not because the systems are poor.

**Q70. Define Recall@10.**
(Relevant documents in the top 10) / |R|. With mostly one relevant document per query, it's close to "did we find the duplicate in the top 10?". Dense 0.9659, TF-IDF 0.8341. For the 13 queries with |R| > 10 it can't reach 1.

**Q71. ⚠ TRAP — Define AP@10 exactly as you implemented it.**
AP@10 = (1/|R|) · Σ_{i=1..10} P@i · rel_i: the sum of precision at each relevant rank in the top 10, divided by the **total** number of relevant documents. MAP@10 is its mean over the 400 queries. Because the divisor is |R| and not min(10, |R|), a perfect top 10 for a query with 33 relevant documents gets AP = 10/33.
**Danger:** stating the min(k, |R|) version, or saying AP divides by "the number of relevant retrieved".

**Q72. Define nDCG@10.**
DCG@10 = Σ_{i=1..10} rel_i / log2(i+1), with binary rel_i. IDCG@10 is the DCG of the ideal ranking, with min(10, |R|) relevant documents at the top. nDCG = DCG/IDCG, in [0, 1]. The log discount rewards putting duplicates near rank 1; rank 1 counts fully, rank 2 counts 0.63.

**Q73. Which metric would you trust most here, and why?**
nDCG@10 and MAP@10: they are rank-sensitive and not capped like P@10. Recall@10 says whether the duplicate was found but not where. With mostly one relevant document, P@1 (Success@1) and MRR are also natural. We report P@1 from the curve data.

**Q74. ⚠ TRAP — What is your MRR?**
We didn't compute MRR; it's not among the four specified metrics. The closest number we have is P@1, which here equals Success@1: Dense 0.9075, Hybrid 0.8925, BM25 0.7000, TF-IDF 0.6450. MRR would be at least that high, since it adds partial credit for lower ranks.
**Danger:** making up an MRR value.

**Q75. How is your PR curve built?**
For each method and each k from 1 to 50: average P@k and R@k over the 400 queries, then plot mean recall (x) against mean precision (y). It isn't the interpolated 11-point curve. It's a cutoff-based curve, so it starts at the recall reached at k = 1 (0.53 for TF-IDF, 0.74 for Dense). Big markers are k = 10, and they match Table 6.

**Q76. ⚠ TRAP — Are the differences statistically significant?**
We didn't run significance tests. The right test would be a paired test over per-query scores (paired t-test, Wilcoxon, or a randomisation test). Our evaluation stored only averages, so it would need a re-run. Large gaps like Dense vs TF-IDF (0.94 vs 0.73 nDCG) are likely real; the 0.006 P@10 gap between BM25 and TF-IDF should not be claimed significant.
**Danger:** "yes, clearly significant" with no test.

**Q77. Why a cutoff of 10?**
It's the assignment's cutoff for P@10 and nDCG@10, and `top_k: 10` in config. Ten results is a typical results page. Recall and MAP are reported at the same cutoff for consistency, and the PR curve goes to 50.

### G.10 Results interpretation

**Q78. Summarise the results in three sentences.**
Dense is best on every metric (nDCG@10 0.9431, MAP@10 0.9087). Hybrid-RRF is second (0.9251), and BM25 beats TF-IDF (0.7755 vs 0.7314). The ordering Dense > Hybrid > BM25 > TF-IDF holds for precision and recall at every k from 1 to 50.

**Q79. Why does BM25 beat TF-IDF?**
Plausibly saturating tf, explicit length normalisation and its idf form suit uneven question lengths [general]. This was not isolated by an ablation. Also note that both are multi-space RRF ensembles, so the comparison is ensemble against ensemble.

**Q80. ⚠ TRAP — Is Dense really better, or is it the pool size?**
We can't fully separate them. Dense searches 100,000 questions, the lexical models 537,929, so Dense faces about 5.4× fewer distractors. Its training data also includes Quora triplets. The large gap (0.94 vs 0.78 nDCG) and examples like qid 9495 suggest a real advantage, but the clean test, equal pools and an encoder without Quora data, wasn't run.
**Danger:** "Dense is clearly better, full stop."

**Q81. What does the segmented analysis show?**
TF-IDF is stronger on short queries than long (nDCG@10 0.7908 vs 0.7247); BM25 the reverse (0.7314 vs 0.7804). Table 6's lexical rows are the query-weighted means of these. The short group is only 40 queries, so treat the differences as noisy.

**Q82. Why might TF-IDF do better on short queries than BM25?**
TF-IDF's short-query path adds the character n-gram list; BM25's doesn't. That's a hypothesis only; it wasn't tested, and n = 40 is small.

**Q83. Interpret Figure 4.**
TF-IDF nDCG@10 by design stage and query group:
- Original 0.7274, but inflated by ties.
- Global light 0.6184 aggregate, 0.5887 short.
- Per-text 0.6784 aggregate, which hid a short-query collapse to 0.3508 (long 0.7148).
- Final 0.7314 aggregate, 0.7908 short, 0.7247 long.

Missing bars were never recorded. The final design beats per-text on both segments.

**Q84. Walk me through a real example.**
qid 1182, "How is time travel possible?", |R| = 33. Qrel hits in the top 10: TF-IDF 7, BM25 8, Dense 8, Hybrid 9. Hybrid's #1, doc 14297, was rank 2 for BM25 and rank 5 for Dense: 1/62 + 1/65 = 0.0315, as printed. Also, both lexical models put 77805 "Is Time travel possible?" at the top, which is clearly a duplicate but unlabelled: incomplete qrels in action.

**Q85. What does Recall@50 tell you?**
Dense 0.999, Hybrid 0.996, BM25 0.945, TF-IDF 0.929. Dense finds nearly every labelled duplicate within 50, while the lexical models still miss about 5–7%. That suggests a dense first stage plus re-ranking is a sound architecture [general].

### G.11 Engineering, UI and Docker

**Q86. Describe the code architecture.**
- `RetrieverBase.rank(query, top_k) -> [(doc_id, score)]` is the common interface.
- `TfidfRetriever`, `BM25Retriever`, `DenseRetriever` and `HybridRRFRetriever` implement it, all sharing `reciprocal_rank_fusion`.
- `build_corpus.py` makes the parquets; `run_evaluation.py` writes the comparison table, PR data and spot-checks; `run_segmented_evaluation.py` writes the segmented CSV.
- `app.py` is the UI.

**Q87. How does the Streamlit UI work?**
Three tabs. Search: a text box plus method selector, calling `rank(query, 10)` live. Performance Comparison: reads `reports/comparison_table.csv`. Precision-Recall Curve: reads `reports/pr_curve_data.json` into Plotly. `@st.cache_resource` fits all retrievers once per server process. Scores aren't comparable across methods (RRF sums vs cosine).

**Q88. How long does start-up take?**
The evaluation script's identical fitting took 7.3 minutes from loading the data to all retrievers fitted: TF-IDF 126 s, BM25 297 s, Dense 0.8 s from cached embeddings, plus loading and model checks. The UI itself wasn't timed. Without the embeddings cache, encoding 100,000 questions would add more (not measured).

**Q89. Explain the Docker image.**
- `python:3.11-slim`.
- CPU-only torch installed first (avoids the CUDA wheel).
- requirements installed, then NLTK data and the MiniLM model baked in.
- `HF_HUB_OFFLINE=1`, so no network at runtime; exposes 8501 and runs Streamlit.
- `data/` and `reports/` are mounted at runtime, not baked in (`docs/DEPLOYMENT.md`).

**Q90. How is the work reproducible?**
- Fixed seed 42 for query sampling and the dense fill.
- Config-driven paths and parameters (`config.yaml`).
- Library defaults recorded.
- Embedding cache checked against the exact ID list.
- Committed result files (`reports/*.csv`, `pr_curve_data.json`).
- A Docker image with pinned-at-build dependencies.

Exact library versions are visible in `.venv`, but `requirements.txt` is unpinned [project], which is a weakness.

**Q91. What tests exist?**
27 pytest test functions, which all passed when run during validation on 2026-10-03. They cover:
- corpus building;
- preprocessing;
- the short-text bug (four tests);
- the RRF arithmetic (D1 = 1/61 + 1/62);
- metrics;
- retriever smoke tests;
- UI helper functions.

**Q92. How would you scale this to millions of questions?**
- Lexical: an inverted index with Lucene/Pyserini, with pruning (WAND or block-max).
- Dense: batch encoding plus an approximate index (HNSW or IVF-PQ in FAISS).
- Re-ranking: a cross-encoder on the top 100.
- Caching of popular queries.

None of this is implemented here.

### G.12 Limitations and reflection

**Q93. What is the biggest weakness of the study?**
The comparison isn't fully fair or clean. Designs were chosen on the evaluation queries, Dense has a smaller pool, and its encoder was trained on Quora triplets. All three favour some systems by unmeasured amounts. The ranking of methods is probably right, but the exact gaps are uncertain.

**Q94. What would you do differently?**
- Split dev and test before design work.
- Use equal pools.
- Use an encoder without Quora data.
- Put BM25 on an inverted index.
- Sweep k and try weighted fusion.
- Add a cross-encoder re-ranker.
- Exclude self-matches by qid.
- Report MRR and significance tests.

**Q95. Can your numbers be compared with published QQP or BEIR results?**
No, not directly. Our corpus, qrels construction (bidirectional, no transitive closure), query sample and cutoffs are our own. BEIR's Quora task has its own splits [CHECK details]. Our numbers are internally comparable across our four models only.

**Q96. How do unlabelled duplicates affect your results?**
They make every system look worse than it is: a true duplicate ranked first counts as a miss. They could also change the ordering if one model finds more unlabelled duplicates. Examples exist in the spot-check (77805, 26235 and 380676 for qid 1182; 2598 for qid 1353). The effect wasn't measured; pooled re-judging would fix it.

**Q97. What is the self-exclusion issue?**
Each retriever removes documents whose text exactly equals the query (after `strip()`). That removes the query's own document, but also any other qid with identical text, which would be a perfect duplicate and possibly relevant. Exclusion by qid would be cleaner. The number of affected queries wasn't measured.

**Q98. Is the hybrid worth having if Dense wins?**
It's a required output [assignment]. It's second on every metric and keeps an exact-match component, which can matter for names and numbers. On qid 1182 it found more labelled duplicates in the top 10 than either input (9 vs 8 and 8). On average, though, equal-weight RRF didn't beat Dense, so a weighted or re-ranked hybrid would be the next experiment.

<!-- END PART G -->

---

## Part H — One-page cheat sheet

**Formulas (as implemented)**
- TF-IDF (scikit-learn 1.9.1): w = tf · (ln((1+N)/(1+df)) + 1), raw tf, L2-normalised. cos(q,d) = q·d / (‖q‖‖d‖).
- BM25 (rank_bm25 0.2.2 BM25Okapi): Σ_{t∈q} idf(t) · f(k1+1) / (f + k1(1 − b + b·|d|/avgdl)), with idf = ln((N − n + 0.5)/(n + 0.5)). Negative idf → 0.25 × mean idf. k1 = 1.5, b = 0.75, ε = 0.25, untuned.
- Dense: sim = f(q)·f(d), unit vectors, so inner product = cosine. ‖u − v‖² = 2 − 2cos.
- RRF: Σ_{m∈M} 1/(60 + rank_m(d)). Max with one list 1/61 = 0.0164; with two lists 2/61 = 0.0328.
- P@10 = |R∩T10|/10 · R@10 = |R∩T10|/|R| · AP@10 = (1/|R|) Σ_{i≤10} P@i·rel_i · nDCG@10 = DCG/IDCG with DCG = Σ rel_i/log2(i+1).
- P@10 ceiling = mean of min(|R|,10)/10 = **0.184** (loose bound 2.19/10 = 0.219).

**Pipeline**
1. QQP 404,290 pairs → corpus 537,929 (unique qid) + qrels 298,526 (both directions) + 400 queries (seed 42; 40 short / 360 long).
2. Preprocessing: lowercase → punctuation to space → `word_tokenize` → [heavy: drop 198 NLTK stop-words → Porter].
3. TF-IDF = RRF(light + bigrams, heavy, char_wb 3–5 only for ≤5-word queries).
4. BM25 = RRF(light, heavy).
5. Dense = MiniLM-L6 → FAISS IndexFlatIP over 100,000 questions (1,276 forced in).
6. Hybrid = RRF(BM25 list, Dense list); TF-IDF is not in it.
7. Each fusion uses top-100 per list; evaluation ranks top 50; metrics at 10.
8. Self-exclusion is by exact text.

**Model**: all-MiniLM-L6-v2 — BERT, 6 layers, 384-d, 12 heads, FFN 1536, vocab 30,522 uncased, max_seq_length 256, mean pooling + Normalize, contrastive fine-tuning on more than 1B pairs **including Quora triplets (103,663)**.

**Results (P@10 / R@10 / MAP@10 / nDCG@10)**
| | P@10 | R@10 | MAP@10 | nDCG@10 |
|---|---|---|---|---|
| TF-IDF | 0.1310 | 0.8341 | 0.6631 | 0.7314 |
| BM25 | 0.1368 | 0.8582 | 0.7129 | 0.7755 |
| **Dense** | **0.1738** | **0.9659** | **0.9087** | **0.9431** |
| Hybrid | 0.1708 | 0.9535 | 0.8843 | 0.9251 |

- P@1: .9075 / .8925 / .7000 / .6450 (Dense / Hybrid / BM25 / TF-IDF).
- R@50: .999 / .996 / .945 / .929.
- Dense − Hybrid nDCG@10 = 0.0180.

**Segmented nDCG@10**: TF-IDF short 0.7908 / long 0.7247; BM25 short 0.7314 / long 0.7804 (n = 40 / 360).

**Design stages, TF-IDF nDCG@10**: original 0.7274 (inflated by ties) → global light 0.6184 (short 0.5887) → per-text 0.6784 (short **0.3508**, long 0.7148) → final 0.7314 (short **0.7908**, long 0.7247). BM25: 0.7613 → 0.7526 → 0.7137 → 0.7755. Collapse: 6 of 40 short queries.

**Run time (one CPU run)**: fit TF-IDF 126 s, BM25 297 s, Dense 0.8 s (cached); per query 0.57 / 1.99 / 0.035 / 2.09 s; start-up 7.3 min; full evaluation ≈ 39 min.

**Data facts**: mean |R| 2.19, median 1, max 33; 294 queries with |R| = 1; 13 with |R| > 10; median question 10 words; 8.2% of questions ≤5 words; dense pool is 5.4× smaller.

**Say early, never deny**: development-set tuning; unequal pools; MiniLM saw Quora triplets; no inverted index at query time; RRF over RRF; |R|-normalised AP; text-based self-exclusion; incomplete qrels; nothing swept or tuned (k, k1, b, threshold); no significance tests, no MRR, no re-ranker.

**Tags**: [assignment] = TF-IDF, BM25, Sentence Transformer, RRF k≈60, P@10/Recall/MAP/nDCG@10, preprocessing steps, three outputs, QQP. [project] = everything in `DECISIONS.md` and the code. [general] = textbook reasoning.

<!-- END PART H -->

---

## Part I — Course topics map (CS4144)

**Source note:** no CS4144 handout or syllabus is in `docs/` (it contains only `CODEBASE_WALKTHROUGH.md`, `DEPLOYMENT.md` and the report template). This map therefore follows the standard textbook syllabus of Manning, Raghavan & Schütze, *Introduction to Information Retrieval* (2008) [general]. Check it against your actual lecture list [CHECK].

| Topic | Must-know (2–3 lines) | Where the project touches it |
|---|---|---|
| **Boolean retrieval** | Queries are logical expressions (AND/OR/NOT) over terms, answered by intersecting or merging sorted postings lists. Output is an unranked set. Intersect the shortest list first (cost O(x + y) per merge). | Not used: all our models rank. It's useful as contrast: a paraphrase with different words fails an AND query. Q2. |
| **Inverted index construction** | Pipeline: tokenise → normalise (case, stemming, stop-words) → sort (term, docID) pairs → dictionary + postings with df. Large collections use blocked sort-based (BSBI) or single-pass in-memory (SPIMI) indexing plus merging. Skip pointers and positional postings support faster and phrase queries. | Our tokenise and normalise steps are exactly this front half (`clean.py`). `inverted_index.py` builds term → [(doc, tf)] postings but isn't used in results. TF-IDF and BM25 score by brute force instead (Part F, item 4). Q3, Q40. |
| **Tolerant retrieval** (k-gram index, spelling) | Character k-gram indexes (with `$` boundary markers) support wildcard queries and spelling correction via k-gram overlap (Jaccard) plus edit distance. | Our `char_wb` 3–5-gram space is the same idea used for scoring: word-boundary-padded character n-grams give sub-word overlap for short queries. Q33. |
| **Term weighting and the vector space model** | Documents and queries are vectors of term weights. Similarity is cosine. Length normalisation stops long documents dominating. SMART notation (e.g. lnc.ltc) names tf / idf / normalisation choices. | TF-IDF ranker: three spaces, cosine via L2-normalised sparse vectors. Unit vectors make cosine = dot product (D.7). Q4–Q5, Q28–Q31. |
| **TF-IDF** | tf rewards repeated terms in a document; idf = log(N/df) rewards rare terms; product, often with sublinear tf and normalisation. Variants differ in tf scaling, idf smoothing and normalisation. | scikit-learn's form: raw tf, idf = ln((1+N)/(1+df)) + 1, L2 norm (verified). Toy computation in D.1. The one-term collapse causes cosine 1.0 ties (D.3). |
| **Probabilistic model** (PRP, BIM, BM25) | Probability Ranking Principle: rank by P(relevant \| d, q). The Binary Independence Model leads to RSJ term weights. BM25 adds tf saturation (k1) and length normalisation (b) to the BIM idf. | BM25 via rank_bm25 (k1 = 1.5, b = 0.75, ε floor). Worked example in D.2. BM25 > TF-IDF on all four metrics here. Q36–Q39. |
| **Language models for IR** | Query likelihood: rank documents by P(q \| M_d), with smoothing (Jelinek–Mercer, Dirichlet) so unseen terms don't zero the score. | Not implemented; it's an alternative to BM25 (C.7). |
| **Evaluation** | Cranfield paradigm: collection + queries + relevance judgements. Set metrics: P, R, F. Ranked metrics: P@k, R-precision, MAP, nDCG, MRR. PR curves (interpolated in the textbook). Pooling for judgements; kappa for annotator agreement; significance tests. | The entire Section 4 of the report: our own qrels, P@10 / R@10 / MAP@10 / nDCG@10, a cutoff-based (non-interpolated) PR curve, the P@10 ceiling of 0.184, incomplete qrels (pooling would help), no significance tests. Part D.6, Q69–Q77. |
| **Web search basics and near-duplicate detection** | Web scale, spam, user intent. Near-duplicates are detected with w-shingles, Jaccard similarity and MinHash sketches. | Our task is *semantic* duplicate detection (paraphrase), which shingling can't catch: "How do I speak English fluently?" vs "How can I take my spoken English to a higher level…" share few shingles. The dense model handles this; lexical overlap fails. |
| **Link analysis (PageRank, HITS)** | PageRank: the stationary distribution of a random surfer who follows out-links with probability d (often 0.85) and teleports otherwise. PR(p) = (1−d)/N + d·Σ_{q→p} PR(q)/out(q), computed by power iteration. HITS: mutually reinforcing hub and authority scores. | Not applicable: QQP questions have no link graph, and we rank by content only. If asked, explain PageRank correctly and say a link-based prior wasn't relevant to this dataset. |
| **Dense retrieval and embeddings** (beyond the classic textbook) | Neural bi-encoders map text to vectors, with approximate nearest-neighbour search (FAISS: flat, IVF, HNSW, PQ). Cross-encoders re-rank. Hybrid lexical and dense retrieval, fused by score or rank (RRF). | MiniLM-L6 bi-encoder + FAISS IndexFlatIP + RRF hybrid. Parts C.10–C.12, Q44–Q61. |

**How to use this in the viva:** if a question drifts to a course topic, give the must-know summary first, then the one-sentence link to the project. For topics we didn't use (Boolean, language models, PageRank), say so plainly and explain the concept correctly; don't stretch a link that isn't there.

<!-- END PART I -->
