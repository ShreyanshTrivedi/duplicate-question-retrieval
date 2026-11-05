# Basics and the Whole Pipeline, Explained From Zero

Read this first. It assumes you know nothing about information retrieval. Part 1 teaches the building blocks. Part 2 shows where each block sits in this project. Part 3 walks the whole system from raw data to the screen. Part 4 follows one query through every stage. Part 5 is what the results mean and what to say about them. Part 6 is a self-test.

Numbers marked (real) come from the project's files. Examples marked (toy) are made up to teach, so do not quote them as results.

---

## Part 1 — The basics

### 1.1 What is "information retrieval"?

Information retrieval (IR) is the problem behind every search box: you type something, and a system returns the items most useful to you, best first.

The words you must know:

| Word | Meaning | In this project |
|---|---|---|
| **Collection** (corpus) | All the items that can be returned | 537,929 Quora questions (real) |
| **Document** | One item in the collection | One question |
| **Query** | What the user types | A question, e.g. "What is the google?" |
| **Relevant** | A document that truly answers the need | A question that is a *duplicate* of the query |
| **Ranking** | Putting documents in order, best first | Top 10 shown |
| **Relevance judgements / qrels** | The answer key: which documents are relevant for which query | The duplicate labels from Quora |

Key idea: a search system does **not** return one answer. It returns a **ranked list**, and we judge how good the order is.

### 1.2 Our problem: duplicate question retrieval

Quora is a question-and-answer site. Many people ask the same thing in different words:

- "How do I speak English fluently?"
- "How can I take my spoken English to a higher level when I speak good enough but not fluently?"

Those mean the same thing, but share few words. A site wants to detect that so answers are not split over many threads.

We turn this into search: the new question is the **query**, all older questions are the **collection**, and a document is **relevant** if people labelled it a duplicate of the query.

### 1.3 Classification versus retrieval (why we "rebuilt" the dataset)

The Quora dataset (QQP) comes as **pairs**: question 1, question 2, and a label 1 (duplicate) or 0 (not). That is a *classification* format: "are these two the same?"

Retrieval is different: given one question, find the matching ones among *half a million*. So we reshaped the data:

1. Every distinct question becomes a document → 537,929 documents (real).
2. Every duplicate pair becomes an answer-key entry, in both directions (if A duplicates B, then B duplicates A) → 298,526 rows, which is exactly 2 × 149,263 duplicate pairs (real).
3. Pairs labelled "not duplicate" give no entry. Those questions stay in the collection as ordinary distractors.

Consequence you must be able to state: if the system returns a question that is *really* a duplicate but nobody labelled it, it is counted as **wrong**. The labels are incomplete, so scores are conservative.

### 1.4 Text preprocessing: turning raw text into tokens

Computers compare symbols, not meaning, so we first clean the text. The steps in this project, in order:

1. **Lowercase**: "Google" and "google" become the same.
2. **Replace punctuation with spaces**: "don't" becomes "don t".
3. **Tokenise**: split into words. "what is the google" → [what, is, the, google]. Each piece is a **token**.
4. **Stop-word removal** (heavy mode only): delete very common words (what, is, the, how, do, i). They appear everywhere, so they carry little meaning.
5. **Stemming** (heavy mode only): cut words to a rough root so variants match. The **Porter stemmer** turns "learning", "learned", "learns" into "learn". It is crude: "google" becomes "googl", and "geographic" becomes "geograph".

Two modes exist in this project:

- **Light**: steps 1 to 3 only (keep stop-words, no stemming).
- **Heavy**: steps 1 to 5.

Why both? Heavy helps long questions (variants match), but wrecks very short ones (see 1.12). That is the heart of the project's "short-text" work.

### 1.5 Bag of words, term frequency, and vectors

A simple way to compare texts: ignore word order and just count words. That is a **bag of words**.

Imagine a vocabulary with one slot per distinct word. A text becomes a list of numbers, one per slot, saying how often each word occurs. This list is a **vector**. A text is now a point in a space with as many dimensions as there are words.

- **Term frequency (tf)**: how many times a word occurs in the text.

Two texts that share many words end up pointing in similar directions. We measure that with **cosine similarity** (1.7).

### 1.6 TF-IDF: not all words are equally informative

Counting words has a flaw: "the" occurs everywhere, so matching on it tells you nothing. A rare word like "loneliest" is a far stronger clue.

**IDF** (inverse document frequency) gives rare words more weight:

- df(t) = number of documents containing the word t
- N = number of documents
- idf(t) = ln((1 + N) / (1 + df(t))) + 1 (this is the scikit-learn "smoothed" form, which the project uses; verified in the installed library)

**TF-IDF weight** = tf × idf. A word gets a high weight if it is frequent in *this* text and rare across the *collection*. Each vector is then scaled to length 1 (**L2 normalisation**), so long and short texts compare fairly.

**Toy example.** Collection of N = 3 documents: d1 "how to learn python", d2 "learn python fast", d3 "how to cook rice". Query "learn python".

- "learn" and "python" appear in 2 of 3 documents: idf = ln(4/3) + 1 = 1.288.
- "fast" appears in 1 document: idf = ln(4/2) + 1 = 1.693 (rarer, so a bigger weight).
- Cosine of the query with d1 ≈ 0.71, with d2 ≈ 0.73, with d3 = 0 (no shared word).

So d2 ranks first, d1 second, d3 last. d2 beats d1 because fewer of its words are unmatched (one, "fast", against two for d1: "how" and "to"), so more of its vector is made of the query words. Notice no meaning is involved, only word overlap.

### 1.7 Cosine similarity

Two vectors, **q** and **d**. Cosine similarity measures the angle between them:

cos(q, d) = (q · d) / (‖q‖ ‖d‖)

- 1 means pointing the same way (identical direction).
- 0 means nothing in common.

Why use the angle and not distance? So that text *length* does not matter, only the mix of words. If both vectors have length 1 already, then cos(q, d) is just the dot product q · d. Remember this: it is why the dense search later can use a plain "inner product" index and still compute cosine.

### 1.8 BM25: a better lexical score

BM25 is the other classic word-matching model. It fixes two weaknesses of TF-IDF:

1. **Saturation**: mentioning a word 10 times is not 10 times more relevant than mentioning it once. BM25 flattens the gain. The parameter **k1** controls how fast (k1 = 1.5 here).
2. **Length normalisation**: a long document naturally contains more words. The parameter **b** controls how strongly we penalise length (b = 0.75 here).

Formula, so you recognise it:

score(q, d) = Σ over query terms t of idf(t) · f(t,d)(k1 + 1) / (f(t,d) + k1 (1 − b + b · |d| / avgdl))

where f(t,d) is the term count, |d| the document length, and avgdl the average document length.

Facts you can state: we use the Python library `rank_bm25` (class BM25Okapi) with its **defaults** (k1 = 1.5, b = 0.75), untuned. BM25 scores are **not bounded** (they can be any positive number), unlike cosine which lives in [0, 1] here. That matters for fusion (1.10).

### 1.9 Embeddings and dense retrieval (meaning, not words)

TF-IDF and BM25 are **lexical**: they only see shared words. They fail on paraphrases:

- "How do I speak English fluently?" vs "How can I take my spoken English to a higher level..."

An **embedding** model converts a whole sentence into a list of numbers (a vector) such that sentences with similar *meaning* get nearby vectors, even with different words. It is trained on huge amounts of text.

This project uses **all-MiniLM-L6-v2**, a **Sentence Transformer**: a small neural network that outputs a **384-number vector** per sentence. Retrieval with such vectors is called **dense retrieval** ("dense" because every one of the 384 numbers is non-zero, unlike sparse word-count vectors that are mostly zeros).

How it is used:

1. Encode every document once → a matrix of vectors (100,000 × 384 in our dense index, real).
2. At query time, encode the query the same way.
3. Find the document vectors closest to the query vector, using cosine similarity.

Important honesty points (the examiner may ask):

- We **do not train** this model. We use it as shipped.
- Its authors **did** train it on lots of data, and its model card lists Quora Question Triplets (103,663) among them. So it may have seen related Quora pairs. That could inflate its score.
- It is a **bi-encoder**: query and document are encoded separately, so documents can be encoded ahead of time. This is fast, but less precise than a *cross-encoder* that reads both texts together (we did not use one).

### 1.10 Finding nearest vectors fast: FAISS

With 100,000 vectors of 384 numbers, you must compare the query against every vector. **FAISS** is a library for exactly this.

We use `IndexFlatIP`: "flat" means brute-force exact comparison with every vector, "IP" means **inner product**. Since all vectors are normalised to length 1, inner product equals cosine (from 1.7). Exact search means no approximation errors; for 100,000 vectors it is fast enough.

Alternatives that exist (approximate indexes such as IVF or HNSW) are faster on millions of vectors but can miss results. We did not build them.

### 1.11 Rank fusion: combining several ranked lists (RRF)

Suppose BM25 gives one ranking and the dense model gives another. How do you merge them? You cannot just add scores: BM25 can be 14.3 while cosine is 0.82. Different scales.

**Reciprocal Rank Fusion (RRF)** ignores scores and uses only **positions**. For every document, add up, over each list that contains it:

RRF(d) = Σ over lists of 1 / (k + rank(d)), with k = 60 and rank starting at 1.

**Toy example** (k = 60). List 1 = [A, B, C], list 2 = [B, D, A].

- A: 1/61 + 1/63 = 0.0323
- B: 1/62 + 1/61 = 0.0325 (rank 2 in one list, rank 1 in the other)
- D: 1/62 = 0.0161 (only in one list)
- C: 1/63 = 0.0159 (only in one list)

Fused order: B, A, D, C. The idea: a document that *both* lists like beats one that only a single list likes.

What k does: with larger k the curve is flatter, so the difference between rank 1 and rank 5 shrinks and "agreement" matters more. k = 60 is the standard value from the original paper (Cormack et al.) and the assignment. We did not tune it.

### 1.12 The short-text problem (central to the project)

Short questions break the heavy pipeline. Take "What is the google?":

1. Stop-words removed: what, is, the → gone. Only "google" remains.
2. Stemmed: "googl".

So the query is a **single token**. Any document that also reduces to the single token "googl" gets an *identical* vector, so its cosine with the query is exactly **1.0**. Many documents tie at 1.0 and their order is arbitrary. In the dataset, 6 of the 40 short evaluation queries (≤ 5 words) collapsed to zero or one token (real).

### 1.13 Evaluation metrics

To compare systems we need numbers. Four are used, all "at 10" (looking only at the top 10 results).

Setup: R = set of truly relevant documents for the query, |R| = how many there are.

- **Precision@10 (P@10)**: of the 10 shown, what fraction are relevant? = hits / 10.
- **Recall@10**: of all relevant documents, what fraction appear in the top 10? = hits / |R|.
- **AP@10 / MAP@10**: Average Precision rewards putting relevant items *early*. At each rank where a relevant item appears, note the precision so far; add those up and divide by |R|. MAP is the mean of AP over all queries. (In our code the divisor is |R|, not min(10, |R|); this is lower than the usual convention only when a query has more than 10 relevant documents.)
- **nDCG@10**: gives each hit a credit of 1 / log2(rank + 1), so earlier hits count more, then divides by the best possible total for that query. It ranges from 0 to 1.

**Toy worked example.** Relevant set R = {A, C, F}, so |R| = 3. The system's top 10 is [A, B, C, D, E, G, F, H, I, J]. Hits are at ranks 1, 3 and 7.

- P@10 = 3/10 = 0.30
- Recall@10 = 3/3 = 1.00
- AP = (1/1 + 2/3 + 3/7) / 3 = 0.698
- DCG = 1 + 0.5 + 0.333 = 1.833; ideal DCG (hits at ranks 1, 2, 3) = 1 + 0.631 + 0.5 = 2.131; nDCG = 0.860

**Why P@10 is tiny in this project.** Most queries have only one relevant document (294 of 400; the average is 2.19, real). Even a perfect system can fill at most that many slots out of 10, so P@10 has a ceiling. The exact ceiling is the mean of min(|R|, 10)/10 = 0.184 (real). Dense's 0.1738 is about 94% of it. That is why P@10 looks low even though the system is good. Read Recall, MAP and nDCG for the story.

### 1.14 Development set, tuning and why numbers are "optimistic"

In machine learning you normally split data into training, development (tuning) and test sets. Here nothing is *trained* on our labels, but we made design decisions (light mode, per-text mode, dual-space) by looking at results on the same 400 queries we report. There is no untouched test set. So our numbers are **development-set figures** and probably a bit **optimistic**. Say this openly; do not call them "held-out". It is the single most important honesty point.

### 1.15 The tools

- **Python**: the programming language.
- **pandas / parquet**: tables of data; parquet is a compact file format for them.
- **NLTK**: a text library (tokeniser, stop-word list, Porter stemmer).
- **scikit-learn**: provides TfidfVectorizer and cosine similarity.
- **rank_bm25**: BM25.
- **sentence-transformers**: loads the MiniLM model.
- **FAISS**: vector search.
- **Streamlit**: turns a Python script into a small web app (the search interface).

---

## Part 2 — How each basic fits this project

| Basic | Where it appears in the project |
|---|---|
| Collection, documents | `corpus.parquet`: 537,929 questions |
| Relevance judgements | `qrels.parquet`: 298,526 rows built from duplicate labels |
| Queries | `eval_queries.parquet`: 400 random questions that have at least one duplicate (seed 42) |
| Preprocessing (light and heavy) | `src/preprocessing/clean.py` |
| TF-IDF and cosine | `src/retrieval/tfidf.py` (three spaces, see 3.4) |
| BM25 | `src/retrieval/bm25.py` (two spaces) |
| Embeddings + FAISS | `src/retrieval/dense.py` |
| RRF | `reciprocal_rank_fusion()` in `src/retrieval/hybrid_rrf.py`, used three times |
| Metrics | `src/evaluation/metrics.py` |
| Short-text collapse | The reason for the dual-space design |
| Dev-set caveat | Limitations section of the report |
| Streamlit | `src/ui/app.py` |

---

## Part 3 — The whole pipeline, from scratch

### 3.1 Step 1: get the data

`src/data/download.py` downloads the Quora Question Pairs file: 404,290 labelled pairs, of which 149,263 are duplicates (real). Each row: ids and text of two questions, plus a 0/1 label.

### 3.2 Step 2: build the retrieval benchmark

`src/data/build_corpus.py` turns pairs into a search problem:

1. **Corpus**: stack question 1 and question 2 columns, drop duplicates by question id → 537,929 unique questions.
2. **Qrels**: for each pair labelled duplicate, store (A→B) and (B→A) → 298,526 rows over 149,650 query ids.
3. **Evaluation queries**: randomly pick 400 questions that have at least one qrel (seed 42, reproducible).

Two details to remember:

- The 400 queries **stay in the collection**. Only an exact-text copy of the query is removed from its own results (otherwise it would find itself).
- Removal is by *text*, so a different question with identical wording would also be dropped. Rare, unmeasured.

### 3.3 Step 3: preprocessing

Every text goes through `clean.preprocess` in light or heavy mode (1.4). A text is "short" if it has at most 5 whitespace-separated words. The 5 was chosen by looking at the data, never tuned. Of the 400 queries, 40 are short and 360 long.

### 3.4 Step 4: build the four rankers

All four share one interface: `rank(query, top_k)` returns a list of (document id, score), best first.

**(a) TF-IDF retriever: three spaces, fused.** One vocabulary cannot serve both short and long texts, so the same collection is indexed three separate ways:

- **Light space**: lowercase tokens (stop-words kept, no stemming) *plus bigrams* (adjacent word pairs, e.g. "what__is"). Keeps surface form and short phrases.
- **Heavy space**: stop-words removed, stemmed, single words only. Good for long questions.
- **Character space**: overlapping groups of 3 to 5 characters within words ("goo", "oog", "ogl", ...). Gives evidence even when word matches are scarce. Used **only for queries of five words or fewer**.

Each space gives its own top-100 list (cosine against every document). The lists are fused with RRF (k = 60). TF-IDF stores sparse matrices and computes cosine against all 537,929 rows; it does **not** use an inverted index at query time.

**(b) BM25 retriever: two spaces, fused.** Light and heavy spaces, each scored by BM25 (`rank_bm25`), top-100 each, fused with RRF. It has no character space and uses both spaces for every query. `rank_bm25` loops over every document for each query term in Python, which is why BM25 is slow: about 2.0 s per query (real, approximate, single CPU run).

**(c) Dense retriever.** Encode questions with MiniLM into unit vectors, store in a FAISS flat inner-product index, search with the encoded query. **Only 100,000 questions** are indexed, not 537,929: all 400 queries and their relevant documents (1,276 questions) are forced in, plus a seeded random fill. This was done to keep encoding manageable. It means Dense searches a pool about 5.4 times smaller than TF-IDF and BM25, which makes its job easier by an amount we did not measure.

**(d) Hybrid-RRF.** Takes the BM25 retriever's top-100 (which is *already* an RRF fusion of two spaces) and the Dense top-100, and fuses those two lists with RRF again. So it is "RRF over an RRF-fused list", and TF-IDF is **not** part of it. The dense half also uses the 100,000-question index.

### 3.5 Step 5: the short-text story (what we found and fixed)

This is the engineering narrative; know the order.

1. **Original design**: heavy preprocessing for everything. Short queries collapsed ("What is the google?" → [googl]) and produced tied cosine scores of 1.0. TF-IDF nDCG@10 was 0.7274, but partly inflated by arbitrary ties.
2. **Attempt 1 — global light mode** (no stop-words/stemming for any text): removed the ties but hurt long queries (aggregate 0.6184).
3. **Attempt 2 — per-text mode** (light for short texts, heavy for long, inside one shared vocabulary): fixed long queries but fragmented the vocabulary. A short query kept "geographic" while its longer duplicate was stemmed to "geograph", so they never matched; bigrams existed only on the query side. Aggregate looked fine (0.6784) but short queries had collapsed to 0.3508. Only **length-segmented evaluation** (scoring short and long queries separately) revealed this.
4. **Final design — dual-space indexing with rank fusion** (3.4a). Short-query nDCG@10 is 0.7908, long 0.7247, aggregate 0.7314 (real).

Lesson to say aloud: an average can hide a failure on a minority group, so we looked at the groups separately.

### 3.6 Step 6: evaluation

`src/evaluation/run_evaluation.py` runs each of the 400 queries through each of the four rankers:

- takes the top 10 for the four metrics (P@10, Recall@10, MAP@10, nDCG@10),
- takes the top 50 to draw the precision-recall curve (precision and recall averaged at every cutoff k = 1 to 50),
- writes `reports/comparison_table.csv` and `pr_curve_data.json`.

`run_segmented_evaluation.py` splits queries into short (40) and long (360) for TF-IDF and BM25.

### 3.7 Step 7: the user interface

`src/ui/app.py` is a Streamlit app with three tabs:

- **Search**: type a question, choose a method, see the top 10 results. It builds all four rankers once at start-up (about 7 minutes on this machine, real), then answers queries.
- **Performance Comparison**: shows the results table.
- **Precision-Recall**: shows the curves.

The last two tabs only read the saved report files.

### 3.8 Step 8: the report

`src/reporting/build_report.py` generates the Word report from the saved CSV and JSON files, so every number in it traces to a file.

---

## Part 4 — One query, end to end

Query: **"What is the google?"** (4 words, so "short").

**TF-IDF path**
1. Light space: tokens [what, is, the, google] plus bigrams (what__is, is__the, the__google). Cosine against all 537,929 light vectors → top 100.
2. Heavy space: stop-words removed → [google] → [googl]. Cosine against all heavy vectors → top 100. (This is the lane that used to produce 1.0 ties.)
3. Character space (query is short): overlapping 3 to 5 character pieces ("wha", "hat", ...). Cosine → top 100.
4. RRF over the three lists → TF-IDF's final top 10.

**BM25 path**: BM25 scores in light and heavy spaces → two top-100 lists → RRF → top 10.

**Dense path**: encode the raw sentence with MiniLM → a 384-number unit vector → FAISS finds the nearest of the 100,000 stored vectors → top 10 by cosine.

**Hybrid path**: BM25's fused top-100 + Dense's top-100 → RRF → top 10.

In every path the query's own text is removed from the list. During *evaluation* each returned id is checked against the answer key; the four metrics are computed; averaged over 400 queries.

---

## Part 5 — What the results mean

Real results on the 400 evaluation queries (cutoff 10):

| Method | P@10 | Recall@10 | MAP@10 | nDCG@10 |
|---|---|---|---|---|
| TF-IDF | 0.1310 | 0.8341 | 0.6631 | 0.7314 |
| BM25 | 0.1368 | 0.8582 | 0.7129 | 0.7755 |
| Dense | 0.1738 | 0.9659 | 0.9087 | 0.9431 |
| Hybrid-RRF | 0.1708 | 0.9535 | 0.8843 | 0.9251 |

How to read it:

- Dense is best on every metric. Duplicates are *paraphrases*, and embeddings capture meaning. But two caveats: it searches a smaller pool, and its encoder saw Quora data in training. Both could inflate it, by unmeasured amounts.
- Hybrid is second, below Dense. A plausible reason is that the weaker BM25 list dilutes the strong dense list under equal-weight fusion. We did not test this.
- BM25 beats TF-IDF: saturation and length normalisation help.
- At rank 1, Dense is right about 9 times in 10 (precision at k = 1 is 0.9075).
- P@10 looks small only because of the ceiling (0.184).

Things you must not claim:

- "Held-out test set." Say "evaluation queries; development-set figures".
- "We tested why Dense wins / why Hybrid is lower." We did not.
- "We tuned k, BM25 parameters or the 5-word threshold." We did not.
- "It uses an inverted index." It does not at query time.

Speed (approximate, one CPU run): BM25 about 2.0 s per query, TF-IDF 0.57 s, Dense 0.035 s, Hybrid 2.09 s; start-up about 7.3 minutes. Hybrid is slow mostly because BM25 is.

Ten sentences to be able to say without notes:

1. We turned a pair-classification dataset into a retrieval benchmark: 537,929 documents, 298,526 answer-key rows, 400 queries.
2. TF-IDF and BM25 match words; the dense model matches meaning.
3. RRF merges ranked lists using only positions, with k = 60.
4. The hybrid is RRF over BM25's own fused list plus the dense list.
5. Short questions collapsed to one token under heavy preprocessing and tied at cosine 1.0.
6. Two fixes failed in ways only per-length evaluation exposed; the final dual-space design works.
7. Dense wins, then Hybrid, BM25, TF-IDF.
8. P@10 is capped near 0.184 by the number of relevant documents.
9. Our figures are development-set figures and probably optimistic.
10. Dense searches 100,000 questions while the lexical models search 537,929, and its encoder saw Quora data.

---

## Part 6 — Self-test (answer out loud, then check)

1. What are a query, a document and a qrel in this project?
2. Why did we convert QQP from pairs to a collection?
3. Name the five preprocessing steps and which two are "heavy only".
4. What does IDF do, and why is "the" a poor search word?
5. Why does cosine similarity ignore text length?
6. What do k1 and b do in BM25?
7. Why can't we add a BM25 score and a cosine score directly?
8. Compute RRF for list 1 = [X, Y], list 2 = [Y, Z] with k = 60.
9. Why is dense retrieval able to find paraphrases?
10. Why can FAISS use inner product to compute cosine here?
11. What exactly collapsed in "What is the google?" and why did it tie at 1.0?
12. Why did the per-text mode fail, and what showed it?
13. Which components are in the TF-IDF retriever and when is the character space used?
14. What is inside Hybrid-RRF, and is TF-IDF part of it?
15. For a query with R = {A, C}, top 10 = [B, A, D, C, E, F, G, H, I, J], compute P@10, Recall@10 and AP. (Answers: 0.2, 1.0, (1/2 + 2/4)/2 = 0.5.)
16. Why is P@10 only about 0.17 for the best system?
17. Name three reasons the dense score might be flattered.
18. Why do we say "development-set figures" and not "held-out"?

When you can answer all of these without looking, you understand the project at the depth a viva needs. For deeper, line-by-line defence of each number, use `docs/VIVA_MASTER_STUDY_GUIDE.md`; for code locations use `docs/CODEBASE_WALKTHROUGH.md`.
