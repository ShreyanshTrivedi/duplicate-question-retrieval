# Decisions Log

- $(date +%Y-%m-%d): QQP reframed from pairwise classification to retrieval
  via qrels built from is_duplicate=1 pairs. Corpus = deduplicated unique
  questions. Positive pairs create bidirectional qrels (q1->q2 and q2->q1).
  A fixed number of eval queries are held out from the pool of questions 
  that have at least one valid duplicate.
- $(date +%Y-%m-%d): Dense retrieval embeddings are computationally expensive. 
  To ensure FAISS index building and encoding fit reasonably in memory and time 
  during rapid iteration, the corpus is optionally subsampled (e.g., to 100k docs).
- 2026-09-27: Short-text token collapse was inflating TF-IDF. Aggressive
  stopword removal + stemming reduced some queries (e.g. "What is the google?")
  to a single stem, so cosine similarity tied at 1.0 across non-identical docs.
  Fix: light preprocessing mode (no stopword/stem) for TF-IDF and BM25; TF-IDF
  word ngrams (1,2) plus char_wb 3–5 fallback when the word vector is still
  degenerate. Dense is unchanged. Held-out eval (400 queries) before → after:
  TF-IDF P@10 0.1283→0.1158, Recall@10 0.8198→0.7301, MAP@10 0.6642→0.5517,
  nDCG@10 0.7274→0.6184; BM25 0.1345→0.1323 / 0.8442→0.8302 / 0.7013→0.6920 /
  0.7613→0.7526; Dense unchanged at 0.1738 / 0.9659 / 0.9087 / 0.9431;
  Hybrid-RRF 0.1688→0.1673 / 0.9498→0.9416 / 0.8838→0.8643 / 0.9228→0.9081.
  Lexical drops reflect removal of 1.0-tie inflation, not a regression in
  ranking sanity (spot-check qid=1182: TF-IDF 4/33, BM25 8/33, Dense 8/33,
  Hybrid 9/33 hits in top-10).
- 2026-09-27: Segmentation showed the first light-preprocessing fix was
  over-applied to long queries and documents: long-query TF-IDF fell below
  the original aggregate behavior even though the collapse bug was short-text
  specific. Second corrective fix: select preprocessing per text using
  `short_text_word_threshold` (default 5) in both TF-IDF and BM25, at fit and
  rank time. Short text retains stopwords and surface forms; long text uses
  the original stopword-removal and stemming pipeline. The completed scoped
  evaluation restored long-query TF-IDF to 0.1256 / 0.7949 / 0.6564 / 0.7148
  (P@10 / Recall@10 / MAP@10 / nDCG@10), close to the original aggregate.
- 2026-09-27: The per-document scoped approach was rejected after concrete
  qrel inspection showed vocabulary fragmentation inside one shared
  vectorizer. Short queries retained surface forms and bigrams while longer
  duplicate documents used stopword removal and stemming, producing examples
  such as `geographic` vs `geograph` and query-only bigrams. Final decision:
  maintain independent corpus-wide light+bigram and heavy+unigram TF-IDF
  spaces, plus a corpus-wide character space used for short queries, and fuse
  ranked lists with the existing RRF formula. BM25 uses equivalent uniform
  light/heavy spaces. Final segmented TF-IDF: short 0.1450 / 0.8389 / 0.7225 /
  0.7908; long 0.1294 / 0.8335 / 0.6565 / 0.7247.
