# Presentation Script — 4 speakers, about 4 to 5 minutes each

Spoken style, roughly 130 words per minute. Text in [brackets] is a cue (what to show or point at), not something to say. Every number comes from the report. Each speaker ends with a handoff line, so the talk flows.

| Speaker | Report sections | Words | Time |
|---|---|---|---|
| 1 | 1 Introduction, 2 About Dataset | about 640 | 4.5 to 5 min |
| 2 | 3 Methodology, 3.1, 3.2, 3.3 | about 650 | 4.5 to 5 min |
| 3 | 3.4, 4 Setup and metrics, up to Table 5 | about 600 | 4.5 min |
| 4 | 4 after Table 5 (ceiling, results, segmented analysis), 5 Conclusion | about 680 | 5 min |

Rule for everyone: if you are not sure of a number, say "approximately" or read it from the report. Never say "held-out". Say "evaluation queries" and "development-set figures".

---

## Speaker 1 — Introduction and Dataset (Sections 1 and 2)

**[Title slide]**

Good morning. Our project is called Duplicate Question Retrieval on Quora Question Pairs. I'll explain the problem and the data; my teammates will explain the methods and the results.

**[Introduction]**

Question-and-answer sites like Quora collect many versions of the same question. That splits the answers across several threads. So when a new question arrives, we want to find the earlier questions that mean the same thing.

We treat this as a search problem. The new question is the query. All earlier questions are the collection. A question counts as relevant if annotators labelled it a duplicate of the query.

This is different from normal web search in one important way. Both the query and the documents are single short sentences, and relevance means the same intent, not just the same topic. For example, "Which smartphone should I buy under 15 thousand rupees?" and "under 40 thousand?" share almost every word, but they are not duplicates.

There are three families of methods. Sparse lexical methods, TF-IDF and BM25, score a document by how many weighted words it shares with the query. They can't see a paraphrase that shares few words. Dense retrieval turns each question into a vector with a Sentence Transformer and ranks by vector similarity, so it handles re-worded questions, but it can miss rare exact words. A hybrid merges two rankings, and we use Reciprocal Rank Fusion, which works from ranks alone so no score adjusting is needed.

In this report we turn the dataset into a search benchmark, compare four rankers on four metrics, describe a dual-space indexing scheme for short questions, and provide a Streamlit search interface.

**[Section 2: dataset, show Table 1]**

The data is the Quora Question Pairs file. It has 404,290 pairs of questions, each with a yes-or-no duplicate label, and 149,263 of the pairs are duplicates. That is a classification format. It has no collection and no relevance judgements, so we built them.

First, the corpus: every distinct question, 537,929 in all. Second, the qrels, which means the answer key. Every duplicate pair is stored in both directions, so if A duplicates B, then B also duplicates A. That gives 298,526 rows over 149,650 question IDs. Non-duplicate pairs add no rows; those questions simply stay in the collection as distractors. One consequence: a true duplicate that nobody labelled counts as not relevant, so our scores are slightly conservative on that point.

Third, the evaluation queries: 400 questions picked at random with a fixed seed of 42, from the questions that have at least one qrel. Forty of them have at most five words, and 360 have more. They stay in the collection; only a document with identical text is removed from its own results, so a query can't find itself.

**[Figure 1]**

Figure 1 shows the data. On the left, question length. The median is 10 words, and 8.2 percent of questions have five words or fewer, shown in dark. These short questions matter later, because that is where text cleaning can erase most of the question.

On the right, the number of relevant documents per query, on a log scale. 294 of the 400 queries have exactly one relevant document. The average is 2.19 and the maximum is 33. This will matter when we discuss precision, because it limits how high precision can go.

**[Table 2]**

Table 2 shows real examples. Pair 1 is a paraphrase with little word overlap. Pair 2 is a three-word question. Pair 3 is not a duplicate, but shares almost every word. That is why this problem is harder than keyword matching.

I'll now hand over to my teammate to explain the methodology.

**Likely question for you:** "Why are there so few relevant documents per query?" → Quora only labelled pairs that were shown to annotators, and only about 37 percent of pairs are duplicates. The relation is sparse, and that is a property of the dataset, not our choice.

---

## Speaker 2 — Methodology (Sections 3, 3.1, 3.2, 3.3)

**[Figure 2: pipeline]**

Thank you. I'll explain how the system works.

All four rankers share one interface, and none is trained: the qrels are used only for scoring. Figure 2 shows the pipeline. The data becomes a corpus, qrels and queries. TF-IDF, BM25 and Dense are built on it. The Hybrid combines BM25 and Dense. Then we evaluate against the qrels, and the Streamlit interface shows the results.

**[3.1 Preprocessing and indexing]**

First, text preprocessing. We make the text lowercase, replace punctuation with spaces, and split it into words with NLTK. We have two modes. Heavy mode also removes common English stop-words and shortens words with the Porter stemmer. Light mode stops after splitting the words. A text counts as short if it has at most five words.

TF-IDF and BM25 are built once over all 537,929 questions. TF-IDF stores sparse matrices in scikit-learn and compares a query with every row. BM25 uses the rank_bm25 library, which goes through every document's word counts for each query word. Neither uses an inverted index at query time.

The dense retriever works differently. It encodes the raw question into a 384-number vector with the model all-MiniLM-L6-v2, and stores the vectors in a FAISS flat index, which does exact search. To keep encoding manageable, the dense index holds 100,000 questions: the 400 queries and their relevant documents, 1,276 questions in all, plus a random fill. We'll come back to that as a limitation.

**[3.2 and Table 3]**

Table 3 shows our design choices. "Spec" means the assignment specified it, like TF-IDF, BM25, a Sentence Transformer and RRF. "Ours" means our choice, like FAISS and Streamlit. We did not benchmark the alternatives; they're listed for context.

**[Equations]**

TF-IDF turns a text into a vector over words. A word's weight is its count times an inverse-document-frequency term, so rare words count more. Each vector is scaled to length one, so similarity is the cosine.

BM25 scores a document with the formula on the slide. k1 limits the gain from repeated words, and b controls how much length is penalised. We used the library defaults, k1 equals 1.5 and b equals 0.75, untuned. BM25 scores have no upper limit, so they can't simply be added to cosine scores.

Dense retrieval ranks by cosine between the query vector and each question vector. Because it compares meaning instead of words, it can match a re-worded duplicate.

Hybrid-RRF merges two lists of 100 documents: the output of BM25, and the dense list. TF-IDF is not included. The BM25 list is itself an RRF merge of its own two spaces, so the hybrid is RRF over an RRF-merged list. The dense side also searches the same 100,000 questions.

**[3.3 and Table 4]**

Now the RRF formula. For each document, we add up one divided by sixty plus its rank, over every list that contains it. A list that does not contain the document adds nothing. Only positions are used, so scores from different methods need no adjusting. The value 60 comes from the assignment and from the original paper.

Table 4 is a small example. Document B is rank 2 in one list and rank 1 in the other, and gets 0.0325. Document E is rank 5 in both lists and gets 0.0308, which is higher than D, which is rank 2 in only one list and gets 0.0161. So agreement between lists beats a single good rank. The same function performs the merging inside TF-IDF, inside BM25, and in the hybrid.

Over to my teammate for the short-text problem.

**Likely question for you:** "Why is TF-IDF not in the hybrid?" → The assignment defined the hybrid as lexical plus dense, and we used BM25 as the lexical half. We did not test adding TF-IDF.

---

## Speaker 3 — Short-text indexing and the evaluation setup (Section 3.4, Section 4 up to Table 5)

**[3.4 and Figure 3]**

Thank you. This is the most interesting engineering part of the project.

Take the query "What is the google?" Heavy preprocessing removes what, is and the, and the stemmer turns "google" into "googl". The query becomes a single word. Any document that also shrinks to that single word has exactly the same vector as the query, so its cosine is exactly 1.0. Many different questions tie at 1.0 and their order is arbitrary. In our data, six of the 40 short evaluation queries shrank to zero or one word.

We tried two fixes, and rejected both. The first was global light mode: no stop-word removal or stemming for any text. That removed the ties but lowered TF-IDF on long queries below its original level. The second was per-text selection: light for short texts and heavy for long ones, inside one shared vocabulary. That fixed long queries but broke the vocabulary. A short query kept the word "geographic", while its longer duplicate was stemmed to "geograph", so they never matched.

The design we adopted is dual-space indexing with rank fusion. Every question is indexed in separate spaces over the whole collection. A space is one version of the index built with one set of text rules. The light space keeps plain words plus word pairs. The heavy space removes stop-words and stems. And for TF-IDF only, a character space uses groups of 3 to 5 letters inside words, but only for queries of five words or fewer. A query is processed in each space exactly like that space's documents. Each space returns up to 100 documents, and the lists are merged with RRF. BM25 uses the light and heavy spaces for every query.

**[Point at Figure 3]**

Figure 3 shows the three lanes for "What is the google?" The top lane is heavy and gives just "googl". The middle lane is light. The bottom lane is characters. Three ranked lists go into RRF. One important point: short and long queries don't use separate systems. Every query uses the same index; the only difference is that short queries also get the character list. The five-word threshold was chosen by looking at the data and was not tuned.

**[Section 4: Setup]**

Now the experimental setup. We run all 400 evaluation queries through all four rankers and look at the top 10 results, and the top 50 for the precision-recall curves. Nothing is trained on the qrels, and the queries stay in the collection.

An honesty point you should know: we chose the short-text design by looking at results on these same 400 queries. We had no separate development split. So all our figures are development-set figures, and are probably optimistic. We don't call them held-out.

**[Table 5: metrics]**

Table 5 defines the four metrics, all at cutoff 10. R is the set of relevant documents for a query.

Precision at 10 is the number of hits in the top 10, divided by 10. Recall at 10 is the number of hits divided by the total number of relevant documents. Average precision rewards putting relevant documents early: at each hit, we note the precision so far, add those up, and divide by the number of relevant documents. MAP is its mean over all queries. One detail: we divide by the full count of relevant documents, not by the smaller of 10 and that count, which gives a lower value for the 13 queries with more than ten relevant documents. Finally nDCG gives each hit a credit of one over log of rank plus one, so earlier hits count more, then divides by the best possible total.

Now my teammate will present the results.

**Likely question for you:** "Why did the per-text design look fine but fail?" → Its overall score, 0.6784, hid a collapse to 0.3508 on short queries. Only splitting the evaluation by query length revealed it.

---

## Speaker 4 — Results, Segmented analysis and Conclusion (Section 4 after Table 5, Section 5)

**[Precision ceiling]**

Thank you. Before the results, one point that explains why precision looks low. Most queries have only one relevant document. A perfect ranker can fill at most that many of the 10 slots. Averaged over all 400 queries, the exact ceiling for precision at 10 is 0.184. So the best system's 0.1738 is already 94 percent of the maximum. That's why we look at recall, MAP and nDCG for the real comparison.

**[Table 6 and Figure 5]**

Table 6 shows the results. Dense retrieval is best on every metric: nDCG at 10 of 0.9431, MAP of 0.9087, and Recall at 10 of 0.9659. Hybrid-RRF is second on every metric, with nDCG of 0.9251. BM25 beats TF-IDF on all four, with nDCG of 0.7755 against 0.7314.

At rank 1, Dense's precision is 0.9075, so it puts a correct duplicate first for about nine of ten queries. In Figure 5b the order is Dense, Hybrid, BM25, TF-IDF at every cutoff from 1 to 50. At 50, Dense recall is 0.999, but BM25 only 0.945 and TF-IDF 0.929.

**[Why dense wins, with caveats]**

Why does Dense win? A likely reason is that duplicates are paraphrases, and the encoder captures meaning. But we did not run an experiment to prove that, and there are two caveats. Dense searches 100,000 questions, while TF-IDF and BM25 search 537,929, about 5.4 times more. And the model card says the encoder was fine-tuned on Quora question triplets, so it may have seen related pairs. Both could inflate Dense by amounts we did not measure. The hybrid sits below Dense, perhaps because equal-weight fusion lets the weaker BM25 list pull down the dense list. We did not test that either.

**[Table 7 and Figure 4]**

Now the segmented analysis. Table 7 splits the lexical models by query length. TF-IDF does better on short queries, 0.7908, than on long ones, 0.7247. BM25 does the opposite. We did not test why, and with only 40 short queries the difference may be noise.

Figure 4 follows TF-IDF through the designs. The original design scored 0.7274 but was inflated by the ties. The per-text design looked fine overall at 0.6784 but collapsed to 0.3508 on short queries. Our final design is better on both groups, and its overall score of 0.7314 is within 0.004 of the original, without the ties behind it. The lesson: an average can hide a failure on a smaller group.

**[Section 5: Conclusion]**

To conclude: similarity of meaning was the strongest signal. Dense led on every metric. Hybrid came within 0.018 nDCG of it while keeping an exact-match component, and BM25 beat TF-IDF. Dual-space indexing removed the ties.

Our limitations are these. First, development-set figures, probably optimistic. Second, unequal pools, which favour Dense and the hybrid. Third, BM25 is untuned, we did not test other values of k, and the short group has just 40 queries. Fourth, the encoder's training data included Quora pairs. Fifth, unlabelled true duplicates count as wrong. And finally speed: BM25 takes about 2 seconds per query and start-up takes about 7 minutes in one CPU run, which is slow for interactive use.

For future work: a separate development split, equal-sized pools, tests of other values of k, a re-ranker, and tuned BM25.

Thank you. We're happy to take questions.

**Likely question for you:** "Is Dense really better, or is it an unfair comparison?" → On this benchmark it is better, but the comparison favours it in two ways we did not measure, the smaller pool and the possible training overlap. So we'd say it's better here, not that we've proven by how much.

---

## Timing check

Practise once with a timer. If a speaker runs over, cut in this order: Table 2 and example pairs (Speaker 1), the Table 3 discussion (Speaker 2), the segmented-analysis detail (Speaker 4), and the speed limitation (Speaker 4).
