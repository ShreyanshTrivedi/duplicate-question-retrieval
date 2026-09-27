import math
from collections import defaultdict
from src.retrieval.base import RetrieverBase
from src.preprocessing.clean import preprocess

class InvertedIndexRetriever(RetrieverBase):
    def __init__(self, config: dict):
        self.config = config
        self.index = defaultdict(list)
        self.doc_ids = []
        self.doc_texts = []
        self.num_docs = 0
        
    def fit(self, corpus_df):
        self.num_docs = len(corpus_df)
        for _, row in corpus_df.iterrows():
            doc_id = str(row['qid'])
            text = row['question']
            self.doc_ids.append(doc_id)
            self.doc_texts.append(str(text))
            tokens = preprocess(text, self.config)
            
            tf = defaultdict(int)
            for token in tokens:
                tf[token] += 1
            
            for token, freq in tf.items():
                self.index[token].append((doc_id, freq))
                
    def rank(self, query: str, top_k: int) -> list[tuple[str, float]]:
        tokens = preprocess(query, self.config)
        scores = defaultdict(float)
        
        for token in tokens:
            postings = self.index.get(token, [])
            df = len(postings)
            if df == 0:
                continue
            idf = math.log(self.num_docs / df)
            for doc_id, tf in postings:
                scores[doc_id] += tf * idf
                
        qnorm = query.strip()
        self_ids = {
            doc_id
            for doc_id, text in zip(self.doc_ids, self.doc_texts)
            if text.strip() == qnorm
        }
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        filtered = [(doc_id, score) for doc_id, score in sorted_results if doc_id not in self_ids]
        return filtered[:top_k]
