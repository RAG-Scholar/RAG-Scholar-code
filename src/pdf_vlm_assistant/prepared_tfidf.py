"""Reuse fixed document statistics for the existing query-inclusive TF-IDF."""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

from .image_retrieval import _char_ngrams


class PreparedTfidf:
    def __init__(self, documents):
        self.documents = tuple(documents)
        self.count = len(self.documents)
        postings = defaultdict(lambda: ([], []))
        for position, document in enumerate(self.documents):
            for term, count in Counter(_char_ngrams(document)).items():
                ids, counts = postings[term]
                ids.append(position)
                counts.append(count)
        self.postings = {}
        self.norm_squared = np.zeros(self.count, dtype=np.float64)
        # Original scoring treats the query as one additional document when
        # computing IDF. Terms present in the query get a +1 document frequency.
        for term, (ids, counts) in postings.items():
            ids = np.asarray(ids, dtype=np.int32)
            counts = np.asarray(counts, dtype=np.float64)
            base_idf = math.log((self.count + 2) / (len(ids) + 1)) + 1
            self.norm_squared[ids] += counts * counts * base_idf * base_idf
            self.postings[term] = (ids, counts, base_idf)
        self.last_query, self.last_scores = None, None

    def scores(self, query):
        if query == self.last_query:
            return list(self.last_scores)
        squared_norm = self.norm_squared.copy()
        dot = np.zeros(self.count, dtype=np.float64)
        query_norm_squared = 0.0
        for term, count in Counter(_char_ngrams(query)).items():
            posting = self.postings.get(term)
            frequency = len(posting[0]) if posting is not None else 0
            idf = math.log((self.count + 2) / (frequency + 2)) + 1
            query_norm_squared += count * count * idf * idf
            if posting is not None:
                ids, counts, base_idf = posting
                squared_norm[ids] += counts * counts * (idf * idf - base_idf * base_idf)
                dot[ids] += count * counts * idf * idf
        # Per-document TF normalization cancels in cosine, so integer counts
        # give the same score without rebuilding every weighted document vector.
        denominator = np.sqrt(np.maximum(squared_norm, 0) * query_norm_squared)
        scores = np.divide(dot, denominator, out=np.zeros_like(dot), where=denominator > 0).tolist()
        self.last_query, self.last_scores = query, scores
        return list(scores)
