from __future__ import annotations

import math
from collections import Counter


def normalize_spaces(text: str) -> str:
    return " ".join((text or "").split())


def preview_text(text: str, max_chars: int) -> str:
    text = normalize_spaces(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _char_ngrams(text: str, min_n: int = 2, max_n: int = 5) -> list[str]:
    text = normalize_spaces(text.lower())
    if not text:
        return []
    compact = text.replace(" ", "")
    terms: list[str] = []
    for source in (text, compact):
        if len(source) < min_n:
            terms.append(source)
            continue
        for n in range(min_n, max_n + 1):
            if len(source) < n:
                continue
            terms.extend(source[i : i + n] for i in range(len(source) - n + 1))
    return terms


def tfidf_scores(query: str, docs: list[str]) -> list[float]:
    counters = [Counter(_char_ngrams(text)) for text in [query, *docs]]
    doc_freq: Counter[str] = Counter()
    for counter in counters:
        doc_freq.update(counter.keys())

    doc_count = len(counters)
    idf = {
        term: math.log((1 + doc_count) / (1 + freq)) + 1.0
        for term, freq in doc_freq.items()
    }

    def weighted(counter: Counter[str]) -> tuple[dict[str, float], float]:
        total = sum(counter.values()) or 1
        vector = {
            term: (count / total) * idf[term]
            for term, count in counter.items()
            if term in idf
        }
        norm = math.sqrt(sum(value * value for value in vector.values()))
        return vector, norm

    query_vec, query_norm = weighted(counters[0])
    scores: list[float] = []
    for counter in counters[1:]:
        doc_vec, doc_norm = weighted(counter)
        if not query_norm or not doc_norm:
            scores.append(0.0)
            continue
        dot = sum(value * doc_vec.get(term, 0.0) for term, value in query_vec.items())
        scores.append(dot / (query_norm * doc_norm))
    return scores


