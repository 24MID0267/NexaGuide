"""Lightweight hybrid retrieval: lexical BM25-style scoring + semantic cosine similarity.
No extra retrieval dependency is required, so the project remains easy to run locally.
"""
import math
import re
from collections import Counter

_WORD = re.compile(r"[a-z0-9]+")


def tokens(text):
    return _WORD.findall((text or "").lower())


def _idf(docs):
    n = len(docs)
    df = Counter()
    for d in docs:
        df.update(set(tokens(d)))
    return {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}


def lexical_scores(query, docs):
    q = tokens(query)
    if not q or not docs:
        return [0.0] * len(docs)
    idf = _idf(docs)
    avgdl = sum(max(1, len(tokens(d))) for d in docs) / len(docs)
    out = []
    k1, b = 1.5, 0.75
    for d in docs:
        dt = tokens(d)
        tf = Counter(dt)
        dl = max(1, len(dt))
        s = 0.0
        for term in q:
            f = tf.get(term, 0)
            if f:
                s += idf.get(term, 0.0) * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
        out.append(s)
    m = max(out) if out else 0.0
    return [x / m if m else 0.0 for x in out]


def hybrid_rank(query, docs, vectors=None, top_k=5, lexical_weight=0.35):
    """Return ranked tuples (index, fused_score, lexical, semantic)."""
    lex = lexical_scores(query, docs)
    sem = [0.0] * len(docs)
    if vectors is not None and len(docs):
        import numpy as np
        from .embedder import embed
        qv = embed([query])[0]
        sem = [float(x) for x in (vectors @ qv)]
        lo, hi = min(sem), max(sem)
        if hi > lo:
            sem = [(x - lo) / (hi - lo) for x in sem]
        else:
            sem = [0.0] * len(sem)
    fused = [lexical_weight * a + (1 - lexical_weight) * b for a, b in zip(lex, sem)]
    order = sorted(range(len(docs)), key=lambda i: fused[i], reverse=True)[:top_k]
    return [(i, round(fused[i], 4), round(lex[i], 4), round(sem[i], 4)) for i in order]
