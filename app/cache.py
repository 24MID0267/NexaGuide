"""Semantic cache: a query hits if it is close in meaning to any stored query/paraphrase."""
import json, os
from pathlib import Path
import numpy as np

CACHE_FILE = Path(__file__).resolve().parent.parent / "data" / "cache.json"


class SemanticCache:
    def __init__(self):
        self.entries = []   # {"queries": [...], "result": {...}}
        self.matrix = np.zeros((0, 384), dtype="float32")
        self.owner = []     # row -> entry index
        self.threshold = float(os.getenv("CACHE_THRESHOLD", 0.78))
        self.load()

    def load(self):
        if CACHE_FILE.exists():
            raw = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
            self.entries = raw["entries"]
            self.matrix = np.asarray(raw["vectors"], dtype="float32").reshape(-1, 384)
            self.owner = raw["owner"]

    def save(self):
        CACHE_FILE.write_text(json.dumps({
            "entries": self.entries, "vectors": self.matrix.tolist(), "owner": self.owner}),
            encoding="utf-8")

    def lookup(self, qvec):
        if len(self.owner) == 0:
            return None, 0.0
        sims = self.matrix @ qvec
        i = int(np.argmax(sims))
        if sims[i] >= self.threshold:
            return self.entries[self.owner[i]]["result"], float(sims[i])
        return None, float(sims[i])

    def add(self, queries, vectors, result):
        idx = len(self.entries)
        self.entries.append({"queries": queries, "result": result})
        self.matrix = np.vstack([self.matrix, vectors]) if len(self.owner) else vectors
        self.owner += [idx] * len(queries)
