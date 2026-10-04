"""Measures latency, schema validity, URL leaks, cache hit rate -> writes metrics.md
Usage:  python -m scripts.evaluate   (run prewarm first)"""
import json, statistics
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from app import pipeline
from app.validators import has_url
from schema import ContextDeeplinkResponse

queries = [l.strip() for l in Path("data/input.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
paraphrases = ["phone display goes black randomly", "my screen keeps flickering", "touch is laggy on my phone",
               "tablet screen blank when opening apps", "cracked display cannot use phone"]
lat, valid, leaks, hits = [], 0, 0, 0
for q in queries:
    out = pipeline.run(q)
    lat.append(out["meta"]["latency_ms"])
    hits += out["meta"]["cache_hit"]
    try:
        ContextDeeplinkResponse(**out["response"]); valid += 1
    except Exception:
        pass
    leaks += has_url(json.dumps(out))
p_hits = sum(pipeline.run(p)["meta"]["cache_hit"] for p in paraphrases)
lat.sort()
p50, p95 = statistics.median(lat), lat[max(0, int(len(lat) * 0.95) - 1)]
md = f"""# System Performance Metrics
**Model:** Groq (see GROQ_MODEL) | **Embeddings:** all-MiniLM-L6-v2 | **Environment:** CPU

| Metric | Target | Measured |
|---|---|---|
| Schema-valid responses | >=99% | {100*valid/len(queries):.0f}% |
| Absolute URL leaks | 0 | {leaks} |
| Cache hit rate (sample queries) | - | {100*hits/len(queries):.0f}% |
| Cache hit rate (hand-written paraphrases) | >=80% | {100*p_hits/len(paraphrases):.0f}% |
| Cache-hit latency P50 / P95 (ms) | <=300 | {p50:.0f} / {p95:.0f} |

Cold-path latency and cost per query: see `meta` in the prewarm logs.

## Ablations (fill in after you try them)
| Variant | Step accuracy | Latency | Notes |
|---|---|---|---|
| Baseline: dense deeplink match (this repo) | | | |
| Variant A: add BM25 to the match | | | |
| Variant B: rules only (no LLM) | | | |

## Known limitations
- Step quality depends on the reference text; empty/irrelevant text returns `no_match`.
- Deeplink matching is embedding-only; unusual screens fall back to `dummy_positive`.
"""
Path("metrics.md").write_text(md, encoding="utf-8")
print(md)
