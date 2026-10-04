"""Match an action to a catalog deeplink using description text (never the masked URI)."""
import json
from pathlib import Path
import numpy as np
from .embedder import embed

DATA = Path(__file__).resolve().parent.parent / "data"
MATCH_THRESHOLD = 0.35

_catalog = json.loads((DATA / "deeplinks.json").read_text(encoding="utf-8"))["deeplinks"]
DUMMY = next(d for d in _catalog if "dummy_positive" in d["deeplink"])
_entries = [d for d in _catalog if d is not DUMMY]
_vecs = None


def _load():
    global _vecs
    if _vecs is None:
        texts = [f"{d.get('description','')} {d.get('message','')} {d.get('qna_description','')}" for d in _entries]
        _vecs = embed(texts)


def best_match(text: str):
    """Returns (catalog_entry, similarity)."""
    _load()
    q = embed([text])[0]
    sims = _vecs @ q
    i = int(np.argmax(sims))
    return _entries[i], float(sims[i])


def to_actionable(entry: dict) -> dict:
    return {
        "deeplink": entry["deeplink"],
        "description": entry["description"],
        "message": entry.get("message") or "",
        "originalType": entry.get("originalType"),
    }


def to_validation(entry: dict):
    v = entry.get("validation")
    if not v:
        return None
    return {"deeplink": v["deeplink"], "key": v["key"],
            "resultType": "boolean", "condition": "equal", "value": "True"}
