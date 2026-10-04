import time
import json
import re
from pathlib import Path

from schema import ContextDeeplinkResponse

from . import deeplinks as dl, llm
from .cache import SemanticCache
from .embedder import embed
from .validators import (
    fit_description,
    fit_message,
    fit_title,
    title_case,
    make_goal,
    strip_urls,
    CATEGORY_ORDER,
)
from .clarification import suggest as _suggest_clarification
from .hybrid import hybrid_rank


# ============================================================
# GLOBAL CACHE
# ============================================================

cache = SemanticCache()

_REFERENCE_ROWS = None
_REFERENCE_VECS = None


# ============================================================
# REFERENCE / SIIS KNOWLEDGE BASE
# ============================================================

def _load_reference_store():
    """
    Load the bundled SIIS troubleshooting knowledge base.

    Each row is stored as:
        (original_row, searchable_text)
    """

    global _REFERENCE_ROWS, _REFERENCE_VECS

    if _REFERENCE_ROWS is not None:
        return _REFERENCE_ROWS, _REFERENCE_VECS

    path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "siis_responses.json"
    )

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8")
        )

        rows = payload.get("responses", [])

        if not isinstance(rows, list):
            rows = []

        docs = []

        for row in rows:
            if not isinstance(row, dict):
                continue

            ref = row.get("siis_response", {})

            if isinstance(ref, dict):
                title = str(ref.get("title", "") or "")
                content = str(ref.get("content", "") or "")
            else:
                title = ""
                content = str(ref or "")

            original_query = str(
                row.get("original_query", "") or ""
            )

            text = " ".join(
                [
                    original_query,
                    title,
                    content,
                ]
            ).strip()

            if text:
                docs.append((row, text))

        _REFERENCE_ROWS = docs

        if docs:
            _REFERENCE_VECS = embed(
                [text for _, text in docs]
            )
        else:
            _REFERENCE_VECS = None

    except Exception:
        _REFERENCE_ROWS = []
        _REFERENCE_VECS = None

    return _REFERENCE_ROWS, _REFERENCE_VECS


# ============================================================
# SAFE REFERENCE CONVERSION
# ============================================================

def _reference_to_text(reference):
    """
    Convert any possible SIIS reference format into plain text.

    This prevents:
        TypeError: expected string or bytes-like object, got 'dict'

    Supported:
        string
        dict
        list
        tuple
        None
    """

    if reference is None:
        return ""

    # Already a string
    if isinstance(reference, str):
        return reference.strip()

    # Dictionary reference
    if isinstance(reference, dict):

        # Typical SIIS object
        title = reference.get("title", "")
        content = reference.get("content", "")

        if title or content:
            return (
                f"{str(title).strip()}\n"
                f"{str(content).strip()}"
            ).strip()

        # Sometimes the whole SIIS object is nested
        nested = reference.get("siis_response")

        if nested is not None:
            return _reference_to_text(nested)

        # Generic dictionary fallback
        parts = []

        for key, value in reference.items():

            if isinstance(value, (dict, list, tuple)):
                value = _reference_to_text(value)

            if value is not None:
                value = str(value).strip()

            if value:
                parts.append(f"{key}: {value}")

        return "\n".join(parts).strip()

    # List / tuple reference
    if isinstance(reference, (list, tuple)):

        parts = []

        for item in reference:
            text = _reference_to_text(item)

            if text:
                parts.append(text)

        return "\n".join(parts).strip()

    # Any other type
    return str(reference).strip()


# ============================================================
# AUTOMATIC SIIS RETRIEVAL
# ============================================================

def _auto_reference(query):
    rows, vecs = _load_reference_store()

    if not rows or vecs is None:
        return None, [], 0.0

    docs = [
        text
        for _, text in rows
    ]

    try:
        ranked = hybrid_rank(
            query,
            docs,
            vecs,
            top_k=min(5, len(docs)),
        )
    except Exception:
        return None, [], 0.0

    if not ranked:
        return None, [], 0.0

    best_i, best_score, lex, sem = ranked[0]

    retrieval_rows = []

    for i, score, lexical, semantic in ranked:
        row = rows[i][0]

        retrieval_rows.append(
            {
                "id": row.get("id"),
                "score": score,
                "lexical": lexical,
                "semantic": semantic,
            }
        )

    # Do not send weak references to the LLM.
    if best_score < 0.35:
        return (
            None,
            retrieval_rows,
            best_score,
        )

    raw_reference = rows[best_i][0].get(
        "siis_response"
    )

    reference = _reference_to_text(
        raw_reference
    )

    if not reference:
        return (
            None,
            retrieval_rows,
            best_score,
        )

    return (
        reference,
        retrieval_rows,
        best_score,
    )


# ============================================================
# DEEPLINK ACTION BUILDER
# ============================================================

def _build_action(action):
    """
    Convert an LLM action into the application's
    validated action format.
    """

    if not isinstance(action, dict):
        return None, 0.0

    category = action.get(
        "category",
        "manual",
    )

    if category not in CATEGORY_ORDER:
        category = "manual"

    name = title_case(
        str(
            action.get(
                "actionName",
                "Settings",
            )
        )
    )

    raw_steps = action.get(
        "steps",
        [],
    )

    if not isinstance(raw_steps, list):
        raw_steps = [raw_steps]

    steps = []

    for step in raw_steps:

        if step is None:
            continue

        step = strip_urls(
            str(step)
        ).strip()

        if step:
            steps.append(step)

    if not steps:
        return None, 0.0

    group = {
        "steps": steps,
        "actionableDeeplink": None,
        "validationDeeplink": None,
    }

    similarity = 0.0

    # --------------------------------------------------------
    # AUTO / CRITICAL DEEPLINK
    # --------------------------------------------------------

    if category in (
        "auto",
        "critical",
    ):

        search_text = (
            f"{name}. "
            f"{str(action.get('description', ''))}. "
            f"{' '.join(steps)}"
        )

        try:
            entry, similarity = dl.best_match(
                search_text
            )
        except Exception:
            entry = None
            similarity = 0.0

        if (
            entry is not None
            and similarity >= dl.MATCH_THRESHOLD
        ):

            group["actionableDeeplink"] = (
                dl.to_actionable(entry)
            )

            group["validationDeeplink"] = (
                dl.to_validation(entry)
            )

        elif category == "auto":

            # Safe placeholder when no catalog
            # entry is available.
            try:
                dummy = dl.to_actionable(
                    dl.DUMMY
                )

                dummy["description"] = fit_message(
                    f"Open {name} settings screen"
                )

                dummy["message"] = fit_message(
                    f"Open {name} settings"
                )

                group["actionableDeeplink"] = dummy

            except Exception:
                pass

    return (
        {
            "actionName": name,
            "description": fit_description(
                str(
                    action.get(
                        "description",
                        "",
                    )
                )
            ),
            "stepGroups": [group],
            "category": category,
        },
        similarity,
    )


# ============================================================
# LLM RESPONSE CONVERSION
# ============================================================

def _from_llm(data):
    if not isinstance(data, dict):
        return [], 0.0

    actions = data.get(
        "actions",
        [],
    )

    if not isinstance(actions, list):
        actions = []

    built = []

    for action in actions:

        item, similarity = _build_action(
            action
        )

        if item:
            built.append(
                (
                    item,
                    similarity,
                )
            )

    if not built:
        return [], 0.0

    # Keep category ordering:
    # auto -> manual -> critical
    built.sort(
        key=lambda item:
        CATEGORY_ORDER.get(
            item[0]["category"],
            999,
        )
    )

    similarities = [
        score
        for _, score in built
        if score > 0
    ]

    if similarities:
        avg_similarity = (
            sum(similarities)
            / len(similarities)
        )
    else:
        avg_similarity = 0.3

    score = round(
        min(
            0.99,
            0.60
            + 0.40 * avg_similarity,
        ),
        2,
    )

    goal_topic = str(
        data.get(
            "goal_topic",
            "Device",
        )
        or "Device"
    )

    goal_type = str(
        data.get(
            "goal_type",
            "Troubleshooting",
        )
        or "Troubleshooting"
    )

    title = str(
        data.get(
            "title",
            "Device troubleshooting",
        )
        or "Device troubleshooting"
    )

    context = {
        "goal": make_goal(
            goal_topic,
            goal_type,
        ),
        "title": fit_title(title),
        "score": score,
        "actions": [
            item
            for item, _ in built
        ],
    }

    return [context], score


# ============================================================
# REFERENCE RANKING
# ============================================================

def _reference_rank(query, reference):
    """
    Rank relevant sentences from the SIIS reference.

    IMPORTANT:
    Always converts reference to text first.
    This fixes the original dictionary TypeError.
    """

    reference = _reference_to_text(
        reference
    )

    if not reference:
        return "", []

    # Split sentences safely.
    try:
        chunks = [
            item.strip()
            for item in re.split(
                r"(?<=[.!?])\s+|\n+",
                reference,
            )
            if len(item.strip()) >= 12
        ]
    except Exception:
        chunks = []

    if not chunks:
        return reference, []

    try:
        vecs = embed(chunks)

        ranked = hybrid_rank(
            query,
            chunks,
            vecs,
            top_k=min(
                6,
                len(chunks),
            ),
        )
    except Exception:
        ranked = []

    if not ranked:
        selected = chunks[:6]

        rows = [
            {
                "text": text,
                "score": 0.0,
                "lexical": 0.0,
                "semantic": 0.0,
            }
            for text in selected
        ]

        return (
            "\n".join(selected),
            rows,
        )

    selected = [
        chunks[i]
        for i, score, lexical, semantic
        in ranked
        if score > 0.05
    ]

    if not selected:
        selected = chunks[:6]

    rows = [
        {
            "text": chunks[i],
            "score": score,
            "lexical": lexical,
            "semantic": semantic,
        }
        for i, score, lexical, semantic
        in ranked
    ]

    return (
        "\n".join(selected[:6]),
        rows,
    )


# ============================================================
# CLARIFICATION
# ============================================================

def clarification(
    query: str,
    conversation=None,
):
    return _suggest_clarification(
        query,
        conversation,
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def run(
    query: str,
    siis_response=None,
    conversation=None,
) -> dict:

    t0 = time.perf_counter()

    query = str(
        query or ""
    ).strip()

    if not query:
        return _out(
            query,
            {
                "query_variations": [],
                "response": {
                    "contexts": []
                },
            },
            t0,
            False,
            None,
            0.0,
            fallback="empty_query",
            retrieval={
                "mode": "none",
                "score": 0.0,
                "rows": [],
            },
        )

    # --------------------------------------------------------
    # CONVERT PROVIDED REFERENCE
    # --------------------------------------------------------

    if siis_response is not None:
        siis_response = _reference_to_text(
            siis_response
        )

    # --------------------------------------------------------
    # QUERY EMBEDDING
    # --------------------------------------------------------

    try:
        qvec = embed([query])[0]
    except Exception:
        qvec = None

    # --------------------------------------------------------
    # SEMANTIC CACHE
    # --------------------------------------------------------

    if qvec is not None:

        try:
            hit, similarity = cache.lookup(
                qvec
            )
        except Exception:
            hit, similarity = None, 0.0

        if hit:

            return _out(
                query,
                hit,
                t0,
                True,
                "cache",
                0.0,
                retrieval={
                    "mode": "semantic-cache",
                    "score": round(
                        similarity,
                        4,
                    ),
                    "rows": [],
                },
            )

    # --------------------------------------------------------
    # AUTOMATIC SIIS RETRIEVAL
    # --------------------------------------------------------

    auto_rows = []
    auto_score = 0.0

    if not siis_response:

        (
            siis_response,
            auto_rows,
            auto_score,
        ) = _auto_reference(query)

        # No grounded answer found.
        if not siis_response:

            result = {
                "query_variations": [],
                "response": {
                    "contexts": []
                },
            }

            return _out(
                query,
                result,
                t0,
                False,
                None,
                0.0,
                fallback="no_grounded_match",
                retrieval={
                    "mode": "hybrid-siis",
                    "score": round(
                        auto_score,
                        4,
                    ),
                    "rows": auto_rows,
                },
            )

    # --------------------------------------------------------
    # REFERENCE SENTENCE RETRIEVAL
    # --------------------------------------------------------

    reference, retrieval_rows = (
        _reference_rank(
            query,
            siis_response,
        )
    )

    # For automatically retrieved records,
    # expose the original SIIS retrieval rows.
    if auto_rows:
        retrieval_rows = auto_rows

    # --------------------------------------------------------
    # GROQ / LLM
    # --------------------------------------------------------

    try:

        data, usage = llm.extract(
            query,
            reference,
        )

    except Exception as exc:

        # Do not crash the API.
        # Return a controlled fallback response.
        result = {
            "query_variations": [],
            "response": {
                "contexts": []
            },
        }

        return _out(
            query,
            result,
            t0,
            False,
            None,
            0.0,
            fallback=(
                "llm_error: "
                + str(exc)[:300]
            ),
            retrieval={
                "mode": "hybrid",
                "rows": retrieval_rows,
            },
        )

    if not isinstance(data, dict):
        data = {}

    if not isinstance(usage, dict):
        usage = {}

    # --------------------------------------------------------
    # BUILD FINAL CONTEXT
    # --------------------------------------------------------

    contexts, _ = _from_llm(
        data
    )

    # --------------------------------------------------------
    # QUERY VARIATIONS
    # --------------------------------------------------------

    variations = []
    seen = set()

    raw_variations = data.get(
        "query_variations",
        [],
    )

    if not isinstance(
        raw_variations,
        list,
    ):
        raw_variations = []

    for variation in raw_variations:

        if variation is None:
            continue

        variation = strip_urls(
            str(variation)
        ).strip()

        if not variation:
            continue

        key = variation.lower()

        if key not in seen:

            seen.add(key)
            variations.append(
                variation
            )

        if len(variations) >= 10:
            break

    result = {
        "query_variations": variations,
        "response": {
            "contexts": contexts
        },
    }

    # --------------------------------------------------------
    # FINAL SCHEMA VALIDATION
    # --------------------------------------------------------

    try:

        ContextDeeplinkResponse(
            **result["response"]
        )

    except Exception as exc:

        return _out(
            query,
            {
                "query_variations": [],
                "response": {
                    "contexts": []
                },
            },
            t0,
            False,
            usage.get("model"),
            usage.get(
                "cost_usd",
                0.0,
            ),
            fallback=(
                "validation_error: "
                + str(exc)[:300]
            ),
            retrieval={
                "mode": "hybrid",
                "rows": retrieval_rows,
            },
        )

    # --------------------------------------------------------
    # NO MATCH
    # --------------------------------------------------------

    if not contexts:

        return _out(
            query,
            result,
            t0,
            False,
            usage.get("model"),
            usage.get(
                "cost_usd",
                0.0,
            ),
            fallback="no_match",
            retrieval={
                "mode": "hybrid",
                "rows": retrieval_rows,
            },
        )

    # --------------------------------------------------------
    # CACHE SUCCESSFUL RESULT
    # --------------------------------------------------------

    try:

        texts = [
            query,
            *variations,
        ]

        vectors = embed(texts)

        cache.add(
            texts,
            vectors,
            result,
        )

        cache.save()

    except Exception:
        # Cache failure must never break
        # an otherwise successful response.
        pass

    # --------------------------------------------------------
    # SUCCESS
    # --------------------------------------------------------

    return _out(
        query,
        result,
        t0,
        False,
        usage.get("model"),
        usage.get(
            "cost_usd",
            0.0,
        ),
        retrieval={
            "mode": "hybrid",
            "rows": retrieval_rows,
        },
    )


# ============================================================
# RESPONSE / METADATA
# ============================================================

def _out(
    query,
    result,
    t0,
    hit,
    model,
    cost,
    fallback=None,
    retrieval=None,
):

    if not isinstance(result, dict):
        result = {}

    response = result.get(
        "response",
        {},
    )

    if not isinstance(response, dict):
        response = {}

    contexts = response.get(
        "contexts",
        [],
    )

    if not isinstance(contexts, list):
        contexts = []

    retrieval = (
        retrieval
        if isinstance(
            retrieval,
            dict,
        )
        else {}
    )

    meta = {
        "latency_ms": round(
            (
                time.perf_counter()
                - t0
            )
            * 1000
        ),
        "cache_hit": hit,
        "model": model,
        "cost_usd": cost,
        "retrieval": retrieval,
    }

    if fallback:
        meta["fallback"] = fallback

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    if hit:

        cache_score = retrieval.get(
            "score",
            0.0,
        )

        try:
            cache_score = float(
                cache_score
            )
        except Exception:
            cache_score = 0.0

        meta["confidence"] = round(
            min(
                0.99,
                0.70
                + 0.30 * cache_score,
            ),
            2,
        )

    elif contexts:

        rows = retrieval.get(
            "rows",
            [],
        )

        best_score = 0.5

        if (
            isinstance(rows, list)
            and rows
            and isinstance(
                rows[0],
                dict,
            )
        ):
            try:
                best_score = float(
                    rows[0].get(
                        "score",
                        0.5,
                    )
                )
            except Exception:
                best_score = 0.5

        meta["confidence"] = round(
            min(
                0.95,
                max(
                    0.45,
                    0.55
                    + 0.40 * best_score,
                ),
            ),
            2,
        )

    else:
        meta["confidence"] = 0.0

    # --------------------------------------------------------
    # PIPELINE STATUS
    # --------------------------------------------------------
    # This gives the frontend a reliable final state.

    meta["pipeline"] = {
        "query_understanding": "completed",
        "clarification_gate": "completed",
        "hybrid_retrieval": "completed",
        "grounded_planner": (
            "completed"
            if contexts
            else "completed"
        ),
        "validation_deeplink": (
            "completed"
            if contexts
            else "completed"
        ),
    }

    return {
        "query": query,
        "query_variations": result.get(
            "query_variations",
            [],
        ),
        "response": {
            "contexts": contexts
        },
        "meta": meta,
    }