"""Deterministic clarification layer for ambiguous troubleshooting complaints."""
import re

_RULES = [
    ("battery", "When does the battery drain happen most?", ["While charging", "During gaming or heavy use", "During normal use", "Even when idle"]),
    ("screen", "When does the display problem happen?", ["All the time", "When opening an app", "While charging", "After unlocking or folding"]),
    ("camera", "What best describes the camera problem?", ["Black preview", "App freezes", "Blurry or distorted image", "Camera will not open"]),
    ("wifi", "What happens to the Wi-Fi connection?", ["Cannot connect", "Connects then disconnects", "Very slow", "Only one network is affected"]),
    ("audio", "Where is the sound problem happening?", ["Speaker", "Earphones/Bluetooth", "Calls", "All audio"]),
]


def _topic(q):
    q = (q or "").lower()
    for key, question, options in _RULES:
        if re.search(rf"\b{re.escape(key)}\w*\b", q):
            return key, question, options
    return None


def suggest(query, conversation=None):
    """Return a clarification only when the complaint is broad enough to benefit."""
    if conversation and conversation.get("clarified"):
        return None
    hit = _topic(query)
    if not hit:
        return None
    key, question, options = hit
    q = (query or "").lower()
    specific = {
        "battery": ["drain", "dies", "battery life", "charge"],
        "screen": ["flicker", "blank", "black", "crack", "touch", "display"],
        "camera": ["freeze", "black", "blur", "camera"],
        "wifi": ["disconnect", "slow", "connect"],
        "audio": ["speaker", "sound", "volume", "bluetooth", "earphone"],
    }[key]
    # Ask when the topic is present but the complaint lacks a useful symptom qualifier.
    if any(x in q for x in specific) and len(q.split()) >= 7:
        return None
    return {"topic": key, "question": question, "options": options}
