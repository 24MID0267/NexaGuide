"""Rules from the problem statement, enforced in CODE (not just in the prompt)."""
import re

MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
URL = re.compile(r"(https?://|www\.)\S+", re.I)
BARE_DOMAIN = re.compile(r"\b[\w-]+\.(com|org|net|io|co|in|gov|edu)(/\S*)?\b", re.I)
FILLER = ["on", "your", "device", "settings"]


def strip_urls(text: str) -> str:
    text = MD_LINK.sub(r"\1", text or "")
    text = URL.sub("", text)
    text = BARE_DOMAIN.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def has_url(text: str) -> bool:
    return bool(URL.search(text or "") or BARE_DOMAIN.search(text or "") or MD_LINK.search(text or ""))


def _pad(words, minimum):
    i = 0
    while len(words) < minimum:
        words.append(FILLER[i % len(FILLER)])
        i += 1
    return words


def fit_description(text: str) -> str:
    """Exactly 5-7 words, starting with 'It will'."""
    words = strip_urls(text).rstrip(".").split()
    if not (len(words) >= 2 and words[0].lower() == "it" and words[1].lower() == "will"):
        rest = words[:]
        if rest:
            rest[0] = rest[0].lower()
            if rest[0].endswith("s") and not rest[0].endswith("ss") and len(rest[0]) > 3:
                rest[0] = rest[0][:-1]  # "allows" -> "allow" 
        words = ["It", "will"] + rest
    words = _pad(words[:7], 5)
    return " ".join(words)


def fit_message(text: str) -> str:
    """5-7 word message (used for the dummy_positive placeholder)."""
    words = strip_urls(text).split()[:7]
    return " ".join(_pad(words, 5))


def fit_title(text: str) -> str:
    """2-3 words, sentence case."""
    words = strip_urls(text).split()[:3]
    words = _pad(words, 2) if words else ["Device", "settings"]
    return " ".join(words).capitalize()


def title_case(text: str) -> str:
    return " ".join(w[:1].upper() + w[1:] for w in strip_urls(text).split())


def make_goal(topic: str, kind: str = "Troubleshooting") -> str:
    kind = "Configuration" if str(kind).lower().startswith("config") else "Troubleshooting"
    topic = re.sub(r"\b(troubleshooting|configuration)\b", "", strip_urls(topic), flags=re.I).strip()
    topic = title_case(topic) or "Device"
    return f"Follow these steps to perform this {topic} {kind}"


CATEGORY_ORDER = {"auto": 0, "manual": 1, "critical": 2}  # least disruptive first, critical last
