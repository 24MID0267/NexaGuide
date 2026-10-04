"""One Groq call does: query enrichment + paraphrases + structured steps."""
import json, os

SYSTEM = """You are a smartphone troubleshooting planner. Output ONLY one JSON object, no markdown.
Use ONLY the REFERENCE TEXT for steps. Never invent steps. Never output any URL, domain or web link.
If the reference text contains no usable solution, return "actions": [].

JSON shape:
{
 "enriched_query": "short canonical technical version of the complaint",
 "query_variations": [8 to 10 DISTINCT paraphrases: formal, casual, keyword-only, frustrated, with typos],
 "goal_topic": "2-3 word topic e.g. Screen Flicker",
 "goal_type": "Troubleshooting" or "Configuration",
 "title": "2-3 words, sentence case, e.g. Battery fast drain",
 "actions": [
  {"actionName": "Title Case name of exactly ONE screen or feature",
   "description": "It will <benefit>, 5-7 words total, starts with 'It will'",
   "category": "auto | manual | critical",
   "steps": ["Short imperative UI step, one physical interaction each", "..."]}
 ]
}
Category rules:
- auto: a Settings screen the user can reach (toggle/configure). Steps like "Navigate to and open Settings.", "Tap on Display.".
- manual: physical things (clean port, replace hardware, visit a service center).
- critical: disruptive or irreversible (factory reset, restart, firmware update, safe mode). Put them LAST.
One action = one screen. Order actions from least disruptive to most disruptive."""


def _client():
    from groq import Groq
    return Groq(api_key=os.environ["GROQ_API_KEY"])


def extract(query: str, reference: str):
    """Returns (parsed_json, usage_dict)."""
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    user = f"COMPLAINT:\n{query}\n\nREFERENCE TEXT:\n{(reference or '')[:6000]}"
    last_err = None
    for _ in range(2):  # one retry if the JSON is broken
        r = _client().chat.completions.create(
            model=model, temperature=0, max_tokens=4000, extra_body={"reasoning_effort": "low"},
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        )
        try:
            data = json.loads(r.choices[0].message.content)
            u = r.usage
            cost = (u.prompt_tokens * float(os.getenv("GROQ_IN_RATE", 0)) +
                    u.completion_tokens * float(os.getenv("GROQ_OUT_RATE", 0))) / 1_000_000
            return data, {"model": model, "cost_usd": round(cost, 6)}
        except Exception as e:  # noqa
            last_err = e
    raise RuntimeError(f"LLM returned invalid JSON: {last_err}")
