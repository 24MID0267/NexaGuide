# NexaGuide — Smart Guided Troubleshooting Engine (Samsung PRISM Theme 2)

NexaGuide turns vague device complaints into a guided, grounded troubleshooting path.

## Highlights
- Interactive clarification for broad complaints
- Hybrid lexical + semantic retrieval
- Grounded LLM planning using retrieved reference text
- Confidence-aware result presentation
- Technical pipeline inspector
- Existing semantic-cache fast path retained
- Catalog-based settings deeplinks retained
- Original `/v1/troubleshoot` API retained

## Run locally (Windows)

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload --port 8000
```

Open `http://localhost:8000/`.

## API
- `GET /health`
- `POST /v1/clarify`
- `POST /v1/troubleshoot`
- `GET /docs`

## Architecture
`Complaint → Clarification → Semantic cache → Hybrid retrieval → Grounded LLM → Validation → Deeplink mapping → Schema → Cache → UI`

See `docs/ARCHITECTURE.md` and `docs/DEMO_SCRIPT.md`.

For new complaints, suitable reference text and a valid `GROQ_API_KEY` are required. The shipped semantic cache can answer previously warmed/similar examples without another generation call.
