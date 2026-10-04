# NexaGuide Architecture

`Complaint → Clarification gate → Semantic cache → Hybrid retrieval → Grounded LLM → Validators → Deeplink catalog → Pydantic schema → Cache → UI`

NexaGuide adds a deterministic clarification layer, lightweight BM25-style lexical ranking, semantic similarity ranking, a fused retrieval score, and a transparent pipeline inspector while retaining the original semantic-cache and deeplink architecture.
