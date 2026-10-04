# AI / GenAI Disclosure

NexaGuide uses a Groq-hosted large language model for query enrichment, paraphrase generation and structured troubleshooting-plan generation. The model is constrained to supplied reference text and the response is validated by application rules and a Pydantic schema.

Semantic retrieval uses `sentence-transformers/all-MiniLM-L6-v2`. A lexical BM25-style scorer is combined with semantic similarity for hybrid ranking. A semantic cache reduces repeated generation.
