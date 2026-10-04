"""Run the full pipeline on the 20 sample complaints and save results to data/cache.json.
Usage:  python -m scripts.prewarm"""
import json, re
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from app import pipeline

rows = json.loads((Path("data") / "siis_responses.json").read_text(encoding="utf-8"))["responses"]
for i, r in enumerate(rows, 1):
    q = re.sub(r'^\s*1\.\s*', '', r["original_query"]).strip().strip('"')
    out = pipeline.run(q, r["siis_response"])
    print(i, out["meta"], len(out["response"]["contexts"]), "context(s)")
print("Saved data/cache.json  (commit this file!)")
