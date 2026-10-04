from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from typing import Optional
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from . import pipeline

STATIC = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="Smart Guided Troubleshooting Engine")


class Req(BaseModel):
    query: str
    siis_response: Optional[str] = None
    conversation: Optional[dict] = None

class ClarifyReq(BaseModel):
    query: str
    conversation: Optional[dict] = None


@app.on_event("startup")
def warm_up():
    pipeline.embed(["warm up"])  # load the model once, so the first request is fast


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC / "index.html")  # simple demo page


@app.post("/v1/troubleshoot")
def troubleshoot(req: Req):
    return pipeline.run(req.query, req.siis_response, req.conversation)


@app.post("/v1/clarify")
def clarify(req: ClarifyReq):
    return {"clarification": pipeline.clarification(req.query, req.conversation)}

@app.get("/health")
def health():
    return {"status": "ok"}
