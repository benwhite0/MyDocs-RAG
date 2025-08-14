from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from .index import build as rebuild_index
from .rag import Retriever, generate_answer, make_prompt
from .settings import settings

app = FastAPI(title="MyDocs-RAG")

retriever = None


class ChatRequest(BaseModel):
    question: str
    k: int | None = None


@app.on_event("startup")
def _startup():
    global retriever
    retriever = Retriever()


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    k = req.k or settings.k
    hits = retriever.retrieve(req.question, k)
    prompt = make_prompt(req.question, hits)
    answer = generate_answer(req.question, hits)
    sources = [{"source": h["meta"].get("source", ""), "score": h["score"]} for h in hits]
    return {"answer": answer, "prompt": prompt, "sources": sources}


# Upload endpoint to accept PDF/TXT, rebuild index, and reload retriever
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".txt"}:
        raise HTTPException(status_code=400, detail="Only .pdf and .txt files are supported")
    dest = settings.raw_dir / Path(file.filename).name
    try:
        settings.raw_dir.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as out:
            while True:
                chunk = await file.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        # rebuild index and hot-reload retriever
        rebuild_index()
        global retriever
        retriever = Retriever()
        return {"status": "ok", "path": str(dest)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")
