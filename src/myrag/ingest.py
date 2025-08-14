from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class DocChunk:
    doc_id: str
    chunk_id: int
    text: str
    meta: dict


def _read_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def _simple_tokenize(s: str) -> list[str]:
    return s.split()


def chunk_text(text: str, tokens=900, overlap=120) -> list[str]:
    toks = _simple_tokenize(text)
    out = []
    i = 0
    while i < len(toks):
        out.append(" ".join(toks[i : i + tokens]))
        i += max(1, tokens - overlap)
    return [c for c in out if c.strip()]


def checksum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 15)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:16]


def ingest_docs(input_dir: Path, out_dir: Path, tokens=900, overlap=120) -> list[DocChunk]:
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[DocChunk] = []
    for p in sorted(Path(input_dir).rglob("*")):
        if not p.is_file() or p.suffix.lower() not in {".pdf", ".txt"}:
            continue
        cs = checksum(p)
        text = _read_text(p)
        for i, part in enumerate(chunk_text(text, tokens, overlap)):
            chunks.append(
                DocChunk(doc_id=cs, chunk_id=i, text=part, meta={"source": str(p), "checksum": cs})
            )
    # persist raw chunks for transparency
    (out_dir / "chunks.jsonl").write_text(
        "\n".join(json.dumps(c.__dict__) for c in chunks), encoding="utf-8"
    )
    return chunks
