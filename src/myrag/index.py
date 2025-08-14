import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .ingest import ingest_docs
from .settings import settings


def build():
    chunks = ingest_docs(
        settings.raw_dir, settings.processed_dir, settings.chunk_tokens, settings.chunk_overlap
    )
    texts = [c.text for c in chunks]
    metas = [c.meta for c in chunks]
    if not texts:
        print("No documents found in", settings.raw_dir)
        return
    model = SentenceTransformer(settings.embedding_model)
    embs = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs.astype(np.float32))
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(settings.index_dir / "faiss.index"))
    (settings.index_dir / "metas.json").write_text(json.dumps(metas, ensure_ascii=False))
    print(f"Built index with {len(texts)} chunks → {settings.index_dir}")


if __name__ == "__main__":
    build()
