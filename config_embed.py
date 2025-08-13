# config_embed.py
import os
from txtai import Embeddings

def build_embeddings():
    backend = os.getenv("TXT_BACKEND", "sqlite-vec")
    model = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    if backend == "sqlite-vec":
        return Embeddings(path=model, content=True, backend="sqlite", sqlite={"quantize": 8})

    if backend == "sqlite-faiss":
        return Embeddings(path=model, content=True, backend="faiss")

    if backend == "pgvector":
        pgurl = os.environ["PGURL"]
        os.environ["CLIENT_URL"] = pgurl
        return Embeddings(
            path=os.getenv("EMB_MODEL", "sentence-transformers/nli-mpnet-base-v2"),
            content="client", client={"schema": "public"},
            backend="pgvector", pgvector={"url": pgurl, "precision": "half"}
        )

    raise ValueError(f"Unknown TXT_BACKEND: {backend}")
