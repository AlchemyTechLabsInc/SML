# graph_rag.py
from pathlib import Path
import json
import networkx as nx
from networkx.readwrite import json_graph
from typing import Iterable, List, Set

from txtai import Embeddings
from txtai.pipeline import RAG

GRAPH_PATH = Path("data/graph.json")
CHUNKMAP_PATH = Path("data/hashmaps/chunk_map.json")
INDEX_DIR = "index"

def load_graph():
    return json_graph.node_link_graph(json.loads(GRAPH_PATH.read_text()))

def load_chunkmap():
    return json.loads(CHUNKMAP_PATH.read_text())

def load_embeddings():
    # For local backends we saved to "index"
    return Embeddings().load(INDEX_DIR)

def build_rag(emb):
    return RAG(
        emb,
        path="google/flan-t5-base",   # swap to your LLM/API
        context=8,
        output="reference",
        template="""
Answer ONLY using the context. If adding numbers, show the math and units.

Question:
{question}

Context:
{context}
"""
    )

def cids_for_entity(G, entity_name: str) -> Set[str]:
    """Return all chunk IDs connected to an entity with this exact name (case-insensitive)."""
    targets = {n for n, d in G.nodes(data=True) if d.get("label") == "Entity" and str(d.get("name","")).lower() == entity_name.lower()}
    cids: Set[str] = set()
    for ent in targets:
        for u, v, data in G.in_edges(ent, data=True):
            if G.nodes[u].get("label") == "Chunk":
                cids.add(u)
        for u, v, data in G.out_edges(ent, data=True):
            if G.nodes[v].get("label") == "Chunk":
                cids.add(v)
    return cids

def top_context(emb, chunk_map, question: str, scope_cids: Set[str] | None, k: int = 12) -> List[str]:
    hits = emb.search(question, limit=60)  # get more, then filter
    texts = []
    for h in hits:
        cid = h["id"]
        if scope_cids and cid not in scope_cids:
            continue
        rec = chunk_map.get(cid)
        if not rec:
            continue
        texts.append(rec["text"])
        if len(texts) >= k:
            break
    return texts

def resolve_refs(chunk_map, refs):
    out = []
    for rid in refs:
        rec = chunk_map.get(rid, {})
        out.append({"id": rid, "docname": rec.get("docname"), "page": rec.get("page"), "type": rec.get("type")})
    return out

def ask(question: str, entities: List[str] | None = None):
    G = load_graph()
    chunk_map = load_chunkmap()
    emb = load_embeddings()
    rag = build_rag(emb)

    scope = set()
    if entities:
        for e in entities:
            scope |= cids_for_entity(G, e)

    ctx = top_context(emb, chunk_map, question, scope_cids=scope or None, k=12)
    name, answer, refs = rag(question, context=ctx)
    return answer, resolve_refs(chunk_map, refs)

if __name__ == "__main__":
    a, r = ask("List each document’s total estimate and compute the combined total.")
    print(a, r)
