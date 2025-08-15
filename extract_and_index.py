# extract_and_index.py
# cspell:ignore pdfplumber
from pathlib import Path
import hashlib, json, re
from typing import Iterable, Dict, Any, List, Tuple
from pydantic import BaseModel
import pdfplumber
import networkx as nx

from txtai.pipeline import Textractor
from config_embed import build_embeddings

PDF_DIR   = Path("data/pdfs")
INDEX_DIR = "index"
DATA_DIR  = Path("data")
HM_DIR    = DATA_DIR / "hashmaps"
for d in (PDF_DIR, DATA_DIR, HM_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------- simple schema for mapped fields ----------
class LineItem(BaseModel):
    desc: str
    qty: float | None = None
    unit: str | None = None
    unit_cost: float | None = None
    ext_cost: float | None = None
    page: int | None = None

class DocMap(BaseModel):
    doc_id: str
    docname: str
    vendor: str | None = None
    grand_total: float | None = None
    items: List[LineItem] = []

money_rx = re.compile(r"\$?\s*([0-9]{1,3}(?:[, ][0-9]{3})*(?:\.[0-9]{2})?)")

def parse_money(s: str) -> float | None:
    s = (s or "").replace(",", "").replace(" ", "")
    m = money_rx.search(s)
    return float(m.group(1)) if m else None

# --------- hashing (stable, content-addressed) ----------
def make_id(prefix: str, payload: dict | str) -> str:
    if isinstance(payload, dict):
        payload = json.dumps(payload, ensure_ascii=False, sort_keys=True)[:2048]
    h = hashlib.blake2s(payload.encode(), digest_size=8).hexdigest()
    return f"{prefix}:{h}"

# --------- chunkers ----------
textract = Textractor(paragraphs=True, sections=True, minlength=80)

def paragraph_chunks(pdfpath: Path, doc_id: str) -> Iterable[Dict[str, Any]]:
    chunks = textract(str(pdfpath))
    if isinstance(chunks, str):
        chunks = [chunks]
    for i, text in enumerate(chunks):
        cid = make_id("PCH", {"doc": doc_id, "i": i, "t": text[:300]})
        yield {
            "id": cid, "text": text, "docname": pdfpath.name,
            "source": str(pdfpath), "type": "paragraph"
        }

def table_rows(pdfpath: Path, doc_id: str) -> Iterable[Dict[str, Any]]:
    try:
        with pdfplumber.open(pdfpath) as pdf:
            for page_index, page in enumerate(pdf.pages, start=1):
                for table in page.extract_tables() or []:
                    for r_i, row in enumerate(table):
                        cells = [(c or "").strip() for c in row]
                        if not any(cells):
                            continue
                        rowtext = " | ".join(cells)
                        cid = make_id("TBL", {"doc": doc_id, "p": page_index, "r": r_i, "c": cells[:10]})
                        yield {
                            "id": cid, "text": rowtext, "docname": pdfpath.name,
                            "source": str(pdfpath), "type": "table-row", "page": page_index
                        }
    except Exception:
        return

# --------- lightweight metadata sniffers ----------
def sniff_vendor(text: str) -> str | None:
    for pat in [r"Vendor\s*[:\-]\s*(.+)", r"Client\s*[:\-]\s*(.+)"]:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()[:120]
    return None

def sniff_grand_total(fulltext: str) -> float | None:
    for label in ["grand total", "contract total", "total"]:
        m = re.search(label + r".{0,40}\$?\s*([0-9][0-9,\. ]+)", fulltext, flags=re.IGNORECASE)
        if m:
            try:
                return float(m.group(1).replace(",", "").replace(" ", ""))
            except Exception:
                pass
    return None

# --------- build graph + maps ----------
def build_graph_and_maps(pdf: Path) -> Tuple[nx.Graph, DocMap, List[Dict[str, Any]]]:
    doc_id = make_id("DOC", {"name": pdf.name, "size": pdf.stat().st_size})
    g = nx.DiGraph()
    g.add_node(doc_id, label="Document", docname=pdf.name, path=str(pdf))
    

    # Gather all text to sniff vendor/grand total
    pchunks = list(paragraph_chunks(pdf, doc_id))
    fulltext = "\n".join(ch["text"] for ch in pchunks)
    vendor = sniff_vendor(fulltext)
    gt = sniff_grand_total(fulltext)

    # Optional entity node (Vendor)
    ent_id = None
    if vendor:
        ent_id = make_id("ENT", {"type": "Vendor", "name": vendor})
        g.add_node(ent_id, label="Entity", type="Vendor", name=vendor)
        g.add_edge(doc_id, ent_id, rel="HAS_ENTITY")

    # Table rows → line items + chunk nodes
    row_chunks = list(table_rows(pdf, doc_id))
    line_items: List[LineItem] = []
    for rc in row_chunks:
        g.add_node(rc["id"], label="Chunk", type=rc["type"], page=rc.get("page"), docname=rc["docname"])
        g.add_edge(doc_id, rc["id"], rel="HAS_CHUNK")
        # Make a light LineItem node if last cell looks like money
        parts = [p.strip() for p in rc["text"].split("|")]
        ext_cost = parse_money(parts[-1]) if len(parts) >= 2 else None
        if ext_cost is not None:
            li = LineItem(desc=" | ".join(parts[:-1])[:200], ext_cost=ext_cost, page=rc.get("page"))
            line_items.append(li)
            lid = make_id("LIT", {"doc": doc_id, "page": rc.get("page"), "desc": li.desc, "ext": ext_cost})
            g.add_node(lid, label="LineItem", ext_cost=ext_cost, page=rc.get("page"), desc=li.desc)
            g.add_edge(rc["id"], lid, rel="HAS_LINEITEM")
            if ent_id:
                g.add_edge(lid, ent_id, rel="ITEM_FOR")

    # Paragraph chunks → chunk nodes (link to doc and vendor entity)
    for pc in pchunks:
        g.add_node(pc["id"], label="Chunk", type=pc["type"], docname=pc["docname"])
        g.add_edge(doc_id, pc["id"], rel="HAS_CHUNK")
        if ent_id:
            g.add_edge(pc["id"], ent_id, rel="MENTIONS")

    docmap = DocMap(doc_id=doc_id, docname=pdf.name, vendor=vendor, grand_total=gt, items=line_items)
    all_chunks = row_chunks + pchunks
    return g, docmap, all_chunks

def save_json(path: Path, data: Any):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    G = nx.DiGraph()
    docmaps : List[DocMap] = []
    all_chunks_for_index : List[Dict[str, Any]] = []

    for pdf in sorted(PDF_DIR.glob("*.pdf")):
        print(f"→ Processing {pdf.name}")
        g, dm, chunks = build_graph_and_maps(pdf)
        G = nx.compose(G, g)
        docmaps.append(dm)
        all_chunks_for_index.extend(chunks)

    # Persist graph + maps (human/audit friendly)
    from networkx.readwrite import json_graph
    save_json(DATA_DIR / "graph.json", json_graph.node_link_data(G))
    save_json(DATA_DIR / "mapped.json", [d.model_dump() for d in docmaps])

    # Tiny hashmaps (can swap to SQLite/Redis later)
    chunk_map = {c["id"]: c for c in all_chunks_for_index}
    save_json(HM_DIR / "chunk_map.json", chunk_map)

    # ---- Index with txtai ----
    emb = build_embeddings()
    def stream():
        for ch in all_chunks_for_index:
            meta = {
                "text": ch["text"],
                "docname": ch["docname"],
                "source": ch["source"],
                "type": ch["type"],
            }
            if "page" in ch:
                meta["page"] = ch["page"]
            yield (ch["id"], meta)

    with emb:
        emb.index(stream())
        emb.save(INDEX_DIR)

    print(" Graph saved → data/graph.json")
    print(" Mappings  → data/mapped.json")
    print(" Hashmaps  → data/hashmaps/chunk_map.json")
    print(" Vector index saved →", INDEX_DIR)
