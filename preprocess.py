# preprocess.py
import re
from typing import List
from langchain_core.documents import Document

QUESTION_START = re.compile(r"\b[1-9]\d{0,2}\.\s+")
# (opcional) si querés extraer número y enunciado cuando haya '?'
Q_HEADER_RE = re.compile(r"^\s*([1-9]\d{0,2})\.\s*(.+?)\?", re.DOTALL)

def flatten_text(txt: str) -> str:
    return re.sub(r"\s+", " ", txt).strip()

def insert_separators(txt: str) -> List[str]:
    marked = re.sub(rf"(?={QUESTION_START.pattern})", "\n|||Q|||\n", txt)
    return [b.strip() for b in marked.split("|||Q|||") if b.strip()]

def make_qa_chunks_from_docs(docs: List[Document]) -> List[Document]:
    qa_chunks: List[Document] = []
    for d in docs:
        base_meta = {
            "source": d.metadata.get("source"),
            "page": d.metadata.get("page"),
            "page_label": d.metadata.get("page_label"),
        }

        flat = flatten_text(d.page_content)
        raw_blocks = insert_separators(flat)

        # (Opcional, recomendado) descartar preámbulo que no es “N. ”
        if raw_blocks and not QUESTION_START.match(raw_blocks[0]):
            raw_blocks = raw_blocks[1:]

        # usar enrich_q_metadata para agregar q_no y q_text cuando se pueda
        for b in raw_blocks:
            qa_chunks.append(enrich_q_metadata(b, base_meta))

    return qa_chunks


def enrich_q_metadata(block: str, base_meta: dict) -> Document:
    """Agrega q_no y q_text si se pueden extraer; sino, deja solo base_meta."""
    meta = dict(base_meta)
    m = Q_HEADER_RE.search(block)
    if m:
        meta["q_no"] = int(m.group(1))
        meta["q_text"] = (m.group(2) + "?").strip()
    return Document(page_content=block, metadata=meta)
