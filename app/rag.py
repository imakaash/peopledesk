"""RAG: put HR docs into Chroma, then retrieve the most relevant chunks for a question.

    python -m app.rag ingest                  # (re)build the index from data/docs/*.md
    python -m app.rag search "sick leave"     # see what retrieval returns, with scores
"""
from __future__ import annotations

import sys
from functools import cache
from pathlib import Path

import chromadb
import yaml
from chromadb.config import Settings

from app import tracing
from app.llm import CONFIG, ROOT

DOCS_DIR = ROOT / "data" / "docs"
COLLECTION = "hr_docs"


@cache
def chroma_client():
    return chromadb.PersistentClient(
        path=str(ROOT / CONFIG["rag"]["chroma_path"]),
        settings=Settings(anonymized_telemetry=False),
    )


def collection(name: str):
    # Chroma embeds text locally with all-MiniLM-L6-v2 (it downloads ~80 MB on first use).
    # Cosine distance: similarity = 1 - distance, so 1.0 means identical meaning.
    return chroma_client().get_or_create_collection(name, configuration={"hnsw": {"space": "cosine"}})


def _collection():
    return collection(COLLECTION)


def parse_frontmatter(path: Path) -> tuple[dict, str]:
    """Split a markdown file into its frontmatter (title, access) and its body."""
    _, front, body = path.read_text().split("---", 2)
    return yaml.safe_load(front), body


def chunk_doc(path: Path) -> list[dict]:
    """One chunk per '## ' section.

    Policy docs are written as self-contained sections, so cutting at headings keeps
    each rule whole. Fixed-size chunks could cut a rule in half. The doc title and
    section heading are put at the start of the chunk text, so the embedding (and the
    model) knows where the text came from.
    """
    meta, body = parse_frontmatter(path)
    chunks = []
    for i, block in enumerate(s for s in body.split("\n## ") if s.strip()):
        heading, _, text = block.lstrip("# ").partition("\n")
        chunks.append({
            "id": f"{path.name}#{i}",
            "text": f"{meta['title']} - {heading.strip()}\n{text.strip()}",
            "metadata": {
                "source": path.name,
                "section": heading.strip(),
                "access": meta["access"],   # "all" or "hr"; filtered at query time
            },
        })
    return chunks


def ingest() -> int:
    col = _collection()
    existing = col.get()["ids"]
    if existing:
        col.delete(ids=existing)  # full rebuild keeps the index in sync with the files
    chunks = [c for p in sorted(DOCS_DIR.glob("*.md")) for c in chunk_doc(p)]
    col.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )
    return len(chunks)


def retrieve(query: str, access: list[str] | None = None, k: int | None = None) -> list[dict]:
    """Top-k chunks the caller is allowed to see, best first.

    `access` is the list of doc tags the user may read. For now everyone gets ["all"];
    milestone 5 (RBAC) passes ["all", "hr"] for HR admins. The filter runs inside the
    vector search, so restricted chunks are never returned at all, rather than being
    removed afterwards.
    """
    cfg = CONFIG["rag"]
    access = access or ["all"]
    with tracing.span("rag.retrieve", "RETRIEVER", {"rag.access": access}, input=query) as s:
        res = _collection().query(
            query_texts=[query],
            n_results=k or cfg["top_k"],
            where={"access": {"$in": access}},
        )
        hits = []
        for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            score = 1 - dist
            if score >= cfg["min_score"]:  # drop weak matches so they don't distract the model
                hits.append({"text": text, "score": round(score, 3), **meta})
        # OpenInference document attributes: Phoenix shows these as a ranked list with scores.
        for i, h in enumerate(hits):
            tracing.set_attrs(s, {
                f"retrieval.documents.{i}.document.id": f"{h['source']} > {h['section']}",
                f"retrieval.documents.{i}.document.score": h["score"],
                f"retrieval.documents.{i}.document.content": h["text"] if tracing.content_enabled() else None,
            })
        return hits


def format_documents(hits: list[dict]) -> str:
    """Render retrieved chunks for the prompt. The agent places this in the user turn."""
    if not hits:
        docs = "<no_matching_documents/>"
    else:
        docs = "\n".join(
            f'<document source="{h["source"]}" section="{h["section"]}">\n{h["text"]}\n</document>'
            for h in hits
        )
    return f"<company_documents>\n{docs}\n</company_documents>"


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "ingest":
        print(f"Indexed {ingest()} chunks from {DOCS_DIR}")
    elif cmd == "search" and len(sys.argv) > 2:
        access = ["all", "hr"] if "--hr" in sys.argv else ["all"]
        query = " ".join(a for a in sys.argv[2:] if a != "--hr")
        for h in retrieve(query, access=access):
            print(f"{h['score']:.3f}  [{h['access']}] {h['source']} > {h['section']}")
    else:
        print(__doc__)
