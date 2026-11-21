"""Semantic answer cache: reuse the answer to a policy FAQ that was asked before.

Unlike prompt caching (which makes the model read a repeated *prefix* cheaply), this
skips the model entirely: embed the question, find a near-identical earlier
question, and return its stored answer. 0 tokens, and a few milliseconds.

What makes a cached answer safe to reuse for someone else:
  - Only FAQ-route answers are stored. Those are generated from docs alone, with no
    tools, no personal memory and no chat history, so they don't depend on who asked.
  - Entries are scoped by role: an HR admin's answer may quote HR-only docs.
  - Only answers that passed the groundedness check are stored.
  - The key includes a fingerprint of the system prompt, docs and skills. Editing
    any of them makes every old entry miss, with no manual invalidation needed.
  - Two-stage match. Embeddings alone can't do it: "primary caregiver" vs "secondary
    caregiver" scores 0.974, above real paraphrases, but the answers differ (16 vs 6
    weeks). So similarity >= min_similarity only picks a *candidate*, and a tiny
    Haiku call confirms that both questions have the same correct answer.

    python -m app.cache clear      # drop every cached answer
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import uuid
from functools import cache as memoize

from app import llm, rag

CFG = llm.CONFIG["answer_cache"]
COLLECTION = "answer_cache"


@memoize
def fingerprint() -> str:
    h = hashlib.sha256(llm.load_system_prompt().encode())
    for folder in ("data/docs", "skills"):
        for p in sorted((llm.ROOT / folder).glob("*.md")):
            h.update(p.name.encode() + p.read_bytes())
    return h.hexdigest()[:16]


def lookup(question: str, role: str) -> tuple[dict | None, str]:
    """Return (hit or None, reason). The reason makes misses explainable in the UI and traces."""
    if not CFG["enabled"]:
        return None, "answer cache disabled"
    col = rag.collection(COLLECTION)
    where = {"$and": [{"role": role}, {"fingerprint": fingerprint()}, {"created_at": {"$gt": time.time() - CFG["ttl_hours"] * 3600}}]}
    res = col.query(query_texts=[question], n_results=CFG["candidates"], where=where)
    if not res["ids"][0]:
        return None, f"no cached answers yet for role '{role}'"
    rejected = []
    # Try the closest few: the best embedding match isn't always the right one
    # ("do I get" can score above "do employees get" for "do we get").
    for cached_q, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        score = round(1 - dist, 3)
        if score < CFG["min_similarity"]:
            break  # results are sorted best first, so the rest are worse
        if not CFG["verify_with_llm"] or same_question(question, cached_q):
            return {"answer": meta["answer"], "cached_question": cached_q, "score": score}, "hit"
        rejected.append(f"'{cached_q}' ({score})")
    if rejected:
        return None, "verifier rejected: " + ", ".join(rejected)
    best_q, best = res["documents"][0][0], round(1 - res["distances"][0][0], 3)
    return None, f"closest cached question too different (similarity {best}): '{best_q}'"


VERIFY_SYSTEM = """Decide whether two questions to an HR assistant ask for exactly the same information, so that one correct general-policy answer would fully and correctly answer both.

They are different when they differ in:
- which group the policy treats differently: primary vs secondary caregiver, full-time vs part-time, new joiners vs everyone
- what is asked about: sick vs annual leave, hotels vs flights
- where: Paris vs Berlin
- which rule: carry over vs pay out, notice vs approval time

These do NOT make them different: the person asking ("I", "we", "employees", "staff" asking about the same general entitlement), rewording, word order, punctuation and politeness."""

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {"same": {"type": "boolean"}},
    "required": ["same"],
    "additionalProperties": False,
}


def same_question(a: str, b: str) -> bool:
    reply = llm.complete(
        [{"role": "user", "content": f"<question_a>\n{a}\n</question_a>\n<question_b>\n{b}\n</question_b>"}],
        system=VERIFY_SYSTEM, tier="cheap", json_schema=VERIFY_SCHEMA,
        effort=CFG["verify_effort"],  # at "low" the same pair flipped between yes and no
    )
    return json.loads(reply.text)["same"]


def store(question: str, role: str, answer: str) -> None:
    if CFG["enabled"]:
        rag.collection(COLLECTION).add(
            ids=[str(uuid.uuid4())],
            documents=[question],  # the question is what gets embedded and matched
            metadatas=[{"role": role, "fingerprint": fingerprint(), "answer": answer, "created_at": time.time()}],
        )


def clear() -> None:
    try:
        rag.chroma_client().delete_collection(COLLECTION)
    except Exception:  # collection doesn't exist yet: nothing to clear
        pass


if __name__ == "__main__":
    if sys.argv[1:] == ["clear"]:
        clear()
        print("Answer cache cleared.")
    else:
        print(__doc__)
