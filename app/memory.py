"""The memory types from the harness diagram, apart from working memory.

  procedural  skills/*.md       how to do things; loaded when a question matches a trigger
  episodic    SQLite episodes   what happened: every Q&A turn, timestamped
  semantic    Chroma user_facts what we know: facts distilled from episodes by the summarizer
              SQLite profiles   plus one short running profile per user

Working memory, the fourth type, is the context window itself. agent.py assembles
it from these stores on every turn.
"""
from __future__ import annotations

import re
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from functools import cache

from app.llm import CONFIG, ROOT
from app.rag import collection, parse_frontmatter

SKILLS_DIR = ROOT / "skills"
FACTS = "user_facts"

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    created_at TEXT NOT NULL,
    consolidated INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS profiles (
    user_id TEXT PRIMARY KEY,
    profile TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _db() -> sqlite3.Connection:
    # A new connection per call: the summarizer runs in a background thread, and
    # SQLite connections can't be shared across threads.
    conn = sqlite3.connect(ROOT / CONFIG["memory"]["db_path"])
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


# --- procedural ---------------------------------------------------------------

@cache
def load_skills() -> list[dict]:
    skills = []
    for path in sorted(SKILLS_DIR.glob("*.md")):
        meta, body = parse_frontmatter(path)
        skills.append({**meta, "body": body.strip()})
    return skills


def match_skills(question: str) -> list[dict]:
    """Keyword triggers: cheap, predictable, and easy to debug.
    (Embedding-based matching is the next step up if this misses too often.)"""
    q = question.lower()
    return [
        s for s in load_skills()
        # whole words, optional plural: "training" matches "trainings", but "train" doesn't
        if any(re.search(rf"\b{re.escape(t)}s?\b", q) for t in s["triggers"])
    ]


# --- episodic -----------------------------------------------------------------

def save_episode(user_id: str, session_id: str, question: str, answer: str) -> None:
    with closing(_db()) as db, db:
        db.execute(
            "INSERT INTO episodes (user_id, session_id, question, answer, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, session_id, question, answer, _now()),
        )


def recent_episodes(user_id: str, exclude_session: str, n: int | None = None) -> list[dict]:
    """The last n Q&A pairs from *earlier* sessions, oldest first. The current
    session is already in the chat history, so it is left out here."""
    n = n or CONFIG["memory"]["recent_episodes"]
    with closing(_db()) as db:
        rows = db.execute(
            "SELECT * FROM episodes WHERE user_id = ? AND session_id != ? ORDER BY id DESC LIMIT ?",
            (user_id, exclude_session, n),
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


def unconsolidated_episodes(user_id: str) -> list[dict]:
    with closing(_db()) as db:
        rows = db.execute(
            "SELECT * FROM episodes WHERE user_id = ? AND consolidated = 0 ORDER BY id", (user_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def mark_consolidated(ids: list[int]) -> None:
    with closing(_db()) as db, db:
        db.executemany("UPDATE episodes SET consolidated = 1 WHERE id = ?", [(i,) for i in ids])


# --- semantic -----------------------------------------------------------------

def get_profile(user_id: str) -> str:
    with closing(_db()) as db:
        row = db.execute("SELECT profile FROM profiles WHERE user_id = ?", (user_id,)).fetchone()
    return row["profile"] if row else ""


def set_profile(user_id: str, profile: str) -> None:
    with closing(_db()) as db, db:
        db.execute(
            "INSERT INTO profiles (user_id, profile, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET profile = excluded.profile, updated_at = excluded.updated_at",
            (user_id, profile, _now()),
        )


def add_facts(user_id: str, facts: list[str]) -> None:
    if facts:
        collection(FACTS).add(
            ids=[str(uuid.uuid4()) for _ in facts],
            documents=facts,
            metadatas=[{"user_id": user_id, "created_at": _now()} for _ in facts],
        )


def all_facts(user_id: str) -> list[str]:
    return collection(FACTS).get(where={"user_id": user_id})["documents"]


def relevant_facts(user_id: str, question: str, k: int | None = None) -> list[str]:
    """Semantic search over this user's facts only. The user_id filter keeps one
    user's memories out of another user's context."""
    col = collection(FACTS)
    n = min(k or CONFIG["memory"]["facts_top_k"], len(all_facts(user_id)))
    if n == 0:
        return []
    res = col.query(query_texts=[question], n_results=n, where={"user_id": user_id})
    return res["documents"][0]


def forget_user(user_id: str) -> None:
    """Delete everything remembered about a user (useful while experimenting)."""
    collection(FACTS).delete(where={"user_id": user_id})
    with closing(_db()) as db, db:
        db.execute("DELETE FROM episodes WHERE user_id = ?", (user_id,))
        db.execute("DELETE FROM profiles WHERE user_id = ?", (user_id,))
