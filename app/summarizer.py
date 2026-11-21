"""Memory consolidation: turn raw episodes into lasting semantic memory.

Every `consolidate_every_n` turns, a cheap Haiku call reads the new episodes and
updates the user's profile and long-term facts. It's the "Summarizer Agent" box in
the diagram: episodic -> semantic.

    python -m app.summarizer alice     # force a consolidation run for one user
"""
from __future__ import annotations

import json
import sys
from datetime import date

from app import llm, memory

SYSTEM = """You maintain the long-term memory of PeopleDesk, the HR assistant at Nimbus Labs.

You receive a user's current profile, the facts already stored about them, and their newest conversations. Return:
- profile: an updated 2-4 sentence profile of the user (role, team, start date, how they like answers), keeping what is still true and adding what is new.
- new_facts: facts worth remembering for future conversations that are not already stored. Each fact must stand alone (name the user, and give absolute dates instead of "next month"). Good facts: plans ("Alice plans parental leave from 2027-03-01"), preferences ("Alice prefers short answers"), situations ("Alice joined on 2026-11-03 and is in probation"). Return an empty list if nothing new is worth keeping.

Don't store medical details, salary figures, passwords, or personal data about other people. Don't store general policy facts, since those live in the company documents."""

SCHEMA = {
    "type": "object",
    "properties": {
        "profile": {"type": "string"},
        "new_facts": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["profile", "new_facts"],
    "additionalProperties": False,
}


def consolidate(user_id: str) -> dict | None:
    episodes = memory.unconsolidated_episodes(user_id)
    if not episodes:
        return None
    convo = "\n\n".join(
        f"[{e['created_at']}]\nUser: {e['question']}\nAssistant: {e['answer']}" for e in episodes
    )
    known = "\n".join(f"- {f}" for f in memory.all_facts(user_id)) or "(none)"
    prompt = (
        f"<today>{date.today().isoformat()}</today>\n<user_id>{user_id}</user_id>\n"
        f"<current_profile>\n{memory.get_profile(user_id) or '(empty)'}\n</current_profile>\n"
        f"<known_facts>\n{known}\n</known_facts>\n"
        f"<new_conversations>\n{convo}\n</new_conversations>"
    )
    reply = llm.complete([{"role": "user", "content": prompt}], system=SYSTEM, tier="cheap", json_schema=SCHEMA)
    data = json.loads(reply.text)
    memory.set_profile(user_id, data["profile"])
    memory.add_facts(user_id, data["new_facts"])
    memory.mark_consolidated([e["id"] for e in episodes])
    return data


def maybe_consolidate(user_id: str) -> dict | None:
    if len(memory.unconsolidated_episodes(user_id)) >= llm.CONFIG["memory"]["consolidate_every_n"]:
        return consolidate(user_id)
    return None


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    print(json.dumps(consolidate(sys.argv[1]), indent=2))
