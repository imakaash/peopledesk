"""THE HARNESS: everything around the model that turns one LLM call into an assistant.

Per turn:
  1. assemble working memory: profile + facts + past episodes + skills + RAG docs
  2. call the model (milestone 4 turns this into a tool loop)
  3. write the turn to episodic memory
  4. in the background, let the summarizer consolidate episodes into semantic memory

The system prompt stays fixed. Everything that changes per user or per question
goes into the final user turn, so the system prefix can be cached.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import date
from typing import Iterator

from app import llm, memory, rag, summarizer


@dataclass
class WorkingMemory:
    user_id: str
    question: str
    profile: str = ""
    facts: list[str] = field(default_factory=list)
    episodes: list[dict] = field(default_factory=list)
    skills: list[dict] = field(default_factory=list)
    hits: list[dict] = field(default_factory=list)

    def render(self) -> str:
        parts = [
            f'<context today="{date.today().isoformat()}" user_id="{self.user_id}"/>',
            f"<user_profile>\n{self.profile or 'No profile yet.'}\n</user_profile>",
        ]
        if self.facts:
            parts.append("<long_term_memory>\n" + "\n".join(f"- {f}" for f in self.facts) + "\n</long_term_memory>")
        if self.episodes:
            eps = "\n\n".join(
                f"[{e['created_at'][:10]}] User: {e['question']}\nAssistant: {e['answer'][:400]}"
                for e in self.episodes
            )
            parts.append(f"<recent_conversations>\n{eps}\n</recent_conversations>")
        if self.skills:
            parts.append("<skills>\n" + "\n".join(
                f'<skill name="{s["name"]}">\n{s["body"]}\n</skill>' for s in self.skills
            ) + "\n</skills>")
        parts.append(rag.format_documents(self.hits))
        parts.append(f"<question>\n{self.question}\n</question>")
        return "\n\n".join(parts)


def window(history: list[dict], n: int | None = None) -> list[dict]:
    """The last n chat messages, as role/content only. Trimmed so the window
    starts with a user turn, as the API requires."""
    msgs = [{"role": m["role"], "content": m["content"]} for m in history[-(n or llm.CONFIG["memory"]["history_window"]):]]
    while msgs and msgs[0]["role"] != "user":
        msgs.pop(0)
    return msgs


class Agent:
    def __init__(self, user_id: str, session_id: str):
        self.user_id = user_id
        self.session_id = session_id

    def assemble(self, question: str) -> WorkingMemory:
        facts = memory.relevant_facts(self.user_id, question)
        # Query expansion: "when do I tell HR about my leave?" doesn't say *which* leave,
        # but memory might ("Alice plans parental leave..."). Searching with the top facts
        # added to the question lets memory steer retrieval to the right policy section.
        search_query = " ".join([question, *facts[:2]])
        return WorkingMemory(
            user_id=self.user_id,
            question=question,
            profile=memory.get_profile(self.user_id),
            facts=facts,
            episodes=memory.recent_episodes(self.user_id, exclude_session=self.session_id),
            skills=memory.match_skills(question),
            hits=rag.retrieve(search_query),  # milestone 5 passes the user's access tags here
        )

    def stream(self, question: str, history: list[dict], result: dict) -> Iterator[str]:
        """Yield the answer as it streams. Afterwards result holds 'reply' and 'memory'.

        `history` is the earlier turns of this session, without the new question.
        """
        wm = self.assemble(question)
        messages = window(history) + [{"role": "user", "content": wm.render()}]
        inner: dict = {}
        yield from llm.stream_chat(messages, result=inner)
        reply = inner["reply"]

        memory.save_episode(self.user_id, self.session_id, question, reply.text)
        # Consolidation is a separate LLM call, so it runs off the request path.
        threading.Thread(target=summarizer.maybe_consolidate, args=(self.user_id,), daemon=True).start()

        result.update(reply=reply, memory=wm)
