"""THE HARNESS: everything around the model that turns one LLM call into an assistant.

Per turn:
  0. input guardrails: redact PII, block injection / off-topic (ends the turn early)
  1. assemble working memory: profile + facts + past episodes + skills + RAG docs
  2. run the agent loop: model -> tool calls (via MCP) -> model ... until it answers
  3. output guardrails: redact PII, check groundedness
  4. write the turn to episodic memory
  5. in the background, let the summarizer consolidate episodes into semantic memory

The system prompt stays fixed. Everything that changes per user or per question
goes into the final user turn, so the system prefix can be cached.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import date
from typing import Iterator

from app import auth, guardrails, llm, mcp_client, memory, rag, summarizer


@dataclass
class WorkingMemory:
    user_id: str
    role: str
    question: str
    profile: str = ""
    facts: list[str] = field(default_factory=list)
    episodes: list[dict] = field(default_factory=list)
    skills: list[dict] = field(default_factory=list)
    hits: list[dict] = field(default_factory=list)

    def render(self) -> str:
        parts = [
            f'<context today="{date.today().isoformat()}" user_id="{self.user_id}" role="{self.role}"/>',
            f"<user_profile>\n{self.profile or 'No profile yet.'}\n</user_profile>",
        ]
        if self.facts:
            parts.append("<long_term_memory>\n" + "\n".join(f"- {f}" for f in self.facts) + "\n</long_term_memory>")
        if self.episodes:
            eps = "\n\n".join(
                f"[{e['created_at'][:10]}] User: {e['question']}\nAssistant: {e['answer'][:400]}"
                for e in self.episodes
            )
            # Labelled at the point of use: statuses and balances in old chats go stale.
            parts.append(
                '<recent_conversations note="From earlier sessions. Statuses, balances and requests mentioned '
                'here may have changed; check with a tool before stating them as current.">\n'
                f"{eps}\n</recent_conversations>"
            )
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
    def __init__(self, token: str, session_id: str):
        # Identity comes only from the verified token: never from the chat or the model.
        claims = auth.verify_token(token)
        self.token = token
        self.user_id = claims["sub"]
        self.role = claims["role"]
        self.session_id = session_id

    def assemble(self, question: str) -> WorkingMemory:
        facts = memory.relevant_facts(self.user_id, question)
        # Query expansion: "when do I tell HR about my leave?" doesn't say *which* leave,
        # but memory might ("Alice plans parental leave..."). Searching with the top facts
        # added to the question lets memory steer retrieval to the right policy section.
        search_query = " ".join([question, *facts[:2]])
        return WorkingMemory(
            user_id=self.user_id,
            role=self.role,
            question=question,
            profile=memory.get_profile(self.user_id),
            facts=facts,
            episodes=memory.recent_episodes(self.user_id, exclude_session=self.session_id),
            skills=memory.match_skills(question),
            # RBAC for documents: the role's tags filter the vector search itself.
            hits=rag.retrieve(search_query, access=auth.doc_access(self.role)),
        )

    def stream(self, question: str, history: list[dict], result: dict) -> Iterator[str]:
        """Yield the answer as it streams. Afterwards result holds 'reply', 'memory'
        and 'tool_calls'.

        `history` is the earlier turns of this session, without the new question.
        """
        # INPUT GUARDRAILS: redact PII, then screen for injection / off-topic.
        # A blocked turn ends here: no retrieval, no tools, and it's not saved to memory.
        guard_in = guardrails.check_input(question)
        question = guard_in.text
        if guard_in.blocked:
            yield guard_in.safe_reply
            result.update(
                reply=llm.Reply(guard_in.safe_reply, model="guardrail", tier="none", stop_reason="guardrail_blocked", usage={}),
                memory=WorkingMemory(user_id=self.user_id, role=self.role, question=question),
                tool_calls=[], guard_in=guard_in, guard_out=None, question=question,
            )
            return

        wm = self.assemble(question)
        messages = window(history) + [{"role": "user", "content": wm.render()}]
        mcp = mcp_client.client()
        tools = mcp.claude_tools(self.role)
        tool_calls: list[dict] = []
        texts: list[str] = []
        usage: dict = {}

        # THE AGENT LOOP: call the model; if it asks for tools, run them, send back
        # the results, and call it again. Stop when it answers without a tool call.
        for _ in range(llm.CONFIG["agent"]["max_tool_rounds"]):
            inner: dict = {}
            if texts:
                yield "\n\n"  # separate text written before and after a tool round
            yield from llm.stream_chat(messages, tools=tools, result=inner)
            reply = inner["reply"]
            usage = {k: usage.get(k, 0) + v for k, v in reply.usage.items()}
            if reply.text:
                texts.append(reply.text)
            # Append the assistant turn exactly as returned, thinking blocks included.
            messages.append({"role": "assistant", "content": reply.content})

            if reply.stop_reason != "tool_use":
                break  # end_turn, refusal, or max_tokens: never run tools from a cut-off turn

            # One assistant turn may ask for several tools at once (parallel tool use).
            # Every result goes back in ONE user message, matched by tool_use_id.
            results = []
            for block in (b for b in reply.content if b.type == "tool_use"):
                out = mcp.call(block.name, block.input, token=self.token)
                tool_calls.append({"name": block.name, "input": block.input, "output": out.text, "is_error": out.is_error})
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": out.text, "is_error": out.is_error})
            messages.append({"role": "user", "content": results})
        else:
            stop = "I had to stop after too many tool steps. Could you narrow the request down?"
            texts.append(stop)
            yield "\n\n" + stop

        # OUTPUT GUARDRAILS: redact PII, and check the answer against its sources.
        # Trade-off: the answer has already streamed, so a redaction only takes effect
        # in the stored copy (what the UI re-renders and memory keeps). For strict
        # deployments, buffer the answer and check it before showing anything.
        guard_out = guardrails.check_output(
            "\n\n".join(texts), wm.hits, tool_calls, wm.skills, [t["name"] for t in tools]
        )
        if guard_out.grounded is False:
            yield guardrails.UNGROUNDED_NOTE
        answer = guard_out.text

        memory.save_episode(self.user_id, self.session_id, question, answer)
        # Consolidation is a separate LLM call, so it runs off the request path.
        threading.Thread(
            target=summarizer.maybe_consolidate, args=(self.user_id,), daemon=True, name="summarizer"
        ).start()

        reply.text, reply.usage = answer, usage
        result.update(reply=reply, memory=wm, tool_calls=tool_calls, guard_in=guard_in, guard_out=guard_out, question=question)
