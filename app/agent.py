"""THE HARNESS: everything around the model that turns one LLM call into an assistant.

Per turn:
  0. redact PII locally, so nothing below sees raw card numbers or IBANs
  1. triage, in parallel: input classifier | router | memory + RAG assembly
     -> blocked (injection / off-topic)? safe reply, stop
  2. FAQ route: answer cache hit? return it (no model call), stop
  3. run the agent loop on the routed model: model -> tools (MCP) -> model ... until done
  4. output guardrails: redact PII, check groundedness; store FAQ answers in the cache
  5. save the turn to episodic memory, consolidate in the background

The system prompt stays fixed. Everything that changes per user or per question
goes into the final user turn, so the tools + system prefix is prompt-cached.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from typing import Iterator

from app import auth, cache, guardrails, llm, mcp_client, memory, rag, router, summarizer


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
        """Full working memory: personal memory + skills + docs."""
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

    def assemble_faq(self, question: str) -> WorkingMemory:
        """Docs-only working memory for the FAQ route. No profile, facts, episodes or
        history, so the answer doesn't depend on who asked. That's what makes it safe
        to cache and give to other users with the same role."""
        return WorkingMemory(
            user_id=self.user_id, role=self.role, question=question,
            skills=memory.match_skills(question),
            hits=rag.retrieve(question, access=auth.doc_access(self.role)),
        )

    def stream(self, question: str, history: list[dict], result: dict) -> Iterator[str]:
        """Yield the answer as it streams. Afterwards `result` holds reply, memory,
        tool_calls, guard_in, guard_out, question (redacted) and meta (route, cache, cost, timings).

        `history` is the earlier turns of this session, without the new question.
        """
        t_start = time.perf_counter()
        timings: dict[str, float] = {}

        def lap(name: str, since: float) -> float:
            now = time.perf_counter()
            timings[name] = round(now - since, 3)
            return now

        # 0. PII first: it's local and fast, and nothing after it sees the raw text.
        question, pii = guardrails.redact_pii(question)
        prev_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), None)

        # 1. TRIAGE IN PARALLEL. These three don't depend on each other, so running them
        # side by side costs max(latencies), not their sum. On a blocked turn the
        # speculative router/memory work is wasted, but it's cheap.
        t = time.perf_counter()
        with ThreadPoolExecutor(max_workers=3) as ex:
            f_verdict = ex.submit(guardrails.classify, question) if guardrails.CFG["classify_input"] else None
            f_route = ex.submit(router.route, question, prev_user)
            f_wm = ex.submit(self.assemble, question)
            verdict = f_verdict.result() if f_verdict else {"verdict": "allow", "reason": "classifier disabled"}
            route, wm = f_route.result(), f_wm.result()
        t = lap("triage", t)

        guard_in = guardrails.InputCheck(text=question, pii=pii, verdict=verdict["verdict"], reason=verdict["reason"])
        meta = {"route": route["route"], "route_reason": route["reason"], "tier": route["tier"],
                "model": llm.CONFIG["models"][route["tier"]]["id"], "cache": None, "cost_usd": 0.0, "timings": timings}
        result.update(guard_in=guard_in, guard_out=None, question=question, tool_calls=[], meta=meta)

        if guard_in.blocked:
            # "End loop" guardrail: no model answer, no tools, not saved to memory.
            yield guard_in.safe_reply
            meta.update(route="blocked", model="guardrail")
            result.update(
                reply=llm.Reply(guard_in.safe_reply, model="guardrail", tier="none", stop_reason="guardrail_blocked", usage={}),
                memory=WorkingMemory(user_id=self.user_id, role=self.role, question=question),
            )
            lap("total", t_start)
            return

        # 2. FAQ ROUTE: standalone policy question -> try the answer cache first.
        is_faq = route["route"] == "faq"
        mcp = mcp_client.client()
        if is_faq:
            hit = cache.lookup(question, self.role)
            t = lap("cache_lookup", t)
            if hit:
                yield hit["answer"]
                meta.update(cache=hit, model="answer-cache")
                memory.save_episode(self.user_id, self.session_id, question, hit["answer"])
                result.update(
                    reply=llm.Reply(hit["answer"], model="answer-cache", tier="none", stop_reason="cache_hit", usage={}),
                    memory=WorkingMemory(user_id=self.user_id, role=self.role, question=question),
                )
                lap("total", t_start)
                return
            wm = self.assemble_faq(question)
            messages = [{"role": "user", "content": wm.render()}]  # no history: standalone by definition
            tools: list[dict] = []                                 # docs-only: nothing user-specific
        else:
            messages = window(history) + [{"role": "user", "content": wm.render()}]
            tools = mcp.claude_tools(self.role)

        # 3. THE AGENT LOOP: call the model; if it asks for tools, run them, send back
        # the results, and call it again. Stop when it answers without a tool call.
        tier = route["tier"]
        tool_calls: list[dict] = []
        texts: list[str] = []
        usage: dict = {}
        for _ in range(llm.CONFIG["agent"]["max_tool_rounds"]):
            inner: dict = {}
            if texts:
                yield "\n\n"  # separate text written before and after a tool round
            yield from llm.stream_chat(messages, tools=tools or None, tier=tier, result=inner)
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
        t = lap("agent_loop", t)

        # 4. OUTPUT GUARDRAILS: redact PII, and check the answer against its sources.
        # Trade-off: the answer has already streamed, so a redaction only takes effect
        # in the stored copy (what the UI re-renders and memory keeps). For strict
        # deployments, buffer the answer and check it before showing anything.
        guard_out = guardrails.check_output(
            "\n\n".join(texts), wm.hits, tool_calls, wm.skills, [t["name"] for t in tools]
        )
        if guard_out.grounded is False:
            yield guardrails.UNGROUNDED_NOTE
        answer = guard_out.text
        # Cache only answers that are safe to share: FAQ route, no tools, grounded, no PII.
        if is_faq and not tool_calls and guard_out.grounded and not guard_out.pii:
            cache.store(question, self.role, answer)
        t = lap("output_guard", t)

        # 5. MEMORY
        memory.save_episode(self.user_id, self.session_id, question, answer)
        # Consolidation is a separate LLM call, so it runs off the request path.
        threading.Thread(
            target=summarizer.maybe_consolidate, args=(self.user_id,), daemon=True, name="summarizer"
        ).start()

        reply.text, reply.usage = answer, usage
        meta["cost_usd"] = round(llm.cost_usd(usage, tier), 6)  # main model only; see tracing (m8) for the rest
        lap("total", t_start)
        result.update(reply=reply, memory=wm, tool_calls=tool_calls, guard_out=guard_out)
