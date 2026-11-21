"""Every call to Claude goes through this module.

Keeping it in one place means routing (milestone 7), tracing (milestone 8) and
evals (milestone 9) only need to hook in here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Iterator

import anthropic
import yaml
from dotenv import load_dotenv

from app import tracing

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())

# Sonnet 5.5 can hand a refused request to a fallback model inside the same call.
# Haiku 5.5 has no server-side fallback, so this only applies to the "smart" tier.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
REFUSAL_TEXT = "Sorry, I can't help with that request. Please contact hr@nimbuslabs.example."


def load_system_prompt(version: str | None = None) -> str:
    """Prompts are versioned files; config.yaml decides which one is live."""
    version = version or CONFIG["prompt_version"]
    return (ROOT / "prompts" / f"system_{version}.md").read_text().strip()


@cache
def client() -> anthropic.Anthropic:
    # Tracing must be registered before the client exists, so the SDK's own spans
    # go to the same tracer provider as ours.
    tracing.init(CONFIG["tracing"])
    mode = "content" if tracing.content_enabled() else "metadata_only"
    # Reads ANTHROPIC_API_KEY from the environment (.env is loaded above).
    return anthropic.Anthropic(open_telemetry={"traces": {"content_mode": mode}})


@dataclass
class Reply:
    text: str
    model: str
    tier: str
    stop_reason: str | None
    usage: dict
    content: list = field(default_factory=list)  # raw blocks (text, thinking, tool_use) for the tool loop


def _request(tier: str, system: str, messages: list[dict], tools: list[dict] | None = None) -> dict:
    m = CONFIG["models"][tier]
    req = dict(
        model=m["id"],
        max_tokens=m["max_tokens"],
        # Thinking is adaptive by default on these models; effort sets how hard it thinks.
        output_config={"effort": m["effort"]},
        # The system prompt is the stable prefix of every request, so mark it for
        # prompt caching. (It only caches once the prefix is long enough; it grows
        # past that once RAG and tools are added. Milestone 7 checks the hit rate.)
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=messages,
    )
    if tools:
        # Tools render before the system prompt, so the cache breakpoint on the
        # system block covers both.
        req["tools"] = tools
        # Automatic breakpoint on the last block, but only for tool requests. In a tool
        # loop, round N+1 re-sends round N's conversation and reads it back from the
        # cache. A single call with no tools would pay the write premium on a unique
        # question that's never read again.
        req["cache_control"] = {"type": "ephemeral"}
    return req


def cost_usd(usage: dict, tier: str) -> float:
    """Price a usage dict. Cached reads cost a tenth (Haiku) or a twentieth (Sonnet) of
    normal input. Cache writes cost 1.25x, so a prefix pays off from its second use."""
    p = CONFIG["models"][tier]["usd_per_mtok"]
    return (
        usage.get("input_tokens", 0) * p["input"]
        + usage.get("output_tokens", 0) * p["output"]
        + usage.get("cache_read_input_tokens", 0) * p["cache_read"]
        + usage.get("cache_creation_input_tokens", 0) * p["cache_write"]
    ) / 1_000_000


def _open_stream(tier: str, req: dict):
    if tier == "smart":
        return client().beta.messages.stream(**req, betas=[FALLBACK_BETA], fallbacks="default")
    return client().messages.stream(**req)


def _to_reply(final, tier: str) -> Reply:
    u = final.usage
    text = "".join(b.text for b in final.content if b.type == "text")
    if final.stop_reason == "refusal":
        text = text or REFUSAL_TEXT
    return Reply(
        text=text,
        model=final.model,
        tier=tier,
        stop_reason=final.stop_reason,
        content=list(final.content),
        usage={
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "cache_read_input_tokens": u.cache_read_input_tokens or 0,
            "cache_creation_input_tokens": u.cache_creation_input_tokens or 0,
        },
    )


def stream_chat(
    messages: list[dict],
    system: str | None = None,
    tier: str | None = None,
    result: dict | None = None,
    tools: list[dict] | None = None,
    span_name: str = "llm.answer",
) -> Iterator[str]:
    """Yield the reply text as it arrives.

    When the stream ends, the full Reply (model, stop reason, token usage, raw
    content blocks) is stored in result["reply"], so the UI can stream text and
    the agent can still run any tool calls.
    """
    tier = tier or CONFIG["default_tier"]
    req = _request(tier, system or load_system_prompt(), messages, tools)
    # CHAIN, not LLM: the SDK's child span is the LLM span (with token counts).
    # Ours names the purpose ("llm.answer round 2") and carries the cost.
    with tracing.span(span_name, "CHAIN", {"peopledesk.effort": req["output_config"]["effort"], "peopledesk.tools": len(tools or [])},
                      input=messages[-1]["content"]) as s:
        with _open_stream(tier, req) as stream:
            yield from stream.text_stream
            final = stream.get_final_message()
        reply = _to_reply(final, tier)
        tracing.record_llm(s, reply.model, reply.usage, cost_usd(reply.usage, tier))
        tracing.io(s, output=reply.text or [b.type for b in reply.content])
        s.set_attribute("llm.stop_reason", str(reply.stop_reason))
    if final.stop_reason == "refusal" and not any(b.type == "text" for b in final.content):
        yield reply.text
    if result is not None:
        result["reply"] = reply


def complete(
    messages: list[dict],
    system: str | None = None,
    tier: str | None = None,
    json_schema: dict | None = None,
    effort: str | None = None,
    span_name: str = "llm.complete",
) -> Reply:
    """Non-streaming call, for background jobs such as classifiers, summaries and judges.

    With json_schema, structured outputs guarantee reply.text is JSON matching the
    schema, so callers can json.loads() it without defensive parsing.
    `effort` overrides the tier's default (e.g. a judge needs more reasoning than chat).
    """
    tier = tier or CONFIG["default_tier"]
    req = _request(tier, system or load_system_prompt(), messages)
    if effort:
        req["output_config"]["effort"] = effort
    if json_schema:
        req["output_config"]["format"] = {"type": "json_schema", "schema": json_schema}
    with tracing.span(span_name, "CHAIN", {"peopledesk.effort": req["output_config"]["effort"]},
                      input=messages[-1]["content"]) as s:
        if tier == "smart":
            final = client().beta.messages.create(**req, betas=[FALLBACK_BETA], fallbacks="default")
        else:
            final = client().messages.create(**req)
        reply = _to_reply(final, tier)
        tracing.record_llm(s, reply.model, reply.usage, cost_usd(reply.usage, tier))
        tracing.io(s, output=reply.text)
    return reply
