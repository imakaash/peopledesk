"""Observability: one OpenTelemetry trace per chat turn, viewed in Arize Phoenix.

    phoenix serve                  # local UI + collector at http://localhost:6006
    streamlit run ui.py            # every turn now shows up as a trace

Two sources of spans end up in one trace:
  - ours: the harness steps (turn, guardrails, router, retrieval, cache, agent loop,
    tools, judge, summarizer), using OpenInference attributes so Phoenix knows each
    span's kind and shows inputs, outputs, documents and token counts.
  - the Anthropic SDK's own `anthropic.messages.create` spans (gen_ai.* attributes,
    request id), nested under our LLM spans automatically because ours is "current".

Context propagation is the one subtle part. A span knows its parent through the
current context, which lives in contextvars. A new thread starts with an empty
context, so work handed to a thread pool would start a separate, parentless trace.
bind() copies the current context into the thread.

The same contextvar also carries a per-turn cost meter, so the turn's total includes
the small calls (classifier, router, verifier, judge, summarizer), not just the answer.
"""
from __future__ import annotations

import contextvars
import json
import socket
from contextlib import contextmanager
from typing import Any, Callable
from urllib.parse import urlparse

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

KIND = "openinference.span.kind"  # LLM | CHAIN | TOOL | RETRIEVER | AGENT | GUARDRAIL
_tracer = trace.get_tracer("peopledesk")  # a proxy: it starts exporting once init() registers a provider
_state = {"initialized": False, "enabled": False, "content": False}
_cost: contextvars.ContextVar[list | None] = contextvars.ContextVar("turn_cost", default=None)


def init(cfg: dict) -> bool:
    """Register Phoenix as the global tracer provider. Must run before the Anthropic
    client is created, so the SDK's spans go to the same provider. Idempotent."""
    if _state["initialized"]:
        return _state["enabled"]
    _state["initialized"] = True
    _state["content"] = cfg.get("content_mode") == "content"
    if not cfg.get("enabled"):
        return False
    url = urlparse(cfg["endpoint"])
    try:  # don't spam export errors on every turn when Phoenix isn't running
        socket.create_connection((url.hostname, url.port or 80), timeout=0.5).close()
    except OSError:
        print(f"[tracing] Phoenix not reachable at {cfg['endpoint']}; tracing is off. Start it with `phoenix serve`.")
        return False
    from phoenix.otel import register  # imported lazily: only needed when tracing is on
    register(project_name=cfg["project"], endpoint=cfg["endpoint"] + "/v1/traces", batch=True, verbose=False)
    _state["enabled"] = True
    return True


def content_enabled() -> bool:
    return _state["content"]


# --- spans -------------------------------------------------------------------------

def _value(v: Any) -> Any:
    if isinstance(v, (str, bool, int, float)):
        return v
    if isinstance(v, (list, tuple)) and all(isinstance(x, (str, bool, int, float)) for x in v):
        return list(v)
    return json.dumps(v, default=str, ensure_ascii=False)


def set_attrs(span, attrs: dict) -> None:
    for k, v in attrs.items():
        if v is not None:
            span.set_attribute(k, _value(v))


def io(span, input: Any = None, output: Any = None) -> None:
    """Record input/output text, only in content mode (it may contain HR data)."""
    if not _state["content"]:
        return
    if input is not None:
        set_attrs(span, {"input.value": input, "input.mime_type": "text/plain" if isinstance(input, str) else "application/json"})
    if output is not None:
        set_attrs(span, {"output.value": output, "output.mime_type": "text/plain" if isinstance(output, str) else "application/json"})


@contextmanager
def span(name: str, kind: str, attrs: dict | None = None, input: Any = None):
    """A step in the trace. It nests under whatever span is current."""
    if not _state["initialized"]:
        # The first span of a turn (the root) usually starts before any LLM call,
        # so set up tracing here too, or the root would come from a no-op tracer.
        from app import llm  # imported here to avoid a circular import (llm imports tracing)
        init(llm.CONFIG["tracing"])
    with _tracer.start_as_current_span(name) as s:
        s.set_attribute(KIND, kind)
        set_attrs(s, attrs or {})
        io(s, input=input)
        yield s


def error(span, message: str) -> None:
    span.set_status(Status(StatusCode.ERROR, message))


def bind(fn: Callable) -> Callable:
    """Run fn in a copy of the *current* context: the trace parent and the cost meter
    follow the work into another thread."""
    ctx = contextvars.copy_context()
    return lambda *a, **k: ctx.run(fn, *a, **k)


# --- LLM spans and cost -----------------------------------------------------------

def record_llm(span, model: str, usage: dict, cost: float) -> None:
    """Cost and usage on our wrapper span, under our own `peopledesk.*` names.

    Token counts deliberately do NOT use the llm.token_count.* names here: the SDK's
    child span (anthropic.messages.create) already reports them, and Phoenix would
    count every call twice if both spans did.
    """
    set_attrs(span, {
        "peopledesk.model": model,
        "peopledesk.cost_usd": cost,
        "peopledesk.usage": usage,  # raw Messages API usage: input excludes cache reads/writes
    })
    if (meter := _cost.get()) is not None:
        meter[0] += cost


@contextmanager
def cost_meter():
    """Collect the cost of every LLM call made in this context (and bound threads)."""
    meter = [0.0]
    token = _cost.set(meter)
    try:
        yield meter
    finally:
        _cost.reset(token)
