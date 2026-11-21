"""Model routing: a cheap Haiku call decides how much model a request needs.

  faq      general policy question that stands alone    -> Haiku, docs only, answer cache
  simple   one lookup/action on the user's records, chat -> Haiku with tools
  complex  multi-step, approvals, comparisons, unclear   -> Sonnet with tools

The point is cost and latency: most HR traffic is FAQs and single lookups, so
paying Sonnet prices only for the hard cases lowers the average cost. The router
is itself an LLM call, but a tiny one (~400 tokens on Haiku).
"""
from __future__ import annotations

import json

from app import llm

CFG = llm.CONFIG["routing"]

SYSTEM = """You route messages for PeopleDesk, the HR assistant of Nimbus Labs. Pick the cheapest route that can handle the message well:

- "faq": a general HR policy question that can be answered from company policy documents alone. It is not about the user's own records (balances, requests, payslips, tasks), and it doesn't depend on earlier messages. Example: "How many sick days do employees get?"
- "simple": needs the user's own records or the conversation so far, with at most one straightforward lookup or action. Also greetings, thanks and short follow-ups. Examples: "How many leave days do I have left?", "And for sick leave?", "Thanks!"
- "complex": several steps or tools, approving or changing things that affect others, comparisons or calculations across records, or anything ambiguous. Examples: "Check my balance and book 10-12 Dec if I have enough", "Approve all of Dave's pending requests"."""

SCHEMA = {
    "type": "object",
    "properties": {
        "route": {"type": "string", "enum": ["faq", "simple", "complex"]},
        "reason": {"type": "string"},
    },
    "required": ["route", "reason"],
    "additionalProperties": False,
}


def route(question: str, previous_user_message: str | None = None) -> dict:
    """Return {"route", "reason", "tier"}. `previous_user_message` lets the router
    tell a standalone question from a follow-up ("and for sick leave?")."""
    if not CFG["enabled"]:
        return {"route": "simple", "reason": "routing disabled", "tier": llm.CONFIG["default_tier"]}
    prev = f"<previous_user_message>\n{previous_user_message}\n</previous_user_message>\n" if previous_user_message else ""
    reply = llm.complete(
        [{"role": "user", "content": f"{prev}<message_to_route>\n{question}\n</message_to_route>"}],
        system=SYSTEM, tier="cheap", json_schema=SCHEMA,
    )
    r = json.loads(reply.text)
    return {**r, "tier": CFG["tiers"][r["route"]]}
