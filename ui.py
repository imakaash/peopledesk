"""Streamlit chat UI. Run with: streamlit run ui.py"""
import uuid

import anthropic
import streamlit as st

from app import auth, llm, memory, summarizer
from app.agent import Agent

st.set_page_config(page_title="PeopleDesk", page_icon="💼")
st.title("PeopleDesk")
st.caption("HR assistant for Nimbus Labs")


def reset_chat() -> None:
    """Start a new session: a new session_id means this chat becomes 'past episodes' next time."""
    for k in ("messages", "last_memory", "last_reply", "last_meta"):
        st.session_state.pop(k, None)


def logout(message: str | None = None) -> None:
    st.session_state.clear()
    if message:
        st.session_state.flash = message


# --- login (authN) ------------------------------------------------------------

def login_screen() -> None:
    if msg := st.session_state.pop("flash", None):
        st.warning(msg)
    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")
    st.caption("Demo users: alice, dave (employee) · bob, frank (manager) · carol (hr_admin). Password: nimbus123")
    if submitted:
        try:
            st.session_state.token = auth.login(username, password)
            st.rerun()
        except auth.AuthError as e:
            st.error(str(e))
        except RuntimeError as e:  # JWT_SECRET missing or too weak
            st.error(str(e))


if "token" not in st.session_state:
    login_screen()
    st.stop()

try:
    # Checked again on every rerun, so an expired session logs out.
    claims = auth.verify_token(st.session_state.token)
except auth.AuthError as e:
    logout(str(e))
    st.rerun()
user_id = claims["sub"]

if "messages" not in st.session_state:
    # [{"role": "user"|"assistant", "content": str, "sources": [...], "tools": [...]}]
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())


# --- rendering helpers ----------------------------------------------------------

def show_sources(hits: list[dict]) -> None:
    if hits:
        with st.expander(f"Sources ({len(hits)})"):
            for h in hits:
                st.markdown(f"`{h['score']:.2f}` **{h['source']}** > {h['section']}")


def show_tools(calls: list[dict]) -> None:
    if calls:
        with st.expander(f"Tools used ({len(calls)})"):
            for c in calls:
                icon = "❌" if c["is_error"] else "✅"
                st.markdown(f"{icon} **{c['name']}** `{c['input']}`")
                st.code(c["output"][:1500], language="json")


def show_meta(m: dict | None) -> None:
    """One line per answer: how it was produced and what it cost."""
    if not m:
        return
    if m.get("cache"):
        c = m["cache"]
        st.caption(f"⚡ answer cache hit · matched “{c['cached_question']}” (similarity {c['score']}) · "
                   f"{m['timings']['total']:.2f}s · $0 for the answer")
    else:
        st.caption(f"route `{m['route']}` · `{m['model']}` · {m['timings'].get('total', 0):.2f}s · ${m['cost_usd']:.5f}")


def show_guard(g: dict | None) -> None:
    if not g:
        return
    gi, go = g["in"], g.get("out")
    flags = []
    if gi.pii or (go and go.pii):
        flags.append("PII redacted")
    if gi.blocked:
        flags.append(f"blocked: {gi.verdict}")
    if go and go.grounded is False:
        flags.append("ungrounded")
    with st.expander("Guardrails" + (f" ⚠️ {', '.join(flags)}" if flags else " ✅")):
        st.markdown(f"**Input:** `{gi.verdict}`: {gi.reason or 'n/a'}")
        if gi.pii:
            st.markdown(f"**PII masked in your message:** {', '.join(gi.pii)}")
        if go:
            st.markdown(f"**Grounded:** {'not checked' if go.grounded is None else go.grounded}")
            for c in go.unsupported_claims:
                st.markdown(f"- unsupported: {c}")
            if go.pii:
                st.markdown(f"**PII masked in the answer:** {', '.join(go.pii)}")


def show_memory(wm) -> None:
    """What the harness put into working memory for the last turn."""
    with st.expander("Working memory (last turn)"):
        st.markdown("**Profile**")
        st.write(wm.profile or "_empty_")
        st.markdown(f"**Facts retrieved:** {len(wm.facts)}")
        for f in wm.facts:
            st.markdown(f"- {f}")
        st.markdown(f"**Past episodes:** {len(wm.episodes)}")
        st.markdown(f"**Skills:** {', '.join(s['name'] for s in wm.skills) or '_none_'}")
        st.markdown(f"**Docs:** {len(wm.hits)}")


# --- sidebar ----------------------------------------------------------------------

with st.sidebar:
    st.markdown(f"Signed in as **{claims['name']}**  \n`{user_id}` · role `{claims['role']}`")
    if st.button("Log out"):
        logout()
        st.rerun()
    st.divider()
    tiers = llm.CONFIG["routing"]["tiers"]
    st.markdown("**Routing:** " + " · ".join(f"{r} → `{llm.CONFIG['models'][t]['id']}`" for r, t in tiers.items()))
    st.markdown(f"**Prompt:** `system_{llm.CONFIG['prompt_version']}.md`")
    if wm := st.session_state.get("last_memory"):
        show_memory(wm)
    if last := st.session_state.get("last_reply"):
        with st.expander("Last reply: usage and timings"):
            m = st.session_state.get("last_meta", {})
            st.markdown(f"**Route:** `{m.get('route')}`: {m.get('route_reason', '')}")
            st.json({"model": last.model, "stop_reason": last.stop_reason, **last.usage,
                     "cost_usd (main model)": m.get("cost_usd"), "timings_s": m.get("timings")})
    c1, c2 = st.columns(2)
    if c1.button("New chat"):
        reset_chat()
        st.rerun()
    if c2.button("Consolidate now", help="Run the summarizer for this user right away"):
        with st.spinner("Consolidating..."):
            res = summarizer.consolidate(user_id)
        if res:
            st.success(f"Profile updated, {len(res['new_facts'])} new facts")
        else:
            st.info("Nothing new to consolidate")
    with st.expander("Everything remembered about you"):
        st.write(memory.get_profile(user_id) or "_no profile_")
        for f in memory.all_facts(user_id):
            st.markdown(f"- {f}")
        if st.button("Forget me", type="secondary"):
            memory.forget_user(user_id)
            st.rerun()


# --- chat ----------------------------------------------------------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        show_meta(msg.get("meta"))
        show_tools(msg.get("tools", []))
        show_sources(msg.get("sources", []))
        show_guard(msg.get("guard"))

if prompt := st.chat_input("Ask about leave, benefits, payroll..."):
    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    agent = Agent(st.session_state.token, st.session_state.session_id)
    with st.chat_message("assistant"):
        result: dict = {}
        try:
            st.write_stream(agent.stream(prompt, history, result))
        except anthropic.AuthenticationError:
            st.session_state.messages.pop()
            st.error("Invalid or missing ANTHROPIC_API_KEY. Put it in .env (see .env.example).")
            st.stop()
        except anthropic.APIError as e:
            st.session_state.messages.pop()
            st.error(f"Claude API error: {e}")
            st.stop()

    reply, wm = result["reply"], result["memory"]
    # Keep the PII-redacted question in history, so raw card numbers etc. never get re-sent.
    st.session_state.messages[-1]["content"] = result["question"]
    st.session_state.messages.append({
        "role": "assistant", "content": reply.text, "sources": wm.hits, "tools": result["tool_calls"],
        "guard": {"in": result["guard_in"], "out": result["guard_out"]},
        "meta": result["meta"],
    })
    st.session_state.last_reply = reply
    st.session_state.last_memory = wm
    st.session_state.last_meta = result["meta"]
    st.rerun()  # refresh the sidebar panels
