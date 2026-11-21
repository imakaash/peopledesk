"""Streamlit chat UI. Run with: streamlit run ui.py"""
import uuid

import anthropic
import streamlit as st

from app import llm, memory, summarizer
from app.agent import Agent

USERS = ["alice", "bob", "carol"]  # milestone 5 replaces this picker with a real login

st.set_page_config(page_title="PeopleDesk", page_icon="💼")
st.title("PeopleDesk")
st.caption("HR assistant for Nimbus Labs")

def reset_chat() -> None:
    """Start a new session: a new session_id means this chat becomes 'past episodes' next time."""
    for k in ("messages", "last_memory", "last_reply"):
        st.session_state.pop(k, None)


if "messages" not in st.session_state:
    # [{"role": "user"|"assistant", "content": str, "sources": [...] (assistant only)}]
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())


def show_sources(hits: list[dict]) -> None:
    if hits:
        with st.expander(f"Sources ({len(hits)})"):
            for h in hits:
                st.markdown(f"`{h['score']:.2f}` **{h['source']}** > {h['section']}")


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


with st.sidebar:
    user_id = st.selectbox("Acting as", USERS, key="user_id", on_change=reset_chat)
    tier = llm.CONFIG["default_tier"]
    st.markdown(f"**Model:** `{llm.CONFIG['models'][tier]['id']}`")
    st.markdown(f"**Prompt:** `system_{llm.CONFIG['prompt_version']}.md`")
    if wm := st.session_state.get("last_memory"):
        show_memory(wm)
    if last := st.session_state.get("last_reply"):
        with st.expander("Last reply usage"):
            st.json({"model": last.model, "stop_reason": last.stop_reason, **last.usage})
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
    with st.expander(f"Everything remembered about {user_id}"):
        st.write(memory.get_profile(user_id) or "_no profile_")
        for f in memory.all_facts(user_id):
            st.markdown(f"- {f}")
        if st.button("Forget me", type="secondary"):
            memory.forget_user(user_id)
            st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        show_sources(msg.get("sources", []))

if prompt := st.chat_input("Ask about leave, benefits, payroll..."):
    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    agent = Agent(user_id, st.session_state.session_id)
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
    st.session_state.messages.append({"role": "assistant", "content": reply.text, "sources": wm.hits})
    st.session_state.last_reply = reply
    st.session_state.last_memory = wm
    st.rerun()  # refresh the sidebar panels
