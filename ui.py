"""Streamlit chat UI. Run with: streamlit run ui.py"""
import anthropic
import streamlit as st

from app import llm, rag

st.set_page_config(page_title="PeopleDesk", page_icon="💼")
st.title("PeopleDesk")
st.caption("HR assistant for Nimbus Labs")

if "messages" not in st.session_state:
    # [{"role": "user"|"assistant", "content": str, "sources": [...] (assistant only)}]
    st.session_state.messages = []


def show_sources(hits: list[dict]) -> None:
    if hits:
        with st.expander(f"Sources ({len(hits)})"):
            for h in hits:
                st.markdown(f"`{h['score']:.2f}` **{h['source']}** > {h['section']}")


def api_messages(history: list[dict], last_user_turn: str) -> list[dict]:
    """What Claude sees: earlier turns as plain text, and the newest question
    replaced by its RAG-augmented version. Only role/content are sent; the API
    rejects extra keys such as 'sources'."""
    msgs = [{"role": m["role"], "content": m["content"]} for m in history[:-1]]
    return msgs + [{"role": "user", "content": last_user_turn}]


with st.sidebar:
    tier = llm.CONFIG["default_tier"]
    st.markdown(f"**Model:** `{llm.CONFIG['models'][tier]['id']}`")
    st.markdown(f"**Prompt:** `system_{llm.CONFIG['prompt_version']}.md`")
    if last := st.session_state.get("last_reply"):
        st.markdown("**Last reply**")
        st.json({"model": last.model, "stop_reason": last.stop_reason, **last.usage})
    if st.button("New chat"):
        st.session_state.clear()
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        show_sources(msg.get("sources", []))

if prompt := st.chat_input("Ask about leave, benefits, payroll..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        hits = rag.retrieve(prompt)  # milestone 5 passes the user's access tags here
        result: dict = {}
        try:
            st.write_stream(llm.stream_chat(
                api_messages(st.session_state.messages, rag.augment(prompt, hits)),
                result=result,
            ))
        except anthropic.AuthenticationError:
            st.session_state.messages.pop()
            st.error("Invalid or missing ANTHROPIC_API_KEY. Put it in .env (see .env.example).")
            st.stop()
        except anthropic.APIError as e:
            st.session_state.messages.pop()
            st.error(f"Claude API error: {e}")
            st.stop()

    reply = result["reply"]
    st.session_state.messages.append({"role": "assistant", "content": reply.text, "sources": hits})
    st.session_state.last_reply = reply
    st.rerun()  # refresh the sidebar usage panel
