"""Streamlit chat UI. Run with: streamlit run ui.py"""
import anthropic
import streamlit as st

from app import llm

st.set_page_config(page_title="PeopleDesk", page_icon="💼")
st.title("PeopleDesk")
st.caption("HR assistant for Nimbus Labs")

if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "user"|"assistant", "content": str}]

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

if prompt := st.chat_input("Ask about leave, benefits, payroll..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        result: dict = {}
        try:
            st.write_stream(llm.stream_chat(st.session_state.messages, result=result))
        except anthropic.AuthenticationError:
            st.session_state.messages.pop()
            st.error("Invalid or missing ANTHROPIC_API_KEY. Put it in .env (see .env.example).")
            st.stop()
        except anthropic.APIError as e:
            st.session_state.messages.pop()
            st.error(f"Claude API error: {e}")
            st.stop()

    reply = result["reply"]
    st.session_state.messages.append({"role": "assistant", "content": reply.text})
    st.session_state.last_reply = reply
    st.rerun()  # refresh the sidebar usage panel
