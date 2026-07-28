"""
app.py
======
The UI layer, and only the UI layer. It reads/writes memory through
MemoryEngine and gets replies through ChatEngine -- it never touches
Mem0 or Groq directly. That split is what lets you swap either piece
out later without rewriting this file.

Per-message flow:
  1. user types something
  2. MemoryEngine.recall() -> facts relevant to *this* message
  3. ChatEngine.reply()   -> answer, using those facts + this session's
                             chat history
  4. MemoryEngine.remember() -> saves whatever's worth keeping long-term
"""

import streamlit as st
from dotenv import load_dotenv

from chat_engine import ChatEngine
from memory_engine import MemoryEngine

load_dotenv()

st.set_page_config(page_title="Memoria", page_icon="🤖", layout="centered")

st.markdown(
    """
    <style>
      .fact-chip {
          display: inline-block; padding: 3px 9px; margin: 2px 3px 2px 0;
          border-radius: 8px; background: #F0F5FF; color: #24408C;
          font-size: 0.82rem; border: 1px solid #D3E0FF;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def _memory() -> MemoryEngine:
    return MemoryEngine()

@st.cache_resource
def _chat() -> ChatEngine:
    return ChatEngine()

memory = _memory()
chat = _chat()

if "history" not in st.session_state:
    st.session_state.history = []  # this browser tab's visible messages only
if "user_id" not in st.session_state:
    st.session_state.user_id = "guest"

##sidebar
with st.sidebar:
    st.markdown(
    """
    <div style="text-align: center; padding-bottom: 20px;">
        <h1 style="margin-bottom: 5px;">🤖 Memoria</h1>
        <p style="color: gray; font-size: 18px;">
            Tell it something once — it remembers, even after a restart..
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

    st.session_state.user_id = st.text_input(
        "Your User-ID",
        value=st.session_state.user_id,
        help="Each ID keeps a completely separate set of memories.",
    )
    active_user = st.session_state.user_id.strip() or "Guest"

    st.divider()
    st.subheader("What's remembered")
    known_facts = memory.everything(active_user)
    if known_facts:
        st.markdown(
            "".join(f'<span class="fact-chip">{f}</span>' for f in known_facts),
            unsafe_allow_html=True,
        )
    else:
        st.caption("Nothing stored yet — say something personal to get started.")

    st.divider()
    left, right = st.columns(2)
    with left:
        if st.button("Clear chat", use_container_width=True, help="Wipes only this window's visible messages."):
            st.session_state.history = []
            st.rerun()
    with right:
        if st.button("Clear mem", use_container_width=True, help="Permanently deletes stored facts for this user ID."):
            memory.forget_everything(active_user)
            st.rerun()

##main
st.title("Memoria")
st.caption("A small chatbot that keeps long-term facts about you across sessions.")

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

user_message = st.chat_input("Say something...")

if user_message:
    st.session_state.history.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.markdown(user_message)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            relevant_facts = memory.recall(user_message, active_user)
            answer = chat.reply(
                user_text=user_message,
                known_facts=relevant_facts,
                history=st.session_state.history[:-1],
            )
            st.markdown(answer)
            if relevant_facts:
                with st.expander("Memory used for this reply"):
                    for f in relevant_facts:
                        st.write(f"• {f}")

    st.session_state.history.append({"role": "assistant", "content": answer})
    memory.remember(user_message, active_user)
    st.rerun()  
