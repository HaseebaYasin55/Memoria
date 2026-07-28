"""
app.py

Streamlit chatbot with long-term memory powered by Mem0.

This file only handles:
- Page/UI setup
- Button and input handling
- Session state (via SessionHandler)
- Calling MemoryHandler and LLMConnector

All Mem0 logic lives in memory_handler.py and all LLM logic lives in
llm_connector.py, so this file stays simple and readable.
"""

import os
import streamlit as st
from dotenv import load_dotenv

from session_handler import SessionHandler
from memory_handler import MemoryHandler, MemoryHandlerError
from llm_connector import LLMConnector, LLMConnectorError

# Load GROQ_API_KEY from a .env file, if present
load_dotenv()


def load_api_key():
    """
    Read the Groq API key from the environment.

    Returns None if it is missing, so the caller can show a friendly
    error instead of crashing.
    """
    return os.getenv("GROQ_API_KEY")


@st.cache_resource(show_spinner=False)
def get_memory_handler(api_key: str):
    """
    Create (and cache) a single MemoryHandler for the app's lifetime.

    Using st.cache_resource means Mem0 is only initialized once,
    instead of on every Streamlit rerun.
    """
    return MemoryHandler(groq_api_key=api_key)


@st.cache_resource(show_spinner=False)
def get_llm_connector(api_key: str):
    """Create (and cache) a single LLMConnector for the app's lifetime."""
    return LLMConnector(api_key=api_key)


def main():
    st.set_page_config(page_title="Memory Chatbot", page_icon="🧠")
    st.title("🧠 Memory Chatbot")
    st.caption("A simple chatbot that remembers facts about you using Mem0.")

    # --- Session state setup ---
    SessionHandler.init_session()

    # --- API key check ---
    api_key = load_api_key()
    if not api_key:
        st.error(
            "GROQ_API_KEY is missing. Please add it to your .env file "
            "and restart the app."
        )
        st.stop()

    # --- Initialize Mem0 and the LLM connector ---
    try:
        memory_handler = get_memory_handler(api_key)
    except MemoryHandlerError as error:
        st.error(f"Could not start the memory system: {error}")
        st.stop()

    llm_connector = get_llm_connector(api_key)

    # --- Sidebar: user id and controls ---
    with st.sidebar:
        st.subheader("Settings")
        user_id = st.text_input("User ID", value=SessionHandler.get_user_id())
        SessionHandler.set_user_id(user_id)

        if st.button("Clear Chat"):
            SessionHandler.clear_chat_history()
            st.rerun()

    # --- Display existing chat history ---
    for message in SessionHandler.get_chat_history():
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # --- Chat input ---
    user_input = st.chat_input("Type your message here...")

    if user_input is not None:
        # Handle empty input (e.g. spaces only)
        if not user_input.strip():
            st.warning("Please enter a message before sending.")
            st.stop()

        # Show the user's message immediately
        SessionHandler.add_message("user", user_input)
        with st.chat_message("user"):
            st.write(user_input)

        # --- Search relevant memories ---
        try:
            memories = memory_handler.search_memories(
                query=user_input, user_id=user_id
            )
        except MemoryHandlerError as error:
            st.warning(f"Could not retrieve memories: {error}")
            memories = []

        # --- Get the assistant's reply ---
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    reply = llm_connector.get_response(
                        user_message=user_input,
                        memories=memories,
                        chat_history=SessionHandler.get_chat_history()[:-1],
                    )
                    st.write(reply)
                except LLMConnectorError as error:
                    reply = None
                    st.error(f"Sorry, something went wrong: {error}")

        # --- Save the exchange to chat history and memory ---
        if reply:
            SessionHandler.add_message("assistant", reply)
            try:
                memory_handler.add_memory(
                    user_message=user_input,
                    assistant_message=reply,
                    user_id=user_id,
                )
            except MemoryHandlerError as error:
                # Saving memory failing should not break the chat itself
                st.warning(f"Could not save this exchange to memory: {error}")


if __name__ == "__main__":
    main()