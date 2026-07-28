"""
session_handler.py

Handles Streamlit session state.
This keeps track of things like chat history and the current user id
while the app is running, so app.py does not have to deal with these
details directly.
"""

import streamlit as st


class SessionHandler:
    """Manages Streamlit session state for the chatbot."""

    @staticmethod
    def init_session():
        """
        Create the session state variables if they do not exist yet.
        This runs once per browser session (Streamlit calls the script
        again on every interaction, so we guard with "if not in state").
        """
        if "chat_history" not in st.session_state:
            # Each item will look like: {"role": "user"/"assistant", "content": "..."}
            st.session_state.chat_history = []

        if "user_id" not in st.session_state:
            # Default user id used to store/retrieve memories in Mem0
            st.session_state.user_id = "default_user"

    @staticmethod
    def get_chat_history():
        """Return the list of chat messages stored in this session."""
        return st.session_state.chat_history

    @staticmethod
    def add_message(role: str, content: str):
        """Add a single message to the chat history."""
        st.session_state.chat_history.append({"role": role, "content": content})

    @staticmethod
    def clear_chat_history():
        """Empty the chat history (used by the 'Clear Chat' button)."""
        st.session_state.chat_history = []

    @staticmethod
    def get_user_id() -> str:
        """Return the current user id."""
        return st.session_state.user_id

    @staticmethod
    def set_user_id(user_id: str):
        """Update the current user id."""
        st.session_state.user_id = user_id