"""
memory_handler.py

Wraps the Mem0 library so the rest of the app never talks to Mem0
directly. This keeps all memory-related logic (adding memories,
searching memories) in one place.

Mem0 is configured to use:
- Groq for the LLM that Mem0 uses internally to extract memories
- HuggingFace (local sentence-transformers model) for embeddings,
  so no OpenAI key is required
- Chroma as the local vector store, saved to disk in ./mem0_db
"""

import os
from mem0 import Memory


class MemoryHandlerError(Exception):
    """Raised when Mem0 cannot be initialized or a memory call fails."""
    pass


class MemoryHandler:
    """Handles storing and retrieving conversation memories using Mem0."""

    def __init__(self, groq_api_key: str):
        """
        Set up the Mem0 Memory instance.

        Args:
            groq_api_key: API key used by Mem0's internal LLM calls.

        Raises:
            MemoryHandlerError: if Mem0 fails to initialize.
        """
        try:
            config = {
                "llm": {
                    "provider": "groq",
                    "config": {
                        "model": "llama-3.3-70b-versatile",
                        "api_key": groq_api_key,
                        "temperature": 0.1,
                    },
                },
                "embedder": {
                    "provider": "huggingface",
                    "config": {
                        "model": "sentence-transformers/all-MiniLM-L6-v2",
                    },
                },
                "vector_store": {
                    "provider": "chroma",
                    "config": {
                        "collection_name": "chat_memories",
                        "path": os.path.join(os.getcwd(), "mem0_db"),
                    },
                },
            }
            self.memory = Memory.from_config(config)
        except Exception as error:
            # Wrap any Mem0 setup error in our own exception so app.py
            # can show one consistent, friendly error message.
            raise MemoryHandlerError(f"Failed to initialize Mem0: {error}")

    def add_memory(self, user_message: str, assistant_message: str, user_id: str):
        """
        Store a user/assistant exchange as a memory for this user.

        Any failure here is non-fatal for the chat itself, so callers
        may choose to just show a warning instead of stopping the app.
        """
        try:
            conversation = [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message},
            ]
            self.memory.add(conversation, user_id=user_id)
        except Exception as error:
            raise MemoryHandlerError(f"Failed to save memory: {error}")

    def search_memories(self, query: str, user_id: str, limit: int = 5):
        """
        Search past memories relevant to the current query.

        Returns:
            A list of memory text strings (empty list if none found
            or if the search fails).
        """
        try:
            results = self.memory.search(query, user_id=user_id, limit=limit)

            # Mem0 returns a dict like {"results": [ {"memory": "..."}, ... ]}
            memories = results.get("results", [])
            return [item.get("memory", "") for item in memories if item.get("memory")]
        except Exception:
            # If memory search fails, we simply return no memories
            # instead of breaking the whole chat response.
            return []