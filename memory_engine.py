"""
memory_engine.py
================
Everything related to long-term memory lives here. The rest of the app
never talks to Mem0 directly -- it only calls the small set of methods
this class exposes (remember / recall / everything / forget_everything).

Why Mem0?
---------
Before writing this, three memory tools were compared for this project:

  * Mem0      - open source, has a simple add/search API, runs fully
                local (no external DB needed) or against a hosted cloud
                store. Good docs, easy to swap the underlying LLM and
                vector store.
  * Zep       - stronger on long-conversation "session summaries" and
                built-in temporal knowledge graphs, but it wants its own
                server (self-hosted or cloud) running alongside the app,
                which is heavier than needed for a single Streamlit demo.
  * OpenMemory - an MCP-flavoured memory server meant to be plugged into
                MCP-compatible clients/agents rather than called as a
                plain Python library, so it doesn't fit a simple
                Streamlit script as cleanly.

Mem0 was picked because it gives fact-level extraction (it decides on
its own that "I turned 26" should replace "I am 25", instead of us
writing that logic by hand) and it persists to a folder on disk with
zero extra services to run.

Local vs cloud
--------------
If a Mem0 Platform key (MEM0_API_KEY, looks like "m0-...") is present in
the environment, the hosted Mem0 service is used. Otherwise everything
runs locally: Chroma as the on-disk vector store, a HuggingFace sentence
embedder, and Groq as the small "reasoning" model Mem0 uses internally
to decide what fact to extract from a message. Nothing here ever needs
an OpenAI key.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from mem0 import Memory, MemoryClient

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("memoria.memory_engine")

# Where the local vector store lives on disk. This folder is what makes
# memory survive an app restart -- Chroma reads it back in on startup.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory_store")
COLLECTION = "memoria_facts"

# Small, fast Groq model used only for fact extraction (not for chat
# replies -- that's a separate, bigger model in chat_engine.py).
EXTRACTION_MODEL = "openai/gpt-oss-20b"

# Must match the embedder's real output size, or Chroma silently breaks
# if the collection was ever created with a different embedder.
EMBEDDER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDER_DIMS = 384


@dataclass
class _LocalSetup:
    """Bundles the pieces needed to build a fully local Mem0 config."""

    groq_key: str

    def as_config(self) -> dict:
        return {
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": COLLECTION,
                    "path": DATA_DIR,
                    "embedding_model_dims": EMBEDDER_DIMS,
                },
            },
            "llm": {
                "provider": "groq",
                "config": {
                    "model": EXTRACTION_MODEL,
                    "api_key": self.groq_key,
                    "temperature": 0.2,
                },
            },
            "embedder": {
                "provider": "huggingface",
                "config": {"model": EMBEDDER_NAME},
            },
        }


# A handful of stock phrases that carry no personal information about
# the user. Messages that reduce to (roughly) just one of these are
# skipped before they ever reach Mem0, so small talk doesn't get sent
# for fact-extraction and doesn't clutter what's remembered.
_LOW_SIGNAL = {
    "hi", "hello", "hey", "yo", "sup",
    "ok", "okay", "k", "cool", "nice", "great", "thanks", "thank you",
    "thanks!", "ty", "yes", "no", "yep", "nope", "sure", "bye", "goodbye",
    "lol", "haha", "good morning", "good night",
}


def is_low_signal(message: str) -> bool:
    """True for greetings/acks that shouldn't be written to memory."""
    stripped = message.strip().lower().strip("!.,? ")
    return len(stripped) == 0 or stripped in _LOW_SIGNAL or len(stripped) < 3


class MemoryEngine:
    """Thin, uniform wrapper around Mem0 local/cloud clients."""

    def __init__(self) -> None:
        cloud_key = os.environ.get("MEM0_API_KEY", "").strip()

        if cloud_key.startswith("m0-"):
            log.info("Using Mem0 Platform (cloud) storage.")
            self._client = MemoryClient(api_key=cloud_key)
            self._cloud = True
            return

        groq_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not groq_key:
            raise RuntimeError(
                "GROQ_API_KEY is missing. Add it to your .env file -- it's "
                "needed both for chat replies and for local fact extraction."
            )

        log.info("Using local Mem0 storage at %s", DATA_DIR)
        self._client = Memory.from_config(_LocalSetup(groq_key).as_config())
        self._cloud = False

    # -- writing -----------------------------------------------------

    def remember(self, message: str, user_id: str) -> bool:
        """Hand a raw user message to Mem0 for fact extraction/merging."""
        if not user_id.strip() or is_low_signal(message):
            return False
        try:
            self._client.add(message, user_id=user_id)
            return True
        except Exception as exc:  # pragma: no cover - defensive logging
            log.error("remember() failed: %s", exc)
            return False

    # -- reading -------------------------------------------------------

    def recall(self, query: str, user_id: str, top_k: int = 5) -> list[str]:
        """Semantic search: facts relevant to `query`, meaning-based."""
        if not query.strip() or not user_id.strip():
            return []
        try:
            if self._cloud:
                result = self._client.search(query, filters={"user_id": user_id}, limit=top_k)
            else:
                result = self._client.search(query, filters={"user_id": user_id}, top_k=top_k)
            return self._unpack(result)
        except Exception as exc:
            log.error("recall() failed: %s", exc)
            return []

    def everything(self, user_id: str) -> list[str]:
        """All facts currently on file for this user (for the sidebar)."""
        if not user_id.strip():
            return []
        try:
            result = self._client.get_all(filters={"user_id": user_id})
            return self._unpack(result)
        except Exception as exc:
            log.error("everything() failed: %s", exc)
            return []

    def forget_everything(self, user_id: str) -> None:
        """Wipe every stored fact for this user -- the 'Clear memory' button."""
        if not user_id.strip():
            return
        try:
            self._client.delete_all(filters={"user_id": user_id})
        except Exception as exc:
            log.error("forget_everything() failed: %s", exc)

    @staticmethod
    def _unpack(result) -> list[str]:
        rows = result.get("results", []) if isinstance(result, dict) else result
        return [row["memory"] for row in rows if isinstance(row, dict) and row.get("memory")]
