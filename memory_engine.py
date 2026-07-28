from __future__ import annotations
import logging
import os
from dataclasses import dataclass
from typing import Any, Iterable
from mem0 import Memory, MemoryClient

log = logging.getLogger("memoria.memory_engine")
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

# Where the local vector store lives on disk. This folder is what makes
# memory survive an app restart -- Chroma reads it back in on startup.
DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "memory_store",
)
COLLECTION = "memoria_facts"
EXTRACTION_MODEL = "openai/gpt-oss-20b"    #for fact extraction

##embeds new and existing facts with the same model.
EMBEDDER_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class _LocalSetup:
    """Builds the fully-local Mem0 config (Chroma + HuggingFace + Groq)."""

    groq_key: str

    def as_config(self) -> dict:
        return {
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": COLLECTION,
                    "path": DATA_DIR,
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

_LOW_SIGNAL = {
    "hi", "hello", "hey", "yo", "sup",
    "ok", "okay", "k", "cool", "nice", "great", "thanks", "thank you",
    "thanks!", "ty", "yes", "no", "yep", "nope", "sure", "bye", "goodbye",
    "lol", "haha", "good morning", "good night", "please",
}


def is_low_signal(message: str) -> bool:
    """Skip greetings/acks so the memory store stays clean."""
    stripped = message.strip().lower().strip("!.,? ")
    return len(stripped) == 0 or stripped in _LOW_SIGNAL or len(stripped) < 3


class MemoryEngine:
    def __init__(self) -> None:
        cloud = self._cloud_key()
        if cloud:
            # Mem0 Platform (cloud) -- your facts live at app.mem0.ai
            # and survive the local folder being deleted.
            self._backend = "cloud"
            self._client = MemoryClient(api_key=cloud)
            log.info("MemoryEngine: backend = Mem0 Platform (cloud)")
            return

        groq_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not groq_key:
            raise RuntimeError(
                "No MEM0_API_KEY (cloud) and no GROQ_API_KEY (needed for "
                "local fact extraction). Add at least one to your .env."
            )
        os.makedirs(DATA_DIR, exist_ok=True)
        self._backend = "local"
        self._client = Memory.from_config(_LocalSetup(groq_key).as_config())
        log.info("MemoryEngine: backend = local Chroma @ %s", DATA_DIR)

    @staticmethod
    def _cloud_key() -> str | None:
        key = os.environ.get("MEM0_API_KEY", "").strip()
        return key if key.startswith("m0-") else None

    @property
    def backend(self) -> str:
        return self._backend

    def remember(self, message: str, user_id: str) -> bool:
        """Hand a raw user message to Mem0 for fact extraction/merging.

        Returns True if Mem0 accepted the message, False on filter or error.
        """
        if not (user_id and user_id.strip()) or is_low_signal(message):
            return False
        try:
            self._client.add(
                messages=[{"role": "user", "content": message}],
                user_id=user_id,
            )
            log.info("remember(): saved fact for user=%s (msg=%r)",
                     user_id, message[:60])
            return True
        except Exception:
            log.exception("remember() failed for user=%s", user_id)
            return False

    def recall(self, query: str, user_id: str, top_k: int = 5) -> list[str]:
        """Semantic search: facts relevant to `query`, meaning-based."""
        if not (query and query.strip()) or not (user_id and user_id.strip()):
            return []
        try:
            # FIX: top-level user_id (the way mem0ai>=1.0.0 wants it).
            result = self._client.search(
                query=query,
                user_id=user_id,
                limit=top_k,
            )
            return _unpack(result)
        except TypeError:
            try:
                result = self._client.search(
                    query=query,
                    filters={"user_id": user_id},
                    limit=top_k,
                )
                return _unpack(result)
            except Exception:
                log.exception("recall() failed for user=%s", user_id)
                return []
        except Exception:
            log.exception("recall() failed for user=%s", user_id)
            return []

    def everything(self, user_id: str) -> list[str]:
        """All facts for this user (powers the sidebar's 'What's remembered')."""
        if not (user_id and user_id.strip()):
            return []
        try:
            result = self._client.get_all(user_id=user_id)
            return _unpack(result)
        except TypeError:
            try:
                result = self._client.get_all(filters={"user_id": user_id})
                return _unpack(result)
            except Exception:
                log.exception("everything() failed for user=%s", user_id)
                return []
        except Exception:
            log.exception("everything() failed for user=%s", user_id)
            return []

    def forget_everything(self, user_id: str) -> None:
        """Wipe every fact for this user -- the 'Clear memory' button."""
        if not (user_id and user_id.strip()):
            return
        try:
            self._client.delete_all(user_id=user_id)
        except TypeError:
            try:
                self._client.delete_all(filters={"user_id": user_id})
            except Exception:
                log.exception("forget_everything() failed for user=%s", user_id)
        except Exception:
            log.exception("forget_everything() failed for user=%s", user_id)

    def health_check(self, user_id: str = "__healthcheck__") -> dict:
        """Tiny round-trip probe: writes a marker fact, reads it back.

        Returns a dict you can paste into a bug report:
          {
            "backend":   "cloud" | "local",
            "add_ok":    bool,
            "recall_ok": bool,
            "count":     int,
            "error":     str | None,
          }
        """
        report = {
            "backend": self._backend,
            "user_id": user_id,
            "add_ok": False,
            "recall_ok": False,
            "count": 0,
            "error": None,
        }
        marker = "mem-bot healthcheck probe"
        try:
            report["add_ok"] = self.remember(marker, user_id)
            hits = self.recall(marker, user_id, top_k=3)
            report["count"] = len(hits)
            report["recall_ok"] = bool(hits)
        except Exception as exc:  # pragma: no cover - diagnostic
            report["error"] = str(exc)
        return report


def _unpack(result: Any) -> list[str]:
    """Normalize Mem0 cloud/local response shapes into a flat list[str]."""
    if result is None:
        return []
    rows: Iterable[Any]
    if isinstance(result, dict):
        rows = result.get("results", []) or []
    elif isinstance(result, list):
        rows = result
    else:
        return []

    out: list[str] = []
    for row in rows:
        if isinstance(row, dict) and row.get("memory"):
            out.append(row["memory"])
        elif isinstance(row, str):
            out.append(row)
    return out







