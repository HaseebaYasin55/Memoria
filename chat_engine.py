from __future__ import annotations

import os

from groq import Groq

CHAT_MODEL = "openai/gpt-oss-120b"

SYSTEM_PREFACE = (
    "You are a friendly personal assistant with a good memory. Speak "
    "naturally, like you simply know this person -- never say things "
    "like 'according to my memory' or 'I retrieved this fact'. If you "
    "genuinely don't know something about the user, just say so instead "
    "of guessing."
)


def _system_prompt(known_facts: list[str]) -> str:
    """Fold retrieved long-term facts straight into the system message."""
    if not known_facts:
        return SYSTEM_PREFACE
    bullet_list = "\n".join(f"- {fact}" for fact in known_facts)
    return f"{SYSTEM_PREFACE}\n\nThings you know about this person:\n{bullet_list}"


class ChatEngine:
    def __init__(self) -> None:
        self._client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    def reply(self, user_text: str, known_facts: list[str], history: list[dict]) -> str:
        """
        user_text   -- the message just typed
        known_facts -- long-term memories pulled from Mem0 (survive restarts)
        history     -- this session's messages only (short-term context)
        """
        conversation = [{"role": "system", "content": _system_prompt(known_facts)}]
        conversation.extend(history)
        conversation.append({"role": "user", "content": user_text})

        completion = self._client.chat.completions.create(
            model=CHAT_MODEL,
            messages=conversation,
            temperature=0.7,
            max_tokens=600,
        )
        return completion.choices[0].message.content
