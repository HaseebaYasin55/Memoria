# 🧩 Memoria — Persistent Memory Chatbot

A **memory-first AI chatbot** built with **Streamlit**, powered by **Groq** (LLM reasoning) and **Mem0** (long-term memory persistence), that remembers facts about you across sessions — tell it something once, and it recalls it even after the app restarts.

---

# Key Features

- Persistent long-term memory that survives app restarts (stored on disk, not just in-session)
- Fact-level memory extraction — Mem0 decides what's worth remembering from a raw message, no manual tagging needed
- Automatic fact updates — new info overrides old info (e.g. "I am 26 now" replaces "I am 25") instead of duplicating it
- Semantic recall — retrieves memories relevant to *this* message, not just a dump of everything stored
- Per-user memory isolation via a simple user ID, so different people keep separate memory sets
- Sidebar view of everything currently remembered about the active user
- One-click **Clear memory** to permanently wipe a user's stored facts, and **Clear chat** to reset the visible conversation only
- "Memory used for this reply" expander showing exactly which facts informed each answer
- Runs fully local by default — no external memory service required, and no OpenAI key needed anywhere

---

# Project Structure

```
mem-bot/
│
├── app.py
├── chat_engine.py
├── memory_engine.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

# How It Works

Memoria follows this flow on every message:

1. The user types a message in the Streamlit chat input.
2. `MemoryEngine.recall()` runs a semantic search over that user's stored memories in Mem0, pulling back only the facts relevant to the current message.
3. `ChatEngine.reply()` sends the user's message, the recalled facts, and this session's chat history to the Groq LLM, which replies naturally — without ever saying things like "according to my memory."
4. The reply is shown to the user, along with an optional expander showing which stored facts were used.
5. `MemoryEngine.remember()` hands the raw user message to Mem0, which decides on its own whether it contains something worth storing long-term, and merges/updates existing facts if needed.
6. The sidebar refreshes to show the full, current list of everything remembered about that user.

Example of the intended behavior:

- Session 1: *"I like football"* → stored as a fact.
- Session 2 (after a full app restart): *"What sport should I try?"* → the bot recalls the football fact and suggests it.

---

# Memory Tools Considered

Before building this, three memory tools were compared for the project:

| Tool | Notes |
| --- | --- |
| **Mem0** | Open source, simple add/search API, runs fully local (no external DB needed) or against a hosted cloud store. Good docs, easy to swap the underlying LLM and vector store. **Chosen for this project.** |
| **Zep** | Stronger on long-conversation session summaries and built-in temporal knowledge graphs, but requires its own server (self-hosted or cloud) running alongside the app — heavier than needed for a single Streamlit demo. |
| **OpenMemory** | An MCP-flavoured memory server meant to be plugged into MCP-compatible clients/agents rather than called as a plain Python library, so it doesn't fit a simple Streamlit script as cleanly. |

**Mem0** was picked because it gives fact-level extraction — it decides on its own that "I turned 26" should replace "I am 25," instead of that logic being hand-written — and it persists to disk with zero extra services to run.

---

# Memory Strategy

**What gets stored:** Any personal detail the user shares that Mem0's extraction model judges worth keeping — age, preferences, location, interests, and similar facts. Low-signal messages (greetings, acknowledgements like "ok" / "thanks" / "yes") are filtered out before they ever reach Mem0, so small talk doesn't clutter memory.

**How it's recalled:** On every new message, a semantic search is run over the user's stored memories, returning only the facts relevant to that message's meaning — not a full memory dump — which are then injected into the LLM's system prompt.

**How updates are handled:** Mem0's extraction step compares new statements against existing memories and updates/replaces conflicting facts automatically (e.g. an updated age overwrites the old one) rather than accumulating duplicates.

---

# Persistent Storage

- **Local mode (default):** Facts are embedded with a HuggingFace sentence-transformer and stored in a local **Chroma** vector store on disk (`memory_store/`), plus a small Mem0 history log (`~/.mem0/history.db`) tracking add/update/delete events. Both are read back automatically on the next app launch — no facts are lost on restart.
- **Cloud mode (optional):** If a Mem0 Platform API key (`MEM0_API_KEY`, starting with `m0-`) is present in the environment, Memoria uses Mem0's hosted cloud service instead of local storage.

---

# AI Models Used

| Model | Purpose |
| --- | --- |
| **`openai/gpt-oss-120b`** (via Groq) | Drives the chatbot's actual conversational replies. |
| **`openai/gpt-oss-20b`** (via Groq) | Smaller, faster model used internally by Mem0 only for fact extraction — deciding what to remember from a message. |
| **`sentence-transformers/all-MiniLM-L6-v2`** (HuggingFace) | Embeds memories for local semantic search via Chroma. |

---

# Tech Stack

- Python
- Streamlit
- Groq API (LLM reasoning + fact extraction)
- Mem0 (`mem0ai`) — memory persistence layer
- Chroma — local on-disk vector store
- sentence-transformers — local embedding model
- python-dotenv

---

# Installation

## 1. Clone the repository

```
git clone https://github.com/HaseebaYasin55/Mem-Bot.git
```

## 2. Move into the project folder

```
cd Mem-Bot
```

## 3. Open the project in VS Code

```
code .
```

## 4. Install all dependencies

Open the terminal inside VS Code and run:

```
pip install -r requirements.txt
```

## 5. Create a `.env` file

Inside the project folder, create a file named:

```
.env
```

Add your API key:

```
GROQ_API_KEY=your_groq_api_key_here
```

Optionally, to use Mem0's hosted cloud storage instead of local storage, also add:

```
MEM0_API_KEY=your_mem0_platform_key_here
```

Get your free keys here:

- **Groq**: <https://console.groq.com/keys>
- **Mem0 Platform** (optional): <https://app.mem0.ai>

---

# Run the Application

Launch the app using Streamlit:

```
streamlit run app.py
```

This opens at `http://localhost:8501`. Enter a user ID in the sidebar, tell it something personal (e.g. *"I'm 25 and I love football"*), then restart the app and ask a related question (e.g. *"what sport should I try?"*) — it should recall the fact and use it in its answer.

---

# Test Cases

Manual test cases used to verify the memory layer behaves as described:

| # | Scenario | Steps | Expected Result |
| --- | --- | --- | --- |
| 1 | **Basic fact storage** | Send *"I am 25 years old."* | Fact appears as a chip in the "What's remembered" sidebar section. |
| 2 | **Recall across the same session** | After Test 1, ask *"How old am I?"* | Bot answers "25" using the stored fact, shown in the "Memory used for this reply" expander. |
| 3 | **Persistence across a restart** | After Test 1, fully stop and restart `streamlit run app.py` with the same user ID | The "25 years old" fact still appears in the sidebar without being re-entered. |
| 4 | **Fact update / overwrite** | Send *"I am 26 now."* after Test 1 | The stored fact updates to 26 instead of both 25 and 26 existing side by side. |
| 5 | **Preference recall in a new context** | Send *"I like football."*, then later ask *"What sport should I try?"* | Bot suggests football, referencing the earlier preference. |
| 6 | **Low-signal messages are ignored** | Send *"ok"*, *"thanks"*, *"hi"* | Sidebar's "What's remembered" list stays unchanged — no new facts added. |
| 7 | **Per-user isolation** | Store a fact under user ID `alice`, then switch the sidebar user ID to `bob` | `bob` sees an empty "Nothing stored yet" state; Alice's facts are not visible. |
| 8 | **Clear memory button** | Click **Clear memory** after some facts exist | All facts for the active user ID are permanently deleted; sidebar shows "Nothing stored yet." |
| 9 | **Clear chat button** | Click **Clear chat** after a conversation | Visible chat history resets, but stored long-term facts remain (verify via sidebar). |
| 10 | **Missing API key handling** | Remove `GROQ_API_KEY` from `.env` and start the app | App raises a clear `RuntimeError` explaining the key is missing, instead of failing silently or crashing unhelpfully. |

---

# Notes & Limitations

- Requires a valid `GROQ_API_KEY` at all times (used for both chat replies and local fact extraction); the app raises a clear error if it's missing.
- In local mode, memory quality depends on the small extraction model correctly identifying what's worth storing — very ambiguous messages may be skipped or mis-extracted.
- Each user ID keeps a fully separate memory set — there's no cross-user memory sharing by design.
- This-session chat history (`st.session_state.history`) is separate from long-term memory: it resets on page reload or **Clear chat**, while long-term facts persist independently in Mem0/Chroma.

---

# 💡 Future Improvements

Some ideas for future enhancements:

- Manual fact editing/deletion (not just "clear everything")
- Support for additional memory backends (Zep, OpenMemory) as swappable providers
- Memory categorization (preferences, facts, goals) shown separately in the sidebar
- Multi-turn memory summarization for very long-running users
- Authentication instead of a free-text user ID field

---

# 👩‍💻 Author

**Haseeba Yasin**

If you found this project helpful, feel free to ⭐ the repository.