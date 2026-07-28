"""
llm_connector.py

Wraps the Groq API so the rest of the app never talks to Groq
directly. This keeps all LLM-related logic (building messages,
sending the request, handling errors) in one place.
"""

from groq import Groq, APITimeoutError, APIConnectionError, APIStatusError


class LLMConnectorError(Exception):
    """Raised when the LLM call fails for any reason."""
    pass


class LLMConnector:
    """Handles sending chat messages to Groq and returning the reply."""

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        """
        Args:
            api_key: Groq API key.
            model: Groq model name to use for chat completions.
        """
        self.client = Groq(api_key=api_key)
        self.model = model

    def get_response(self, user_message: str, memories: list, chat_history: list) -> str:
        """
        Send the user's message to Groq, including relevant memories
        and recent chat history as context, and return the reply text.

        Args:
            user_message: The latest message typed by the user.
            memories: List of relevant past-memory strings from Mem0.
            chat_history: List of {"role", "content"} dicts for context.

        Raises:
            LLMConnectorError: if the API call fails or the response
                is invalid/empty.
        """
        try:
            system_prompt = self._build_system_prompt(memories)

            # Build the message list: system prompt + recent history + new message
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(chat_history)
            messages.append({"role": "user", "content": user_message})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
                timeout=30,
            )

            reply = response.choices[0].message.content

            if not reply or not reply.strip():
                raise LLMConnectorError("Received an empty response from the model.")

            return reply.strip()

        except APITimeoutError:
            raise LLMConnectorError(
                "The request to Groq timed out. Please check your internet connection and try again."
            )
        except APIConnectionError:
            raise LLMConnectorError(
                "Could not connect to Groq. Please check your internet connection."
            )
        except APIStatusError as error:
            raise LLMConnectorError(f"Groq API returned an error: {error}")
        except LLMConnectorError:
            raise
        except Exception as error:
            raise LLMConnectorError(f"Unexpected error while calling the LLM: {error}")

    @staticmethod
    def _build_system_prompt(memories: list) -> str:
        """Build a system prompt that includes any relevant memories."""
        base_prompt = (
            "You are a helpful assistant. Use the conversation history and "
            "any remembered facts about the user to give personalized, "
            "relevant answers."
        )

        if memories:
            memory_text = "\n".join(f"- {memory}" for memory in memories)
            base_prompt += f"\n\nRelevant facts you remember about the user:\n{memory_text}"

        return base_prompt