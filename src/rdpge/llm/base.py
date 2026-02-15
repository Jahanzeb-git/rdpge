# base.py — LLM Provider Interface
#
# Defines the protocol (interface) that any LLM provider must implement.
# RDPGE is provider-agnostic: OpenAI, Anthropic, Together, Ollama — all work.

from typing import Protocol, runtime_checkable
from dataclasses import dataclass, field
from ..core.models import Message


@runtime_checkable
class LLMProvider(Protocol):
    """
    Protocol that any LLM adapter must implement.

    The entire contract is ONE method:
        messages in → text out

    Usage:
        class MyProvider:
            async def generate(self, messages: list[Message]) -> str:
                # Call your LLM API
                return response_text

        agent = Agent(llm=MyProvider())
    """
    async def generate(self, messages: list[Message]) -> str:
        """
        Send messages to the LLM and return the response text.

        Args:
            messages: list of Message(role, content) objects

        Returns:
            The LLM's response as a plain string
        """
        ...


@dataclass
class LLMConfig:
    """Configuration for LLM adapters."""
    model: str
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.2
    max_tokens: int = 4096
    timeout: float = 60.0
    extra: dict = field(default_factory=dict)
