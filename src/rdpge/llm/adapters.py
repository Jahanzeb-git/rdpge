# adapters.py — LLM Provider Adapters
#
# Concrete implementations of LLMProvider for:
#   - OpenAI (GPT-4, GPT-4o, etc.)
#   - Anthropic (Claude Sonnet, Opus, Haiku)
#
# Uses httpx for async HTTP calls. No SDK dependencies.

import httpx
from ..core.models import Message
from .base import LLMConfig


class OpenAIAdapter:
    """
    LLM adapter for OpenAI-compatible APIs.
    Works with: OpenAI, Together AI, Groq, OpenRouter, local vLLM, etc.

    Usage:
        llm = OpenAIAdapter(LLMConfig(
            model="gpt-4o",
            api_key="sk-...",
        ))
        response = await llm.generate(messages)
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self.base_url = config.base_url or "https://api.openai.com/v1"

    async def generate(self, messages: list[Message]) -> str:
        # Translate RDPGE messages → OpenAI format (1:1 mapping)
        openai_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.config.model,
                    "messages": openai_messages,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                    **self.config.extra,
                },
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"OpenAI API error {response.status_code}: {response.text}"
                )

            data = response.json()
            return data["choices"][0]["message"]["content"]


class AnthropicAdapter:
    """
    LLM adapter for Anthropic's Claude API.
    The ONLY difference: system prompt is extracted to a separate field.

    Usage:
        llm = AnthropicAdapter(LLMConfig(
            model="claude-sonnet-4-20250514",
            api_key="sk-ant-...",
        ))
        response = await llm.generate(messages)
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self.base_url = config.base_url or "https://api.anthropic.com/v1"

    async def generate(self, messages: list[Message]) -> str:
        # Anthropic needs system prompt SEPARATED from messages
        system_content = ""
        anthropic_messages = []

        for msg in messages:
            if msg.role == "system":
                system_content = msg.content
            else:
                anthropic_messages.append(
                    {"role": msg.role, "content": msg.content}
                )

        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            response = await client.post(
                f"{self.base_url}/messages",
                headers={
                    "x-api-key": self.config.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.config.model,
                    "system": system_content,
                    "messages": anthropic_messages,
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature,
                    **self.config.extra,
                },
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"Anthropic API error {response.status_code}: {response.text}"
                )

            data = response.json()
            return data["content"][0]["text"]
