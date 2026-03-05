"""Unified LLM client for agent-based JobScout."""

import os
import json
from typing import Optional, TypeVar, Type
from dataclasses import dataclass
from functools import lru_cache


T = TypeVar('T')


@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    model: str
    provider: str

    def parse_json(self) -> dict:
        """Parse response as JSON."""
        try:
            return json.loads(self.content)
        except json.JSONDecodeError as e:
            # Try to extract JSON from markdown code blocks
            content = self.content.strip()
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                if end != -1:
                    return json.loads(content[start:end].strip())
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                if end != -1:
                    return json.loads(content[start:end].strip())
            raise ValueError(f"Failed to parse JSON from LLM response: {e}")

    def parse_object(self, cls: Type[T]) -> T:
        """Parse response as a dataclass object."""
        data = self.parse_json()
        # Convert dict keys to match field names if needed
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


class LLMClient:
    """Unified client for OpenAI, Anthropic, and DeepSeek."""

    # Default models per provider
    DEFAULT_MODELS = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-5-haiku-20241022",
        "deepseek": "deepseek-chat",
    }

    # Environment variable names for API keys
    API_KEY_ENV_VARS = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
    }

    def __init__(
        self,
        provider: str = "openai",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize LLM client.

        Args:
            provider: "openai", "anthropic", or "deepseek"
            api_key: API key (defaults to env var)
            model: Model name (defaults to provider default)
        """
        self.provider = provider.lower()
        self.api_key = api_key or self._get_default_api_key()
        self.model = model or self._get_default_model()

        if not self.api_key:
            raise ValueError(f"API key not found for provider: {provider}")

        self._client = self._create_client()

    def _get_default_api_key(self) -> Optional[str]:
        """Get API key from environment."""
        env_var = self.API_KEY_ENV_VARS.get(self.provider, "")
        return os.getenv(env_var)

    def _get_default_model(self) -> str:
        """Get default model for provider."""
        return self.DEFAULT_MODELS.get(self.provider, "gpt-4o-mini")

    def _create_client(self):
        """Create the appropriate client instance."""
        if self.provider == "openai":
            from openai import OpenAI
            return OpenAI(api_key=self.api_key)
        elif self.provider == "anthropic":
            from anthropic import Anthropic
            return Anthropic(api_key=self.api_key)
        elif self.provider == "deepseek":
            from openai import OpenAI
            return OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com"
            )
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        response_format: Optional[dict] = None,
    ) -> LLMResponse:
        """
        Send chat completion request.

        Args:
            messages: List of {role, content} dicts
            temperature: Sampling temperature (0-1)
            response_format: For OpenAI, {"type": "json_object"}

        Returns:
            LLMResponse with content and metadata
        """
        if self.provider == "openai":
            return self._openai_chat(messages, temperature, response_format)
        elif self.provider == "anthropic":
            return self._anthropic_chat(messages, temperature)
        elif self.provider == "deepseek":
            return self._deepseek_chat(messages, temperature)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _openai_chat(
        self,
        messages: list[dict],
        temperature: float,
        response_format: Optional[dict],
    ) -> LLMResponse:
        """OpenAI chat completion."""
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format:
            kwargs["response_format"] = response_format

        response = self._client.chat.completions.create(**kwargs)
        return LLMResponse(
            content=response.choices[0].message.content or "",
            model=response.model,
            provider="openai",
        )

    def _anthropic_chat(
        self,
        messages: list[dict],
        temperature: float,
    ) -> LLMResponse:
        """Anthropic chat completion."""
        # Anthropic requires system message to be separate
        system_msg = ""
        user_msgs = []

        for msg in messages:
            if msg["role"] == "system":
                system_msg += msg["content"] + "\n\n"
            else:
                user_msgs.append(msg)

        response = self._client.messages.create(
            model=self.model,
            system=system_msg.strip() or None,
            messages=user_msgs,
            temperature=temperature,
            max_tokens=4096,
        )

        return LLMResponse(
            content=response.content[0].text,
            model=response.model,
            provider="anthropic",
        )

    def _deepseek_chat(
        self,
        messages: list[dict],
        temperature: float,
    ) -> LLMResponse:
        """DeepSeek chat completion (OpenAI-compatible)."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return LLMResponse(
            content=response.choices[0].message.content or "",
            model=response.model,
            provider="deepseek",
        )

    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> dict:
        """
        Generate JSON response.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt

        Returns:
            Parsed JSON dict
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # For OpenAI, use JSON mode
        if self.provider == "openai":
            response = self.chat(messages, response_format={"type": "json_object"})
        else:
            # For non-OpenAI, add instruction to prompt
            messages[-1]["content"] += "\n\nRespond ONLY with valid JSON, no markdown."
            response = self.chat(messages)

        return response.parse_json()

    def generate_structured(
        self,
        prompt: str,
        structure: Type[T],
        system_prompt: Optional[str] = None,
    ) -> T:
        """
        Generate structured response matching a dataclass.

        Args:
            prompt: User prompt
            structure: Dataclass type to parse into
            system_prompt: Optional system prompt

        Returns:
            Instance of the dataclass
        """
        json_response = self.generate_json(prompt, system_prompt)
        return structure(**{k: v for k, v in json_response.items() if hasattr(structure, k)})


@lru_cache(maxsize=1)
def get_default_client() -> Optional[LLMClient]:
    """Get a default LLM client based on available API keys."""
    for provider in ["openai", "anthropic", "deepseek"]:
        env_var = LLMClient.API_KEY_ENV_VARS.get(provider)
        if os.getenv(env_var):
            return LLMClient(provider=provider)
    return None
