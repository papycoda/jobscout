"""Multi-provider LLM system supporting OpenAI, DeepSeek, and others."""

import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum


logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    ANTHROPIC = "anthropic"  # For future support


@dataclass
class LLMModel:
    """LLM model definition."""
    id: str
    name: str
    provider: LLMProvider
    description: str
    max_tokens: int
    supports_reasoning: bool = False
    cost_per_1m_input: float = 0.0
    cost_per_1m_output: float = 0.0


# Available models catalog
AVAILABLE_MODELS: List[LLMModel] = [
    # OpenAI GPT-5 Models (Latest)
    LLMModel(
        id="gpt-5.2",
        name="GPT-5.2",
        provider=LLMProvider.OPENAI,
        description="OpenAI's best general-purpose model - complex reasoning, broad knowledge, multi-step tasks",
        max_tokens=100000,
        supports_reasoning=True,
        cost_per_1m_input=2.50,
        cost_per_1m_output=10.00
    ),
    LLMModel(
        id="gpt-5.2-pro",
        name="GPT-5.2 Pro",
        provider=LLMProvider.OPENAI,
        description="OpenAI's most intelligent model - toughest problems requiring deep thinking",
        max_tokens=100000,
        supports_reasoning=True,
        cost_per_1m_input=5.00,
        cost_per_1m_output=20.00
    ),
    LLMModel(
        id="gpt-5-mini",
        name="GPT-5 Mini",
        provider=LLMProvider.OPENAI,
        description="Cost-optimized reasoning and chat - balances speed, cost, and capability",
        max_tokens=128000,
        supports_reasoning=True,
        cost_per_1m_input=0.15,
        cost_per_1m_output=0.60
    ),
    LLMModel(
        id="gpt-5-nano",
        name="GPT-5 Nano",
        provider=LLMProvider.OPENAI,
        description="High-throughput tasks - simple instruction-following or classification",
        max_tokens=128000,
        supports_reasoning=False,
        cost_per_1m_input=0.10,
        cost_per_1m_output=0.40
    ),
    LLMModel(
        id="gpt-5.1-codex-max",
        name="GPT-5.1-Codex-Max",
        provider=LLMProvider.OPENAI,
        description="Best for coding tasks - interactive coding products, full spectrum coding",
        max_tokens=128000,
        supports_reasoning=True,
        cost_per_1m_input=1.50,
        cost_per_1m_output=6.00
    ),

    # Older OpenAI Models (for backward compatibility)
    LLMModel(
        id="gpt-4o",
        name="GPT-4o",
        provider=LLMProvider.OPENAI,
        description="Legacy model - use GPT-5 models instead (deprecated)",
        max_tokens=128000,
        supports_reasoning=False,
        cost_per_1m_input=2.50,
        cost_per_1m_output=10.00
    ),
    LLMModel(
        id="gpt-4o-mini",
        name="GPT-4o Mini",
        provider=LLMProvider.OPENAI,
        description="Legacy cost-effective model - use GPT-5 Mini instead (deprecated)",
        max_tokens=128000,
        supports_reasoning=False,
        cost_per_1m_input=0.15,
        cost_per_1m_output=0.60
    ),

    # DeepSeek Models (cost-effective alternative)
    LLMModel(
        id="deepseek-chat",
        name="DeepSeek Chat",
        provider=LLMProvider.DEEPSEEK,
        description="DeepSeek's conversational model - excellent value",
        max_tokens=128000,
        supports_reasoning=False,
        cost_per_1m_input=0.14,
        cost_per_1m_output=0.28
    ),
    LLMModel(
        id="deepseek-reasoner",
        name="DeepSeek Reasoner",
        provider=LLMProvider.DEEPSEEK,
        description="DeepSeek's reasoning model - great for complex tasks",
        max_tokens=64000,
        supports_reasoning=True,
        cost_per_1m_input=0.55,
        cost_per_1m_output=2.19
    ),
]

DEFAULT_MODEL = "gpt-5-mini"


def get_model(model_id: str) -> Optional[LLMModel]:
    """Get model by ID."""
    for model in AVAILABLE_MODELS:
        if model.id == model_id:
            return model
    return None


def get_models_by_provider(provider: LLMProvider) -> List[LLMModel]:
    """Get all models for a specific provider."""
    return [m for m in AVAILABLE_MODELS if m.provider == provider]


class MultiProviderLLMClient:
    """Unified client for multiple LLM providers."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        provider_api_keys: Optional[Dict[LLMProvider, str]] = None
    ):
        """
        Initialize multi-provider LLM client.

        Args:
            model_id: Model ID to use
            api_key: API key for the model's provider (overrides provider_api_keys)
            provider_api_keys: Dict mapping providers to their API keys
        """
        self.model_id = model_id
        self.model = get_model(model_id)

        if not self.model:
            raise ValueError(f"Unknown model: {model_id}. Available: {[m.id for m in AVAILABLE_MODELS]}")

        # Determine which provider to use
        self.provider = self.model.provider

        # Get API key
        if api_key:
            self.api_key = api_key
        elif provider_api_keys and self.provider in provider_api_keys:
            self.api_key = provider_api_keys[self.provider]
        else:
            raise ValueError(f"API key required for provider: {self.provider.value}")

        # Initialize provider-specific client
        self._init_client()

    def _init_client(self):
        """Initialize the provider-specific client."""
        if self.provider == LLMProvider.OPENAI:
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("OpenAI package not installed. Run: pip install openai")

        elif self.provider == LLMProvider.DEEPSEEK:
            try:
                import openai
                # DeepSeek uses OpenAI-compatible API
                self.client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url="https://api.deepseek.com"
                )
            except ImportError:
                raise ImportError("OpenAI package not installed. Run: pip install openai")

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        """
        Generate chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific params

        Returns:
            Generated response text
        """
        # Provider-specific parameters
        params = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
            **kwargs
        }

        try:
            response = self.client.chat.completions.create(
                **params,
                max_completion_tokens=max_tokens,
            )
        except Exception as e:
            msg = str(e).lower()
            # Some legacy models require max_tokens instead of max_completion_tokens
            if "max_completion_tokens" in msg and "unsupported" in msg:
                response = self.client.chat.completions.create(
                    **params,
                    max_tokens=max_tokens,
                )
            else:
                logger.error(f"LLM API call failed: {e}")
                raise

        return response.choices[0].message.content.strip()

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "id": self.model.id,
            "name": self.model.name,
            "provider": self.model.provider.value,
            "description": self.model.description,
            "max_tokens": self.model.max_tokens,
            "supports_reasoning": self.model.supports_reasoning,
            "cost_per_1m_input": self.model.cost_per_1m_input,
            "cost_per_1m_output": self.model.cost_per_1m_output,
        }


def get_available_models_for_frontend() -> List[Dict[str, Any]]:
    """
    Get all available models formatted for frontend.

    Returns:
        List of model dicts with provider grouping
    """
    models_by_provider = {}

    for model in AVAILABLE_MODELS:
        provider = model.provider.value
        if provider not in models_by_provider:
            models_by_provider[provider] = []

        models_by_provider[provider].append({
            "id": model.id,
            "name": model.name,
            "description": model.description,
            "supports_reasoning": model.supports_reasoning,
            "cost_estimate": f"${model.cost_per_1m_input:.2f}/1M input",
        })

    # Group by provider with provider info
    result = []
    for provider, models in models_by_provider.items():
        result.append({
            "provider": provider,
            "models": models
        })

    return result
