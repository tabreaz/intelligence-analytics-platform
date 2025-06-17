# src/core/llm/base_llm.py
from abc import ABC, abstractmethod
from typing import cast, Any

from src.core.logger import get_logger
from ..config_manager import LLMConfig

logger = get_logger(__name__)


class BaseLLMClient(ABC):
    """Base class for LLM clients"""

    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    async def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """Generate response from LLM"""
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI client implementation"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        from openai import OpenAI
        self.client = OpenAI(api_key=config.api_key)

    async def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """Generate response using OpenAI"""
        try:
            # Use the new v1 API
            # Cast messages to Any to satisfy type checker
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=cast(Any, messages),  # Type cast for PyCharm
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                **kwargs
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise


class AnthropicClient(BaseLLMClient):
    """Anthropic client implementation"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        import anthropic
        self.client = anthropic.Anthropic(api_key=config.api_key)

    async def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """Generate response using Anthropic"""
        try:
            # Cast messages to Any to satisfy type checker
            messages = [{"role": "user", "content": user_prompt}]
            
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=system_prompt,
                messages=cast(Any, messages),  # Type cast for PyCharm
                **kwargs
            )

            return response.content[0].text

        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise


class LLMClientFactory:
    """Factory for creating LLM clients"""

    @staticmethod
    def create_client(config: LLMConfig) -> BaseLLMClient:
        """Create LLM client based on configuration"""

        if 'gpt' in config.model.lower():
            return OpenAIClient(config)
        elif 'claude' in config.model.lower():
            return AnthropicClient(config)
        else:
            raise ValueError(f"Unsupported LLM model: {config.model}")