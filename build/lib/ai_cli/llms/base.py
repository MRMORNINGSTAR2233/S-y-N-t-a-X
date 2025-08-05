"""Base classes for LLM clients and data structures."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum


@dataclass
class ChatMessage:
    """Represents a chat message."""
    role: str  # "user", "assistant", "system"
    content: str
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary format."""
        return {"role": self.role, "content": self.content}


@dataclass
class LLMResponse:
    """Response from an LLM."""
    content: str
    model: str
    provider: str = ""
    tokens_used: int = 0
    cost_usd: float = 0.0
    function_calls: List[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None
    usage: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None
    
    def __post_init__(self):
        """Initialize mutable default values."""
        if self.function_calls is None:
            self.function_calls = []
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "content": self.content,
            "model": self.model,
            "provider": self.provider,
            "tokens_used": self.tokens_used,
            "cost_usd": self.cost_usd,
            "function_calls": self.function_calls,
            "metadata": self.metadata,
            "usage": self.usage,
            "finish_reason": self.finish_reason
        }


@dataclass
class ModelInfo:
    """Information about a language model."""
    name: str
    provider: str
    context_length: int
    supports_streaming: bool = True
    supports_functions: bool = False
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "name": self.name,
            "provider": self.provider,
            "context_length": self.context_length,
            "supports_streaming": self.supports_streaming,
            "supports_functions": self.supports_functions,
            "cost_per_1k_input": self.cost_per_1k_input,
            "cost_per_1k_output": self.cost_per_1k_output
        }


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize the client with API key and optional parameters."""
        self.api_key = api_key
        self.config = kwargs
    
    @abstractmethod
    async def chat_completion(self, messages: List[ChatMessage], model: str, **kwargs) -> LLMResponse:
        """Send a chat completion request."""
        pass
    
    @abstractmethod
    def get_model_info(self, model: str) -> ModelInfo:
        """Get information about a specific model."""
        pass
    
    @abstractmethod
    def list_models(self) -> List[str]:
        """List available models for this provider."""
        pass
    
    def calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Calculate the cost for a request. Override in subclasses for provider-specific logic."""
        model_info = self.get_model_info(model)
        input_cost = (input_tokens / 1000) * model_info.cost_per_1k_input
        output_cost = (output_tokens / 1000) * model_info.cost_per_1k_output
        return input_cost + output_cost
    
    async def test_connection(self) -> bool:
        """Test if the client can connect to the provider."""
        try:
            # Test with a simple message
            test_messages = [ChatMessage(role="user", content="Hello")]
            models = self.list_models()
            if models:
                await self.chat_completion(test_messages, models[0])
                return True
            return False
        except Exception:
            return False
