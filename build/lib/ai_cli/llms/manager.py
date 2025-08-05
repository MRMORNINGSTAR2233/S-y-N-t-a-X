"""LLM provider manager for handling multiple AI providers with unified interface."""

from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
import asyncio
import toml
from pathlib import Path

from ai_cli.llms.base import BaseLLMClient, ChatMessage, LLMResponse, ModelInfo
from ai_cli.llms.openai_client import OpenAIClient
from ai_cli.llms.anthropic_client import AnthropicClient
from ai_cli.llms.groq_client import GroqClient
from ai_cli.llms.ollama_client import OllamaClient
from ai_cli.llms.gemini_client import GeminiClient
from ai_cli.config.database import DatabaseManager


class ModelType(Enum):
    """Supported model types."""
    GPT_4 = "gpt-4"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4O = "gpt-4o"
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20240620"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"
    GROQ_LLAMA_70B = "llama2-70b-4096"
    GROQ_MIXTRAL = "mixtral-8x7b-32768"
    OLLAMA_LLAMA2 = "llama2"
    OLLAMA_CODELLAMA = "codellama"
    OLLAMA_MISTRAL = "mistral"
    GEMINI_1_5_PRO = "gemini-1.5-pro"
    GEMINI_1_5_FLASH = "gemini-1.5-flash"
    GEMINI_1_0_PRO = "gemini-1.0-pro"


class LLMManager:
    """Manages multiple LLM providers with unified interface."""
    
    # Model catalog
    MODEL_CATALOG = {
        ModelType.GPT_4: ModelInfo(
            name="gpt-4",
            provider="openai",
            context_length=8192,
            supports_functions=True,
            cost_per_1k_input=0.03,
            cost_per_1k_output=0.09000,
        ),
        ModelType.GPT_4_TURBO: ModelInfo(
            name="gpt-4-turbo",
            provider="openai",
            context_length=128000,
            supports_functions=True,
            cost_per_1k_input=0.01,
            cost_per_1k_output=0.03000,
        ),
        ModelType.GPT_4O: ModelInfo(
            name="gpt-4o",
            provider="openai",
            context_length=128000,
            supports_functions=True,
            cost_per_1k_input=0.005,
            cost_per_1k_output=0.01500,
        ),
        ModelType.CLAUDE_3_OPUS: ModelInfo(
            name="claude-3-opus-20240229",
            provider="anthropic",
            context_length=200000,
            supports_functions=True,
            cost_per_1k_input=0.015,
            cost_per_1k_output=0.04500,
        ),
        ModelType.CLAUDE_3_5_SONNET: ModelInfo(
            name="claude-3-5-sonnet-20240620",
            provider="anthropic",
            context_length=200000,
            supports_functions=True,
            cost_per_1k_input=0.003,
            cost_per_1k_output=0.00900,
        ),
        ModelType.CLAUDE_3_HAIKU: ModelInfo(
            name="claude-3-haiku-20240307",
            provider="anthropic",
            context_length=200000,
            supports_functions=True,
            cost_per_1k_input=0.00025,
            cost_per_1k_output=0.00075,
        ),
        ModelType.GROQ_LLAMA_70B: ModelInfo(
            name="llama2-70b-4096",
            provider="groq",
            context_length=4096,
            supports_functions=False,
            cost_per_1k_input=0.0008,
            cost_per_1k_output=0.00240,
        ),
        ModelType.GROQ_MIXTRAL: ModelInfo(
            name="mixtral-8x7b-32768",
            provider="groq",
            context_length=32768,
            supports_functions=False,
            cost_per_1k_input=0.0002,
            cost_per_1k_output=0.00060,
        ),
        ModelType.OLLAMA_LLAMA2: ModelInfo(
            name="llama2:7b",
            provider="ollama",
            context_length=4096,
            supports_functions=False,
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.00000,
        ),
        ModelType.OLLAMA_CODELLAMA: ModelInfo(
            name="llama3.2:latest",
            provider="ollama",
            context_length=8192,
            supports_functions=False,
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.00000,
        ),
        ModelType.GEMINI_1_5_PRO: ModelInfo(
            name="gemini-1.5-pro",
            provider="gemini",
            context_length=2097152,
            supports_functions=False,
            cost_per_1k_input=0.0035,
            cost_per_1k_output=0.01050,
        ),
        ModelType.GEMINI_1_5_FLASH: ModelInfo(
            name="gemini-1.5-flash",
            provider="gemini",
            context_length=1048576,
            supports_functions=False,
            cost_per_1k_input=0.00035,
            cost_per_1k_output=0.00105,
        ),
        ModelType.GEMINI_1_0_PRO: ModelInfo(
            name="gemini-1.0-pro",
            provider="gemini",
            context_length=32768,
            supports_functions=False,
            cost_per_1k_input=0.0005,
            cost_per_1k_output=0.00150,
        ),
    }
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
        self.clients: Dict[str, BaseLLMClient] = {}
        self.default_model = self.config.get("general", {}).get("default_model", "gpt-4")
        
        # Initialize database manager
        self.db_manager = DatabaseManager()
        
        # Initialize clients
        self._initialize_clients()
    
    def _get_default_config_path(self) -> Path:
        """Get the default configuration file path."""
        home_dir = Path.home()
        config_dir = home_dir / ".config" / "syntax"
        return config_dir / "config.toml"
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from TOML file."""
        if not self.config_path.exists():
            return {}
        
        try:
            with open(self.config_path, 'r') as f:
                return toml.load(f)
        except Exception as e:
            print(f"Warning: Failed to load config from {self.config_path}: {e}")
            return {}
    
    def _get_api_key(self, provider_name: str) -> Optional[str]:
        """Get API key for a provider from config or environment."""
        # First check config file
        providers_config = self.config.get("providers", {})
        provider_config = providers_config.get(provider_name, {})
        api_key = provider_config.get("api_key")
        
        # If not in config, check environment variables
        if not api_key:
            import os
            env_var_map = {
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY", 
                "groq": "GROQ_API_KEY",
                "gemini": "GOOGLE_API_KEY"
            }
            env_var = env_var_map.get(provider_name)
            if env_var:
                api_key = os.getenv(env_var)
        
        return api_key
    
    def _initialize_clients(self) -> None:
        """Initialize LLM clients for available providers."""
        providers = {
            "openai": OpenAIClient,
            "anthropic": AnthropicClient,
            "groq": GroqClient,
            "ollama": OllamaClient,
            "gemini": GeminiClient
        }
        
        for provider_name, client_class in providers.items():
            api_key = self._get_api_key(provider_name)
            
            if api_key or provider_name == "ollama":  # Ollama doesn't need API key
                try:
                    if provider_name == "ollama":
                        # Ollama uses local server
                        base_url = self.config.get("providers", {}).get("ollama", {}).get("base_url", "http://localhost:11434")
                        self.clients[provider_name] = client_class(base_url=base_url)
                    else:
                        self.clients[provider_name] = client_class(api_key)
                    
                    print(f"✅ Initialized {provider_name} client")
                except Exception as e:
                    print(f"⚠️  Failed to initialize {provider_name}: {e}")
            else:
                print(f"⚠️  No API key found for {provider_name}")
        
        if not self.clients:
            print("❌ No LLM providers available. Please configure API keys.")
        else:
            # Update default model if the configured one isn't available
            try:
                self._get_provider_for_model(self.default_model)
                print(f"🔄 Using configured default model: {self.default_model}")
            except ValueError:
                # Current default model isn't available, pick the first available model
                available_models = self.get_available_models()
                if available_models:
                    # Prefer models in order of general capability
                    preferred_order = [
                        "gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro",  # Gemini models
                        "claude-3-5-sonnet-20240620", "claude-3-opus-20240229", "claude-3-haiku-20240307",  # Anthropic
                        "gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo",  # OpenAI
                        "mixtral-8x7b-32768", "llama2-70b-4096",  # Groq
                        "llama2:7b", "llama3.2:latest"  # Ollama
                    ]
                    
                    # Find the first available model from preferred order
                    for preferred_model in preferred_order:
                        for available_model in available_models:
                            if available_model.name == preferred_model:
                                self.default_model = preferred_model
                                print(f"🔄 Using {self.default_model} as default model")
                                break
                        if self.default_model in [m.name for m in available_models]:
                            break
                    
                    # If none of the preferred models are available, use the first available
                    if self.default_model not in [m.name for m in available_models]:
                        self.default_model = available_models[0].name
                        print(f"🔄 Using {self.default_model} as default model")
        
        # Set fallback order
        self.fallback_order = ["anthropic", "openai", "gemini", "groq", "ollama"]
    
    async def chat_completion(self, messages: List[ChatMessage], 
                            model: Optional[str] = None, 
                            provider: Optional[str] = None,
                            **kwargs) -> LLMResponse:
        """Generate a chat completion using the specified or default model."""
        
        # Determine model and provider
        if not model:
            model = self.default_model
        
        if not provider:
            provider = self._get_provider_for_model(model)
        
        if provider not in self.clients:
            raise ValueError(f"Provider {provider} not available")
        
        client = self.clients[provider]
        
        # Generate completion
        response = await client.chat_completion(messages, model, **kwargs)
        
        # Log usage
        self.db_manager.log_usage(
            provider=provider,
            model=model,
            command="chat_completion",
            tokens_used=response.tokens_used,
            cost_usd=str(response.cost_usd),
            success=True
        )
        
        return response
    
    def _get_provider_for_model(self, model: str) -> str:
        """Get the provider for a given model."""
        # First check exact matches in the model catalog
        for model_type, model_info in self.MODEL_CATALOG.items():
            if (model_info.name == model or model_type.value == model) and model_info.provider in self.clients:
                return model_info.provider
        
        # Check if model matches any available provider patterns
        for provider in self.clients.keys():
            if provider == "ollama" and any(ollama_name in model.lower() 
                                          for ollama_name in ["llama", "mistral", "codellama", "qwen", "granite", "phi", "gemma", "deepseek"]):
                return "ollama"
            elif provider == "openai" and "gpt" in model.lower():
                return "openai"
            elif provider == "anthropic" and "claude" in model.lower():
                return "anthropic"
            elif provider == "groq" and any(groq_name in model.lower() 
                                          for groq_name in ["llama", "mixtral", "gemma"]):
                return "groq"
            elif provider == "gemini" and "gemini" in model.lower():
                return "gemini"
        
        # If no pattern matches, the model is not available
        raise ValueError(f"Model {model} not available with current providers: {list(self.clients.keys())}")
    
    def get_available_models(self, provider: Optional[str] = None) -> List[ModelInfo]:
        """Get available models for a provider or all providers."""
        if provider:
            return [info for info in self.MODEL_CATALOG.values() 
                   if info.provider == provider and provider in self.clients]
        else:
            return [info for info in self.MODEL_CATALOG.values() 
                   if info.provider in self.clients]
    
    def get_model_info(self, model: str) -> Optional[ModelInfo]:
        """Get information about a specific model."""
        for model_info in self.MODEL_CATALOG.values():
            if model_info.name == model:
                return model_info
        return None
    
    def is_provider_available(self, provider: str) -> bool:
        """Check if a provider is available."""
        return provider in self.clients
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers."""
        return list(self.clients.keys())
    
    async def get_best_model_for_task(self, task_type: str, 
                                    requirements: Dict[str, Any] = None) -> str:
        """Get the best model for a specific task type."""
        requirements = requirements or {}
        
        # Task-specific model recommendations
        task_models = {
            "code_generation": [
                ModelType.CLAUDE_3_5_SONNET.value,
                ModelType.GPT_4O.value,
                ModelType.OLLAMA_CODELLAMA.value
            ],
            "debugging": [
                ModelType.CLAUDE_3_OPUS.value,
                ModelType.GPT_4_TURBO.value,
                ModelType.CLAUDE_3_5_SONNET.value
            ],
            "code_review": [
                ModelType.CLAUDE_3_OPUS.value,
                ModelType.GPT_4.value,
                ModelType.CLAUDE_3_5_SONNET.value
            ],
            "search": [
                ModelType.CLAUDE_3_5_SONNET.value,
                ModelType.GPT_4O.value,
                ModelType.GROQ_MIXTRAL.value
            ],
            "fast_tasks": [
                ModelType.GROQ_MIXTRAL.value,
                ModelType.GROQ_LLAMA_70B.value,
                ModelType.CLAUDE_3_HAIKU.value
            ]
        }
        
        preferred_models = task_models.get(task_type, [self.default_model])
        
        # Filter by requirements
        max_cost = requirements.get("max_cost_per_1k", float('inf'))
        min_context = requirements.get("min_context_length", 0)
        requires_functions = requirements.get("requires_functions", False)
        
        for model in preferred_models:
            model_info = self.get_model_info(model)
            if model_info:
                # Check if provider is available
                if not self.is_provider_available(model_info.provider):
                    continue
                
                # Check requirements
                if (model_info.cost_per_1k_tokens <= max_cost and
                    model_info.context_length >= min_context and
                    (not requires_functions or model_info.supports_functions)):
                    return model
        
        # Fallback to default model
        return self.default_model
    
    async def multi_provider_completion(self, messages: List[ChatMessage], 
                                      providers: List[str] = None,
                                      temperature: float = 0.7) -> Dict[str, LLMResponse]:
        """Get completions from multiple providers for comparison."""
        if not providers:
            providers = self.get_available_providers()
        
        tasks = []
        for provider in providers:
            if provider in self.clients:
                # Get best model for this provider
                available_models = self.get_available_models(provider)
                if available_models:
                    model = available_models[0].name  # Use first available model
                    task = self.chat_completion(
                        messages=messages,
                        model=model,
                        provider=provider,
                        temperature=temperature
                    )
                    tasks.append((provider, task))
        
        # Execute all tasks concurrently
        results = {}
        if tasks:
            completed_tasks = await asyncio.gather(
                *[task for _, task in tasks],
                return_exceptions=True
            )
            
            for i, (provider, _) in enumerate(tasks):
                result = completed_tasks[i]
                if isinstance(result, Exception):
                    print(f"Error with {provider}: {result}")
                else:
                    results[provider] = result
        
        return results
    
    def get_usage_stats(self, days: int = 30) -> Dict[str, Any]:
        """Get usage statistics across all providers."""
        return self.db_manager.get_usage_stats(days=days)
    
    def estimate_cost(self, text: str, model: str) -> float:
        """Estimate cost for processing given text with a model."""
        # Rough token estimation (1 token ≈ 4 characters)
        estimated_tokens = len(text) // 4
        
        model_info = self.get_model_info(model)
        if model_info:
            return (estimated_tokens / 1000) * model_info.cost_per_1k_tokens
        
        return 0.0
    
    def set_default_model(self, model: str) -> bool:
        """Set the default model, returns True if successful"""
        try:
            # Verify the model is available
            provider = self._get_provider_for_model(model)
            if provider in self.clients:
                self.default_model = model
                print(f"✅ Default model set to: {model} ({provider})")
                return True
            else:
                print(f"❌ Provider {provider} not available for model {model}")
                return False
        except ValueError as e:
            print(f"❌ Invalid model {model}: {e}")
            return False
    
    def get_current_configuration(self) -> Dict[str, Any]:
        """Get current LLM manager configuration"""
        try:
            default_provider = self._get_provider_for_model(self.default_model)
        except ValueError:
            default_provider = "unknown"
        
        return {
            "default_model": self.default_model,
            "default_provider": default_provider,
            "available_providers": self.get_available_providers(),
            "available_models": [m.name for m in self.get_available_models()],
            "total_clients": len(self.clients)
        }
    
    def get_provider_for_model(self, model: str) -> str:
        """Public method to get provider for a model"""
        return self._get_provider_for_model(model)
