"""Ollama client for local LLM models."""

import httpx
from typing import List, Dict, Any
import time
import json

from ai_cli.llms.base import BaseLLMClient, ChatMessage, LLMResponse, ModelInfo


class OllamaClient(BaseLLMClient):
    """Ollama client for local model inference."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        # Ollama doesn't require an API key
        super().__init__("")
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}
    
    async def chat_completion(self, messages: List[ChatMessage], 
                            model: str, **kwargs) -> LLMResponse:
        """Generate a chat completion using Ollama."""
        
        # Convert our ChatMessage format to Ollama format
        ollama_messages = []
        for msg in messages:
            ollama_messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        start_time = time.time()
        
        payload = {
            "model": model,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 1.0),
                "num_predict": kwargs.get("max_tokens", -1),
            }
        }
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:  # Longer timeout for local inference
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Extract response content
            content = data.get("message", {}).get("content", "")
            
            # Ollama doesn't support function calling
            function_calls = []
            
            # Estimate tokens (Ollama doesn't always provide exact counts)
            tokens_used = self._estimate_tokens(content)
            
            return LLMResponse(
                content=content,
                model=model,
                provider="ollama",
                tokens_used=tokens_used,
                cost_usd=0.0,  # Local models are free
                function_calls=function_calls,
                metadata={
                    "duration_ms": duration_ms,
                    "eval_count": data.get("eval_count", 0),
                    "eval_duration": data.get("eval_duration", 0),
                    "load_duration": data.get("load_duration", 0),
                    "prompt_eval_count": data.get("prompt_eval_count", 0),
                    "prompt_eval_duration": data.get("prompt_eval_duration", 0),
                }
            )
            
        except httpx.ConnectError:
            return LLMResponse(
                content="Error: Ollama server not running. Please start Ollama and ensure it's accessible at " + self.base_url,
                model=model,
                provider="ollama",
                tokens_used=0,
                cost_usd=0.0,
                function_calls=[],
                metadata={"error": "connection_failed"}
            )
        
        except httpx.HTTPStatusError as e:
            error_detail = "Unknown error"
            try:
                error_data = e.response.json()
                error_detail = error_data.get("error", str(e))
            except:
                error_detail = str(e)
            
            return LLMResponse(
                content=f"Ollama Error: {error_detail}",
                model=model,
                provider="ollama",
                tokens_used=0,
                cost_usd=0.0,
                function_calls=[],
                metadata={"error": error_detail, "status_code": e.response.status_code}
            )
        
        except Exception as e:
            return LLMResponse(
                content=f"Error: {str(e)}",
                model=model,
                provider="ollama",
                tokens_used=0,
                cost_usd=0.0,
                function_calls=[],
                metadata={"error": str(e)}
            )
    
    def get_available_models(self) -> List[ModelInfo]:
        """Get list of common Ollama models."""
        return [
            ModelInfo(
                name="llama2:7b",
                provider="ollama",
                context_length=4096,
                supports_functions=False,
                cost_per_1k_input=0.0,
                cost_per_1k_output=0.0
            ),
            ModelInfo(
                name="llama3.2:latest",
                provider="ollama",
                context_length=8192,
                supports_functions=False,
                cost_per_1k_input=0.0,
                cost_per_1k_output=0.0
            ),
            ModelInfo(
                name="granite3.3:latest",
                provider="ollama",
                context_length=8192,
                supports_functions=False,
                cost_per_1k_input=0.0,
                cost_per_1k_output=0.0
            ),
            ModelInfo(
                name="qwen3:14b",
                provider="ollama",
                context_length=16384,
                supports_functions=False,
                cost_per_1k_input=0.0,
                cost_per_1k_output=0.0
            ),
            ModelInfo(
                name="gemma3:12b",
                provider="ollama",
                context_length=8192,
                supports_functions=False,
                cost_per_1k_input=0.0,
                cost_per_1k_output=0.0
            ),
        ]

    def get_model_info(self, model: str) -> ModelInfo:
        """Get information about a specific model."""
        available_models = self.get_available_models()
        for model_info in available_models:
            if model_info.name == model:
                return model_info
        
        # Return default model info if not found
        return ModelInfo(
            name=model,
            provider="ollama",
            context_length=4096,
            supports_functions=False,
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.0
        )

    def list_models(self) -> List[str]:
        """List available models for this provider."""
        return [model.name for model in self.get_available_models()]
    
    def calculate_cost(self, tokens: int, model: str) -> float:
        """Calculate cost for token usage (always 0 for local models)."""
        return 0.0
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)."""
        # Very rough estimate: 1 token ≈ 4 characters
        return max(1, len(text) // 4)
    
    async def get_installed_models(self) -> List[Dict[str, Any]]:
        """Get actually installed models from Ollama."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                data = response.json()
                models = data.get("models", [])
                
                # Filter out empty models and add basic validation
                valid_models = []
                for model in models:
                    if model.get("name") and model.get("name").strip():
                        valid_models.append(model)
                
                return valid_models
                
        except httpx.ConnectError as e:
            raise Exception(f"Cannot connect to Ollama server at {self.base_url}. Is Ollama running?")
        except httpx.TimeoutException as e:
            raise Exception(f"Timeout connecting to Ollama server at {self.base_url}")
        except httpx.HTTPStatusError as e:
            raise Exception(f"HTTP error from Ollama server: {e.response.status_code}")
        except Exception as e:
            raise Exception(f"Error fetching Ollama models: {str(e)}")
    
    async def pull_model(self, model_name: str) -> bool:
        """Pull/download a model in Ollama."""
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:  # Long timeout for model downloads
                response = await client.post(
                    f"{self.base_url}/api/pull",
                    json={"name": model_name},
                    headers=self.headers
                )
                response.raise_for_status()
                return True
                
        except Exception as e:
            print(f"Error pulling model {model_name}: {e}")
            return False
    
    async def check_model_exists(self, model_name: str) -> bool:
        """Check if a model is available locally."""
        try:
            installed_models = await self.get_installed_models()
            return any(model["name"].startswith(model_name) for model in installed_models)
        except:
            return False
