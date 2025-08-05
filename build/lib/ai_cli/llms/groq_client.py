"""Groq client for ultra-fast LLaMA and Mixtral models."""

import httpx
from typing import List, Dict, Any
import time
import json

from ai_cli.llms.base import BaseLLMClient, ChatMessage, LLMResponse, ModelInfo


class GroqClient(BaseLLMClient):
    """Groq API client for ultra-fast inference."""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.base_url = "https://api.groq.com/openai/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Token pricing (per 1k tokens) - Groq is very cost-effective
        self.pricing = {
            "llama2-70b-4096": 0.0008,
            "mixtral-8x7b-32768": 0.0002,
            "gemma-7b-it": 0.0001,
        }
    
    async def chat_completion(self, messages: List[ChatMessage], 
                            model: str, **kwargs) -> LLMResponse:
        """Generate a chat completion using Groq."""
        
        # Convert our ChatMessage format to OpenAI-compatible format
        groq_messages = []
        for msg in messages:
            groq_messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        start_time = time.time()
        
        payload = {
            "model": model,
            "messages": groq_messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", None),
            "top_p": kwargs.get("top_p", 1.0),
            "stream": False,
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Extract response content
            content = data["choices"][0]["message"]["content"] or ""
            
            # Groq typically doesn't support function calling yet
            function_calls = []
            
            # Calculate tokens and cost
            usage = data.get("usage", {})
            tokens_used = usage.get("total_tokens", 0)
            cost = self.calculate_cost(tokens_used, model)
            
            return LLMResponse(
                content=content,
                model=model,
                provider="groq",
                tokens_used=tokens_used,
                cost_usd=cost,
                function_calls=function_calls,
                metadata={
                    "finish_reason": data["choices"][0]["finish_reason"],
                    "duration_ms": duration_ms,
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                    "x_groq": data.get("x_groq", {}),  # Groq-specific metadata
                }
            )
            
        except httpx.HTTPStatusError as e:
            error_detail = "Unknown error"
            try:
                error_data = e.response.json()
                error_detail = error_data.get("error", {}).get("message", str(e))
            except:
                error_detail = str(e)
            
            return LLMResponse(
                content=f"Groq API Error: {error_detail}",
                model=model,
                provider="groq",
                tokens_used=0,
                cost_usd=0.0,
                function_calls=[],
                metadata={"error": error_detail, "status_code": e.response.status_code}
            )
        
        except Exception as e:
            return LLMResponse(
                content=f"Error: {str(e)}",
                model=model,
                provider="groq",
                tokens_used=0,
                cost_usd=0.0,
                function_calls=[],
                metadata={"error": str(e)}
            )
    
    def get_available_models(self) -> List[ModelInfo]:
        """Get list of available Groq models."""
        return [
            ModelInfo(
                name="llama2-70b-4096",
                provider="groq",
                context_length=4096,
                supports_functions=False,
                cost_per_1k_tokens=0.0008,
                description="Llama 2 70B - Ultra-fast inference"
            ),
            ModelInfo(
                name="mixtral-8x7b-32768",
                provider="groq",
                context_length=32768,
                supports_functions=False,
                cost_per_1k_tokens=0.0002,
                description="Mixtral 8x7B - Large context, fast"
            ),
            ModelInfo(
                name="gemma-7b-it",
                provider="groq",
                context_length=8192,
                supports_functions=False,
                cost_per_1k_tokens=0.0001,
                description="Google Gemma 7B - Instruction tuned"
            ),
        ]
    
    def calculate_cost(self, tokens: int, model: str) -> float:
        """Calculate cost for token usage."""
        price_per_1k = self.pricing.get(model, 0.001)  # Default if unknown
        return (tokens / 1000) * price_per_1k
    
    async def get_models(self) -> List[Dict[str, Any]]:
        """Get available models from Groq API."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers=self.headers
                )
                response.raise_for_status()
                data = response.json()
                return data.get("data", [])
                
        except Exception as e:
            print(f"Error fetching Groq models: {e}")
            return []
