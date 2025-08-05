"""OpenAI client for GPT models."""

import openai
from typing import List, Dict, Any
import time

from ai_cli.llms.base import BaseLLMClient, ChatMessage, LLMResponse, ModelInfo


class OpenAIClient(BaseLLMClient):
    """OpenAI API client."""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.client = openai.AsyncOpenAI(api_key=api_key)
        
        # Token pricing (per 1k tokens)
        self.pricing = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
        }
    
    async def chat_completion(self, messages: List[ChatMessage], 
                            model: str, **kwargs) -> LLMResponse:
        """Generate a chat completion using OpenAI."""
        
        # Convert our ChatMessage format to OpenAI format
        openai_messages = []
        for msg in messages:
            openai_msg = {"role": msg.role, "content": msg.content}
            if msg.function_call:
                openai_msg["function_call"] = msg.function_call
            openai_messages.append(openai_msg)
        
        start_time = time.time()
        
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=openai_messages,
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", None),
                top_p=kwargs.get("top_p", 1.0),
                frequency_penalty=kwargs.get("frequency_penalty", 0.0),
                presence_penalty=kwargs.get("presence_penalty", 0.0),
                functions=kwargs.get("functions", None),
                function_call=kwargs.get("function_call", None),
            )
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Extract function calls if any
            function_calls = []
            if (response.choices[0].message.function_call):
                function_calls.append({
                    "name": response.choices[0].message.function_call.name,
                    "arguments": response.choices[0].message.function_call.arguments
                })
            
            # Calculate cost
            tokens_used = response.usage.total_tokens
            cost = self.calculate_cost(tokens_used, model)
            
            return LLMResponse(
                content=response.choices[0].message.content or "",
                model=model,
                provider="openai",
                tokens_used=tokens_used,
                cost_usd=cost,
                function_calls=function_calls,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "duration_ms": duration_ms,
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                }
            )
            
        except Exception as e:
            # Return error response
            return LLMResponse(
                content=f"Error: {str(e)}",
                model=model,
                provider="openai",
                tokens_used=0,
                cost_usd=0.0,
                function_calls=[],
                metadata={"error": str(e)}
            )
    
    def get_available_models(self) -> List[ModelInfo]:
        """Get list of available OpenAI models."""
        return [
            ModelInfo(
                name="gpt-4",
                provider="openai",
                context_length=8192,
                supports_functions=True,
                cost_per_1k_tokens=0.03,
                description="Most capable GPT-4 model"
            ),
            ModelInfo(
                name="gpt-4-turbo",
                provider="openai",
                context_length=128000,
                supports_functions=True,
                cost_per_1k_tokens=0.01,
                description="Latest GPT-4 Turbo with vision"
            ),
            ModelInfo(
                name="gpt-4o",
                provider="openai",
                context_length=128000,
                supports_functions=True,
                cost_per_1k_tokens=0.005,
                description="GPT-4 Omni multimodal model"
            ),
            ModelInfo(
                name="gpt-3.5-turbo",
                provider="openai",
                context_length=16384,
                supports_functions=True,
                cost_per_1k_tokens=0.001,
                description="Fast and efficient GPT-3.5"
            ),
        ]
    
    def calculate_cost(self, tokens: int, model: str) -> float:
        """Calculate cost for token usage."""
        # For simplicity, use average of input/output pricing
        if model in self.pricing:
            avg_price = (self.pricing[model]["input"] + self.pricing[model]["output"]) / 2
            return (tokens / 1000) * avg_price
        else:
            # Default pricing for unknown models
            return (tokens / 1000) * 0.02
