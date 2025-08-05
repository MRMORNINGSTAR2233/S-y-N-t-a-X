"""Anthropic client for Claude models."""

import anthropic
from typing import List, Dict, Any
import time

from ai_cli.llms.base import BaseLLMClient, ChatMessage, LLMResponse, ModelInfo


class AnthropicClient(BaseLLMClient):
    """Anthropic API client for Claude models."""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        
        # Token pricing (per 1k tokens)
        self.pricing = {
            "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
            "claude-3-5-sonnet-20240620": {"input": 0.003, "output": 0.015},
            "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
        }
    
    async def chat_completion(self, messages: List[ChatMessage], 
                            model: str, **kwargs) -> LLMResponse:
        """Generate a chat completion using Anthropic Claude."""
        
        # Separate system message from other messages
        system_message = ""
        claude_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                claude_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        start_time = time.time()
        
        try:
            # Prepare request parameters
            request_params = {
                "model": model,
                "messages": claude_messages,
                "max_tokens": kwargs.get("max_tokens", 4096),
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 1.0),
            }
            
            if system_message:
                request_params["system"] = system_message
            
            # Handle tools/functions if provided
            if kwargs.get("functions"):
                tools = []
                for func in kwargs["functions"]:
                    tools.append({
                        "name": func["name"],
                        "description": func["description"],
                        "input_schema": func["parameters"]
                    })
                request_params["tools"] = tools
            
            response = await self.client.messages.create(**request_params)
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Extract content and function calls
            content = ""
            function_calls = []
            
            for content_block in response.content:
                if content_block.type == "text":
                    content += content_block.text
                elif content_block.type == "tool_use":
                    function_calls.append({
                        "name": content_block.name,
                        "arguments": content_block.input,
                        "id": content_block.id
                    })
            
            # Calculate cost
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            total_tokens = input_tokens + output_tokens
            cost = self.calculate_cost_detailed(input_tokens, output_tokens, model)
            
            return LLMResponse(
                content=content,
                model=model,
                provider="anthropic",
                tokens_used=total_tokens,
                cost_usd=cost,
                function_calls=function_calls,
                metadata={
                    "stop_reason": response.stop_reason,
                    "duration_ms": duration_ms,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                }
            )
            
        except Exception as e:
            return LLMResponse(
                content=f"Error: {str(e)}",
                model=model,
                provider="anthropic",
                tokens_used=0,
                cost_usd=0.0,
                function_calls=[],
                metadata={"error": str(e)}
            )
    
    def get_available_models(self) -> List[ModelInfo]:
        """Get list of available Anthropic models."""
        return [
            ModelInfo(
                name="claude-3-opus-20240229",
                provider="anthropic",
                context_length=200000,
                supports_functions=True,
                cost_per_1k_tokens=0.015,
                description="Most powerful Claude 3 model"
            ),
            ModelInfo(
                name="claude-3-5-sonnet-20240620",
                provider="anthropic",
                context_length=200000,
                supports_functions=True,
                cost_per_1k_tokens=0.003,
                description="Balanced Claude 3.5 model"
            ),
            ModelInfo(
                name="claude-3-sonnet-20240229",
                provider="anthropic",
                context_length=200000,
                supports_functions=True,
                cost_per_1k_tokens=0.003,
                description="Balanced Claude 3 model"
            ),
            ModelInfo(
                name="claude-3-haiku-20240307",
                provider="anthropic",
                context_length=200000,
                supports_functions=True,
                cost_per_1k_tokens=0.00025,
                description="Fastest Claude 3 model"
            ),
        ]
    
    def calculate_cost(self, tokens: int, model: str) -> float:
        """Calculate cost for token usage (simplified)."""
        if model in self.pricing:
            # Use average of input/output pricing for simplification
            avg_price = (self.pricing[model]["input"] + self.pricing[model]["output"]) / 2
            return (tokens / 1000) * avg_price
        else:
            return (tokens / 1000) * 0.01  # Default pricing
    
    def calculate_cost_detailed(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Calculate cost with separate input/output pricing."""
        if model in self.pricing:
            input_cost = (input_tokens / 1000) * self.pricing[model]["input"]
            output_cost = (output_tokens / 1000) * self.pricing[model]["output"]
            return input_cost + output_cost
        else:
            return ((input_tokens + output_tokens) / 1000) * 0.01
