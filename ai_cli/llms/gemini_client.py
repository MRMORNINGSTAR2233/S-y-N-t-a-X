"""Google Gemini client for the AI CLI."""

import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from ai_cli.llms.base import BaseLLMClient, ChatMessage, LLMResponse, ModelInfo


@dataclass
class GeminiClient:
    """Client for Google Gemini models."""
    
    def __init__(self, api_key: str):
        """Initialize Gemini client."""
        self.api_key = api_key
        genai.configure(api_key=api_key)
        
        # Available models
        self.models = [
            "gemini-1.5-pro",
            "gemini-1.5-flash", 
            "gemini-1.0-pro"
        ]
        
        # Safety settings - allow most content for code analysis
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    
    async def chat_completion(
        self,
        messages: List[ChatMessage],
        model: str = "gemini-1.5-pro",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate chat completion using Gemini."""
        
        try:
            # Convert messages to Gemini format
            gemini_messages = self._convert_messages(messages)
            
            # Create model instance
            gemini_model = genai.GenerativeModel(
                model_name=model,
                safety_settings=self.safety_settings
            )
            
            # Generation config
            generation_config = {
                "temperature": temperature,
                "candidate_count": 1,
            }
            
            if max_tokens:
                generation_config["max_output_tokens"] = max_tokens
            
            # Generate response
            response = await asyncio.to_thread(
                gemini_model.generate_content,
                gemini_messages,
                generation_config=generation_config
            )
            
            # Extract response text
            if response.candidates and len(response.candidates) > 0:
                content = response.candidates[0].content.parts[0].text
            else:
                content = "No response generated"
            
            # Calculate usage (approximate for Gemini)
            prompt_tokens = self._estimate_tokens(messages)
            completion_tokens = self._estimate_tokens([ChatMessage(role="assistant", content=content)])
            
            return LLMResponse(
                content=content,
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            )
            
        except Exception as e:
            # Handle rate limits and other errors
            if "quota" in str(e).lower() or "rate" in str(e).lower():
                raise Exception(f"Gemini rate limit exceeded: {e}")
            elif "safety" in str(e).lower():
                raise Exception(f"Gemini safety filter triggered: {e}")
            else:
                raise Exception(f"Gemini API error: {e}")
    
    def _convert_messages(self, messages: List[ChatMessage]) -> str:
        """Convert chat messages to Gemini format."""
        
        # Gemini uses a simple text format for generation
        # Combine all messages into a single prompt
        prompt_parts = []
        
        for message in messages:
            if message.role == "system":
                prompt_parts.append(f"System: {message.content}")
            elif message.role == "user":
                prompt_parts.append(f"User: {message.content}")
            elif message.role == "assistant":
                prompt_parts.append(f"Assistant: {message.content}")
        
        return "\n\n".join(prompt_parts)
    
    def _estimate_tokens(self, messages: List[ChatMessage]) -> int:
        """Estimate token count for messages."""
        
        total_chars = sum(len(msg.content) for msg in messages)
        # Rough estimate: 1 token ≈ 4 characters for English text
        return max(1, total_chars // 4)
    
    def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get information about a specific model."""
        
        model_info = {
            "gemini-1.5-pro": {
                "max_tokens": 2097152,  # 2M tokens context
                "cost_per_1k_prompt": 0.0035,
                "cost_per_1k_completion": 0.0105,
                "description": "Most capable Gemini model with large context"
            },
            "gemini-1.5-flash": {
                "max_tokens": 1048576,  # 1M tokens context
                "cost_per_1k_prompt": 0.00035,
                "cost_per_1k_completion": 0.00105,
                "description": "Fast and efficient Gemini model"
            },
            "gemini-1.0-pro": {
                "max_tokens": 32768,  # 32K tokens context
                "cost_per_1k_prompt": 0.0005,
                "cost_per_1k_completion": 0.0015,
                "description": "Original Gemini Pro model"
            }
        }
        
        return model_info.get(model, {
            "max_tokens": 32768,
            "cost_per_1k_prompt": 0.001,
            "cost_per_1k_completion": 0.002,
            "description": "Unknown Gemini model"
        })
    
    def calculate_cost(self, prompt_tokens: int, completion_tokens: int, model: str) -> float:
        """Calculate cost for the API call."""
        
        info = self.get_model_info(model)
        
        prompt_cost = (prompt_tokens / 1000) * info["cost_per_1k_prompt"]
        completion_cost = (completion_tokens / 1000) * info["cost_per_1k_completion"]
        
        return prompt_cost + completion_cost
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return self.models.copy()
    
    async def test_connection(self) -> bool:
        """Test the connection to Gemini API."""
        
        try:
            test_messages = [
                ChatMessage(role="user", content="Hello, can you respond with just 'OK'?")
            ]
            
            response = await self.chat_completion(
                messages=test_messages,
                model="gemini-1.5-flash",  # Use fastest model for testing
                temperature=0.1,
                max_tokens=10
            )
            
            return "ok" in response.content.lower()
            
        except Exception as e:
            print(f"Gemini connection test failed: {e}")
            return False
