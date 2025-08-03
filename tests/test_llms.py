"""Tests for LLM manager and providers."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ai_cli.llms.manager import LLMManager, ModelInfo
from ai_cli.llms.openai_client import OpenAIClient
from ai_cli.llms.anthropic_client import AnthropicClient
from ai_cli.llms.groq_client import GroqClient
from ai_cli.llms.ollama_client import OllamaClient


class TestLLMManager:
    """Test cases for LLM manager."""
    
    @pytest.fixture
    def mock_settings(self):
        """Mock settings for testing."""
        settings = MagicMock()
        settings.providers = MagicMock()
        settings.providers.openai = MagicMock()
        settings.providers.openai.api_key = "test_key"
        settings.providers.openai.enabled = True
        settings.providers.anthropic = MagicMock()
        settings.providers.anthropic.api_key = "test_key"
        settings.providers.anthropic.enabled = True
        settings.providers.groq = MagicMock()
        settings.providers.groq.api_key = "test_key"
        settings.providers.groq.enabled = True
        settings.providers.ollama = MagicMock()
        settings.providers.ollama.base_url = "http://localhost:11434"
        settings.providers.ollama.enabled = True
        return settings
    
    @pytest.fixture
    def mock_db_manager(self):
        """Mock database manager for testing."""
        db_manager = MagicMock()
        db_manager.get_api_key = MagicMock(return_value="test_key")
        db_manager.log_llm_usage = MagicMock()
        return db_manager
    
    def test_llm_manager_initialization(self, mock_settings, mock_db_manager):
        """Test LLM manager initialization."""
        manager = LLMManager(mock_settings, mock_db_manager)
        
        assert manager.settings == mock_settings
        assert manager.db_manager == mock_db_manager
        assert len(manager.available_models) > 0
    
    @pytest.mark.asyncio
    async def test_get_best_model_for_task(self, mock_settings, mock_db_manager):
        """Test getting best model for specific tasks."""
        manager = LLMManager(mock_settings, mock_db_manager)
        
        # Test different task types
        code_model = await manager.get_best_model_for_task("code_generation")
        assert code_model in ["gpt-4", "claude-3-5-sonnet-20241022"]
        
        debug_model = await manager.get_best_model_for_task("debugging")
        assert debug_model in ["gpt-4", "claude-3-5-sonnet-20241022"]
        
        review_model = await manager.get_best_model_for_task("code_review")
        assert review_model in ["gpt-4", "claude-3-5-sonnet-20241022"]
    
    @pytest.mark.asyncio
    async def test_chat_completion_openai(self, mock_settings, mock_db_manager):
        """Test chat completion with OpenAI."""
        manager = LLMManager(mock_settings, mock_db_manager)
        
        # Mock OpenAI client
        with patch.object(manager, '_get_provider_client') as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat_completion = AsyncMock(return_value=MagicMock(
                content="Test response",
                usage={"prompt_tokens": 10, "completion_tokens": 5}
            ))
            mock_get_client.return_value = mock_client
            
            from ai_cli.llms.manager import ChatMessage
            messages = [ChatMessage(role="user", content="Hello")]
            
            response = await manager.chat_completion(
                messages=messages,
                model="gpt-4"
            )
            
            assert response.content == "Test response"
            mock_client.chat_completion.assert_called_once()
    
    def test_estimate_cost(self, mock_settings, mock_db_manager):
        """Test cost estimation."""
        manager = LLMManager(mock_settings, mock_db_manager)
        
        cost = manager.estimate_cost("gpt-4", prompt_tokens=1000, completion_tokens=500)
        assert cost > 0
        
        # Test with unknown model
        cost_unknown = manager.estimate_cost("unknown-model", prompt_tokens=1000, completion_tokens=500)
        assert cost_unknown == 0.0


class TestOpenAIClient:
    """Test cases for OpenAI client."""
    
    @pytest.mark.asyncio
    async def test_chat_completion(self):
        """Test OpenAI chat completion."""
        client = OpenAIClient("test_key")
        
        with patch('openai.AsyncOpenAI') as mock_openai:
            # Mock the OpenAI response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Test response"
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 5
            
            mock_openai.return_value.chat.completions.create = AsyncMock(return_value=mock_response)
            
            from ai_cli.llms.manager import ChatMessage
            messages = [ChatMessage(role="user", content="Hello")]
            
            response = await client.chat_completion(
                messages=messages,
                model="gpt-4"
            )
            
            assert response.content == "Test response"
            assert response.usage["prompt_tokens"] == 10
            assert response.usage["completion_tokens"] == 5


class TestAnthropicClient:
    """Test cases for Anthropic client."""
    
    @pytest.mark.asyncio
    async def test_chat_completion(self):
        """Test Anthropic chat completion."""
        client = AnthropicClient("test_key")
        
        with patch('anthropic.AsyncAnthropic') as mock_anthropic:
            # Mock the Anthropic response
            mock_response = MagicMock()
            mock_response.content = [MagicMock()]
            mock_response.content[0].text = "Test response"
            mock_response.usage.input_tokens = 10
            mock_response.usage.output_tokens = 5
            
            mock_anthropic.return_value.messages.create = AsyncMock(return_value=mock_response)
            
            from ai_cli.llms.manager import ChatMessage
            messages = [ChatMessage(role="user", content="Hello")]
            
            response = await client.chat_completion(
                messages=messages,
                model="claude-3-5-sonnet-20241022"
            )
            
            assert response.content == "Test response"
            assert response.usage["prompt_tokens"] == 10
            assert response.usage["completion_tokens"] == 5


class TestGroqClient:
    """Test cases for Groq client."""
    
    @pytest.mark.asyncio
    async def test_chat_completion(self):
        """Test Groq chat completion."""
        client = GroqClient("test_key")
        
        with patch('groq.AsyncGroq') as mock_groq:
            # Mock the Groq response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Test response"
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 5
            
            mock_groq.return_value.chat.completions.create = AsyncMock(return_value=mock_response)
            
            from ai_cli.llms.manager import ChatMessage
            messages = [ChatMessage(role="user", content="Hello")]
            
            response = await client.chat_completion(
                messages=messages,
                model="llama-3.1-70b-versatile"
            )
            
            assert response.content == "Test response"
            assert response.usage["prompt_tokens"] == 10
            assert response.usage["completion_tokens"] == 5


class TestOllamaClient:
    """Test cases for Ollama client."""
    
    @pytest.mark.asyncio
    async def test_chat_completion(self):
        """Test Ollama chat completion."""
        client = OllamaClient("http://localhost:11434")
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            # Mock the Ollama response
            mock_response = MagicMock()
            mock_response.json = AsyncMock(return_value={
                "message": {"content": "Test response"},
                "prompt_eval_count": 10,
                "eval_count": 5
            })
            mock_response.status = 200
            mock_post.return_value.__aenter__.return_value = mock_response
            
            from ai_cli.llms.manager import ChatMessage
            messages = [ChatMessage(role="user", content="Hello")]
            
            response = await client.chat_completion(
                messages=messages,
                model="llama3.1"
            )
            
            assert response.content == "Test response"
            assert response.usage["prompt_tokens"] == 10
            assert response.usage["completion_tokens"] == 5
