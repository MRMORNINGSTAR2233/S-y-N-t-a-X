"""Test configuration and fixtures."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from ai_cli.config.settings import get_settings
from ai_cli.config.database import DatabaseManager
from ai_cli.memory.vector_store import VectorStore
from ai_cli.llms.manager import LLMManager
from ai_cli.tools.git_integration import GitManager


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = get_settings()
    settings.general.debug = True
    settings.general.data_dir = Path(tempfile.mkdtemp())
    return settings


@pytest.fixture
def mock_database_manager(mock_settings):
    """Mock database manager for testing."""
    db_manager = DatabaseManager(mock_settings)
    return db_manager


@pytest.fixture
def mock_vector_store(mock_settings):
    """Mock vector store for testing."""
    vector_store = VectorStore(mock_settings)
    return vector_store


@pytest.fixture
def mock_llm_manager():
    """Mock LLM manager for testing."""
    manager = MagicMock(spec=LLMManager)
    
    # Mock async methods
    manager.get_best_model_for_task = AsyncMock(return_value="gpt-4")
    manager.chat_completion = AsyncMock()
    manager.estimate_cost = MagicMock(return_value=0.01)
    
    # Mock chat completion response
    mock_response = MagicMock()
    mock_response.content = "Test response"
    mock_response.usage = {"prompt_tokens": 100, "completion_tokens": 50}
    manager.chat_completion.return_value = mock_response
    
    return manager


@pytest.fixture
def mock_git_manager(temp_dir):
    """Mock git manager for testing."""
    git_manager = GitManager(str(temp_dir))
    return git_manager


@pytest.fixture
def sample_python_code():
    """Sample Python code for testing."""
    return '''
def calculate_fibonacci(n):
    """Calculate nth Fibonacci number."""
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

def main():
    """Main function."""
    result = calculate_fibonacci(10)
    print(f"Fibonacci number: {result}")

if __name__ == "__main__":
    main()
'''


@pytest.fixture
def sample_javascript_code():
    """Sample JavaScript code for testing."""
    return '''
function calculateFactorial(n) {
    if (n <= 0) return 1;
    return n * calculateFactorial(n - 1);
}

function main() {
    const result = calculateFactorial(5);
    console.log(`Factorial: ${result}`);
}

main();
'''
