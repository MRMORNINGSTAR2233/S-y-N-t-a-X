"""Integration tests for the complete system."""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path
import tempfile
import shutil

from ai_cli.main import CliContext
from ai_cli.agents.code_generator import CodeGeneratorAgent
from ai_cli.agents.debugger import DebuggerAgent
from ai_cli.agents.navigator import NavigatorAgent
from ai_cli.agents.reviewer import ReviewerAgent


class TestIntegration:
    """Integration tests for the complete system."""
    
    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory with sample files."""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create a sample Python project structure
        (temp_dir / "src").mkdir()
        (temp_dir / "tests").mkdir()
        (temp_dir / "docs").mkdir()
        
        # Add sample Python files
        (temp_dir / "src" / "__init__.py").write_text("")
        (temp_dir / "src" / "main.py").write_text('''
"""Main application module."""

def calculate_fibonacci(n):
    """Calculate nth Fibonacci number using recursion."""
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

def main():
    """Main function."""
    print("Fibonacci Calculator")
    for i in range(10):
        result = calculate_fibonacci(i)
        print(f"F({i}) = {result}")

if __name__ == "__main__":
    main()
''')
        
        (temp_dir / "src" / "utils.py").write_text('''
"""Utility functions."""

def format_output(value):
    """Format output value."""
    return f"Result: {value}"

def validate_input(value):
    """Validate input value."""
    if not isinstance(value, int):
        raise ValueError("Input must be an integer")
    if value < 0:
        raise ValueError("Input must be non-negative")
    return True
''')
        
        (temp_dir / "tests" / "test_main.py").write_text('''
"""Tests for main module."""
import unittest
from src.main import calculate_fibonacci

class TestFibonacci(unittest.TestCase):
    """Test Fibonacci calculation."""
    
    def test_fibonacci_base_cases(self):
        """Test base cases."""
        self.assertEqual(calculate_fibonacci(0), 0)
        self.assertEqual(calculate_fibonacci(1), 1)
    
    def test_fibonacci_sequence(self):
        """Test sequence values."""
        self.assertEqual(calculate_fibonacci(5), 5)
        self.assertEqual(calculate_fibonacci(10), 55)

if __name__ == "__main__":
    unittest.main()
''')
        
        (temp_dir / "README.md").write_text('''
# Sample Project

This is a sample Python project for testing the AI CLI.

## Features
- Fibonacci calculation
- Input validation
- Utility functions

## Usage
```python
python src/main.py
```
''')
        
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_full_workflow_code_generation_to_review(self, temp_project_dir, mock_llm_manager, mock_vector_store, mock_git_manager):
        """Test complete workflow from code generation to review."""
        
        # Change to temp directory for the test
        original_cwd = Path.cwd()
        import os
        os.chdir(temp_project_dir)
        
        try:
            # Initialize agents
            code_gen = CodeGeneratorAgent(mock_llm_manager, mock_vector_store, mock_git_manager)
            debugger = DebuggerAgent(mock_llm_manager, mock_vector_store, mock_git_manager)
            navigator = NavigatorAgent(mock_llm_manager, mock_vector_store, mock_git_manager)
            reviewer = ReviewerAgent(mock_llm_manager, mock_vector_store, mock_git_manager)
            
            # Mock LLM responses for realistic interaction
            mock_llm_manager.chat_completion.return_value = MagicMock(
                content='''
def memoized_fibonacci(n, memo={}):
    """Calculate nth Fibonacci number with memoization."""
    if n in memo:
        return memo[n]
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        memo[n] = memoized_fibonacci(n-1, memo) + memoized_fibonacci(n-2, memo)
        return memo[n]
''',
                usage={"prompt_tokens": 100, "completion_tokens": 150}
            )
            
            # Step 1: Generate optimized Fibonacci function
            with patch.object(code_gen.file_ops, 'write_file') as mock_write:
                mock_write.return_value = True
                
                gen_result = await code_gen.generate_feature(
                    description="Create an optimized Fibonacci function with memoization",
                    file_patterns=["src/*.py"]
                )
                
                assert gen_result["success"] is True
            
            # Step 2: Navigate to existing Fibonacci function
            nav_result = await navigator.navigate_to_symbol(
                symbol_name="calculate_fibonacci",
                symbol_type="function"
            )
            
            assert nav_result["success"] is True
            assert nav_result["symbol_found"] is True
            assert "main.py" in nav_result["file_path"]
            
            # Step 3: Search for performance-related code
            search_result = await navigator.search_codebase(
                query="fibonacci performance optimization",
                search_type="semantic"
            )
            
            assert search_result["success"] is True
            
            # Step 4: Debug performance issues
            with patch.object(debugger, '_find_error_files') as mock_find_errors:
                mock_find_errors.return_value = [str(temp_project_dir / "src" / "main.py")]
                
                debug_result = await debugger.debug_issue(
                    description="Fix performance issues in Fibonacci calculation",
                    file_patterns=["src/*.py"]
                )
                
                assert debug_result["success"] is True
            
            # Step 5: Review the entire codebase
            review_result = await reviewer.review_code(
                file_patterns=["src/*.py", "tests/*.py"]
            )
            
            assert review_result["success"] is True
            assert review_result["files_analyzed"] > 0
            
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.asyncio
    async def test_cli_context_integration(self, temp_project_dir):
        """Test CLI context integration with all components."""
        
        original_cwd = Path.cwd()
        import os
        os.chdir(temp_project_dir)
        
        try:
            with patch('ai_cli.config.settings.get_settings') as mock_settings, \
                 patch('ai_cli.config.database.DatabaseManager') as mock_db, \
                 patch('ai_cli.memory.vector_store.VectorStore') as mock_vector, \
                 patch('ai_cli.llms.manager.LLMManager') as mock_llm:
                
                # Mock all components
                mock_settings.return_value = MagicMock()
                mock_db.return_value = MagicMock()
                mock_vector.return_value = MagicMock()
                mock_llm.return_value = MagicMock()
                
                # Create CLI context
                ctx = CliContext()
                
                # Verify all components are initialized
                assert ctx.settings is not None
                assert ctx.db_manager is not None
                assert ctx.vector_store is not None
                assert ctx.llm_manager is not None
                assert ctx.git_manager is not None
                
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self, temp_project_dir, mock_llm_manager, mock_vector_store, mock_git_manager):
        """Test error handling across different components."""
        
        original_cwd = Path.cwd()
        import os
        os.chdir(temp_project_dir)
        
        try:
            # Test with LLM errors
            mock_llm_manager.chat_completion.side_effect = Exception("API Error")
            
            code_gen = CodeGeneratorAgent(mock_llm_manager, mock_vector_store, mock_git_manager)
            
            # Should handle LLM errors gracefully
            result = await code_gen.generate_feature("Test feature")
            assert result["success"] is False
            assert "error" in result
            
            # Test with file system errors
            navigator = NavigatorAgent(mock_llm_manager, mock_vector_store, mock_git_manager)
            
            with patch.object(navigator.file_ops, 'read_file', side_effect=PermissionError("Access denied")):
                result = await navigator.navigate_to_symbol("test_symbol", "function")
                assert result["success"] is True  # Should handle gracefully
                assert result["symbol_found"] is False
            
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, temp_project_dir, mock_llm_manager, mock_vector_store, mock_git_manager):
        """Test concurrent operations across agents."""
        
        original_cwd = Path.cwd()
        import os
        os.chdir(temp_project_dir)
        
        try:
            # Initialize multiple agents
            navigator = NavigatorAgent(mock_llm_manager, mock_vector_store, mock_git_manager)
            reviewer = ReviewerAgent(mock_llm_manager, mock_vector_store, mock_git_manager)
            
            # Mock responses
            mock_llm_manager.chat_completion.return_value = MagicMock(
                content="Test response",
                usage={"prompt_tokens": 50, "completion_tokens": 30}
            )
            
            # Run operations concurrently
            tasks = [
                navigator.search_codebase("fibonacci"),
                reviewer.review_code(file_patterns=["src/*.py"])
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Both operations should complete successfully
            assert len(results) == 2
            for result in results:
                assert not isinstance(result, Exception)
                assert result["success"] is True
                
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.asyncio
    async def test_memory_integration(self, temp_project_dir, mock_llm_manager, mock_git_manager):
        """Test vector store memory integration."""
        
        original_cwd = Path.cwd()
        import os
        os.chdir(temp_project_dir)
        
        try:
            # Use a real vector store for this test (mocked ChromaDB)
            with patch('chromadb.Client') as mock_chroma:
                mock_collection = MagicMock()
                mock_chroma.return_value.create_collection.return_value = mock_collection
                mock_chroma.return_value.get_collection.return_value = mock_collection
                
                from ai_cli.memory.vector_store import VectorStore
                from ai_cli.config.settings import get_settings
                
                settings = get_settings()
                vector_store = VectorStore(settings)
                
                # Test storing and retrieving code snippets
                vector_store.store_code_snippet(
                    content="def test_function(): pass",
                    file_path="test.py",
                    language="python"
                )
                
                # Mock search results
                mock_collection.query.return_value = {
                    "documents": [["def test_function(): pass"]],
                    "metadatas": [[{"file_path": "test.py", "language": "python"}]],
                    "distances": [[0.1]]
                }
                
                results = vector_store.search_similar("test function", collection_name="code_snippets")
                assert len(results) > 0
                
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.asyncio
    async def test_configuration_integration(self, temp_project_dir):
        """Test configuration system integration."""
        
        original_cwd = Path.cwd()
        import os
        os.chdir(temp_project_dir)
        
        try:
            from ai_cli.config.settings import get_settings, update_setting
            
            settings = get_settings()
            
            # Test updating settings
            original_model = settings.general.default_model
            update_setting("general.default_model", "gpt-4")
            
            # Reload settings
            updated_settings = get_settings()
            assert updated_settings.general.default_model == "gpt-4"
            
            # Restore original setting
            update_setting("general.default_model", original_model)
            
        finally:
            os.chdir(original_cwd)
