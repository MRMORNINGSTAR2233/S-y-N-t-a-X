"""Tests for agent functionality."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from ai_cli.agents.code_generator import CodeGeneratorAgent
from ai_cli.agents.debugger import DebuggerAgent
from ai_cli.agents.navigator import NavigatorAgent
from ai_cli.agents.reviewer import ReviewerAgent, ReviewIssue


class TestCodeGeneratorAgent:
    """Test cases for code generator agent."""
    
    @pytest.fixture
    def agent(self, mock_llm_manager, mock_vector_store, mock_git_manager):
        """Create code generator agent for testing."""
        return CodeGeneratorAgent(mock_llm_manager, mock_vector_store, mock_git_manager)
    
    @pytest.mark.asyncio
    async def test_generate_feature_success(self, agent):
        """Test successful feature generation."""
        
        # Mock the agent's internal methods
        with patch.object(agent, '_analyze_requirements') as mock_analyze, \
             patch.object(agent, '_gather_context') as mock_context, \
             patch.object(agent, '_generate_code') as mock_generate, \
             patch.object(agent, '_validate_code') as mock_validate, \
             patch.object(agent, '_write_files') as mock_write:
            
            # Set up mock returns
            mock_analyze.return_value = {"complexity": "medium", "files_needed": ["test.py"]}
            mock_context.return_value = {"existing_code": ""}
            mock_generate.return_value = {"test.py": "print('Hello World')"}
            mock_validate.return_value = {"valid": True, "issues": []}
            mock_write.return_value = {"files_created": ["test.py"], "files_modified": []}
            
            result = await agent.generate_feature(
                description="Create a hello world function",
                file_patterns=["*.py"]
            )
            
            assert result["success"] is True
            assert "test.py" in result["files_created"]
    
    @pytest.mark.asyncio
    async def test_generate_feature_validation_failure(self, agent):
        """Test feature generation with validation failure."""
        
        with patch.object(agent, '_analyze_requirements') as mock_analyze, \
             patch.object(agent, '_gather_context') as mock_context, \
             patch.object(agent, '_generate_code') as mock_generate, \
             patch.object(agent, '_validate_code') as mock_validate:
            
            # Set up mock returns with validation failure
            mock_analyze.return_value = {"complexity": "medium", "files_needed": ["test.py"]}
            mock_context.return_value = {"existing_code": ""}
            mock_generate.return_value = {"test.py": "invalid syntax code"}
            mock_validate.return_value = {"valid": False, "issues": ["Syntax error"]}
            
            result = await agent.generate_feature(
                description="Create invalid code",
                file_patterns=["*.py"]
            )
            
            assert result["success"] is False
            assert "validation failed" in result["error"].lower()


class TestDebuggerAgent:
    """Test cases for debugger agent."""
    
    @pytest.fixture
    def agent(self, mock_llm_manager, mock_vector_store, mock_git_manager):
        """Create debugger agent for testing."""
        return DebuggerAgent(mock_llm_manager, mock_vector_store, mock_git_manager)
    
    @pytest.mark.asyncio
    async def test_debug_issue_success(self, agent, temp_dir, sample_python_code):
        """Test successful debugging."""
        
        # Create a test file with issues
        test_file = temp_dir / "test.py"
        test_file.write_text(sample_python_code)
        
        with patch.object(agent, '_find_error_files') as mock_find, \
             patch.object(agent, '_analyze_errors') as mock_analyze, \
             patch.object(agent, '_generate_fixes') as mock_generate, \
             patch.object(agent, '_validate_fixes') as mock_validate, \
             patch.object(agent, '_apply_fixes') as mock_apply:
            
            # Set up mock returns
            mock_find.return_value = [str(test_file)]
            mock_analyze.return_value = [{"type": "performance", "description": "Inefficient recursion"}]
            mock_generate.return_value = [{"file": str(test_file), "fix": "Use memoization"}]
            mock_validate.return_value = {"valid": True, "issues": []}
            mock_apply.return_value = {"files_modified": [str(test_file)]}
            
            result = await agent.debug_issue(
                description="Fix performance issues",
                file_patterns=["*.py"]
            )
            
            assert result["success"] is True
            assert result["issues_found"] > 0
    
    @pytest.mark.asyncio
    async def test_debug_no_issues_found(self, agent):
        """Test debugging when no issues are found."""
        
        with patch.object(agent, '_find_error_files') as mock_find:
            mock_find.return_value = []
            
            result = await agent.debug_issue(
                description="Fix issues",
                file_patterns=["*.py"]
            )
            
            assert result["success"] is True
            assert result["issues_found"] == 0


class TestNavigatorAgent:
    """Test cases for navigator agent."""
    
    @pytest.fixture
    def agent(self, mock_llm_manager, mock_vector_store, mock_git_manager):
        """Create navigator agent for testing."""
        return NavigatorAgent(mock_llm_manager, mock_vector_store, mock_git_manager)
    
    @pytest.mark.asyncio
    async def test_navigate_to_symbol_found(self, agent, temp_dir, sample_python_code):
        """Test successful symbol navigation."""
        
        # Create a test file
        test_file = temp_dir / "test.py"
        test_file.write_text(sample_python_code)
        
        with patch.object(agent.file_ops, 'find_files') as mock_find:
            mock_find.return_value = [{"path": str(test_file), "size": 1000}]
            
            result = await agent.navigate_to_symbol(
                symbol_name="calculate_fibonacci",
                symbol_type="function"
            )
            
            assert result["success"] is True
            assert result["symbol_found"] is True
            assert "test.py" in result["file_path"]
    
    @pytest.mark.asyncio
    async def test_navigate_to_symbol_not_found(self, agent):
        """Test symbol navigation when symbol is not found."""
        
        with patch.object(agent.file_ops, 'find_files') as mock_find:
            mock_find.return_value = []
            
            result = await agent.navigate_to_symbol(
                symbol_name="nonexistent_function",
                symbol_type="function"
            )
            
            assert result["success"] is True
            assert result["symbol_found"] is False
    
    @pytest.mark.asyncio
    async def test_search_codebase_semantic(self, agent):
        """Test semantic codebase search."""
        
        with patch.object(agent.vector_store, 'search_similar') as mock_search:
            mock_search.return_value = [
                {"metadata": {"file_path": "test.py", "line_number": 10}, "distance": 0.3}
            ]
            
            result = await agent.search_codebase(
                query="authentication logic",
                search_type="semantic"
            )
            
            assert result["success"] is True
            assert len(result["results"]) > 0


class TestReviewerAgent:
    """Test cases for reviewer agent."""
    
    @pytest.fixture
    def agent(self, mock_llm_manager, mock_vector_store, mock_git_manager):
        """Create reviewer agent for testing."""
        return ReviewerAgent(mock_llm_manager, mock_vector_store, mock_git_manager)
    
    @pytest.mark.asyncio
    async def test_review_code_success(self, agent, temp_dir, sample_python_code):
        """Test successful code review."""
        
        # Create a test file
        test_file = temp_dir / "test.py"
        test_file.write_text(sample_python_code)
        
        with patch.object(agent, '_get_files_for_review') as mock_get_files, \
             patch.object(agent, '_review_single_file') as mock_review:
            
            mock_get_files.return_value = [str(test_file)]
            mock_review.return_value = [
                ReviewIssue(
                    type="performance",
                    severity="minor",
                    file_path=str(test_file),
                    line_number=5,
                    title="Inefficient recursion",
                    description="Fibonacci function uses inefficient recursion",
                    suggestion="Consider using memoization"
                )
            ]
            
            result = await agent.review_code(
                file_patterns=["*.py"]
            )
            
            assert result["success"] is True
            assert result["files_analyzed"] > 0
            assert result["summary"]["total_issues"] > 0
    
    @pytest.mark.asyncio
    async def test_review_code_no_files(self, agent):
        """Test code review when no files are found."""
        
        with patch.object(agent, '_get_files_for_review') as mock_get_files:
            mock_get_files.return_value = []
            
            result = await agent.review_code()
            
            assert result["success"] is False
            assert "No files found" in result["error"]
    
    def test_python_specific_checks(self, agent, temp_dir):
        """Test Python-specific review checks."""
        
        # Code with security issues
        dangerous_code = '''
import os
user_input = input("Enter command: ")
eval(user_input)  # Dangerous!
'''
        
        issues = agent._python_specific_checks("test.py", dangerous_code)
        
        # Should find the eval security issue
        security_issues = [issue for issue in issues if issue.type == "security"]
        assert len(security_issues) > 0
        assert "eval" in security_issues[0].description.lower()
    
    def test_javascript_specific_checks(self, agent, temp_dir):
        """Test JavaScript-specific review checks."""
        
        # Code with security issues
        dangerous_code = '''
const userInput = getUserInput();
eval(userInput);  // Dangerous!
document.getElementById("output").innerHTML = userInput;  // XSS risk
'''
        
        issues = agent._javascript_specific_checks("test.js", dangerous_code)
        
        # Should find security issues
        security_issues = [issue for issue in issues if issue.type == "security"]
        assert len(security_issues) > 0
    
    def test_quality_score_calculation(self, agent):
        """Test quality score calculation."""
        
        # Test with no issues
        score = agent._calculate_quality_score([], 1)
        assert score == 100.0
        
        # Test with critical issues
        critical_issue = ReviewIssue(
            type="security",
            severity="critical",
            file_path="test.py",
            line_number=1,
            title="Critical issue",
            description="Test",
            suggestion="Fix it"
        )
        
        score = agent._calculate_quality_score([critical_issue], 1)
        assert score == 80.0  # 100 - 20 for critical issue
