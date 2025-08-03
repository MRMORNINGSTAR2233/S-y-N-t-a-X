"""Tests for the main CLI module."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from click.testing import CliRunner

from ai_cli.main import cli, CliContext


class TestCLI:
    """Test cases for the main CLI."""
    
    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()
    
    def test_cli_help(self):
        """Test that CLI help works."""
        result = self.runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'AI-powered terminal CLI' in result.output
    
    def test_cli_version(self):
        """Test version command."""
        result = self.runner.invoke(cli, ['--version'])
        assert result.exit_code == 0
        assert '1.0.0' in result.output
    
    @patch('ai_cli.main.DatabaseManager')
    @patch('ai_cli.main.VectorStore')  
    @patch('ai_cli.main.LLMManager')
    def test_cli_context_initialization(self, mock_llm, mock_vector, mock_db):
        """Test CLI context initialization."""
        
        # Mock the managers
        mock_db_instance = MagicMock()
        mock_vector_instance = MagicMock()
        mock_llm_instance = MagicMock()
        
        mock_db.return_value = mock_db_instance
        mock_vector.return_value = mock_vector_instance
        mock_llm.return_value = mock_llm_instance
        
        # Create CLI context
        ctx = CliContext()
        
        # Verify managers are created
        assert ctx.db_manager is not None
        assert ctx.vector_store is not None
        assert ctx.llm_manager is not None
    
    @patch('ai_cli.main.CodeGeneratorAgent')
    def test_generate_feature_command(self, mock_agent_class):
        """Test generate-feature command."""
        
        # Mock the agent
        mock_agent = MagicMock()
        mock_agent.generate_feature = AsyncMock(return_value={
            "success": True,
            "files_created": ["test.py"],
            "files_modified": []
        })
        mock_agent_class.return_value = mock_agent
        
        # Run command
        result = self.runner.invoke(cli, [
            'generate-feature',
            'Create a simple calculator function'
        ])
        
        # Check that command executed without error
        assert result.exit_code == 0
    
    @patch('ai_cli.main.DebuggerAgent')
    def test_debug_command(self, mock_agent_class):
        """Test debug command."""
        
        # Mock the agent
        mock_agent = MagicMock()
        mock_agent.debug_issue = AsyncMock(return_value={
            "success": True,
            "issues_found": 1,
            "fixes_applied": 1
        })
        mock_agent_class.return_value = mock_agent
        
        # Run command
        result = self.runner.invoke(cli, [
            'debug',
            'Fix syntax errors in main.py'
        ])
        
        # Check that command executed without error
        assert result.exit_code == 0
    
    @patch('ai_cli.main.NavigatorAgent')
    def test_navigate_command(self, mock_agent_class):
        """Test navigate command."""
        
        # Mock the agent
        mock_agent = MagicMock()
        mock_agent.navigate_to_symbol = AsyncMock(return_value={
            "success": True,
            "symbol_found": True,
            "file_path": "test.py",
            "line_number": 10
        })
        mock_agent_class.return_value = mock_agent
        
        # Run command
        result = self.runner.invoke(cli, [
            'navigate',
            'find_function',
            'test_function'
        ])
        
        # Check that command executed without error
        assert result.exit_code == 0
    
    @patch('ai_cli.main.NavigatorAgent')  
    def test_search_command(self, mock_agent_class):
        """Test search command."""
        
        # Mock the agent
        mock_agent = MagicMock()
        mock_agent.search_codebase = AsyncMock(return_value={
            "success": True,
            "results": [
                {"file_path": "test.py", "line_number": 5, "context": "test code"}
            ]
        })
        mock_agent_class.return_value = mock_agent
        
        # Run command
        result = self.runner.invoke(cli, [
            'search',
            'authentication logic'
        ])
        
        # Check that command executed without error
        assert result.exit_code == 0
    
    @patch('ai_cli.main.ReviewerAgent')
    def test_review_command(self, mock_agent_class):
        """Test review command."""
        
        # Mock the agent
        mock_agent = MagicMock()
        mock_agent.review_code = AsyncMock(return_value={
            "success": True,
            "files_analyzed": 5,
            "total_issues": 10,
            "quality_score": 85.0
        })
        mock_agent_class.return_value = mock_agent
        
        # Run command
        result = self.runner.invoke(cli, ['review'])
        
        # Check that command executed without error
        assert result.exit_code == 0
    
    def test_config_list_command(self):
        """Test config list command."""
        result = self.runner.invoke(cli, ['config', 'list'])
        assert result.exit_code == 0
    
    @patch('ai_cli.main.get_settings')
    def test_config_set_command(self, mock_get_settings):
        """Test config set command."""
        
        # Mock settings
        mock_settings = MagicMock()
        mock_get_settings.return_value = mock_settings
        
        result = self.runner.invoke(cli, [
            'config', 'set',
            'general.default_model',
            'gpt-4'
        ])
        
        # Should succeed (even if it doesn't actually save in test)
        assert result.exit_code == 0


class TestInteractiveMode:
    """Test cases for interactive mode."""
    
    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()
    
    @patch('ai_cli.main.get_input')
    @patch('ai_cli.main.CodeGeneratorAgent')
    def test_interactive_mode_generate(self, mock_agent_class, mock_input):
        """Test interactive mode generation."""
        
        # Mock user input
        mock_input.side_effect = [
            'generate',
            'Create a hello world function',
            'exit'
        ]
        
        # Mock agent
        mock_agent = MagicMock()
        mock_agent.generate_feature = AsyncMock(return_value={
            "success": True,
            "files_created": ["hello.py"]
        })
        mock_agent_class.return_value = mock_agent
        
        # Run interactive mode
        result = self.runner.invoke(cli, ['interactive'])
        
        # Should exit cleanly
        assert result.exit_code == 0
    
    @patch('ai_cli.main.get_input')
    def test_interactive_mode_help(self, mock_input):
        """Test interactive mode help."""
        
        # Mock user input
        mock_input.side_effect = ['help', 'exit']
        
        # Run interactive mode
        result = self.runner.invoke(cli, ['interactive'])
        
        # Should show help and exit cleanly
        assert result.exit_code == 0
        assert 'Available commands:' in result.output
    
    @patch('ai_cli.main.get_input')
    def test_interactive_mode_exit(self, mock_input):
        """Test interactive mode exit."""
        
        # Mock user input
        mock_input.side_effect = ['exit']
        
        # Run interactive mode
        result = self.runner.invoke(cli, ['interactive'])
        
        # Should exit cleanly
        assert result.exit_code == 0
