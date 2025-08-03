"""Tests for utility tools."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import json

from ai_cli.tools.file_operations import FileOperations
from ai_cli.tools.git_integration import GitManager
from ai_cli.tools.code_analysis import CodeAnalyzer


class TestFileOperations:
    """Test cases for file operations."""
    
    @pytest.fixture
    def file_ops(self):
        """Create file operations instance for testing."""
        return FileOperations()
    
    def test_read_file_success(self, file_ops, temp_dir):
        """Test successful file reading."""
        
        # Create a test file
        test_file = temp_dir / "test.txt"
        test_content = "Hello, World!"
        test_file.write_text(test_content)
        
        content = file_ops.read_file(str(test_file))
        assert content == test_content
    
    def test_read_file_not_found(self, file_ops):
        """Test reading non-existent file."""
        
        content = file_ops.read_file("/nonexistent/file.txt")
        assert content is None
    
    def test_write_file_success(self, file_ops, temp_dir):
        """Test successful file writing."""
        
        test_file = temp_dir / "output.txt"
        test_content = "Test content"
        
        success = file_ops.write_file(str(test_file), test_content)
        assert success is True
        assert test_file.read_text() == test_content
    
    def test_backup_file(self, file_ops, temp_dir):
        """Test file backup."""
        
        # Create original file
        original_file = temp_dir / "original.txt"
        original_content = "Original content"
        original_file.write_text(original_content)
        
        backup_path = file_ops.backup_file(str(original_file))
        assert backup_path is not None
        assert Path(backup_path).exists()
        assert Path(backup_path).read_text() == original_content
    
    def test_read_json_file(self, file_ops, temp_dir):
        """Test reading JSON file."""
        
        # Create JSON file
        json_file = temp_dir / "test.json"
        test_data = {"key": "value", "number": 42}
        json_file.write_text(json.dumps(test_data))
        
        data = file_ops.read_json_file(str(json_file))
        assert data == test_data
    
    def test_write_json_file(self, file_ops, temp_dir):
        """Test writing JSON file."""
        
        json_file = temp_dir / "output.json"
        test_data = {"test": True, "items": [1, 2, 3]}
        
        success = file_ops.write_json_file(str(json_file), test_data)
        assert success is True
        
        # Verify content
        with open(json_file, 'r') as f:
            loaded_data = json.load(f)
        assert loaded_data == test_data
    
    def test_find_files(self, file_ops, temp_dir):
        """Test finding files by pattern."""
        
        # Create test files
        (temp_dir / "test1.py").write_text("# Python file 1")
        (temp_dir / "test2.py").write_text("# Python file 2")
        (temp_dir / "test.txt").write_text("Text file")
        (temp_dir / "subdir").mkdir()
        (temp_dir / "subdir" / "nested.py").write_text("# Nested Python file")
        
        # Find Python files
        python_files = file_ops.find_files(temp_dir, ["*.py"])
        assert len(python_files) >= 2
        
        # All found files should be Python files
        for file_info in python_files:
            assert file_info["path"].endswith(".py")
    
    def test_is_text_file(self, file_ops, temp_dir):
        """Test text file detection."""
        
        # Create text file
        text_file = temp_dir / "text.txt"
        text_file.write_text("This is a text file")
        
        assert file_ops.is_text_file(str(text_file)) is True
        
        # Create binary file
        binary_file = temp_dir / "binary.bin"
        binary_file.write_bytes(b"\x00\x01\x02\x03")
        
        assert file_ops.is_text_file(str(binary_file)) is False
    
    def test_get_file_info(self, file_ops, temp_dir):
        """Test getting file information."""
        
        test_file = temp_dir / "info_test.txt"
        test_content = "Test content for info"
        test_file.write_text(test_content)
        
        info = file_ops.get_file_info(str(test_file))
        assert info["exists"] is True
        assert info["size"] == len(test_content)
        assert "modified_time" in info


class TestGitManager:
    """Test cases for Git integration."""
    
    @pytest.fixture
    def git_manager(self, temp_dir):
        """Create git manager for testing."""
        return GitManager(str(temp_dir))
    
    def test_is_git_repository_false(self, git_manager):
        """Test detection of non-git directory."""
        
        assert git_manager.is_git_repository() is False
    
    @patch('subprocess.run')
    def test_get_current_branch(self, mock_run, git_manager):
        """Test getting current branch."""
        
        # Mock git command
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="main\n"
        )
        
        branch = git_manager.get_current_branch()
        assert branch == "main"
    
    @patch('subprocess.run')
    def test_get_changed_files(self, mock_run, git_manager):
        """Test getting changed files."""
        
        # Mock git status output
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="M  modified_file.py\nA  new_file.py\nD  deleted_file.py\n"
        )
        
        changed_files = git_manager.get_changed_files()
        assert len(changed_files) == 3
        
        # Check file statuses
        statuses = {f["status"] for f in changed_files}
        assert "modified" in statuses
        assert "added" in statuses
        assert "deleted" in statuses
    
    @patch('subprocess.run')
    def test_get_commit_history(self, mock_run, git_manager):
        """Test getting commit history."""
        
        # Mock git log output
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="abc123|Author Name|2024-01-01 12:00:00|Initial commit\ndef456|Author Name|2024-01-02 12:00:00|Add feature\n"
        )
        
        commits = git_manager.get_commit_history(limit=2)
        assert len(commits) == 2
        assert commits[0]["hash"] == "abc123"
        assert commits[0]["message"] == "Initial commit"


class TestCodeAnalyzer:
    """Test cases for code analysis."""
    
    @pytest.fixture
    def analyzer(self):
        """Create code analyzer for testing."""
        return CodeAnalyzer()
    
    def test_detect_language_python(self, analyzer):
        """Test Python language detection."""
        
        assert analyzer.detect_language("test.py") == "python"
        assert analyzer.detect_language("script.pyw") == "python"
    
    def test_detect_language_javascript(self, analyzer):
        """Test JavaScript language detection."""
        
        assert analyzer.detect_language("script.js") == "javascript"
        assert analyzer.detect_language("app.jsx") == "javascript"
        assert analyzer.detect_language("component.tsx") == "typescript"
    
    def test_detect_language_unknown(self, analyzer):
        """Test unknown language detection."""
        
        assert analyzer.detect_language("file.xyz") is None
        assert analyzer.detect_language("") is None
    
    def test_validate_code_python_valid(self, analyzer, sample_python_code):
        """Test validation of valid Python code."""
        
        result = analyzer.validate_code(sample_python_code, "python")
        assert result["valid"] is True
        assert len(result["errors"]) == 0
    
    def test_validate_code_python_invalid(self, analyzer):
        """Test validation of invalid Python code."""
        
        invalid_code = "def broken_function(\n    print('Missing closing parenthesis')"
        
        result = analyzer.validate_code(invalid_code, "python")
        assert result["valid"] is False
        assert len(result["errors"]) > 0
    
    def test_validate_code_javascript_valid(self, analyzer, sample_javascript_code):
        """Test validation of valid JavaScript code."""
        
        result = analyzer.validate_code(sample_javascript_code, "javascript")
        assert result["valid"] is True
        assert len(result["errors"]) == 0
    
    def test_validate_code_javascript_invalid(self, analyzer):
        """Test validation of invalid JavaScript code."""
        
        invalid_code = "function broken() { console.log('Missing closing brace'"
        
        result = analyzer.validate_code(invalid_code, "javascript")
        assert result["valid"] is False
        assert len(result["errors"]) > 0
    
    def test_analyze_complexity_python(self, analyzer, sample_python_code):
        """Test complexity analysis for Python code."""
        
        result = analyzer.analyze_complexity(sample_python_code, "python")
        assert "cyclomatic_complexity" in result
        assert "lines_of_code" in result
        assert result["lines_of_code"] > 0
    
    def test_analyze_complexity_javascript(self, analyzer, sample_javascript_code):
        """Test complexity analysis for JavaScript code."""
        
        result = analyzer.analyze_complexity(sample_javascript_code, "javascript")
        assert "cyclomatic_complexity" in result
        assert "lines_of_code" in result
        assert result["lines_of_code"] > 0
    
    def test_extract_functions_python(self, analyzer, sample_python_code):
        """Test function extraction from Python code."""
        
        functions = analyzer.extract_functions(sample_python_code, "python")
        assert len(functions) >= 2  # calculate_fibonacci and main
        
        function_names = [f["name"] for f in functions]
        assert "calculate_fibonacci" in function_names
        assert "main" in function_names
    
    def test_extract_functions_javascript(self, analyzer, sample_javascript_code):
        """Test function extraction from JavaScript code."""
        
        functions = analyzer.extract_functions(sample_javascript_code, "javascript")
        assert len(functions) >= 2  # calculateFactorial and main
        
        function_names = [f["name"] for f in functions]
        assert "calculateFactorial" in function_names
        assert "main" in function_names
