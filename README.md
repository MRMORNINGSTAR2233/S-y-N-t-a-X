# S-y-N-t-a-X: AI-Powered Terminal CLI

**S-y-N-t-a-X** is an open-source, Python-based AI-powered Terminal CLI application inspired by Claude Code and Gemini CLI. It's designed to deeply understand and manipulate codebases, integrate with multiple LLM providers, and support agentic workflows for software development.

## 🚀 Features

### Core Capabilities
- **Multi-LLM Support**: OpenAI, Anthropic, Groq, and Ollama integration
- **Intelligent Code Generation**: Natural language to code with context awareness
- **Advanced Debugging**: AI-powered error detection and automatic fixing
- **Smart Navigation**: Semantic search and symbol navigation across codebases
- **Comprehensive Code Review**: Security, performance, style, and logic analysis
- **Git Integration**: Repository-aware operations and change tracking
- **Vector Memory**: Persistent context and learning from interactions

### Agentic Intelligence
- **Multi-step Reasoning**: LangGraph-powered workflow orchestration
- **Context Gathering**: Automatic codebase analysis and understanding
- **Self-Validation**: Code quality checks and iterative improvements
- **Adaptive Learning**: Memory of past interactions and patterns

### Developer Experience
- **Interactive Mode**: Conversational interface for iterative development
- **Rich Terminal UI**: Beautiful output with syntax highlighting
- **Flexible Configuration**: Project-specific and global settings
- **Secure API Management**: Encrypted key storage and usage tracking

## 📦 Installation

### Prerequisites
- Python 3.9 or higher
- Git (for repository integration)
- Optional: Ollama for local LLM support

### Install from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/S-y-N-t-a-X.git
cd S-y-N-t-a-X

# Install in development mode
pip install -e .

# Or install for production
pip install .
```

### Install Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# For development (includes testing dependencies)
pip install -e .[dev]
```

## 🔧 Configuration

### Initial Setup

```bash
# Configure API keys
syntax config set providers.openai.api_key "your-openai-key"
syntax config set providers.anthropic.api_key "your-anthropic-key"
syntax config set providers.groq.api_key "your-groq-key"

# Set default model
syntax config set general.default_model "gpt-4"

# Enable debugging
syntax config set general.debug true
```

### Configuration Files

The CLI uses TOML configuration files:

- **Global**: `~/.config/syntax/config.toml`
- **Project**: `.syntax/config.toml` (in project root)

Example configuration:

```toml
[general]
default_model = "gpt-4"
max_context_files = 10
debug = false
data_dir = "~/.local/share/syntax"

[providers]
[providers.openai]
api_key = "sk-..."
enabled = true
models = ["gpt-4", "gpt-3.5-turbo"]

[providers.anthropic]
api_key = "sk-ant-..."
enabled = true
models = ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"]

[memory]
enabled = true
max_entries = 1000
similarity_threshold = 0.7

[security]
encrypt_keys = true
require_confirmation = true
```

## 🎯 Usage

### Command Line Interface

#### Code Generation
```bash
# Generate a new feature
syntax generate-feature "Create a REST API endpoint for user authentication"

# Generate with specific files
syntax generate-feature "Add logging to the database module" --files "src/db/*.py"

# Generate with context
syntax generate-feature "Implement rate limiting" --context "web server, redis"
```

#### Debugging
```bash
# Debug errors in specific files
syntax debug "Fix the memory leak in the connection pool" --files "src/db/pool.py"

# Debug by error message
syntax debug "TypeError: 'NoneType' object is not subscriptable"

# Debug performance issues
syntax debug "Optimize slow query performance" --type performance
```

#### Navigation
```bash
# Find a function or class
syntax navigate function "authenticate_user"
syntax navigate class "DatabaseConnection"

# Navigate to file
syntax navigate file "user_model.py"

# Find symbol in specific scope
syntax navigate method "save" --scope "UserModel"
```

#### Search
```bash
# Semantic search
syntax search "authentication logic"

# Keyword search
syntax search "password" --type keyword

# Search in specific files
syntax search "database connection" --files "*.py"

# Search with filters
syntax search "API endpoint" --language python --modified-since "2024-01-01"
```

#### Code Review
```bash
# Review all code
syntax review

# Review specific files
syntax review --files "src/api/*.py"

# Focus on specific issues
syntax review --scope security

# Output to file
syntax review --output review_report.md
```

#### Configuration
```bash
# List all settings
syntax config list

# Get specific setting
syntax config get general.default_model

# Set configuration
syntax config set providers.openai.api_key "new-key"

# Show API usage
syntax config usage
```

### Interactive Mode

```bash
# Start interactive session
syntax interactive
```

Interactive commands:
- `generate <description>` - Generate code
- `debug <issue>` - Debug problems
- `search <query>` - Search codebase
- `navigate <symbol>` - Navigate to code
- `review` - Review code quality
- `help` - Show available commands
- `exit` - Exit interactive mode

### Advanced Usage

#### Custom Workflows
```bash
# Multi-step code generation
syntax generate-feature "Add user authentication" \
  --workflow "analyze -> design -> implement -> test -> review"

# Debug with automatic fixing
syntax debug "Performance issues" --auto-fix --backup

# Comprehensive code review with fix suggestions
syntax review --scope all --suggest-fixes --output detailed_review.md
```

#### Git Integration
```bash
# Review only changed files
syntax review --changed-only

# Generate code for specific branch
syntax generate-feature "New feature" --branch feature/auth

# Debug issues introduced in recent commits
syntax debug --since "1 week ago"
```

## 🏗️ Architecture

### Core Components

```
ai_cli/
├── main.py              # CLI entry point and command routing
├── config/              # Configuration management
│   ├── settings.py      # Pydantic settings models
│   └── database.py      # Encrypted storage
├── memory/              # Vector storage and context
│   └── vector_store.py  # ChromaDB integration
├── llms/                # LLM provider management
│   ├── manager.py       # Unified LLM interface
│   ├── openai_client.py # OpenAI integration
│   ├── anthropic_client.py # Anthropic integration
│   ├── groq_client.py   # Groq integration
│   └── ollama_client.py # Ollama integration
├── agents/              # AI agents with LangGraph
│   ├── code_generator.py # Code generation workflows
│   ├── debugger.py      # Debugging and fixing
│   ├── navigator.py     # Code navigation and search
│   └── reviewer.py      # Code review and analysis
└── tools/               # Utility tools
    ├── file_operations.py # File system operations
    ├── git_integration.py # Git repository management
    └── code_analysis.py   # Code parsing and analysis
```

### Agent Workflows

Each agent uses LangGraph for multi-step reasoning:

**Code Generator Agent:**
1. Analyze requirements
2. Gather context from codebase
3. Generate code with AI
4. Validate syntax and logic
5. Refine based on feedback

**Debugger Agent:**
1. Identify error sources
2. Analyze error patterns
3. Generate fix suggestions
4. Validate fixes
5. Apply changes with backup

**Navigator Agent:**
1. Parse query intent
2. Search vector memory
3. Analyze code structure
4. Return ranked results
5. Extract relevant context

**Reviewer Agent:**
1. Analyze code files
2. Run static analysis
3. Generate AI review
4. Categorize issues
5. Provide fix suggestions

## 🧪 Testing

### Run Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ai_cli --cov-report=html

# Run specific test categories
pytest tests/test_agents.py
pytest tests/test_llms.py
pytest tests/test_integration.py

# Run with verbose output
pytest -v --tb=short
```

### Test Structure
```
tests/
├── conftest.py          # Test configuration and fixtures
├── test_main.py         # CLI interface tests
├── test_llms.py         # LLM provider tests
├── test_agents.py       # Agent functionality tests
├── test_tools.py        # Utility tool tests
└── test_integration.py  # Full system integration tests
```

## 🔒 Security

### API Key Management
- Keys are encrypted using Fernet symmetric encryption
- Keys are stored in SQLite database with secure access
- Option to use environment variables for CI/CD
- Automatic key rotation support

### Code Safety
- Sandbox execution for generated code (optional)
- Backup creation before modifications
- Confirmation prompts for destructive operations
- Input validation and sanitization

### Privacy
- Local vector storage (no data sent to external services)
- Configurable data retention policies
- Option to disable telemetry completely

## 🤝 Contributing

### Development Setup
```bash
# Clone and install for development
git clone https://github.com/yourusername/S-y-N-t-a-X.git
cd S-y-N-t-a-X
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run linting
flake8 ai_cli/
black ai_cli/
mypy ai_cli/
```

### Code Style
- Follow PEP 8 guidelines
- Use Black for code formatting
- Add type hints for all functions
- Write comprehensive docstrings
- Include unit tests for new features

### Submitting Changes
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run the test suite
5. Submit a pull request

## 📚 Examples

### Example 1: API Development
```bash
# Generate a complete REST API
syntax generate-feature "Create a FastAPI application with user management, authentication, and CRUD operations for a blog platform"

# Review the generated code
syntax review --scope security,performance

# Debug any issues
syntax debug "Fix validation errors in user endpoints"
```

### Example 2: Refactoring
```bash
# Search for code patterns to refactor
syntax search "database queries without error handling"

# Generate improved version
syntax generate-feature "Add proper error handling and logging to database operations"

# Review changes
syntax review --changed-only
```

### Example 3: Bug Investigation
```bash
# Start with navigation
syntax navigate function "process_payment"

# Search for related issues
syntax search "payment processing errors"

# Debug the specific issue
syntax debug "Payment processor returns 500 error intermittently"
```

## 🗺️ Roadmap

### Current Version (1.0.0)
- ✅ Multi-LLM provider support
- ✅ Core agent functionality
- ✅ Basic CLI interface
- ✅ Configuration system
- ✅ Vector memory storage

### Upcoming Features (1.1.0)
- 🔄 Plugin system for custom agents
- 🔄 Web interface for visual code exploration
- 🔄 Integration with popular IDEs
- 🔄 Advanced code metrics and analytics
- 🔄 Team collaboration features

### Future Vision (2.0.0)
- 🎯 Multi-language support beyond Python
- 🎯 Cloud deployment options
- 🎯 Enterprise features and SSO
- 🎯 Advanced AI model fine-tuning
- 🎯 Integration with CI/CD pipelines

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by Claude Code and Gemini CLI
- Built with LangChain and LangGraph
- Uses ChromaDB for vector storage
- Thanks to the open-source community

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/S-y-N-t-a-X/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/S-y-N-t-a-X/discussions)
- **Email**: support@syntax-cli.dev

---

**S-y-N-t-a-X** - Empowering developers with AI-driven code intelligence.: AI-Powered Terminal CLI

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI version](https://badge.fury.io/py/syntax-ai-cli.svg)](https://badge.fury.io/py/syntax-ai-cli)

An open-source, Python-based AI-powered Terminal CLI application inspired by Claude Code and Gemini CLI, designed to deeply understand and manipulate codebases, integrate with multiple LLM providers, and support agentic workflows.

## 🎯 Core Capabilities

- **Build new features** from natural language descriptions
- **Debug and fix issues** from code and error traces
- **Navigate and search** large codebases contextually
- **Automate repetitive** and tedious development tasks
- **Support for both one-shot and multi-step** agentic workflows

## 🧠 Agentic Intelligence

- Uses LangChain and LangGraph to define modular and coordinated multi-step agents
- Understands high-level developer intent and translates into actionable code changes
- Uses agentic search and reasoning to coordinate across multiple files
- Never modifies files without explicit user approval

## 🌐 Supported LLM Providers

- **Claude** (Opus 4, Sonnet 4, Haiku 3.5)
- **OpenAI** (GPT-4, GPT-4o, GPT-3.5)
- **Groq** (ultra-fast LLaMA/Mixtral)
- **Ollama** (local models like LLaMA2, Mistral, CodeLlama)
- **OpenRouter** (multi-provider API access)
- **Grok** (xAI support)

## 🚀 Quick Start

### Installation

```bash
pip install syntax-ai-cli
```

### Basic Usage

```bash
# Interactive mode
syntax

# Generate a new feature
syntax generate-feature "Add user authentication with JWT tokens"

# Debug a file
syntax debug src/auth.py

# Search codebase
syntax search "authentication logic"

# Navigate to symbol
syntax navigate UserController

# Edit with natural language
syntax edit "Add error handling to the login function"

# Code review
syntax review
```

### Configuration

```bash
# Set up API keys
syntax config set-key openai sk-...
syntax config set-key anthropic sk-ant-...
syntax config set-key groq gsk_...

# Configure preferences
syntax config set model claude-3-sonnet
syntax config set interactive true
```

## 🧰 Core Commands

| Command | Description | Example |
|---------|-------------|---------|
| `generate-feature` | Build new features from description | `syntax generate-feature "REST API endpoint"` |
| `debug` | Debug files or error traces | `syntax debug app.py` |
| `navigate` | Find and jump to code symbols | `syntax navigate UserClass` |
| `search` | Semantic or keyword search | `syntax search "database connection"` |
| `edit` | Interactive code modifications | `syntax edit "refactor this function"` |
| `review` | AI-powered code review | `syntax review --files src/` |
| `config` | Manage settings and API keys | `syntax config list` |

## 🏗️ Project Structure

```
ai_cli/
├── main.py              # CLI entry point
├── agents/              # LangGraph agent definitions
│   ├── code_generator.py
│   ├── debugger.py
│   ├── navigator.py
│   └── reviewer.py
├── tools/               # Agent tools and utilities
│   ├── file_operations.py
│   ├── code_analysis.py
│   ├── git_integration.py
│   └── test_runner.py
├── llms/               # LLM provider integrations
│   ├── openai_client.py
│   ├── anthropic_client.py
│   ├── groq_client.py
│   └── ollama_client.py
├── config/             # Configuration management
│   ├── settings.py
│   └── database.py
├── db/                 # SQLite database models
│   └── models.py
└── memory/             # ChromaDB vector storage
    └── vector_store.py
```

## ⚙️ Configuration

### Global Configuration (`~/.syntax/config.toml`)

```toml
[general]
default_model = "claude-3-sonnet"
interactive_mode = true
auto_save = true
max_context_length = 8000

[providers]
preferred_order = ["anthropic", "openai", "groq", "ollama"]

[memory]
enable_vector_store = true
max_history_items = 1000
```

### Project Configuration (`.syntax.toml`)

```toml
[project]
name = "my-project"
language = "python"
framework = "fastapi"

[prompts]
system_prompt = "You are a Python FastAPI expert..."
coding_style = "Follow PEP 8 and use type hints"

[exclusions]
ignore_patterns = ["*.pyc", "__pycache__", ".git", "node_modules"]
```

## 🧪 Advanced Features

### Agentic Workflows

```bash
# Multi-step feature development
syntax workflow create "user-auth" \
  --steps "design,implement,test,document" \
  --review-each-step

# Custom agent chains
syntax agent-chain \
  --agents "analyzer,planner,coder,tester" \
  --goal "optimize database queries"
```

### Memory and Context

```bash
# Vector search across codebase
syntax memory search "authentication patterns"

# Context-aware suggestions
syntax suggest --context "current file and imports"

# Learning from interactions
syntax memory learn --from-session
```

### IDE Integration

```bash
# VS Code extension
syntax install vscode-extension

# JetBrains plugin
syntax install jetbrains-plugin

# LSP server mode
syntax lsp-server --port 8080
```

## 🔧 Development

### Setup Development Environment

```bash
git clone https://github.com/MRMORNINGSTAR2233/S-y-N-t-a-X.git
cd S-y-N-t-a-X
pip install -e ".[dev]"
pre-commit install
```

### Running Tests

```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# All tests with coverage
pytest --cov=ai_cli tests/
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🤝 Community

- **GitHub Discussions**: [Community Forum](https://github.com/MRMORNINGSTAR2233/S-y-N-t-a-X/discussions)
- **Issues**: [Bug Reports & Feature Requests](https://github.com/MRMORNINGSTAR2233/S-y-N-t-a-X/issues)
- **Wiki**: [Documentation](https://github.com/MRMORNINGSTAR2233/S-y-N-t-a-X/wiki)

## 🙏 Acknowledgments

- Inspired by Claude Code and Gemini CLI
- Built with LangChain and LangGraph
- Powered by multiple LLM providers
- Community-driven development

---

**Made with ❤️ by the S-y-N-t-a-X team**