<div align="center">
  <img src="banner.png" alt="S-y-N-t-a-X Logo" width="600"/>
</div>

# S-y-N-t-a-X: AI-Powered Terminal CLI

**S-y-N-t-a-X** is an open-source, Python-based AI-powered Terminal CLI application inspired by Claude Code and Gemini CLI. It's designed to deeply understand and manipulate codebases, integrate with multiple LLM providers, and support agentic workflows for software development.

> **Getting Started:** After installation, simply run `syntax` to launch the interactive CLI!

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
syntax
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
├── agents/              # AI agents with LangGraph workflows
│   ├── autonomous_agent.py # ✅ Advanced autonomous agent with task planning
│   ├── code_generator.py   # ✅ Code generation workflows
│   ├── debugger.py         # ✅ Debugging and fixing workflows
│   ├── navigator.py        # ✅ Code navigation and search
│   └── reviewer.py         # ✅ Code review and analysis
├── llms/                # LLM provider management
│   ├── manager.py          # ✅ Unified LLM interface with provider selection
│   ├── base.py             # ✅ Base LLM client interface
│   ├── openai_client.py    # ✅ OpenAI GPT integration
│   ├── anthropic_client.py # ✅ Anthropic Claude integration
│   ├── gemini_client.py    # ✅ Google Gemini integration
│   ├── groq_client.py      # ✅ Groq integration
│   └── ollama_client.py    # ✅ Ollama local models integration
├── ui/                  # User interface components
│   ├── interactive.py      # ✅ Rich terminal interactive interface
│   ├── llm_config.py       # ✅ LLM provider configuration UI
│   └── demo.py             # ✅ Demo interface
├── config/              # Configuration management
│   ├── settings.py         # ✅ Pydantic settings models
│   └── database.py         # ✅ Encrypted storage
├── memory/              # Vector storage and context
│   └── vector_store.py     # ✅ ChromaDB integration
├── tools/               # Utility tools
│   ├── file_operations.py  # ✅ File system operations
│   ├── git_integration.py  # ✅ Git repository management
│   └── code_analysis.py    # ✅ Code parsing and analysis
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

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**S-y-N-t-a-X** - Empowering developers with AI-driven code intelligence.
