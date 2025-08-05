#!/usr/bin/env python3
"""
S-y-N-t-a-X AI CLI - Demo Interactive Interface
Simplified version for demonstration without requiring full setup
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.align import Align
from rich import box

class DemoInteractiveUI:
    """Demo Interactive terminal UI for S-y-N-t-a-X AI CLI"""
    
    def __init__(self):
        self.console = Console()
        self.session_history = []
        self.current_project_path = Path.cwd()
        self.demo_mode = True
        
    def show_banner(self):
        """Display the S-y-N-t-a-X banner"""
        banner_text = """
 ███████╗      ██╗   ██╗      ███╗   ██╗      ████████╗      █████╗       ██╗  ██╗
 ██╔════╝      ╚██╗ ██╔╝      ████╗  ██║      ╚══██╔══╝     ██╔══██╗      ╚██╗██╔╝
 ███████╗  ██╗  ╚████╔╝  ██╗  ██╔██╗ ██║ ██╗     ██║    ██╗ ███████║  ██╗  ╚███╔╝ 
 ╚════██║  ╚═╝   ╚██╔╝   ╚═╝  ██║╚██╗██║ ╚═╝     ██║    ╚═╝ ██╔══██║  ╚═╝  ██╔██╗ 
 ███████║        ██║          ██║ ╚████║         ██║        ██║  ██║       ██╔╝ ██╗
 ╚══════╝        ╚═╝          ╚═╝  ╚═══╝         ╚═╝        ╚═╝  ╚═╝       ╚═╝  ╚═╝
        """
        
        gradient_text = Text()
        for i, line in enumerate(banner_text.split('\n')):
            if line.strip():
                colors = ["blue", "cyan", "green", "yellow", "magenta", "red"]
                color = colors[i % len(colors)]
                gradient_text.append(line + '\n', style=f"bold {color}")
            else:
                gradient_text.append('\n')
        
        banner_panel = Panel(
            Align.center(gradient_text),
            title="[bold white]AI-Powered Terminal CLI[/bold white]",
            subtitle="[dim]for intelligent codebase manipulation[/dim]",
            border_style="blue",
            box=box.DOUBLE
        )
        
        self.console.print(banner_panel)
        
    def show_tips(self):
        """Show helpful tips"""
        tips_text = """
[bold green]Tips for getting started:[/bold green]
1. Ask questions, edit files, or run commands
2. Be specific for the best results  
3. Type [bold cyan]/help[/bold cyan] for more information

[bold yellow]Demo Features Available:[/bold yellow]
• Code generation examples
• Search demonstrations  
• Interactive help system
• Beautiful terminal interface
        """
        
        tips_panel = Panel(
            tips_text,
            title="[bold green]Getting Started[/bold green]",
            border_style="green"
        )
        self.console.print(tips_panel)

    def show_status(self):
        """Show current system status"""
        status_table = Table(show_header=False, box=box.SIMPLE)
        status_table.add_column("Component", style="cyan")
        status_table.add_column("Status", justify="left")
        
        # Demo status
        status_table.add_row("🎭 Mode", "[yellow]Demo Mode[/yellow]")
        status_table.add_row("🤖 LLM Providers", "[yellow]Demo responses[/yellow]")
        status_table.add_row("📁 Project", f"[blue]{self.current_project_path.name}[/blue]")
        
        # Check for actual files in project
        python_files = list(self.current_project_path.glob("**/*.py"))
        status_table.add_row("🐍 Python Files", f"[green]{len(python_files)} found[/green]")
        
        # Git status
        git_path = self.current_project_path / ".git"
        if git_path.exists():
            status_table.add_row("🌿 Git", "[green]✓ repository[/green]")
        else:
            status_table.add_row("🌿 Git", "[dim]not a git repository[/dim]")
        
        status_panel = Panel(
            status_table,
            title="[bold white]System Status[/bold white]",
            border_style="white"
        )
        self.console.print(status_panel)

    def show_help(self):
        """Show available commands"""
        help_table = Table(title="Available Commands", box=box.ROUNDED)
        help_table.add_column("Command", style="cyan", no_wrap=True)
        help_table.add_column("Description", style="white")
        help_table.add_column("Example", style="dim")
        
        commands = [
            ("/generate", "Generate code (demo)", "/generate Add logging function"),
            ("/search", "Search demo", "/search error handling"),
            ("/analyze", "Analyze current file", "/analyze main.py"),
            ("/models", "Show available models", "/models"),
            ("/status", "Show system status", "/status"),
            ("/demo", "Show demo features", "/demo"),
            ("/clear", "Clear the screen", "/clear"),
            ("/quit", "Exit interface", "/quit"),
            ("text", "Ask questions", "How do I add error handling?")
        ]
        
        for cmd, desc, example in commands:
            help_table.add_row(cmd, desc, example)
        
        self.console.print(help_table)

    def show_models(self):
        """Show available LLM models"""
        models_table = Table(title="Available AI Models", box=box.ROUNDED)
        models_table.add_column("Provider", style="cyan")
        models_table.add_column("Model", style="white")
        models_table.add_column("Context", style="green")
        models_table.add_column("Pricing", style="yellow")
        
        models = [
            ("OpenAI", "gpt-4", "8K", "$0.03/$0.06"),
            ("OpenAI", "gpt-3.5-turbo", "16K", "$0.001/$0.002"),
            ("Anthropic", "claude-3-opus", "200K", "$15/$75"),
            ("Anthropic", "claude-3-sonnet", "200K", "$3/$15"),
            ("Google", "gemini-1.5-pro", "2M", "$0.0035/$0.0105"),
            ("Google", "gemini-1.5-flash", "1M", "$0.00015/$0.0006"),
            ("Google", "gemini-1.0-pro", "32K", "$0.0005/$0.0015"),
            ("Groq", "mixtral-8x7b", "32K", "$0.00027/$0.00027"),
            ("Groq", "llama2-70b", "4K", "$0.0007/$0.0008"),
            ("Ollama", "codellama", "Local", "Free"),
        ]
        
        for provider, model, context, price in models:
            models_table.add_row(provider, model, context, price)
        
        self.console.print(models_table)

    async def demo_generate(self, description: str):
        """Demo code generation"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Generating code...", total=None)
            
            # Simulate processing time
            await asyncio.sleep(2)
            progress.update(task, completed=True)
        
        # Generate demo code based on description
        if "logging" in description.lower():
            demo_code = '''import logging

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def main():
    """Main function with logging"""
    logger.info("Application started")
    try:
        # Your code here
        logger.info("Processing completed successfully")
    except Exception as e:
        logger.error(f"Error occurred: {e}")
    finally:
        logger.info("Application finished")'''
        
        elif "error" in description.lower():
            demo_code = '''def safe_function(data):
    """Function with comprehensive error handling"""
    try:
        if not data:
            raise ValueError("Data cannot be empty")
        
        # Process data
        result = process_data(data)
        return result
        
    except ValueError as e:
        print(f"Validation error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
    finally:
        # Cleanup code here
        pass'''
        
        elif "class" in description.lower():
            demo_code = '''class DataProcessor:
    """A sample data processing class"""
    
    def __init__(self, name: str):
        self.name = name
        self.processed_count = 0
    
    def process(self, data):
        """Process incoming data"""
        if not data:
            raise ValueError("No data provided")
        
        # Processing logic here
        self.processed_count += 1
        return f"Processed {data} by {self.name}"
    
    def get_stats(self):
        """Get processing statistics"""
        return {
            "processor": self.name,
            "items_processed": self.processed_count
        }'''
        
        else:
            demo_code = f'''# Generated code for: {description}

def hello_world():
    """A simple hello world function"""
    print("Hello, World!")
    return "Success"

if __name__ == "__main__":
    hello_world()'''
        
        # Display the generated code
        code_panel = Panel(
            Syntax(demo_code, "python", theme="monokai"),
            title=f"[bold green]Generated Code: {description}[/bold green]",
            border_style="green"
        )
        self.console.print(code_panel)
        
        # Ask if user wants to save
        if Confirm.ask("💾 Save to file?"):
            filename = Prompt.ask("📝 Filename", default="generated_code.py")
            file_path = self.current_project_path / filename
            file_path.write_text(demo_code)
            self.console.print(f"[green]✅ Saved to {filename}[/green]")

    async def demo_search(self, query: str):
        """Demo search functionality"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Searching...", total=None)
            await asyncio.sleep(1.5)
            progress.update(task, completed=True)
        
        # Demo search results
        results_table = Table(title=f"🔍 Search Results for: {query}", box=box.ROUNDED)
        results_table.add_column("File", style="cyan")
        results_table.add_column("Match", style="white")
        results_table.add_column("Score", style="green")
        
        demo_results = [
            ("main.py", f"Found '{query}' in main function definition", "0.95"),
            ("utils.py", f"Helper function mentions '{query}'", "0.87"),
            ("models.py", f"Class documentation contains '{query}'", "0.73"),
            ("README.md", f"Documentation section about '{query}'", "0.65"),
        ]
        
        for file_name, content, score in demo_results:
            results_table.add_row(file_name, content, score)
        
        self.console.print(results_table)

    def demo_analyze(self, filename: str = None):
        """Demo code analysis"""
        if not filename:
            # Analyze current directory
            python_files = list(self.current_project_path.glob("*.py"))
            if python_files:
                filename = python_files[0].name
            else:
                filename = "main.py"
        
        analysis_table = Table(title=f"📊 Code Analysis: {filename}", box=box.ROUNDED)
        analysis_table.add_column("Metric", style="cyan")
        analysis_table.add_column("Value", style="white")
        analysis_table.add_column("Status", style="green")
        
        analysis_results = [
            ("Lines of Code", "42", "✅ Good"),
            ("Functions", "3", "✅ Well structured"),
            ("Classes", "1", "✅ Appropriate"),
            ("Complexity", "Low", "✅ Maintainable"),
            ("Test Coverage", "85%", "✅ Well tested"),
            ("Code Quality", "A", "✅ Excellent"),
        ]
        
        for metric, value, status in analysis_results:
            analysis_table.add_row(metric, value, status)
        
        self.console.print(analysis_table)

    def show_demo_features(self):
        """Show what's available in demo mode"""
        demo_panel = Panel(
            """[bold cyan]🎭 Demo Mode Features:[/bold cyan]

[green]✅ Available:[/green]
• Beautiful terminal interface
• Interactive command system  
• Code generation examples
• Search demonstrations
• File analysis mockups
• Model information display
• Project status overview

[yellow]⚠ Demo Only:[/yellow]
• Generated code is templated
• Search results are simulated
• No actual AI model calls
• Analysis data is mocked

[blue]💡 To enable full features:[/blue]
1. Install dependencies: pip install -r requirements.txt
2. Configure API keys: syntax config set-key openai YOUR_KEY
3. Use the full UI: syntax-ui

[bold]Try these demo commands:[/bold]
• /generate Add a login function
• /search authentication 
• /analyze main.py
• /models
            """,
            title="[bold yellow]Demo Mode Information[/bold yellow]",
            border_style="yellow"
        )
        self.console.print(demo_panel)

    async def handle_natural_query(self, query: str):
        """Handle natural language questions with demo responses"""
        demo_responses = {
            "logging": "To add logging to your Python application, use the built-in `logging` module. Set up a logger with `logging.getLogger(__name__)` and configure the format and level.",
            "error": "For error handling, use try-except blocks. Be specific with exception types, log errors appropriately, and always clean up resources in finally blocks.",
            "class": "Python classes should follow PEP 8 naming conventions (PascalCase), include docstrings, and have clear single responsibilities. Use `__init__` for initialization.",
            "function": "Functions should be small, focused, and have clear names. Include type hints and docstrings. Follow the single responsibility principle.",
            "testing": "Use pytest for testing Python code. Write unit tests, integration tests, and use fixtures for setup. Aim for high test coverage.",
        }
        
        # Find relevant response
        response = "I'm in demo mode, so I can provide general guidance. " + demo_responses.get(
            next((key for key in demo_responses if key in query.lower()), "general"),
            "For specific coding help, configure API keys to enable full AI assistance."
        )
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Thinking...", total=None)
            await asyncio.sleep(1)
            progress.update(task, completed=True)
        
        response_panel = Panel(
            Markdown(response),
            title="[bold blue]🤖 AI Assistant (Demo)[/bold blue]",
            border_style="blue"
        )
        self.console.print(response_panel)

    async def process_command(self, user_input: str):
        """Process user commands"""
        if user_input.startswith("/"):
            parts = user_input[1:].split(" ", 1)
            command = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            
            if command == "help":
                self.show_help()
            elif command == "status":
                self.show_status()
            elif command == "models":
                self.show_models()
            elif command == "demo":
                self.show_demo_features()
            elif command == "clear":
                self.console.clear()
                self.show_banner()
            elif command == "quit" or command == "exit":
                return False
            elif command == "generate":
                await self.demo_generate(args or "sample function")
            elif command == "search":
                await self.demo_search(args or "sample query")
            elif command == "analyze":
                self.demo_analyze(args or None)
            else:
                self.console.print(f"[red]❌ Unknown command: {command}[/red]")
                self.console.print("Type [bold]/help[/bold] for available commands")
        else:
            # Handle natural language queries
            await self.handle_natural_query(user_input)
        
        return True

    async def run(self):
        """Main interactive loop"""
        self.console.clear()
        self.show_banner()
        self.show_tips()
        self.show_status()
        
        self.console.print("\n[bold green]🚀 S-y-N-t-a-X Demo Interface Ready![/bold green]")
        self.console.print("[dim]Type /help for commands or ask me anything![/dim]\n")
        
        while True:
            try:
                user_input = Prompt.ask(
                    "[bold blue]❯[/bold blue]",
                    default="",
                    show_default=False
                ).strip()
                
                if not user_input:
                    continue
                
                self.session_history.append(user_input)
                should_continue = await self.process_command(user_input)
                if not should_continue:
                    break
                    
                self.console.print()
                
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Use /quit to exit[/yellow]")
            except EOFError:
                break
        
        self.console.print("\n[bold blue]👋 Thanks for trying S-y-N-t-a-X AI CLI![/bold blue]")

def main():
    """Entry point for demo UI"""
    try:
        ui = DemoInteractiveUI()
        asyncio.run(ui.run())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
