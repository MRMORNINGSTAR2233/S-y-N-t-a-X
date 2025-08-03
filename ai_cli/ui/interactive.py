#!/usr/bin/env python3
"""
Interactive Terminal UI for S-y-N-t-a-X AI CLI
Inspired by Gemini CLI interface with rich terminal features
"""

import asyncio
import os
import sys
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

# Rich imports for beautiful terminal UI
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.layout import Layout
from rich.live import Live
from rich.align import Align
from rich import box

# Our AI CLI imports
from ai_cli.llms.manager import LLMManager
from ai_cli.agents.code_generator import CodeGeneratorAgent
from ai_cli.agents.reviewer import ReviewerAgent
from ai_cli.memory.vector_store import VectorStore
from ai_cli.config.settings import Settings
from ai_cli.tools.file_operations import FileOperations


# Import demo UI from the project root
try:
    demo_ui_path = Path(__file__).parent.parent.parent / "demo_ui.py"
    if demo_ui_path.exists():
        sys.path.insert(0, str(demo_ui_path.parent))
        from demo_ui import DemoInteractiveUI
    else:
        DemoInteractiveUI = None
except ImportError:
    DemoInteractiveUI = None


class InteractiveUI:
    """Interactive terminal UI for S-y-N-t-a-X AI CLI"""
    
    def __init__(self):
        self.console = Console()
        self.settings = Settings()
        self.llm_manager = LLMManager()
        self.file_ops = FileOperations()
        self.code_generator = None
        self.code_reviewer = None
        self.vector_store = VectorStore()
        self.session_history = []
        self.current_project_path = Path.cwd()
        
    async def initialize(self):
        """Initialize all components"""
        try:
            # Initialize agents if LLM is available
            if self.llm_manager.clients:
                # Import required dependencies for agents
                from ai_cli.tools.git_integration import GitManager
                
                # Initialize Git manager
                git_manager = GitManager(self.current_project_path)
                
                # Initialize agents with all required parameters
                self.code_generator = CodeGeneratorAgent(
                    llm_manager=self.llm_manager,
                    vector_store=self.vector_store,
                    git_manager=git_manager
                )
                self.code_reviewer = ReviewerAgent(
                    llm_manager=self.llm_manager,
                    vector_store=self.vector_store,
                    git_manager=git_manager
                )
            return True
        except Exception as e:
            self.console.print(f"[red]Error initializing: {e}[/red]")
            # Continue without agents if initialization fails
            return True

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
        
        banner_panel = Panel(
            Align.center(Text(banner_text, style="bold blue")),
            title="[bold white]AI-Powered Terminal CLI[/bold white]",
            subtitle="[dim]for intelligent codebase manipulation[/dim]",
            border_style="blue",
            box=box.DOUBLE
        )
        
        self.console.print(banner_panel)
        
    def show_tips(self):
        """Show helpful tips for getting started"""
        tips = [
            "1. Ask questions, edit files, or run commands",
            "2. Use file operations: /read, /write, /update, /delete",
            "3. Browse with /list, /tree, /info for file management",
            "4. Be specific for the best results",
            "5. Type /help for more information"
        ]
        
        tips_text = "\n".join(tips)
        tips_panel = Panel(
            tips_text,
            title="[bold green]Tips for getting started[/bold green]",
            border_style="green"
        )
        self.console.print(tips_panel)

    def show_status(self):
        """Show current system status"""
        # Create status table
        status_table = Table(show_header=False, box=box.SIMPLE)
        status_table.add_column("Component", style="cyan")
        status_table.add_column("Status", justify="left")
        
        # Check LLM providers
        provider_count = len(self.llm_manager.clients)
        if provider_count > 0:
            status_table.add_row("🤖 LLM Providers", f"[green]{provider_count} active[/green]")
            for provider in self.llm_manager.clients.keys():
                status_table.add_row(f"  └─ {provider}", "[green]✓ ready[/green]")
        else:
            status_table.add_row("🤖 LLM Providers", "[yellow]⚠ No API keys configured[/yellow]")
        
        # Project info
        status_table.add_row("📁 Project", f"[blue]{self.current_project_path.name}[/blue]")
        
        # Git status
        git_path = self.current_project_path / ".git"
        if git_path.exists():
            status_table.add_row("🌿 Git", "[green]✓ repository[/green]")
        else:
            status_table.add_row("🌿 Git", "[dim]not a git repository[/dim]")
        
        # File operations
        status_table.add_row("📁 File Operations", "[green]✓ ready[/green]")
        
        # Count files in current directory
        try:
            file_count = len([f for f in self.current_project_path.iterdir() if f.is_file()])
            dir_count = len([d for d in self.current_project_path.iterdir() if d.is_dir() and not d.name.startswith('.')])
            status_table.add_row("  └─ Current Dir", f"[blue]{file_count} files, {dir_count} directories[/blue]")
        except:
            status_table.add_row("  └─ Current Dir", "[dim]unable to scan[/dim]")
        
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
            ("/generate", "Generate new code features", "/generate Add logging to main function"),
            ("/edit", "Edit existing code", "/edit Add error handling to utils.py"),
            ("/review", "Review code for issues", "/review Check main.py for bugs"),
            ("/search", "Search codebase", "/search error handling"),
            ("/read", "Read file content", "/read src/main.py"),
            ("/write", "Write content to file", "/write test.py 'print(\"hello\")'"),
            ("/update", "Update/modify file content", "/update src/main.py line 10 'new content'"),
            ("/delete", "Delete file or directory", "/delete old_file.py"),
            ("/copy", "Copy file", "/copy src/old.py src/new.py"),
            ("/move", "Move/rename file", "/move old.py new.py"),
            ("/list", "List files in directory", "/list src/ *.py"),
            ("/info", "Get file information", "/info main.py"),
            ("/tree", "Show directory structure", "/tree src/"),
            ("/config", "Configure settings", "/config set provider openai"),
            ("/status", "Show system status", "/status"),
            ("/clear", "Clear the screen", "/clear"),
            ("/quit", "Exit the interface", "/quit"),
            ("text", "Natural language queries", "How do I add logging to this function?")
        ]
        
        for cmd, desc, example in commands:
            help_table.add_row(cmd, desc, example)
        
        self.console.print(help_table)

    async def handle_generate_command(self, args: str):
        """Handle code generation command"""
        if not self.code_generator:
            self.console.print("[red]❌ No LLM providers available. Configure API keys first.[/red]")
            return
        
        # Show progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Generating code...", total=None)
            
            try:
                # Simple generation for demo - in real implementation would use the full workflow
                result = await self.code_generator.generate_feature(
                    description=args,
                    language="python",
                    project_context=str(self.current_project_path)
                )
                
                progress.update(task, completed=True)
                
                # Display result
                if hasattr(result, 'final_result') and result.final_result:
                    code_panel = Panel(
                        Syntax(result.final_result.get('code', 'No code generated'), "python"),
                        title="[bold green]Generated Code[/bold green]",
                        border_style="green"
                    )
                    self.console.print(code_panel)
                    
                    # Ask if user wants to save
                    if Confirm.ask("Save to file?"):
                        filename = Prompt.ask("Filename", default="generated_code.py")
                        file_path = self.current_project_path / filename
                        file_path.write_text(result.final_result.get('code', ''))
                        self.console.print(f"[green]✓ Saved to {filename}[/green]")
                else:
                    self.console.print("[yellow]No code was generated[/yellow]")
                    
            except Exception as e:
                progress.update(task, completed=True)
                self.console.print(f"[red]❌ Error generating code: {e}[/red]")

    async def handle_search_command(self, query: str):
        """Handle search command"""
        if not self.vector_store:
            self.console.print("[red]❌ Vector store not available[/red]")
            return
            
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Searching...", total=None)
            
            try:
                results = self.vector_store.search_code(
                    query=query,
                    n_results=5
                )
                
                progress.update(task, completed=True)
                
                if results:
                    results_table = Table(title=f"Search Results for: {query}", box=box.ROUNDED)
                    results_table.add_column("File", style="cyan")
                    results_table.add_column("Content", style="white")
                    results_table.add_column("Similarity", style="green")
                    
                    for result in results:
                        file_name = result.get('metadata', {}).get('file_path', 'Unknown')
                        content = result.get('code', '')[:100] + "..." if len(result.get('code', '')) > 100 else result.get('code', '')
                        similarity = f"{result.get('similarity', 0):.2f}"
                        results_table.add_row(file_name, content, similarity)
                    
                    self.console.print(results_table)
                else:
                    self.console.print("[yellow]No results found[/yellow]")
                    
            except Exception as e:
                progress.update(task, completed=True)
                self.console.print(f"[red]❌ Search error: {e}[/red]")

    async def handle_config_command(self, args: str):
        """Handle configuration command"""
        parts = args.split()
        if len(parts) < 2:
            self.console.print("[red]Usage: /config set <key> <value> or /config get <key>[/red]")
            return
        
        action = parts[0]
        key = parts[1]
        
        if action == "set" and len(parts) >= 3:
            value = " ".join(parts[2:])
            
            if key == "provider":
                # Handle provider setting
                self.console.print(f"[green]✓ Set default provider to: {value}[/green]")
            elif key.endswith("_api_key"):
                # Handle API key setting
                self.settings.set_api_key(key.replace("_api_key", ""), value)
                self.console.print(f"[green]✓ API key set for {key.replace('_api_key', '')}[/green]")
                # Reinitialize LLM manager
                await self.initialize()
            else:
                self.console.print(f"[green]✓ Set {key} = {value}[/green]")
                
        elif action == "get":
            # Show current configuration
            config_table = Table(title="Current Configuration", box=box.ROUNDED)
            config_table.add_column("Setting", style="cyan")
            config_table.add_column("Value", style="white")
            
            config_table.add_row("Project Path", str(self.current_project_path))
            config_table.add_row("LLM Providers", str(len(self.llm_manager.clients)))
            
            self.console.print(config_table)

    async def handle_read_command(self, args: str):
        """Handle file read command"""
        if not args.strip():
            self.console.print("[red]Usage: /read <file_path>[/red]")
            return
        
        file_path = Path(args.strip())
        if not file_path.is_absolute():
            file_path = self.current_project_path / file_path
        
        if not file_path.exists():
            self.console.print(f"[red]❌ File not found: {file_path}[/red]")
            return
        
        if not file_path.is_file():
            self.console.print(f"[red]❌ Path is not a file: {file_path}[/red]")
            return
        
        try:
            content = self.file_ops.read_file(file_path)
            if content is not None:
                # Determine language for syntax highlighting
                language = "text"
                if file_path.suffix:
                    ext_to_lang = {
                        '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
                        '.java': 'java', '.cpp': 'cpp', '.c': 'c', '.h': 'c',
                        '.css': 'css', '.html': 'html', '.xml': 'xml',
                        '.json': 'json', '.yaml': 'yaml', '.yml': 'yaml',
                        '.md': 'markdown', '.sql': 'sql', '.sh': 'bash',
                        '.go': 'go', '.rs': 'rust', '.php': 'php'
                    }
                    language = ext_to_lang.get(file_path.suffix.lower(), "text")
                
                # Show file info
                file_info = self.file_ops.get_file_info(file_path)
                info_text = f"📁 {file_path.name} | {file_info.get('size_human', 'Unknown size')} | {file_info.get('line_count', 0)} lines"
                
                self.console.print(f"[dim]{info_text}[/dim]")
                
                # Display content with syntax highlighting
                content_panel = Panel(
                    Syntax(content, language, theme="monokai", line_numbers=True),
                    title=f"[bold green]📖 {file_path.name}[/bold green]",
                    border_style="green"
                )
                self.console.print(content_panel)
            else:
                self.console.print(f"[red]❌ Could not read file: {file_path}[/red]")
        except Exception as e:
            self.console.print(f"[red]❌ Error reading file: {e}[/red]")

    async def handle_write_command(self, args: str):
        """Handle file write command"""
        parts = args.split(" ", 1)
        if len(parts) < 2:
            self.console.print("[red]Usage: /write <file_path> '<content>' or /write <file_path> (for interactive input)[/red]")
            return
        
        file_path = Path(parts[0].strip())
        if not file_path.is_absolute():
            file_path = self.current_project_path / file_path
        
        # Check if content is provided or need interactive input
        if len(parts) == 1 or not parts[1].strip():
            # Interactive input
            self.console.print(f"[cyan]Enter content for {file_path.name} (Ctrl+D when done):[/cyan]")
            content_lines = []
            try:
                while True:
                    line = input()
                    content_lines.append(line)
            except EOFError:
                content = "\n".join(content_lines)
        else:
            content = parts[1].strip()
            # Remove surrounding quotes if present
            if (content.startswith('"') and content.endswith('"')) or \
               (content.startswith("'") and content.endswith("'")):
                content = content[1:-1]
        
        # Check if file exists and ask for confirmation
        if file_path.exists():
            if not Confirm.ask(f"File {file_path.name} exists. Overwrite?"):
                self.console.print("[yellow]Operation cancelled[/yellow]")
                return
        
        success = self.file_ops.write_file(file_path, content, backup=True)
        if success:
            file_info = self.file_ops.get_file_info(file_path)
            self.console.print(f"[green]✓ File written successfully: {file_path.name} ({file_info.get('size_human', 'Unknown size')})[/green]")
        else:
            self.console.print(f"[red]❌ Failed to write file: {file_path}[/red]")

    async def handle_update_command(self, args: str):
        """Handle file update/modification command"""
        parts = args.split(" ", 2)
        if len(parts) < 3:
            self.console.print("[red]Usage: /update <file_path> line <line_number> '<new_content>' or /update <file_path> append '<content>'[/red]")
            return
        
        file_path = Path(parts[0].strip())
        if not file_path.is_absolute():
            file_path = self.current_project_path / file_path
        
        if not file_path.exists() or not file_path.is_file():
            self.console.print(f"[red]❌ File not found: {file_path}[/red]")
            return
        
        operation = parts[1].strip().lower()
        content = parts[2].strip()
        
        # Remove surrounding quotes if present
        if (content.startswith('"') and content.endswith('"')) or \
           (content.startswith("'") and content.endswith("'")):
            content = content[1:-1]
        
        try:
            if operation == "append":
                success = self.file_ops.append_to_file(file_path, "\n" + content)
                if success:
                    self.console.print(f"[green]✓ Content appended to {file_path.name}[/green]")
                else:
                    self.console.print(f"[red]❌ Failed to append to file[/red]")
            
            elif operation == "line":
                # This is a more complex operation - read, modify specific line, write back
                current_content = self.file_ops.read_file(file_path)
                if current_content is None:
                    self.console.print(f"[red]❌ Could not read file for updating[/red]")
                    return
                
                try:
                    line_num = int(content.split("'")[0].strip())
                    new_line_content = content.split("'", 1)[1].strip("'")
                    
                    lines = current_content.split('\n')
                    if 1 <= line_num <= len(lines):
                        lines[line_num - 1] = new_line_content
                        updated_content = '\n'.join(lines)
                        
                        success = self.file_ops.write_file(file_path, updated_content, backup=True)
                        if success:
                            self.console.print(f"[green]✓ Line {line_num} updated in {file_path.name}[/green]")
                        else:
                            self.console.print(f"[red]❌ Failed to update file[/red]")
                    else:
                        self.console.print(f"[red]❌ Line number {line_num} out of range (1-{len(lines)})[/red]")
                        
                except (ValueError, IndexError):
                    self.console.print("[red]❌ Invalid line number or format[/red]")
            else:
                self.console.print(f"[red]❌ Unknown update operation: {operation}[/red]")
                self.console.print("[yellow]Supported operations: append, line[/yellow]")
                
        except Exception as e:
            self.console.print(f"[red]❌ Error updating file: {e}[/red]")

    async def handle_delete_command(self, args: str):
        """Handle file/directory deletion command"""
        if not args.strip():
            self.console.print("[red]Usage: /delete <file_or_directory_path>[/red]")
            return
        
        target_path = Path(args.strip())
        if not target_path.is_absolute():
            target_path = self.current_project_path / target_path
        
        if not target_path.exists():
            self.console.print(f"[red]❌ Path not found: {target_path}[/red]")
            return
        
        # Show what will be deleted
        if target_path.is_file():
            file_info = self.file_ops.get_file_info(target_path)
            self.console.print(f"[yellow]File to delete: {target_path.name} ({file_info.get('size_human', 'Unknown size')})[/yellow]")
        else:
            self.console.print(f"[yellow]Directory to delete: {target_path.name} (and all contents)[/yellow]")
        
        # Confirm deletion
        if not Confirm.ask("Are you sure you want to delete this?"):
            self.console.print("[yellow]Operation cancelled[/yellow]")
            return
        
        try:
            if target_path.is_file():
                success = self.file_ops.delete_file(target_path)
                if success:
                    self.console.print(f"[green]✓ File deleted: {target_path.name}[/green]")
                else:
                    self.console.print(f"[red]❌ Failed to delete file[/red]")
            else:
                # For directories, use shutil.rmtree
                import shutil
                shutil.rmtree(target_path)
                self.console.print(f"[green]✓ Directory deleted: {target_path.name}[/green]")
        except Exception as e:
            self.console.print(f"[red]❌ Error deleting: {e}[/red]")

    async def handle_copy_command(self, args: str):
        """Handle file copy command"""
        parts = args.split(" ", 1)
        if len(parts) < 2:
            self.console.print("[red]Usage: /copy <source_path> <destination_path>[/red]")
            return
        
        source_dest = parts[1].split(" ", 1)
        if len(source_dest) < 2:
            self.console.print("[red]Usage: /copy <source_path> <destination_path>[/red]")
            return
        
        source_path = Path(parts[0].strip())
        dest_path = Path(source_dest[1].strip())
        
        if not source_path.is_absolute():
            source_path = self.current_project_path / source_path
        if not dest_path.is_absolute():
            dest_path = self.current_project_path / dest_path
        
        if not source_path.exists():
            self.console.print(f"[red]❌ Source file not found: {source_path}[/red]")
            return
        
        if dest_path.exists():
            if not Confirm.ask(f"Destination {dest_path.name} exists. Overwrite?"):
                self.console.print("[yellow]Operation cancelled[/yellow]")
                return
        
        success = self.file_ops.copy_file(source_path, dest_path)
        if success:
            self.console.print(f"[green]✓ File copied: {source_path.name} → {dest_path.name}[/green]")
        else:
            self.console.print(f"[red]❌ Failed to copy file[/red]")

    async def handle_move_command(self, args: str):
        """Handle file move/rename command"""
        parts = args.split(" ", 1)
        if len(parts) < 2:
            self.console.print("[red]Usage: /move <source_path> <destination_path>[/red]")
            return
        
        source_dest = parts[1].split(" ", 1)
        if len(source_dest) < 2:
            self.console.print("[red]Usage: /move <source_path> <destination_path>[/red]")
            return
        
        source_path = Path(parts[0].strip())
        dest_path = Path(source_dest[1].strip())
        
        if not source_path.is_absolute():
            source_path = self.current_project_path / source_path
        if not dest_path.is_absolute():
            dest_path = self.current_project_path / dest_path
        
        if not source_path.exists():
            self.console.print(f"[red]❌ Source file not found: {source_path}[/red]")
            return
        
        if dest_path.exists():
            if not Confirm.ask(f"Destination {dest_path.name} exists. Overwrite?"):
                self.console.print("[yellow]Operation cancelled[/yellow]")
                return
        
        success = self.file_ops.move_file(source_path, dest_path)
        if success:
            self.console.print(f"[green]✓ File moved: {source_path.name} → {dest_path.name}[/green]")
        else:
            self.console.print(f"[red]❌ Failed to move file[/red]")

    async def handle_list_command(self, args: str):
        """Handle file listing command"""
        parts = args.split(" ", 1)
        directory = parts[0].strip() if parts[0].strip() else "."
        pattern = parts[1].strip() if len(parts) > 1 else "*"
        
        dir_path = Path(directory)
        if not dir_path.is_absolute():
            dir_path = self.current_project_path / dir_path
        
        if not dir_path.exists():
            self.console.print(f"[red]❌ Directory not found: {dir_path}[/red]")
            return
        
        if not dir_path.is_dir():
            self.console.print(f"[red]❌ Path is not a directory: {dir_path}[/red]")
            return
        
        try:
            files = self.file_ops.list_files(dir_path, pattern)
            
            if not files:
                self.console.print(f"[yellow]No files found matching pattern '{pattern}' in {dir_path.name}[/yellow]")
                return
            
            # Create files table
            files_table = Table(title=f"Files in {dir_path.name} (pattern: {pattern})", box=box.ROUNDED)
            files_table.add_column("Name", style="cyan")
            files_table.add_column("Type", style="blue")
            files_table.add_column("Size", style="green", justify="right")
            files_table.add_column("Modified", style="dim")
            
            for file_path in sorted(files):
                try:
                    if file_path.is_file():
                        file_info = self.file_ops.get_file_info(file_path)
                        files_table.add_row(
                            file_path.name,
                            "📄 File",
                            file_info.get('size_human', 'Unknown'),
                            time.strftime('%Y-%m-%d %H:%M', time.localtime(file_info.get('modified', 0)))
                        )
                    else:
                        files_table.add_row(
                            file_path.name,
                            "📁 Directory",
                            "-",
                            "-"
                        )
                except Exception:
                    files_table.add_row(file_path.name, "❓ Unknown", "-", "-")
            
            self.console.print(files_table)
            
        except Exception as e:
            self.console.print(f"[red]❌ Error listing files: {e}[/red]")

    async def handle_info_command(self, args: str):
        """Handle file information command"""
        if not args.strip():
            self.console.print("[red]Usage: /info <file_path>[/red]")
            return
        
        file_path = Path(args.strip())
        if not file_path.is_absolute():
            file_path = self.current_project_path / file_path
        
        if not file_path.exists():
            self.console.print(f"[red]❌ Path not found: {file_path}[/red]")
            return
        
        try:
            file_info = self.file_ops.get_file_info(file_path)
            
            if "error" in file_info:
                self.console.print(f"[red]❌ Error getting file info: {file_info['error']}[/red]")
                return
            
            # Create info table
            info_table = Table(title=f"File Information: {file_path.name}", box=box.ROUNDED)
            info_table.add_column("Property", style="cyan")
            info_table.add_column("Value", style="white")
            
            info_table.add_row("📁 Name", file_info['name'])
            info_table.add_row("📍 Path", file_info['path'])
            info_table.add_row("🏷️ Extension", file_info['extension'] or 'None')
            info_table.add_row("📏 Size", file_info['size_human'])
            info_table.add_row("📄 Type", "File" if file_info['is_file'] else "Directory")
            info_table.add_row("🔐 Permissions", file_info['permissions'])
            
            if file_info['is_file'] and file_info['line_count'] > 0:
                info_table.add_row("📝 Lines", str(file_info['line_count']))
            
            info_table.add_row("📅 Created", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(file_info['created'])))
            info_table.add_row("✏️ Modified", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(file_info['modified'])))
            
            # Check if it's a text file
            if file_path.is_file():
                is_text = self.file_ops.is_text_file(file_path)
                info_table.add_row("📖 Text File", "Yes" if is_text else "No")
            
            self.console.print(info_table)
            
        except Exception as e:
            self.console.print(f"[red]❌ Error getting file info: {e}[/red]")

    async def handle_tree_command(self, args: str):
        """Handle directory tree command"""
        directory = args.strip() if args.strip() else "."
        
        dir_path = Path(directory)
        if not dir_path.is_absolute():
            dir_path = self.current_project_path / dir_path
        
        if not dir_path.exists():
            self.console.print(f"[red]❌ Directory not found: {dir_path}[/red]")
            return
        
        if not dir_path.is_dir():
            self.console.print(f"[red]❌ Path is not a directory: {dir_path}[/red]")
            return
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                task = progress.add_task("Building directory tree...", total=None)
                
                tree_data = self.file_ops.get_directory_structure(dir_path, max_depth=3)
                progress.update(task, completed=True)
            
            def format_tree(node, prefix="", is_last=True):
                """Recursively format directory tree"""
                if "error" in node:
                    return f"{prefix}❌ {node['error']}"
                
                name = node['name']
                if node['type'] == 'file':
                    icon = "📄"
                    if 'size' in node:
                        size_info = f" ({self.file_ops._format_size(node['size'])})"
                        name += size_info
                elif node['type'] == 'directory':
                    icon = "📁"
                elif node['type'] == 'truncated':
                    icon = "⋯"
                else:
                    icon = "❓"
                
                current_prefix = "└── " if is_last else "├── "
                result = f"{prefix}{current_prefix}{icon} {name}\n"
                
                if node['type'] == 'directory' and 'children' in node:
                    next_prefix = prefix + ("    " if is_last else "│   ")
                    children = node['children']
                    for i, child in enumerate(children):
                        is_last_child = i == len(children) - 1
                        result += format_tree(child, next_prefix, is_last_child)
                
                return result
            
            tree_output = format_tree(tree_data)
            
            tree_panel = Panel(
                tree_output,
                title=f"[bold blue]📁 Directory Tree: {dir_path.name}[/bold blue]",
                border_style="blue"
            )
            self.console.print(tree_panel)
            
        except Exception as e:
            self.console.print(f"[red]❌ Error generating directory tree: {e}[/red]")

    async def process_command(self, user_input: str):
        """Process user commands"""
        if user_input.startswith("/"):
            # Handle special commands
            parts = user_input[1:].split(" ", 1)
            command = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            
            if command == "help":
                self.show_help()
            elif command == "status":
                self.show_status()
            elif command == "clear":
                self.console.clear()
                self.show_banner()
            elif command == "quit" or command == "exit":
                return False
            elif command == "generate":
                await self.handle_generate_command(args)
            elif command == "search":
                await self.handle_search_command(args)
            elif command == "config":
                await self.handle_config_command(args)
            elif command == "read":
                await self.handle_read_command(args)
            elif command == "write":
                await self.handle_write_command(args)
            elif command == "update":
                await self.handle_update_command(args)
            elif command == "delete":
                await self.handle_delete_command(args)
            elif command == "copy":
                await self.handle_copy_command(args)
            elif command == "move":
                await self.handle_move_command(args)
            elif command == "list":
                await self.handle_list_command(args)
            elif command == "info":
                await self.handle_info_command(args)
            elif command == "tree":
                await self.handle_tree_command(args)
            elif command == "edit":
                self.console.print("[yellow]Edit command coming soon![/yellow]")
            elif command == "review":
                self.console.print("[yellow]Review command coming soon![/yellow]")
            else:
                self.console.print(f"[red]Unknown command: {command}[/red]")
                self.console.print("Type [bold]/help[/bold] for available commands")
        else:
            # Handle natural language queries
            if self.llm_manager.clients:
                await self.handle_natural_query(user_input)
            else:
                self.console.print("[yellow]Natural language queries require API keys to be configured.[/yellow]")
                self.console.print("Use [bold]/config set <provider>_api_key <your_key>[/bold] to configure.")
        
        return True

    async def handle_natural_query(self, query: str):
        """Handle natural language queries"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Processing query...", total=None)
            
            try:
                # Simple response using LLM
                from ai_cli.llms.base import ChatMessage
                
                messages = [
                    ChatMessage(
                        role="system",
                        content="You are a helpful coding assistant. Provide clear, concise answers about programming and software development."
                    ),
                    ChatMessage(role="user", content=query)
                ]
                
                response = await self.llm_manager.chat_completion(messages=messages)
                progress.update(task, completed=True)
                
                # Display response in a nice panel
                response_panel = Panel(
                    Markdown(response.content),
                    title="[bold blue]AI Response[/bold blue]",
                    border_style="blue"
                )
                self.console.print(response_panel)
                
            except Exception as e:
                progress.update(task, completed=True)
                self.console.print(f"[red]❌ Error processing query: {e}[/red]")

    async def run(self):
        """Main interactive loop"""
        # Initialize system
        self.console.clear()
        self.show_banner()
        
        init_success = await self.initialize()
        if not init_success:
            self.console.print("[yellow]⚠ Some features may be limited due to initialization issues[/yellow]")
        
        self.show_tips()
        self.show_status()
        
        self.console.print("\n[bold green]🚀 S-y-N-t-a-X Interactive CLI is ready![/bold green]")
        self.console.print("[dim]Type /help for commands or ask me anything![/dim]\n")
        
        # Main interaction loop
        while True:
            try:
                # Custom prompt
                user_input = Prompt.ask(
                    "[bold blue]❯[/bold blue]",
                    default="",
                    show_default=False
                ).strip()
                
                if not user_input:
                    continue
                
                # Add to session history
                self.session_history.append(user_input)
                
                # Process the command
                should_continue = await self.process_command(user_input)
                if not should_continue:
                    break
                    
                self.console.print()  # Add spacing
                
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Use /quit to exit[/yellow]")
            except EOFError:
                break
        
        # Goodbye message
        self.console.print("\n[bold blue]👋 Thanks for using S-y-N-t-a-X AI CLI![/bold blue]")


def main():
    """Entry point for interactive UI"""
    ui = InteractiveUI()
    asyncio.run(ui.run())


if __name__ == "__main__":
    main()
