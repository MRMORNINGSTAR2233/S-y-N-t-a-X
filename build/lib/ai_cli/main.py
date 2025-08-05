"""
S-y-N-t-a-X AI CLI Main Entry Point

This module provides the main CLI interface for the AI-powered terminal application.
It handles command routing, configuration, and interaction with various agents and tools.
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.prompt import Confirm
from rich.traceback import install

from ai_cli.config.settings import Settings
from ai_cli.config.database import DatabaseManager
from ai_cli.memory.vector_store import VectorStore
from ai_cli.agents.code_generator import CodeGeneratorAgent
from ai_cli.agents.debugger import DebuggerAgent
from ai_cli.agents.navigator import NavigatorAgent
from ai_cli.agents.reviewer import ReviewerAgent
from ai_cli.tools.git_integration import GitManager

# Install rich traceback handler
install(show_locals=True)

console = Console()


class CLIContext:
    """Global CLI context object."""
    
    def __init__(self):
        self.settings = Settings()
        self.db_manager = DatabaseManager()
        self.vector_store = VectorStore()
        self.git_manager = GitManager()
        self.interactive_mode = True


pass_context = click.make_pass_decorator(CLIContext, ensure=True)


@click.group(invoke_without_command=True)
@click.option('--interactive/--no-interactive', default=True, help='Enable interactive mode')
@click.option('--config-path', type=click.Path(), help='Path to configuration file')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx: click.Context, interactive: bool, config_path: Optional[str], verbose: bool):
    """S-y-N-t-a-X: AI-Powered Terminal CLI for intelligent codebase manipulation."""
    
    # Initialize context
    ctx.ensure_object(CLIContext)
    ctx.obj.interactive_mode = interactive
    
    if verbose:
        console.print("[dim]Verbose mode enabled[/dim]")
    
    # If no command provided, start interactive mode
    if ctx.invoked_subcommand is None:
        if interactive:
            start_interactive_mode(ctx.obj)
        else:
            console.print("[red]No command provided. Use --help for available commands.[/red]")
            sys.exit(1)


@cli.command()
@click.argument('description', required=True)
@click.option('--file', '-f', help='Target file for the feature')
@click.option('--language', '-l', help='Programming language')
@click.option('--framework', help='Framework to use')
@click.option('--dry-run', is_flag=True, help='Show what would be generated without applying')
@pass_context
def generate_feature(ctx: CLIContext, description: str, file: Optional[str], 
                    language: Optional[str], framework: Optional[str], dry_run: bool):
    """Generate a new feature from natural language description."""
    
    console.print(f"[bold blue]Generating feature:[/bold blue] {description}")
    
    agent = CodeGeneratorAgent(
        llm_manager=ctx.settings.get_llm_manager(),
        vector_store=ctx.vector_store,
        git_manager=ctx.git_manager
    )
    
    # Run the generation in async context
    result = asyncio.run(agent.generate_feature(
        description=description,
        target_file=file,
        language=language,
        framework=framework,
        dry_run=dry_run
    ))
    
    if not dry_run and ctx.interactive_mode:
        if Confirm.ask("Apply the generated changes?"):
            console.print("[green]✓ Changes applied successfully![/green]")
        else:
            console.print("[yellow]Changes cancelled.[/yellow]")


@cli.command()
@click.argument('target', required=True)
@click.option('--trace', '-t', help='Error trace file')
@click.option('--context', '-c', help='Additional context files')
@click.option('--fix/--analyze-only', default=True, help='Apply fixes or analyze only')
@pass_context
def debug(ctx: CLIContext, target: str, trace: Optional[str], 
          context: Optional[str], fix: bool):
    """Debug and fix issues in code files or error traces."""
    
    console.print(f"[bold red]Debugging:[/bold red] {target}")
    
    agent = DebuggerAgent(
        llm_manager=ctx.settings.get_llm_manager(),
        vector_store=ctx.vector_store,
        git_manager=ctx.git_manager
    )
    
    result = asyncio.run(agent.debug_issue(
        target=target,
        trace_file=trace,
        context_files=context.split(',') if context else None,
        apply_fixes=fix and not ctx.interactive_mode
    ))
    
    if fix and ctx.interactive_mode and result.get('fixes'):
        if Confirm.ask("Apply the suggested fixes?"):
            console.print("[green]✓ Fixes applied successfully![/green]")
        else:
            console.print("[yellow]Fixes cancelled.[/yellow]")


@cli.command()
@click.argument('symbol', required=True)
@click.option('--type', 'symbol_type', help='Symbol type (function, class, variable)')
@click.option('--file', '-f', help='Limit search to specific file')
@pass_context
def navigate(ctx: CLIContext, symbol: str, symbol_type: Optional[str], file: Optional[str]):
    """Find and navigate to code symbols in the codebase."""
    
    console.print(f"[bold green]Navigating to:[/bold green] {symbol}")
    
    agent = NavigatorAgent(
        llm_manager=ctx.settings.get_llm_manager(),
        vector_store=ctx.vector_store,
        git_manager=ctx.git_manager
    )
    
    result = asyncio.run(agent.navigate_to_symbol(
        symbol=symbol,
        symbol_type=symbol_type,
        target_file=file
    ))
    
    if result.get('found'):
        console.print(f"[green]✓ Found {symbol} at {result['location']}[/green]")
    else:
        console.print(f"[red]✗ Symbol '{symbol}' not found[/red]")


@cli.command()
@click.argument('query', required=True)
@click.option('--semantic/--keyword', default=True, help='Search type')
@click.option('--files', '-f', help='Limit search to specific files/patterns')
@click.option('--exclude', '-e', help='Exclude files/patterns')
@pass_context
def search(ctx: CLIContext, query: str, semantic: bool, files: Optional[str], 
           exclude: Optional[str]):
    """Search codebase with semantic or keyword matching."""
    
    search_type = "semantic" if semantic else "keyword"
    console.print(f"[bold cyan]Searching ({search_type}):[/bold cyan] {query}")
    
    agent = NavigatorAgent(
        llm_manager=ctx.settings.get_llm_manager(),
        vector_store=ctx.vector_store,
        git_manager=ctx.git_manager
    )
    
    result = asyncio.run(agent.search_codebase(
        query=query,
        search_type=search_type,
        include_patterns=files.split(',') if files else None,
        exclude_patterns=exclude.split(',') if exclude else None
    ))
    
    console.print(f"[green]Found {len(result.get('results', []))} matches[/green]")


@cli.command()
@click.argument('instruction', required=True)
@click.option('--file', '-f', help='Target file to edit')
@click.option('--scope', help='Scope of changes (function, class, file)')
@pass_context
def edit(ctx: CLIContext, instruction: str, file: Optional[str], scope: Optional[str]):
    """Make interactive code modifications with natural language instructions."""
    
    console.print(f"[bold magenta]Editing:[/bold magenta] {instruction}")
    
    agent = CodeGeneratorAgent(
        llm_manager=ctx.settings.get_llm_manager(),
        vector_store=ctx.vector_store,
        git_manager=ctx.git_manager
    )
    
    result = asyncio.run(agent.edit_code(
        instruction=instruction,
        target_file=file,
        scope=scope
    ))
    
    if ctx.interactive_mode and result.get('changes'):
        if Confirm.ask("Apply the suggested changes?"):
            console.print("[green]✓ Changes applied successfully![/green]")
        else:
            console.print("[yellow]Changes cancelled.[/yellow]")


@cli.command()
@click.option('--files', '-f', help='Files or patterns to review')
@click.option('--scope', help='Review scope (security, performance, style, all)')
@click.option('--output', '-o', help='Output file for review report')
@pass_context
def review(ctx: CLIContext, files: Optional[str], scope: Optional[str], 
           output: Optional[str]):
    """AI-powered code review with comprehensive analysis."""
    
    console.print("[bold purple]Starting code review...[/bold purple]")
    
    agent = ReviewerAgent(
        llm_manager=ctx.settings.get_llm_manager(),
        vector_store=ctx.vector_store,
        git_manager=ctx.git_manager
    )
    
    result = asyncio.run(agent.review_code(
        file_patterns=files.split(',') if files else None,
        review_scope=scope or 'all',
        output_file=output
    ))
    
    console.print(f"[green]✓ Review completed with {len(result.get('issues', []))} findings[/green]")


@cli.group()
@pass_context
def config(ctx: CLIContext):
    """Configuration management commands."""
    pass


@config.command('set-key')
@click.argument('provider', required=True)
@click.argument('api_key', required=True)
@pass_context
def set_api_key(ctx: CLIContext, provider: str, api_key: str):
    """Set API key for LLM provider."""
    
    success = ctx.db_manager.store_api_key(provider, api_key)
    if success:
        console.print(f"[green]✓ API key set for {provider}[/green]")
    else:
        console.print(f"[red]✗ Failed to set API key for {provider}[/red]")


@config.command('list')
@pass_context
def list_config(ctx: CLIContext):
    """List current configuration."""
    
    settings = ctx.settings.get_all_settings()
    console.print("[bold]Current Configuration:[/bold]")
    for key, value in settings.items():
        console.print(f"  {key}: {value}")


@config.command('set')
@click.argument('key', required=True)
@click.argument('value', required=True)
@pass_context
def set_config(ctx: CLIContext, key: str, value: str):
    """Set configuration value."""
    
    success = ctx.settings.set_setting(key, value)
    if success:
        console.print(f"[green]✓ Set {key} = {value}[/green]")
    else:
        console.print(f"[red]✗ Failed to set {key}[/red]")


@cli.command()
@click.option('--demo', is_flag=True, help='Run in demo mode (no API keys required)')
@pass_context
def ui(ctx: CLIContext, demo: bool):
    """Launch the rich interactive terminal interface."""
    
    if demo:
        console.print("[bold yellow]🎭 Starting Demo Mode...[/bold yellow]")
        try:
            from ai_cli.ui.demo import DemoInteractiveUI
            ui_instance = DemoInteractiveUI()
            asyncio.run(ui_instance.run())
        except ImportError as e:
            console.print(f"[red]❌ Demo UI not available: {e}[/red]")
    else:
        console.print("[bold blue]🚀 Starting Interactive UI...[/bold blue]")
        try:
            from ai_cli.ui.interactive import InteractiveUI
            ui_instance = InteractiveUI()
            asyncio.run(ui_instance.run())
        except ImportError as e:
            console.print(f"[red]❌ Interactive UI not available: {e}[/red]")
            console.print("[dim]Try using --demo flag for demo mode[/dim]")


@cli.command('ui')
@click.option('--demo', is_flag=True, help='Run in demo mode (no API keys required)')
@pass_context
def interactive_ui(ctx: CLIContext, demo: bool):
    """Launch the rich interactive terminal interface."""
    
    try:
        if demo:
            # Import and run demo UI
            from .ui.interactive import DemoInteractiveUI
            console.print("[yellow]🎭 Starting S-y-N-t-a-X in Demo Mode...[/yellow]")
            ui = DemoInteractiveUI()
        else:
            # Import and run full UI
            from .ui.interactive import InteractiveUI
            console.print("[blue]🚀 Starting S-y-N-t-a-X Interactive UI...[/blue]")
            ui = InteractiveUI()
        
        # Run the UI
        asyncio.run(ui.run())
        
    except ImportError as e:
        console.print(f"[red]❌ Could not import UI module: {e}[/red]")
        console.print("[yellow]💡 Try running the demo UI directly: python demo_ui.py[/yellow]")
    except KeyboardInterrupt:
        console.print("\n[blue]👋 Goodbye![/blue]")
    except Exception as e:
        console.print(f"[red]❌ Error starting UI: {e}[/red]")


def start_interactive_mode(ctx: CLIContext):
    """Start interactive CLI session with rich UI."""
    
    try:
        # Import and run the enhanced interactive UI
        from ai_cli.ui.interactive import InteractiveUI
        console.print("[bold blue]🚀 Starting S-y-N-t-a-X Interactive UI...[/bold blue]")
        ui = InteractiveUI()
        asyncio.run(ui.run())
        
    except ImportError as e:
        console.print(f"[red]❌ Could not import interactive UI: {e}[/red]")
        console.print("[yellow]💡 Falling back to basic CLI mode...[/yellow]")
        
        # Fallback to basic CLI
        console.print("[bold blue]Welcome to S-y-N-t-a-X AI CLI![/bold blue]")
        console.print("Type 'help' for commands or 'exit' to quit.\n")
        
        while True:
            try:
                command = console.input("[bold green]syntax>[/bold green] ").strip()
                
                if command.lower() in ['exit', 'quit', 'q']:
                    console.print("[dim]Goodbye![/dim]")
                    break
                elif command.lower() in ['help', 'h']:
                    console.print(cli.get_help(click.Context(cli)))
                elif command:
                    # Parse and execute command
                    try:
                        cli.main(command.split(), standalone_mode=False)
                    except SystemExit:
                        pass  # Ignore sys.exit from click
                    except Exception as e:
                        console.print(f"[red]Error: {e}[/red]")
                
            except KeyboardInterrupt:
                console.print("\n[dim]Use 'exit' to quit.[/dim]")
            except EOFError:
                break
                
    except KeyboardInterrupt:
        console.print("\n[blue]👋 Goodbye![/blue]")
    except Exception as e:
        console.print(f"[red]❌ Error starting interactive mode: {e}[/red]")


if __name__ == '__main__':
    cli()
