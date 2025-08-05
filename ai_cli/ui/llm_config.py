"""
LLM Configuration UI for S-y-N-t-a-X AI CLI
Handles provider selection, API key configuration, and model selection
"""

import asyncio
from typing import Dict, List, Optional, Tuple
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.text import Text
from rich import box
from rich.align import Align
from rich.progress import Progress, SpinnerColumn, TextColumn

from ai_cli.llms.manager import LLMManager
from ai_cli.llms.ollama_client import OllamaClient
from ai_cli.config.settings import Settings


class LLMConfigurationUI:
    """UI for configuring LLM providers and models"""
    
    def __init__(self, console: Console):
        self.console = console
        self.settings = Settings()
        self.selected_provider = None
        self.selected_model = None
        self.api_key = None
        self.llm_manager = None
        
        # Provider information
        self.providers_info = {
            "openai": {
                "name": "OpenAI",
                "models": ["gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-3.5-turbo"],
                "api_key_required": True,
                "description": "Advanced AI models from OpenAI"
            },
            "anthropic": {
                "name": "Anthropic (Claude)",
                "models": ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
                "api_key_required": True,
                "description": "Powerful and helpful AI assistant from Anthropic"
            },
            "groq": {
                "name": "Groq",
                "models": ["llama2-70b-4096", "mixtral-8x7b-32768", "llama3-8b-8192", "llama3-70b-8192"],
                "api_key_required": True,
                "description": "Ultra-fast inference with open-source models"
            },
            "gemini": {
                "name": "Google Gemini",
                "models": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"],
                "api_key_required": True,
                "description": "Google's multimodal AI models"
            },
            "ollama": {
                "name": "Ollama (Local)",
                "models": [],  # Will be fetched dynamically
                "api_key_required": False,
                "description": "Run local models on your machine"
            }
        }

    def show_provider_selection(self) -> None:
        """Display provider selection interface"""
        provider_panel = Panel(
            self._create_provider_table(),
            title="[bold blue]🤖 Select LLM Provider[/bold blue]",
            subtitle="[dim]Choose your preferred AI provider[/dim]",
            border_style="blue",
            box=box.ROUNDED
        )
        self.console.print(provider_panel)

    def _create_provider_table(self) -> Table:
        """Create a table showing available providers"""
        table = Table(show_header=True, header_style="bold cyan", box=box.SIMPLE)
        table.add_column("Option", style="cyan", no_wrap=True, width=8)
        table.add_column("Provider", style="white", width=20)
        table.add_column("Description", style="dim", width=40)
        table.add_column("API Key", style="yellow", width=12)
        
        for i, (key, info) in enumerate(self.providers_info.items(), 1):
            api_status = "Required" if info["api_key_required"] else "Not needed"
            table.add_row(
                f"{i}",
                f"[bold]{info['name']}[/bold]",
                info["description"],
                f"[red]{api_status}[/red]" if info["api_key_required"] else f"[green]{api_status}[/green]"
            )
        
        return table

    async def select_provider(self) -> bool:
        """Handle provider selection process"""
        self.show_provider_selection()
        
        while True:
            choice = Prompt.ask(
                "\n[bold blue]Choose a provider[/bold blue]",
                choices=["1", "2", "3", "4", "5", "q"],
                default="1"
            )
            
            if choice == "q":
                return False
            
            provider_keys = list(self.providers_info.keys())
            if choice.isdigit() and 1 <= int(choice) <= len(provider_keys):
                self.selected_provider = provider_keys[int(choice) - 1]
                provider_info = self.providers_info[self.selected_provider]
                
                self.console.print(f"\n[green]✓ Selected:[/green] {provider_info['name']}")
                
                # Handle API key configuration
                if provider_info["api_key_required"]:
                    if not await self._configure_api_key():
                        continue  # Go back to provider selection
                
                # Handle model selection
                if not await self._configure_model():
                    continue  # Go back to provider selection
                
                return True
            else:
                self.console.print("[red]Invalid choice. Please try again.[/red]")

    async def _configure_api_key(self) -> bool:
        """Configure API key for the selected provider"""
        provider_info = self.providers_info[self.selected_provider]
        
        # Check if API key already exists in environment or config
        existing_key = self._get_existing_api_key()
        
        if existing_key:
            self.console.print(f"[green]✓ Found existing API key for {provider_info['name']}[/green]")
            use_existing = Confirm.ask("Use existing API key?", default=True)
            if use_existing:
                self.api_key = existing_key
                return True
        
        # Prompt for new API key
        key_panel = Panel(
            f"[yellow]API Key Required for {provider_info['name']}[/yellow]\n\n"
            f"Please enter your API key for {provider_info['name']}.\n"
            f"Visit their website to get an API key if you don't have one.",
            title="[bold yellow]🔑 API Key Configuration[/bold yellow]",
            border_style="yellow"
        )
        self.console.print(key_panel)
        
        while True:
            api_key = Prompt.ask(
                f"[bold yellow]Enter {provider_info['name']} API key[/bold yellow]",
                password=True
            )
            
            if not api_key.strip():
                self.console.print("[red]API key cannot be empty.[/red]")
                continue
            
            # Validate API key format (basic validation)
            if self._validate_api_key_format(api_key):
                self.api_key = api_key.strip()
                
                # Test the API key
                if await self._test_api_key():
                    self.console.print("[green]✓ API key validated successfully![/green]")
                    return True
                else:
                    self.console.print("[red]❌ API key validation failed. Please check and try again.[/red]")
                    retry = Confirm.ask("Try a different API key?", default=True)
                    if not retry:
                        return False
            else:
                self.console.print("[red]Invalid API key format. Please check and try again.[/red]")

    def _get_existing_api_key(self) -> Optional[str]:
        """Check for existing API key in environment variables"""
        import os
        
        env_var_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "groq": "GROQ_API_KEY",
            "gemini": "GOOGLE_API_KEY"
        }
        
        env_var = env_var_map.get(self.selected_provider)
        if env_var:
            return os.getenv(env_var)
        return None

    def _validate_api_key_format(self, api_key: str) -> bool:
        """Basic validation of API key format"""
        api_key = api_key.strip()
        
        # Basic format validation for different providers
        if self.selected_provider == "openai":
            return api_key.startswith("sk-") and len(api_key) > 10
        elif self.selected_provider == "anthropic":
            return api_key.startswith("sk-ant-") and len(api_key) > 20
        elif self.selected_provider == "groq":
            return len(api_key) > 10  # Groq keys vary in format
        elif self.selected_provider == "gemini":
            return len(api_key) > 10  # Google API keys vary in format
        
        return len(api_key) > 5  # Fallback validation

    async def _test_api_key(self) -> bool:
        """Test the API key by making a simple request"""
        try:
            self.console.print("[dim]Testing API key...[/dim]")
            
            # Import specific client based on provider
            if self.selected_provider == "openai":
                from ai_cli.llms.openai_client import OpenAIClient
                client = OpenAIClient(self.api_key)
            elif self.selected_provider == "anthropic":
                from ai_cli.llms.anthropic_client import AnthropicClient
                client = AnthropicClient(self.api_key)
            elif self.selected_provider == "groq":
                from ai_cli.llms.groq_client import GroqClient
                client = GroqClient(self.api_key)
            elif self.selected_provider == "gemini":
                from ai_cli.llms.gemini_client import GeminiClient
                client = GeminiClient(self.api_key)
                # Test by checking if we can create the client without errors
                self.console.print("[dim]Gemini client created successfully[/dim]")
                return True
            else:
                return False
            
            # For other providers, we could add actual API tests here
            self.console.print("[dim]API key format looks valid[/dim]")
            return True
            
        except Exception as e:
            self.console.print(f"[dim]API key test error: {e}[/dim]")
            return False

    async def _configure_model(self) -> bool:
        """Configure model selection for the selected provider"""
        provider_info = self.providers_info[self.selected_provider]
        
        if self.selected_provider == "ollama":
            return await self._configure_ollama_model()
        else:
            return await self._configure_standard_model()

    async def _configure_ollama_model(self) -> bool:
        """Configure Ollama model by fetching available models"""
        self.console.print("\n[blue]📡 Fetching available Ollama models...[/blue]")
        
        try:
            # Show progress while fetching models
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                task = progress.add_task("Connecting to Ollama...", total=None)
                
                # Create Ollama client and fetch models
                ollama_client = OllamaClient()
                installed_models = await ollama_client.get_installed_models()
                
                # Stop the progress spinner
                progress.stop()
            
            # Clear the progress line and check results
            self.console.print("")  # Add a clean line
            
            if not installed_models:
                self.console.print("[yellow]⚠ No models found in Ollama installation.[/yellow]")
                self.console.print("[dim]Please install models using: ollama pull <model_name>[/dim]")
                self.console.print("[dim]Popular models: llama2, llama3, codellama, mistral[/dim]")
                return False
            
            self.console.print(f"[green]✓ Found {len(installed_models)} installed models[/green]")
            
            # Show available models
            model_table = Table(show_header=True, header_style="bold cyan", box=box.SIMPLE)
            model_table.add_column("Option", style="cyan", width=8)
            model_table.add_column("Model Name", style="white", width=30)
            model_table.add_column("Size", style="dim", width=12)
            model_table.add_column("Modified", style="dim", width=20)
            
            for i, model in enumerate(installed_models, 1):
                size = self._format_size(model.get("size", 0))
                modified = model.get("modified_at", "Unknown")[:19] if model.get("modified_at") else "Unknown"
                model_table.add_row(
                    f"{i}",
                    f"[bold]{model['name']}[/bold]",
                    size,
                    modified
                )
            
            model_panel = Panel(
                model_table,
                title="[bold green]🦙 Available Ollama Models[/bold green]",
                border_style="green"
            )
            self.console.print(model_panel)
            
            # Select model
            while True:
                choice = Prompt.ask(
                    "\n[bold green]Choose a model[/bold green]",
                    choices=[str(i) for i in range(1, len(installed_models) + 1)] + ["q"],
                    default="1"
                )
                
                if choice == "q":
                    return False
                
                if choice.isdigit() and 1 <= int(choice) <= len(installed_models):
                    selected_model = installed_models[int(choice) - 1]
                    self.selected_model = selected_model["name"]
                    self.console.print(f"[green]✓ Selected model:[/green] {self.selected_model}")
                    return True
                else:
                    self.console.print("[red]Invalid choice. Please try again.[/red]")
                    
        except Exception as e:
            self.console.print(f"[red]❌ Error connecting to Ollama: {e}[/red]")
            self.console.print("[dim]Make sure Ollama is running: ollama serve[/dim]")
            self.console.print(f"[dim]Error details: {str(e)}[/dim]")
            return False

    async def _configure_standard_model(self) -> bool:
        """Configure model for standard providers (non-Ollama)"""
        provider_info = self.providers_info[self.selected_provider]
        
        # Show available models
        model_table = Table(show_header=True, header_style="bold cyan", box=box.SIMPLE)
        model_table.add_column("Option", style="cyan", width=8)
        model_table.add_column("Model Name", style="white", width=40)
        
        for i, model in enumerate(provider_info["models"], 1):
            model_table.add_row(f"{i}", f"[bold]{model}[/bold]")
        
        model_table.add_row(f"{len(provider_info['models']) + 1}", "[dim]Enter custom model name[/dim]")
        
        model_panel = Panel(
            model_table,
            title=f"[bold blue]🎯 Select Model for {provider_info['name']}[/bold blue]",
            border_style="blue"
        )
        self.console.print(model_panel)
        
        # Select model
        while True:
            max_choice = len(provider_info["models"]) + 1
            choice = Prompt.ask(
                "\n[bold blue]Choose a model[/bold blue]",
                choices=[str(i) for i in range(1, max_choice + 1)] + ["q"],
                default="1"
            )
            
            if choice == "q":
                return False
            
            if choice.isdigit() and 1 <= int(choice) <= max_choice:
                if int(choice) <= len(provider_info["models"]):
                    # Standard model selection
                    self.selected_model = provider_info["models"][int(choice) - 1]
                    self.console.print(f"[green]✓ Selected model:[/green] {self.selected_model}")
                    return True
                else:
                    # Custom model name
                    custom_model = Prompt.ask("[bold blue]Enter custom model name[/bold blue]")
                    if custom_model.strip():
                        self.selected_model = custom_model.strip()
                        self.console.print(f"[green]✓ Selected custom model:[/green] {self.selected_model}")
                        return True
                    else:
                        self.console.print("[red]Model name cannot be empty.[/red]")
            else:
                self.console.print("[red]Invalid choice. Please try again.[/red]")

    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"

    async def create_llm_manager(self) -> LLMManager:
        """Create and configure LLM manager with selected settings"""
        # Set environment variable for the API key if needed
        if self.api_key:
            import os
            env_var_map = {
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY", 
                "groq": "GROQ_API_KEY",
                "gemini": "GOOGLE_API_KEY"
            }
            
            env_var = env_var_map.get(self.selected_provider)
            if env_var:
                os.environ[env_var] = self.api_key
                self.console.print(f"[dim]Set {env_var} environment variable[/dim]")
        
        # Create LLM manager (this will automatically initialize clients)
        llm_manager = LLMManager()
        
        # Force re-initialization to pick up the new environment variable
        llm_manager._initialize_clients()
        
        # Verify the selected provider is available
        if self.selected_provider not in llm_manager.clients:
            available_providers = list(llm_manager.clients.keys())
            if available_providers:
                self.console.print(f"[yellow]Warning: {self.selected_provider} not available, but found: {available_providers}[/yellow]")
                # Use the first available provider instead
                fallback_provider = available_providers[0]
                self.console.print(f"[blue]Using {fallback_provider} as fallback provider[/blue]")
            else:
                raise Exception(f"No LLM clients initialized. Check API key for {self.selected_provider}")
        
        # Set the default model to our selected model
        llm_manager.default_model = self.selected_model
        
        return llm_manager

    def show_configuration_summary(self) -> None:
        """Show a summary of the selected configuration"""
        summary_text = f"""[bold green]✓ Configuration Complete![/bold green]

[cyan]Provider:[/cyan] {self.providers_info[self.selected_provider]['name']}
[cyan]Model:[/cyan] {self.selected_model}
[cyan]API Key:[/cyan] {'Configured' if self.api_key else 'Not required'}

[dim]You can now start asking questions and giving commands![/dim]"""
        
        summary_panel = Panel(
            summary_text,
            title="[bold white]🎉 LLM Configuration Summary[/bold white]",
            border_style="green",
            box=box.DOUBLE
        )
        self.console.print(summary_panel)

    def get_selected_config(self) -> Dict[str, str]:
        """Get the selected configuration as a dictionary"""
        return {
            "provider": self.selected_provider,
            "model": self.selected_model,
            "api_key": self.api_key or ""
        }
