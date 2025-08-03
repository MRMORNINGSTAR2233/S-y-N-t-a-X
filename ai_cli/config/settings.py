"""Configuration and settings management for the AI CLI."""

import os
import toml
from pathlib import Path
from typing import Dict, Any, Optional, TYPE_CHECKING
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ai_cli.llms.manager import LLMManager


class GeneralSettings(BaseModel):
    """General application settings."""
    default_model: str = Field(default="claude-3-sonnet")
    interactive_mode: bool = Field(default=True)
    auto_save: bool = Field(default=True)
    max_context_length: int = Field(default=8000)
    verbose_logging: bool = Field(default=False)


class ProviderSettings(BaseModel):
    """LLM provider settings."""
    preferred_order: list[str] = Field(default=["anthropic", "openai", "groq", "ollama"])
    timeout_seconds: int = Field(default=30)
    max_retries: int = Field(default=3)


class MemorySettings(BaseModel):
    """Memory and vector store settings."""
    enable_vector_store: bool = Field(default=True)
    max_history_items: int = Field(default=1000)
    embedding_model: str = Field(default="text-embedding-ada-002")
    chroma_persist_directory: str = Field(default=".chroma")


class SecuritySettings(BaseModel):
    """Security and safety settings."""
    require_confirmation: bool = Field(default=True)
    safe_mode: bool = Field(default=True)
    max_file_size_mb: int = Field(default=10)
    allowed_file_types: list[str] = Field(default=[
        ".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rs", ".rb", ".php",
        ".html", ".css", ".json", ".yaml", ".yml", ".toml", ".md", ".txt"
    ])


class Settings:
    """Main settings manager for the AI CLI."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or self._get_default_config_path()
        self.general = GeneralSettings()
        self.providers = ProviderSettings()
        self.memory = MemorySettings()
        self.security = SecuritySettings()
        self._llm_manager: Optional["LLMManager"] = None
        
        # Load configuration if exists
        if self.config_path.exists():
            self.load_config()
    
    def _get_default_config_path(self) -> Path:
        """Get the default configuration file path."""
        home_dir = Path.home()
        config_dir = home_dir / ".syntax"
        config_dir.mkdir(exist_ok=True)
        return config_dir / "config.toml"
    
    def load_config(self) -> None:
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                config_data = toml.load(f)
            
            # Load each section
            if 'general' in config_data:
                self.general = GeneralSettings(**config_data['general'])
            if 'providers' in config_data:
                self.providers = ProviderSettings(**config_data['providers'])
            if 'memory' in config_data:
                self.memory = MemorySettings(**config_data['memory'])
            if 'security' in config_data:
                self.security = SecuritySettings(**config_data['security'])
                
        except Exception as e:
            print(f"Warning: Failed to load config from {self.config_path}: {e}")
    
    def save_config(self) -> bool:
        """Save current configuration to file."""
        try:
            config_data = {
                'general': self.general.model_dump(),
                'providers': self.providers.model_dump(),
                'memory': self.memory.model_dump(),
                'security': self.security.model_dump()
            }
            
            # Ensure directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                toml.dump(config_data, f)
            
            return True
        except Exception as e:
            print(f"Error: Failed to save config to {self.config_path}: {e}")
            return False
    
    def set_setting(self, key: str, value: Any) -> bool:
        """Set a configuration setting by key path."""
        try:
            # Parse key path (e.g., "general.default_model")
            if '.' in key:
                section, setting = key.split('.', 1)
                section_obj = getattr(self, section, None)
                if section_obj and hasattr(section_obj, setting):
                    # Convert value to appropriate type
                    field_info = section_obj.model_fields.get(setting)
                    if field_info:
                        if field_info.annotation == bool:
                            value = value.lower() in ('true', '1', 'yes', 'on')
                        elif field_info.annotation == int:
                            value = int(value)
                        elif field_info.annotation == list:
                            value = [item.strip() for item in value.split(',')]
                    
                    setattr(section_obj, setting, value)
                    self.save_config()
                    return True
            
            return False
        except Exception:
            return False
    
    def get_setting(self, key: str) -> Any:
        """Get a configuration setting by key path."""
        try:
            if '.' in key:
                section, setting = key.split('.', 1)
                section_obj = getattr(self, section, None)
                if section_obj and hasattr(section_obj, setting):
                    return getattr(section_obj, setting)
            return None
        except Exception:
            return None
    
    def get_all_settings(self) -> Dict[str, Any]:
        """Get all configuration settings as a flat dictionary."""
        settings = {}
        
        for section_name in ['general', 'providers', 'memory', 'security']:
            section = getattr(self, section_name)
            for key, value in section.model_dump().items():
                settings[f"{section_name}.{key}"] = value
        
        return settings
    
    def get_llm_manager(self) -> "LLMManager":
        """Get or create the LLM manager instance."""
        if self._llm_manager is None:
            from ai_cli.llms.manager import LLMManager
            self._llm_manager = LLMManager(config_path=self.config_path)
        return self._llm_manager
    
    def load_project_config(self, project_path: Path) -> Dict[str, Any]:
        """Load project-specific configuration."""
        project_config_path = project_path / ".syntax.toml"
        
        if not project_config_path.exists():
            return {}
        
        try:
            with open(project_config_path, 'r') as f:
                return toml.load(f)
        except Exception as e:
            print(f"Warning: Failed to load project config: {e}")
            return {}
    
    def create_default_project_config(self, project_path: Path) -> bool:
        """Create a default project configuration file."""
        project_config_path = project_path / ".syntax.toml"
        
        default_config = {
            'project': {
                'name': project_path.name,
                'language': 'auto-detect',
                'framework': 'auto-detect'
            },
            'prompts': {
                'system_prompt': 'You are an expert software engineer.',
                'coding_style': 'Follow best practices and maintain consistency.'
            },
            'exclusions': {
                'ignore_patterns': [
                    '*.pyc', '__pycache__', '.git', 'node_modules',
                    '*.log', '.env', '.venv', 'venv'
                ]
            },
            'tools': {
                'auto_format': True,
                'auto_lint': True,
                'auto_test': False
            }
        }
        
        try:
            with open(project_config_path, 'w') as f:
                toml.dump(default_config, f)
            return True
        except Exception as e:
            print(f"Error: Failed to create project config: {e}")
            return False
