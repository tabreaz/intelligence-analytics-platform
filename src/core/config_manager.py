# src/core/config_manager.py
import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import logging
from src.core.logger import get_logger
from dataclasses import dataclass

logger = get_logger(__name__)


@dataclass
class DatabaseConfig:
    host: str
    port: int
    user: str = None
    password: str = None
    database: str = None
    secure: bool = False
    pool_size: int = 10
    min_pool_size: int = 2
    max_pool_size: int = 10
    timeout: int = 30
    compression: bool = True
    db: int = 0  # For Redis compatibility
    ssl: bool = False  # For Redis compatibility
    retry_on_timeout: bool = True  # For Redis compatibility
    command_timeout: int = 30


@dataclass
class LLMConfig:
    api_key: str
    base_url: str
    model: str
    max_tokens: int = 4000
    temperature: float = 0.1
    timeout: int = 60


class ConfigManager:
    """Centralized configuration management with environment variable support"""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self._configs = {}
        self._load_all_configs()

    def _load_all_configs(self):
        """Load all YAML configuration files"""
        config_files = [
            'database.yaml',
            'llm_models.yaml',
            'agents.yaml',
            'logging.yaml'
        ]

        for config_file in config_files:
            config_path = self.config_dir / config_file
            if config_path.exists():
                with open(config_path, 'r') as f:
                    content = f.read()
                    # Replace environment variables
                    content = self._substitute_env_vars(content)
                    config_data = yaml.safe_load(content)
                    self._configs[config_file.split('.')[0]] = config_data
                    logger.info(f"Loaded config: {config_file}")
            else:
                logger.warning(f"Config file not found: {config_file}")

    def _substitute_env_vars(self, content: str) -> str:
        """Replace ${VAR:default} patterns with environment variables"""
        import re

        def replace_var(match):
            var_expr = match.group(1)
            if ':' in var_expr:
                var_name, default_value = var_expr.split(':', 1)
            else:
                var_name, default_value = var_expr, ''

            return os.environ.get(var_name, default_value)

        return re.sub(r'\$\{([^}]+)\}', replace_var, content)

    def get_database_config(self, db_name: str) -> DatabaseConfig:
        """Get database configuration"""
        db_config = self._configs.get('database', {}).get('databases', {}).get(db_name, {})
        if not db_config:
            raise ValueError(f"Database config not found: {db_name}")

        return DatabaseConfig(**db_config)

    def get_llm_config(self, provider: str = None) -> LLMConfig:
        """Get LLM provider configuration"""
        llm_configs = self._configs.get('llm_models', {})

        if not provider:
            provider = llm_configs.get('default_provider', 'openai')

        provider_config = llm_configs.get('llm_providers', {}).get(provider, {})
        if not provider_config:
            raise ValueError(f"LLM provider config not found: {provider}")

        return LLMConfig(**provider_config)

    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """Get agent configuration"""
        agents_config = self._configs.get('agents', {}).get('agents', {})
        agent_config = agents_config.get(agent_name, {})

        if not agent_config:
            raise ValueError(f"Agent config not found: {agent_name}")

        return agent_config

    def get_workflow_config(self) -> Dict[str, Any]:
        """Get workflow configuration"""
        return self._configs.get('agents', {}).get('workflow', {})

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated path"""
        keys = key_path.split('.')
        value = self._configs

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value