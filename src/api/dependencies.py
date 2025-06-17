# src/api/dependencies.py
from typing import Optional
from src.agents.agent_manager import AgentManager
from src.core.config_manager import ConfigManager
from src.core.llm.base_llm import BaseLLMClient

# Global instances
_agent_manager: Optional[AgentManager] = None
_config_manager: Optional[ConfigManager] = None
_llm_client: Optional[BaseLLMClient] = None


def set_agent_manager(manager: AgentManager):
    """Set the global agent manager instance"""
    global _agent_manager
    _agent_manager = manager


def get_agent_manager() -> AgentManager:
    """Get the global agent manager instance"""
    if _agent_manager is None:
        raise RuntimeError("Agent manager not initialized")
    return _agent_manager


def set_config_manager(manager: ConfigManager):
    """Set the global config manager instance"""
    global _config_manager
    _config_manager = manager


def get_config_manager() -> ConfigManager:
    """Get the global config manager instance"""
    if _config_manager is None:
        raise RuntimeError("Config manager not initialized")
    return _config_manager


def set_llm_client(client: BaseLLMClient):
    """Set the global LLM client instance"""
    global _llm_client
    _llm_client = client


def get_llm_client() -> Optional[BaseLLMClient]:
    """Get the global LLM client instance (may be None)"""
    return _llm_client