# /src/core/resource_manager.py
"""
Shared Resource Manager for connection pooling and resource reuse.
This manager ensures efficient resource utilization across all agents.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Set
from datetime import datetime

import redis
from redis.connection import ConnectionPool

from src.core.database.clickhouse_pool import get_clickhouse_pool, close_clickhouse_pool
from src.core.database.clickhouse_client import ClickHouseClient
from src.core.session_manager import PostgreSQLSessionManager
from src.core.llm.base_llm import LLMClientFactory
from src.core.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class ResourceManager:
    """
    Centralized resource manager for sharing connections and clients across agents.
    Supports multi-user scenarios with proper isolation.
    """

    def __init__(self, config_manager: ConfigManager):
        """Initialize shared resources from configuration."""
        self.config_manager = config_manager
        self._clickhouse_pool = None
        self._redis_pool = None
        self._redis_client = None
        self._session_manager = None
        self._llm_clients = {}
        self._initialized = False

        # Multi-user support
        self._user_contexts: Dict[str, Dict] = {}
        self._user_semaphores: Dict[str, asyncio.Semaphore] = {}
        self._metrics = ResourceMetrics()

    async def initialize(self):
        """Initialize all shared resources with proper error handling."""
        if self._initialized:
            return

        logger.info("Initializing shared resources...")

        initialization_errors = []

        # Initialize ClickHouse
        try:
            ch_config = self.config_manager.get_database_config('clickhouse')
            # Determine pool size based on deployment
            pool_size = 50 if self._is_production() else 10
            self._clickhouse_pool = get_clickhouse_pool(ch_config, pool_size)
            logger.info(f"ClickHouse pool initialized with size {pool_size}")
        except Exception as e:
            initialization_errors.append(f"ClickHouse: {e}")
            logger.error(f"Failed to initialize ClickHouse pool: {e}")

        # Initialize Redis
        try:
            redis_config = self.config_manager.get_database_config('redis')
            self._redis_pool = ConnectionPool(
                host=redis_config.host,
                port=redis_config.port,
                db=redis_config.db,
                password=redis_config.password,
                max_connections=redis_config.pool_size,
                decode_responses=True,
                socket_timeout=redis_config.timeout,
                retry_on_timeout=redis_config.retry_on_timeout
            )
            self._redis_client = redis.Redis(connection_pool=self._redis_pool)
            # Test connection
            self._redis_client.ping()
            logger.info("Redis pool initialized")
        except Exception as e:
            initialization_errors.append(f"Redis: {e}")
            logger.error(f"Failed to initialize Redis pool: {e}")

        # Initialize PostgreSQL session manager
        try:
            pg_config = self.config_manager.get_database_config('postgresql')
            self._session_manager = PostgreSQLSessionManager(pg_config.__dict__)
            await self._session_manager.initialize()
            logger.info("PostgreSQL session manager initialized")
        except Exception as e:
            initialization_errors.append(f"PostgreSQL: {e}")
            logger.error(f"Failed to initialize session manager: {e}")

        # Initialize LLM clients
        try:
            await self._initialize_llm_clients()
        except Exception as e:
            initialization_errors.append(f"LLM: {e}")
            logger.error(f"Failed to initialize LLM clients: {e}")

        self._initialized = True

        if initialization_errors:
            logger.warning(f"Resource manager initialized with errors: {initialization_errors}")
        else:
            logger.info("All shared resources initialized successfully")

    async def _initialize_llm_clients(self):
        """Initialize LLM clients based on agent configurations."""
        # Get unique LLM models from agent configs
        llm_models = set()
        agents_config = self.config_manager.get('agents.agents', {})

        for agent_name, agent_config in agents_config.items():
            if isinstance(agent_config, dict) and agent_config.get('enabled'):
                llm_model = agent_config.get('llm_model', 'openai')
                llm_models.add(llm_model)

        # Create one client per model
        for model in llm_models:
            try:
                llm_config = self.config_manager.get_llm_config(model)
                self._llm_clients[model] = LLMClientFactory.create_client(llm_config)
                logger.info(f"LLM client for model '{model}' initialized")
            except Exception as e:
                logger.error(f"Failed to initialize LLM client for {model}: {e}")

    def _is_production(self) -> bool:
        """Check if running in production environment."""
        import os
        return os.getenv('ENVIRONMENT', 'development').lower() == 'production'

    async def shutdown(self):
        """Shutdown all shared resources gracefully."""
        logger.info("Shutting down shared resources...")

        shutdown_errors = []

        # Close ClickHouse pool
        if self._clickhouse_pool:
            try:
                close_clickhouse_pool()
                logger.info("ClickHouse pool closed")
            except Exception as e:
                shutdown_errors.append(f"ClickHouse: {e}")

        # Close Redis pool
        if self._redis_pool:
            try:
                self._redis_pool.disconnect()
                logger.info("Redis pool closed")
            except Exception as e:
                shutdown_errors.append(f"Redis: {e}")

        # Close session manager
        if self._session_manager:
            try:
                await self._session_manager.close()
                logger.info("Session manager closed")
            except Exception as e:
                shutdown_errors.append(f"SessionManager: {e}")

        self._initialized = False

        if shutdown_errors:
            logger.warning(f"Shutdown completed with errors: {shutdown_errors}")
        else:
            logger.info("All shared resources shutdown successfully")

    def get_resources_for_agent(self, agent_name: str,
                                user_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Get all resources needed for a specific agent with optional user context.
        Resources are configured in agents.yaml under 'required_resources'.
        """
        resources = {
            'config_manager': self.config_manager,
            'metrics': self._metrics
        }

        # Get agent configuration from agents.yaml
        agent_config = self.config_manager.get_agent_config(agent_name)
        
        # Get required resources from config, default to ['session'] if not specified
        needed_resources = agent_config.get('required_resources', ['session'])
        
        # Log what resources are being loaded for this agent
        logger.debug(f"Agent '{agent_name}' requires resources: {needed_resources}")

        # Add requested resources
        if 'clickhouse' in needed_resources:
            resources['clickhouse_pool'] = self._clickhouse_pool
            resources['clickhouse_client'] = self.get_clickhouse_client(use_pool=True)

        if 'redis' in needed_resources:
            resources['redis_client'] = self.get_redis_client()

        if 'session' in needed_resources:
            resources['session_manager'] = self.get_session_manager()

        if 'llm' in needed_resources:
            agent_config = self.config_manager.get_agent_config(agent_name)
            llm_model = agent_config.get('llm_model', 'openai')
            resources['llm_client'] = self.get_llm_client(llm_model)

        # Add user context if provided
        if user_context:
            resources['user_context'] = user_context
            user_id = user_context.get('user_id')
            if user_id:
                resources['user_semaphore'] = self._get_user_semaphore(user_id)

        return resources

    def _get_user_semaphore(self, user_id: str, max_concurrent: int = 5) -> asyncio.Semaphore:
        """Get or create a semaphore for user-level concurrency control."""
        if user_id not in self._user_semaphores:
            self._user_semaphores[user_id] = asyncio.Semaphore(max_concurrent)
        return self._user_semaphores[user_id]

    def get_clickhouse_client(self, use_pool: bool = True) -> ClickHouseClient:
        """Get a ClickHouse client instance."""
        ch_config = self.config_manager.get_database_config()
        return ClickHouseClient(ch_config, use_pool=use_pool)

    def get_clickhouse_pool(self):
        """Get the shared ClickHouse connection pool."""
        if not self._clickhouse_pool:
            ch_config = self.config_manager.get_database_config()
            self._clickhouse_pool = get_clickhouse_pool(ch_config)
        return self._clickhouse_pool

    def get_redis_client(self) -> redis.Redis:
        """Get a Redis client from the connection pool."""
        if not self._redis_client:
            raise RuntimeError("Redis client not initialized. Call initialize() first.")
        return self._redis_client

    def get_redis_pool(self) -> ConnectionPool:
        """Get the Redis connection pool."""
        if not self._redis_pool:
            raise RuntimeError("Redis pool not initialized. Call initialize() first.")
        return self._redis_pool

    def get_session_manager(self) -> Optional[PostgreSQLSessionManager]:
        """Get the shared session manager."""
        return self._session_manager

    def get_llm_client(self, provider: Optional[str] = None):
        """
        Get an LLM client for the specified provider.
        If no provider specified, returns the default provider client.
        """
        if not self._llm_clients:
            raise RuntimeError("LLM clients not initialized. Call initialize() first.")

        if provider:
            client = self._llm_clients.get(provider)
            if not client:
                # Create client on demand if not exists
                llm_config = self.config_manager.get_llm_config()
                provider_config = {**llm_config, 'provider': provider}
                client = LLMClientFactory.create_client(provider_config)
                self._llm_clients[provider] = client
            return client
        else:
            # Return default provider client
            llm_config = self.config_manager.get_llm_config()
            default_provider = llm_config.get('provider', 'openai')
            return self.get_llm_client(default_provider)


class ResourceMetrics:
    """Simple metrics collector for resource usage."""

    def __init__(self):
        self.query_count = 0
        self.error_count = 0
        self.user_metrics: Dict[str, Dict] = {}

    def record_query(self, user_id: str, duration: float, success: bool = True):
        """Record query metrics."""
        if user_id not in self.user_metrics:
            self.user_metrics[user_id] = {
                'query_count': 0,
                'error_count': 0,
                'total_duration': 0.0
            }

        self.user_metrics[user_id]['query_count'] += 1
        self.user_metrics[user_id]['total_duration'] += duration

        if not success:
            self.user_metrics[user_id]['error_count'] += 1
            self.error_count += 1

        self.query_count += 1
