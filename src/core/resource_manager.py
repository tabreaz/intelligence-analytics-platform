# /src/core/resource_manager.py
"""
Shared Resource Manager for connection pooling and resource reuse.
This manager ensures efficient resource utilization across all agents.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Set
from datetime import datetime
from contextlib import asynccontextmanager

import redis
from redis.connection import ConnectionPool

from src.core.database.clickhouse_pool import get_clickhouse_pool, close_clickhouse_pool
from src.core.database.clickhouse_client import ClickHouseClient
from src.core.session_manager import EnhancedSessionManager
from src.core.llm.base_llm import LLMClientFactory
from src.core.config_manager import ConfigManager
from src.core.metrics_logger import MetricsLogger

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
        
        # Initialize metrics logger instead of in-memory metrics
        metrics_dir = config_manager.get('resource_manager.metrics_dir', 'logs/metrics')
        buffer_size = config_manager.get('resource_manager.metrics_buffer_size', 100)
        flush_interval = config_manager.get('resource_manager.metrics_flush_interval', 30)
        self._metrics = MetricsLogger(metrics_dir, buffer_size, flush_interval)
        
        # Track resource usage
        self._active_agents: Set[str] = set()
        self._resource_locks = {}
        
        # Track background tasks for cleanup
        self._background_tasks: Set[asyncio.Task] = set()

    async def initialize(self):
        """Initialize all shared resources with proper error handling."""
        if self._initialized:
            return

        logger.info("Initializing shared resources...")

        initialization_errors = []

        # Initialize ClickHouse
        try:
            ch_config = self.config_manager.get_database_config('clickhouse')
            # Get pool size from config or use default based on environment
            default_pool_size = 50 if ResourceManager._is_production() else 10
            pool_size = self.config_manager.get('resource_manager.clickhouse_pool_size', default_pool_size)
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
                retry_on_timeout=redis_config.retry_on_timeout,
                socket_keepalive=True,
                socket_keepalive_options={}
            )
            self._redis_client = redis.Redis(connection_pool=self._redis_pool)
            # Test connection (Redis-py ping is synchronous)
            self._redis_client.ping()
            logger.info("Redis pool initialized")
        except Exception as e:
            initialization_errors.append(f"Redis: {e}")
            logger.error(f"Failed to initialize Redis pool: {e}")

        # Initialize PostgreSQL session manager
        try:
            pg_config = self.config_manager.get_database_config('postgresql')
            self._session_manager = EnhancedSessionManager(pg_config.__dict__)
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

        # Start metrics logger
        try:
            await self._metrics.start()
            logger.info("Metrics logger started")
        except Exception as e:
            initialization_errors.append(f"Metrics: {e}")
            logger.error(f"Failed to start metrics logger: {e}")
        
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

    @staticmethod
    def _is_production() -> bool:
        """Check if running in production environment."""
        import os
        return os.getenv('ENVIRONMENT', 'development').lower() == 'production'

    async def shutdown(self):
        """Shutdown all shared resources gracefully."""
        logger.info("Shutting down shared resources...")
        
        # Wait for active agents to complete
        if self._active_agents:
            logger.warning(f"Waiting for {len(self._active_agents)} active agents to complete...")
            # Wait up to 30 seconds for agents to complete
            wait_time = 0
            while self._active_agents and wait_time < 30:
                await asyncio.sleep(1)
                wait_time += 1
            
            if self._active_agents:
                logger.error(f"Force shutdown with {len(self._active_agents)} agents still active: {self._active_agents}")

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
                
        # Stop metrics logger
        try:
            await self._metrics.stop()
            logger.info("Metrics logger stopped")
        except Exception as e:
            shutdown_errors.append(f"Metrics: {e}")
            
        # Wait for any remaining background tasks
        await self.wait_for_background_tasks(timeout=10.0)

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
    
    @asynccontextmanager
    async def track_agent_usage(self, agent_name: str, user_id: Optional[str] = None):
        """Context manager to track agent resource usage with fire-and-forget metrics."""
        self._active_agents.add(agent_name)
        start_time = datetime.now()
        try:
            yield
            # Record successful execution (fire-and-forget)
            duration = (datetime.now() - start_time).total_seconds()
            
            # Fire-and-forget metrics recording
            async def record_success():
                await self._metrics.record_agent_execution(agent_name, duration, success=True)
                if user_id:
                    await self._metrics.record_query(user_id, duration, success=True, agent_name=agent_name)
                    
            self.fire_and_forget(record_success(), task_name=f"metrics_{agent_name}_success")
            
        except Exception as e:
            # Record failed execution (fire-and-forget)
            duration = (datetime.now() - start_time).total_seconds()
            
            # Fire-and-forget error recording
            async def record_failure():
                await self._metrics.record_agent_execution(agent_name, duration, success=False, error=str(e))
                if user_id:
                    await self._metrics.record_query(user_id, duration, success=False, agent_name=agent_name)
                await self._metrics.record_error('agent_execution', str(e), user_id=user_id, agent_name=agent_name)
                
            self.fire_and_forget(record_failure(), task_name=f"metrics_{agent_name}_error")
            raise
        finally:
            self._active_agents.discard(agent_name)

    def get_clickhouse_client(self, use_pool: bool = True) -> ClickHouseClient:
        """Get a ClickHouse client instance. Always uses pool by default for better performance."""
        if not self._clickhouse_pool and use_pool:
            raise RuntimeError("ClickHouse pool not initialized. Call initialize() first.")
        
        ch_config = self.config_manager.get_database_config('clickhouse')
        # Always prefer pool usage for agents
        return ClickHouseClient(ch_config, use_pool=use_pool)

    def get_clickhouse_pool(self):
        """Get the shared ClickHouse connection pool."""
        if not self._clickhouse_pool:
            ch_config = self.config_manager.get_database_config('clickhouse')
            self._clickhouse_pool = get_clickhouse_pool(ch_config)
        return self._clickhouse_pool

    def get_redis_client(self) -> Optional[redis.Redis]:
        """Get a Redis client from the connection pool. Returns None if Redis initialization failed."""
        return self._redis_client  # Can be None if Redis init failed

    def get_redis_pool(self) -> ConnectionPool:
        """Get the Redis connection pool."""
        if not self._redis_pool:
            raise RuntimeError("Redis pool not initialized. Call initialize() first.")
        return self._redis_pool

    def get_session_manager(self) -> Optional[EnhancedSessionManager]:
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
                llm_config = self.config_manager.get_llm_config(provider)
                client = LLMClientFactory.create_client(llm_config)
                self._llm_clients[provider] = client
            return client
        else:
            # Return default provider client
            default_provider = self.config_manager.get('llm.provider', 'openai')
            return self.get_llm_client(default_provider)
            
    async def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get metrics summary for the specified time window."""
        return await self._metrics.get_summary(hours)
        
    async def cleanup_old_metrics(self, days_to_keep: int = 7):
        """Clean up old metrics files."""
        return await self._metrics.cleanup_old_files(days_to_keep)
        
    def fire_and_forget(self, coro, task_name: Optional[str] = None):
        """
        Execute a coroutine in the background without waiting for completion.
        
        This is a centralized fire-and-forget pattern that:
        - Tracks background tasks to prevent them from being garbage collected
        - Logs errors from background tasks
        - Cleans up completed tasks automatically
        
        Args:
            coro: The coroutine to execute
            task_name: Optional name for logging purposes
            
        Example:
            self.fire_and_forget(
                self._metrics.record_agent_execution(agent_name, duration),
                task_name="record_metrics"
            )
        """
        task = asyncio.create_task(self._wrap_background_task(coro, task_name))
        self._background_tasks.add(task)
        # Remove task from set when complete
        task.add_done_callback(self._background_tasks.discard)
        
    async def _wrap_background_task(self, coro, task_name: Optional[str] = None):
        """Wrapper for background tasks to handle errors."""
        try:
            await coro
        except asyncio.CancelledError:
            # Task was cancelled, this is expected during shutdown
            if task_name:
                logger.debug(f"Background task '{task_name}' was cancelled")
        except Exception as e:
            # Log errors from background tasks
            error_msg = f"Error in background task"
            if task_name:
                error_msg += f" '{task_name}'"
            logger.error(f"{error_msg}: {e}", exc_info=True)
            
    async def wait_for_background_tasks(self, timeout: float = 5.0):
        """Wait for all background tasks to complete with timeout."""
        if not self._background_tasks:
            return
            
        logger.info(f"Waiting for {len(self._background_tasks)} background tasks to complete...")
        
        # Wait with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._background_tasks, return_exceptions=True),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for background tasks after {timeout}s")
            # Cancel remaining tasks
            for task in self._background_tasks:
                if not task.done():
                    task.cancel()

