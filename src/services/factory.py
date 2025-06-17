"""
Service Factory
Creates appropriate service implementations based on configuration
"""
import logging
from typing import Optional

from src.services.base.profile_analytics import BaseProfileAnalyticsService
from src.services.base.query_executor import BaseQueryExecutor
from src.services.executors.clickhouse_profile_executor import ClickHouseProfileQueryExecutor
from src.services.clickhouse.profile_analytics_service import ClickHouseProfileAnalyticsService
from src.core.config_manager import ConfigManager
from src.core.session_manager import EnhancedSessionManager

logger = logging.getLogger(__name__)


class ServiceFactory:
    """
    Factory for creating service instances
    Supports different database engines and configurations
    """
    
    @staticmethod
    async def create_profile_analytics_service(
        engine_type: str,
        config_manager: ConfigManager,
        enable_query_history: bool = True
    ) -> BaseProfileAnalyticsService:
        """
        Create a profile analytics service for the specified engine
        
        Args:
            engine_type: Type of engine ('clickhouse', 'spark', etc.)
            config_manager: Configuration manager instance
            enable_query_history: Whether to enable query history storage
            
        Returns:
            Profile analytics service instance
            
        Raises:
            ValueError: If engine type is not supported
        """
        logger.info(f"Creating profile analytics service for engine: {engine_type}")
        
        if engine_type.lower() == "clickhouse":
            # Get ClickHouse configuration
            ch_config = config_manager.get('database.databases.clickhouse')
            if not ch_config:
                raise ValueError("ClickHouse configuration not found")
            
            # Create ClickHouse query executor
            executor = ClickHouseProfileQueryExecutor(ch_config)
            
            # Create session manager for query history if enabled
            session_manager = None
            if enable_query_history:
                pg_config = config_manager.get('database.databases.postgresql')
                if pg_config:
                    session_manager = EnhancedSessionManager(pg_config)
                    await session_manager.initialize()
                else:
                    logger.warning("PostgreSQL config not found, query history disabled")
            
            # Create and return ClickHouse service
            service = ClickHouseProfileAnalyticsService(
                query_executor=executor,
                session_manager=session_manager
            )
            
            return service
            
        elif engine_type.lower() == "spark":
            # TODO: Implement Spark support
            raise NotImplementedError("Spark engine not yet implemented")
            
        elif engine_type.lower() == "postgres":
            # TODO: Implement PostgreSQL support for smaller datasets
            raise NotImplementedError("PostgreSQL engine not yet implemented")
            
        else:
            raise ValueError(f"Unsupported engine type: {engine_type}")
    
    @staticmethod
    async def create_query_executor(
        engine_type: str,
        config: dict
    ) -> BaseQueryExecutor:
        """
        Create a query executor for the specified engine
        
        Args:
            engine_type: Type of engine
            config: Engine configuration
            
        Returns:
            Query executor instance
        """
        if engine_type.lower() == "clickhouse":
            return ClickHouseProfileQueryExecutor(config)
        else:
            raise ValueError(f"Unsupported engine type: {engine_type}")


class ProfileAnalyticsServiceManager:
    """
    Manages profile analytics service instances
    Provides caching and lifecycle management
    """
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self._services = {}
        self._default_engine = None
    
    async def get_service(
        self, 
        engine_type: Optional[str] = None
    ) -> BaseProfileAnalyticsService:
        """
        Get or create a profile analytics service
        
        Args:
            engine_type: Engine type (None = use default)
            
        Returns:
            Profile analytics service instance
        """
        # Determine engine type
        if not engine_type:
            if not self._default_engine:
                # Get default from config
                self._default_engine = self.config_manager.get(
                    'services.default_engine', 
                    'clickhouse'
                )
            engine_type = self._default_engine
        
        # Check cache
        if engine_type in self._services:
            return self._services[engine_type]
        
        # Create new service
        service = await ServiceFactory.create_profile_analytics_service(
            engine_type=engine_type,
            config_manager=self.config_manager,
            enable_query_history=True
        )
        
        # Initialize and cache
        await service.initialize()
        self._services[engine_type] = service
        
        return service
    
    async def close_all(self):
        """Close all service connections"""
        for service in self._services.values():
            try:
                if hasattr(service, 'executor') and service.executor:
                    await service.executor.close()
                if hasattr(service, 'session_manager') and service.session_manager:
                    await service.session_manager.close()
            except Exception as e:
                logger.error(f"Error closing service: {e}")
        
        self._services.clear()