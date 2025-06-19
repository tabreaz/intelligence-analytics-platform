# src/core/database/clickhouse_client.py
import asyncio
import clickhouse_connect
from typing import List, Tuple, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import logging
from src.core.logger import get_logger
from ..config_manager import DatabaseConfig

logger = get_logger(__name__)


class ClickHouseClient:
    """ClickHouse database client with async support"""
    
    # Class-level thread pool shared by all instances
    _executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="clickhouse")

    def __init__(self, config: DatabaseConfig, use_pool: bool = True):
        self.config = config
        self.use_pool = use_pool
        self.client = None
        self._pool = None
        
        if use_pool:
            # Use connection pool for concurrent access
            from .clickhouse_pool import get_clickhouse_pool
            self._pool = get_clickhouse_pool(config)
        else:
            # Single connection mode (legacy)
            self._connect()

    def _connect(self):
        """Connect to ClickHouse"""
        try:
            self.client = clickhouse_connect.get_client(
                host=self.config.host,
                port=self.config.port,
                username=self.config.user,
                password=self.config.password,
                database=self.config.database,
                secure=self.config.secure,
                compress=True
            )

            # Test connection
            self.client.ping()
            logger.info(f"Connected to ClickHouse: {self.config.host}:{self.config.port}")

        except Exception as e:
            logger.error(f"Failed to connect to ClickHouse: {e}")
            raise

    def execute(self, query: str, parameters: Optional[dict] = None) -> List[Tuple]:
        """Execute query and return results"""
        try:
            logger.debug(f"Executing ClickHouse query: {query[:500]}...")
            result = self.client.query(query, parameters=parameters)
            logger.debug(f"Query returned {len(result.result_rows)} rows")
            return result.result_rows
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            logger.error(f"Failed query: {query}")
            raise
    
    def execute_with_params(self, query: str, params: dict) -> List[Tuple]:
        """Execute parameterized query (native support)"""
        try:
            logger.debug(f"Executing parameterized ClickHouse query: {query[:500]}...")
            # ClickHouse-connect supports parameters natively
            result = self.client.query(query, parameters=params)
            logger.debug(f"Query returned {len(result.result_rows)} rows")
            return result.result_rows
        except Exception as e:
            logger.error(f"Parameterized query execution failed: {e}")
            logger.error(f"Failed query: {query}")
            logger.error(f"Parameters: {params}")
            raise

    def insert(self, table: str, data: List[dict], column_names: List[str] = None):
        """Insert data into table (synchronous wrapper)"""
        try:
            if self.use_pool and self._pool:
                # For pool mode, we need to use async
                # Create a new event loop if not in async context
                try:
                    loop = asyncio.get_running_loop()
                    # We're already in an async context - this shouldn't happen
                    # for a sync method, but handle it gracefully
                    raise RuntimeError("Cannot call synchronous insert from async context when using pool")
                except RuntimeError:
                    # No event loop running - create one for this operation
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(self._insert_async_internal(table, data, column_names))
                    finally:
                        loop.close()
                        asyncio.set_event_loop(None)
            elif self.client:
                # Use single connection (synchronous)
                if column_names:
                    self.client.insert(table, data, column_names=column_names)
                else:
                    self.client.insert(table, data)
                logger.info(f"Inserted {len(data)} rows into {table}")
            else:
                raise RuntimeError("No ClickHouse connection available")
        except Exception as e:
            logger.error(f"Insert failed with error: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            if hasattr(e, 'message'):
                logger.error(f"Error message: {e.message}")
            raise
    
    async def _insert_async_internal(self, table: str, data: List[dict], column_names: List[str] = None):
        """Internal async insert method for pool operations"""
        async with self._pool.get_connection() as conn:
            if column_names:
                conn.insert(table, data, column_names=column_names)
            else:
                conn.insert(table, data)
            logger.info(f"Inserted {len(data)} rows into {table}")

    def close(self):
        """Close connection"""
        if self.client:
            self.client.close()
    
    # Async methods for non-blocking operations
    async def execute_async(self, query: str, parameters: Optional[dict] = None) -> List[Tuple]:
        """Execute query asynchronously and return results"""
        if self.use_pool and self._pool:
            # Use connection pool
            return await self._pool.execute_async(query, parameters)
        else:
            # Use thread pool for single connection
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self._executor,
                self.execute,
                query,
                parameters
            )
    
    async def execute_with_params_async(self, query: str, params: dict) -> List[Tuple]:
        """Execute parameterized query asynchronously"""
        if self.use_pool and self._pool:
            # Use connection pool with parameters
            return await self._pool.execute_async(query, params)
        else:
            # Use thread pool for single connection
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self._executor,
                self.execute_with_params,
                query,
                params
            )
    
    async def insert_async(self, table: str, data: List[dict], column_names: List[str] = None):
        """Insert data into table asynchronously"""
        if self.use_pool and self._pool:
            # Use the async pool directly
            await self._insert_async_internal(table, data, column_names)
        else:
            # For single connection, use thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self._executor,
                lambda: self.insert(table, data, column_names)
            )
    
    @classmethod
    def shutdown_executor(cls):
        """Shutdown the shared thread pool executor"""
        cls._executor.shutdown(wait=True)
