"""
ClickHouse Connection Pool Implementation
Manages multiple connections to handle concurrent requests
"""
import asyncio
import clickhouse_connect
from typing import List, Tuple, Any, Optional, Dict
from contextlib import asynccontextmanager
from queue import Queue, Empty
import logging
from threading import Lock
from ..config_manager import DatabaseConfig

logger = logging.getLogger(__name__)


class ClickHouseConnectionPool:
    """
    Connection pool for ClickHouse to handle concurrent queries
    Each connection can only handle one query at a time
    """
    
    def __init__(self, config: DatabaseConfig, pool_size: int = 10):
        """
        Initialize connection pool
        
        Args:
            config: Database configuration
            pool_size: Number of connections to maintain
        """
        self.config = config
        self.pool_size = pool_size
        self._pool = Queue(maxsize=pool_size)
        self._lock = Lock()
        self._closed = False
        self._initialize_pool()
        
    def _initialize_pool(self):
        """Create initial connections"""
        logger.info(f"Initializing ClickHouse connection pool with {self.pool_size} connections")
        
        for i in range(self.pool_size):
            try:
                conn = self._create_connection()
                self._pool.put(conn)
            except Exception as e:
                logger.error(f"Failed to create connection {i}: {e}")
                # Continue with fewer connections rather than failing completely
                
        actual_size = self._pool.qsize()
        if actual_size == 0:
            raise Exception("Failed to create any ClickHouse connections")
            
        logger.info(f"ClickHouse pool initialized with {actual_size} connections")
    
    def _create_connection(self):
        """Create a new ClickHouse connection"""
        return clickhouse_connect.get_client(
            host=self.config.host,
            port=self.config.port,
            username=self.config.user,
            password=self.config.password,
            database=self.config.database,
            secure=self.config.secure,
            compress=True
        )
    
    @asynccontextmanager
    async def get_connection(self, timeout: float = 5.0):
        """
        Get a connection from the pool
        
        Args:
            timeout: Maximum time to wait for a connection
            
        Yields:
            ClickHouse connection
        """
        if self._closed:
            raise Exception("Connection pool is closed")
            
        connection = None
        start_time = asyncio.get_event_loop().time()
        
        while True:
            try:
                # Try to get a connection without blocking
                connection = self._pool.get_nowait()
                break
            except Empty:
                # No connections available
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= timeout:
                    raise Exception(f"Timeout waiting for connection after {timeout}s")
                
                # Wait a bit before trying again
                await asyncio.sleep(0.1)
        
        try:
            # Verify connection is still alive
            connection.ping()
            yield connection
        except Exception as e:
            # Connection is dead, create a new one
            logger.warning(f"Dead connection detected: {e}")
            try:
                connection = self._create_connection()
                yield connection
            except Exception as create_error:
                logger.error(f"Failed to create replacement connection: {create_error}")
                raise
        finally:
            # Return connection to pool
            if connection and not self._closed:
                self._pool.put(connection)
    
    async def execute_async(self, query: str, parameters: Optional[dict] = None) -> List[Tuple]:
        """
        Execute a query using a pooled connection
        
        Args:
            query: SQL query to execute
            parameters: Query parameters
            
        Returns:
            Query results
        """
        async with self.get_connection() as conn:
            logger.debug(f"Executing query: {query[:200]}...")
            
            # Run the synchronous query in a thread to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: conn.query(query, parameters=parameters)
            )
            
            logger.debug(f"Query returned {len(result.result_rows)} rows")
            return result.result_rows
    
    async def insert_async(self, table: str, data: List[dict], column_names: List[str] = None):
        """
        Insert data into table using a pooled connection
        
        Args:
            table: Table name
            data: List of dictionaries to insert
            column_names: Optional column names
        """
        async with self.get_connection() as conn:
            logger.debug(f"Inserting {len(data)} rows into {table}")
            
            # Run the synchronous insert in a thread to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: conn.insert(table, data, column_names=column_names) if column_names else conn.insert(table, data)
            )
            
            logger.info(f"Inserted {len(data)} rows into {table}")
    
    def close(self):
        """Close all connections in the pool"""
        self._closed = True
        
        with self._lock:
            closed_count = 0
            while not self._pool.empty():
                try:
                    conn = self._pool.get_nowait()
                    conn.close()
                    closed_count += 1
                except Exception as e:
                    logger.error(f"Error closing connection: {e}")
                    
            logger.info(f"Closed {closed_count} connections in pool")


# Global pool instance (created on first use)
_global_pool: Optional[ClickHouseConnectionPool] = None
_pool_lock = Lock()


def get_clickhouse_pool(config: DatabaseConfig, pool_size: int = 10) -> ClickHouseConnectionPool:
    """
    Get or create the global ClickHouse connection pool
    
    Args:
        config: Database configuration
        pool_size: Pool size (only used on first call)
        
    Returns:
        ClickHouse connection pool
    """
    global _global_pool
    
    with _pool_lock:
        if _global_pool is None:
            _global_pool = ClickHouseConnectionPool(config, pool_size)
        return _global_pool


def close_clickhouse_pool():
    """Close the global connection pool"""
    global _global_pool
    
    with _pool_lock:
        if _global_pool:
            _global_pool.close()
            _global_pool = None