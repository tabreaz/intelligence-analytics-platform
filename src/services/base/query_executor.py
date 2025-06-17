"""
Base Query Executor Interface
Abstract interface for database query execution
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime


class BaseQueryExecutor(ABC):
    """
    Abstract base class for query executors
    Defines interface for executing queries against different engines
    """
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the query executor (e.g., establish connections)"""
        pass
    
    @abstractmethod
    async def execute_query(self, sql: str, params: Optional[Dict[str, Any]] = None) -> List[Any]:
        """
        Execute a query and return results
        
        Args:
            sql: SQL query to execute
            params: Optional parameters for parameterized queries
            
        Returns:
            Query results as list of rows
        """
        pass
    
    @abstractmethod
    async def execute_count_query(self, sql: str, params: Optional[Dict[str, Any]] = None) -> int:
        """
        Execute a count query and return the count
        
        Args:
            sql: Count query to execute
            params: Optional parameters
            
        Returns:
            Count value
        """
        pass
    
    @abstractmethod
    async def execute_query_with_headers(
        self, 
        sql: str, 
        params: Optional[Dict[str, Any]] = None,
        column_names: Optional[List[str]] = None
    ) -> Tuple[List[str], List[List[Any]]]:
        """
        Execute query and return headers and data separately
        
        Args:
            sql: SQL query to execute
            params: Optional parameters
            column_names: Optional list of column names (if known)
            
        Returns:
            Tuple of (headers, data_rows)
        """
        pass
    
    @abstractmethod
    async def execute_batch(self, queries: List[str]) -> List[Any]:
        """
        Execute multiple queries in batch
        
        Args:
            queries: List of SQL queries
            
        Returns:
            List of results for each query
        """
        pass
    
    @abstractmethod
    def get_engine_type(self) -> str:
        """Return the engine type (e.g., 'clickhouse', 'spark', 'postgres')"""
        pass
    
    @abstractmethod
    def quote_identifier(self, identifier: str) -> str:
        """
        Quote an identifier according to engine rules
        
        Args:
            identifier: Column or table name to quote
            
        Returns:
            Properly quoted identifier
        """
        pass
    
    @abstractmethod
    def format_value(self, value: Any) -> str:
        """
        Format a value for use in SQL according to engine rules
        
        Args:
            value: Value to format
            
        Returns:
            Properly formatted value for SQL
        """
        pass
    
    @abstractmethod
    async def test_connection(self) -> bool:
        """
        Test if connection is alive
        
        Returns:
            True if connection is valid
        """
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close connections and cleanup resources"""
        pass