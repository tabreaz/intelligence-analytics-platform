"""
ClickHouse Profile Query Executor Implementation
Specialized for executing profile-only queries against phone_imsi_uid_latest table
"""
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager

from src.services.base.query_executor import BaseQueryExecutor
from src.core.database.clickhouse_client import ClickHouseClient

logger = logging.getLogger(__name__)


class ClickHouseProfileQueryExecutor(BaseQueryExecutor):
    """
    ClickHouse implementation of query executor for profile queries
    Optimized for phone_imsi_uid_latest table operations
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ClickHouse executor
        
        Args:
            config: ClickHouse configuration dictionary
        """
        self.config = config
        self.client = None
        self._initialized = False
        self._lock = asyncio.Lock()  # For thread-safe initialization
    
    async def initialize(self) -> None:
        """Initialize ClickHouse connection with thread safety"""
        async with self._lock:
            if self._initialized:
                return
                
            # Import DatabaseConfig here to avoid circular imports
            from src.core.config_manager import DatabaseConfig
            
            # Create DatabaseConfig from dictionary
            db_config = DatabaseConfig(**self.config)
            
            self.client = ClickHouseClient(db_config)
            
            # If ClickHouseClient has async connection method, use it
            if hasattr(self.client, 'connect'):
                await self.client.connect()
            
            self._initialized = True
            logger.info("ClickHouse profile query executor initialized")
    
    async def execute_query(
        self, 
        sql: str, 
        params: Optional[Union[Dict[str, Any], List[Any]]] = None
    ) -> List[Any]:
        """
        Execute a query and return results
        
        Args:
            sql: SQL query with optional placeholders
            params: Parameters for the query (dict for named, list for positional)
        """
        await self.initialize()
        
        try:
            # Check if client supports parameterized queries
            if params and hasattr(self.client, 'execute_with_params_async'):
                # Use native parameterized queries (more secure)
                result = await self.client.execute_with_params_async(sql, params)
            else:
                # If parameterized queries not supported, at least sanitize
                if params:
                    sql = self._substitute_params(sql, params)
                result = await self.client.execute_async(sql)
            
            return result
        except Exception as e:
            logger.error(f"ClickHouse query execution failed: {e}")
            logger.debug(f"Query: {sql[:200]}...")  # Log first 200 chars
            raise
    
    def _substitute_params(self, sql: str, params: Union[Dict[str, Any], List[Any]]) -> str:
        """
        Safely substitute parameters into SQL query
        WARNING: This is a fallback - prefer native parameterized queries
        """
        if isinstance(params, dict):
            # Named parameters
            for key, value in params.items():
                placeholder = f":{key}"
                if placeholder in sql:
                    sql = sql.replace(placeholder, self.format_value(value))
        else:
            # Positional parameters - replace ? placeholders
            for value in params:
                sql = sql.replace('?', self.format_value(value), 1)
        
        return sql
    
    async def execute_count_query(self, sql: str, params: Optional[Dict[str, Any]] = None) -> int:
        """Execute a count query and return the count"""
        result = await self.execute_query(sql, params)
        
        if result and len(result) > 0 and len(result[0]) > 0:
            try:
                return int(result[0][0])
            except (ValueError, TypeError):
                logger.error(f"Could not convert count result to int: {result[0][0]}")
                return 0
        return 0
    
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
            Tuple of (headers, data rows)
        """
        await self.initialize()
        
        try:
            # Execute the query to get data
            data = await self.execute_query(sql, params)
            
            # If column names are provided, use them
            if column_names:
                headers = column_names
            else:
                # Try to extract column names from the SQL query
                headers = self._extract_column_names_from_sql(sql)
            
            # If we couldn't get headers, generate generic ones
            if not headers and data and len(data) > 0:
                headers = [f"column_{i}" for i in range(len(data[0]))]
            elif not headers:
                headers = []
            
            return headers, data if data else []
            
        except Exception as e:
            logger.error(f"Failed to execute query with headers: {e}")
            raise
    
    def _extract_column_names_from_sql(self, sql: str) -> List[str]:
        """
        Extract column names from SELECT statement
        This is a simple parser that handles common cases
        """
        import re
        
        try:
            # Remove comments and normalize whitespace
            sql_clean = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
            sql_clean = re.sub(r'/\*.*?\*/', '', sql_clean, flags=re.DOTALL)
            sql_clean = ' '.join(sql_clean.split())
            
            # Extract SELECT clause
            select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql_clean, re.IGNORECASE)
            if not select_match:
                return []
            
            select_clause = select_match.group(1)
            headers = []
            
            # Split by commas but handle nested functions
            parts = []
            current = ''
            paren_depth = 0
            
            for char in select_clause:
                if char == '(' :
                    paren_depth += 1
                elif char == ')':
                    paren_depth -= 1
                elif char == ',' and paren_depth == 0:
                    parts.append(current.strip())
                    current = ''
                    continue
                current += char
            
            if current.strip():
                parts.append(current.strip())
            
            # Extract column names or aliases
            for part in parts:
                # Check for alias (AS or as)
                alias_match = re.search(r'\s+[Aa][Ss]\s+([`"]?)(\w+)\1\s*$', part)
                if alias_match:
                    headers.append(alias_match.group(2))
                else:
                    # Extract the column name
                    # Remove backticks, quotes
                    part = part.strip('`"')
                    # If it's a function call, use the whole expression
                    if '(' in part:
                        # Try to get a simple name
                        func_match = re.match(r'(\w+)\s*\(', part)
                        if func_match:
                            headers.append(func_match.group(1).lower())
                        else:
                            headers.append(part.split('(')[0])
                    else:
                        # Simple column name
                        headers.append(part.split('.')[-1])  # Handle table.column
            
            return headers
            
        except Exception as e:
            logger.debug(f"Failed to extract column names from SQL: {e}")
            return []
    
    async def execute_batch(
        self, 
        queries: List[str], 
        transaction: bool = False
    ) -> List[Optional[List[Any]]]:
        """
        Execute multiple queries in batch
        
        Args:
            queries: List of SQL queries
            transaction: Whether to wrap in a transaction (if supported)
        """
        results = []
        
        if transaction and hasattr(self.client, 'begin_transaction'):
            # Use transaction if available
            async with self._transaction():
                for query in queries:
                    try:
                        result = await self.execute_query(query)
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Batch query failed: {e}")
                        raise  # Rollback transaction
        else:
            # Execute sequentially without transaction
            for query in queries:
                try:
                    result = await self.execute_query(query)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Batch query failed: {e}")
                    results.append(None)
        
        return results
    
    @asynccontextmanager
    async def _transaction(self):
        """Context manager for transactions (if supported)"""
        try:
            if hasattr(self.client, 'begin_transaction'):
                await self.client.begin_transaction()
            yield
            if hasattr(self.client, 'commit'):
                await self.client.commit()
        except Exception:
            if hasattr(self.client, 'rollback'):
                await self.client.rollback()
            raise
    
    def get_engine_type(self) -> str:
        """Return the engine type"""
        return "clickhouse"
    
    def quote_identifier(self, identifier: str) -> str:
        """
        Quote an identifier for ClickHouse
        Handles special characters and reserved words
        """
        # Remove existing quotes if any
        identifier = identifier.strip('`"')
        
        # Always quote to be safe
        return f"`{identifier}`"
    
    def format_value(self, value: Any) -> str:
        """
        Format a value for ClickHouse SQL
        IMPORTANT: This should only be used as a last resort!
        Prefer parameterized queries for security.
        """
        if isinstance(value, str):
            # Proper SQL escaping - double single quotes
            escaped = value.replace("'", "''")
            return f"'{escaped}'"
        elif isinstance(value, datetime):
            # ClickHouse datetime format
            return f"'{value.strftime('%Y-%m-%d %H:%M:%S')}'"
        elif isinstance(value, (list, tuple)):
            # Format array values
            formatted_items = [self.format_value(v) for v in value]
            return f"[{', '.join(formatted_items)}]"
        elif value is None:
            return "NULL"
        elif isinstance(value, bool):
            return "1" if value else "0"
        elif isinstance(value, (int, float)):
            return str(value)
        else:
            # For unknown types, convert to string and escape
            return self.format_value(str(value))
    
    async def test_connection(self) -> bool:
        """Test if connection is alive"""
        try:
            await self.initialize()
            result = await self.execute_query("SELECT 1 AS test")
            return bool(result and len(result) > 0)
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    async def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """
        Get information about a table
        
        Returns:
            Dictionary with table metadata
        """
        await self.initialize()
        
        # Parse table name (might include database)
        parts = table_name.split('.')
        if len(parts) == 2:
            database, table = parts
        else:
            database = "currentDatabase()"
            table = parts[0]
        
        # Get column information
        columns_query = f"""
            SELECT 
                name,
                type,
                default_kind,
                default_expression,
                comment
            FROM system.columns
            WHERE database = {database if database != "currentDatabase()" else database}
            AND table = '{table}'
            ORDER BY position
        """
        
        columns = await self.execute_query(columns_query)
        
        # Get table engine and other metadata
        table_query = f"""
            SELECT 
                engine,
                total_rows,
                total_bytes,
                metadata_modification_time
            FROM system.tables
            WHERE database = {database if database != "currentDatabase()" else database}
            AND name = '{table}'
        """
        
        table_info = await self.execute_query(table_query)
        
        return {
            'columns': [
                {
                    'name': col[0],
                    'type': col[1],
                    'default_kind': col[2],
                    'default_expression': col[3],
                    'comment': col[4]
                }
                for col in columns
            ],
            'engine': table_info[0][0] if table_info else None,
            'total_rows': table_info[0][1] if table_info else None,
            'total_bytes': table_info[0][2] if table_info else None,
            'last_modified': table_info[0][3] if table_info else None
        }
    
    async def close(self) -> None:
        """Close connection and cleanup resources"""
        async with self._lock:
            if self.client:
                try:
                    if hasattr(self.client, 'close'):
                        await self.client.close()
                    else:
                        # Synchronous close
                        self.client.close()
                except Exception as e:
                    logger.error(f"Error closing ClickHouse connection: {e}")
                finally:
                    self.client = None
                    self._initialized = False
                    logger.info("ClickHouse connection closed")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()