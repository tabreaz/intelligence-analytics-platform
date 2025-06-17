"""
ClickHouse query engine implementation
"""
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import hashlib
import logging

from ..base import QueryEngine
from ...models import (
    EngineType, QueryType, GeneratedQuery, QueryPlan,
    QueryExecutorRequest
)
from .translator import ClickHouseFilterTranslator
from .builder import ClickHouseQueryBuilder
from src.core.database.schema_manager import ClickHouseSchemaManager

logger = logging.getLogger(__name__)


class ClickHouseEngine(QueryEngine):
    """ClickHouse query engine implementation"""
    
    def _get_engine_type(self) -> EngineType:
        return EngineType.CLICKHOUSE
    
    def _load_schema(self) -> Dict[str, Any]:
        """Load ClickHouse schema information from schema manager"""
        from src.core.database.schema_manager import ClickHouseSchemaManager
        
        schema_manager = ClickHouseSchemaManager()
        # Get the raw ClickHouse schema
        return schema_manager.SCHEMAS
    
    def quote_identifier(self, identifier: str) -> str:
        """ClickHouse uses backticks for identifiers"""
        return f"`{identifier}`"
    
    def format_value(self, value: Any, field_type: str = None) -> str:
        """Format values for ClickHouse"""
        if isinstance(value, str):
            # Escape single quotes
            escaped = value.replace("'", "\\'")
            return f"'{escaped}'"
        elif isinstance(value, datetime):
            return f"'{value.strftime('%Y-%m-%d %H:%M:%S')}'"
        elif isinstance(value, list):
            # Format array values
            formatted_items = [self.format_value(v) for v in value]
            return f"[{', '.join(formatted_items)}]"
        elif value is None:
            return "NULL"
        elif isinstance(value, bool):
            return "1" if value else "0"
        else:
            return str(value)
    
    def generate_query(self, request: QueryExecutorRequest) -> GeneratedQuery:
        """Generate ClickHouse query based on request"""
        translator = ClickHouseFilterTranslator(self)
        builder = ClickHouseQueryBuilder(self, translator)
        
        # Get unified filter tree
        unified_tree = request.unified_filter_tree
        
        if request.query_type == QueryType.PROFILE_ONLY:
            # Extract filter tree for profile queries
            filter_tree = unified_tree.get('unified_filter_tree', {})
            query_str = builder.build_profile_query(
                filter_tree=filter_tree,
                select_fields=request.select_fields,
                limit=request.limit,
                offset=request.offset,
                order_by=request.order_by
            )
        else:
            # TODO: Implement location-based queries in Phase 2
            raise NotImplementedError(f"Query type {request.query_type} not yet implemented for ClickHouse")
        
        # Generate query hash for caching
        query_hash = hashlib.md5(query_str.encode()).hexdigest()
        
        return GeneratedQuery(
            query=query_str,
            parameters=[],  # ClickHouse queries are not parameterized in this implementation
            query_hash=query_hash,
            engine_type=self.engine_type,
            dialect_version="23.8"  # ClickHouse version
        )
    
    def estimate_query_plan(self, query: GeneratedQuery) -> QueryPlan:
        """Estimate ClickHouse query execution plan"""
        # Parse query to estimate
        query_lower = query.query.lower()
        has_array_ops = any(op in query_lower for op in ['arrayexists', 'has(', 'array_'])
        
        # Simple heuristics for estimation based on filter presence
        if 'imsi' in query_lower and ('=' in query_lower or 'in (' in query_lower):
            estimated_rows = "<100"
        elif 'nationality_code' in query_lower:
            estimated_rows = "10K-1M"
        elif has_array_ops:
            estimated_rows = "100K-10M"
        else:
            estimated_rows = "1M+"
        
        optimization_hints = []
        
        # Add optimization hints
        if 'event_date' not in query_lower:
            optimization_hints.append("Consider adding date filter to use partition pruning")
        
        if has_array_ops:
            optimization_hints.append("Array operations may be slower on large datasets")
        
        if 'order by' in query_lower and 'limit' not in query_lower:
            optimization_hints.append("Consider adding LIMIT to reduce sorting overhead")
        
        return QueryPlan(
            estimated_rows=estimated_rows,
            estimated_cost=None,  # ClickHouse doesn't provide cost estimates
            indexes_used=["primary"],  # ClickHouse always uses primary index
            partitions_scanned=None,  # Would need to run EXPLAIN
            optimization_hints=optimization_hints,
            warnings=[]
        )
    
    def validate_query(self, query: GeneratedQuery) -> Tuple[bool, Optional[str]]:
        """
        Validate generated query using sqlglot
        """
        if not query.query:
            return False, "Empty query"
        
        try:
            import sqlglot
            from sqlglot import parse_one, errors
            
            # Parse the query with ClickHouse dialect
            parsed = parse_one(query.query, dialect="clickhouse")
            
            # Get schema for validation
            schema_manager = ClickHouseSchemaManager()
            schema = schema_manager.get_schema_for_validation()
            
            # Validate using sqlglot
            from sqlglot.optimizer.qualify import qualify
            from sqlglot.optimizer.validate import validate
            
            # Qualify the query (add table references to columns)
            qualified = qualify(parsed, schema=schema, dialect="clickhouse")
            
            # Validate the query
            validate(qualified, schema=schema, dialect="clickhouse")
            
            logger.debug(f"Query validated successfully with sqlglot")
            return True, None
            
        except errors.ParseError as e:
            return False, f"SQL parse error: {str(e)}"
        except errors.OptimizeError as e:
            return False, f"SQL optimization error: {str(e)}"
        except Exception as e:
            # Log the error but don't fail - sqlglot might not support all ClickHouse features
            logger.warning(f"Sqlglot validation warning: {str(e)}")
            # Fall back to basic validation
            query_lower = query.query.lower()
            if 'select' not in query_lower or 'from' not in query_lower:
                return False, "Query missing SELECT or FROM clause"
            return True, None
    
    def get_table_name(self, query_type: QueryType) -> str:
        """Get ClickHouse table name with database"""
        base_table = super().get_table_name(query_type)
        database = self.config.get('database', 'telecom_db')
        return f"{database}.{base_table}"