"""
ClickHouse query builder
"""
from typing import Dict, Any, List, Optional
import logging

from ...models import QueryType
from src.core.database.schema_manager import ClickHouseSchemaManager
from ...field_mapper import FieldMapper

logger = logging.getLogger(__name__)


class ClickHouseQueryBuilder:
    """Builds ClickHouse queries from filter trees"""
    
    def __init__(self, engine, translator):
        self.engine = engine
        self.translator = translator
        # Load schema for field validation
        self.schema_manager = ClickHouseSchemaManager()
        self.schemas = self.schema_manager.SCHEMAS
    
    def build_profile_query(
        self,
        filter_tree: Dict[str, Any],
        select_fields: List[str] = None,
        limit: int = 1000,
        offset: int = 0,
        order_by: List[Dict[str, str]] = None
    ) -> str:
        """Build ClickHouse query for profile-only filters"""
        
        # Get table name
        table = self.engine.get_table_name(QueryType.PROFILE_ONLY)
        
        # Default fields if none specified
        if not select_fields:
            select_fields = [
                "imsi", "phone_no", "uid", "eid",
                "nationality_code", "gender_en", "age",
                "risk_score", "drug_dealing_score", "murder_score",
                "home_city", "work_location",
                "latest_job_title_en"
            ]
        
        # Validate fields against schema
        valid_fields = self._validate_fields(select_fields, table)
        if valid_fields != select_fields:
            invalid_fields = set(select_fields) - set(valid_fields)
            logger.warning(f"Removed invalid fields from SELECT: {invalid_fields}")
            select_fields = valid_fields
        
        # Build SELECT clause
        select_clause = self._build_select_clause(select_fields)
        
        # Clean filter tree to fix field names
        cleaned_filter_tree = FieldMapper.clean_filter_tree(filter_tree) if filter_tree else None
        
        # Build WHERE clause
        where_clause, params = self.translator.translate(cleaned_filter_tree) if cleaned_filter_tree else ("1=1", [])
        
        # Start building query
        query_parts = [
            f"SELECT {select_clause}",
            f"FROM {table}"
        ]
        
        # Add WHERE clause if filters exist
        if where_clause and where_clause != "1=1":
            query_parts.append(f"WHERE {where_clause}")
        
        # Add ORDER BY if specified
        if order_by:
            order_clause = self._build_order_clause(order_by)
            query_parts.append(f"ORDER BY {order_clause}")
        
        # Add LIMIT/OFFSET
        if limit:
            query_parts.append(f"LIMIT {limit}")
            if offset:
                query_parts.append(f"OFFSET {offset}")
        
        # Add ClickHouse-specific settings for large queries
        query_parts.append("SETTINGS max_memory_usage = 3000000000")  # 3GB limit
        
        return "\n".join(query_parts)
    
    def _build_select_clause(self, fields: List[str]) -> str:
        """Build SELECT clause with quoted identifiers"""
        quoted_fields = [self.engine.quote_identifier(f) for f in fields]
        return ", ".join(quoted_fields)
    
    def _build_order_clause(self, order_by: List[Dict[str, str]]) -> str:
        """Build ORDER BY clause"""
        order_parts = []
        for order in order_by:
            field = order.get("field")
            direction = order.get("direction", "ASC").upper()
            if field:
                quoted_field = self.engine.quote_identifier(field)
                order_parts.append(f"{quoted_field} {direction}")
        
        return ", ".join(order_parts) if order_parts else ""
    
    def _validate_fields(self, fields: List[str], table_name: str) -> List[str]:
        """Validate fields against table schema"""
        # Get schema for the table
        table_schema = self.schemas.get(table_name, {})
        if not table_schema:
            logger.warning(f"No schema found for table {table_name}, returning all fields")
            return fields
        
        # Filter out invalid fields
        valid_fields = []
        for field in fields:
            if field in table_schema:
                valid_fields.append(field)
            else:
                logger.debug(f"Field '{field}' not found in {table_name} schema")
        
        return valid_fields
    
    def build_movement_query(
        self,
        location_contexts: List[Dict[str, Any]],
        pattern_type: str,
        time_filters: Dict[str, Any] = None
    ) -> str:
        """Build ClickHouse query for movement patterns - Phase 2"""
        # TODO: Implement in Phase 2
        raise NotImplementedError("Movement queries not yet implemented")
    
    def build_analytical_query(
        self,
        aggregations: List[Dict[str, Any]],
        group_by: List[str],
        filter_tree: Dict[str, Any] = None
    ) -> str:
        """Build analytical queries with aggregations - Phase 3"""
        # TODO: Implement in Phase 3
        raise NotImplementedError("Analytical queries not yet implemented")