"""
Base classes for query engines
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
import logging

from ..models import (
    EngineType, QueryType, GeneratedQuery, QueryPlan,
    QueryExecutorRequest, QueryExecutorResult
)

logger = logging.getLogger(__name__)


class QueryEngine(ABC):
    """Abstract base class for all query engines"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.engine_type = self._get_engine_type()
        self.schema = self._load_schema()
    
    @abstractmethod
    def _get_engine_type(self) -> EngineType:
        """Return the engine type"""
        pass
    
    @abstractmethod
    def _load_schema(self) -> Dict[str, Any]:
        """Load schema information for the engine"""
        pass
    
    @abstractmethod
    def generate_query(self, request: QueryExecutorRequest) -> GeneratedQuery:
        """Generate query based on request"""
        pass
    
    @abstractmethod
    def estimate_query_plan(self, query: GeneratedQuery) -> QueryPlan:
        """Estimate query execution plan"""
        pass
    
    @abstractmethod
    def validate_query(self, query: GeneratedQuery) -> Tuple[bool, Optional[str]]:
        """
        Validate generated query
        Returns: (is_valid, error_message)
        """
        pass
    
    @abstractmethod
    def quote_identifier(self, identifier: str) -> str:
        """Quote table/column names according to engine rules"""
        pass
    
    @abstractmethod
    def format_value(self, value: Any, field_type: str = None) -> str:
        """Format values according to engine rules"""
        pass
    
    def get_table_name(self, query_type: QueryType) -> str:
        """Get table name based on query type"""
        table_mapping = {
            QueryType.PROFILE_ONLY: self.config.get('profile_table', 'phone_imsi_uid_latest'),
            QueryType.LOCATION_BASED: self.config.get('movement_table', 'movements'),
            QueryType.MOVEMENT_PATTERN: self.config.get('movement_table', 'movements'),
        }
        return table_mapping.get(query_type, self.config.get('default_table', 'phone_imsi_uid_latest'))


class FilterTranslator(ABC):
    """Base class for translating unified filter trees to engine-specific SQL"""
    
    def __init__(self, engine: QueryEngine):
        self.engine = engine
        self.operators_map = self._get_operators_map()
    
    @abstractmethod
    def _get_operators_map(self) -> Dict[str, str]:
        """Return operator mapping for the engine"""
        pass
    
    def translate(self, filter_tree: Dict[str, Any]) -> Tuple[str, List[Any]]:
        """Translate filter tree to WHERE clause"""
        if not filter_tree:
            return "1=1", []
        
        return self._translate_node(filter_tree)
    
    def _translate_node(self, node: Dict[str, Any]) -> Tuple[str, List[Any]]:
        """Recursively translate filter nodes"""
        if not isinstance(node, dict):
            raise ValueError(f"Invalid filter node: {node}")
        
        # Handle logical operators
        if "AND" in node:
            return self._translate_logical("AND", node["AND"])
        elif "OR" in node:
            return self._translate_logical("OR", node["OR"])
        elif "NOT" in node:
            clause, params = self._translate_node(node["NOT"])
            return f"NOT ({clause})", params
        else:
            # It's a comparison node
            return self._translate_comparison(node)
    
    def _translate_logical(self, operator: str, nodes: List[Dict]) -> Tuple[str, List[Any]]:
        """Translate AND/OR operations"""
        if not nodes:
            return "1=1", []
        
        clauses = []
        all_params = []
        
        for node in nodes:
            try:
                clause, params = self._translate_node(node)
                clauses.append(clause)
                all_params.extend(params)
            except Exception as e:
                logger.warning(f"Failed to translate node {node}: {e}")
                continue
        
        if not clauses:
            return "1=1", []
        
        # Wrap each clause in parentheses for safety
        wrapped_clauses = [f"({c})" for c in clauses]
        return f" {operator} ".join(wrapped_clauses), all_params
    
    @abstractmethod
    def _translate_comparison(self, node: Dict[str, Any]) -> Tuple[str, List[Any]]:
        """Translate comparison operators - engine specific"""
        pass
    
    def _validate_field(self, field: str) -> bool:
        """Validate field exists in schema"""
        # TODO: Implement schema validation
        return True
    
    def _sanitize_value(self, value: Any) -> Any:
        """Sanitize values to prevent SQL injection"""
        if isinstance(value, str):
            # Basic sanitization - engines should use parameterized queries
            return value.replace("'", "''")
        return value