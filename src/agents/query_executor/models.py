"""
Data models for Query Executor Agent
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Union
from datetime import datetime


class EngineType(Enum):
    """Supported query engines"""
    CLICKHOUSE = "clickhouse"
    SPARK = "spark"
    PYSPARK = "pyspark"  # DataFrame API
    POSTGRESQL = "postgresql"
    ELASTICSEARCH = "elasticsearch"


class QueryType(Enum):
    """Types of queries supported"""
    PROFILE_ONLY = "profile_only"
    LOCATION_BASED = "location_based"
    MOVEMENT_PATTERN = "movement_pattern"
    ANALYTICAL = "analytical"
    AGGREGATION = "aggregation"


class ExecutionMode(Enum):
    """Query execution modes"""
    GENERATE_ONLY = "generate_only"  # Only generate SQL, don't execute
    EXECUTE = "execute"  # Generate and execute
    EXPLAIN = "explain"  # Generate and explain query plan
    VALIDATE = "validate"  # Validate query without execution


@dataclass
class QueryExecutorRequest:
    """Request model for query executor"""
    unified_filter_tree: Dict[str, Any]  # From unified filter agent
    engine_type: EngineType
    query_type: QueryType = QueryType.PROFILE_ONLY
    execution_mode: ExecutionMode = ExecutionMode.GENERATE_ONLY
    
    # Optional parameters
    select_fields: Optional[List[str]] = None
    limit: int = 1000
    offset: int = 0
    order_by: Optional[List[Dict[str, str]]] = None  # [{"field": "risk_score", "direction": "DESC"}]
    
    # Engine-specific options
    engine_options: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    session_id: Optional[str] = None
    query_id: Optional[str] = None
    user_id: Optional[str] = None


@dataclass
class QueryPlan:
    """Query execution plan details"""
    estimated_rows: str
    estimated_cost: Optional[float] = None
    indexes_used: List[str] = field(default_factory=list)
    partitions_scanned: Optional[int] = None
    optimization_hints: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class GeneratedQuery:
    """Generated query details"""
    query: str
    parameters: List[Any] = field(default_factory=list)
    query_hash: Optional[str] = None
    
    # Engine-specific
    engine_type: EngineType = EngineType.CLICKHOUSE
    dialect_version: Optional[str] = None
    
    # PySpark specific
    dataframe_code: Optional[str] = None
    dataframe_operations: Optional[List[Dict[str, Any]]] = None


@dataclass
class ExecutionResult:
    """Query execution results"""
    rows_returned: int
    execution_time_ms: float
    data: Optional[List[Dict[str, Any]]] = None  # Actual results if requested
    column_types: Optional[Dict[str, str]] = None
    has_more_results: bool = False
    next_offset: Optional[int] = None


@dataclass
class QueryExecutorResult:
    """Complete result from query executor"""
    # Core results
    generated_query: GeneratedQuery
    query_plan: Optional[QueryPlan] = None
    execution_result: Optional[ExecutionResult] = None
    
    # Status
    success: bool = True
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    
    # Metadata
    generation_time_ms: float = 0.0
    total_time_ms: float = 0.0
    engine_type: EngineType = EngineType.CLICKHOUSE
    query_type: QueryType = QueryType.PROFILE_ONLY
    
    # Tracking
    session_id: Optional[str] = None
    query_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "generated_query": {
                "query": self.generated_query.query,
                "parameters": self.generated_query.parameters,
                "engine_type": self.generated_query.engine_type.value,
                "dataframe_code": self.generated_query.dataframe_code
            },
            "query_plan": {
                "estimated_rows": self.query_plan.estimated_rows,
                "optimization_hints": self.query_plan.optimization_hints
            } if self.query_plan else None,
            "execution_result": {
                "rows_returned": self.execution_result.rows_returned,
                "execution_time_ms": self.execution_result.execution_time_ms,
                "has_more_results": self.execution_result.has_more_results
            } if self.execution_result else None,
            "success": self.success,
            "error_message": self.error_message,
            "engine_type": self.engine_type.value,
            "query_type": self.query_type.value,
            "timestamp": self.timestamp.isoformat()
        }