# src/agents/profiler/models.py
"""
Data models for Profiler Agent - SQL Generation and Analytics
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum


class QueryType(str, Enum):
    """Specific types of profile queries"""
    DEMOGRAPHIC = "demographic"
    RISK_BASED = "risk_based"
    LOCATION_BASED = "location_based"
    MOVEMENT_ANALYSIS = "movement_analysis"
    APPLICATION_USAGE = "application_usage"
    CRIME_INVESTIGATION = "crime_investigation"
    AGGREGATE_STATS = "aggregate_stats"
    DETAIL_RECORDS = "detail_records"
    COMPLEX_MULTI_FILTER = "complex_multi_filter"


class SQLGenerationMethod(str, Enum):
    """How the SQL was generated"""
    TEMPLATE = "template"  # Template-based
    LLM = "llm"  # Pure LLM generation
    HYBRID = "hybrid"  # Template + LLM enhancement
    MANUAL = "manual"  # Manually crafted


@dataclass
class QueryStats:
    """Statistics about the query results"""
    # Row counts
    estimated_row_count: Optional[int] = None
    actual_row_count: Optional[int] = None
    
    # Risk distribution (if applicable)
    risk_distribution: Optional[Dict[str, int]] = None  # {"high": 100, "medium": 200, "low": 300}
    risk_score_avg: Optional[float] = None
    risk_score_max: Optional[float] = None
    
    # Demographics distribution
    nationality_distribution: Optional[Dict[str, int]] = None  # Top 10
    age_group_distribution: Optional[Dict[str, int]] = None
    gender_distribution: Optional[Dict[str, int]] = None
    residency_status_distribution: Optional[Dict[str, int]] = None
    
    # Geographic coverage
    unique_cities_count: Optional[int] = None
    unique_geohashes_count: Optional[int] = None
    location_radius_meters: Optional[int] = None
    
    # Temporal stats
    time_range_days: Optional[int] = None
    data_recency_hours: Optional[float] = None
    
    # Query complexity
    filter_count: int = 0
    filter_complexity_score: int = 0
    join_count: int = 0
    estimated_scan_gb: Optional[float] = None
    
    # Performance hints
    index_usage: List[str] = field(default_factory=list)
    partition_pruning: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            k: v for k, v in self.__dict__.items() 
            if v is not None and (not isinstance(v, list) or v)
        }


@dataclass
class ProfilerSQLGenerated:
    """Represents a generated SQL script with full metadata"""
    # Core tracking
    session_id: str
    query_id: str
    
    # Query information
    original_query: str
    context_aware_query: str
    
    # Classification details
    classification: Dict[str, Any]  # category, confidence
    domains: List[str]
    agents_required: List[str]
    query_type: QueryType
    
    # Extracted information
    entities_detected: Dict[str, Any]
    filters: Dict[str, Any]  # All filters from orchestrator
    
    # SQL Generation
    sql_generated_script: str
    sql_generation_method: SQLGenerationMethod
    generated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Validation and issues
    validation_warnings: List[str] = field(default_factory=list)
    ambiguities: List[Dict[str, Any]] = field(default_factory=list)
    
    # Filter flags for quick querying
    has_time_filter: bool = False
    has_location_filter: bool = False
    has_profile_filter: bool = False
    has_risk_filter: bool = False
    has_movement_filter: bool = False
    has_crime_filter: bool = False
    has_application_filter: bool = False
    is_aggregate_query: bool = False
    is_detail_query: bool = False
    
    # Statistics
    stats: Optional[QueryStats] = None
    
    # Additional metadata
    sql_dialect: str = "clickhouse"
    table_schema_version: str = "1.0"
    privacy_classification: str = "restricted"  # public/restricted/sensitive
    execution_hints: List[str] = field(default_factory=list)
    related_queries: List[str] = field(default_factory=list)  # Related query IDs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage in any analytical engine"""
        return {
            # Core tracking
            'session_id': self.session_id,
            'query_id': self.query_id,
            
            # Query information
            'original_query': self.original_query,
            'context_aware_query': self.context_aware_query,
            
            # Classification
            'classification': self.classification,
            'domains': self.domains,
            'agents_required': self.agents_required,
            'query_type': self.query_type.value,
            
            # Extracted information
            'entities_detected': self.entities_detected,
            'filters': self.filters,
            
            # SQL Generation
            'sql_generated_script': self.sql_generated_script,
            'sql_generation_method': self.sql_generation_method.value,
            'generated_at': self.generated_at.isoformat(),
            
            # Validation
            'validation_warnings': self.validation_warnings,
            'ambiguities': self.ambiguities,
            
            # Filter flags
            'has_time_filter': self.has_time_filter,
            'has_location_filter': self.has_location_filter,
            'has_profile_filter': self.has_profile_filter,
            'has_risk_filter': self.has_risk_filter,
            'has_movement_filter': self.has_movement_filter,
            'has_crime_filter': self.has_crime_filter,
            'has_application_filter': self.has_application_filter,
            'is_aggregate_query': self.is_aggregate_query,
            'is_detail_query': self.is_detail_query,
            
            # Stats
            'stats': self.stats.to_dict() if self.stats else None,
            
            # Metadata
            'sql_dialect': self.sql_dialect,
            'table_schema_version': self.table_schema_version,
            'privacy_classification': self.privacy_classification,
            'execution_hints': self.execution_hints,
            'related_queries': self.related_queries
        }
    
    def to_analytics_format(self, format_type: str = "parquet") -> Dict[str, Any]:
        """Convert to format suitable for analytical engines"""
        base_dict = self.to_dict()
        
        if format_type == "parquet":
            # Flatten nested structures for Parquet
            flat_dict = {}
            for key, value in base_dict.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        flat_dict[f"{key}_{sub_key}"] = sub_value
                elif isinstance(value, list):
                    flat_dict[key] = str(value)  # Convert lists to strings
                else:
                    flat_dict[key] = value
            return flat_dict
        
        return base_dict


@dataclass
class ProfilerResult:
    """Result from the Profiler Agent"""
    status: str = "pending"
    sql_generated: Optional[ProfilerSQLGenerated] = None
    execution_time_ms: int = 0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'status': self.status,
            'sql_generated': self.sql_generated.to_dict() if self.sql_generated else None,
            'execution_time_ms': self.execution_time_ms,
            'error': self.error
        }