"""
Base Profile Analytics Service Interface
Abstract interface for profile analytics operations
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from uuid import UUID
from dataclasses import dataclass, field
from enum import Enum


class AggregationType(Enum):
    """Types of aggregations supported"""
    COUNT = "count"
    UNIQUE_COUNT = "unique_count"
    DISTRIBUTION = "distribution"
    PERCENTILE = "percentile"
    TIME_SERIES = "time_series"
    CROSS_TAB = "cross_tab"


@dataclass
class FieldStatistics:
    """Statistics for a single field"""
    field_name: str
    total_count: int
    unique_count: Optional[int] = None
    null_count: int = 0
    distribution: Dict[str, int] = field(default_factory=dict)
    percentiles: Dict[float, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProfileQueryResult:
    """Result of a profile query execution"""
    query_id: UUID
    session_id: Optional[str]
    sql_generated: str
    execution_time_ms: float
    result_count: int
    data: List[Dict[str, Any]]
    statistics: Dict[str, FieldStatistics] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseProfileAnalyticsService(ABC):
    """
    Abstract base class for profile analytics services
    Defines interface for profile query execution and analysis
    """
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the service"""
        pass
    
    @abstractmethod
    async def execute_profile_query(
        self,
        where_clause: str,
        select_fields: Optional[List[str]] = None,
        limit: int = 1000,
        offset: int = 0,
        order_by: Optional[List[Dict[str, str]]] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> ProfileQueryResult:
        """
        Execute a profile query with the given WHERE clause
        
        Args:
            where_clause: SQL WHERE clause (without WHERE keyword)
            select_fields: Fields to select (None = default fields)
            limit: Result limit
            offset: Result offset for pagination
            order_by: List of order by specifications
            session_id: Session ID for tracking
            user_id: User ID for tracking
            
        Returns:
            ProfileQueryResult with data and statistics
        """
        pass
    
    @abstractmethod
    async def get_profile_details(
        self,
        where_clause: str,
        fields: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Get detailed profile information with pagination
        
        Args:
            where_clause: SQL WHERE clause
            fields: Fields to return
            limit: Page size
            offset: Page offset
            
        Returns:
            Dictionary with profile details and pagination info
        """
        pass
    
    @abstractmethod
    async def get_unique_counts(
        self,
        where_clause: str,
        identifier_fields: Optional[List[str]] = None
    ) -> Dict[str, int]:
        """
        Get unique counts for identifier fields
        
        Args:
            where_clause: SQL WHERE clause
            identifier_fields: Fields to count (None = default identifiers)
            
        Returns:
            Dictionary of field names to unique counts
        """
        pass
    
    @abstractmethod
    async def get_field_distribution(
        self,
        where_clause: str,
        field_name: str,
        top_n: int = 20,
        include_others: bool = True
    ) -> FieldStatistics:
        """
        Get distribution of values for a specific field
        
        Args:
            where_clause: SQL WHERE clause
            field_name: Field to analyze
            top_n: Number of top values to return
            include_others: Include "Others" category
            
        Returns:
            FieldStatistics with distribution data
        """
        pass
    
    @abstractmethod
    async def get_demographic_statistics(
        self,
        where_clause: str
    ) -> Dict[str, FieldStatistics]:
        """
        Get comprehensive demographic statistics
        
        Args:
            where_clause: SQL WHERE clause
            
        Returns:
            Dictionary of field names to statistics
        """
        pass
    
    @abstractmethod
    async def get_risk_statistics(
        self,
        where_clause: str
    ) -> Dict[str, FieldStatistics]:
        """
        Get risk score and security statistics
        
        Args:
            where_clause: SQL WHERE clause
            
        Returns:
            Dictionary of field names to statistics
        """
        pass
    
    @abstractmethod
    async def get_communication_statistics(
        self,
        where_clause: str
    ) -> Dict[str, FieldStatistics]:
        """
        Get communication pattern statistics
        
        Args:
            where_clause: SQL WHERE clause
            
        Returns:
            Dictionary of field names to statistics
        """
        pass
    
    @abstractmethod
    async def get_travel_statistics(
        self,
        where_clause: str
    ) -> Dict[str, FieldStatistics]:
        """
        Get travel pattern statistics
        
        Args:
            where_clause: SQL WHERE clause
            
        Returns:
            Dictionary of field names to statistics
        """
        pass
    
    @abstractmethod
    async def get_cross_tabulation(
        self,
        where_clause: str,
        field1: str,
        field2: str,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Get cross-tabulation between two fields
        
        Args:
            where_clause: SQL WHERE clause
            field1: First field
            field2: Second field
            limit: Maximum combinations to return
            
        Returns:
            Cross-tabulation data
        """
        pass
    
    @abstractmethod
    def get_supported_fields(self) -> Dict[str, List[str]]:
        """
        Get supported fields by category
        
        Returns:
            Dictionary of field categories to field names
        """
        pass
    
    @abstractmethod
    def get_engine_type(self) -> str:
        """
        Get the underlying query engine type
        
        Returns:
            Engine type string (e.g., 'clickhouse', 'spark')
        """
        pass