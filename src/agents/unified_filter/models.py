# src/agents/unified_filter/models.py
"""
Data models for Unified Filter Agent
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum


class FilterGenerationMethod(Enum):
    """How the unified filter was generated"""
    UNIFIED_FILTERS = "unified_filters"
    DIRECT_LLM = "direct_llm"
    FALLBACK = "fallback"


class QueryPatternType(Enum):
    """Query pattern types - spatial and non-spatial"""
    NON_SPATIAL = "non_spatial"  # No location component (profile/risk/time only)
    SINGLE_LOCATION = "single_location"  # One location with filters
    MULTI_LOCATION_UNION = "multi_location_union"  # Different people in different locations (OR)
    MULTI_LOCATION_INTERSECTION = "multi_location_intersection"  # Same people in multiple locations (AND)
    LOCATION_SEQUENCE = "location_sequence"  # People who visited locations in sequence
    CO_LOCATION = "co_location"  # People together at same time/place


@dataclass
class LocationContext:
    """
    Represents a location with ALL its associated filters (self-contained)
    """
    location_name: str
    location_index: int
    location_type: str  # CITY, EMIRATE, FACILITY, ADDRESS
    location_field: str  # home_city, work_location, residency_emirate, visited_city, visited_emirate, geohash
    location_value: Optional[str] = None  # DUBAI, ABU DHABI, etc. (for CITY/EMIRATE)
    radius_meters: Optional[int] = None  # Only for FACILITY/ADDRESS
    sequence_order: Optional[int] = None  # For location sequences (1, 2, 3...)
    geohash_reference: Optional[Dict[str, Any]] = None  # Reference to stored geohashes for FACILITY/ADDRESS
    
    # Complete filter tree for this location (includes ALL conditions)
    complete_filter_tree: Dict[str, Any] = field(default_factory=dict)
    # This contains ALL filters that apply to this location context:
    # - Time conditions
    # - Profile conditions  
    # - Risk conditions
    # - Location conditions
    # Everything combined in one tree
    
    # Breakdown for analysis (optional, derived from complete_filter_tree)
    time_filters: Optional[Dict[str, Any]] = None
    profile_filters: Optional[Dict[str, Any]] = None
    risk_filters: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "location_name": self.location_name,
            "location_index": self.location_index,
            "location_type": self.location_type,
            "location_field": self.location_field,
            "location_value": self.location_value,
            "radius_meters": self.radius_meters,
            "sequence_order": self.sequence_order,
            "complete_filter_tree": self.complete_filter_tree,
            # Include breakdowns if available
            "time_filters": self.time_filters,
            "profile_filters": self.profile_filters,
            "risk_filters": self.risk_filters
        }
        # Only include geohash_reference if it exists
        if self.geohash_reference:
            result["geohash_reference"] = self.geohash_reference
        return result


@dataclass
class SequenceConfig:
    """
    Configuration for location sequence patterns
    """
    window_hours: int = 24
    order_matters: bool = True
    allow_intermediate_stops: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_hours": self.window_hours,
            "order_matters": self.order_matters,
            "allow_intermediate_stops": self.allow_intermediate_stops
        }


@dataclass
class QueryPattern:
    """
    Describes how the query should be interpreted and executed
    """
    pattern_type: QueryPatternType
    combination_logic: str = ""  # "OR", "AND", "SEQUENCE", etc.
    requires_movement_data: bool = False
    requires_profile_data: bool = True
    common_criteria: Optional[Dict[str, Any]] = None  # Only if truly common across all locations
    sequence_config: Optional[SequenceConfig] = None  # For location sequence patterns
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "pattern_type": self.pattern_type.value,
            "combination_logic": self.combination_logic,
            "requires_movement_data": self.requires_movement_data,
            "requires_profile_data": self.requires_profile_data,
            "common_criteria": self.common_criteria
        }
        if self.sequence_config:
            result["sequence_config"] = self.sequence_config.to_dict()
        return result


@dataclass
class UnifiedFilterTree:
    """
    Unified filter tree combining all agent filters
    Supports both spatial (location-based) and non-spatial queries
    """
    # Query pattern - how to interpret and combine locations
    query_pattern: QueryPattern
    
    # For non-spatial queries: unified filter tree at root level
    unified_filter_tree: Optional[Dict[str, Any]] = None
    
    # For spatial queries: location-specific contexts (each is self-contained)
    location_contexts: List[LocationContext] = field(default_factory=list)
    
    # Metadata about filter composition
    has_time_filters: bool = False
    has_location_filters: bool = False
    has_profile_filters: bool = False
    has_risk_filters: bool = False
    has_city_filters: bool = False  # tracks CITY/EMIRATE types
    has_facility_filters: bool = False  # tracks FACILITY/ADDRESS types
    uses_location_geohashes: bool = False
    
    # Query optimization hints
    suggested_data_source: str = ""  # "profile_only", "movements", "geo_live"
    query_plan: Dict[str, Any] = field(default_factory=dict)
    
    # Generation metadata
    reasoning: str = ""
    confidence: float = 0.0
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "query_pattern": self.query_pattern.to_dict(),
            "unified_filter_tree": self.unified_filter_tree,  # For non-spatial queries
            "location_contexts": [loc.to_dict() for loc in self.location_contexts],
            "metadata": {
                "has_time_filters": self.has_time_filters,
                "has_location_filters": self.has_location_filters,
                "has_profile_filters": self.has_profile_filters,
                "has_risk_filters": self.has_risk_filters,
                "has_city_filters": self.has_city_filters,
                "has_facility_filters": self.has_facility_filters,
                "uses_location_geohashes": self.uses_location_geohashes
            },
            "query_optimization": {
                "suggested_data_source": self.suggested_data_source,
                "query_plan": self.query_plan
            },
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "warnings": self.warnings
        }


@dataclass
class UnifiedFilterResult:
    """Result from Unified Filter Agent"""
    # Unified filter tree
    unified_filter_tree: UnifiedFilterTree
    
    # Query metadata
    query_type: str = ""  # demographic, risk_based, location_based, complex_multi_location
    estimated_complexity: int = 0  # 1-10 scale
    
    # Generation metadata
    generation_method: FilterGenerationMethod = FilterGenerationMethod.UNIFIED_FILTERS
    generation_timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Optimization hints for SQL generation
    optimization_hints: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response"""
        return {
            "unified_filter_tree": self.unified_filter_tree.to_dict(),
            "query_metadata": {
                "query_type": self.query_type,
                "estimated_complexity": self.estimated_complexity,
                "generation_method": self.generation_method.value,
                "generation_timestamp": self.generation_timestamp.isoformat()
            },
            "optimization_hints": self.optimization_hints
        }