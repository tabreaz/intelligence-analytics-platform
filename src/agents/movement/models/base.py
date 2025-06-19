# src/agents/movement/models.py
"""
Data models for Movement Analysis Agent
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional


class QueryType(Enum):
    """Movement query types"""
    SINGLE_LOCATION = "single_location"
    MULTI_LOCATION_AND = "multi_location_and"
    MULTI_LOCATION_OR = "multi_location_or"
    MOVEMENT_PATTERN = "movement_pattern"
    CO_PRESENCE = "co_presence"
    HEATMAP = "heatmap"
    PREDICTIVE_ANALYSIS = "predictive_analysis"
    ANOMALY_DETECTION = "anomaly_detection"
    MULTI_MODAL_MOVEMENT_ANALYSIS = "multi_modal_movement_analysis"


class SpatialMethod(Enum):
    """Spatial filter methods"""
    NAME = "name"
    COORDINATES = "coordinates"
    AREA = "area"
    POLYGON = "polygon"


class AggregationPeriod(Enum):
    """Time aggregation periods"""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass
class DateRange:
    """Date range for time constraints"""
    start: str  # ISO format datetime
    end: str    # ISO format datetime
    
    def to_dict(self) -> Dict[str, str]:
        return {"start": self.start, "end": self.end}
    
    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> 'DateRange':
        return cls(start=data['start'], end=data['end'])


@dataclass
class HourRange:
    """Hour range for time constraints"""
    start_hour: int  # 0-23
    end_hour: int    # 0-23
    
    def to_dict(self) -> Dict[str, int]:
        return {"start_hour": self.start_hour, "end_hour": self.end_hour}
    
    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> 'HourRange':
        return cls(start_hour=data['start_hour'], end_hour=data['end_hour'])


@dataclass
class RecurringPattern:
    """Recurring time pattern"""
    type: str  # daily, weekly, monthly
    on_days: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "on_days": self.on_days}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RecurringPattern':
        return cls(type=data['type'], on_days=data.get('on_days', []))


@dataclass
class SpatialFilter:
    """Spatial filter definition"""
    method: str
    value: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    radius_meters: Optional[int] = 1000  # Default 1km radius
    polygon: Optional[List[List[float]]] = None


@dataclass
class TimeConstraints:
    """Time constraints for location"""
    included_date_ranges: List[DateRange] = field(default_factory=list)
    excluded_date_ranges: List[DateRange] = field(default_factory=list)
    included_hours: List[int] = field(default_factory=list)
    excluded_hours: List[int] = field(default_factory=list)
    included_days_of_week: List[str] = field(default_factory=list)
    excluded_days_of_week: List[str] = field(default_factory=list)
    recurring: Optional[RecurringPattern] = None
    match_granularity: str = "hour"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding empty fields"""
        result = {}
        
        if self.included_date_ranges:
            result['included_date_ranges'] = [dr.to_dict() for dr in self.included_date_ranges]
        if self.excluded_date_ranges:
            result['excluded_date_ranges'] = [dr.to_dict() for dr in self.excluded_date_ranges]
        if self.included_hours:
            result['included_hours'] = self.included_hours
        if self.excluded_hours:
            result['excluded_hours'] = self.excluded_hours
        if self.included_days_of_week:
            result['included_days_of_week'] = self.included_days_of_week
        if self.excluded_days_of_week:
            result['excluded_days_of_week'] = self.excluded_days_of_week
        if self.recurring:
            result['recurring'] = self.recurring.to_dict()
        result['match_granularity'] = self.match_granularity
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TimeConstraints':
        """Create from dictionary"""
        return cls(
            included_date_ranges=[DateRange.from_dict(dr) for dr in data.get('included_date_ranges', [])],
            excluded_date_ranges=[DateRange.from_dict(dr) for dr in data.get('excluded_date_ranges', [])],
            included_hours=data.get('included_hours', []),
            excluded_hours=data.get('excluded_hours', []),
            included_days_of_week=data.get('included_days_of_week', []),
            excluded_days_of_week=data.get('excluded_days_of_week', []),
            recurring=RecurringPattern.from_dict(data['recurring']) if data.get('recurring') else None,
            match_granularity=data.get('match_granularity', 'hour')
        )


@dataclass
class PresenceRequirements:
    """Presence requirements at location"""
    minimum_duration_minutes: Optional[int] = None
    minimum_visits: Optional[int] = None
    aggregation_period: Optional[str] = None


@dataclass
class Geofence:
    """Geofence definition"""
    id: str
    reference: str
    spatial_filter: SpatialFilter
    time_constraints: Optional[TimeConstraints] = None
    presence_requirements: Optional[PresenceRequirements] = None


@dataclass
class MovementFilterResult:
    """Result model for movement filter agent"""
    # Core fields
    reasoning: str = ""
    query_type: str = ""
    
    # Identity filters
    identity_filters: Dict[str, List[str]] = field(default_factory=dict)
    
    # Co-presence analysis
    co_presence: Optional[Dict[str, Any]] = None
    
    # Geofences
    geofences: List[Geofence] = field(default_factory=list)
    
    # Analysis types
    heatmap: Optional[Dict[str, Any]] = None
    sequence_patterns: List[Dict[str, Any]] = field(default_factory=list)
    clustering: Optional[Dict[str, Any]] = None
    pattern_detection: Optional[Dict[str, Any]] = None
    anomaly_detection: Optional[Dict[str, Any]] = None
    predictive_modeling: Optional[Dict[str, Any]] = None
    
    # Global filters
    global_time_filter: Optional[Dict[str, Any]] = None
    
    # Output options
    output_options: Dict[str, Any] = field(default_factory=lambda: {
        "format": "trajectory",
        "include_metadata": True,
        "include_profiles": False
    })
    
    # Metadata
    ambiguities: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    extraction_method: str = "llm"
    raw_extractions: Dict[str, Any] = field(default_factory=dict)
    validation_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response"""
        result = {
            "reasoning": self.reasoning,
            "query_type": self.query_type,
            "confidence": self.confidence,
            "extraction_method": self.extraction_method,
            "validation_warnings": self.validation_warnings
        }
        
        # Only include non-empty fields
        if self.identity_filters:
            result["identity_filters"] = self.identity_filters
            
        if self.co_presence:
            result["co_presence"] = self.co_presence
            
        if self.geofences:
            result["geofences"] = [self._geofence_to_dict(gf) for gf in self.geofences]
            
        if self.heatmap:
            result["heatmap"] = self.heatmap
            
        if self.sequence_patterns:
            result["sequence_patterns"] = self.sequence_patterns
            
        if self.clustering:
            result["clustering"] = self.clustering
            
        if self.pattern_detection:
            result["pattern_detection"] = self.pattern_detection
            
        if self.anomaly_detection:
            result["anomaly_detection"] = self.anomaly_detection
            
        if self.predictive_modeling:
            result["predictive_modeling"] = self.predictive_modeling
            
        if self.global_time_filter:
            result["global_time_filter"] = self.global_time_filter
            
        if self.output_options:
            result["output_options"] = self.output_options
            
        if self.ambiguities:
            result["ambiguities"] = self.ambiguities

        return result
    
    def _geofence_to_dict(self, geofence: Geofence) -> Dict[str, Any]:
        """Convert geofence to dictionary"""
        gf_dict = {
            "id": geofence.id,
            "reference": geofence.reference,
            "spatial_filter": {
                "method": geofence.spatial_filter.method
            }
        }
        
        # Add spatial filter fields
        sf = geofence.spatial_filter
        if sf.value:
            gf_dict["spatial_filter"]["value"] = sf.value
        if sf.latitude is not None:
            gf_dict["spatial_filter"]["latitude"] = sf.latitude
        if sf.longitude is not None:
            gf_dict["spatial_filter"]["longitude"] = sf.longitude
        if sf.radius_meters:
            gf_dict["spatial_filter"]["radius_meters"] = sf.radius_meters
        if sf.polygon:
            gf_dict["spatial_filter"]["polygon"] = sf.polygon
            
        # Add time constraints if present
        if geofence.time_constraints:
            # Use the to_dict method which handles the new structure properly
            tc_dict = geofence.time_constraints.to_dict()
            if tc_dict:
                gf_dict["time_constraints"] = tc_dict
                
        # Add presence requirements if present
        if geofence.presence_requirements:
            pr = geofence.presence_requirements
            pr_dict = {}
            if pr.minimum_duration_minutes is not None:
                pr_dict["minimum_duration_minutes"] = pr.minimum_duration_minutes
            if pr.minimum_visits is not None:
                pr_dict["minimum_visits"] = pr.minimum_visits
            if pr.aggregation_period:
                pr_dict["aggregation_period"] = pr.aggregation_period
            if pr_dict:
                gf_dict["presence_requirements"] = pr_dict
                
        return gf_dict