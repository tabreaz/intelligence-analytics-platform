# src/agents/movement/models/single_location.py
"""
Data models for Single Location Pattern extraction
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

# Import common models from base
from .base import TimeConstraints, DateRange, HourRange, RecurringPattern


@dataclass
class CountryCodeFilter:
    """Country code filtering for single location queries"""
    country_code: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        if self.country_code:
            return {'country_code': self.country_code}
        return {}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CountryCodeFilter':
        """Create from dictionary"""
        return cls(country_code=data.get('country_code', []))


@dataclass
class LocationScope:
    """Location scope filtering (emirate/municipality)"""
    emirate: List[str] = field(default_factory=list)
    municipality: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding empty fields"""
        result = {}
        if self.emirate:
            result['emirate'] = self.emirate
        if self.municipality:
            result['municipality'] = self.municipality
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LocationScope':
        """Create from dictionary"""
        return cls(
            emirate=data.get('emirate', []),
            municipality=data.get('municipality', [])
        )


# Note: DateRange, HourRange, RecurringPattern, and TimeConstraints are now imported from base


@dataclass
class LocationFilter:
    """Location filter for single location queries"""
    method: str  # name, geohash, coordinates, polygon
    value: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    radius_meters: Optional[int] = 1000
    polygon: Optional[List[List[float]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values"""
        result = {"method": self.method}
        
        if self.value is not None:
            result['value'] = self.value
        if self.latitude is not None:
            result['latitude'] = self.latitude
        if self.longitude is not None:
            result['longitude'] = self.longitude
        if self.radius_meters is not None:
            result['radius_meters'] = self.radius_meters
        if self.polygon is not None:
            result['polygon'] = self.polygon
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LocationFilter':
        """Create from dictionary"""
        return cls(
            method=data['method'],
            value=data.get('value'),
            latitude=data.get('latitude'),
            longitude=data.get('longitude'),
            radius_meters=data.get('radius_meters', 1000),
            polygon=data.get('polygon')
        )


@dataclass
class PresenceRequirements:
    """Presence requirements at the location"""
    minimum_duration_minutes: Optional[int] = 30
    minimum_visits: Optional[int] = 1
    aggregation_period: Optional[str] = "day"  # day, week, month
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {}
        if self.minimum_duration_minutes is not None:
            result['minimum_duration_minutes'] = self.minimum_duration_minutes
        if self.minimum_visits is not None:
            result['minimum_visits'] = self.minimum_visits
        if self.aggregation_period is not None:
            result['aggregation_period'] = self.aggregation_period
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PresenceRequirements':
        return cls(
            minimum_duration_minutes=data.get('minimum_duration_minutes', 30),
            minimum_visits=data.get('minimum_visits', 1),
            aggregation_period=data.get('aggregation_period', 'day')
        )


@dataclass
class OutputOptions:
    """Output formatting options"""
    format: str = "density_map"  # presence_count, trajectory, point_cloud, density_map
    include_metadata: bool = True
    include_profiles: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": self.format,
            "include_metadata": self.include_metadata,
            "include_profiles": self.include_profiles
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OutputOptions':
        return cls(
            format=data.get('format', 'presence_count'),
            include_metadata=data.get('include_metadata', True),
            include_profiles=data.get('include_profiles', False)
        )


@dataclass
class Ambiguity:
    """Ambiguity in the query"""
    parameter: str
    issue: str
    suggested_clarification: str
    options: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "parameter": self.parameter,
            "issue": self.issue,
            "suggested_clarification": self.suggested_clarification
        }
        if self.options:
            result['options'] = self.options
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Ambiguity':
        return cls(
            parameter=data['parameter'],
            issue=data['issue'],
            suggested_clarification=data['suggested_clarification'],
            options=data.get('options')
        )


@dataclass
class SingleLocationResult:
    """Complete result for single location pattern extraction"""
    reasoning: str = ""
    country_code_filter: Optional[CountryCodeFilter] = None
    location_scope: Optional[LocationScope] = None
    time_constraints: Optional[TimeConstraints] = None
    location_filter: Optional[LocationFilter] = None
    presence_requirements: Optional[PresenceRequirements] = None
    output_options: OutputOptions = field(default_factory=OutputOptions)
    ambiguities: List[Ambiguity] = field(default_factory=list)
    confidence: float = 0.95
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON response"""
        result = {
            "reasoning": self.reasoning,
            "confidence": self.confidence
        }
        
        # Add non-empty fields
        if self.country_code_filter:
            filter_dict = self.country_code_filter.to_dict()
            if filter_dict:  # Only add if not empty
                result['country_code_filter'] = filter_dict
                
        if self.location_scope:
            scope_dict = self.location_scope.to_dict()
            if scope_dict:  # Only add if not empty
                result['location_scope'] = scope_dict
                
        if self.time_constraints:
            result['time_constraints'] = self.time_constraints.to_dict()
            
        if self.location_filter:
            result['location_filter'] = self.location_filter.to_dict()
            
        if self.presence_requirements:
            result['presence_requirements'] = self.presence_requirements.to_dict()
            
        result['output_options'] = self.output_options.to_dict()
        
        if self.ambiguities:
            result['ambiguities'] = [amb.to_dict() for amb in self.ambiguities]
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SingleLocationResult':
        """Create from dictionary (parsed JSON response)"""
        return cls(
            reasoning=data.get('reasoning', ''),
            country_code_filter=CountryCodeFilter.from_dict(data['country_code_filter']) 
                if data.get('country_code_filter') else None,
            location_scope=LocationScope.from_dict(data['location_scope'])
                if data.get('location_scope') else None,
            time_constraints=TimeConstraints.from_dict(data['time_constraints']) 
                if data.get('time_constraints') else None,
            location_filter=LocationFilter.from_dict(data['location_filter']) 
                if data.get('location_filter') else None,
            presence_requirements=PresenceRequirements.from_dict(data['presence_requirements'])
                if data.get('presence_requirements') else None,
            output_options=OutputOptions.from_dict(data['output_options']) 
                if data.get('output_options') else OutputOptions(),
            ambiguities=[Ambiguity.from_dict(amb) for amb in data.get('ambiguities', [])],
            confidence=data.get('confidence', 0.95)
        )