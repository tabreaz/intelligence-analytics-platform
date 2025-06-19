# src/agents/movement/response_parser.py
"""
Response parser for Movement Analysis Agent
"""
import json
from datetime import datetime
from typing import Dict, Any, List

from .models import (
    MovementFilterResult, Geofence, SpatialFilter,
    TimeConstraints, PresenceRequirements
)
from .utils import parse_json_response
from ...core.logger import get_logger

logger = get_logger(__name__)


class MovementResponseParser:
    """Parser for Movement Analysis Agent LLM responses"""

    def parse(self, llm_response: str) -> MovementFilterResult:
        """Parse LLM response into structured result"""
        result = MovementFilterResult()

        try:
            # Parse JSON response
            data = parse_json_response(llm_response)

            # Extract reasoning (required)
            result.reasoning = data.get('reasoning', 'No reasoning provided')

            # Extract query type (required)
            result.query_type = data.get('query_type', 'single_location')

            # Extract identity filters
            if 'identity_filters' in data and isinstance(data['identity_filters'], dict):
                result.identity_filters = data['identity_filters']

            # Extract co-presence
            if 'co_presence' in data and isinstance(data['co_presence'], dict):
                result.co_presence = data['co_presence']

            # Extract geofences
            if 'geofences' in data and isinstance(data['geofences'], list):
                result.geofences = self._parse_geofences(data['geofences'])

            # Extract heatmap
            if 'heatmap' in data and isinstance(data['heatmap'], dict):
                result.heatmap = data['heatmap']

            # Extract sequence patterns
            if 'sequence_patterns' in data and isinstance(data['sequence_patterns'], list):
                result.sequence_patterns = data['sequence_patterns']

            # Extract clustering
            if 'clustering' in data and isinstance(data['clustering'], dict):
                result.clustering = data['clustering']

            # Extract pattern detection
            if 'pattern_detection' in data and isinstance(data['pattern_detection'], dict):
                result.pattern_detection = data['pattern_detection']

            # Extract anomaly detection
            if 'anomaly_detection' in data and isinstance(data['anomaly_detection'], dict):
                result.anomaly_detection = data['anomaly_detection']

            # Extract predictive modeling
            if 'predictive_modeling' in data and isinstance(data['predictive_modeling'], dict):
                result.predictive_modeling = data['predictive_modeling']

            # Extract global time filter
            if 'global_time_filter' in data and isinstance(data['global_time_filter'], dict):
                result.global_time_filter = data['global_time_filter']

            # Extract output options
            if 'output_options' in data and isinstance(data['output_options'], dict):
                result.output_options = data['output_options']

            # Extract ambiguities
            if 'ambiguities' in data and isinstance(data['ambiguities'], list):
                result.ambiguities = data['ambiguities']

            # Extract confidence
            result.confidence = float(data.get('confidence', 0.8))

            # Store raw response
            result.raw_extractions['llm_response'] = data

            # Validate extracted data
            self._validate_result(result)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            result.extraction_method = "failed"
            result.validation_warnings.append(f"JSON parse error: {str(e)}")
            result.confidence = 0.0
        except Exception as e:
            logger.error(f"Unexpected error parsing response: {e}")
            result.extraction_method = "error"
            result.validation_warnings.append(f"Parse error: {str(e)}")
            result.confidence = 0.0

        return result

    def _parse_geofences(self, geofences_data: List[Dict[str, Any]]) -> List[Geofence]:
        """Parse geofences from JSON data"""
        geofences = []
        
        for idx, gf_data in enumerate(geofences_data):
            try:
                # Extract spatial filter
                sf_data = gf_data.get('spatial_filter', {})
                spatial_filter = SpatialFilter(
                    method=sf_data.get('method', 'name'),
                    value=sf_data.get('value'),
                    latitude=sf_data.get('latitude'),
                    longitude=sf_data.get('longitude'),
                    radius_meters=sf_data.get('radius_meters', 1000),
                    polygon=sf_data.get('polygon')
                )
                
                # Extract time constraints if present
                time_constraints = None
                if 'time_constraints' in gf_data:
                    tc_data = gf_data['time_constraints']
                    # Use TimeConstraints.from_dict to handle parsing properly
                    time_constraints = TimeConstraints.from_dict(tc_data)
                
                # Extract presence requirements if present
                presence_requirements = None
                if 'presence_requirements' in gf_data:
                    pr_data = gf_data['presence_requirements']
                    presence_requirements = PresenceRequirements(
                        minimum_duration_minutes=pr_data.get('minimum_duration_minutes'),
                        minimum_visits=pr_data.get('minimum_visits'),
                        aggregation_period=pr_data.get('aggregation_period')
                    )
                
                # Create geofence
                geofence = Geofence(
                    id=gf_data.get('id', f'gf_{idx + 1}'),
                    reference=gf_data.get('reference', ''),
                    spatial_filter=spatial_filter,
                    time_constraints=time_constraints,
                    presence_requirements=presence_requirements
                )
                
                geofences.append(geofence)
                
            except Exception as e:
                logger.warning(f"Failed to parse geofence {idx}: {e}")
                
        return geofences

    def _validate_result(self, result: MovementFilterResult) -> None:
        """Validate the parsed result"""
        warnings = []
        
        # Validate query type
        valid_query_types = [
            "single_location", "multi_location_and", "multi_location_or",
            "movement_pattern", "co_presence", "heatmap", "predictive_analysis",
            "anomaly_detection", "multi_modal_movement_analysis"
        ]
        if result.query_type not in valid_query_types:
            warnings.append(f"Invalid query_type: {result.query_type}")
        
        # Validate geofences
        for idx, geofence in enumerate(result.geofences):
            # Validate spatial filter
            if geofence.spatial_filter.method == "coordinates":
                if geofence.spatial_filter.latitude is None or geofence.spatial_filter.longitude is None:
                    warnings.append(f"Geofence {geofence.id}: coordinates method requires latitude and longitude")
            elif geofence.spatial_filter.method == "name" or geofence.spatial_filter.method == "area":
                if not geofence.spatial_filter.value:
                    warnings.append(f"Geofence {geofence.id}: {geofence.spatial_filter.method} method requires value")
            elif geofence.spatial_filter.method == "polygon":
                if not geofence.spatial_filter.polygon:
                    warnings.append(f"Geofence {geofence.id}: polygon method requires polygon coordinates")
                    
            # Validate time constraints
            if geofence.time_constraints:
                # Validate date ranges (both included and excluded)
                for date_range in geofence.time_constraints.included_date_ranges:
                    try:
                        datetime.fromisoformat(date_range.start.replace('Z', '+00:00'))
                        datetime.fromisoformat(date_range.end.replace('Z', '+00:00'))
                    except ValueError:
                        warnings.append(f"Geofence {geofence.id}: Invalid date format in included_date_ranges")
                
                for date_range in geofence.time_constraints.excluded_date_ranges:
                    try:
                        datetime.fromisoformat(date_range.start.replace('Z', '+00:00'))
                        datetime.fromisoformat(date_range.end.replace('Z', '+00:00'))
                    except ValueError:
                        warnings.append(f"Geofence {geofence.id}: Invalid date format in excluded_date_ranges")
                        
                # Validate hour values
                for hour in geofence.time_constraints.included_hours:
                    if not (0 <= hour <= 23):
                        warnings.append(f"Geofence {geofence.id}: Hour values must be between 0-23")
                        
                for hour in geofence.time_constraints.excluded_hours:
                    if not (0 <= hour <= 23):
                        warnings.append(f"Geofence {geofence.id}: Hour values must be between 0-23")
        
        # Validate co-presence
        if result.co_presence:
            if 'target_ids' not in result.co_presence or not result.co_presence['target_ids']:
                warnings.append("Co-presence requires target_ids")
                
        # Store validation warnings
        result.validation_warnings.extend(warnings)
        
        # Adjust confidence based on warnings
        if warnings:
            result.confidence = max(0.3, result.confidence - 0.1 * len(warnings))