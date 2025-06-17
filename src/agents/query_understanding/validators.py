# src/agents/query_understanding/validators.py
"""
Input validation utilities for query understanding
"""
import re
import uuid
from typing import Optional, Union, Dict, Any

from .constants import (
    MIN_RADIUS_METERS, MAX_RADIUS_METERS,
    MIN_CONFIDENCE, MAX_CONFIDENCE
)


class QueryValidator:
    """Validate query inputs"""

    MAX_QUERY_LENGTH = 1000  # Maximum characters in a query
    MIN_QUERY_LENGTH = 2  # Minimum meaningful query

    @classmethod
    def validate_query(cls, query: str) -> tuple[bool, Optional[str]]:
        """
        Validate query text
        
        Returns:
            (is_valid, error_message)
        """
        if not query or not isinstance(query, str):
            return False, "Query must be a non-empty string"

        query = query.strip()

        if len(query) < cls.MIN_QUERY_LENGTH:
            return False, f"Query too short (minimum {cls.MIN_QUERY_LENGTH} characters)"

        if len(query) > cls.MAX_QUERY_LENGTH:
            return False, f"Query too long (maximum {cls.MAX_QUERY_LENGTH} characters)"

        # Check for suspicious patterns (SQL injection, etc.)
        suspicious_patterns = [
            r';\s*(DROP|DELETE|UPDATE|INSERT|ALTER)',
            r'<script',
            r'javascript:',
            r'--\s*$'
        ]

        for pattern in suspicious_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return False, "Query contains suspicious patterns"

        return True, None

    @classmethod
    def validate_session_id(cls, session_id: Optional[str]) -> tuple[bool, Optional[str]]:
        """
        Validate session ID format
        
        Returns:
            (is_valid, error_message)
        """
        if not session_id:
            return True, None  # Optional parameter

        if not isinstance(session_id, str):
            return False, "Session ID must be a string"

        # Check if it's a valid UUID
        try:
            uuid.UUID(session_id)
            return True, None
        except ValueError:
            return False, "Session ID must be a valid UUID"

    @classmethod
    def validate_query_id(cls, query_id: str) -> tuple[bool, Optional[str]]:
        """
        Validate query ID format
        
        Returns:
            (is_valid, error_message)
        """
        if not query_id or not isinstance(query_id, str):
            return False, "Query ID must be a non-empty string"

        try:
            uuid.UUID(query_id)
            return True, None
        except ValueError:
            return False, "Query ID must be a valid UUID"


class LocationValidator:
    """Validate location-related inputs"""

    @classmethod
    def validate_radius(cls, radius: Union[int, float]) -> tuple[bool, Optional[str]]:
        """
        Validate radius in meters
        
        Returns:
            (is_valid, error_message)
        """
        try:
            radius_int = int(radius)
            if radius_int < MIN_RADIUS_METERS:
                return False, f"Radius too small (minimum {MIN_RADIUS_METERS}m)"
            if radius_int > MAX_RADIUS_METERS:
                return False, f"Radius too large (maximum {MAX_RADIUS_METERS}m)"
            return True, None
        except (ValueError, TypeError):
            return False, "Radius must be a valid number"

    @classmethod
    def validate_coordinates(cls, lat: float, lng: float) -> tuple[bool, Optional[str]]:
        """
        Validate latitude and longitude
        
        Returns:
            (is_valid, error_message)
        """
        try:
            lat = float(lat)
            lng = float(lng)

            if not -90 <= lat <= 90:
                return False, "Latitude must be between -90 and 90"

            if not -180 <= lng <= 180:
                return False, "Longitude must be between -180 and 180"

            return True, None
        except (ValueError, TypeError):
            return False, "Coordinates must be valid numbers"

    @classmethod
    def validate_geohash(cls, geohash: str, expected_length: Optional[int] = None) -> tuple[bool, Optional[str]]:
        """
        Validate geohash format
        
        Returns:
            (is_valid, error_message)
        """
        if not geohash or not isinstance(geohash, str):
            return False, "Geohash must be a non-empty string"

        # Check valid geohash characters
        valid_chars = '0123456789bcdefghjkmnpqrstuvwxyz'
        if not all(c in valid_chars for c in geohash.lower()):
            return False, "Geohash contains invalid characters"

        if expected_length and len(geohash) != expected_length:
            return False, f"Geohash must be exactly {expected_length} characters"

        return True, None

    @classmethod
    def validate_location_key(cls, loc_key: str) -> tuple[bool, Optional[int]]:
        """
        Validate and extract location index from key like 'location1'
        
        Returns:
            (is_valid, location_index or None)
        """
        if not loc_key or not isinstance(loc_key, str):
            return False, None

        if not loc_key.startswith('location'):
            return False, None

        try:
            index = int(loc_key.replace('location', ''))
            if index < 1:
                return False, None
            return True, index
        except ValueError:
            return False, None


class ConfidenceValidator:
    """Validate confidence scores"""

    @classmethod
    def validate_confidence(cls, confidence: Union[float, int]) -> tuple[bool, Optional[str]]:
        """
        Validate confidence score
        
        Returns:
            (is_valid, error_message)
        """
        try:
            conf_float = float(confidence)
            if not MIN_CONFIDENCE <= conf_float <= MAX_CONFIDENCE:
                return False, f"Confidence must be between {MIN_CONFIDENCE} and {MAX_CONFIDENCE}"
            return True, None
        except (ValueError, TypeError):
            return False, "Confidence must be a valid number"


class StructureValidator:
    """Validate complex data structures"""

    @classmethod
    def validate_location_result(cls, result: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate location extraction result structure
        
        Returns:
            (is_valid, error_message)
        """
        if not isinstance(result, dict):
            return False, "Location result must be a dictionary"

        if 'locations' not in result:
            return False, "Location result must contain 'locations' field"

        locations = result['locations']
        if not isinstance(locations, dict):
            return False, "'locations' must be a dictionary"

        # Validate each location
        for loc_key, loc_data in locations.items():
            is_valid, index = LocationValidator.validate_location_key(loc_key)
            if not is_valid:
                return False, f"Invalid location key: {loc_key}"

            if not isinstance(loc_data, dict):
                return False, f"Location data for {loc_key} must be a dictionary"

            # Required fields
            if 'name' not in loc_data:
                return False, f"Location {loc_key} missing 'name' field"

            if 'radius_meters' in loc_data:
                is_valid, error = LocationValidator.validate_radius(loc_data['radius_meters'])
                if not is_valid:
                    return False, f"Location {loc_key}: {error}"

        return True, None
