# src/agents/location_extractor/validators.py
"""
Input validation utilities for location extractor
"""
from typing import Optional, Union, Dict, Any

from .constants import (
    MIN_RADIUS_METERS, MAX_RADIUS_METERS,
    MIN_CONFIDENCE, MAX_CONFIDENCE
)


class LocationQueryValidator:
    """Validate location extraction queries"""

    MAX_PROMPT_LENGTH = 1000  # Maximum characters in a prompt

    @classmethod
    def validate_prompt(cls, prompt: str) -> tuple[bool, Optional[str]]:
        """
        Validate location extraction prompt
        
        Returns:
            (is_valid, error_message)
        """
        if not prompt or not isinstance(prompt, str):
            return False, "Prompt must be a non-empty string"

        prompt = prompt.strip()

        if len(prompt) > cls.MAX_PROMPT_LENGTH:
            return False, f"Prompt too long (maximum {cls.MAX_PROMPT_LENGTH} characters)"

        return True, None

    @classmethod
    def validate_location_data(cls, location_data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate individual location data structure
        
        Returns:
            (is_valid, error_message)
        """
        if not isinstance(location_data, dict):
            return False, "Location data must be a dictionary"

        # Required field: location
        if 'location' not in location_data or not location_data['location']:
            return False, "Location data must have a 'location' field"

        # Optional field validation
        if 'radius_meters' in location_data:
            try:
                radius = int(location_data['radius_meters'])
                if radius < MIN_RADIUS_METERS or radius > MAX_RADIUS_METERS:
                    return False, f"Radius must be between {MIN_RADIUS_METERS} and {MAX_RADIUS_METERS}"
            except (ValueError, TypeError):
                return False, "Radius must be a valid number"

        if 'confidence' in location_data:
            try:
                conf = float(location_data['confidence'])
                if conf < MIN_CONFIDENCE or conf > MAX_CONFIDENCE:
                    return False, f"Confidence must be between {MIN_CONFIDENCE} and {MAX_CONFIDENCE}"
            except (ValueError, TypeError):
                return False, "Confidence must be a valid number"

        return True, None


class CoordinateValidator:
    """Validate geographic coordinates"""

    @classmethod
    def validate_lat_lng(cls, lat: Union[float, str], lng: Union[float, str]) -> tuple[bool, Optional[str]]:
        """
        Validate latitude and longitude pair
        
        Returns:
            (is_valid, error_message)
        """
        try:
            lat_float = float(lat)
            lng_float = float(lng)

            if not -90 <= lat_float <= 90:
                return False, "Latitude must be between -90 and 90"

            if not -180 <= lng_float <= 180:
                return False, "Longitude must be between -180 and 180"

            return True, None

        except (ValueError, TypeError):
            return False, "Coordinates must be valid numbers"
