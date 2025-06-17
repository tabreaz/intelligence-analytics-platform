# src/agents/location_extractor/response_parser.py
"""
Response parser for location extractor agent
"""
import json
import re
from typing import List, Dict, Any

from src.core.logger import get_logger

logger = get_logger(__name__)

from .constants import (
    DEFAULT_RADIUS_METERS, MIN_RADIUS_METERS, MAX_RADIUS_METERS,
    MIN_CONFIDENCE, MAX_CONFIDENCE, DEFAULT_CONFIDENCE,
    MAX_ERROR_LOG_LENGTH
)


class LocationExtractorResponseParser:
    """Parse LLM responses specific to location extraction"""

    @classmethod
    def parse(
            cls,
            response: str,
            strict: bool = False
    ) -> Dict[str, Any]:
        """
        Parse location extraction response from LLM
        
        Args:
            response: Raw LLM response string
            strict: If True, raise exception on parse failure
            
        Returns:
            Dictionary with 'locations' and 'ambiguities' arrays
        """
        try:
            # Clean response
            cleaned = cls._clean_response(response)

            # Parse JSON
            result = json.loads(cleaned)

            # Handle different response formats
            if isinstance(result, dict) and "locations" in result:
                # New format with locations and ambiguities
                return {
                    "locations": cls._validate_locations(result.get("locations", [])),
                    "ambiguities": cls._validate_ambiguities(result.get("ambiguities", []))
                }
            elif isinstance(result, list):
                # Legacy format - just array of locations
                return {
                    "locations": cls._validate_locations(result),
                    "ambiguities": []
                }
            elif isinstance(result, dict):
                # Single location as dict - legacy format
                return {
                    "locations": cls._validate_locations([result]),
                    "ambiguities": []
                }
            else:
                raise ValueError(f"Unexpected response type: {type(result).__name__}")

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse location response: {e}")
            logger.debug(f"Raw response: {response[:MAX_ERROR_LOG_LENGTH]}...")

            if strict:
                raise

            # Return empty structure for invalid responses
            return {"locations": [], "ambiguities": []}

    @classmethod
    def _clean_response(cls, response: str) -> str:
        """Clean markdown and extract JSON"""
        if not response:
            return "[]"

        cleaned = response.strip()

        # Remove Markdown code blocks
        if cleaned.startswith("```"):
            # Remove opening ```json or ```
            cleaned = re.sub(r'^```\w*\n?', '', cleaned)
            # Remove closing ```
            cleaned = re.sub(r'\n?```$', '', cleaned)
            cleaned = cleaned.strip()

        # Extract JSON array or object
        # First try array
        first_bracket = cleaned.find('[')
        last_bracket = cleaned.rfind(']')

        if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
            return cleaned[first_bracket:last_bracket + 1]

        # Then try object
        first_brace = cleaned.find('{')
        last_brace = cleaned.rfind('}')

        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            return cleaned[first_brace:last_brace + 1]

        return cleaned

    @classmethod
    def _validate_locations(cls, locations: List[Any]) -> List[Dict[str, Any]]:
        """Validate and normalize location entries"""
        validated = []

        for loc in locations:
            if not isinstance(loc, dict):
                continue

            # Must have at least a location name
            if "location" not in loc or not loc["location"]:
                continue

            # Build validated entry - preserve ALL fields from LLM response
            validated_loc = {
                "location": str(loc["location"]),
                "context": str(loc.get("context", "unknown")),
                "confidence": cls._validate_confidence(loc.get("confidence"))
            }
            
            # Preserve the new fields: type, field, value
            if "type" in loc:
                validated_loc["type"] = str(loc["type"])
            
            if "field" in loc:
                validated_loc["field"] = str(loc["field"])
                
            if "value" in loc:
                validated_loc["value"] = str(loc["value"])
            
            # Only add radius_meters for FACILITY/ADDRESS types
            loc_type = loc.get("type", "FACILITY")
            if loc_type in ["FACILITY", "ADDRESS"]:
                validated_loc["radius_meters"] = cls._validate_radius(loc.get("radius_meters"))
            elif "radius_meters" in loc:
                # For CITY/EMIRATE, include radius if provided but don't add default
                validated_loc["radius_meters"] = cls._validate_radius(loc.get("radius_meters"))

            validated.append(validated_loc)

        return validated

    @classmethod
    def _validate_radius(cls, radius: Any) -> int:
        """Validate and return radius in meters"""
        if radius is None:
            return DEFAULT_RADIUS_METERS

        try:
            radius_int = int(radius)
            # Ensure reasonable bounds
            if radius_int < MIN_RADIUS_METERS:
                return MIN_RADIUS_METERS
            elif radius_int > MAX_RADIUS_METERS:
                return MAX_RADIUS_METERS
            return radius_int
        except (ValueError, TypeError):
            return DEFAULT_RADIUS_METERS

    @classmethod
    def _validate_confidence(cls, confidence: Any) -> float:
        """Validate confidence score"""
        if confidence is None:
            return DEFAULT_CONFIDENCE

        try:
            conf_float = float(confidence)
            # Ensure valid range
            return max(MIN_CONFIDENCE, min(MAX_CONFIDENCE, conf_float))
        except (ValueError, TypeError):
            return DEFAULT_CONFIDENCE
    
    @classmethod
    def _validate_ambiguities(cls, ambiguities: List[Any]) -> List[Dict[str, Any]]:
        """Validate and normalize ambiguity entries"""
        validated = []
        
        for amb in ambiguities:
            if not isinstance(amb, dict):
                continue
                
            # Must have at least a reference
            if "reference" not in amb or not amb["reference"]:
                continue
                
            # Build validated entry
            validated_amb = {
                "reference": str(amb["reference"]),
                "ambiguity_type": str(amb.get("ambiguity_type", "unknown")),
                "context": str(amb.get("context", "")),
                "suggestions": amb.get("suggestions", []),
                "potential_count": str(amb.get("potential_count", "unknown")),
                "clarification_needed": bool(amb.get("clarification_needed", True)),
                "severity": str(amb.get("severity", "medium"))
            }
            
            # Ensure suggestions is a list
            if not isinstance(validated_amb["suggestions"], list):
                validated_amb["suggestions"] = [str(validated_amb["suggestions"])]
            
            validated.append(validated_amb)
            
        return validated
