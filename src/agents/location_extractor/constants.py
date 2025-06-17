# src/agents/location_extractor/constants.py
"""
Constants for Location Extractor Agent
"""

# Default radius when location is found but radius not specified
DEFAULT_RADIUS_METERS = 500

# Radius bounds validation
MIN_RADIUS_METERS = 100
MAX_RADIUS_METERS = 5000

# Confidence thresholds
MIN_CONFIDENCE = 0.0
MAX_CONFIDENCE = 1.0
DEFAULT_CONFIDENCE = 0.5

# Response parsing
MAX_ERROR_LOG_LENGTH = 200  # Characters to log from errors
MAX_RESPONSE_LOG_LENGTH = 500  # Characters to log from responses

# Google Places API
DEFAULT_MAX_RESULTS = 15

# Geohash validation (if needed by location extractor)
GEOHASH7_LENGTH = 7
GEOHASH6_LENGTH = 6
