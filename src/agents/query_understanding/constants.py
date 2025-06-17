# src/agents/query_understanding/constants.py
"""
Shared constants for query understanding components
"""

# Continuation phrases that suggest context inheritance
CONTINUATION_PHRASES = [
    "how about", "what about", "and for", "also show", "instead",
    "also at", "and near", "as well as", "in addition to",
    "except at", "but not at", "rather than", "same for"
]

# Location modification phrases
LOCATION_MODIFIERS = [
    "closer to", "further from", "near to", "around",
    "within", "outside", "beyond", "before", "after"
]

# Replacement indicators
REPLACEMENT_INDICATORS = [
    "instead", "rather than", "not at", "except"
]

# Addition indicators
ADDITION_INDICATORS = [
    "also at", "as well as", "in addition", "and near"
]

# Modification indicators
MODIFICATION_INDICATORS = [
    "closer to", "further from", "around", "near"
]

# Token estimation constants
TOKEN_ESTIMATION_DIVISOR = 4  # Rough estimation: 1 token ≈ 4 characters

# Query processing limits
MAX_QUERY_HISTORY_SIZE = 3  # Number of previous queries to consider
MIN_CONTINUATION_QUERY_WORDS = 5  # Minimum words to consider as continuation

# Geohash validation
GEOHASH7_LENGTH = 7
GEOHASH6_LENGTH = 6

# Location validation  
MIN_RADIUS_METERS = 100
MAX_RADIUS_METERS = 5000
DEFAULT_RADIUS_METERS = 500

# Confidence thresholds
MIN_CONFIDENCE = 0.0
MAX_CONFIDENCE = 1.0
DEFAULT_CONFIDENCE = 0.5

# Response parsing
MAX_RESPONSE_LOG_LENGTH = 500  # Characters to log from responses
MAX_ERROR_LOG_LENGTH = 200  # Characters to log from errors

# Database operation timeouts (milliseconds)
DB_INSERT_TIMEOUT_MS = 5000
DB_QUERY_TIMEOUT_MS = 3000

# Session management
DEFAULT_SESSION_DURATION_HOURS = 24

# Location ambiguity radius suggestions
SMALL_RADIUS_SUGGESTIONS = [200, 300, 500]  # For radius <= 300m
MEDIUM_RADIUS_SUGGESTIONS = [500, 750, 1000]  # For radius <= 500m
LARGE_RADIUS_MULTIPLIERS = [1, 2, 5]  # For larger radii
