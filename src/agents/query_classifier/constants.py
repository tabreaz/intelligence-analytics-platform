# src/agents/query_classifier/constants.py
"""
Constants for Query Classifier Agent
"""

# Continuation phrases that suggest context inheritance
CONTINUATION_PHRASES = [
    "how about", "what about", "and for", "also show", "instead",
    "also at", "and near", "as well as", "in addition to",
    "except at", "but not at", "rather than", "same for",
    "but with", "and exclude", "but only", "and also"
]

# Replacement indicators
REPLACEMENT_INDICATORS = [
    "instead", "rather than", "not", "except", "but not"
]

# Addition indicators
ADDITION_INDICATORS = [
    "also", "as well as", "in addition", "and", "plus", "including"
]

# Token estimation constants
TOKEN_ESTIMATION_DIVISOR = 4  # Rough estimation: 1 token ≈ 4 characters

# Query processing limits
MAX_QUERY_HISTORY_SIZE = 5  # Number of previous queries to consider

# Confidence thresholds
MIN_CONFIDENCE = 0.0
MAX_CONFIDENCE = 1.0
DEFAULT_CONFIDENCE = 0.5

# Response parsing
MAX_RESPONSE_LOG_LENGTH = 500  # Characters to log from responses

# Special category keys
UNSUPPORTED_DOMAIN_KEY = "unsupported_domain"
GENERAL_INQUIRY_KEY = "general_inquiry"

# Valid domain names
DOMAIN_PROFILE = "profile"
DOMAIN_MOVEMENT = "movement"
DOMAIN_COMMUNICATION = "communication"
DOMAIN_RISK_PROFILES = "risk_profiles"

VALID_DOMAINS = [
    DOMAIN_PROFILE,
    DOMAIN_MOVEMENT,
    DOMAIN_COMMUNICATION,
    DOMAIN_RISK_PROFILES
]

# Valid agents for agents_required field
AGENT_PROFILE = "profile"
AGENT_TIME = "time"
AGENT_LOCATION = "location"
AGENT_RISK = "risk"
AGENT_COMMUNICATION = "communication"
AGENT_MOVEMENT = "movement"

VALID_AGENTS = [
    AGENT_PROFILE,
    AGENT_TIME,
    AGENT_LOCATION,
    AGENT_RISK,
    AGENT_COMMUNICATION,
    AGENT_MOVEMENT
]
