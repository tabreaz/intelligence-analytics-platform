# src/agents/unified_filter/constants.py
"""
Constants for Unified Filter Agent
"""

# Agent configuration
DEFAULT_CONFIDENCE = 0.85
MAX_RETRIES = 2
RETRY_DELAY = 1.0

# Filter complexity thresholds
COMPLEXITY_LOW = 3
COMPLEXITY_MEDIUM = 6
COMPLEXITY_HIGH = 9

# Cache settings
CACHE_TTL = 300  # 5 minutes

# LLM response parsing
MAX_RESPONSE_LENGTH = 10000
JSON_START_MARKERS = ["```json", "```", "{"]
JSON_END_MARKERS = ["```", "}"]