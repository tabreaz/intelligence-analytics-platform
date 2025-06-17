# src/agents/risk_filter/constants.py
"""
Constants for Risk Filter Agent
"""

# Token estimation
TOKEN_ESTIMATION_DIVISOR = 4

# Response parsing
MAX_RESPONSE_LOG_LENGTH = 500

# Default confidence
DEFAULT_CONFIDENCE = 0.8

# Retry settings
MAX_RETRIES = 2
RETRY_DELAY = 1.0

# Risk score thresholds (for reference in prompts)
RISK_THRESHOLDS = {
    "high": 0.7,
    "medium": 0.3,
    "low": 0.1
}
