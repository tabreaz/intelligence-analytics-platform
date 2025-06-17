# src/agents/entity_annotator/constants.py
"""
Constants for Entity Annotator Agent
"""

# Agent configuration
DEFAULT_CONFIDENCE = 0.85
MAX_RETRIES = 2
RETRY_DELAY = 1.0

# Entity pattern for validation - accepts any uppercase entity type
ENTITY_PATTERN = r'\[([A-Z_]+):([^\]]+)\]'

# Note: We don't restrict entity types - LLM can create new ones as needed
# The prompt provides examples but encourages creating new types when appropriate
