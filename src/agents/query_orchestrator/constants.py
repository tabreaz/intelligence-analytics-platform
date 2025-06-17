# src/agents/query_orchestrator/constants.py
"""
Constants for Query Orchestrator Agent
"""

# Agent names mapping
AGENT_MAPPING = {
    "classifier": "query_classifier",
    "time": "time_parser",
    "location": "location_extractor",
    "profile": "profile_filter",
    "risk": "risk_filter",
    "entity": "entity_annotator"
}

# Agents that process filters
FILTER_AGENTS = ["time", "location", "profile", "risk"]

# Maximum retries for ambiguity resolution
MAX_AMBIGUITY_RETRIES = 3

# Parallel execution settings
PARALLEL_TIMEOUT_SECONDS = 30
MAX_CONCURRENT_AGENTS = 5
