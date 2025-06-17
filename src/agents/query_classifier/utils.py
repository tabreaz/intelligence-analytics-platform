# src/agents/query_classifier/utils.py
"""
Utility functions for Query Classifier Agent
"""
from typing import Dict, Any, Optional


def should_store_in_history(category: str) -> bool:
    """
    Determine if a query with given category should be stored in history
    
    Args:
        category: The query category
        
    Returns:
        bool: True if you should be stored, False otherwise
    """
    # Don't store unsupported queries in history
    excluded_categories = ['unsupported_domain']
    return category not in excluded_categories


def extract_continuation_type(query: str) -> Optional[str]:
    """
    Extract the type of continuation from query
    
    Args:
        query: The query text
        
    Returns:
        Optional[str]: Type of continuation (replace, add, modify) or None
    """
    query_lower = query.lower()

    # Check for replacement
    replace_phrases = ["instead", "rather than", "not", "except"]
    for phrase in replace_phrases:
        if phrase in query_lower:
            return "replace"

    # Check for addition
    add_phrases = ["also", "as well", "in addition", "and"]
    for phrase in add_phrases:
        if phrase in query_lower:
            return "add"

    # Check for modification
    modify_phrases = ["how about", "what about"]
    for phrase in modify_phrases:
        if phrase in query_lower:
            return "modify"

    return None


def merge_context(previous_params: Dict[str, Any],
                  modifications: Dict[str, Any],
                  continuation_type: str) -> Dict[str, Any]:
    """
    Merge previous context with modifications based on continuation type
    
    Args:
        previous_params: Previous query parameters
        modifications: New/changed parameters
        continuation_type: Type of continuation (replace, add, modify)
        
    Returns:
        Dict[str, Any]: Merged parameters
    """
    if continuation_type == "replace":
        # Replace specific values
        result = previous_params.copy()
        for key, value in modifications.items():
            result[key] = value

    elif continuation_type == "add":
        # Add to existing values
        result = previous_params.copy()
        for key, value in modifications.items():
            if key in result and isinstance(result[key], list):
                result[key].extend(value if isinstance(value, list) else [value])
            else:
                result[key] = value

    else:  # modify or default
        # Override with new values
        result = previous_params.copy()
        result.update(modifications)

    return result
