# src/agents/movement/utils.py
"""
Utility functions for movement agents
"""
import json
import re
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def parse_json_response(response: str) -> Dict[str, Any]:
    """
    Parse JSON response from LLM, handling markdown code blocks and finding JSON objects
    
    Args:
        response: Raw response string from LLM
        
    Returns:
        Parsed JSON as dictionary
        
    Raises:
        json.JSONDecodeError: If response is not valid JSON
    """
    if not response:
        raise json.JSONDecodeError("Empty response", response, 0)
    
    # First attempt: Try to parse as-is
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass
    
    # Second attempt: Strip markdown code blocks
    cleaned_response = response.strip()
    
    # Remove opening markdown
    if cleaned_response.startswith("```json"):
        cleaned_response = cleaned_response[7:]  # Remove ```json
    elif cleaned_response.startswith("```"):
        cleaned_response = cleaned_response[3:]  # Remove ```
    
    # Remove closing markdown
    if cleaned_response.endswith("```"):
        cleaned_response = cleaned_response[:-3]
        
    # Clean up any remaining whitespace
    cleaned_response = cleaned_response.strip()
    
    # Try parsing the cleaned response
    try:
        return json.loads(cleaned_response)
    except json.JSONDecodeError:
        pass
    
    # Third attempt: Find JSON object using regex
    # First try to find JSON within code blocks
    code_block_match = re.search(r'```(?:json)?\s*(\{[^`]*\})\s*```', response, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Then look for content between outermost { and }
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    # Fourth attempt: Find JSON array
    json_array_match = re.search(r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]', response, re.DOTALL)
    if json_array_match:
        try:
            return json.loads(json_array_match.group(0))
        except json.JSONDecodeError:
            pass
    
    # If all attempts fail, raise the original error
    raise json.JSONDecodeError(
        f"Could not find valid JSON in response. Response starts with: {response[:100]}...", 
        response, 
        0
    )