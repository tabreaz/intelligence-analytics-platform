# src/agents/movement/parsers/single_location_parser.py
"""
Parser for Single Location queries
"""
import json
import logging
from typing import Dict, Any

from ..models.single_location import SingleLocationResult, Ambiguity
from ..templates.single_location_prompt import SINGLE_LOCATION_PROMPT
from ..utils import parse_json_response

logger = logging.getLogger(__name__)


class SingleLocationParser:
    """
    Parser for single location queries
    Handles both JSON parsing and model conversion
    """
    
    def __init__(self):
        self.prompt = SINGLE_LOCATION_PROMPT
        
    def parse_response(self, llm_response: str) -> SingleLocationResult:
        """
        Parse LLM response and return SingleLocationResult
        
        Args:
            llm_response: Raw JSON string from LLM
            
        Returns:
            SingleLocationResult object with parsed data
        """
        try:
            # Parse JSON
            data = parse_json_response(llm_response)
            
            # Create model from parsed data
            result = SingleLocationResult.from_dict(data)
            
            # Validate the result
            self._validate_result(result)
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            # Return error result
            return SingleLocationResult(
                reasoning="Failed to parse LLM response",
                confidence=0.0,
                ambiguities=[Ambiguity(
                    parameter="response",
                    issue=f"Invalid JSON: {str(e)}",
                    suggested_clarification="Please retry the query"
                )]
            )
        except Exception as e:
            logger.error(f"Error parsing single location response: {e}")
            return SingleLocationResult(
                reasoning="Error processing response",
                confidence=0.0,
                ambiguities=[Ambiguity(
                    parameter="processing",
                    issue=f"Processing error: {str(e)}",
                    suggested_clarification="Please retry the query"
                )]
            )
    
    def _validate_result(self, result: SingleLocationResult) -> None:
        """
        Validate the parsed result
        Adds warnings or adjusts confidence if issues found
        """
        warnings = []
        
        # Check if location filter is present
        if not result.location_filter:
            warnings.append("No location filter specified")
            result.confidence *= 0.8
        
        # Validate location method
        if result.location_filter and result.location_filter.method not in ["name", "coordinates", "polygon"]:
            warnings.append(f"Unknown location method: {result.location_filter.method}")
        
        # Validate time constraints
        if result.time_constraints:
            # Check hour ranges
            for hour_range in result.time_constraints.included_hours:
                if not (0 <= hour_range <= 23):
                    warnings.append(f"Invalid hour value: {hour_range}")
                    
        # Log warnings
        if warnings:
            logger.warning(f"Validation warnings: {warnings}")
            
    def get_prompt(self) -> str:
        """Get the single location prompt"""
        return self.prompt
    
    def format_for_output(self, result: SingleLocationResult) -> Dict[str, Any]:
        """
        Format result for API output
        Returns both the model and the JSON representation
        """
        return {
            "model": result,  # The actual model object
            "json": result.to_dict(),  # JSON representation
            "metadata": {
                "parser": "single_location",
                "confidence": result.confidence,
                "has_ambiguities": len(result.ambiguities) > 0
            }
        }