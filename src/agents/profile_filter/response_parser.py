# src/agents/profile_filter/response_parser.py
"""
Response parser for Profile Filter Agent
"""
import json
from typing import Dict, Any

from .constants import FIELD_ALIASES
from .models import ProfileFilterResult, validate_field_existence
from ...core.logger import get_logger

logger = get_logger(__name__)


class ProfileFilterResponseParser:
    """Parser for Profile Filter Agent LLM responses"""

    def parse(self, llm_response: str) -> ProfileFilterResult:
        """Parse LLM response into structured result"""
        result = ProfileFilterResult()

        try:
            # Parse JSON response
            data = json.loads(llm_response)

            # Extract filter_tree (required)
            if 'filter_tree' in data:
                result.filter_tree = data['filter_tree']
            else:
                logger.warning("No filter_tree found in response")
                result.filter_tree = {}

            # Extract exclusions if present (optional)
            if 'exclusions' in data and isinstance(data['exclusions'], dict):
                result.exclusions = data['exclusions']

            # Extract ambiguities
            if 'ambiguities' in data and isinstance(data['ambiguities'], list):
                result.ambiguities = data['ambiguities']

            # Extract reasoning
            if 'reasoning' in data:
                result.raw_extractions['reasoning'] = data['reasoning']

            # Extract confidence
            result.confidence = float(data.get('confidence', 0.8))

            # Store raw response
            result.raw_extractions['llm_response'] = data

            # Validate extracted filters
            self._validate_filters(result)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            result.extraction_method = "failed"
            result.validation_warnings.append(f"JSON parse error: {str(e)}")
            result.confidence = 0.0
        except Exception as e:
            logger.error(f"Unexpected error parsing response: {e}")
            result.extraction_method = "error"
            result.validation_warnings.append(f"Parse error: {str(e)}")
            result.confidence = 0.0

        return result

    def _process_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Process and normalize filter values"""
        processed = {}

        for field, value in filters.items():
            # Normalize common field name variations
            normalized_field = self._normalize_field_name(field)

            # Skip unknown fields (they'll be caught in validation)
            if normalized_field != field:
                logger.debug(f"Normalized field name: {field} -> {normalized_field}")

            # Handle operator-based filters (e.g., age: {">": 30})
            if isinstance(value, dict) and any(
                    key in ['>', '<', '>=', '<=', '=', '!=', 'operator', 'range', 'between', 'exists'] for key in
                    value):
                # Standardize to operator/value format
                if 'operator' in value:
                    # Handle special operators
                    if value['operator'] == 'range' and 'value' in value and isinstance(value['value'], list):
                        # Convert range to BETWEEN
                        value['operator'] = 'BETWEEN'
                    elif value['operator'] == 'exists':
                        # Convert exists to notEmpty for arrays
                        value['operator'] = 'notEmpty'
                        value.pop('value', None)  # Remove value for exists
                    processed[normalized_field] = value
                else:
                    # Convert {">": 30} to {"operator": ">", "value": 30}
                    for op, val in value.items():
                        processed[normalized_field] = {"operator": op, "value": val}
                        break  # Take first operator
            else:
                # Simple value
                processed[normalized_field] = value

        return processed

    def _normalize_field_name(self, field: str) -> str:
        """Normalize field name using aliases from config"""
        # Use the loaded field aliases mapping
        normalized = FIELD_ALIASES.get(field, field)

        # Log normalization for debugging
        if normalized != field:
            logger.debug(f"Field normalized: '{field}' -> '{normalized}'")

        return normalized

    def _validate_filters(self, result: ProfileFilterResult) -> None:
        """Simple validation - just check if fields exist in schema"""

        # Validate field existence only
        warnings = validate_field_existence(result.filter_tree, result.exclusions)
        result.validation_warnings.extend(warnings)

        # Adjust confidence based on validation warnings
        if result.validation_warnings:
            # Reduce confidence if there are unknown fields
            result.confidence = max(0.3, result.confidence - 0.1 * len(result.validation_warnings))
