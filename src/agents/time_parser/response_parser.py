# src/agents/time_parser/response_parser.py
"""
Response parser for Time Parser Agent
"""
import json
import re
from typing import Dict, Any, List

from src.core.logger import get_logger
from .exceptions import InvalidTimeExpressionError

logger = get_logger(__name__)


class TimeParserResponseParser:
    """
    Parse and validate time parser LLM responses
    """

    @classmethod
    def parse(cls, response: str) -> Dict[str, Any]:
        """
        Parse time extraction response from LLM
        
        Args:
            response: Raw LLM response string
            
        Returns:
            Parsed and validated time data structure
        """
        default_structure = {
            "date_ranges": [],
            "excluded_date_ranges": [],
            "hour_constraints": [],
            "day_constraints": [],
            "composite_constraints": None,
            "event_mappings": [],
            "raw_expressions": [],
            "default_range": None
        }

        if not response:
            logger.warning("Empty response received")
            return default_structure

        try:
            # Clean and extract JSON
            cleaned = cls._clean_response(response)

            # Parse JSON
            result = json.loads(cleaned)

            # Validate structure
            if not isinstance(result, dict):
                raise InvalidTimeExpressionError("Response is not a valid object")

            # Merge with defaults and validate
            validated = cls._validate_structure(result, default_structure)

            return validated

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Raw response: {response[:500]}")
            return default_structure

        except Exception as e:
            logger.error(f"Unexpected error parsing response: {e}")
            return default_structure

    @classmethod
    def _clean_response(cls, response: str) -> str:
        """
        Clean response by removing markdown and extracting JSON
        """
        # Remove markdown code blocks
        cleaned = re.sub(r'```json?\s*', '', response)
        cleaned = re.sub(r'```\s*$', '', cleaned)

        # Remove any leading/trailing whitespace
        cleaned = cleaned.strip()

        # Try to find JSON object boundaries
        start_idx = cleaned.find('{')
        end_idx = cleaned.rfind('}')

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            cleaned = cleaned[start_idx:end_idx + 1]

        # Remove any text before first { or after last }
        cleaned = re.sub(r'^[^{]*', '', cleaned)
        cleaned = re.sub(r'[^}]*$', '', cleaned)

        return cleaned

    @classmethod
    def _validate_structure(
            cls,
            result: Dict[str, Any],
            default_structure: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate and merge response with default structure
        """
        validated = default_structure.copy()

        # Validate date ranges
        if 'date_ranges' in result and isinstance(result['date_ranges'], list):
            validated['date_ranges'] = cls._validate_date_ranges(result['date_ranges'])

        # Validate excluded date ranges
        if 'excluded_date_ranges' in result and isinstance(result['excluded_date_ranges'], list):
            validated['excluded_date_ranges'] = cls._validate_date_ranges(result['excluded_date_ranges'])

        # Validate hour constraints
        if 'hour_constraints' in result and isinstance(result['hour_constraints'], list):
            validated['hour_constraints'] = cls._validate_hour_constraints(result['hour_constraints'])

        # Validate day constraints
        if 'day_constraints' in result and isinstance(result['day_constraints'], list):
            validated['day_constraints'] = cls._validate_day_constraints(result['day_constraints'])

        # Copy composite constraints
        if 'composite_constraints' in result and isinstance(result['composite_constraints'], dict):
            validated['composite_constraints'] = result['composite_constraints']

        # Validate event mappings
        if 'event_mappings' in result and isinstance(result['event_mappings'], list):
            validated['event_mappings'] = cls._validate_event_mappings(result['event_mappings'])

        # Copy raw expressions
        if 'raw_expressions' in result and isinstance(result['raw_expressions'], list):
            validated['raw_expressions'] = result['raw_expressions']

        # Copy default range if valid
        if 'default_range' in result and isinstance(result['default_range'], (dict, type(None))):
            validated['default_range'] = result['default_range']

        return validated

    @classmethod
    def _validate_date_ranges(cls, ranges: List[Any]) -> List[Dict[str, Any]]:
        """
        Validate date range format and data
        """
        validated = []

        for r in ranges:
            if not isinstance(r, dict):
                continue

            # Must have start and end
            if 'start' not in r or 'end' not in r:
                logger.warning(f"Date range missing start/end: {r}")
                continue

            # Validate required fields
            valid_range = {
                'type': r.get('type', 'absolute'),
                'start': str(r['start']),
                'end': str(r['end']),
                'original_text': r.get('original_text', ''),
                'confidence': float(r.get('confidence', 1.0)),
                'constraint_type': r.get('constraint_type', 'include'),
                'expand_to_dates': r.get('expand_to_dates', False)
            }

            validated.append(valid_range)

        return validated

    @classmethod
    def _validate_hour_constraints(cls, constraints: List[Any]) -> List[Dict[str, Any]]:
        """
        Validate hour constraints
        """
        validated = []

        for c in constraints:
            if not isinstance(c, dict):
                continue

            # Must have start_hour and end_hour
            if 'start_hour' not in c or 'end_hour' not in c:
                logger.warning(f"Hour constraint missing start/end: {c}")
                continue

            try:
                start_hour = int(c['start_hour'])
                end_hour = int(c['end_hour'])

                # Validate hour range
                if not (0 <= start_hour <= 23 and 0 <= end_hour <= 23):
                    logger.warning(f"Invalid hour range: {start_hour}-{end_hour}")
                    continue

                valid_constraint = {
                    'start_hour': start_hour,
                    'end_hour': end_hour,
                    'original_text': c.get('original_text', ''),
                    'days_applicable': c.get('days_applicable'),
                    'constraint_type': c.get('constraint_type', 'include'),
                    'excluded_hours': c.get('excluded_hours')
                }

                validated.append(valid_constraint)

            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid hour values: {e}")
                continue

        return validated

    @classmethod
    def _validate_day_constraints(cls, constraints: List[Any]) -> List[Dict[str, Any]]:
        """
        Validate day constraints with enhanced structure
        """
        valid_days = [
            'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
            'saturday', 'sunday', 'weekday', 'weekend', 'workday'
        ]

        validated = []

        for c in constraints:
            if isinstance(c, dict):
                # New format with constraint type
                days = c.get('days', [])
                valid_day_list = [d.lower() for d in days if d.lower() in valid_days]
                if valid_day_list:
                    validated.append({
                        'days': valid_day_list,
                        'constraint_type': c.get('constraint_type', 'include'),
                        'original_text': c.get('original_text', '')
                    })
            elif isinstance(c, str) and c.lower() in valid_days:
                # Legacy format - convert to new structure
                validated.append({
                    'days': [c.lower()],
                    'constraint_type': 'include',
                    'original_text': c
                })

        return validated

    @classmethod
    def _validate_event_mappings(cls, mappings: List[Any]) -> List[Dict[str, Any]]:
        """
        Validate event mappings
        """
        validated = []

        for m in mappings:
            if not isinstance(m, dict):
                continue

            # Must have event_name and dates
            if 'event_name' not in m or 'dates' not in m:
                logger.warning(f"Event mapping missing required fields: {m}")
                continue

            if isinstance(m['dates'], list):
                valid_mapping = {
                    'event_name': str(m['event_name']),
                    'dates': [str(d) for d in m['dates']],
                    'original_text': m.get('original_text', '')
                }
                validated.append(valid_mapping)

        return validated
