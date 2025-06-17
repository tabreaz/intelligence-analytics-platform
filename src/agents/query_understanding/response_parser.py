# src/agents/query_understanding/response_parser.py
"""
Response parser for query understanding agent
"""
import json
import re
from typing import Dict, Any

from src.core.logger import get_logger

logger = get_logger(__name__)


class QueryUnderstandingResponseParser:
    """Parse LLM responses specific to query understanding classification"""

    # Markdown patterns to clean
    MARKDOWN_PATTERNS = [
        (r'^```json\s*\n?', ''),  # Opening ```json
        (r'^```\s*\n?', ''),  # Opening ```
        (r'\n?```$', ''),  # Closing ```
        (r'^\s*```json\s*', ''),  # Inline ```json
        (r'\s*```\s*$', ''),  # Inline closing ```
    ]

    # Default structure specific to query understanding responses
    DEFAULT_STRUCTURE = {
        "classification": {"category": None, "confidence": 0.0},
        "context_aware_query": "",
        "tables_required": [],
        "entities_detected": {},
        "mapped_parameters": {},
        "ambiguities": [],
        "validation_warnings": []
    }

    @classmethod
    def parse(
            cls,
            response: str,
            strict: bool = False
    ) -> Dict[str, Any]:
        """
        Parse classification response from LLM
        
        Args:
            response: Raw LLM response string
            strict: If True, raise exception on parse failure
            
        Returns:
            Parsed classification result
        """
        try:
            # Clean and parse
            cleaned = cls._clean_markdown(response)
            result = json.loads(cleaned)

            if not isinstance(result, dict):
                raise ValueError(f"Expected dictionary, got {type(result).__name__}")

            # Ensure required fields exist
            return cls._ensure_structure(result)

        except (json.JSONDecodeError, ValueError) as e:
            from .exceptions import ParsingError
            from .constants import MAX_ERROR_LOG_LENGTH

            error_msg = f"Failed to parse query understanding response: {e}"
            logger.error(error_msg)
            logger.debug(f"Raw response: {response[:MAX_ERROR_LOG_LENGTH]}...")

            if strict:
                raise ParsingError(error_msg, response)

            # Return safe default
            default = cls.DEFAULT_STRUCTURE.copy()
            default["validation_warnings"].append("Failed to parse LLM response")
            return default

    @classmethod
    def _clean_markdown(cls, response: str) -> str:
        """Remove markdown code blocks"""
        if not response:
            return "{}"

        cleaned = response.strip()

        # Apply patterns
        for pattern, replacement in cls.MARKDOWN_PATTERNS:
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.MULTILINE)

        # Extract JSON object
        first_brace = cleaned.find('{')
        last_brace = cleaned.rfind('}')

        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            cleaned = cleaned[first_brace:last_brace + 1]

        return cleaned.strip()

    @classmethod
    def _ensure_structure(cls, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure all required fields exist"""
        result = cls.DEFAULT_STRUCTURE.copy()

        # Deep merge preserving structure
        for key, value in parsed.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key].update(value)
            else:
                result[key] = value

        return result
