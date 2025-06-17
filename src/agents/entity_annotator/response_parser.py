# src/agents/entity_annotator/response_parser.py
"""
Response parser for Entity Annotator Agent
"""
import json
import re
from typing import List

from .constants import ENTITY_PATTERN
from .models import EntityAnnotatorResult, Entity
from ...core.logger import get_logger

logger = get_logger(__name__)


class EntityAnnotatorResponseParser:
    """Parser for Entity Annotator Agent LLM responses"""

    def parse(self, llm_response: str, original_query: str) -> EntityAnnotatorResult:
        """Parse LLM response into structured result"""
        result = EntityAnnotatorResult(
            query=original_query,
            annotated_query=original_query  # Default to original if parsing fails
        )

        try:
            # Parse JSON response
            data = json.loads(llm_response)

            # Extract basic fields
            result.query = data.get('query', original_query)
            result.annotated_query = data.get('annotated_query', original_query)
            result.confidence = float(data.get('confidence', 0.8))
            result.raw_response = data

            # Extract entities from JSON if provided
            if 'entities' in data and isinstance(data['entities'], list):
                for entity_data in data['entities']:
                    if all(k in entity_data for k in ['type', 'value', 'start_pos', 'end_pos']):
                        entity = Entity(
                            type=entity_data['type'],
                            value=entity_data['value'],
                            start_pos=entity_data['start_pos'],
                            end_pos=entity_data['end_pos']
                        )
                        result.entities.append(entity)

            # Also parse entities from annotated_query as backup
            if not result.entities and result.annotated_query:
                result.entities = self._extract_entities_from_annotation(result.annotated_query)

            # Extract entity types
            if 'entity_types' in data and isinstance(data['entity_types'], list):
                result.entity_types = data['entity_types']
            else:
                # Derive from entities
                result.entity_types = list(set(e.type for e in result.entities))

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            result.extraction_method = "regex_fallback"
            result.confidence = 0.6

            # Try to extract annotated query from raw response
            result.annotated_query = self._extract_annotated_query(llm_response)
            if result.annotated_query:
                result.entities = self._extract_entities_from_annotation(result.annotated_query)
                result.entity_types = list(set(e.type for e in result.entities))

        except Exception as e:
            logger.error(f"Unexpected error parsing response: {e}")
            result.extraction_method = "error"
            result.error = str(e)
            result.confidence = 0.0

        return result

    def _extract_entities_from_annotation(self, annotated_query: str) -> List[Entity]:
        """Extract entities from an annotated query string"""
        entities = []

        # Find all entity annotations
        for match in re.finditer(ENTITY_PATTERN, annotated_query):
            entity_type = match.group(1)
            entity_value = match.group(2)
            start_pos = match.start()
            end_pos = match.end()

            entity = Entity(
                type=entity_type,
                value=entity_value,
                start_pos=start_pos,
                end_pos=end_pos
            )
            entities.append(entity)

            logger.debug(f"Extracted entity: {entity_type}='{entity_value}'")

        return entities

    def _extract_annotated_query(self, raw_response: str) -> str:
        """Try to extract annotated query from raw LLM response"""
        # Look for patterns that might indicate the annotated query
        patterns = [
            r'"annotated_query":\s*"([^"]+)"',
            r'Output:\s*(.+)',
            r'Annotated:\s*(.+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, raw_response, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # If no patterns found, return empty string
        return ""
