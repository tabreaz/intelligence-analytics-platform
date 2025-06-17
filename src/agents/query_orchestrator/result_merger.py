# src/agents/query_orchestrator/result_merger.py
"""
Result merger and deduplication utilities for Query Orchestrator
"""
from typing import Dict, Any, List

from ...core.logger import get_logger

logger = get_logger(__name__)


class ResultMerger:
    """Handles merging and deduplication of agent results"""

    @staticmethod
    def merge_filter_results(filter_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge filter results from multiple agents and deduplicate
        
        Args:
            filter_results: Dictionary of agent_name -> filter results
            
        Returns:
            Merged and deduplicated filters
        """
        merged = {
            "inclusions": {},
            "exclusions": {},
            "time_constraints": {},
            "location_constraints": {},
            "ambiguities": [],
            "validation_warnings": []
        }

        # Collect all inclusions
        for agent_name, result in filter_results.items():
            if isinstance(result, dict):
                # Handle inclusions
                inclusions = result.get('inclusions', {})
                if isinstance(inclusions, dict):
                    merged['inclusions'] = ResultMerger._merge_dictionaries(
                        merged['inclusions'],
                        inclusions
                    )

                # Handle exclusions
                exclusions = result.get('exclusions', {})
                if isinstance(exclusions, dict):
                    merged['exclusions'] = ResultMerger._merge_dictionaries(
                        merged['exclusions'],
                        exclusions
                    )

                # Handle time-specific fields
                if agent_name == "time_parser":
                    for key in ['start_time', 'end_time', 'time_range', 'relative_time']:
                        if key in result:
                            merged['time_constraints'][key] = result[key]

                # Handle location-specific fields
                if agent_name == "location_extractor":
                    for key in ['locations', 'geohashes', 'radius']:
                        if key in result:
                            merged['location_constraints'][key] = result[key]

                # Collect ambiguities
                ambiguities = result.get('ambiguities', [])
                if isinstance(ambiguities, list):
                    merged['ambiguities'].extend(ambiguities)

                # Collect warnings
                warnings = result.get('validation_warnings', [])
                if isinstance(warnings, list):
                    merged['validation_warnings'].extend(warnings)

        # Deduplicate ambiguities
        merged['ambiguities'] = ResultMerger._deduplicate_ambiguities(merged['ambiguities'])

        # Deduplicate warnings
        merged['validation_warnings'] = list(set(merged['validation_warnings']))

        return merged

    @staticmethod
    def _merge_dictionaries(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two dictionaries, handling conflicts intelligently
        """
        result = dict1.copy()

        for key, value in dict2.items():
            if key not in result:
                result[key] = value
            else:
                # Handle conflicts based on data type
                existing_value = result[key]

                # If both are lists, combine and deduplicate
                if isinstance(existing_value, list) and isinstance(value, list):
                    combined = existing_value + value
                    # Deduplicate while preserving order
                    seen = set()
                    result[key] = []
                    for item in combined:
                        if isinstance(item, (str, int, float)):
                            if item not in seen:
                                seen.add(item)
                                result[key].append(item)
                        else:
                            # For complex types, just append
                            result[key].append(item)

                # If both are dicts with operator/value structure
                elif (isinstance(existing_value, dict) and isinstance(value, dict) and
                      'operator' in existing_value and 'operator' in value):
                    # Keep the more restrictive condition
                    # This is a simplified logic - might need enhancement
                    logger.warning(f"Conflicting conditions for field '{key}', keeping first: {existing_value}")

                # If types don't match or other cases
                else:
                    logger.warning(f"Type mismatch for field '{key}': {type(existing_value)} vs {type(value)}")
                    # Keep the existing value

        return result

    @staticmethod
    def _deduplicate_ambiguities(ambiguities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate ambiguities based on parameter and issue"""
        seen = set()
        unique_ambiguities = []

        for ambiguity in ambiguities:
            # Create a key from parameter and issue
            key = (
                ambiguity.get('parameter', ''),
                ambiguity.get('issue', '')
            )

            if key not in seen and key != ('', ''):
                seen.add(key)
                unique_ambiguities.append(ambiguity)

        return unique_ambiguities

    @staticmethod
    def merge_entities(entity_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge and deduplicate entity results
        """
        seen_entities = set()
        unique_entities = []

        for entity in entity_results:
            # Create a unique key for the entity
            key = (
                entity.get('type', ''),
                entity.get('value', ''),
                entity.get('start_pos', -1),
                entity.get('end_pos', -1)
            )

            if key not in seen_entities:
                seen_entities.add(key)
                unique_entities.append(entity)

        return unique_entities
