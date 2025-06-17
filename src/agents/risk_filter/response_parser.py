# src/agents/risk_filter/response_parser_enhanced.py
"""
Enhanced Response parser for Risk Filter Agent that handles filter_tree format
"""
import json
from typing import Dict, Any, Optional, List, Union

from .category_mapper import CategoryMapper
from .constants import MAX_RESPONSE_LOG_LENGTH
from .models import RiskFilterResult, ScoreFilter, OperatorType, CategoryFilter
from ...core.logger import get_logger

logger = get_logger(__name__)


class RiskFilterResponseParser:
    """Parses LLM responses for risk filter extraction - supports both formats"""

    def parse(self, response: str) -> RiskFilterResult:
        """Parse LLM response into RiskFilterResult
        
        Supports both formats:
        1. Old format: { "inclusions": {...}, "exclusions": {...} }
        2. New format: { "filter_tree": {...}, "exclusions": {...} }
        
        Args:
            response: Raw LLM response string
            
        Returns:
            RiskFilterResult with extracted filters
        """
        result = RiskFilterResult()

        try:
            # Truncate response for logging
            log_response = response[:MAX_RESPONSE_LOG_LENGTH] + "..." if len(
                response) > MAX_RESPONSE_LOG_LENGTH else response
            logger.debug(f"Parsing risk filter response: {log_response}")

            # Parse JSON response
            data = json.loads(response)

            # Extract reasoning if available
            if 'reasoning' in data:
                result.raw_extractions['reasoning'] = data['reasoning']

            # Check which format we have
            if 'filter_tree' in data:
                # New format with filter_tree
                self._process_filter_tree(data.get('filter_tree', {}), result, is_exclusion=False)
                if 'exclusions' in data:
                    self._process_filter_tree(data.get('exclusions', {}), result, is_exclusion=True)
            else:
                # Old format with inclusions/exclusions
                if 'inclusions' in data:
                    self._process_inclusions(data['inclusions'], result)
                if 'exclusions' in data:
                    self._process_exclusions(data['exclusions'], result)

            # Extract confidence if provided
            if 'confidence' in data:
                result.confidence = float(data['confidence'])
            else:
                # Set confidence based on extraction success
                result.confidence = self._calculate_confidence(result)

            # Store raw extractions for debugging  
            result.raw_extractions['llm_response'] = data

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            result.validation_warnings.append(f"Invalid JSON response: {str(e)}")
            result.confidence = 0.0
        except Exception as e:
            logger.error(f"Error parsing risk filter response: {e}")
            result.validation_warnings.append(f"Parsing error: {str(e)}")
            result.confidence = 0.0

        return result

    def _process_filter_tree(self, tree: Union[Dict, List], result: RiskFilterResult, is_exclusion: bool = False) -> None:
        """Process filter tree structure recursively"""
        if not tree:
            return
            
        # Handle leaf nodes (actual filter conditions)
        if 'field' in tree and 'operator' in tree and 'value' in tree:
            self._process_filter_node(tree, result, is_exclusion)
            return
            
        # Handle logical operators
        if 'AND' in tree:
            for child in tree['AND']:
                self._process_filter_tree(child, result, is_exclusion)
        elif 'OR' in tree:
            # For OR conditions, we process all branches
            # Note: The current RiskFilterResult structure doesn't support complex OR logic,
            # so we just extract all conditions
            for child in tree['OR']:
                self._process_filter_tree(child, result, is_exclusion)
        else:
            # Direct filter object
            self._process_filter_node(tree, result, is_exclusion)

    def _process_filter_node(self, node: Dict[str, Any], result: RiskFilterResult, is_exclusion: bool) -> None:
        """Process a single filter node"""
        field = node.get('field', '')
        operator = node.get('operator', '=')
        value = node.get('value')
        
        # Determine field type and process accordingly
        if field in ['risk_score', 'drug_dealing_score', 'drug_addict_score', 'murder_score']:
            # Score filter
            # Handle BETWEEN operator where value might be a list [min, max]
            if operator == 'BETWEEN' and isinstance(value, list) and len(value) == 2:
                score_filter = ScoreFilter(
                    field=field,
                    operator=OperatorType(operator),
                    value=float(value[0]),
                    value2=float(value[1])
                )
            else:
                score_filter = ScoreFilter(
                    field=field,
                    operator=OperatorType(operator),
                    value=float(value),
                    value2=float(node.get('value2')) if 'value2' in node else None
                )
            
            if is_exclusion:
                result.exclude_scores[field] = score_filter
            else:
                result.risk_scores[field] = score_filter
                
        elif field in ['has_crime_case', 'has_investigation_case', 'is_in_prison', 'is_diplomat']:
            # Boolean flag
            if is_exclusion:
                result.exclude_flags[field] = bool(value)
            else:
                result.flags[field] = bool(value)
                
        elif field == 'crime_categories_en':
            # Crime categories
            if not result.crime_categories:
                result.crime_categories = CategoryFilter()
                
            categories = value if isinstance(value, list) else [value]
            
            if is_exclusion:
                result.crime_categories.exclude.extend(categories)
            else:
                result.crime_categories.include.extend(categories)

    def _process_inclusions(self, inclusions: Dict[str, Any], result: RiskFilterResult) -> None:
        """Process inclusion filters (old format)"""
        
        # Process risk scores
        if 'risk_scores' in inclusions and inclusions['risk_scores']:
            for field, score_data in inclusions['risk_scores'].items():
                score_filter = self._parse_score_filter(field, score_data)
                if score_filter:
                    result.risk_scores[field] = score_filter

        # Process flags
        if 'flags' in inclusions and inclusions['flags']:
            result.flags = inclusions['flags']

        # Process crime categories
        if 'crime_categories' in inclusions and inclusions['crime_categories']:
            result.crime_categories = self._parse_crime_categories(inclusions['crime_categories'])

    def _process_exclusions(self, exclusions: Dict[str, Any], result: RiskFilterResult) -> None:
        """Process exclusion filters (old format)"""
        
        # Process excluded risk scores
        if 'risk_scores' in exclusions and exclusions['risk_scores']:
            for field, score_data in exclusions['risk_scores'].items():
                score_filter = self._parse_score_filter(field, score_data)
                if score_filter:
                    result.exclude_scores[field] = score_filter

        # Process excluded flags
        if 'flags' in exclusions and exclusions['flags']:
            result.exclude_flags = exclusions['flags']

        # Process excluded crime categories
        if 'crime_categories' in exclusions and exclusions['crime_categories']:
            excluded_categories = self._parse_crime_categories(exclusions['crime_categories'])
            if excluded_categories:
                # Merge exclusions into main crime categories
                if not result.crime_categories:
                    result.crime_categories = CategoryFilter()
                result.crime_categories.exclude.extend(excluded_categories.include)

    def _parse_score_filter(self, field: str, score_data: Dict[str, Any]) -> Optional[ScoreFilter]:
        """Parse a score filter from the response"""
        try:
            operator = score_data.get('operator', '>')
            value = float(score_data.get('value', 0))
            value2 = float(score_data.get('value2')) if 'value2' in score_data else None

            # Validate operator
            try:
                operator_enum = OperatorType(operator)
            except ValueError:
                logger.warning(f"Invalid operator: {operator}")
                return None

            # Create score filter
            return ScoreFilter(
                field=field,
                operator=operator_enum,
                value=value,
                value2=value2
            )

        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse score filter for {field}: {e}")
            return None

    def _parse_crime_categories(self, category_data: Dict[str, Any]) -> Optional[CategoryFilter]:
        """Parse crime categories from the response"""
        try:
            categories = category_data.get('categories', [])
            severity = category_data.get('severity')

            # Keep categories as-is for now (no mapping to database values)
            # TODO: In future, implement fuzzy matching to database categories

            # If severity is specified without categories, we still need to get categories
            if severity and not categories:
                # Get all categories for the severity level
                categories = CategoryMapper.get_categories_by_severity(severity)

            if categories or severity:
                return CategoryFilter(
                    include=categories,
                    severity_filter=severity
                )

            return None

        except Exception as e:
            logger.warning(f"Failed to parse crime categories: {e}")
            return None

    def _calculate_confidence(self, result: RiskFilterResult) -> float:
        """Calculate confidence score based on extraction success"""
        
        # Start with base confidence
        confidence = 0.5

        # Increase confidence for each type of filter found
        if result.risk_scores:
            confidence += 0.2
        if result.flags:
            confidence += 0.2
        if result.crime_categories:
            confidence += 0.1

        # Decrease confidence for warnings
        if result.validation_warnings:
            confidence -= 0.1 * len(result.validation_warnings)

        # Ensure confidence is between 0 and 1
        return max(0.0, min(1.0, confidence))