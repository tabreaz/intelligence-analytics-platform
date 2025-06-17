# src/agents/risk_filter/response_parser.py
"""
Response parser for Risk Filter Agent
"""
import json
from typing import Dict, Any, Optional

from .category_mapper import CategoryMapper
from .constants import MAX_RESPONSE_LOG_LENGTH
from .models import RiskFilterResult, ScoreFilter, OperatorType, CategoryFilter
from ...core.logger import get_logger

logger = get_logger(__name__)


class RiskFilterResponseParser:
    """Parses LLM responses for risk filter extraction"""

    def parse(self, response: str) -> RiskFilterResult:
        """Parse LLM response into RiskFilterResult
        
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

            # Process inclusions
            if 'inclusions' in data:
                self._process_inclusions(data['inclusions'], result)

            # Process exclusions
            if 'exclusions' in data:
                self._process_exclusions(data['exclusions'], result)

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

    def _process_inclusions(self, inclusions: Dict[str, Any], result: RiskFilterResult) -> None:
        """Process inclusion filters"""

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
        """Process exclusion filters"""

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
