# src/agents/risk_filter/validator.py
"""
Validator for Risk Filter Agent - validates against database schema
"""
from typing import List, Tuple

from ...core.logger import get_logger

logger = get_logger(__name__)


class RiskFilterValidator:
    """Validates risk filter fields against database schema"""

    # Define valid risk score fields from schema
    VALID_RISK_SCORE_FIELDS = {
        'risk_score',
        'drug_dealing_score',
        'drug_addict_score',
        'murder_score'
    }

    # Define valid boolean flag fields from schema
    VALID_FLAG_FIELDS = {
        'has_crime_case',
        'has_investigation_case',
        'is_in_prison',
        'is_diplomat'
    }

    # Define valid crime category field
    CRIME_CATEGORY_FIELD = 'crime_categories_en'

    @classmethod
    def validate_risk_scores(cls, risk_scores: dict) -> Tuple[dict, List[str]]:
        """Validate risk score fields against schema
        
        Args:
            risk_scores: Dictionary of risk score filters
            
        Returns:
            Tuple of (validated_scores, warnings)
        """
        validated = {}
        warnings = []

        for field, score_filter in risk_scores.items():
            if field not in cls.VALID_RISK_SCORE_FIELDS:
                warnings.append(f"Invalid risk score field: {field}")
                logger.warning(f"Skipping invalid risk score field: {field}")
                continue

            # Validate score values are between 0 and 1
            if hasattr(score_filter, 'value'):
                if not 0 <= score_filter.value <= 1:
                    warnings.append(f"{field} value {score_filter.value} not in range 0-1")
                    continue

                # Check value2 for BETWEEN operator
                if hasattr(score_filter, 'value2') and score_filter.value2 is not None:
                    if not 0 <= score_filter.value2 <= 1:
                        warnings.append(f"{field} value2 {score_filter.value2} not in range 0-1")
                        continue

            validated[field] = score_filter

        return validated, warnings

    @classmethod
    def validate_flags(cls, flags: dict) -> Tuple[dict, List[str]]:
        """Validate flag fields against schema
        
        Args:
            flags: Dictionary of boolean flags
            
        Returns:
            Tuple of (validated_flags, warnings)
        """
        validated = {}
        warnings = []

        for field, value in flags.items():
            if field not in cls.VALID_FLAG_FIELDS:
                warnings.append(f"Invalid flag field: {field}")
                logger.warning(f"Skipping invalid flag field: {field}")
                continue

            # Ensure value is boolean
            if not isinstance(value, bool):
                warnings.append(f"{field} value must be boolean, got {type(value).__name__}")
                continue

            validated[field] = value

        return validated, warnings

    @classmethod
    def get_schema_info(cls) -> dict:
        """Get information about valid schema fields
        
        Returns:
            Dictionary with schema information
        """
        return {
            "risk_score_fields": list(cls.VALID_RISK_SCORE_FIELDS),
            "flag_fields": list(cls.VALID_FLAG_FIELDS),
            "crime_category_field": cls.CRIME_CATEGORY_FIELD,
            "field_types": {
                "risk_scores": "Float32 (0.0-1.0)",
                "flags": "Bool",
                "crime_categories": "Array(String)"
            }
        }

    @classmethod
    def validate_and_clean(cls, risk_scores: dict, flags: dict,
                           exclude_scores: dict, exclude_flags: dict) -> Tuple[dict, List[str]]:
        """Validate and clean all risk filter fields
        
        Args:
            risk_scores: Risk score filters
            flags: Boolean flag filters
            exclude_scores: Excluded risk scores
            exclude_flags: Excluded flags
            
        Returns:
            Tuple of (cleaned_data, all_warnings)
        """
        all_warnings = []

        # Validate inclusions
        validated_scores, score_warnings = cls.validate_risk_scores(risk_scores)
        all_warnings.extend(score_warnings)

        validated_flags, flag_warnings = cls.validate_flags(flags)
        all_warnings.extend(flag_warnings)

        # Validate exclusions
        validated_exclude_scores, exclude_score_warnings = cls.validate_risk_scores(exclude_scores)
        all_warnings.extend(exclude_score_warnings)

        validated_exclude_flags, exclude_flag_warnings = cls.validate_flags(exclude_flags)
        all_warnings.extend(exclude_flag_warnings)

        cleaned_data = {
            "risk_scores": validated_scores,
            "flags": validated_flags,
            "exclude_scores": validated_exclude_scores,
            "exclude_flags": validated_exclude_flags
        }

        return cleaned_data, all_warnings
