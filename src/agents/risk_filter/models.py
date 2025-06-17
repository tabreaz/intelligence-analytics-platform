# src/agents/risk_filter/models.py
"""
Data models for Risk Filter Agent
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional


class OperatorType(str, Enum):
    """SQL operator types"""
    EQUALS = "="
    NOT_EQUALS = "!="
    GREATER_THAN = ">"
    GREATER_THAN_EQUALS = ">="
    LESS_THAN = "<"
    LESS_THAN_EQUALS = "<="
    BETWEEN = "BETWEEN"
    IN = "IN"
    NOT_IN = "NOT IN"
    CONTAINS = "CONTAINS"
    NOT_CONTAINS = "NOT CONTAINS"


@dataclass
class ScoreFilter:
    """Filter for numeric score fields"""
    field: str
    operator: OperatorType
    value: float
    value2: Optional[float] = None  # For BETWEEN operator


@dataclass
class FlagFilter:
    """Filter for boolean flag fields"""
    field: str
    value: bool


@dataclass
class CategoryFilter:
    """Filter for crime category arrays"""
    field_name: str = "crime_categories_en"
    include: List[str] = field(default_factory=list)
    exclude: List[str] = field(default_factory=list)
    severity_filter: Optional[str] = None  # "severe", "moderate", "minor"


@dataclass
class RiskFilterResult:
    """Complete risk filter extraction result"""
    # Main filters
    risk_scores: Dict[str, ScoreFilter] = field(default_factory=dict)
    flags: Dict[str, bool] = field(default_factory=dict)
    crime_categories: Optional[CategoryFilter] = None

    # Exclusions
    exclude_scores: Dict[str, ScoreFilter] = field(default_factory=dict)
    exclude_flags: Dict[str, bool] = field(default_factory=dict)

    # Metadata
    confidence: float = 0.0
    extraction_method: str = "llm"
    raw_extractions: Dict[str, Any] = field(default_factory=dict)
    validation_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            "risk_scores": {},
            "flags": self.flags,
            "crime_categories": None,
            "exclusions": {
                "risk_scores": {},
                "flags": self.exclude_flags
            },
            "confidence": self.confidence,
            "validation_warnings": self.validation_warnings,
            "extraction_method": self.extraction_method
        }

        # Include reasoning if available
        if self.raw_extractions.get('reasoning'):
            result['reasoning'] = self.raw_extractions['reasoning']

        # Convert score filters
        for field, score_filter in self.risk_scores.items():
            result["risk_scores"][field] = {
                "operator": score_filter.operator.value,
                "value": score_filter.value
            }
            if score_filter.value2 is not None:
                result["risk_scores"][field]["value2"] = score_filter.value2

        # Convert exclusion scores
        for field, score_filter in self.exclude_scores.items():
            result["exclusions"]["risk_scores"][field] = {
                "operator": score_filter.operator.value,
                "value": score_filter.value
            }
            if score_filter.value2 is not None:
                result["exclusions"]["risk_scores"][field]["value2"] = score_filter.value2

        # Convert crime categories
        if self.crime_categories:
            result["crime_categories"] = {
                "include": self.crime_categories.include,
                "exclude": self.crime_categories.exclude,
                "severity_filter": self.crime_categories.severity_filter
            }

        return result

    def has_filters(self) -> bool:
        """Check if any filters were extracted"""
        return bool(
            self.risk_scores or
            self.flags or
            self.crime_categories or
            self.exclude_scores or
            self.exclude_flags
        )
