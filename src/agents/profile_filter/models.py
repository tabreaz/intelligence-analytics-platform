# src/agents/profile_filter/models.py
"""
Data models for Profile Filter Agent
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class ProfileFilterResult:
    """Result model for profile filter agent"""
    # Core extraction fields - using filter_tree format
    filter_tree: Dict[str, Any] = field(default_factory=dict)
    exclusions: Dict[str, Any] = field(default_factory=dict)  # Optional exclusions tree
    ambiguities: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    confidence: float = 0.0
    extraction_method: str = "llm"
    raw_extractions: Dict[str, Any] = field(default_factory=dict)
    validation_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response"""
        result = {
            "filter_tree": self.filter_tree,
            "exclusions": self.exclusions,
            "ambiguities": self.ambiguities,
            "confidence": self.confidence,
            "extraction_method": self.extraction_method,
            "validation_warnings": self.validation_warnings
        }

        # Include reasoning if available
        if self.raw_extractions.get('reasoning'):
            result['reasoning'] = self.raw_extractions['reasoning']

        return result


# Field data type definitions for operator validation
# Based on telecom_db.phone_imsi_uid_latest schema
FIELD_DATA_TYPES = {
    # Identity fields
    "imsi": "string",
    "phone_no": "string",
    "uid": "string",
    "eid": "array",  # Array(String)

    # Demographics
    "fullname_en": "string",
    "gender_en": "string",  # Enum but treat as string for validation
    "date_of_birth": "date",
    "age": "number",
    "age_group": "string",  # Enum but treat as string
    "marital_status_en": "string",  # Enum but treat as string

    # Nationality & Residency
    "nationality_code": "string",  # FixedString(3)
    "nationality_name_en": "string",
    "previous_nationality_code": "string",  # Nullable(FixedString(3))
    "previous_nationality_en": "string",  # Nullable(String)
    "residency_status": "string",  # Enum but treat as string
    "dwell_duration_tag": "string",  # Nullable(Enum)

    # Travel
    "last_travelled_country_code": "string",  # FixedString(3)
    "travelled_country_codes": "array",  # Array(FixedString(3))
    "communicated_country_codes": "array",  # Array(FixedString(3))

    # Location
    "home_city": "string",
    "home_location": "string",  # Nullable(String)
    "work_location": "string",  # Nullable(String)

    # Work & Sponsorship
    "latest_sponsor_name_en": "string",  # Nullable(String)
    "latest_job_title_en": "string",  # Nullable(String)

    # Criminal & Risk
    "has_investigation_case": "boolean",
    "has_crime_case": "boolean",
    "is_in_prison": "boolean",
    "crime_categories_en": "array",  # Array(String)
    "crime_sub_categories_en": "array",  # Array(String)
    "is_diplomat": "boolean",

    # Risk Scores
    "drug_addict_score": "number",  # Float32
    "drug_dealing_score": "number",  # Float32
    "murder_score": "number",  # Float32
    "risk_score": "number",  # Float32

    # Risk Rules
    "drug_addict_rules": "array",  # Array(String)
    "drug_dealing_rules": "array",  # Array(String)
    "murder_rules": "array",  # Array(String)
    "risk_rules": "array",  # Array(String)

    # Lifestyle
    "applications_used": "array",  # Array(String)
    "driving_license_type": "array",  # Array(String)
}


def get_field_data_type(field_name: str) -> Optional[str]:
    """Get the data type for a field"""
    return FIELD_DATA_TYPES.get(field_name)


def field_exists(field_name: str) -> bool:
    """Check if a field exists in the schema"""
    return field_name in FIELD_DATA_TYPES


def validate_field_existence(filter_tree: Dict[str, Any], exclusions: Dict[str, Any] = None) -> List[str]:
    """
    Simple validation to check if fields exist in the schema.
    Works with the new filter_tree format.
    """
    warnings = []

    def validate_filter_conditions(conditions: Any, context: str = "filter_tree"):
        """Recursively validate filter conditions"""
        if isinstance(conditions, dict):
            # Handle logical operators
            for key, value in conditions.items():
                if key in ["AND", "OR", "NOT"]:
                    if isinstance(value, list):
                        for condition in value:
                            validate_filter_conditions(condition, context)
                    else:
                        validate_filter_conditions(value, context)
                elif key == "field":
                    # This is a filter condition
                    if not field_exists(value):
                        warnings.append(f"Unknown field in {context}: {value}")
                
    # Check filter_tree
    if filter_tree:
        validate_filter_conditions(filter_tree, "filter_tree")
    
    # Check exclusions if provided
    if exclusions:
        validate_filter_conditions(exclusions, "exclusions")

    return warnings
