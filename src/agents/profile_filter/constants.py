# src/agents/profile_filter/constants.py
"""
Constants for Profile Filter Agent
"""
from pathlib import Path
from typing import Dict

import yaml

# Agent configuration
DEFAULT_CONFIDENCE = 0.8
MAX_RETRIES = 2
RETRY_DELAY = 1.0


# Load field aliases from config
def load_field_aliases() -> Dict[str, str]:
    """Load field aliases from configuration file"""
    config_path = Path(__file__).parent.parent.parent.parent / 'config' / 'field_aliases.yaml'

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Create reverse mapping: alias -> canonical field name
        alias_mapping = {}
        field_aliases = config.get('field_aliases', {})

        for canonical_field, aliases in field_aliases.items():
            # Map each alias to the canonical field name
            for alias in aliases:
                alias_mapping[alias] = canonical_field
            # Also map the canonical name to itself
            alias_mapping[canonical_field] = canonical_field

        return alias_mapping
    except Exception as e:
        # If config loading fails, return empty mapping
        print(f"Warning: Could not load field aliases config: {e}")
        return {}


# Load field aliases at module level
FIELD_ALIASES = load_field_aliases()

# EID format - stored without hyphens in database
EID_PREFIX = "784"
EID_PATTERN = r"^\d{15}$"  # 15 digits without hyphens

# Field type groups
IDENTITY_FIELDS = ["imsi", "phone_no", "uid", "eid"]
DEMOGRAPHIC_FIELDS = ["fullname_en", "gender_en", "date_of_birth", "age", "age_group", "marital_status_en"]
NATIONALITY_FIELDS = ["nationality_code", "nationality_name_en", "previous_nationality_code",
                      "previous_nationality_en", "residency_status", "dwell_duration_tag"]
WORK_FIELDS = ["latest_sponsor_name_en", "latest_job_title_en"]
TRAVEL_FIELDS = ["last_travelled_country_code", "travelled_country_codes", "communicated_country_codes"]
LOCATION_FIELDS = ["home_city", "home_location", "work_location"]
LIFESTYLE_FIELDS = ["applications_used", "driving_license_type"]
RISK_FIELDS = ["has_investigation_case", "has_crime_case", "is_in_prison", "crime_categories_en",
               "crime_sub_categories_en", "is_diplomat", "drug_addict_score", "drug_dealing_score",
               "murder_score", "risk_score"]

# Common age descriptors
AGE_DESCRIPTORS = {
    "young": {"operator": "<", "value": 35},
    "middle-aged": {"operator": "BETWEEN", "value": [35, 50]},
    "middle aged": {"operator": "BETWEEN", "value": [35, 50]},
    "elderly": {"operator": ">", "value": 60},
    "senior": {"operator": ">", "value": 60},
    "adult": {"operator": ">=", "value": 18},
    "minor": {"operator": "<", "value": 18},
    "teenager": {"operator": "BETWEEN", "value": [13, 19]},
    "teen": {"operator": "BETWEEN", "value": [13, 19]}
}
