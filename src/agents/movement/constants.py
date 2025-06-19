# src/agents/movement/constants.py
"""
Constants for Movement Analysis Agent
"""

# Maximum retry attempts for LLM calls
MAX_RETRIES = 2

# Default values
DEFAULT_RADIUS_METERS = 1000  # 1km default radius
DEFAULT_TIME_WINDOW_DAYS = 7  # 7 days default for single visits
DEFAULT_PATTERN_WINDOW_DAYS = 30  # 30 days for pattern analysis
DEFAULT_MINIMUM_OVERLAP_MINUTES = 30  # 30 minutes for co-presence
DEFAULT_MATCH_GRANULARITY = "geohash7"

# Time mappings (hours)
LOCATION_TIME_MAPPINGS = {
    "home": {"hours": [20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6], "name": "night hours"},
    "residence": {"hours": [20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6], "name": "night hours"},
    "work": {"hours": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], "days": ["monday", "tuesday", "wednesday", "thursday", "friday"]},
    "office": {"hours": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], "days": ["monday", "tuesday", "wednesday", "thursday", "friday"]},
    "mall": {"hours": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]},
    "shopping": {"hours": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]},
    "restaurant": {"hours": [12, 13, 14, 19, 20, 21, 22]},
    "dining": {"hours": [12, 13, 14, 19, 20, 21, 22]},
    "nightclub": {"hours": [22, 23, 0, 1, 2, 3, 4]},
    "bar": {"hours": [22, 23, 0, 1, 2, 3, 4]},
    "mosque": {"hours": [4, 5, 12, 13, 15, 16, 18, 19, 20]},  # Prayer times
    "prayer": {"hours": [4, 5, 12, 13, 15, 16, 18, 19, 20]},
    "gym": {"hours": [5, 6, 7, 8, 17, 18, 19, 20, 21]},
    "fitness": {"hours": [5, 6, 7, 8, 17, 18, 19, 20, 21]},
    "school": {"hours": [7, 8, 9, 10, 11, 12, 13, 14, 15, 16], "days": ["monday", "tuesday", "wednesday", "thursday", "friday"]},
    "university": {"hours": [7, 8, 9, 10, 11, 12, 13, 14, 15, 16], "days": ["monday", "tuesday", "wednesday", "thursday", "friday"]}
}

# Movement pattern mappings
MOVEMENT_PATTERNS = {
    "commuting": {
        "morning": {"hours": [6, 7, 8, 9]},
        "evening": {"hours": [16, 17, 18, 19]},
        "days": ["monday", "tuesday", "wednesday", "thursday", "friday"]
    },
    "visiting": {"hours": list(range(9, 21))},  # 9am to 9pm
    "frequenting": {"min_visits": 3, "window_days": 30},
    "passing_through": {"max_duration_minutes": 30},
    "staying": {"min_duration_minutes": 120},  # 2 hours
    "dwelling": {"min_duration_minutes": 120}
}

# Time period definitions
TIME_PERIODS = {
    "night_hours": [23, 0, 1, 2, 3, 4, 5],
    "social_hours": [18, 19, 20, 21, 22],
    "work_hours": [7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    "morning_commute": [6, 7, 8],
    "evening_commute": [16, 17, 18],
    "early_morning": [5, 6, 7, 8],
    "morning": [6, 7, 8, 9, 10, 11],
    "afternoon": [12, 13, 14, 15, 16],
    "evening": [17, 18, 19, 20, 21],
    "night": [22, 23, 0, 1, 2, 3, 4]
}

# Valid query types
VALID_QUERY_TYPES = [
    "single_location",
    "multi_location_and",
    "multi_location_or",
    "movement_pattern",
    "co_presence",
    "heatmap",
    "predictive_analysis",
    "anomaly_detection",
    "multi_modal_movement_analysis"
]

# Valid spatial methods
VALID_SPATIAL_METHODS = ["name", "coordinates", "area", "polygon"]

# Valid match granularities
VALID_MATCH_GRANULARITIES = ["geohash7", "geohash6", "geohash5", "coordinates"]

# Valid aggregation periods
VALID_AGGREGATION_PERIODS = ["hourly", "daily", "weekly", "monthly"]

# Valid days of week
VALID_DAYS_OF_WEEK = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

# Location name variations (for normalization)
LOCATION_ALIASES = {
    # Malls
    "dubai mall": ["the dubai mall", "dubai shopping mall"],
    "mall of emirates": ["mall of the emirates", "moe", "emirates mall"],
    "ibn battuta": ["ibn battuta mall", "ibn batuta"],
    
    # Areas
    "difc": ["dubai international financial centre", "financial centre"],
    "jbr": ["jumeirah beach residence", "jumeira beach residence"],
    "downtown": ["downtown dubai", "burj khalifa area"],
    "marina": ["dubai marina", "marina walk"],
    
    # Emirates
    "dubai": ["dxb", "dubai city"],
    "abu dhabi": ["abu dhabi city", "auh", "ad"],
    "sharjah": ["shj", "sharjah city"],
    "ajman": ["ajm", "ajman city"],
    "rak": ["ras al khaimah", "ras al-khaimah"],
    "fujairah": ["fuj", "fujairah city"],
    "uaq": ["umm al quwain", "umm al-quwain"]
}