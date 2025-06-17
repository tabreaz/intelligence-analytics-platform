# src/agents/time_parser/constants.py
"""
Constants for Time Parser Agent
"""

# Time indicators for quick validation
TIME_INDICATORS = [
    'yesterday', 'today', 'tomorrow', 'last', 'next', 'ago',
    'morning', 'afternoon', 'evening', 'night', 'weekend',
    'hour', 'day', 'week', 'month', 'year',
    'AM', 'PM', 'between', 'from', 'to', 'since', 'until',
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
    'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
    'january', 'february', 'march', 'april', 'june', 'july', 'august',
    'september', 'october', 'november', 'december',
    'except', 'excluding', 'exclude', 'not', 'without', 'skip', 'omit',  # Exclusion indicators
    'only', 'just', 'specifically', 'every', 'each'  # Specific selection indicators
]

# Exclusion indicators
EXCLUSION_INDICATORS = [
    'except', 'excluding', 'exclude', 'not including', 'without',
    'skip', 'omit', 'minus', 'but not', 'apart from'
]

# Specific selection indicators
SPECIFIC_INDICATORS = [
    'only', 'just', 'specifically', 'exactly', 'strictly',
    'every', 'each', 'all'
]

# Hour mappings for natural language
HOUR_MAPPINGS = {
    'morning': (6, 12),
    'afternoon': (12, 18),
    'evening': (18, 22),
    'night': (22, 6),
    'dawn': (4, 6),
    'dusk': (18, 20),
    'noon': (12, 13),
    'midnight': (0, 1)
}

# Day of week mappings
DAY_MAPPINGS = {
    'weekday': ['monday', 'tuesday', 'wednesday', 'thursday', 'friday'],
    'weekend': ['saturday', 'sunday'],
    'workday': ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']
}

# Relative time units in days
RELATIVE_TIME_UNITS = {
    'day': 1,
    'week': 7,
    'fortnight': 14,
    'month': 30,  # Approximate
    'quarter': 90,
    'year': 365
}

# Default settings
DEFAULT_TIME_RANGE_DAYS = 2  # Default to last 2 days if no time specified
MAX_TIME_RANGE_DAYS = 365  # Maximum allowed range
DEFAULT_TIMEZONE = "UTC"

# Regex patterns
DATE_PATTERNS = [
    r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
    r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
    r'\d{2}-\d{2}-\d{4}',  # DD-MM-YYYY
    r'\d{1,2}[-/]\d{1,2}',  # M/D or MM/DD
    r'\d{4}',  # Year only
]

TIME_PATTERNS = [
    r'\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?',  # Time with optional seconds and AM/PM
    r'\d{1,2}\s*[AP]M',  # Hour with AM/PM
]
