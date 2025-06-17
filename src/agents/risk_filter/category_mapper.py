# src/agents/risk_filter/category_mapper.py
"""
Crime category mapping utilities
"""
from typing import List, Optional, Set

from .prompt import CRIME_CATEGORIES, SEVERITY_LEVELS


class CategoryMapper:
    """Maps crime category keywords to actual database values"""

    @staticmethod
    def get_categories_by_type(crime_type: str) -> List[str]:
        """Get all crime categories for a given type
        
        Args:
            crime_type: Type like 'drug_related', 'violent_crimes', etc.
            
        Returns:
            List of crime category values
        """
        if crime_type in CRIME_CATEGORIES:
            category_data = CRIME_CATEGORIES[crime_type]
            if isinstance(category_data, list):
                return category_data
            elif isinstance(category_data, dict) and 'all' in category_data:
                return category_data['all']
        return []

    @staticmethod
    def get_categories_by_severity(severity: str) -> List[str]:
        """Get all crime categories for a given severity level
        
        Args:
            severity: 'severe', 'moderate', or 'minor'
            
        Returns:
            List of crime category values
        """
        if severity in SEVERITY_LEVELS:
            return SEVERITY_LEVELS[severity]
        return []

    @staticmethod
    def expand_category_keywords(keywords: List[str]) -> Set[str]:
        """Expand category keywords to actual database values
        
        Args:
            keywords: List of keywords like ['drug-related', 'violent crimes']
            
        Returns:
            Set of actual crime category values
        """
        expanded = set()

        keyword_mapping = {
            'drug-related': 'drug_related',
            'drug related': 'drug_related',
            'drugs': 'drug_related',
            'violent crimes': 'violent_crimes',
            'violent': 'violent_crimes',
            'violence': 'violent_crimes',
            'financial crimes': 'financial_crimes',
            'financial': 'financial_crimes',
            'white collar': 'financial_crimes',
            'cyber crimes': 'cyber_crimes',
            'cyber': 'cyber_crimes',
            'property crimes': 'property_crimes',
            'property': 'property_crimes',
            'traffic': 'traffic_crimes',
            'traffic crimes': 'traffic_crimes',
            'human trafficking': 'human_crimes',
            'human crimes': 'human_crimes',
            'national security': 'national_security',
            'terrorism': 'national_security',
            'immigration': 'immigration_crimes',
            'immigration crimes': 'immigration_crimes',
            'organized crime': 'organized_crime',
            'organized': 'organized_crime'
        }

        for keyword in keywords:
            keyword_lower = keyword.lower().strip()

            # Check direct mapping
            if keyword_lower in keyword_mapping:
                crime_type = keyword_mapping[keyword_lower]
                categories = CategoryMapper.get_categories_by_type(crime_type)
                expanded.update(categories)
            # Check if it's a specific crime category
            elif keyword in SEVERITY_LEVELS.get('severe', []) + \
                    SEVERITY_LEVELS.get('moderate', []) + \
                    SEVERITY_LEVELS.get('minor', []):
                expanded.add(keyword)

        return expanded

    @staticmethod
    def filter_by_severity(categories: List[str], severity: Optional[str]) -> List[str]:
        """Filter crime categories by severity level
        
        Args:
            categories: List of crime categories
            severity: 'severe', 'moderate', 'minor', or None
            
        Returns:
            Filtered list of categories
        """
        if not severity or severity not in SEVERITY_LEVELS:
            return categories

        severity_crimes = set(SEVERITY_LEVELS[severity])
        return [cat for cat in categories if cat in severity_crimes]
