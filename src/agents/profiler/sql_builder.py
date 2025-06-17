# src/agents/profiler/sql_builder.py
"""
SQL Builder for Profiler Agent - Generates ClickHouse SQL scripts
Based on profile and risk filters extracted by profile_filter and risk_filter agents
"""
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json

from .models import QueryType, SQLGenerationMethod
from ...core.logger import get_logger

logger = get_logger(__name__)


class ProfilerSQLBuilder:
    """
    Builds SQL scripts for profile queries based on filters
    
    This builder understands the output format of profile_filter and risk_filter agents:
    - Profile filters: inclusions/exclusions with field mappings
    - Risk filters: risk_scores, flags, crime_categories sections
    """
    
    def __init__(self):
        self.base_table = "telecom_db.phone_imsi_uid_latest"
        
        # Define field groups for different query types
        self.basic_fields = [
            "imsi", "phone_no", "fullname_en", "nationality_code", 
            "gender_en", "age", "home_city"
        ]
        
        self.risk_fields = [
            "risk_score", "drug_dealing_score", "murder_score",
            "has_crime_case", "is_in_prison", "crime_categories_en"
        ]
        
        self.detail_fields = [
            "uid", "eid", "residency_status", "age_group", "marital_status_en",
            "latest_job_title_en", "latest_sponsor_name_en", "is_diplomat"
        ]
        
    def build_sql_script(self, 
                        filters: Dict[str, Any],
                        query_type: QueryType,
                        context_aware_query: str,
                        original_query: str,
                        limit: int = 1000) -> Tuple[str, SQLGenerationMethod, List[str]]:
        """
        Build SQL script based on filters from profile_filter and risk_filter agents
        
        Returns:
            - SQL script
            - Generation method used
            - List of validation warnings
        """
        warnings = []
        
        # Extract profile and risk filters
        profile_filters = filters.get('profile', {})
        risk_filters = filters.get('risk', {})
        
        # Select appropriate fields based on query type and filters
        select_fields = self._get_fields_for_query_type(query_type, profile_filters, risk_filters)
        
        # Build WHERE conditions
        where_conditions = []
        
        # Add profile conditions from filter_tree
        if profile_filters and profile_filters.get('filter_tree'):
            profile_condition = self._build_conditions_from_filter_tree(profile_filters['filter_tree'])
            if profile_condition:
                where_conditions.append(profile_condition)
        
        # Add profile exclusions
        if profile_filters and profile_filters.get('exclusions'):
            exclusion_condition = self._build_conditions_from_filter_tree(profile_filters['exclusions'], is_exclusion=True)
            if exclusion_condition:
                where_conditions.append(exclusion_condition)
        
        # Add risk conditions from filter_tree
        if risk_filters and risk_filters.get('filter_tree'):
            risk_condition = self._build_conditions_from_filter_tree(risk_filters['filter_tree'])
            if risk_condition:
                where_conditions.append(risk_condition)
                
        # Add risk exclusions
        if risk_filters and risk_filters.get('exclusions'):
            exclusion_condition = self._build_conditions_from_filter_tree(risk_filters['exclusions'], is_exclusion=True)
            if exclusion_condition:
                where_conditions.append(exclusion_condition)
        
        # Always have at least one condition
        if not where_conditions:
            where_conditions.append("1=1")
            warnings.append("No filters found, returning all records")
        
        # Build ORDER BY clause
        order_by = self._build_order_by(query_type, risk_filters)
        
        # Construct the SQL
        sql = self._construct_sql(
            select_fields=select_fields,
            where_conditions=where_conditions,
            order_by=order_by,
            limit=limit
        )
        
        # Format with metadata
        formatted_sql = self._format_sql_script(
            sql=sql,
            context_query=context_aware_query,
            original_query=original_query,
            filters=filters,
            query_type=query_type
        )
        
        return formatted_sql, SQLGenerationMethod.TEMPLATE, warnings
    
    def _get_fields_for_query_type(self, 
                                  query_type: QueryType, 
                                  profile_filters: Dict[str, Any],
                                  risk_filters: Dict[str, Any]) -> List[str]:
        """Select appropriate fields based on query type and filters present"""
        fields = self.basic_fields.copy()
        
        # Add risk fields if any risk filters present
        if risk_filters.get('inclusions') or risk_filters.get('exclusions'):
            fields.extend(self.risk_fields)
        
        # Add detail fields for certain query types
        if query_type in [QueryType.DETAIL_RECORDS, QueryType.COMPLEX_MULTI_FILTER]:
            fields.extend(self.detail_fields)
        
        # Add specific fields based on profile filter inclusions
        profile_inclusions = profile_filters.get('inclusions', {})
        
        # Travel fields
        if 'travelled_country_codes' in profile_inclusions:
            fields.extend(['travelled_country_codes', 'last_travelled_country_code'])
        
        # Communication fields  
        if 'communicated_country_codes' in profile_inclusions:
            fields.append('communicated_country_codes')
            
        # Application fields
        if 'applications_used' in profile_inclusions:
            fields.append('applications_used')
            
        # License fields
        if 'driving_license_type' in profile_inclusions:
            fields.append('driving_license_type')
        
        # Remove duplicates while preserving order
        seen = set()
        unique_fields = []
        for field in fields:
            if field not in seen:
                seen.add(field)
                unique_fields.append(field)
        
        return unique_fields
    
    def _build_conditions_from_filter_tree(self, filter_tree: Dict[str, Any], is_exclusion: bool = False) -> str:
        """
        Build SQL conditions from filter_tree structure
        
        Args:
            filter_tree: Filter tree with AND/OR structure
            is_exclusion: If True, negate the entire condition
            
        Returns:
            SQL condition string
        """
        if 'AND' in filter_tree:
            conditions = []
            for condition in filter_tree['AND']:
                if isinstance(condition, dict):
                    if 'OR' in condition or 'AND' in condition:
                        # Nested logical operator
                        nested = self._build_conditions_from_filter_tree(condition)
                        if nested:
                            conditions.append(f"({nested})")
                    else:
                        # Single condition
                        cond = self._build_single_condition(condition)
                        if cond:
                            conditions.append(cond)
            
            result = ' AND '.join(conditions)
            return f"NOT ({result})" if is_exclusion and result else result
            
        elif 'OR' in filter_tree:
            conditions = []
            for condition in filter_tree['OR']:
                if isinstance(condition, dict):
                    if 'AND' in condition or 'OR' in condition:
                        # Nested logical operator
                        nested = self._build_conditions_from_filter_tree(condition)
                        if nested:
                            conditions.append(f"({nested})")
                    else:
                        # Single condition
                        cond = self._build_single_condition(condition)
                        if cond:
                            conditions.append(cond)
            
            result = ' OR '.join(conditions)
            return f"NOT ({result})" if is_exclusion and result else result
        
        else:
            # Single condition at root level
            cond = self._build_single_condition(filter_tree)
            return f"NOT ({cond})" if is_exclusion and cond else cond
    
    def _build_single_condition(self, condition: Dict[str, Any]) -> str:
        """Build a single SQL condition from filter format"""
        field = condition.get('field')
        operator = condition.get('operator', '=')
        value = condition.get('value')
        
        if not field:
            return ""
            
        # Handle different operators
        if operator.upper() == 'IN':
            # Special handling for array fields
            if field in ['crime_categories_en', 'travelled_country_codes', 'communicated_country_codes', 
                        'applications_used', 'driving_license_type', 'eid']:
                # Use has() for array fields
                if isinstance(value, list):
                    if len(value) == 1:
                        return f"has({field}, '{value[0]}')"
                    else:
                        or_conditions = [f"has({field}, '{v}')" for v in value]
                        return f"({' OR '.join(or_conditions)})"
                else:
                    return f"has({field}, '{value}')"
            else:
                # Regular IN operator for non-array fields
                if isinstance(value, list):
                    if len(value) == 1:
                        return f"{field} = '{value[0]}'"
                    else:
                        value_list = "', '".join(str(v) for v in value)
                        return f"{field} IN ('{value_list}')"
                else:
                    return f"{field} = '{value}'"
                
        elif operator.upper() == 'CONTAINS':
            return f"has({field}, '{value}')"
            
        elif operator.upper() == 'CONTAINS_ANY':
            if isinstance(value, list):
                or_conditions = [f"has({field}, '{v}')" for v in value]
                return f"({' OR '.join(or_conditions)})"
            else:
                return f"has({field}, '{value}')"
                
        elif operator.upper() == 'CONTAINS_ALL':
            if isinstance(value, list):
                and_conditions = [f"has({field}, '{v}')" for v in value]
                return f"({' AND '.join(and_conditions)})"
            else:
                return f"has({field}, '{value}')"
                
        elif operator.upper() in ['EQ', '=']:
            # Handle boolean values
            if isinstance(value, bool):
                return f"{field} = {1 if value else 0}"
            else:
                return f"{field} = '{value}'"
                
        elif operator.upper() == 'BETWEEN':
            value2 = condition.get('value2')
            if value2 is not None:
                return f"{field} BETWEEN {value} AND {value2}"
            else:
                return f"{field} >= {value}"
                
        elif operator.upper() == 'IS_NULL':
            return f"{field} IS NULL"
            
        elif operator.upper() == 'IS_NOT_NULL':
            return f"{field} IS NOT NULL"
            
        else:
            # Numeric operators (>, <, >=, <=, !=)
            if isinstance(value, (int, float)):
                return f"{field} {operator} {value}"
            else:
                return f"{field} {operator} '{value}'"
    
    def _build_profile_conditions(self, profile_filters: Dict[str, Any]) -> List[str]:
        """
        Build WHERE conditions from profile_filter agent output
        
        Profile filter format:
        {
            "inclusions": {
                "nationality_code": ["IND", "PAK"],
                "age": {"operator": "<", "value": 35},
                "gender_en": "Male"
            },
            "exclusions": {
                "home_city": ["Dubai"]
            }
        }
        """
        conditions = []
        
        # Process inclusions
        inclusions = profile_filters.get('inclusions', {})
        for field, value in inclusions.items():
            if not value:
                continue
                
            # Handle array fields (travelled_country_codes, applications_used, etc.)
            if field in ['travelled_country_codes', 'communicated_country_codes', 
                        'applications_used', 'driving_license_type', 'eid']:
                if isinstance(value, list):
                    for item in value:
                        conditions.append(f"has({field}, '{item}')")
                else:
                    conditions.append(f"has({field}, '{value}')")
                    
            # Handle age with operators
            elif field == 'age' and isinstance(value, dict):
                operator = value.get('operator', '=')
                val = value.get('value')
                if val is not None:
                    conditions.append(f"age {operator} {val}")
                    
            # Handle list values (nationalities, etc.)
            elif isinstance(value, list):
                if len(value) == 1:
                    conditions.append(f"{field} = '{value[0]}'")
                else:
                    value_list = "', '".join(str(v) for v in value)
                    conditions.append(f"{field} IN ('{value_list}')")
                    
            # Handle single string/enum values
            else:
                conditions.append(f"{field} = '{value}'")
        
        # Process exclusions
        exclusions = profile_filters.get('exclusions', {})
        for field, value in exclusions.items():
            if not value:
                continue
                
            if isinstance(value, list):
                value_list = "', '".join(str(v) for v in value)
                conditions.append(f"{field} NOT IN ('{value_list}')")
            else:
                conditions.append(f"{field} != '{value}'")
        
        return conditions
    
    def _build_risk_conditions(self, risk_filters: Dict[str, Any]) -> List[str]:
        """
        Build WHERE conditions from risk_filter agent output
        
        Risk filter format:
        {
            "inclusions": {
                "risk_scores": {
                    "risk_score": {"operator": ">", "value": 0.7}
                },
                "flags": {
                    "has_crime_case": true
                },
                "crime_categories": {
                    "categories": ["violent crimes", "drug-related"]
                }
            },
            "exclusions": {...}
        }
        """
        conditions = []
        
        inclusions = risk_filters.get('inclusions', {})
        
        # Process risk scores
        risk_scores = inclusions.get('risk_scores', {}) or inclusions.get('risk_score', {})
        
        # Handle both nested and flat structures
        if isinstance(risk_scores, dict):
            # Check if it's a single score or multiple
            if 'operator' in risk_scores:
                # Flat structure: {"operator": ">", "value": 0.7}
                operator = risk_scores.get('operator', '>=')
                value = risk_scores.get('value', 0.5)
                conditions.append(f"risk_score {operator} {value}")
            else:
                # Nested structure: {"risk_score": {"operator": ">", "value": 0.7}}
                for score_field, score_def in risk_scores.items():
                    if isinstance(score_def, dict):
                        operator = score_def.get('operator', '>=')
                        value = score_def.get('value', 0.5)
                        
                        # Handle BETWEEN operator
                        if operator.upper() == 'BETWEEN' and 'value2' in score_def:
                            conditions.append(f"{score_field} BETWEEN {value} AND {score_def['value2']}")
                        else:
                            conditions.append(f"{score_field} {operator} {value}")
        
        # Process flags
        flags = inclusions.get('flags', {})
        for flag, value in flags.items():
            if value is True:
                conditions.append(f"{flag} = 1")
            elif value is False:
                conditions.append(f"{flag} = 0")
        
        # Process crime categories
        crime_cats = inclusions.get('crime_categories', {})
        if crime_cats and crime_cats.get('categories'):
            categories = crime_cats['categories']
            if isinstance(categories, list):
                for category in categories:
                    conditions.append(f"has(crime_categories_en, '{category}')")
        
        # Process exclusions
        exclusions = risk_filters.get('exclusions', {})
        
        # Exclusion flags
        excl_flags = exclusions.get('flags', {})
        for flag, value in excl_flags.items():
            if value is True:
                conditions.append(f"{flag} = 0")  # Exclude true means include false
        
        # Exclusion crime categories
        excl_crime = exclusions.get('crime_categories', {})
        if excl_crime and excl_crime.get('categories'):
            categories = excl_crime['categories']
            if isinstance(categories, list):
                for category in categories:
                    conditions.append(f"NOT has(crime_categories_en, '{category}')")
        
        return conditions
    
    def _build_order_by(self, query_type: QueryType, risk_filters: Dict[str, Any]) -> str:
        """Build ORDER BY clause based on query type and filters"""
        # If risk filters present, order by risk score
        if risk_filters.get('inclusions'):
            return "risk_score DESC, has_crime_case DESC"
        else:
            return "fullname_en"  # Default ordering
    
    def _construct_sql(self, 
                      select_fields: List[str],
                      where_conditions: List[str],
                      order_by: str,
                      limit: int) -> str:
        """Construct the final SQL query"""
        # Format fields for readability
        formatted_fields = self._format_select_fields(select_fields)
        
        sql = f"""SELECT 
{formatted_fields}
FROM {self.base_table}
WHERE {' AND '.join(where_conditions)}
ORDER BY {order_by}
LIMIT {limit}"""
        
        return sql
    
    def _format_select_fields(self, fields: List[str]) -> str:
        """Format SELECT fields for readability"""
        if len(fields) <= 5:
            return "    " + ", ".join(fields)
        else:
            # Group fields for better readability
            lines = []
            for i in range(0, len(fields), 4):
                group = fields[i:i+4]
                lines.append("    " + ", ".join(group))
            return ",\n".join(lines)
    
    def _format_sql_script(self, 
                          sql: str, 
                          context_query: str,
                          original_query: str,
                          filters: Dict[str, Any],
                          query_type: QueryType) -> str:
        """Format SQL with comments and metadata"""
        timestamp = datetime.now().isoformat()
        
        # Create filter summary
        filter_summary = []
        profile_filters = filters.get('profile', {})
        risk_filters = filters.get('risk', {})
        
        if profile_filters:
            prof_inc = len(profile_filters.get('inclusions', {}))
            prof_exc = len(profile_filters.get('exclusions', {}))
            filter_summary.append(f"Profile filters: {prof_inc} inclusions, {prof_exc} exclusions")
            
        if risk_filters:
            risk_inc = risk_filters.get('inclusions', {})
            risk_count = len(risk_inc.get('risk_scores', {})) + len(risk_inc.get('flags', {}))
            if risk_inc.get('crime_categories'):
                risk_count += 1
            filter_summary.append(f"Risk filters: {risk_count} conditions")
        
        formatted = f"""-- ============================================================================
-- Profiler Agent Generated SQL Script
-- ============================================================================
-- Generated at: {timestamp}
-- Query Type: {query_type.value}
-- Original Query: {original_query}
-- Context-Aware Query: {context_query}
-- 
-- Filter Summary:
-- {chr(10).join('-- ' + s for s in filter_summary)}
-- 
-- Table: {self.base_table}
-- ============================================================================

{sql}

-- ============================================================================
-- End of generated script
-- ============================================================================
"""
        return formatted