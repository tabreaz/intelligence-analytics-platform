"""
ClickHouse-specific filter translation
"""
from typing import Dict, Any, List, Tuple
import logging

from ..base import FilterTranslator

logger = logging.getLogger(__name__)


class ClickHouseFilterTranslator(FilterTranslator):
    """Translates unified filter trees to ClickHouse SQL"""
    
    def _get_operators_map(self) -> Dict[str, str]:
        """ClickHouse operator mappings"""
        return {
            "=": "=",
            "!=": "!=",
            ">": ">",
            "<": "<",
            ">=": ">=",
            "<=": "<=",
            "IN": "IN",
            "NOT IN": "NOT IN",
            "BETWEEN": "BETWEEN",
            "LIKE": "LIKE",
            "NOT LIKE": "NOT LIKE",
            "IS NULL": "IS NULL",
            "IS NOT NULL": "IS NOT NULL",
            "LENGTH >": "LENGTH >",
            "LENGTH <": "LENGTH <",
            "LENGTH =": "LENGTH =",
            "LENGTH >=": "LENGTH >=",
            "LENGTH <=": "LENGTH <="
        }
    
    def _translate_comparison(self, node: Dict[str, Any]) -> Tuple[str, List[Any]]:
        """Translate comparison operators for ClickHouse"""
        field = node.get("field")
        operator = node.get("operator")
        value = node.get("value")
        
        if not field or not operator:
            raise ValueError(f"Invalid comparison node: missing field or operator in {node}")
        
        # Quote field name
        quoted_field = self.engine.quote_identifier(field)
        
        # Define field types for proper operator selection
        string_fields = ['latest_job_title_en', 'fullname_en', 'latest_sponsor_name_en']
        array_fields = [
            'eid', 'travelled_country_codes', 'communicated_country_codes',
            'crime_categories_en', 'crime_sub_categories_en', 'applications_used',
            'driving_license_type', 'drug_addict_rules', 'drug_dealing_rules',
            'murder_rules', 'risk_rules'
        ]
        
        # Handle special operators
        if operator == "CONTAINS":
            # For string fields, use position() or LIKE
            if field in string_fields:
                return f"position({quoted_field}, {self.engine.format_value(value)}) > 0", []
            # For array fields use has()
            return f"has({quoted_field}, {self.engine.format_value(value)})", []
        
        elif operator == "CONTAINS_ANY":
            # For array fields, use hasAny
            if field in array_fields:
                if isinstance(value, list):
                    formatted_values = ", ".join(self.engine.format_value(v) for v in value)
                    return f"hasAny({quoted_field}, [{formatted_values}])", []
                else:
                    return f"has({quoted_field}, {self.engine.format_value(value)})", []
            else:
                # For string fields, use OR of LIKE conditions
                if isinstance(value, list):
                    conditions = [f"position({quoted_field}, {self.engine.format_value(v)}) > 0" for v in value]
                    return f"({' OR '.join(conditions)})", []
                else:
                    return f"position({quoted_field}, {self.engine.format_value(value)}) > 0", []
        
        elif operator.startswith("LENGTH"):
            # Handle array length operations
            if field in array_fields:
                # Extract the actual operator (LENGTH >, LENGTH <, etc.)
                if operator == "LENGTH >":
                    return f"length({quoted_field}) > {self.engine.format_value(value)}", []
                elif operator == "LENGTH <":
                    return f"length({quoted_field}) < {self.engine.format_value(value)}", []
                elif operator == "LENGTH =":
                    return f"length({quoted_field}) = {self.engine.format_value(value)}", []
                elif operator == "LENGTH >=":
                    return f"length({quoted_field}) >= {self.engine.format_value(value)}", []
                elif operator == "LENGTH <=":
                    return f"length({quoted_field}) <= {self.engine.format_value(value)}", []
                else:
                    raise ValueError(f"Unknown LENGTH operator: {operator}")
            else:
                raise ValueError(f"LENGTH operator only valid for array fields, not {field}")
        
        elif operator == "CONTAINS_ALL":
            # For array fields, use hasAll
            if field in array_fields:
                if isinstance(value, list):
                    formatted_values = ", ".join(self.engine.format_value(v) for v in value)
                    return f"hasAll({quoted_field}, [{formatted_values}])", []
                else:
                    return f"has({quoted_field}, {self.engine.format_value(value)})", []
            else:
                # For string fields, use AND of position conditions
                if isinstance(value, list):
                    conditions = [f"position({quoted_field}, {self.engine.format_value(v)}) > 0" for v in value]
                    return f"({' AND '.join(conditions)})", []
                else:
                    return f"position({quoted_field}, {self.engine.format_value(value)}) > 0", []
        
        elif operator == "BETWEEN":
            # Handle BETWEEN for dates/numbers
            if isinstance(value, list) and len(value) == 2:
                start_val = self._format_date_value(value[0])
                end_val = self._format_date_value(value[1])
                return f"{quoted_field} BETWEEN {start_val} AND {end_val}", []
            else:
                raise ValueError(f"BETWEEN requires exactly 2 values, got: {value}")
        
        elif operator in self.operators_map:
            # Standard operators
            if operator in ["IN", "NOT IN"]:
                # Check if this is an array field
                if field in array_fields:
                    # For array fields, use hasAny/hasAll
                    if isinstance(value, list):
                        formatted_values = ", ".join(self.engine.format_value(v) for v in value)
                        if operator == "IN":
                            # Use hasAny for array contains any of the values
                            return f"hasAny({quoted_field}, [{formatted_values}])", []
                        else:  # NOT IN
                            # Use NOT hasAny for array doesn't contain any of the values
                            return f"NOT hasAny({quoted_field}, [{formatted_values}])", []
                    else:
                        # Single value
                        if operator == "IN":
                            return f"has({quoted_field}, {self.engine.format_value(value)})", []
                        else:  # NOT IN
                            return f"NOT has({quoted_field}, {self.engine.format_value(value)})", []
                else:
                    # Regular field (non-array)
                    if isinstance(value, list):
                        formatted_values = ", ".join(self.engine.format_value(v) for v in value)
                        return f"{quoted_field} {operator} ({formatted_values})", []
                    else:
                        # Single value - convert to IN syntax
                        return f"{quoted_field} {operator} ({self.engine.format_value(value)})", []
            else:
                # Simple comparison
                formatted_value = self._format_field_value(field, value)
                return f"{quoted_field} {operator} {formatted_value}", []
        
        else:
            raise ValueError(f"Unknown operator: {operator}")
    
    def _format_date_value(self, value: Any) -> str:
        """Format date values for ClickHouse"""
        if isinstance(value, str):
            # Check if it's already in date format
            if 'T' in value:  # ISO format
                # Convert to ClickHouse datetime
                return f"toDateTime('{value}')"
            else:
                # Assume it's a date
                return self.engine.format_value(value)
        return self.engine.format_value(value)
    
    def _format_field_value(self, field: str, value: Any) -> str:
        """Format value based on field type"""
        # Special handling for certain fields
        if field.endswith('_date'):
            return self._format_date_value(value)
        
        return self.engine.format_value(value)