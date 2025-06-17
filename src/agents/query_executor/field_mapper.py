"""
Field mapping for query generation to handle incorrect field names
"""

class FieldMapper:
    """Maps incorrect or ambiguous field names to correct schema fields"""
    
    # Field name corrections
    FIELD_MAPPINGS = {
        # Common misnamed fields
        "murder_risk_score": "murder_score",
        "drug_risk_score": "drug_dealing_score", 
        "drug_addict_risk_score": "drug_addict_score",
        "last_travel_date": None,  # This field doesn't exist
        "residency_emirate": "home_city",  # Map residency_emirate to home_city
        
        # Prison/Criminal related
        "prison_record": "is_in_prison",
        "criminal_record_type": "crime_categories_en",
        "crime_types": "crime_categories_en",
        "crime_type": "crime_categories_en",
        
        # Alternative names that might be used
        "job_title": "latest_job_title_en",
        "sponsor": "latest_sponsor_name_en",
        "name": "fullname_en",
        "gender": "gender_en",
        "marital_status": "marital_status_en",
        "dob": "date_of_birth",
        "nationality": "nationality_code",
        "nationality_name": "nationality_name_en",
        "crimes": "crime_categories_en",
        "crime_subcategories": "crime_sub_categories_en",
        "apps": "applications_used",
        "licenses": "driving_license_type",
        
        # Travel related
        "traveled_countries": "travelled_country_codes",
        "last_travel_country": "last_travelled_country_code",
        "travel_countries": "travelled_country_codes",
        
        # Location fields
        "home_geohash": "home_location",
        "work_geohash": "work_location",
    }
    
    @classmethod
    def map_field(cls, field_name: str) -> str:
        """Map field name to correct schema field"""
        # Check if field needs mapping
        if field_name in cls.FIELD_MAPPINGS:
            mapped = cls.FIELD_MAPPINGS[field_name]
            if mapped is None:
                # Field doesn't exist in schema
                raise ValueError(f"Field '{field_name}' does not exist in the schema")
            return mapped
        
        # Return original field name if no mapping needed
        return field_name
    
    @classmethod
    def is_valid_field(cls, field_name: str) -> bool:
        """Check if field is valid (exists in schema)"""
        if field_name in cls.FIELD_MAPPINGS:
            return cls.FIELD_MAPPINGS[field_name] is not None
        return True  # Assume valid if not in mapping
    
    @classmethod
    def clean_filter_tree(cls, filter_tree: dict) -> dict:
        """Clean filter tree by mapping field names"""
        if isinstance(filter_tree, dict):
            # Handle logical operators
            if "AND" in filter_tree:
                cleaned = [cls.clean_filter_tree(child) for child in filter_tree["AND"]]
                # Filter out None values
                cleaned = [c for c in cleaned if c is not None]
                return {"AND": cleaned} if cleaned else None
            elif "OR" in filter_tree:
                cleaned = [cls.clean_filter_tree(child) for child in filter_tree["OR"]]
                # Filter out None values
                cleaned = [c for c in cleaned if c is not None]
                return {"OR": cleaned} if cleaned else None
            elif "NOT" in filter_tree:
                cleaned = cls.clean_filter_tree(filter_tree["NOT"])
                return {"NOT": cleaned} if cleaned else None
            
            # Handle field nodes
            elif "field" in filter_tree:
                field = filter_tree["field"]
                try:
                    mapped_field = cls.map_field(field)
                    # Create new node with mapped field
                    new_node = filter_tree.copy()
                    new_node["field"] = mapped_field
                    
                    # Special handling for travel date queries
                    if field == "last_travel_date" and filter_tree.get("operator") in [">", "<", ">=", "<=", "BETWEEN"]:
                        # This is a travel history query, but we can't filter by date
                        # Return None to remove this filter
                        return None
                    
                    # Special handling for EXISTS operator on boolean fields
                    if new_node.get("operator") == "EXISTS":
                        if mapped_field == "is_in_prison":
                            # Convert EXISTS to boolean check
                            new_node["operator"] = "="
                            new_node["value"] = True
                        else:
                            # For other fields, EXISTS doesn't make sense
                            return None
                        
                    return new_node
                except ValueError:
                    # Field doesn't exist, return None to filter it out
                    return None
        
        return filter_tree