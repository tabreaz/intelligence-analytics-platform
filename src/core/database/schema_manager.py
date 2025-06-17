# src/core/database/schema_manager.py
from typing import Dict


class ClickHouseSchemaManager:
    """
    Centralized schema management for validation

    Purpose:
    - Single source of truth for table schemas
    - Provide schema to validators
    - Support schema evolution
    """

    SCHEMAS = {
        "telecom_db.geo_live": {
            # "doc_id": "String",
            "imsi": "String",
            "phone_no": "Nullable(String)",
            "uid": "Nullable(String)",
            "event_timestamp": "DateTime64(3)",
            "event_date": "Date",
            "time_bucket": "DateTime",
            "day_of_week": "UInt8",
            "hour_of_day": "UInt8",
            "is_weekend": "Bool",
            # "latitude": "Float32",
            # "longitude": "Float32",
            "geohash7": "FixedString(7)",
            # "geohash6": "String",
            # "geohash5": "String",
            "event_type": "LowCardinality(String)",
            "country_code": "FixedString(3)",
            "emirate": "LowCardinality(String)",
            "municipality": "LowCardinality(String)",
            "location_category": "Nullable(Enum8('DENSE' = 1, 'MEDIUM' = 2, 'LOW' = 3, 'LEAST' = 4))",
            # "profile_category": "LowCardinality(String)",
            "fullname_en": "Nullable(String)",
            "gender_en": "Nullable(Enum8('Male' = 1, 'Female' = 2))",
            "age": "Nullable(UInt8)",
            "age_group": "Nullable(Enum8('20-30' = 1, '30-40' = 2, '40-50' = 3, '50-60' = 4, '60-70' = 5))",
            "nationality_code": "Nullable(FixedString(3))",
            "previous_nationality_code": "Nullable(FixedString(3))",
            "residency_status": "Nullable(Enum8('CITIZEN' = 1, 'RESIDENT' = 2, 'VISITOR' = 3, 'INACTIVE' = 4))",
            "dwell_duration_tag": "Nullable(Enum8('LESS_THAN_1_YEAR' = 1, '1_TO_3_YEARS' = 2, '3_TO_5_YEARS' = 3, '5_TO_10_YEARS' = 4, 'MORE_THAN_10_YEARS' = 5))",
            "latest_sponsor_name_en": "Nullable(String)",
            "latest_job_title_en": "Nullable(String)",
            "home_city": "LowCardinality(String)",
            "travelled_country_codes": "Array(FixedString(3))",
            "communicated_country_codes": "Array(FixedString(3))",
            "risk_score": "Float32",
            "drug_addict_score": "Float32",
            "drug_dealing_score": "Float32",
            "murder_score": "Float32",
            "has_investigation_case": "Bool",
            "has_crime_case": "Bool",
            "is_in_prison": "Bool",
            "crime_categories_en": "Array(String)",
            "crime_sub_categories_en": "Array(String)",
            "is_diplomat": "Bool",
            "drug_addict_rules": "Array(String)",
            "drug_dealing_rules": "Array(String)",
            "murder_rules": "Array(String)",
            "risk_rules": "Array(String)",
            "applications_used": "Array(String)",
            "driving_license_type": "Array(String)"
        },
        "telecom_db.movements": {
            "imsi": "String",
            "phone_no": "String",
            # "profile_category": "LowCardinality(String)",
            "event_timestamp": "DateTime64(3)",
            # "latitude": "Float32",
            # "longitude": "Float32",
            "geohash7": "FixedString(7)",
            "emirate": "LowCardinality(String)",
            "municipality": "LowCardinality(String)",
            "event_type": "LowCardinality(String)",
            "is_weekend": "Bool",
            "hour_of_day": "UInt8",
            "event_date": "Date",
            # "geohash5": "String",
            # "geohash4": "String",
            "time_bucket": "DateTime"
        },
        "telecom_db.phone_imsi_uid_latest": {
            "imsi": "String",
            "phone_no": "String",
            "uid": "String",
            "eid": "Array(String)",
            "residency_status": "Enum8('CITIZEN' = 1, 'RESIDENT' = 2, 'VISITOR' = 3, 'INACTIVE' = 4)",
            "fullname_en": "String",
            "gender_en": "Enum8('Male' = 1, 'Female' = 2)",
            "date_of_birth": "Date",
            "age": "UInt8",
            "age_group": "Enum8('20-30' = 1, '30-40' = 2, '40-50' = 3, '50-60' = 4, '60-70' = 5)",
            "marital_status_en": "Enum8('DIVORCED' = 1, 'MARRIED' = 2, 'SINGLE' = 3, 'WIDOWED' = 4)",
            "nationality_code": "FixedString(3)",
            "nationality_name_en": "LowCardinality(String)",
            "previous_nationality_code": "Nullable(FixedString(3))",
            "previous_nationality_en": "Nullable(String)",
            "last_travelled_country_code": "FixedString(3)",
            "travelled_country_codes": "Array(FixedString(3))",
            "dwell_duration_tag": "Nullable(Enum8('LESS_THAN_1_YEAR' = 1, '1_TO_3_YEARS' = 2, '3_TO_5_YEARS' = 3, '5_TO_10_YEARS' = 4, 'MORE_THAN_10_YEARS' = 5))",
            "home_city": "LowCardinality(String)",
            "home_location": "Nullable(String)",
            "work_location": "Nullable(String)",
            "latest_sponsor_name_en": "Nullable(String)",
            "latest_job_title_en": "Nullable(String)",
            "has_investigation_case": "Bool",
            "has_crime_case": "Bool",
            "is_in_prison": "Bool",
            "crime_categories_en": "Array(String)",
            "crime_sub_categories_en": "Array(String)",
            "drug_addict_score": "Float32",
            "drug_dealing_score": "Float32",
            "murder_score": "Float32",
            "risk_score": "Float32",
            "drug_addict_rules": "Array(String)",
            "drug_dealing_rules": "Array(String)",
            "murder_rules": "Array(String)",
            "risk_rules": "Array(String)",
            "communicated_country_codes": "Array(FixedString(3))",
            "applications_used": "Array(String)",
            "driving_license_type": "Array(String)",
            "is_diplomat": "Bool"
        },
        "telecom_db.query_location_geohashes": {
            "query_id": "UUID",
            "location_name": "String",
            "location_index": "UInt8",
            "geohash7": "FixedString(7)",
            # "geohash6": "FixedString(6)",
            # "latitude": "Float32",
            # "longitude": "Float32",
            "radius_meters": "UInt32",
            "confidence": "Float32",
            # "created_at": "DateTime",
            # "part_date": "Date"
        }
    }

    def get_schema_for_validation(self) -> Dict[str, Dict[str, str]]:
        """Get schema in sqlglot format"""
        # Transform to sqlglot compatible format
        return self._transform_to_sqlglot_format()

    def _transform_to_sqlglot_format(self) -> Dict[str, Dict[str, str]]:
        """
        Transform ClickHouse types to sqlglot compatible format

        This method converts ClickHouse-specific types to more generic SQL types
        that sqlglot can understand for validation purposes.
        """
        sqlglot_schemas = {}

        for table_name, columns in self.SCHEMAS.items():
            sqlglot_table = {}
            for col_name, col_type in columns.items():
                # Basic type mapping for sqlglot compatibility
                sqlglot_type = self._map_clickhouse_to_sqlglot_type(col_type)
                sqlglot_table[col_name] = sqlglot_type
            sqlglot_schemas[table_name] = sqlglot_table

        return sqlglot_schemas

    def _map_clickhouse_to_sqlglot_type(self, ch_type: str) -> str:
        """
        Map ClickHouse types to sqlglot-compatible types

        Args:
            ch_type: ClickHouse type definition

        Returns:
            Simplified type string for sqlglot
        """
        # Remove Nullable wrapper
        if ch_type.startswith("Nullable("):
            ch_type = ch_type[9:-1]

        # Remove LowCardinality wrapper
        if ch_type.startswith("LowCardinality("):
            ch_type = ch_type[15:-1]

        # Type mappings
        type_map = {
            "String": "VARCHAR",
            "FixedString": "CHAR",
            "UInt8": "TINYINT",
            "UInt16": "SMALLINT",
            "UInt32": "INT",
            "UInt64": "BIGINT",
            "Int8": "TINYINT",
            "Int16": "SMALLINT",
            "Int32": "INT",
            "Int64": "BIGINT",
            "Float32": "FLOAT",
            "Float64": "DOUBLE",
            "Bool": "BOOLEAN",
            "Date": "DATE",
            "DateTime": "TIMESTAMP",
            "DateTime64": "TIMESTAMP",
            "UUID": "UUID",
            "Array": "ARRAY",
            "Enum8": "VARCHAR",
            "Enum16": "VARCHAR"
        }

        # Extract base type
        base_type = ch_type.split("(")[0]

        return type_map.get(base_type, "VARCHAR")