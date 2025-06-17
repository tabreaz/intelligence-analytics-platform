"""
Profile Analytics Service
Handles profile-only (non-spatial) queries with comprehensive analytics
"""
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
from uuid import UUID, uuid4
import logging
from dataclasses import dataclass, field
from enum import Enum

from src.core.database.clickhouse_client import ClickHouseClient
from src.core.database.postgres_client import PostgresClient
from src.core.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class AggregationType(Enum):
    """Types of aggregations supported"""
    COUNT = "count"
    UNIQUE_COUNT = "unique_count"
    DISTRIBUTION = "distribution"
    PERCENTILE = "percentile"
    TIME_SERIES = "time_series"
    CROSS_TAB = "cross_tab"


@dataclass
class FieldStatistics:
    """Statistics for a single field"""
    field_name: str
    total_count: int
    unique_count: Optional[int] = None
    null_count: int = 0
    distribution: Dict[str, int] = field(default_factory=dict)
    percentiles: Dict[float, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProfileQueryResult:
    """Result of a profile query execution"""
    query_id: UUID
    session_id: Optional[str]
    sql_generated: str
    execution_time_ms: float
    result_count: int
    data: List[Dict[str, Any]]
    statistics: Dict[str, FieldStatistics] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProfileAnalyticsService:
    """
    Service for executing and analyzing profile-only queries
    Provides detailed analytics and statistics for dashboard visualization
    """
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.clickhouse_client = None
        self.postgres_client = None
        self._initialized = False
        
        # Field categorization for analytics
        self.IDENTIFIER_FIELDS = {'imsi', 'phone_no', 'uid', 'eid'}
        self.NUMERIC_FIELDS = {
            'age', 'risk_score', 'drug_dealing_score', 
            'drug_addict_score', 'murder_score'
        }
        self.CATEGORICAL_FIELDS = {
            'nationality_code', 'gender_en', 'age_group', 
            'residency_status', 'marital_status_en', 'home_city',
            'dwell_duration_tag', 'latest_job_title_en'
        }
        self.ARRAY_FIELDS = {
            'travelled_country_codes', 'communicated_country_codes',
            'crime_categories_en', 'applications_used', 'driving_license_type'
        }
        self.BOOLEAN_FIELDS = {
            'has_investigation_case', 'has_crime_case', 
            'is_in_prison', 'is_diplomat'
        }
    
    async def initialize(self):
        """Initialize database connections"""
        if self._initialized:
            return
            
        # Initialize ClickHouse
        ch_config = self.config.get('database.clickhouse')
        self.clickhouse_client = ClickHouseClient(
            host=ch_config['host'],
            port=ch_config.get('port', 8123),
            user=ch_config.get('user', 'default'),
            password=ch_config.get('password', ''),
            database=ch_config.get('database', 'telecom_db')
        )
        await self.clickhouse_client.connect()
        
        # Initialize PostgreSQL for storing query history
        pg_config = self.config.get('database.postgres')
        if pg_config:
            self.postgres_client = PostgresClient(pg_config)
            await self.postgres_client.connect()
        
        self._initialized = True
    
    async def execute_profile_query(
        self,
        where_clause: str,
        select_fields: Optional[List[str]] = None,
        limit: int = 1000,
        offset: int = 0,
        order_by: Optional[List[Dict[str, str]]] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> ProfileQueryResult:
        """
        Execute a profile query with the given WHERE clause
        
        Args:
            where_clause: SQL WHERE clause (without WHERE keyword)
            select_fields: Fields to select (None = default fields)
            limit: Result limit
            offset: Result offset for pagination
            order_by: List of order by specifications
            session_id: Session ID for tracking
            user_id: User ID for tracking
            
        Returns:
            ProfileQueryResult with data and statistics
        """
        await self.initialize()
        
        query_id = uuid4()
        start_time = datetime.utcnow()
        
        # Build SQL query
        sql = self._build_profile_sql(
            where_clause, select_fields, limit, offset, order_by
        )
        
        try:
            # Execute query
            result = await self.clickhouse_client.execute_query(sql)
            execution_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Convert to list of dicts
            data = []
            if result and len(result) > 1:  # First row is headers
                headers = result[0]
                for row in result[1:]:
                    data.append(dict(zip(headers, row)))
            
            # Store query history
            if self.postgres_client:
                await self._store_query_history(
                    query_id, session_id, user_id, where_clause, 
                    sql, len(data), execution_time_ms, None
                )
            
            return ProfileQueryResult(
                query_id=query_id,
                session_id=session_id,
                sql_generated=sql,
                execution_time_ms=execution_time_ms,
                result_count=len(data),
                data=data
            )
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            
            # Store failed query
            if self.postgres_client:
                await self._store_query_history(
                    query_id, session_id, user_id, where_clause,
                    sql, 0, 0, str(e)
                )
            
            raise
    
    async def get_profile_details(
        self,
        where_clause: str,
        fields: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Get detailed profile information with pagination
        
        Returns basic user profile details for dashboard display
        """
        if not fields:
            # Default profile fields for display
            fields = [
                'imsi', 'phone_no', 'uid', 'fullname_en',
                'nationality_code', 'gender_en', 'age', 'age_group',
                'residency_status', 'home_city', 'risk_score',
                'has_crime_case', 'is_diplomat'
            ]
        
        result = await self.execute_profile_query(
            where_clause=where_clause,
            select_fields=fields,
            limit=limit,
            offset=offset,
            order_by=[{"field": "risk_score", "direction": "DESC"}]
        )
        
        return {
            "total_count": await self._get_total_count(where_clause),
            "page_count": result.result_count,
            "limit": limit,
            "offset": offset,
            "data": result.data,
            "execution_time_ms": result.execution_time_ms
        }
    
    async def get_unique_counts(
        self,
        where_clause: str,
        identifier_fields: Optional[List[str]] = None
    ) -> Dict[str, int]:
        """
        Get unique counts for identifier fields
        
        Returns counts of unique IMSIs, UIDs, phone numbers, and EIDs
        """
        if not identifier_fields:
            identifier_fields = list(self.IDENTIFIER_FIELDS)
        
        counts = {}
        
        # Execute count queries in parallel
        tasks = []
        for field in identifier_fields:
            if field == 'eid':
                # Special handling for array field
                sql = f"""
                    SELECT COUNT(DISTINCT arrayJoin({field})) as count
                    FROM telecom_db.phone_imsi_uid_latest
                    WHERE {where_clause}
                """
            else:
                sql = f"""
                    SELECT COUNT(DISTINCT {field}) as count
                    FROM telecom_db.phone_imsi_uid_latest
                    WHERE {where_clause}
                """
            tasks.append(self._execute_count_query(field, sql))
        
        results = await asyncio.gather(*tasks)
        
        for field, count in results:
            counts[field] = count
        
        return counts
    
    async def get_field_distribution(
        self,
        where_clause: str,
        field_name: str,
        top_n: int = 20,
        include_others: bool = True
    ) -> FieldStatistics:
        """
        Get distribution of values for a specific field
        
        Handles different field types appropriately:
        - Categorical: frequency counts
        - Numeric: binned distribution
        - Array: flattened frequency counts
        - Boolean: true/false counts
        """
        await self.initialize()
        
        # Determine field type and build appropriate query
        if field_name in self.CATEGORICAL_FIELDS:
            return await self._get_categorical_distribution(
                where_clause, field_name, top_n, include_others
            )
        elif field_name in self.NUMERIC_FIELDS:
            return await self._get_numeric_distribution(
                where_clause, field_name
            )
        elif field_name in self.ARRAY_FIELDS:
            return await self._get_array_distribution(
                where_clause, field_name, top_n, include_others
            )
        elif field_name in self.BOOLEAN_FIELDS:
            return await self._get_boolean_distribution(
                where_clause, field_name
            )
        else:
            raise ValueError(f"Unknown field type: {field_name}")
    
    async def get_demographic_statistics(
        self,
        where_clause: str
    ) -> Dict[str, FieldStatistics]:
        """
        Get comprehensive demographic statistics
        
        Includes distributions for:
        - nationality_code (top 20)
        - gender_en
        - age_group
        - marital_status_en
        - residency_status
        """
        demographic_fields = [
            'nationality_code', 'gender_en', 'age_group',
            'marital_status_en', 'residency_status'
        ]
        
        tasks = []
        for field in demographic_fields:
            top_n = 20 if field == 'nationality_code' else 10
            tasks.append(self.get_field_distribution(
                where_clause, field, top_n=top_n
            ))
        
        results = await asyncio.gather(*tasks)
        
        return {
            field: stats 
            for field, stats in zip(demographic_fields, results)
        }
    
    async def get_risk_statistics(
        self,
        where_clause: str
    ) -> Dict[str, FieldStatistics]:
        """
        Get risk score statistics
        
        Includes:
        - risk_score distribution
        - drug_dealing_score distribution
        - drug_addict_score distribution
        - murder_score distribution
        - Crime flags (has_crime_case, is_in_prison)
        """
        risk_fields = [
            'risk_score', 'drug_dealing_score', 
            'drug_addict_score', 'murder_score'
        ]
        flag_fields = ['has_crime_case', 'is_in_prison']
        
        tasks = []
        
        # Get numeric distributions
        for field in risk_fields:
            tasks.append(self.get_field_distribution(where_clause, field))
        
        # Get boolean distributions
        for field in flag_fields:
            tasks.append(self.get_field_distribution(where_clause, field))
        
        results = await asyncio.gather(*tasks)
        
        stats = {}
        for i, field in enumerate(risk_fields + flag_fields):
            stats[field] = results[i]
        
        return stats
    
    async def get_communication_statistics(
        self,
        where_clause: str
    ) -> Dict[str, FieldStatistics]:
        """
        Get communication pattern statistics
        
        Includes:
        - applications_used distribution
        - communicated_country_codes distribution
        """
        comm_fields = ['applications_used', 'communicated_country_codes']
        
        tasks = []
        for field in comm_fields:
            tasks.append(self.get_field_distribution(
                where_clause, field, top_n=20
            ))
        
        results = await asyncio.gather(*tasks)
        
        return {
            field: stats 
            for field, stats in zip(comm_fields, results)
        }
    
    async def get_travel_statistics(
        self,
        where_clause: str
    ) -> Dict[str, FieldStatistics]:
        """
        Get travel pattern statistics
        
        Includes:
        - travelled_country_codes distribution
        - last_travelled_country_code distribution
        """
        travel_fields = ['travelled_country_codes', 'last_travelled_country_code']
        
        tasks = []
        for field in travel_fields:
            # Skip if field doesn't exist
            if field == 'last_travelled_country_code':
                field = 'nationality_code'  # Fallback for now
            tasks.append(self.get_field_distribution(
                where_clause, field, top_n=30
            ))
        
        results = await asyncio.gather(*tasks)
        
        return {
            field: stats 
            for field, stats in zip(travel_fields, results)
        }
    
    async def get_cross_tabulation(
        self,
        where_clause: str,
        field1: str,
        field2: str,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Get cross-tabulation between two fields
        
        Useful for understanding relationships like:
        - nationality vs risk_score ranges
        - gender vs age_group
        - has_crime_case vs nationality
        """
        sql = f"""
            SELECT 
                {field1},
                {field2},
                COUNT(*) as count
            FROM telecom_db.phone_imsi_uid_latest
            WHERE {where_clause}
            GROUP BY {field1}, {field2}
            ORDER BY count DESC
            LIMIT {limit}
        """
        
        result = await self.clickhouse_client.execute_query(sql)
        
        # Convert to nested dict structure
        cross_tab = {}
        if result and len(result) > 1:
            for row in result[1:]:
                val1, val2, count = row
                if val1 not in cross_tab:
                    cross_tab[val1] = {}
                cross_tab[val1][val2] = count
        
        return {
            "field1": field1,
            "field2": field2,
            "data": cross_tab,
            "total_combinations": len(result) - 1 if result else 0
        }
    
    # Private helper methods
    
    def _build_profile_sql(
        self,
        where_clause: str,
        select_fields: Optional[List[str]],
        limit: int,
        offset: int,
        order_by: Optional[List[Dict[str, str]]]
    ) -> str:
        """Build SQL query for profile data"""
        # Default fields if none specified
        if not select_fields:
            select_fields = [
                'imsi', 'phone_no', 'uid', 'fullname_en',
                'nationality_code', 'gender_en', 'age',
                'risk_score', 'home_city'
            ]
        
        # Build SELECT clause
        select_clause = ", ".join(f"`{f}`" for f in select_fields)
        
        # Build ORDER BY clause
        order_clause = ""
        if order_by:
            order_parts = []
            for order in order_by:
                field = order.get('field', 'risk_score')
                direction = order.get('direction', 'DESC')
                order_parts.append(f"`{field}` {direction}")
            order_clause = f"ORDER BY {', '.join(order_parts)}"
        
        sql = f"""
            SELECT {select_clause}
            FROM telecom_db.phone_imsi_uid_latest
            WHERE {where_clause}
            {order_clause}
            LIMIT {limit}
            OFFSET {offset}
            SETTINGS max_memory_usage = 3000000000
        """
        
        return sql.strip()
    
    async def _get_total_count(self, where_clause: str) -> int:
        """Get total count for a WHERE clause"""
        sql = f"""
            SELECT COUNT(*) as count
            FROM telecom_db.phone_imsi_uid_latest
            WHERE {where_clause}
        """
        
        result = await self.clickhouse_client.execute_query(sql)
        if result and len(result) > 1:
            return result[1][0]
        return 0
    
    async def _execute_count_query(
        self, 
        field_name: str, 
        sql: str
    ) -> Tuple[str, int]:
        """Execute a count query and return field name with count"""
        result = await self.clickhouse_client.execute_query(sql)
        count = 0
        if result and len(result) > 1:
            count = result[1][0]
        return (field_name, count)
    
    async def _get_categorical_distribution(
        self,
        where_clause: str,
        field_name: str,
        top_n: int,
        include_others: bool
    ) -> FieldStatistics:
        """Get distribution for categorical fields"""
        sql = f"""
            SELECT 
                {field_name} as value,
                COUNT(*) as count
            FROM telecom_db.phone_imsi_uid_latest
            WHERE {where_clause}
            GROUP BY {field_name}
            ORDER BY count DESC
            LIMIT {top_n + 1}
        """
        
        result = await self.clickhouse_client.execute_query(sql)
        
        distribution = {}
        total_count = 0
        
        if result and len(result) > 1:
            for i, row in enumerate(result[1:]):
                if i < top_n:
                    value, count = row
                    distribution[str(value)] = count
                    total_count += count
                else:
                    # Count for "Others"
                    if include_others:
                        _, count = row
                        if 'Others' not in distribution:
                            distribution['Others'] = 0
                        distribution['Others'] += count
                        total_count += count
        
        # Get unique count
        unique_sql = f"""
            SELECT COUNT(DISTINCT {field_name}) as unique_count
            FROM telecom_db.phone_imsi_uid_latest
            WHERE {where_clause}
        """
        
        unique_result = await self.clickhouse_client.execute_query(unique_sql)
        unique_count = unique_result[1][0] if unique_result and len(unique_result) > 1 else 0
        
        return FieldStatistics(
            field_name=field_name,
            total_count=total_count,
            unique_count=unique_count,
            distribution=distribution
        )
    
    async def _get_numeric_distribution(
        self,
        where_clause: str,
        field_name: str,
        num_bins: int = 20
    ) -> FieldStatistics:
        """Get distribution for numeric fields"""
        # Get min, max, percentiles
        stats_sql = f"""
            SELECT 
                MIN({field_name}) as min_val,
                MAX({field_name}) as max_val,
                AVG({field_name}) as avg_val,
                quantile(0.25)({field_name}) as p25,
                quantile(0.5)({field_name}) as p50,
                quantile(0.75)({field_name}) as p75,
                quantile(0.95)({field_name}) as p95,
                COUNT(*) as total_count,
                COUNT(DISTINCT {field_name}) as unique_count
            FROM telecom_db.phone_imsi_uid_latest
            WHERE {where_clause}
        """
        
        result = await self.clickhouse_client.execute_query(stats_sql)
        
        if not result or len(result) < 2:
            return FieldStatistics(field_name=field_name, total_count=0)
        
        stats = result[1]
        min_val, max_val, avg_val, p25, p50, p75, p95, total_count, unique_count = stats
        
        # Create histogram bins
        distribution = {}
        if min_val is not None and max_val is not None and min_val < max_val:
            bin_width = (max_val - min_val) / num_bins
            
            # Get histogram data
            hist_sql = f"""
                SELECT 
                    floor(({field_name} - {min_val}) / {bin_width}) as bin_num,
                    COUNT(*) as count
                FROM telecom_db.phone_imsi_uid_latest
                WHERE {where_clause} AND {field_name} IS NOT NULL
                GROUP BY bin_num
                ORDER BY bin_num
            """
            
            hist_result = await self.clickhouse_client.execute_query(hist_sql)
            
            if hist_result and len(hist_result) > 1:
                for row in hist_result[1:]:
                    bin_num, count = row
                    bin_start = min_val + (bin_num * bin_width)
                    bin_end = bin_start + bin_width
                    bin_label = f"{bin_start:.2f}-{bin_end:.2f}"
                    distribution[bin_label] = count
        
        return FieldStatistics(
            field_name=field_name,
            total_count=int(total_count),
            unique_count=int(unique_count),
            distribution=distribution,
            percentiles={
                0.0: float(min_val) if min_val is not None else 0,
                0.25: float(p25) if p25 is not None else 0,
                0.5: float(p50) if p50 is not None else 0,
                0.75: float(p75) if p75 is not None else 0,
                0.95: float(p95) if p95 is not None else 0,
                1.0: float(max_val) if max_val is not None else 0
            },
            metadata={
                "min": float(min_val) if min_val is not None else 0,
                "max": float(max_val) if max_val is not None else 0,
                "avg": float(avg_val) if avg_val is not None else 0
            }
        )
    
    async def _get_array_distribution(
        self,
        where_clause: str,
        field_name: str,
        top_n: int,
        include_others: bool
    ) -> FieldStatistics:
        """Get distribution for array fields"""
        sql = f"""
            SELECT 
                arrayJoin({field_name}) as value,
                COUNT(*) as count
            FROM telecom_db.phone_imsi_uid_latest
            WHERE {where_clause} AND notEmpty({field_name})
            GROUP BY value
            ORDER BY count DESC
            LIMIT {top_n}
        """
        
        result = await self.clickhouse_client.execute_query(sql)
        
        distribution = {}
        total_count = 0
        
        if result and len(result) > 1:
            for row in result[1:]:
                value, count = row
                distribution[str(value)] = count
                total_count += count
        
        # Get unique values count
        unique_sql = f"""
            SELECT COUNT(DISTINCT arrayJoin({field_name})) as unique_count
            FROM telecom_db.phone_imsi_uid_latest
            WHERE {where_clause} AND notEmpty({field_name})
        """
        
        unique_result = await self.clickhouse_client.execute_query(unique_sql)
        unique_count = unique_result[1][0] if unique_result and len(unique_result) > 1 else 0
        
        return FieldStatistics(
            field_name=field_name,
            total_count=total_count,
            unique_count=unique_count,
            distribution=distribution,
            metadata={"field_type": "array"}
        )
    
    async def _get_boolean_distribution(
        self,
        where_clause: str,
        field_name: str
    ) -> FieldStatistics:
        """Get distribution for boolean fields"""
        sql = f"""
            SELECT 
                {field_name} as value,
                COUNT(*) as count
            FROM telecom_db.phone_imsi_uid_latest
            WHERE {where_clause}
            GROUP BY {field_name}
            ORDER BY {field_name}
        """
        
        result = await self.clickhouse_client.execute_query(sql)
        
        distribution = {"true": 0, "false": 0}
        total_count = 0
        
        if result and len(result) > 1:
            for row in result[1:]:
                value, count = row
                key = "true" if value else "false"
                distribution[key] = count
                total_count += count
        
        return FieldStatistics(
            field_name=field_name,
            total_count=total_count,
            unique_count=2,
            distribution=distribution,
            metadata={
                "field_type": "boolean",
                "true_percentage": (distribution["true"] / total_count * 100) if total_count > 0 else 0
            }
        )
    
    async def _store_query_history(
        self,
        query_id: UUID,
        session_id: Optional[str],
        user_id: Optional[str],
        where_clause: str,
        sql_generated: str,
        result_count: int,
        execution_time_ms: float,
        error_message: Optional[str]
    ):
        """Store query execution history in PostgreSQL"""
        if not self.postgres_client:
            return
        
        try:
            sql = """
                INSERT INTO query_execution_history (
                    query_id, session_id, user_id, 
                    context_aware_query, original_query, sql_generated,
                    query_status, result_count, execution_time_ms,
                    error_message, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """
            
            status = "failed" if error_message else "completed"
            
            await self.postgres_client.execute(
                sql,
                str(query_id), session_id, user_id,
                where_clause, where_clause, sql_generated,
                status, result_count, execution_time_ms,
                error_message, datetime.utcnow()
            )
        except Exception as e:
            logger.error(f"Failed to store query history: {e}")