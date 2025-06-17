"""
ClickHouse Profile Analytics Service Implementation
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from uuid import UUID, uuid4

from src.services.base.profile_analytics import (
    BaseProfileAnalyticsService, 
    FieldStatistics, 
    ProfileQueryResult,
    AggregationType
)
from src.services.base.query_executor import BaseQueryExecutor
from src.core.session_manager import EnhancedSessionManager

logger = logging.getLogger(__name__)


class ClickHouseProfileAnalyticsService(BaseProfileAnalyticsService):
    """
    ClickHouse implementation of profile analytics service
    Optimized for phone_imsi_uid_latest table analysis
    """
    
    def __init__(
        self, 
        query_executor: BaseQueryExecutor,
        session_manager: Optional[EnhancedSessionManager] = None
    ):
        """
        Initialize service with query executor
        
        Args:
            query_executor: Query executor instance (dependency injection)
            session_manager: Optional session manager for query history
        """
        self.executor = query_executor
        self.session_manager = session_manager
        self._initialized = False
        
        # ClickHouse-specific field categorization
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
        
        # Default table for profile queries
        self.profile_table = "telecom_db.phone_imsi_uid_latest"
    
    async def initialize(self) -> None:
        """Initialize the service and its dependencies"""
        if self._initialized:
            return
            
        await self.executor.initialize()
        
        # Session manager is already initialized in factory
        
        self._initialized = True
        logger.info("ClickHouse profile analytics service initialized")
    
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
        """Execute a profile query with the given WHERE clause"""
        await self.initialize()
        
        query_id = uuid4()
        start_time = datetime.utcnow()
        
        # Build SQL query
        sql = self._build_profile_sql(
            where_clause, select_fields, limit, offset, order_by
        )
        
        try:
            # Execute query - we know the columns we selected
            headers, data_rows = await self.executor.execute_query_with_headers(
                sql, column_names=select_fields
            )
            execution_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Convert to list of dicts
            data = []
            for row in data_rows:
                data.append(dict(zip(headers, row)))
            
            # Store query history if session manager available
            if self.session_manager:
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
            logger.error(f"Profile query execution failed: {e}")
            
            # Store failed query
            if self.session_manager:
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
        """Get detailed profile information with pagination"""
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
        """Get unique counts for identifier fields"""
        if not identifier_fields:
            identifier_fields = list(self.IDENTIFIER_FIELDS)
        
        counts = {}
        
        # Execute count queries in parallel
        tasks = []
        for field in identifier_fields:
            if field == 'eid':
                # Special handling for array field in ClickHouse
                sql = f"""
                    SELECT COUNT(DISTINCT arrayJoin({field})) as count
                    FROM {self.profile_table}
                    WHERE {where_clause}
                """
            else:
                sql = f"""
                    SELECT COUNT(DISTINCT {field}) as count
                    FROM {self.profile_table}
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
        """Get distribution of values for a specific field"""
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
        """Get comprehensive demographic statistics"""
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
        """Get risk score statistics"""
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
        """Get communication pattern statistics"""
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
        """Get travel pattern statistics"""
        travel_fields = ['travelled_country_codes']
        
        tasks = []
        for field in travel_fields:
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
        """Get cross-tabulation between two fields"""
        sql = f"""
            SELECT 
                {field1},
                {field2},
                COUNT(*) as count
            FROM {self.profile_table}
            WHERE {where_clause}
            GROUP BY {field1}, {field2}
            ORDER BY count DESC
            LIMIT {limit}
        """
        
        # We know the column names
        headers, data_rows = await self.executor.execute_query_with_headers(
            sql, column_names=[field1, field2, 'count']
        )
        
        # Convert to nested dict structure
        cross_tab = {}
        for row in data_rows:
            val1, val2, count = row
            # Convert values to strings to ensure they're hashable
            # Handle bytes values from ClickHouse
            if isinstance(val1, bytes):
                val1_str = val1.decode('utf-8')
            else:
                val1_str = str(val1) if val1 is not None else 'NULL'
            
            if isinstance(val2, bytes):
                val2_str = val2.decode('utf-8')
            else:
                val2_str = str(val2) if val2 is not None else 'NULL'
            
            if val1_str not in cross_tab:
                cross_tab[val1_str] = {}
            cross_tab[val1_str][val2_str] = count
        
        return {
            "field1": field1,
            "field2": field2,
            "data": cross_tab,
            "total_combinations": len(data_rows)
        }
    
    def get_supported_fields(self) -> Dict[str, List[str]]:
        """Get supported fields by category"""
        return {
            "identifiers": list(self.IDENTIFIER_FIELDS),
            "numeric": list(self.NUMERIC_FIELDS),
            "categorical": list(self.CATEGORICAL_FIELDS),
            "arrays": list(self.ARRAY_FIELDS),
            "boolean": list(self.BOOLEAN_FIELDS)
        }
    
    def get_engine_type(self) -> str:
        """Get the underlying query engine type"""
        return self.executor.get_engine_type()
    
    # Private helper methods specific to ClickHouse
    
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
        
        # Build SELECT clause with proper quoting
        select_clause = ", ".join(
            self.executor.quote_identifier(f) for f in select_fields
        )
        
        # Build ORDER BY clause
        order_clause = ""
        if order_by:
            order_parts = []
            for order in order_by:
                field = order.get('field', 'risk_score')
                direction = order.get('direction', 'DESC')
                quoted_field = self.executor.quote_identifier(field)
                order_parts.append(f"{quoted_field} {direction}")
            order_clause = f"ORDER BY {', '.join(order_parts)}"
        
        sql = f"""
            SELECT {select_clause}
            FROM {self.profile_table}
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
            FROM {self.profile_table}
            WHERE {where_clause}
        """
        
        return await self.executor.execute_count_query(sql)
    
    async def _execute_count_query(
        self, 
        field_name: str, 
        sql: str
    ) -> Tuple[str, int]:
        """Execute a count query and return field name with count"""
        count = await self.executor.execute_count_query(sql)
        return (field_name, count)
    
    async def _get_categorical_distribution(
        self,
        where_clause: str,
        field_name: str,
        top_n: int,
        include_others: bool
    ) -> FieldStatistics:
        """Get distribution for categorical fields - ClickHouse specific"""
        sql = f"""
            SELECT 
                {field_name} as value,
                COUNT(*) as count
            FROM {self.profile_table}
            WHERE {where_clause}
            GROUP BY {field_name}
            ORDER BY count DESC
            LIMIT {top_n + 1}
        """
        
        # We know the column names
        headers, data_rows = await self.executor.execute_query_with_headers(
            sql, column_names=['value', 'count']
        )
        
        distribution = {}
        total_count = 0
        
        for i, row in enumerate(data_rows):
            if i < top_n:
                value, count = row
                # Handle bytes values from ClickHouse
                if isinstance(value, bytes):
                    value = value.decode('utf-8')
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
            FROM {self.profile_table}
            WHERE {where_clause}
        """
        
        unique_count = await self.executor.execute_count_query(unique_sql)
        
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
        """Get distribution for numeric fields - ClickHouse specific"""
        # Get statistics using ClickHouse quantile functions
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
            FROM {self.profile_table}
            WHERE {where_clause}
        """
        
        # We know the column names
        headers, data_rows = await self.executor.execute_query_with_headers(
            stats_sql, 
            column_names=['min_val', 'max_val', 'avg_val', 'p25', 'p50', 'p75', 'p95', 'total_count', 'unique_count']
        )
        
        if not data_rows:
            return FieldStatistics(field_name=field_name, total_count=0)
        
        stats = data_rows[0]
        min_val, max_val, avg_val, p25, p50, p75, p95, total_count, unique_count = stats
        
        # Create histogram bins
        distribution = {}
        if min_val is not None and max_val is not None and min_val < max_val:
            bin_width = (max_val - min_val) / num_bins
            
            # Get histogram data using ClickHouse histogram function
            hist_sql = f"""
                SELECT 
                    floor(({field_name} - {min_val}) / {bin_width}) as bin_num,
                    COUNT(*) as count
                FROM {self.profile_table}
                WHERE {where_clause} AND {field_name} IS NOT NULL
                GROUP BY bin_num
                ORDER BY bin_num
            """
            
            # We know the column names
            hist_headers, hist_rows = await self.executor.execute_query_with_headers(
                hist_sql, column_names=['bin_num', 'count']
            )
            
            for row in hist_rows:
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
        """Get distribution for array fields - ClickHouse specific"""
        # Use arrayJoin to flatten arrays
        sql = f"""
            SELECT 
                arrayJoin({field_name}) as value,
                COUNT(*) as count
            FROM {self.profile_table}
            WHERE {where_clause} AND notEmpty({field_name})
            GROUP BY value
            ORDER BY count DESC
            LIMIT {top_n}
        """
        
        # We know the column names
        headers, data_rows = await self.executor.execute_query_with_headers(
            sql, column_names=['value', 'count']
        )
        
        distribution = {}
        total_count = 0
        
        for row in data_rows:
            value, count = row
            distribution[str(value)] = count
            total_count += count
        
        # Get unique values count using ClickHouse array functions
        unique_sql = f"""
            SELECT COUNT(DISTINCT arrayJoin({field_name})) as unique_count
            FROM {self.profile_table}
            WHERE {where_clause} AND notEmpty({field_name})
        """
        
        unique_count = await self.executor.execute_count_query(unique_sql)
        
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
            FROM {self.profile_table}
            WHERE {where_clause}
            GROUP BY {field_name}
            ORDER BY {field_name}
        """
        
        # We know the column names
        headers, data_rows = await self.executor.execute_query_with_headers(
            sql, column_names=['value', 'count']
        )
        
        distribution = {"true": 0, "false": 0}
        total_count = 0
        
        for row in data_rows:
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
        if not self.session_manager or not self.session_manager.db_pool:
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
            
            async with self.session_manager.db_pool.acquire() as conn:
                await conn.execute(
                    sql,
                    str(query_id), session_id, user_id,
                    where_clause, where_clause, sql_generated,
                    status, result_count, execution_time_ms,
                    error_message, datetime.utcnow()
                )
        except Exception as e:
            logger.error(f"Failed to store query history: {e}")