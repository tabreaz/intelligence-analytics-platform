"""
Profile Analytics Integration Service
Combines profile analytics with extended statistics for comprehensive analysis
"""
import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

from src.services.clickhouse.profile_analytics_service import ClickHouseProfileAnalyticsService
from src.services.statistics.profiler_extended_stats import ProfilerExtendedStatistics
from src.services.response_generator import ResponseGenerator, ResponseConfig, GeneratedResponse
from src.services.base.profile_analytics import ProfileQueryResult
from src.core.llm.base_llm import BaseLLMClient

logger = logging.getLogger(__name__)


class ProfileAnalyticsIntegration:
    """
    Integrates profile analytics with extended statistics
    Provides unified interface for comprehensive profile analysis
    """
    
    def __init__(
        self,
        profile_service: ClickHouseProfileAnalyticsService,
        llm_client: Optional[BaseLLMClient] = None
    ):
        self.profile_service = profile_service
        self.extended_stats = ProfilerExtendedStatistics(profile_service)
        self.response_generator = ResponseGenerator(llm_client)
        
    async def analyze_with_statistics(
        self,
        where_clause: str,
        query_context: Dict[str, Any],
        fields_to_analyze: Optional[List[str]] = None,
        extended_stat_ids: Optional[List[str]] = None,
        limit: int = 100,
        response_config: Optional[ResponseConfig] = None
    ) -> Dict[str, Any]:
        """
        Execute profile query with comprehensive statistics
        
        Args:
            where_clause: SQL WHERE clause for filtering
            query_context: Context from query understanding (category, original query, etc.)
            fields_to_analyze: Fields for distribution analysis
            extended_stat_ids: Specific extended statistics to calculate
            limit: Number of profiles to return
            response_config: Configuration for response generation
            
        Returns:
            Dictionary with query results, statistics, and generated response
        """
        start_time = datetime.utcnow()
        
        # Execute profile query
        query_result = await self.profile_service.execute_profile_query(
            where_clause=where_clause,
            limit=limit,
            session_id=query_context.get("session_id")
        )
        
        # Determine which statistics to calculate
        if not fields_to_analyze:
            fields_to_analyze = self._select_fields_by_category(query_context.get("category"))
        
        if not extended_stat_ids:
            extended_stat_ids = await self._select_extended_stats(where_clause, query_context)
        
        # Calculate statistics in parallel
        stats_tasks = []
        
        # Field distributions
        for field in fields_to_analyze:
            stats_tasks.append(self.profile_service.get_field_distribution(
                where_clause, field, top_n=15
            ))
        
        # Extended statistics
        for stat_id in extended_stat_ids:
            stats_tasks.append(self.extended_stats.calculate_extended_statistic(
                stat_id, where_clause
            ))
        
        # Execute all statistics calculations
        stats_results = await asyncio.gather(*stats_tasks, return_exceptions=True)
        
        # Organize results
        field_distributions = {}
        extended_statistics = []
        
        for i, result in enumerate(stats_results):
            if isinstance(result, Exception):
                logger.error(f"Statistics calculation failed: {result}")
                continue
                
            if i < len(fields_to_analyze):
                # Field distribution result
                field_distributions[fields_to_analyze[i]] = {
                    "total_count": result.total_count,
                    "unique_count": result.unique_count,
                    "distribution": result.distribution,
                    "percentiles": result.percentiles,
                    "metadata": result.metadata
                }
            else:
                # Extended statistic result
                extended_statistics.append(result)
        
        # Combine all statistics
        all_statistics = {
            "field_distributions": field_distributions,
            "extended_statistics": extended_statistics
        }
        
        # Generate intelligent response
        generated_response = await self.response_generator.generate_response(
            query_result=query_result,
            statistics=all_statistics,
            query_context=query_context,
            config=response_config
        )
        
        total_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return {
            "query_result": {
                "query_id": str(query_result.query_id),
                "result_count": query_result.result_count,
                "profiles": query_result.data[:limit],  # Limit profiles returned
                "sql_generated": query_result.sql_generated,
                "execution_time_ms": query_result.execution_time_ms
            },
            "statistics": all_statistics,
            "generated_response": {
                "summary": generated_response.summary,
                "key_insights": generated_response.key_insights,
                "statistics_shown": generated_response.statistics_shown,
                "profiles_shown": generated_response.profiles_shown,
                "sql_filter": generated_response.sql_filter,
                "total_matches": generated_response.total_matches,
                "metadata": generated_response.metadata
            },
            "total_execution_time_ms": total_time_ms,
            "engine_type": self.profile_service.get_engine_type()
        }
    
    async def get_dashboard_statistics(
        self,
        where_clause: str,
        dashboard_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get statistics for dashboard visualization
        
        Args:
            where_clause: SQL WHERE clause for filtering
            dashboard_config: Configuration specifying which stats to include
            
        Returns:
            Dashboard-ready statistics data
        """
        # Get requested statistics
        requested_fields = dashboard_config.get("field_distributions", [])
        requested_extended = dashboard_config.get("extended_statistics", [])
        
        # Calculate all requested statistics
        tasks = []
        
        for field in requested_fields:
            tasks.append(self.profile_service.get_field_distribution(
                where_clause, field, top_n=10
            ))
        
        for stat_id in requested_extended:
            tasks.append(self.extended_stats.calculate_extended_statistic(
                stat_id, where_clause
            ))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Format for dashboard
        dashboard_data = {
            "filter": where_clause,
            "timestamp": datetime.utcnow().isoformat(),
            "widgets": []
        }
        
        # Add field distributions
        for i, field in enumerate(requested_fields):
            if i < len(results) and not isinstance(results[i], Exception):
                result = results[i]
                dashboard_data["widgets"].append({
                    "id": f"field_{field}",
                    "type": "field_distribution",
                    "title": f"{field.replace('_', ' ').title()} Distribution",
                    "data": {
                        "labels": list(result.distribution.keys()),
                        "values": list(result.distribution.values()),
                        "total": result.total_count,
                        "unique": result.unique_count
                    },
                    "visualization": "bar"
                })
        
        # Add extended statistics
        extended_start = len(requested_fields)
        for i, stat_id in enumerate(requested_extended):
            idx = extended_start + i
            if idx < len(results) and not isinstance(results[idx], Exception):
                result = results[idx]
                dashboard_data["widgets"].append(result)
        
        return dashboard_data
    
    def _select_fields_by_category(self, category: Optional[str]) -> List[str]:
        """Select relevant fields based on query category"""
        category_fields = {
            "demographic": ["nationality_code", "gender_en", "age_group", "residency_status"],
            "risk_analysis": ["risk_score", "has_crime_case", "drug_dealing_score"],
            "travel": ["travelled_country_codes", "home_city"],
            "communication": ["applications_used", "communicated_country_codes"],
            "employment": ["latest_job_title_en", "is_diplomat"],
            "location": ["home_city", "residency_status", "dwell_duration_tag"]
        }
        
        # Default fields if category not found
        default_fields = ["nationality_code", "risk_score", "gender_en"]
        
        return category_fields.get(category, default_fields)
    
    async def _select_extended_stats(
        self,
        where_clause: str,
        query_context: Dict[str, Any]
    ) -> List[str]:
        """Select relevant extended statistics based on context"""
        category = query_context.get("category", "")
        query_text = query_context.get("original_query", "").lower()
        
        # Category-based selection
        category_stats = {
            "risk_analysis": ["avg_risk_score", "risk_rules_distribution", "combined_risk_indicators"],
            "demographic": ["nationality_changes", "long_term_residents"],
            "travel": ["travel_frequency", "work_home_comparison"],
            "crime": ["crime_category_breakdown", "substance_risk_correlation"],
            "employment": ["top_job_categories", "diplomat_analysis"],
            "digital": ["social_media_apps"]
        }
        
        selected = category_stats.get(category, [])
        
        # Add stats based on keywords
        if "risk" in query_text and "avg_risk_score" not in selected:
            selected.append("avg_risk_score")
        if "travel" in query_text and "travel_frequency" not in selected:
            selected.append("travel_frequency")
        if "crime" in query_text and "crime_category_breakdown" not in selected:
            selected.append("crime_category_breakdown")
        
        # Default to top priority stats if none selected
        if not selected:
            selected = ["avg_risk_score", "risk_rules_distribution", "travel_frequency"]
        
        # Limit to 5 stats
        return selected[:5]
    
    async def get_available_statistics(self) -> Dict[str, Any]:
        """Get all available statistics for configuration"""
        return {
            "field_distributions": {
                "categories": {
                    "identifiers": list(self.profile_service.IDENTIFIER_FIELDS),
                    "numeric": list(self.profile_service.NUMERIC_FIELDS),
                    "categorical": list(self.profile_service.CATEGORICAL_FIELDS),
                    "arrays": list(self.profile_service.ARRAY_FIELDS),
                    "boolean": list(self.profile_service.BOOLEAN_FIELDS)
                }
            },
            "extended_statistics": self.extended_stats.get_available_extended_statistics()
        }