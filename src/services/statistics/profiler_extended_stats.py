"""
Extended Profiler Statistics Service
Provides additional statistics beyond basic field distributions
"""
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import asyncio

from src.services.clickhouse.profile_analytics_service import ClickHouseProfileAnalyticsService
from src.services.base.profile_analytics import FieldStatistics

logger = logging.getLogger(__name__)


@dataclass
class ExtendedStatistic:
    """Extended statistic definition"""
    id: str
    name: str
    description: str
    category: str
    sql_template: str
    visualization: str
    priority: int = 0  # Higher = more important


class ProfilerExtendedStatistics:
    """
    Extended statistics for profiler data
    Complements the basic field distributions with specialized queries
    """
    
    def __init__(self, profile_service: ClickHouseProfileAnalyticsService):
        self.profile_service = profile_service
        self.table = "telecom_db.phone_imsi_uid_latest"
        self.extended_stats = self._define_extended_statistics()
    
    def _define_extended_statistics(self) -> Dict[str, ExtendedStatistic]:
        """Define extended statistics beyond basic distributions"""
        return {
            # Risk Analysis
            "avg_risk_score": ExtendedStatistic(
                id="avg_risk_score",
                name="Average Risk Score",
                description="Average risk score with percentiles",
                category="risk",
                sql_template="""
                    SELECT 
                        AVG(risk_score) as avg_score,
                        quantile(0.25)(risk_score) as p25,
                        quantile(0.50)(risk_score) as p50,
                        quantile(0.75)(risk_score) as p75,
                        quantile(0.95)(risk_score) as p95,
                        COUNT(*) as total_count
                    FROM {table}
                    WHERE {filter}
                """,
                visualization="gauge",
                priority=10
            ),
            
            "risk_rules_distribution": ExtendedStatistic(
                id="risk_rules_distribution",
                name="Risk Rules Triggered",
                description="Distribution of risk rules that were triggered",
                category="risk",
                sql_template="""
                    SELECT 
                        arrayJoin(risk_rules) as rule,
                        COUNT(*) as count
                    FROM {table}
                    WHERE {filter} AND notEmpty(risk_rules)
                    GROUP BY rule
                    ORDER BY count DESC
                    LIMIT 15
                """,
                visualization="horizontal_bar",
                priority=8
            ),
            
            # Travel Patterns
            "travel_frequency": ExtendedStatistic(
                id="travel_frequency",
                name="Travel Frequency Distribution",
                description="How many countries people have visited",
                category="travel",
                sql_template="""
                    SELECT 
                        CASE 
                            WHEN length(travelled_country_codes) = 0 THEN 'Never Traveled'
                            WHEN length(travelled_country_codes) <= 3 THEN '1-3 Countries'
                            WHEN length(travelled_country_codes) <= 10 THEN '4-10 Countries'
                            WHEN length(travelled_country_codes) <= 20 THEN '11-20 Countries'
                            ELSE '>20 Countries'
                        END as travel_category,
                        COUNT(*) as count,
                        AVG(risk_score) as avg_risk_score
                    FROM {table}
                    WHERE {filter}
                    GROUP BY travel_category
                    ORDER BY 
                        CASE travel_category
                            WHEN 'Never Traveled' THEN 1
                            WHEN '1-3 Countries' THEN 2
                            WHEN '4-10 Countries' THEN 3
                            WHEN '11-20 Countries' THEN 4
                            ELSE 5
                        END
                """,
                visualization="column",
                priority=7
            ),
            
            # Location Analysis
            "work_home_comparison": ExtendedStatistic(
                id="work_home_comparison",
                name="Work vs Home Location Pattern",
                description="Analysis of work and home location patterns",
                category="location",
                sql_template="""
                    SELECT 
                        CASE 
                            WHEN home_location IS NULL AND work_location IS NULL THEN 'No Location Data'
                            WHEN home_location = work_location THEN 'Same Location'
                            WHEN home_location IS NOT NULL AND work_location IS NULL THEN 'Home Only'
                            WHEN home_location IS NULL AND work_location IS NOT NULL THEN 'Work Only'
                            ELSE 'Different Locations'
                        END as pattern,
                        COUNT(*) as count,
                        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() as percentage
                    FROM {table}
                    WHERE {filter}
                    GROUP BY pattern
                    ORDER BY count DESC
                """,
                visualization="donut",
                priority=5
            ),
            
            # Previous Nationality
            "nationality_changes": ExtendedStatistic(
                id="nationality_changes",
                name="Nationality Changes",
                description="Profiles that changed nationality",
                category="demographic",
                sql_template="""
                    SELECT 
                        CASE 
                            WHEN previous_nationality_code IS NULL THEN 'No Change'
                            WHEN previous_nationality_code = nationality_code THEN 'Same (Error?)'
                            ELSE 'Changed Nationality'
                        END as change_status,
                        COUNT(*) as count
                    FROM {table}
                    WHERE {filter}
                    GROUP BY change_status
                """,
                visualization="pie",
                priority=4
            ),
            
            # Dwell Duration
            "long_term_residents": ExtendedStatistic(
                id="long_term_residents",
                name="Long-term Residents Analysis",
                description="Residents by duration of stay",
                category="demographic",
                sql_template="""
                    SELECT 
                        dwell_duration_tag,
                        COUNT(*) as count,
                        AVG(risk_score) as avg_risk_score,
                        COUNT(*) FILTER (WHERE has_crime_case = 1) as crime_cases
                    FROM {table}
                    WHERE {filter} AND dwell_duration_tag IS NOT NULL
                    GROUP BY dwell_duration_tag
                    ORDER BY 
                        CASE dwell_duration_tag
                            WHEN 'LESS_THAN_1_YEAR' THEN 1
                            WHEN '1_TO_3_YEARS' THEN 2
                            WHEN '3_TO_5_YEARS' THEN 3
                            WHEN '5_TO_10_YEARS' THEN 4
                            WHEN 'MORE_THAN_10_YEARS' THEN 5
                        END
                """,
                visualization="stacked_bar",
                priority=6
            ),
            
            # Crime Patterns
            "crime_category_breakdown": ExtendedStatistic(
                id="crime_category_breakdown",
                name="Crime Categories Breakdown",
                description="Types of crimes in the filtered population",
                category="crime",
                sql_template="""
                    SELECT 
                        arrayJoin(crime_categories_en) as category,
                        COUNT(*) as count,
                        COUNT(DISTINCT imsi) as unique_profiles
                    FROM {table}
                    WHERE {filter} AND notEmpty(crime_categories_en)
                    GROUP BY category
                    ORDER BY count DESC
                    LIMIT 15
                """,
                visualization="treemap",
                priority=9
            ),
            
            # Drug Analysis
            "substance_risk_correlation": ExtendedStatistic(
                id="substance_risk_correlation",
                name="Substance Risk Correlation",
                description="Correlation between drug dealing and addiction",
                category="substance",
                sql_template="""
                    SELECT 
                        CASE 
                            WHEN drug_dealing_score > 0.7 AND drug_addict_score > 0.7 THEN 'High Risk Both'
                            WHEN drug_dealing_score > 0.7 THEN 'High Dealing Risk Only'
                            WHEN drug_addict_score > 0.7 THEN 'High Addiction Risk Only'
                            WHEN drug_dealing_score > 0.3 OR drug_addict_score > 0.3 THEN 'Medium Risk'
                            ELSE 'Low/No Risk'
                        END as risk_category,
                        COUNT(*) as count,
                        AVG(risk_score) as avg_overall_risk
                    FROM {table}
                    WHERE {filter}
                    GROUP BY risk_category
                    ORDER BY avg_overall_risk DESC
                """,
                visualization="scatter",
                priority=7
            ),
            
            # Employment Analysis
            "top_job_categories": ExtendedStatistic(
                id="top_job_categories",
                name="Top Job Categories",
                description="Most common job titles in filtered population",
                category="employment",
                sql_template="""
                    SELECT 
                        latest_job_title_en as job_title,
                        COUNT(*) as count,
                        AVG(risk_score) as avg_risk_score
                    FROM {table}
                    WHERE {filter} AND latest_job_title_en IS NOT NULL
                    GROUP BY latest_job_title_en
                    ORDER BY count DESC
                    LIMIT 20
                """,
                visualization="word_cloud",
                priority=3
            ),
            
            # Application Usage Patterns
            "social_media_apps": ExtendedStatistic(
                id="social_media_apps",
                name="Social Media Usage",
                description="Popular social media applications",
                category="digital",
                sql_template="""
                    SELECT 
                        arrayJoin(arrayFilter(
                            app -> app IN ['WhatsApp', 'Facebook', 'Instagram', 'Twitter', 
                                          'Snapchat', 'TikTok', 'Telegram', 'WeChat'],
                            applications_used
                        )) as app,
                        COUNT(*) as user_count
                    FROM {table}
                    WHERE {filter} AND notEmpty(applications_used)
                    GROUP BY app
                    ORDER BY user_count DESC
                """,
                visualization="bubble_chart",
                priority=5
            ),
            
            # Special Status
            "diplomat_analysis": ExtendedStatistic(
                id="diplomat_analysis",
                name="Diplomatic Status Analysis",
                description="Analysis of diplomatic personnel",
                category="special",
                sql_template="""
                    SELECT 
                        nationality_code,
                        COUNT(*) as diplomat_count,
                        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() as percentage
                    FROM {table}
                    WHERE {filter} AND is_diplomat = 1
                    GROUP BY nationality_code
                    ORDER BY diplomat_count DESC
                    LIMIT 10
                """,
                visualization="pie",
                priority=2
            ),
            
            # Multi-factor Risk
            "combined_risk_indicators": ExtendedStatistic(
                id="combined_risk_indicators",
                name="Combined Risk Indicators",
                description="Profiles with multiple risk factors",
                category="risk",
                sql_template="""
                    SELECT 
                        'High Risk Score' as indicator,
                        countIf(risk_score > 0.7) as count
                    FROM {table} WHERE {filter}
                    UNION ALL
                    SELECT 
                        'Has Crime Case' as indicator,
                        countIf(has_crime_case = 1) as count
                    FROM {table} WHERE {filter}
                    UNION ALL
                    SELECT 
                        'In Prison' as indicator,
                        countIf(is_in_prison = 1) as count
                    FROM {table} WHERE {filter}
                    UNION ALL
                    SELECT 
                        'High Drug Risk' as indicator,
                        countIf(drug_dealing_score > 0.7 OR drug_addict_score > 0.7) as count
                    FROM {table} WHERE {filter}
                    UNION ALL
                    SELECT 
                        'Multiple Risks' as indicator,
                        countIf(
                            risk_score > 0.7 AND 
                            (has_crime_case = 1 OR drug_dealing_score > 0.7)
                        ) as count
                    FROM {table} WHERE {filter}
                """,
                visualization="radar",
                priority=10
            )
        }
    
    async def calculate_extended_statistic(
        self,
        stat_id: str,
        filter_condition: str
    ) -> Dict[str, Any]:
        """Calculate a single extended statistic"""
        stat_def = self.extended_stats.get(stat_id)
        if not stat_def:
            raise ValueError(f"Unknown extended statistic: {stat_id}")
        
        sql = stat_def.sql_template.format(
            table=self.table,
            filter=filter_condition
        )
        
        try:
            # Execute query
            result = await self.profile_service.executor.execute_query(sql)
            
            # Format result
            formatted_data = self._format_result(stat_def, result)
            
            return {
                "id": stat_def.id,
                "name": stat_def.name,
                "description": stat_def.description,
                "category": stat_def.category,
                "visualization": stat_def.visualization,
                "priority": stat_def.priority,
                "data": formatted_data,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate {stat_id}: {e}")
            raise
    
    async def calculate_category_statistics(
        self,
        category: str,
        filter_condition: str
    ) -> List[Dict[str, Any]]:
        """Calculate all statistics for a category"""
        category_stats = [
            stat for stat in self.extended_stats.values()
            if stat.category == category
        ]
        
        # Sort by priority
        category_stats.sort(key=lambda x: x.priority, reverse=True)
        
        # Calculate in parallel
        tasks = [
            self.calculate_extended_statistic(stat.id, filter_condition)
            for stat in category_stats
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out errors
        valid_results = []
        for result in results:
            if not isinstance(result, Exception):
                valid_results.append(result)
            else:
                logger.error(f"Statistic calculation failed: {result}")
        
        return valid_results
    
    async def get_top_insights(
        self,
        filter_condition: str,
        query_context: Dict[str, Any],
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get top insights based on query context"""
        # Determine relevant categories based on context
        query_text = query_context.get("original_query", "").lower()
        categories = []
        
        # Map keywords to categories
        if any(word in query_text for word in ["risk", "score", "danger", "threat"]):
            categories.append("risk")
        if any(word in query_text for word in ["travel", "country", "visit"]):
            categories.append("travel")
        if any(word in query_text for word in ["crime", "prison", "investigation"]):
            categories.append("crime")
        if any(word in query_text for word in ["drug", "substance", "addict", "dealing"]):
            categories.append("substance")
        
        # Default categories if none matched
        if not categories:
            categories = ["risk", "demographic"]
        
        # Get high priority stats from relevant categories
        relevant_stats = []
        for stat in self.extended_stats.values():
            if stat.category in categories:
                relevant_stats.append(stat)
        
        # Sort by priority and take top N
        relevant_stats.sort(key=lambda x: x.priority, reverse=True)
        top_stats = relevant_stats[:limit]
        
        # Calculate them
        tasks = [
            self.calculate_extended_statistic(stat.id, filter_condition)
            for stat in top_stats
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Return valid results
        insights = []
        for result in results:
            if not isinstance(result, Exception):
                insights.append(result)
        
        return insights
    
    def _format_result(self, stat_def: ExtendedStatistic, raw_result: List[Tuple]) -> Any:
        """Format query result based on statistic type"""
        if not raw_result:
            return None
        
        # Handle different result structures
        if stat_def.id == "avg_risk_score":
            row = raw_result[0]
            return {
                "average": round(row[0], 3) if row[0] else 0,
                "percentiles": {
                    "p25": round(row[1], 3) if row[1] else 0,
                    "p50": round(row[2], 3) if row[2] else 0,
                    "p75": round(row[3], 3) if row[3] else 0,
                    "p95": round(row[4], 3) if row[4] else 0
                },
                "total_count": row[5] if len(row) > 5 else 0
            }
        
        elif stat_def.id in ["risk_rules_distribution", "crime_category_breakdown", 
                            "top_job_categories", "social_media_apps"]:
            # Simple name-value pairs
            return [
                {"name": row[0], "value": row[1]}
                for row in raw_result
            ]
        
        elif stat_def.id == "travel_frequency":
            # Category with additional metrics
            return [
                {
                    "category": row[0],
                    "count": row[1],
                    "avg_risk_score": round(row[2], 3) if row[2] else 0
                }
                for row in raw_result
            ]
        
        elif stat_def.id == "combined_risk_indicators":
            # Indicator-value pairs for radar chart
            return [
                {"indicator": row[0], "value": row[1]}
                for row in raw_result
            ]
        
        elif stat_def.id == "substance_risk_correlation":
            # Risk categories with metrics
            return [
                {
                    "category": row[0],
                    "count": row[1],
                    "avg_risk": round(row[2], 3) if row[2] else 0
                }
                for row in raw_result
            ]
        
        else:
            # Generic handling - return as list of dicts
            if len(raw_result[0]) == 2:
                return [{"name": row[0], "value": row[1]} for row in raw_result]
            else:
                return [{"values": list(row)} for row in raw_result]
    
    def get_available_extended_statistics(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all available extended statistics grouped by category"""
        grouped = {}
        
        for stat in self.extended_stats.values():
            if stat.category not in grouped:
                grouped[stat.category] = []
            
            grouped[stat.category].append({
                "id": stat.id,
                "name": stat.name,
                "description": stat.description,
                "visualization": stat.visualization,
                "priority": stat.priority
            })
        
        # Sort by priority within each category
        for category in grouped:
            grouped[category].sort(key=lambda x: x["priority"], reverse=True)
        
        return grouped