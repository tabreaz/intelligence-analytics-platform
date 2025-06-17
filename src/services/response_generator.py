"""
Response Generator Service
Generates intelligent responses from query results and statistics
"""
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import json
import random

from src.core.llm.base_llm import BaseLLMClient
from src.services.base.profile_analytics import ProfileQueryResult, FieldStatistics

logger = logging.getLogger(__name__)


@dataclass
class ResponseConfig:
    """Configuration for response generation"""
    max_profiles_to_show: int = 5
    max_statistics_to_show: int = 5
    include_raw_sql: bool = False
    include_execution_metrics: bool = True
    response_style: str = "analytical"  # analytical, concise, detailed
    markdown_format: bool = True


@dataclass
class GeneratedResponse:
    """Generated response with all components"""
    summary: str
    key_insights: List[str]
    statistics_shown: List[Dict[str, Any]]
    profiles_shown: List[Dict[str, Any]]
    sql_filter: str
    total_matches: int
    execution_time_ms: float
    metadata: Dict[str, Any]


class ResponseGenerator:
    """
    Generates intelligent responses from query results
    Selects relevant statistics and formats for presentation
    """
    
    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        self.llm_client = llm_client
        self.config = ResponseConfig()
        
    async def generate_response(
        self,
        query_result: ProfileQueryResult,
        statistics: Dict[str, Any],
        query_context: Dict[str, Any],
        config: Optional[ResponseConfig] = None
    ) -> GeneratedResponse:
        """
        Generate comprehensive response from query results
        
        Args:
            query_result: Profile query execution result
            statistics: Dictionary of calculated statistics
            query_context: Original query context (query text, category, etc.)
            config: Optional response configuration
        """
        if config:
            self.config = config
            
        # Extract SQL WHERE clause
        sql_filter = self._extract_where_clause(query_result.sql_generated)
        
        # Select top statistics based on relevance
        selected_stats = await self._select_top_statistics(
            statistics, query_context, self.config.max_statistics_to_show
        )
        
        # Select representative profiles
        selected_profiles = self._select_profiles(
            query_result.data, self.config.max_profiles_to_show
        )
        
        # Generate summary and insights
        summary = await self._generate_summary(
            query_result, selected_stats, query_context
        )
        
        key_insights = await self._generate_insights(
            selected_stats, query_result, query_context
        )
        
        # Build metadata
        metadata = {
            "query_id": str(query_result.query_id),
            "session_id": query_result.session_id,
            "query_category": query_context.get("category", "unknown"),
            "original_query": query_context.get("original_query", ""),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if self.config.include_raw_sql:
            metadata["sql_generated"] = query_result.sql_generated
            
        return GeneratedResponse(
            summary=summary,
            key_insights=key_insights,
            statistics_shown=selected_stats,
            profiles_shown=selected_profiles,
            sql_filter=sql_filter,
            total_matches=query_result.result_count,
            execution_time_ms=query_result.execution_time_ms,
            metadata=metadata
        )
    
    async def _select_top_statistics(
        self,
        statistics: Dict[str, Any],
        query_context: Dict[str, Any],
        limit: int
    ) -> List[Dict[str, Any]]:
        """Select most relevant statistics based on query context"""
        
        # Get all available statistics with metadata
        all_stats = []
        
        # Process field distributions
        if "field_distributions" in statistics:
            for field_name, field_stats in statistics["field_distributions"].items():
                all_stats.append({
                    "type": "field_distribution",
                    "name": f"{field_name} Distribution",
                    "field": field_name,
                    "data": field_stats,
                    "priority": self._calculate_priority(field_name, query_context)
                })
        
        # Process extended statistics
        if "extended_statistics" in statistics:
            for stat in statistics["extended_statistics"]:
                all_stats.append({
                    "type": "extended",
                    "name": stat["name"],
                    "id": stat["id"],
                    "data": stat["data"],
                    "visualization": stat.get("visualization", "table"),
                    "priority": stat.get("priority", 0) + self._calculate_context_bonus(stat, query_context)
                })
        
        # Sort by priority and select top N
        all_stats.sort(key=lambda x: x["priority"], reverse=True)
        
        # If we have many stats, try to get a diverse mix
        selected = []
        if len(all_stats) > limit * 2:
            # Get at least one from each category
            categories_seen = set()
            for stat in all_stats:
                category = stat.get("data", {}).get("category", stat["type"])
                if category not in categories_seen and len(selected) < limit:
                    selected.append(stat)
                    categories_seen.add(category)
            
            # Fill remaining slots with highest priority
            for stat in all_stats:
                if stat not in selected and len(selected) < limit:
                    selected.append(stat)
        else:
            selected = all_stats[:limit]
        
        # Format the selected statistics before returning
        formatted_stats = []
        for stat in selected:
            formatted_stat = self._format_statistic(stat)
            if formatted_stat:
                formatted_stats.append(formatted_stat)
        
        return formatted_stats
    
    def _format_statistic(self, stat: Dict[str, Any]) -> Dict[str, Any]:
        """Format a statistic based on its type and field characteristics"""
        stat_type = stat.get("type")
        field_name = stat.get("field", "")
        data = stat.get("data", {})
        
        # Determine the formatting based on field type
        if stat_type == "field_distribution":
            return self._format_field_distribution(field_name, data, stat.get("name"))
        elif stat_type == "extended":
            return self._format_extended_statistic(stat)
        else:
            # Default formatting
            return {
                "name": stat.get("name", "Unknown Statistic"),
                "type": stat_type,
                "formatted_data": data
            }
    
    def _format_field_distribution(self, field_name: str, data: Dict[str, Any], display_name: str) -> Dict[str, Any]:
        """Format field distribution based on field type"""
        
        # Extract basic info
        total_count = data.get("total_count", 0)
        unique_count = data.get("unique_count", 0)
        distribution = data.get("distribution", {})
        percentiles = data.get("percentiles", {})
        metadata = data.get("metadata", {})
        
        formatted_result = {
            "name": display_name,
            "field": field_name,
            "type": "field_distribution",
            "total_count": total_count,
            "unique_count": unique_count
        }
        
        # Format based on field type
        if field_name == "risk_score" or field_name.endswith("_score"):
            # Format risk/score fields
            formatted_result["formatted_data"] = {
                "average": round(metadata.get("avg", 0), 3),
                "min": round(metadata.get("min", 0), 3),
                "max": round(metadata.get("max", 0), 3),
                "percentiles": {
                    "25th": round(percentiles.get("0.25", 0), 3),
                    "median": round(percentiles.get("0.5", 0), 3),
                    "75th": round(percentiles.get("0.75", 0), 3),
                    "95th": round(percentiles.get("0.95", 0), 3)
                },
                "risk_levels": self._categorize_risk_distribution(distribution)
            }
            formatted_result["visualization"] = "gauge"
            
        elif field_name in ["nationality_code", "nationality", "home_city", "work_city"]:
            # Format location/nationality fields
            top_values = sorted(distribution.items(), key=lambda x: x[1], reverse=True)[:5]
            formatted_result["formatted_data"] = {
                "top_values": [
                    {
                        "value": value,
                        "count": count,
                        "percentage": round((count / total_count) * 100, 1)
                    }
                    for value, count in top_values
                ],
                "others_count": sum(count for _, count in distribution.items()) - sum(count for _, count in top_values)
            }
            formatted_result["visualization"] = "bar_chart"
            
        elif field_name in ["gender", "gender_en", "marital_status"]:
            # Format demographic categorical fields
            formatted_result["formatted_data"] = {
                "distribution": [
                    {
                        "category": category,
                        "count": count,
                        "percentage": round((count / total_count) * 100, 1)
                    }
                    for category, count in distribution.items()
                ]
            }
            formatted_result["visualization"] = "pie_chart"
            
        elif field_name == "age" or field_name == "age_group":
            # Format age fields
            if percentiles:
                formatted_result["formatted_data"] = {
                    "average": round(metadata.get("avg", 0), 1),
                    "median": round(percentiles.get("0.5", 0), 1),
                    "range": {
                        "min": round(metadata.get("min", 0)),
                        "max": round(metadata.get("max", 0))
                    },
                    "age_groups": self._create_age_groups(distribution)
                }
            else:
                # Categorical age groups
                formatted_result["formatted_data"] = {
                    "groups": [
                        {
                            "group": group,
                            "count": count,
                            "percentage": round((count / total_count) * 100, 1)
                        }
                        for group, count in sorted(distribution.items())
                    ]
                }
            formatted_result["visualization"] = "histogram"
            
        elif field_name == "has_crime_case" or field_name.startswith("is_"):
            # Format boolean fields
            true_count = distribution.get("true", 0) + distribution.get("1", 0) + distribution.get(1, 0)
            false_count = distribution.get("false", 0) + distribution.get("0", 0) + distribution.get(0, 0)
            
            formatted_result["formatted_data"] = {
                "yes": {
                    "count": true_count,
                    "percentage": round((true_count / total_count) * 100, 1)
                },
                "no": {
                    "count": false_count,
                    "percentage": round((false_count / total_count) * 100, 1)
                }
            }
            formatted_result["visualization"] = "donut_chart"
            
        elif field_name in ["applications_used", "travelled_country_codes", "crime_categories"]:
            # Format array fields
            # These need special handling as they're arrays
            formatted_result["formatted_data"] = {
                "most_common": [
                    {
                        "value": value,
                        "count": count,
                        "percentage": round((count / total_count) * 100, 1)
                    }
                    for value, count in sorted(distribution.items(), key=lambda x: x[1], reverse=True)[:10]
                ],
                "total_unique": unique_count
            }
            formatted_result["visualization"] = "horizontal_bar"
            
        else:
            # Default formatting for unknown fields
            top_values = sorted(distribution.items(), key=lambda x: x[1], reverse=True)[:5]
            formatted_result["formatted_data"] = {
                "values": [
                    {
                        "value": str(value),
                        "count": count,
                        "percentage": round((count / total_count) * 100, 1)
                    }
                    for value, count in top_values
                ]
            }
            formatted_result["visualization"] = "table"
        
        return formatted_result
    
    def _categorize_risk_distribution(self, distribution: Dict) -> Dict[str, Any]:
        """Categorize risk score distribution into risk levels"""
        low_risk = 0
        medium_risk = 0
        high_risk = 0
        
        for range_str, count in distribution.items():
            if isinstance(range_str, str) and "-" in range_str:
                try:
                    min_val = float(range_str.split("-")[0])
                    if min_val < 0.33:
                        low_risk += count
                    elif min_val < 0.67:
                        medium_risk += count
                    else:
                        high_risk += count
                except:
                    pass
        
        total = low_risk + medium_risk + high_risk
        return {
            "low": {
                "count": low_risk,
                "percentage": round((low_risk / total) * 100, 1) if total > 0 else 0
            },
            "medium": {
                "count": medium_risk,
                "percentage": round((medium_risk / total) * 100, 1) if total > 0 else 0
            },
            "high": {
                "count": high_risk,
                "percentage": round((high_risk / total) * 100, 1) if total > 0 else 0
            }
        }
    
    def _create_age_groups(self, distribution: Dict) -> List[Dict[str, Any]]:
        """Create age groups from age distribution"""
        age_groups = {
            "0-18": 0,
            "19-25": 0,
            "26-35": 0,
            "36-45": 0,
            "46-60": 0,
            "60+": 0
        }
        
        for age_str, count in distribution.items():
            try:
                age = int(age_str)
                if age <= 18:
                    age_groups["0-18"] += count
                elif age <= 25:
                    age_groups["19-25"] += count
                elif age <= 35:
                    age_groups["26-35"] += count
                elif age <= 45:
                    age_groups["36-45"] += count
                elif age <= 60:
                    age_groups["46-60"] += count
                else:
                    age_groups["60+"] += count
            except:
                pass
        
        return [
            {
                "group": group,
                "count": count,
                "percentage": round((count / sum(age_groups.values())) * 100, 1) if sum(age_groups.values()) > 0 else 0
            }
            for group, count in age_groups.items()
            if count > 0
        ]
    
    def _format_extended_statistic(self, stat: Dict[str, Any]) -> Dict[str, Any]:
        """Format extended statistics"""
        return {
            "name": stat.get("name"),
            "id": stat.get("id"),
            "type": "extended",
            "category": stat.get("category", "general"),
            "visualization": stat.get("visualization", "custom"),
            "formatted_data": stat.get("data", {}),
            "description": stat.get("description", "")
        }
    
    def _calculate_priority(self, field_name: str, query_context: Dict[str, Any]) -> int:
        """Calculate priority for a field based on query context"""
        query_text = query_context.get("original_query", "").lower()
        priority = 0
        
        # Field-specific priorities
        field_priorities = {
            "risk_score": 10,
            "nationality_code": 8,
            "gender_en": 6,
            "age_group": 6,
            "residency_status": 7,
            "has_crime_case": 9,
            "travelled_country_codes": 5,
            "applications_used": 4
        }
        
        priority += field_priorities.get(field_name, 3)
        
        # Boost if field is mentioned in query
        if field_name.replace("_", " ") in query_text:
            priority += 10
        
        # Category-based boosts
        category = query_context.get("category", "")
        if category == "risk_analysis" and "risk" in field_name:
            priority += 5
        elif category == "demographic" and field_name in ["nationality_code", "gender_en", "age_group"]:
            priority += 5
        elif category == "travel" and "travel" in field_name:
            priority += 5
        
        return priority
    
    def _calculate_context_bonus(self, stat: Dict[str, Any], query_context: Dict[str, Any]) -> int:
        """Calculate context-based priority bonus for extended statistics"""
        query_text = query_context.get("original_query", "").lower()
        bonus = 0
        
        # Check if stat name or category matches query
        stat_name = stat.get("name", "").lower()
        stat_category = stat.get("category", "").lower()
        
        for keyword in query_text.split():
            if len(keyword) > 3:  # Skip short words
                if keyword in stat_name or keyword in stat_category:
                    bonus += 5
        
        return bonus
    
    def _select_profiles(self, profiles: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        """Select representative profiles to show"""
        if not profiles:
            return []
        
        selected = []
        
        # Always include highest risk if available
        if any(p.get("risk_score", 0) for p in profiles):
            sorted_by_risk = sorted(profiles, key=lambda x: x.get("risk_score", 0), reverse=True)
            selected.append(self._clean_profile(sorted_by_risk[0]))
        
        # Add diverse profiles
        remaining = limit - len(selected)
        if remaining > 0 and len(profiles) > len(selected):
            # Try to get diverse nationalities/demographics
            seen_nationalities = {selected[0].get("nationality_code")} if selected else set()
            
            for profile in profiles:
                if len(selected) >= limit:
                    break
                    
                nationality = profile.get("nationality_code")
                if nationality not in seen_nationalities:
                    selected.append(self._clean_profile(profile))
                    seen_nationalities.add(nationality)
            
            # Fill remaining with random selection
            while len(selected) < limit and len(selected) < len(profiles):
                profile = random.choice(profiles)
                cleaned = self._clean_profile(profile)
                if cleaned not in selected:
                    selected.append(cleaned)
        
        return selected
    
    def _clean_profile(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Clean profile data for display"""
        # Key fields to always include
        key_fields = [
            "imsi", "phone_no", "fullname_en", "nationality_code",
            "gender_en", "age", "risk_score", "home_city",
            "has_crime_case", "residency_status"
        ]
        
        cleaned = {}
        for field in key_fields:
            if field in profile and profile[field] is not None:
                cleaned[field] = profile[field]
        
        return cleaned
    
    async def _generate_summary(
        self,
        query_result: ProfileQueryResult,
        selected_stats: List[Dict[str, Any]],
        query_context: Dict[str, Any]
    ) -> str:
        """Generate executive summary of results"""
        
        if self.llm_client:
            # Use LLM for intelligent summary
            prompt = self._build_summary_prompt(query_result, selected_stats, query_context)
            summary = await self.llm_client.generate(
                system_prompt="You are an intelligence analyst generating concise summaries.",
                user_prompt=prompt
            )
            return summary.strip()
        else:
            # Fallback to template-based summary
            return self._generate_template_summary(query_result, selected_stats, query_context)
    
    async def _generate_insights(
        self,
        selected_stats: List[Dict[str, Any]],
        query_result: ProfileQueryResult,
        query_context: Dict[str, Any]
    ) -> List[str]:
        """Generate key insights from statistics"""
        insights = []
        
        # Analyze each statistic for insights
        for stat in selected_stats:
            insight = self._extract_insight_from_stat(stat, query_result.result_count)
            if insight:
                insights.append(insight)
        
        # Add overall insights
        if query_result.result_count > 0:
            if query_result.result_count < 100:
                insights.append(f"The query returned a focused set of {query_result.result_count} profiles")
            elif query_result.result_count > 10000:
                insights.append(f"Large dataset identified with {query_result.result_count:,} matching profiles")
        
        return insights[:5]  # Limit to 5 key insights
    
    def _extract_insight_from_stat(self, stat: Dict[str, Any], total_count: int) -> Optional[str]:
        """Extract insight from a single statistic"""
        stat_type = stat.get("type")
        stat_name = stat.get("name", "")
        data = stat.get("data", {})
        
        if stat_type == "field_distribution":
            # Analyze distribution
            distribution = data.get("distribution", {})
            field_total = data.get("total_count", total_count)
            if distribution:
                top_value = max(distribution.items(), key=lambda x: x[1])
                # Use the field's total count, not the query total count
                percentage = (top_value[1] / field_total * 100) if field_total > 0 else 0
                if percentage > 50:
                    return f"Majority ({percentage:.1f}%) have {stat['field']} = {top_value[0]}"
                elif percentage > 20:
                    return f"Significant portion ({percentage:.1f}%) have {stat['field']} = {top_value[0]}"
        
        elif stat_type == "extended" and isinstance(data, dict):
            # Handle specific extended statistics
            stat_id = stat.get("id", "")
            
            if stat_id == "avg_risk_score" and "average" in data:
                avg = data["average"]
                if avg > 0.7:
                    return f"Very high average risk score of {avg:.3f}"
                elif avg > 0.5:
                    return f"Elevated average risk score of {avg:.3f}"
            
            elif stat_id == "risk_rules_distribution" and isinstance(data, list):
                if len(data) > 0:
                    top_rule = data[0]
                    return f"Most common risk rule: {top_rule.get('name', 'Unknown')} ({top_rule.get('value', 0)} cases)"
            
            elif stat_id == "travel_frequency" and isinstance(data, list):
                frequent_travelers = sum(item.get("count", 0) for item in data if "10" in item.get("category", "") or "20" in item.get("category", ""))
                if frequent_travelers > total_count * 0.2:
                    return "High proportion of frequent international travelers"
        
        return None
    
    def _build_summary_prompt(
        self,
        query_result: ProfileQueryResult,
        selected_stats: List[Dict[str, Any]],
        query_context: Dict[str, Any]
    ) -> str:
        """Build prompt for LLM summary generation"""
        prompt = f"""Generate a concise analytical summary for the following query results:

Original Query: {query_context.get('original_query', 'N/A')}
Total Matches: {query_result.result_count:,} profiles
Query Category: {query_context.get('category', 'general')}

Key Statistics:
"""
        for stat in selected_stats[:3]:  # Include top 3 stats in prompt
            prompt += f"\n- {stat['name']}: "
            if isinstance(stat.get('data'), dict):
                prompt += json.dumps(stat['data'], indent=2)[:200] + "..."
            else:
                prompt += str(stat.get('data'))[:200] + "..."
        
        prompt += """

Provide a 2-3 sentence analytical summary highlighting the most important findings. Focus on actionable insights and patterns."""
        
        return prompt
    
    def _generate_template_summary(
        self,
        query_result: ProfileQueryResult,
        selected_stats: List[Dict[str, Any]],
        query_context: Dict[str, Any]
    ) -> str:
        """Generate template-based summary without LLM"""
        category = query_context.get("category", "general")
        total = query_result.result_count
        
        templates = {
            "risk_analysis": f"Risk analysis identified {total:,} profiles matching the criteria. ",
            "demographic": f"Demographic analysis found {total:,} profiles with specified characteristics. ",
            "travel": f"Travel pattern analysis revealed {total:,} profiles meeting the travel criteria. ",
            "general": f"Query returned {total:,} profiles based on the specified filters. "
        }
        
        summary = templates.get(category, templates["general"])
        
        # Add execution time if configured
        if self.config.include_execution_metrics:
            summary += f"Query executed in {query_result.execution_time_ms:.0f}ms."
        
        return summary
    
    def _extract_where_clause(self, sql: str) -> str:
        """Extract WHERE clause from SQL query"""
        sql_upper = sql.upper()
        where_start = sql_upper.find("WHERE")
        if where_start == -1:
            return ""
        
        # Find the end of WHERE clause (before ORDER BY, GROUP BY, LIMIT, etc.)
        end_keywords = ["ORDER BY", "GROUP BY", "LIMIT", "OFFSET", "HAVING", "UNION", "SETTINGS"]
        where_end = len(sql)
        
        for keyword in end_keywords:
            pos = sql_upper.find(keyword, where_start)
            if pos != -1 and pos < where_end:
                where_end = pos
        
        where_clause = sql[where_start + 5:where_end].strip()
        return where_clause
    
    def format_markdown_response(self, response: GeneratedResponse) -> str:
        """Format response as markdown for display"""
        md_lines = []
        
        # Summary
        md_lines.append(f"## Summary\n\n{response.summary}\n")
        
        # Key Insights
        if response.key_insights:
            md_lines.append("## Key Insights\n")
            for insight in response.key_insights:
                md_lines.append(f"- {insight}")
            md_lines.append("")
        
        # Statistics
        if response.statistics_shown:
            md_lines.append("## Statistics\n")
            for stat in response.statistics_shown:
                md_lines.append(f"### {stat['name']}")
                if stat.get('visualization'):
                    md_lines.append(f"*Recommended visualization: {stat['visualization']}*")
                
                # Format data based on type
                data = stat.get('data', {})
                if isinstance(data, list):
                    for item in data[:5]:  # Show top 5
                        if isinstance(item, dict):
                            md_lines.append(f"- {item}")
                        else:
                            md_lines.append(f"- {item}")
                elif isinstance(data, dict):
                    for key, value in list(data.items())[:5]:
                        md_lines.append(f"- **{key}**: {value}")
                
                md_lines.append("")
        
        # Sample Profiles
        if response.profiles_shown:
            md_lines.append("## Sample Profiles\n")
            for i, profile in enumerate(response.profiles_shown, 1):
                md_lines.append(f"**Profile {i}:**")
                for key, value in profile.items():
                    md_lines.append(f"- {key}: {value}")
                md_lines.append("")
        
        # Metadata
        md_lines.append("---")
        md_lines.append(f"*Total matches: {response.total_matches:,} | ")
        md_lines.append(f"Execution time: {response.execution_time_ms:.0f}ms*")
        
        return "\n".join(md_lines)