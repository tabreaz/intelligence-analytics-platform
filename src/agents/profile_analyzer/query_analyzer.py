# src/agents/profile_analyzer/query_analyzer.py
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Any

from src.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class QueryAnalysis:
    """Analysis result from LLM about the query and SQL"""
    query_type: str
    analysis_focus: List[str]
    key_metrics: List[str]
    grouping_fields: List[str]
    filter_conditions: List[str]
    expected_insights: List[str]
    summary_structure: Dict[str, Any]
    comparison_groups: List[str] = field(default_factory=list)  # For comparison queries


@dataclass
class DataStatistics:
    """Statistics extracted from data without exposing raw values"""
    total_records: int
    numeric_fields: Dict[str, Dict[str, float]]  # field -> {min, max, avg, count}
    categorical_fields: Dict[str, Dict[str, int]]  # field -> {value -> count}
    null_counts: Dict[str, int]
    patterns: List[str]
    comparison_metrics: Dict[str, Any]  # For comparison queries


class ColumnNameParser:
    """Intelligent parser for SQL column aliases"""

    @staticmethod
    def extract_column_info(column_name: str) -> Dict[str, Any]:
        """
        Extract meaningful information from any SQL column name/alias
        Returns: {
            'original': original column name,
            'base_metric': extracted metric name,
            'group_identifier': group/category if found,
            'operation': SQL operation if found (AVG, SUM, etc),
            'display_name': human-readable name
        }
        """
        info = {
            'original': column_name,
            'base_metric': column_name,
            'group_identifier': None,
            'operation': None,
            'display_name': column_name
        }

        # Extract SQL operations
        operations = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'COUNTIf']
        for op in operations:
            if op + '(' in column_name.upper():
                info['operation'] = op
                # Extract the field name from operation
                match = re.search(f'{op}\s*\(\s*([^)]+)\s*\)', column_name, re.IGNORECASE)
                if match:
                    info['base_metric'] = match.group(1).strip()
                break

        # Handle AS aliases
        if ' AS ' in column_name:
            parts = column_name.split(' AS ')
            info['base_metric'] = parts[0].strip()
            info['display_name'] = parts[-1].strip()

        # Clean up the base metric
        base = info['base_metric'].lower()

        # Try to extract group identifiers (for comparisons)
        # Pattern 1: metric_GROUP (e.g., total_users_afg)
        suffix_match = re.search(r'(.+?)_([a-z]{2,5})$', base)
        if suffix_match:
            info['base_metric'] = suffix_match.group(1)
            info['group_identifier'] = suffix_match.group(2).upper()

        # Pattern 2: GROUP_metric (e.g., afg_total_users)
        prefix_match = re.search(r'^([a-z]{2,5})_(.+)', base)
        if prefix_match and not info['group_identifier']:
            info['group_identifier'] = prefix_match.group(1).upper()
            info['base_metric'] = prefix_match.group(2)

        # Clean up display name
        display = info['display_name']
        display = display.replace('_', ' ')
        display = display.title()
        info['display_name'] = display

        return info


class LLMQueryAnalyzer:
    """Uses LLM to analyze queries and SQL for intelligent summarization"""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    async def analyze_query_and_sql(self, user_query: str, sql_query: str) -> QueryAnalysis:
        """Analyze the user query and generated SQL to understand intent and structure"""

        system_prompt = """
        You are an expert data analyst. Analyze the user's query and the generated SQL to understand:
        1. The type of analysis being requested
        2. What insights the user is looking for
        3. How to best summarize the results

        Pay special attention to:
        - Comparison queries (vs, versus, compare, between different groups)
        - Aggregate summaries (single row with multiple statistics)
        - Individual lookups (single profile details)
        - Group analysis (demographics, distributions)
        - always alias the fields appropriately don't use col_.

        For comparisons, identify the groups being compared by looking at:
        - CTE names in the SQL
        - Column naming patterns (e.g., metric_groupname)
        - WHERE clauses filtering different groups

        Return a JSON object with this structure:
        {
            "query_type": "one of: comparison, aggregate_summary, individual_profile, group_analysis, trend_analysis, risk_assessment, demographic_breakdown, statistical_analysis",
            "comparison_groups": ["list of groups if comparison, e.g., ['SYR', 'IRN'] or ['Denmark', 'France']"],
            "analysis_focus": ["list of main topics like: drug_dealing, risk_scores, demographics, etc."],
            "key_metrics": ["list of important metrics/scores being analyzed"],
            "grouping_fields": ["fields used for grouping/categorization"],
            "filter_conditions": ["main filtering criteria applied"],
            "expected_insights": ["what insights the user likely wants"],
            "summary_structure": {
                "main_sections": ["ordered list of sections to include"],
                "highlight_patterns": ["patterns to look for in results"],
                "visualization_hints": ["table", "distribution", "statistics", "comparison_table"]
            }
        }
        """

        user_prompt = f"""
        Analyze this query and SQL:

        User Query: "{user_query}"

        Generated SQL:
        ```sql
        {sql_query}
        ```

        Provide the analysis in the JSON format specified.
        """

        try:
            response = await self.llm_client.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )

            # Parse the JSON response
            analysis_data = json.loads(response)

            # Add comparison_groups to QueryAnalysis
            return QueryAnalysis(
                query_type=analysis_data.get('query_type', 'general'),
                analysis_focus=analysis_data.get('analysis_focus', []),
                key_metrics=analysis_data.get('key_metrics', []),
                grouping_fields=analysis_data.get('grouping_fields', []),
                filter_conditions=analysis_data.get('filter_conditions', []),
                expected_insights=analysis_data.get('expected_insights', []),
                summary_structure=analysis_data.get('summary_structure', {}),
                comparison_groups=analysis_data.get('comparison_groups', [])
            )

        except Exception as e:
            logger.error(f"Failed to analyze query: {e}")
            # Return default analysis
            return QueryAnalysis(
                query_type='general',
                analysis_focus=[],
                key_metrics=[],
                grouping_fields=[],
                filter_conditions=[],
                expected_insights=[],
                summary_structure={'main_sections': ['overview', 'data', 'insights']},
                comparison_groups=[]
            )

    async def generate_insights_from_statistics(self, analysis: QueryAnalysis,
                                                statistics: DataStatistics) -> str:
        """Generate insights based on statistics, not raw data"""

        system_prompt = """
        You are a data analyst providing insights based on statistical summaries.
        You will receive statistics and patterns, NOT raw data.

        Generate meaningful insights and recommendations based on:
        - Statistical distributions
        - Patterns and trends
        - Comparison differences (if applicable)

        Be specific about what the statistics reveal, but don't make up data values.
        """

        # Prepare statistics summary for LLM
        stats_summary = {
            "query_type": analysis.query_type,
            "total_records": statistics.total_records,
            "numeric_summaries": {
                field: {
                    "range": f"{stats['min']:.2f} to {stats['max']:.2f}",
                    "average": f"{stats['avg']:.2f}",
                    "has_outliers": stats['max'] > stats['avg'] * 3
                }
                for field, stats in statistics.numeric_fields.items()
            },
            "categorical_distributions": {
                field: {
                    "unique_values": len(counts),
                    "dominant_value_percentage": max(counts.values()) / sum(counts.values()) * 100 if counts else 0
                }
                for field, counts in statistics.categorical_fields.items()
            },
            "data_quality": {
                field: f"{count}/{statistics.total_records} null values"
                for field, count in statistics.null_counts.items() if count > 0
            },
            "patterns": statistics.patterns
        }

        if analysis.query_type == 'comparison' and statistics.comparison_metrics:
            stats_summary["comparison_insights"] = statistics.comparison_metrics

        user_prompt = f"""
        Generate insights for this analysis:

        Query Focus: {', '.join(analysis.analysis_focus)}
        Expected Insights: {', '.join(analysis.expected_insights)}

        Statistics Summary:
        {json.dumps(stats_summary, indent=2)}

        Provide 3-5 key insights based on these statistics.
        """

        try:
            insights = await self.llm_client.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
            return insights
        except Exception as e:
            logger.error(f"Failed to generate insights: {e}")
            return "Unable to generate automated insights."


class IntelligentResultSummarizer:
    """Summarizer that uses LLM analysis to create intelligent summaries"""

    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.query_analyzer = LLMQueryAnalyzer(llm_client)

    async def generate_summary_with_analysis(self, user_query: str, sql_query: str,
                                             results: List[Dict],
                                             pre_computed_analysis: QueryAnalysis) -> str:
        """Generate summary using pre-computed analysis (for parallel execution)"""

        analysis = pre_computed_analysis
        logger.info(f"Using pre-computed analysis: {analysis.query_type}, focus: {analysis.analysis_focus}")

        # Handle different query types
        if analysis.query_type == 'comparison' and len(results) == 1:
            # Special handling for comparison queries
            return await self._generate_comparison_summary_local(user_query, results[0], analysis)

        # Extract statistics from results
        statistics = self._extract_statistics(results, analysis)

        # Generate insights from statistics (not raw data)
        llm_insights = await self.query_analyzer.generate_insights_from_statistics(analysis, statistics)

        # Build the summary locally with actual data
        summary_sections = []

        # Add header
        summary_sections.append(self._generate_header(user_query, results, analysis))

        # Add data sections based on query type
        if analysis.query_type in ['group_analysis', 'demographic_breakdown']:
            summary_sections.append(self._build_distribution_section(results, statistics, analysis))
        elif analysis.query_type == 'individual_profile':
            summary_sections.append(self._build_profile_section(results[0] if results else {}))
        else:
            summary_sections.append(self._build_statistics_section(statistics, analysis))

        # Add table for small result sets
        if len(results) <= 20 and len(results) > 0:
            summary_sections.append(self._build_table_section(results, analysis))

        # Add LLM-generated insights
        if llm_insights:
            summary_sections.append(f"## 💡 Key Insights\n\n{llm_insights}")

        return "\n\n".join(summary_sections)

    async def generate_summary(self, user_query: str, sql_query: str,
                               results: List[Dict]) -> str:
        """Generate intelligent summary using LLM analysis"""

        # Step 1: Analyze query and SQL
        analysis = await self.query_analyzer.analyze_query_and_sql(user_query, sql_query)

        # Step 2: Use the analysis to generate summary
        return await self.generate_summary_with_analysis(user_query, sql_query, results, analysis)

    async def _generate_comparison_summary_local(self, user_query: str,
                                                 result: Dict, analysis: QueryAnalysis) -> str:
        """Generate comparison summary WITHOUT sending data to LLM"""

        # Extract comparison statistics locally
        comparison_stats = self._extract_comparison_statistics(result, analysis)

        # Generate insights based on statistics only
        llm_insights = await self._get_comparison_insights(
            user_query, analysis, comparison_stats
        )

        # Build the summary with actual data
        sections = []

        # Header
        sections.append(f"# 🔄 Comparison Analysis\n")
        sections.append(f"**Query:** {user_query}\n")
        sections.append(f"**Comparing:** {' vs '.join(analysis.comparison_groups)}\n")

        # Overview statistics
        sections.append(self._build_comparison_overview(result, analysis, comparison_stats))

        # Comparison table
        sections.append(self._build_comparison_table(result, analysis))

        # Key differences
        sections.append(self._build_comparison_differences(comparison_stats))

        # LLM insights (generated without seeing raw data)
        sections.append(f"## 📊 Analysis & Insights\n\n{llm_insights}")

        return "\n".join(sections)

    async def _get_comparison_insights(self, user_query: str, analysis: QueryAnalysis,
                                       comparison_stats: Dict) -> str:
        """Get LLM insights based on statistics only"""

        system_prompt = """
        You are a data analyst providing insights for a comparison analysis.
        You'll receive statistical summaries and patterns, NOT raw data.

        Provide insights about:
        1. Which group shows stronger metrics
        2. Key differences between groups
        3. Recommendations based on the patterns
        """

        user_prompt = f"""
        Generate insights for this comparison:

        Query: "{user_query}"
        Groups compared: {' vs '.join(analysis.comparison_groups)}
        Focus areas: {', '.join(analysis.analysis_focus)}

        Comparison Statistics:
        {json.dumps(comparison_stats, indent=2)}

        Provide 3-5 specific insights and recommendations.
        """

        try:
            insights = await self.llm_client.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
            return insights
        except Exception as e:
            logger.error(f"Failed to generate comparison insights: {e}")
            return self._generate_fallback_insights(comparison_stats)

    def _extract_statistics(self, results: List[Dict], analysis: QueryAnalysis) -> DataStatistics:
        """Extract statistics from results without exposing raw data"""

        if not results:
            return DataStatistics(
                total_records=0,
                numeric_fields={},
                categorical_fields={},
                null_counts={},
                patterns=[],
                comparison_metrics={}
            )

        stats = DataStatistics(
            total_records=len(results),
            numeric_fields={},
            categorical_fields=defaultdict(lambda: defaultdict(int)),
            null_counts=defaultdict(int),
            patterns=[],
            comparison_metrics={}
        )

        # Analyze first result to understand structure
        sample = results[0]

        # Process all results
        for row in results:
            for field, value in row.items():
                if value is None:
                    stats.null_counts[field] += 1
                elif isinstance(value, (int, float)):
                    if field not in stats.numeric_fields:
                        stats.numeric_fields[field] = {
                            'min': float('inf'),
                            'max': float('-inf'),
                            'sum': 0,
                            'count': 0
                        }
                    stats.numeric_fields[field]['min'] = min(stats.numeric_fields[field]['min'], value)
                    stats.numeric_fields[field]['max'] = max(stats.numeric_fields[field]['max'], value)
                    stats.numeric_fields[field]['sum'] += value
                    stats.numeric_fields[field]['count'] += 1
                else:
                    stats.categorical_fields[field][str(value)] += 1

        # Calculate averages
        for field, field_stats in stats.numeric_fields.items():
            if field_stats['count'] > 0:
                field_stats['avg'] = field_stats['sum'] / field_stats['count']
            else:
                field_stats['avg'] = 0

        # Identify patterns
        stats.patterns = self._identify_patterns(stats, analysis)

        return stats

    def _extract_comparison_statistics(self, result: Dict, analysis: QueryAnalysis) -> Dict:
        """Extract comparison statistics from a single-row comparison result"""

        stats = {
            "groups": analysis.comparison_groups,
            "metrics_compared": {},
            "differences": {},
            "totals": {},
            "debug_info": {}  # For troubleshooting
        }

        # Use the new parser
        parser = ColumnNameParser()

        # Group metrics by base name and comparison group
        metrics_by_base = defaultdict(dict)
        unmapped_columns = []  # Track columns we couldn't map

        for field, value in result.items():
            col_info = parser.extract_column_info(field)

            # Try to match with comparison groups
            matched = False

            # First try: use extracted group identifier
            if col_info['group_identifier']:
                for group in analysis.comparison_groups:
                    if col_info['group_identifier'].upper() == group.upper():
                        metrics_by_base[col_info['base_metric']][group] = value
                        matched = True

                        # Track totals
                        if 'total' in col_info['base_metric'].lower():
                            stats['totals'][group] = value
                        break

            # Second try: fuzzy matching on the original field name
            if not matched:
                field_lower = field.lower()
                for group in analysis.comparison_groups:
                    group_lower = group.lower()

                    # Check various patterns
                    patterns = [
                        f'_{group_lower}',  # suffix
                        f'{group_lower}_',  # prefix
                        f'_{group_lower}_',  # middle
                        f' {group_lower}',  # space separated
                        group_lower  # just the identifier
                    ]

                    for pattern in patterns:
                        if pattern in field_lower:
                            # Extract base metric by removing the group identifier
                            base_metric = field_lower
                            for p in patterns:
                                base_metric = base_metric.replace(p, '')

                            base_metric = base_metric.strip('_ ')
                            metrics_by_base[base_metric][group] = value
                            matched = True

                            # Track totals
                            if 'total' in base_metric:
                                stats['totals'][group] = value
                            break

                    if matched:
                        break

            if not matched:
                unmapped_columns.append((field, value))

        # Log unmapped columns for debugging
        if unmapped_columns:
            stats['debug_info']['unmapped'] = unmapped_columns
            logger.debug(f"Unmapped columns in comparison: {unmapped_columns}")

        # Calculate differences for numeric metrics
        for metric, group_values in metrics_by_base.items():
            if len(group_values) == 2 and all(isinstance(v, (int, float)) for v in group_values.values()):
                values = list(group_values.values())
                diff = abs(values[0] - values[1])
                pct_diff = (diff / max(values) * 100) if max(values) > 0 else 0

                stats['differences'][metric] = {
                    "absolute_diff": diff,
                    "percentage_diff": pct_diff,
                    "higher_group": max(group_values.items(), key=lambda x: x[1])[0]
                }

        stats['metrics_compared'] = dict(metrics_by_base)

        return stats

    def _build_comparison_overview(self, result: Dict, analysis: QueryAnalysis,
                                   stats: Dict) -> str:
        """Build overview section for comparison"""

        lines = ["## 📈 Overview\n"]

        # Add totals if available
        if stats.get('totals'):
            for group, total in stats['totals'].items():
                lines.append(f"- **{group}**: {total:,} total records")

        # Add summary of metrics compared
        lines.append(f"\n**Metrics Compared:** {len(stats['metrics_compared'])}")

        # Highlight biggest differences
        if stats.get('differences'):
            biggest_diff = max(stats['differences'].items(),
                               key=lambda x: x[1]['percentage_diff'])
            metric_name, diff_info = biggest_diff
            lines.append(f"\n**Largest Difference:** {metric_name} "
                         f"({diff_info['percentage_diff']:.1f}% higher in {diff_info['higher_group']})")

        return "\n".join(lines)

    def _build_comparison_table(self, result: Dict, analysis: QueryAnalysis) -> str:
        """Build comparison table from actual results"""

        # Use the same parsing logic
        parser = ColumnNameParser()
        metrics_by_base = defaultdict(dict)

        for field, value in result.items():
            col_info = parser.extract_column_info(field)

            # Try to match with comparison groups
            matched = False

            # Use extracted group identifier
            if col_info['group_identifier']:
                for group in analysis.comparison_groups:
                    if col_info['group_identifier'].upper() == group.upper():
                        # Use cleaned display name
                        display_metric = col_info['base_metric'].replace('_', ' ').title()
                        metrics_by_base[display_metric][group] = value
                        matched = True
                        break

            # Fallback to pattern matching
            if not matched:
                field_lower = field.lower()
                for group in analysis.comparison_groups:
                    group_lower = group.lower()

                    if group_lower in field_lower:
                        # Extract and clean metric name
                        base_metric = field_lower
                        patterns = [f'_{group_lower}', f'{group_lower}_', group_lower]
                        for pattern in patterns:
                            base_metric = base_metric.replace(pattern, '')

                        base_metric = base_metric.strip('_ ').replace('_', ' ').title()
                        metrics_by_base[base_metric][group] = value
                        break

        # Build table
        lines = ["\n## 📊 Detailed Comparison\n"]

        if not metrics_by_base:
            lines.append("*Unable to parse comparison data. Please check the SQL query structure.*")
            return "\n".join(lines)

        # Create header
        header = ["Metric"] + analysis.comparison_groups
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")

        # Add rows (sorted by metric name)
        for metric, values in sorted(metrics_by_base.items()):
            row = [metric]
            for group in analysis.comparison_groups:
                val = values.get(group, "N/A")

                # Smart formatting based on metric type
                if isinstance(val, float):
                    metric_lower = metric.lower()
                    if 'age' in metric_lower:
                        val = f"{val:.1f}"
                    elif 'score' in metric_lower or 'avg' in metric_lower:
                        val = f"{val:.3f}"
                    elif 'percent' in metric_lower or 'pct' in metric_lower:
                        val = f"{val:.1f}%"
                    else:
                        val = f"{val:,.0f}"
                elif isinstance(val, int):
                    val = f"{val:,}"

                row.append(str(val))

            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)

    def _build_comparison_differences(self, stats: Dict) -> str:
        """Build key differences section"""

        if not stats.get('differences'):
            return ""

        lines = ["\n## 🔍 Key Differences\n"]

        # Sort by percentage difference
        sorted_diffs = sorted(stats['differences'].items(),
                              key=lambda x: x[1]['percentage_diff'],
                              reverse=True)

        for metric, diff_info in sorted_diffs[:5]:  # Top 5 differences
            if diff_info['percentage_diff'] > 5:  # Only show significant differences
                lines.append(f"- **{metric}**: {diff_info['percentage_diff']:.1f}% "
                             f"higher in {diff_info['higher_group']}")

        return "\n".join(lines)

    def _generate_fallback_insights(self, stats: Dict) -> str:
        """Generate comprehensive insights when LLM fails"""

        insights = ["### Analysis\n"]

        # Overall comparison if totals exist
        if 'totals' in stats and len(stats['totals']) == 2:
            groups = list(stats['totals'].keys())
            values = list(stats['totals'].values())

            larger_idx = 0 if values[0] > values[1] else 1
            smaller_idx = 1 - larger_idx

            ratio = values[larger_idx] / values[smaller_idx] if values[smaller_idx] > 0 else float('inf')

            insights.append(f"- **Population Size:** {groups[larger_idx]} has {values[larger_idx]:,} individuals "
                            f"compared to {values[smaller_idx]:,} for {groups[smaller_idx]} "
                            f"({ratio:.1f}x larger)")

        # Analyze differences
        if 'differences' in stats:
            # Sort by percentage difference to find most significant
            sorted_diffs = sorted(stats['differences'].items(),
                                  key=lambda x: x[1]['percentage_diff'],
                                  reverse=True)

            # Key insights based on metric patterns
            for metric, diff in sorted_diffs[:5]:  # Top 5 differences
                pct_diff = diff['percentage_diff']
                higher_group = diff['higher_group']

                if pct_diff < 5:
                    continue  # Skip small differences

                # Generate insight based on metric type
                metric_lower = metric.lower()

                if 'risk' in metric_lower or 'score' in metric_lower:
                    if pct_diff > 20:
                        insights.append(f"- **Significant Risk Difference:** {higher_group} shows "
                                        f"{pct_diff:.1f}% higher {metric.replace('_', ' ')} score")
                    else:
                        insights.append(f"- **Moderate Risk Difference:** {higher_group} has "
                                        f"{pct_diff:.1f}% higher {metric.replace('_', ' ')} score")

                elif 'crime' in metric_lower or 'investigation' in metric_lower or 'prison' in metric_lower:
                    insights.append(f"- **Criminal Activity:** {higher_group} has "
                                    f"{pct_diff:.1f}% more {metric.replace('_', ' ')} cases")

                elif 'age' in metric_lower:
                    insights.append(f"- **Age Demographics:** {higher_group} population is "
                                    f"{pct_diff:.1f}% older on average")

                elif any(app in metric_lower for app in ['whatsapp', 'facebook', 'instagram']):
                    app_name = metric.replace('_users', '').replace('_', ' ').title()
                    insights.append(f"- **App Usage:** {higher_group} has {pct_diff:.1f}% more "
                                    f"{app_name} users")

                else:
                    insights.append(f"- **{metric.replace('_', ' ').title()}:** "
                                    f"{pct_diff:.1f}% higher in {higher_group}")

        # Add context about the metrics compared
        if 'metrics_compared' in stats:
            total_metrics = len(stats['metrics_compared'])
            insights.append(f"\n- **Comprehensive Analysis:** Compared {total_metrics} different metrics "
                            f"between the two groups")

            # Categorize metrics
            categories = {
                'Demographics': [],
                'Risk & Security': [],
                'Legal Status': [],
                'Digital Behavior': []
            }

            for metric in stats['metrics_compared'].keys():
                metric_lower = metric.lower()
                if 'age' in metric_lower or 'total' in metric_lower:
                    categories['Demographics'].append(metric)
                elif 'risk' in metric_lower or 'score' in metric_lower:
                    categories['Risk & Security'].append(metric)
                elif any(term in metric_lower for term in ['crime', 'investigation', 'prison']):
                    categories['Legal Status'].append(metric)
                elif any(app in metric_lower for app in ['whatsapp', 'facebook', 'instagram']):
                    categories['Digital Behavior'].append(metric)

            for category, metrics in categories.items():
                if metrics:
                    insights.append(f"- **{category}:** Analyzed {len(metrics)} metrics")

        # Summary recommendation
        insights.append("\n### Recommendations\n")

        # Find the group with higher risk/concerns
        risk_metrics = {group: 0 for group in stats.get('groups', [])}

        for metric, diff in stats.get('differences', {}).items():
            if any(term in metric.lower() for term in ['risk', 'crime', 'investigation', 'prison']):
                risk_metrics[diff['higher_group']] = risk_metrics.get(diff['higher_group'], 0) + 1

        if risk_metrics:
            higher_risk_group = max(risk_metrics.items(), key=lambda x: x[1])[0]
            insights.append(f"1. **Priority Focus:** {higher_risk_group} nationality shows higher risk "
                            f"indicators across multiple metrics")
            insights.append(f"2. **Enhanced Monitoring:** Consider additional screening for {higher_risk_group} "
                            f"nationals based on the identified patterns")
            insights.append(f"3. **Data-Driven Decisions:** Use these comparative insights to inform "
                            f"policy and resource allocation decisions")

        return "\n".join(insights)

    def _identify_patterns(self, stats: DataStatistics, analysis: QueryAnalysis) -> List[str]:
        """Identify patterns in the data"""

        patterns = []

        # Check for high concentration in categorical fields
        for field, counts in stats.categorical_fields.items():
            if counts:
                total = sum(counts.values())
                max_count = max(counts.values())
                if max_count / total > 0.7:
                    patterns.append(f"High concentration in {field}")

        # Check for outliers in numeric fields
        for field, field_stats in stats.numeric_fields.items():
            if field_stats['count'] > 0:
                range_val = field_stats['max'] - field_stats['min']
                if range_val > field_stats['avg'] * 3:
                    patterns.append(f"Wide range in {field}")

        # Check for data quality issues
        for field, null_count in stats.null_counts.items():
            if null_count > stats.total_records * 0.3:
                patterns.append(f"High null rate in {field}")

        return patterns

    def _build_statistics_section(self, stats: DataStatistics, analysis: QueryAnalysis) -> str:
        """Build statistics section"""

        lines = ["## 📊 Statistics Summary\n"]

        # Numeric field statistics
        if stats.numeric_fields:
            lines.append("### Numeric Fields")
            for field, field_stats in stats.numeric_fields.items():
                if field in analysis.key_metrics:
                    lines.append(f"\n**{field}:**")
                    lines.append(f"- Range: {field_stats['min']:.2f} - {field_stats['max']:.2f}")
                    lines.append(f"- Average: {field_stats['avg']:.2f}")

        # Categorical distributions
        if stats.categorical_fields:
            lines.append("\n### Distributions")
            for field, counts in stats.categorical_fields.items():
                if field in analysis.grouping_fields and counts:
                    lines.append(f"\n**{field}:** {len(counts)} unique values")
                    # Show top 3 values
                    top_values = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:3]
                    for value, count in top_values:
                        pct = count / stats.total_records * 100
                        lines.append(f"- {value}: {count:,} ({pct:.1f}%)")

        return "\n".join(lines)

    def _build_distribution_section(self, results: List[Dict], stats: DataStatistics,
                                    analysis: QueryAnalysis) -> str:
        """Build distribution analysis section"""

        lines = ["## 📊 Distribution Analysis\n"]

        # Focus on grouping fields
        for field in analysis.grouping_fields:
            if field in stats.categorical_fields:
                counts = stats.categorical_fields[field]
                if counts:
                    lines.append(f"\n### {field} Distribution")

                    # Sort by count
                    sorted_values = sorted(counts.items(), key=lambda x: x[1], reverse=True)

                    # Show all values if less than 10, otherwise top 10
                    display_values = sorted_values[:10] if len(sorted_values) > 10 else sorted_values

                    for value, count in display_values:
                        pct = count / stats.total_records * 100
                        bar_length = int(pct / 5)  # Scale to max 20 chars
                        bar = "█" * bar_length
                        lines.append(f"{value:<30} {bar} {count:>6,} ({pct:>5.1f}%)")

                    if len(sorted_values) > 10:
                        lines.append(f"\n... and {len(sorted_values) - 10} more values")

        return "\n".join(lines)

    def _build_profile_section(self, profile: Dict) -> str:
        """Build individual profile section"""

        if not profile:
            return "## Profile\n\nNo profile data available."

        lines = ["## 👤 Profile Details\n"]

        # Group fields by category
        categories = {
            "Basic Information": ['imsi', 'phone_no', 'fullname_en', 'gender_en', 'age_group'],
            "Demographics": ['nationality_name_en', 'residency_status', 'home_city'],
            "Risk Indicators": ['risk_score', 'drug_addict_score', 'drug_dealing_score', 'murder_score'],
            "Criminal Record": ['has_crime_case', 'has_investigation_case', 'is_in_prison']
        }

        for category, fields in categories.items():
            category_data = [(f, profile.get(f)) for f in fields if f in profile]
            if category_data:
                lines.append(f"\n### {category}")
                for field, value in category_data:
                    display_name = field.replace('_', ' ').title()
                    if isinstance(value, float):
                        value = f"{value:.2f}"
                    elif isinstance(value, bool):
                        value = "Yes" if value else "No"
                    lines.append(f"- **{display_name}:** {value}")

        return "\n".join(lines)

    def _build_table_section(self, results: List[Dict], analysis: QueryAnalysis) -> str:
        """Build table section for small result sets"""

        if not results:
            return ""

        lines = ["\n## 📋 Data Table\n"]

        # Select columns to display (prioritize key metrics and grouping fields)
        all_columns = list(results[0].keys())
        priority_columns = analysis.key_metrics + analysis.grouping_fields

        # Get unique priority columns that exist in results
        display_columns = [col for col in priority_columns if col in all_columns]

        # Add other columns if space allows
        for col in all_columns:
            if col not in display_columns and len(display_columns) < 8:
                display_columns.append(col)

        # Build header
        lines.append("| " + " | ".join(display_columns) + " |")
        lines.append("|" + "|".join(["---"] * len(display_columns)) + "|")

        # Add rows
        for row in results[:10]:  # Limit to 10 rows
            values = []
            for col in display_columns:
                val = row.get(col, "")
                if isinstance(val, float):
                    val = f"{val:.2f}"
                elif isinstance(val, int):
                    val = f"{val:,}"
                elif isinstance(val, str) and len(val) > 20:
                    val = val[:17] + "..."
                values.append(str(val))

            lines.append("| " + " | ".join(values) + " |")

        if len(results) > 10:
            lines.append(f"\n*... and {len(results) - 10} more rows*")

        return "\n".join(lines)

    def _generate_header(self, query: str, results: List[Dict],
                         analysis: QueryAnalysis) -> str:
        """Generate summary header"""

        header = f"# 📊 Query Analysis: {analysis.query_type.replace('_', ' ').title()}\n\n"
        header += f"**Query:** {query}\n"
        header += f"**Records Found:** {len(results):,}\n"

        if analysis.analysis_focus:
            header += f"**Focus Areas:** {', '.join(analysis.analysis_focus)}\n"

        if analysis.filter_conditions:
            header += f"**Filters Applied:** {', '.join(analysis.filter_conditions)}\n"

        return header

    def _generate_insights_section(self, results: List[Dict],
                                   analysis: QueryAnalysis) -> str:
        """Generate final insights based on expected insights"""

        content = "## 💡 Key Insights\n\n"

        for expected in analysis.expected_insights:
            content += f"- {expected}\n"

        return content
