# src/agents/profile_analyzer/nationality_comparator.py
from dataclasses import dataclass
from typing import List, Dict, Optional

from src.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class NationalityGroup:
    """Represents a nationality or group of nationalities"""
    name: str  # Display name (e.g., "Middle East", "AFG")
    codes: List[str]  # ISO 3-letter codes
    is_group: bool = False  # True if it's a region/group


class NationalityComparator:
    """Specialized handler for nationality comparisons"""

    # Regional groupings
    REGION_GROUPS = {
        'middle_east': {
            'name': 'Middle East',
            'codes': ['SAU', 'ARE', 'KWT', 'QAT', 'BHR', 'OMN', 'YEM', 'IRQ', 'SYR', 'JOR', 'LBN', 'PSE']
        },
        'gcc': {
            'name': 'GCC',
            'codes': ['SAU', 'ARE', 'KWT', 'QAT', 'BHR', 'OMN']
        },
        'europe': {
            'name': 'Europe',
            'codes': ['GBR', 'FRA', 'DEU', 'ITA', 'ESP', 'NLD', 'BEL', 'CHE', 'AUT', 'SWE', 'NOR', 'DNK', 'FIN', 'POL',
                      'ROU', 'GRC', 'PRT']
        },
        'south_asia': {
            'name': 'South Asia',
            'codes': ['IND', 'PAK', 'BGD', 'LKA', 'NPL', 'BTN', 'MDV', 'AFG']
        },
        'southeast_asia': {
            'name': 'Southeast Asia',
            'codes': ['IDN', 'MYS', 'SGP', 'THA', 'PHL', 'VNM', 'MMR', 'KHM', 'LAO']
        },
        'africa': {
            'name': 'Africa',
            'codes': ['EGY', 'ZAF', 'NGA', 'KEN', 'ETH', 'GHA', 'TZA', 'UGA', 'DZA', 'MAR', 'TUN']
        },
        'americas': {
            'name': 'Americas',
            'codes': ['USA', 'CAN', 'MEX', 'BRA', 'ARG', 'CHL', 'COL', 'PER', 'VEN']
        },
        'central_asia': {
            'name': 'Central Asia',
            'codes': ['KAZ', 'KGZ', 'TJK', 'TKM', 'UZB']
        },
        'east_asia': {
            'name': 'East Asia',
            'codes': ['CHN', 'JPN', 'KOR', 'PRK', 'MNG', 'TWN', 'HKG', 'MAC']
        },
        'caucasus': {
            'name': 'Caucasus',
            'codes': ['GEO', 'ARM', 'AZE']
        }
    }

    def __init__(self):
        self.metrics_config = self._get_metrics_config()

    def _get_metrics_config(self) -> Dict[str, Dict]:
        """Define what metrics to compare"""
        return {
            'demographics': {
                'total_count': 'COUNT(*)',
                'avg_age': 'AVG(age)',
                'male_count': "SUM(CASE WHEN gender_en = 'Male' THEN 1 ELSE 0 END)",
                'female_count': "SUM(CASE WHEN gender_en = 'Female' THEN 1 ELSE 0 END)",
                'citizen_count': "SUM(CASE WHEN residency_status = 'CITIZEN' THEN 1 ELSE 0 END)",
                'resident_count': "SUM(CASE WHEN residency_status = 'RESIDENT' THEN 1 ELSE 0 END)",
                'visitor_count': "SUM(CASE WHEN residency_status = 'VISITOR' THEN 1 ELSE 0 END)"
            },
            'risk_scores': {
                'avg_risk_score': 'AVG(risk_score)',
                'high_risk_count': 'SUM(CASE WHEN risk_score > 0.2 THEN 1 ELSE 0 END)',
                'avg_drug_dealing_score': 'AVG(drug_dealing_score)',
                'avg_drug_addict_score': 'AVG(drug_addict_score)',
                'avg_murder_score': 'AVG(murder_score)'
            },
            'criminal_activity': {
                'investigation_cases': 'SUM(CASE WHEN has_investigation_case THEN 1 ELSE 0 END)',
                'crime_cases': 'SUM(CASE WHEN has_crime_case THEN 1 ELSE 0 END)',
                'prison_cases': 'SUM(CASE WHEN is_in_prison THEN 1 ELSE 0 END)'
            },
            'digital_behavior': {
                'whatsapp_users': "SUM(CASE WHEN has(applications_used, 'WhatsApp') THEN 1 ELSE 0 END)",
                'facebook_users': "SUM(CASE WHEN has(applications_used, 'Facebook') THEN 1 ELSE 0 END)",
                'instagram_users': "SUM(CASE WHEN has(applications_used, 'Instagram') THEN 1 ELSE 0 END)",
                'telegram_users': "SUM(CASE WHEN has(applications_used, 'Telegram') THEN 1 ELSE 0 END)"
            },
            'geographic': {
                'dubai_residents': "SUM(CASE WHEN home_city = 'Dubai' THEN 1 ELSE 0 END)",
                'abudhabi_residents': "SUM(CASE WHEN home_city = 'AbuDhabi' THEN 1 ELSE 0 END)",
                'sharjah_residents': "SUM(CASE WHEN home_city = 'Sharjah' THEN 1 ELSE 0 END)"
            }
        }

    def parse_nationality_request(self, query: str, comparison_groups: List[str]) -> List[NationalityGroup]:
        """Parse nationality comparison request and identify groups"""
        groups = []

        for group_str in comparison_groups:
            group_lower = group_str.lower().replace(' ', '_')

            # Check if it's a predefined region
            if group_lower in self.REGION_GROUPS:
                region_info = self.REGION_GROUPS[group_lower]
                groups.append(NationalityGroup(
                    name=region_info['name'],
                    codes=region_info['codes'],
                    is_group=True
                ))
            else:
                # Single nationality - ensure it's uppercase 3-letter code
                code = group_str.upper()[:3]
                groups.append(NationalityGroup(
                    name=code,
                    codes=[code],
                    is_group=False
                ))

        return groups

    def generate_comparison_sql(self, groups: List[NationalityGroup],
                                additional_filters: Optional[str] = None) -> str:
        """Generate standardized SQL for nationality comparison"""

        # Build CTEs for each group
        cte_parts = []

        for i, group in enumerate(groups):
            # Create WHERE clause for nationality codes
            if len(group.codes) == 1:
                nationality_filter = f"nationality_code = '{group.codes[0]}'"
            else:
                codes_str = "', '".join(group.codes)
                nationality_filter = f"nationality_code IN ('{codes_str}')"

            # Build metrics selection
            metric_selections = []
            for category, metrics in self.metrics_config.items():
                for metric_name, metric_sql in metrics.items():
                    metric_selections.append(f"{metric_sql} AS {metric_name}")

            # Add group identifier
            metric_selections.append(f"'{group.name}' AS group_name")

            # Build CTE
            additional_filter_clause = f'AND {additional_filters}' if additional_filters else ''
            metric_selections_str = ',\n                '.join(metric_selections)
            cte_sql = f"""
        group_{i} AS (
            SELECT 
                {metric_selections_str}
            FROM telecom_db.phone_imsi_uid_latest
            WHERE {nationality_filter}
            {additional_filter_clause}
        )"""
            cte_parts.append(cte_sql)

        # Build final UNION query
        union_parts = [f"SELECT * FROM group_{i}" for i in range(len(groups))]

        cte_parts_str = ',\n    '.join(cte_parts)
        union_parts_str = ' UNION ALL '.join(union_parts)

        sql = f"""
WITH {cte_parts_str}
{union_parts_str}
ORDER BY total_count DESC
"""

        return sql

    def generate_2way_comparison_sql(self, group1: NationalityGroup, group2: NationalityGroup,
                                     additional_filters: Optional[str] = None) -> str:
        """Generate optimized SQL for 2-way comparison (single row result)"""

        # Build WHERE clauses
        if len(group1.codes) == 1:
            where1 = f"nationality_code = '{group1.codes[0]}'"
        else:
            codes_str = "', '".join(group1.codes)
            where1 = f"nationality_code IN ('{codes_str}')"

        if len(group2.codes) == 1:
            where2 = f"nationality_code = '{group2.codes[0]}'"
        else:
            codes_str = "', '".join(group2.codes)
            where2 = f"nationality_code IN ('{codes_str}')"

        # Build metric selections with group suffixes
        selections = []

        for category, metrics in self.metrics_config.items():
            for metric_name, metric_sql in metrics.items():
                # Add both groups' metrics
                selections.append(f"g1.{metric_name} AS {metric_name}_{group1.name.lower()}")
                selections.append(f"g2.{metric_name} AS {metric_name}_{group2.name.lower()}")

        # Build metric selections for CTEs
        cte_selections = []
        for category, metrics in self.metrics_config.items():
            for metric_name, metric_sql in metrics.items():
                cte_selections.append(f"{metric_sql} AS {metric_name}")

        additional_filter_clause = f'AND {additional_filters}' if additional_filters else ''
        cte_selections_str = ',\n            '.join(cte_selections)
        selections_str = ',\n    '.join(selections)

        sql = f"""
WITH 
    g1 AS (
        SELECT 
            {cte_selections_str}
        FROM telecom_db.phone_imsi_uid_latest
        WHERE {where1}
        {additional_filter_clause}
    ),
    g2 AS (
        SELECT 
            {cte_selections_str}
        FROM telecom_db.phone_imsi_uid_latest
        WHERE {where2}
        {additional_filter_clause}
    )
SELECT 
    {selections_str}
FROM g1
CROSS JOIN g2
"""

        return sql


class NationalityComparisonSummarizer:
    """Specialized summarizer for nationality comparisons"""

    def __init__(self):
        self.metric_categories = {
            'Demographics': ['total_count', 'avg_age', 'male_count', 'female_count',
                             'citizen_count', 'resident_count', 'visitor_count'],
            'Risk Assessment': ['avg_risk_score', 'high_risk_count', 'avg_drug_dealing_score',
                                'avg_drug_addict_score', 'avg_murder_score'],
            'Criminal Activity': ['investigation_cases', 'crime_cases', 'prison_cases'],
            'Digital Behavior': ['whatsapp_users', 'facebook_users', 'instagram_users', 'telegram_users'],
            'Geographic Distribution': ['dubai_residents', 'abudhabi_residents', 'sharjah_residents']
        }

    def summarize_comparison(self, results: List[Dict], groups: List[NationalityGroup],
                             is_single_row: bool = False) -> str:
        """Generate summary for nationality comparison"""

        if is_single_row and len(results) == 1:
            return self._summarize_2way_comparison(results[0], groups)
        else:
            return self._summarize_multiway_comparison(results, groups)

    def _summarize_2way_comparison(self, result: Dict, groups: List[NationalityGroup]) -> str:
        """Summarize 2-way comparison from single row result"""

        sections = []

        # Header
        sections.append(f"# 🌍 Nationality Comparison Analysis\n")
        sections.append(f"**Comparing:** {groups[0].name} vs {groups[1].name}\n")

        # Parse results into structured data
        metrics_by_group = self._parse_2way_results(result, groups)

        # Overview
        sections.append(self._build_overview_section(metrics_by_group))

        # Detailed comparison by category
        for category, metric_names in self.metric_categories.items():
            category_section = self._build_category_comparison(category, metric_names, metrics_by_group)
            if category_section:
                sections.append(category_section)

        # Key insights
        sections.append(self._build_insights_section(metrics_by_group))

        # Recommendations
        sections.append(self._build_recommendations_section(metrics_by_group))

        return "\n".join(sections)

    def _summarize_multiway_comparison(self, results: List[Dict], groups: List[NationalityGroup]) -> str:
        """Summarize multi-way comparison from multiple rows"""

        sections = []

        # Header
        sections.append(f"# 🌍 Multi-Nationality Comparison Analysis\n")
        group_names = ', '.join([g.name for g in groups])
        sections.append(f"**Comparing:** {group_names}\n")

        # Overview table
        sections.append(self._build_overview_table(results))

        # Detailed analysis by category
        for category, metric_names in self.metric_categories.items():
            category_section = self._build_multiway_category_section(category, metric_names, results)
            if category_section:
                sections.append(category_section)

        # Rankings
        sections.append(self._build_rankings_section(results))

        return "\n".join(sections)

    def _parse_2way_results(self, result: Dict, groups: List[NationalityGroup]) -> Dict:
        """Parse single-row results into structured format"""
        metrics_by_group = {group.name: {} for group in groups}

        for key, value in result.items():
            # Extract metric name and group
            for group in groups:
                suffix = f"_{group.name.lower()}"
                if key.endswith(suffix):
                    metric_name = key[:-len(suffix)]
                    metrics_by_group[group.name][metric_name] = value
                    break

        return metrics_by_group

    def _build_overview_section(self, metrics_by_group: Dict) -> str:
        """Build overview section for 2-way comparison"""
        lines = ["## 📊 Overview\n"]

        # Get group names
        groups = list(metrics_by_group.keys())

        # Total populations
        for group in groups:
            total = metrics_by_group[group].get('total_count', 0)
            lines.append(f"- **{group}**: {total:,} individuals")

        # Population ratio
        if len(groups) == 2:
            count1 = metrics_by_group[groups[0]].get('total_count', 0)
            count2 = metrics_by_group[groups[1]].get('total_count', 0)
            if count2 > 0:
                ratio = count1 / count2
                larger, smaller = (groups[0], groups[1]) if count1 > count2 else (groups[1], groups[0])
                lines.append(f"\n**Population Ratio:** {larger} has {ratio:.1f}x more individuals than {smaller}")

        return "\n".join(lines)

    def _build_category_comparison(self, category: str, metric_names: List[str],
                                   metrics_by_group: Dict) -> str:
        """Build comparison section for a category"""
        lines = [f"\n## {category}\n"]

        # Create comparison table
        groups = list(metrics_by_group.keys())

        # Table header
        header_sep = '|'.join(['-------'] * (len(groups) + 1))
        lines.append(f"| Metric | {' | '.join(groups)} | Difference |")
        lines.append(f"|--------|{header_sep}|")

        # Add rows for each metric
        for metric in metric_names:
            # Get values for each group
            values = []
            for group in groups:
                val = metrics_by_group[group].get(metric, 0)
                values.append(val)

            # Skip if all zeros
            if all(v == 0 for v in values):
                continue

            # Format metric name
            display_name = metric.replace('_', ' ').title()

            # Calculate difference for 2-way comparison
            if len(values) == 2:
                if isinstance(values[0], (int, float)) and values[1] != 0:
                    diff_pct = ((values[0] - values[1]) / values[1]) * 100
                    diff_str = f"{diff_pct:+.1f}%"
                else:
                    diff_str = "N/A"
            else:
                diff_str = "-"

            # Format values
            formatted_values = []
            for val in values:
                if isinstance(val, float):
                    if 'score' in metric or 'avg' in metric:
                        formatted_values.append(f"{val:.3f}")
                    else:
                        formatted_values.append(f"{val:.0f}")
                else:
                    formatted_values.append(f"{val:,}")

            # Add row
            lines.append(f"| {display_name} | {' | '.join(formatted_values)} | {diff_str} |")

        return "\n".join(lines) if len(lines) > 3 else ""  # Only return if there's content

    def _build_insights_section(self, metrics_by_group: Dict) -> str:
        """Build insights section"""
        lines = ["\n## 💡 Key Insights\n"]

        groups = list(metrics_by_group.keys())

        # Risk comparison
        risk_scores = {g: metrics_by_group[g].get('avg_risk_score', 0) for g in groups}
        if any(risk_scores.values()):
            higher_risk = max(risk_scores.items(), key=lambda x: x[1])
            lines.append(f"- **Risk Profile:** {higher_risk[0]} shows higher average risk ({higher_risk[1]:.3f})")

        # Criminal activity
        crime_totals = {g: sum([
            metrics_by_group[g].get('investigation_cases', 0),
            metrics_by_group[g].get('crime_cases', 0),
            metrics_by_group[g].get('prison_cases', 0)
        ]) for g in groups}

        if any(crime_totals.values()):
            higher_crime = max(crime_totals.items(), key=lambda x: x[1])
            lines.append(
                f"- **Criminal Activity:** {higher_crime[0]} has more criminal involvement ({higher_crime[1]} total cases)")

        # Digital behavior
        digital_usage = {g: sum([
            metrics_by_group[g].get('whatsapp_users', 0),
            metrics_by_group[g].get('facebook_users', 0),
            metrics_by_group[g].get('instagram_users', 0)
        ]) for g in groups}

        if len(groups) == 2 and all(digital_usage.values()):
            total1 = metrics_by_group[groups[0]].get('total_count', 1)
            total2 = metrics_by_group[groups[1]].get('total_count', 1)

            digital_rate1 = digital_usage[groups[0]] / total1 if total1 > 0 else 0
            digital_rate2 = digital_usage[groups[1]] / total2 if total2 > 0 else 0

            if digital_rate1 > digital_rate2:
                lines.append(f"- **Digital Engagement:** {groups[0]} shows higher social media usage rate")
            else:
                lines.append(f"- **Digital Engagement:** {groups[1]} shows higher social media usage rate")

        # Demographics
        for g in groups:
            male = metrics_by_group[g].get('male_count', 0)
            female = metrics_by_group[g].get('female_count', 0)
            total = male + female
            if total > 0:
                male_pct = (male / total) * 100
                lines.append(f"- **{g} Gender Split:** {male_pct:.1f}% male, {100 - male_pct:.1f}% female")

        return "\n".join(lines)

    def _build_recommendations_section(self, metrics_by_group: Dict) -> str:
        """Build recommendations based on comparison"""
        lines = ["\n## 📋 Recommendations\n"]

        groups = list(metrics_by_group.keys())

        # Find group with higher risk indicators
        risk_indicators = {}
        for g in groups:
            risk_indicators[g] = (
                    metrics_by_group[g].get('avg_risk_score', 0) * 1000 +
                    metrics_by_group[g].get('crime_cases', 0) +
                    metrics_by_group[g].get('investigation_cases', 0)
            )

        higher_risk_group = max(risk_indicators.items(), key=lambda x: x[1])[0]

        lines.append(
            f"1. **Enhanced Screening:** Implement additional security measures for {higher_risk_group} nationals")
        lines.append(f"2. **Risk Profiling:** Develop specific risk profiles based on the identified patterns")
        lines.append(f"3. **Resource Allocation:** Allocate monitoring resources proportionally to risk levels")
        lines.append(
            f"4. **Policy Review:** Consider policy adjustments based on demographic and behavioral differences")

        return "\n".join(lines)

    def _build_overview_table(self, results: List[Dict]) -> str:
        """Build overview table for multi-way comparison"""
        lines = ["## 📊 Overview Statistics\n"]

        # Sort by total count
        sorted_results = sorted(results, key=lambda x: x.get('total_count', 0), reverse=True)

        # Create table
        lines.append("| Nationality/Group | Population | Avg Age | Risk Score | Crime Cases |")
        lines.append("|-------------------|------------|---------|------------|-------------|")

        for row in sorted_results:
            group = row.get('group_name', 'Unknown')
            total = row.get('total_count', 0)
            age = row.get('avg_age', 0)
            risk = row.get('avg_risk_score', 0)
            crimes = row.get('crime_cases', 0)

            lines.append(f"| {group} | {total:,} | {age:.1f} | {risk:.3f} | {crimes:,} |")

        return "\n".join(lines)

    def _build_multiway_category_section(self, category: str, metric_names: List[str],
                                         results: List[Dict]) -> str:
        """Build category section for multi-way comparison"""
        lines = [f"\n## {category}\n"]

        # Get all groups
        groups = [r.get('group_name', 'Unknown') for r in results]

        # Create table
        header_sep = '|'.join(['-------'] * len(groups))
        lines.append(f"| Metric | {' | '.join(groups)} |")
        lines.append(f"|--------|{header_sep}|")

        # Add metrics
        for metric in metric_names:
            # Skip if all zeros
            if all(r.get(metric, 0) == 0 for r in results):
                continue

            display_name = metric.replace('_', ' ').title()

            values = []
            for r in results:
                val = r.get(metric, 0)
                if isinstance(val, float):
                    if 'score' in metric or 'avg' in metric:
                        values.append(f"{val:.3f}")
                    else:
                        values.append(f"{val:.0f}")
                else:
                    values.append(f"{val:,}")

            lines.append(f"| {display_name} | {' | '.join(values)} |")

        return "\n".join(lines) if len(lines) > 3 else ""

    def _build_rankings_section(self, results: List[Dict]) -> str:
        """Build rankings section"""
        lines = ["\n## 🏆 Rankings\n"]

        # Risk ranking
        risk_sorted = sorted(results, key=lambda x: x.get('avg_risk_score', 0), reverse=True)
        lines.append("### Highest Risk")
        for i, r in enumerate(risk_sorted[:3], 1):
            lines.append(f"{i}. **{r.get('group_name')}**: {r.get('avg_risk_score', 0):.3f} average risk score")

        # Crime ranking
        crime_sorted = sorted(results, key=lambda x: x.get('crime_cases', 0), reverse=True)
        lines.append("\n### Most Criminal Cases")
        for i, r in enumerate(crime_sorted[:3], 1):
            lines.append(f"{i}. **{r.get('group_name')}**: {r.get('crime_cases', 0):,} cases")

        # Population ranking
        pop_sorted = sorted(results, key=lambda x: x.get('total_count', 0), reverse=True)
        lines.append("\n### Largest Populations")
        for i, r in enumerate(pop_sorted[:3], 1):
            lines.append(f"{i}. **{r.get('group_name')}**: {r.get('total_count', 0):,} individuals")

        return "\n".join(lines)
