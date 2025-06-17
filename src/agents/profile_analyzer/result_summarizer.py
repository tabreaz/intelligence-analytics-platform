# src/agents/profile_analyzer/result_summarizer.py
import re
import statistics
from collections import Counter, defaultdict
from typing import List, Dict, Any

from src.core.logger import get_logger

logger = get_logger(__name__)


class ResultSummarizer:
    """Smart pattern-based result summarizer that generates markdown summaries"""

    def __init__(self):
        # Define patterns for different analysis types
        self.numeric_fields = {
            'risk_score', 'drug_addict_score', 'drug_dealing_score', 'murder_score',
            'age', 'activity_count', 'unique_users', 'total_visits'
        }

        self.categorical_fields = {
            'gender_en', 'age_group', 'residency_status', 'marital_status_en',
            'nationality_code', 'nationality_name_en', 'home_city', 'dwell_duration_tag'
        }

        self.array_fields = {
            'applications_used', 'crime_categories_en', 'crime_sub_categories_en',
            'travelled_country_codes', 'communicated_country_codes', 'driving_license_type',
            'drug_addict_rules', 'drug_dealing_rules', 'murder_rules', 'risk_rules'
        }

        self.boolean_fields = {
            'has_investigation_case', 'has_crime_case', 'is_in_prison'
        }

        self.identifier_fields = {
            'imsi', 'phone_no', 'uid', 'fullname_en'
        }

    def _get_field_value(self, row: Dict, field_name: str) -> Any:
        """Get field value handling SQL aliases"""
        # Direct match
        if field_name in row:
            return row[field_name]

        # Check for SQL alias pattern
        for key in row.keys():
            if ' AS ' in key:
                original_col = key.split(' AS ')[0].strip()
                if original_col == field_name:
                    return row[key]
            elif key.lower() == field_name.lower():
                return row[key]

        return None

    def generate_summary(self, query: str, results: List[Dict], sql_query: str = None) -> str:
        """Generate comprehensive markdown summary of results"""

        if not results:
            return "## 📊 Query Summary\n\n❌ **No results found for your query.**"

        # Check if this is a comparison query
        if self._is_comparison_query(query, sql_query, results):
            return self._generate_comparison_summary(query, results[0], sql_query)

        # Check if this is a single-row aggregate result
        if len(results) == 1 and self._is_aggregate_result(results[0]):
            return self._generate_aggregate_summary(query, results[0], sql_query)

        # Regular multi-row processing
        query_type = self._classify_query_type(query, sql_query, results)
        available_fields = self._detect_fields(results)

        # Build summary sections
        sections = []

        # Header section
        sections.append(self._generate_header(query, results, query_type))

        # Key insights based on patterns
        insights = self._generate_insights(results, available_fields, query_type)
        if insights:
            sections.append("## 🔍 Key Insights\n\n" + insights)

        # Statistical analysis
        stats = self._generate_statistics(results, available_fields)
        if stats:
            sections.append("## 📈 Statistical Analysis\n\n" + stats)

        # Distribution analysis
        distributions = self._generate_distributions(results, available_fields)
        if distributions:
            sections.append("## 📊 Distribution Analysis\n\n" + distributions)

        # Pattern analysis
        patterns = self._analyze_patterns(results, available_fields)
        if patterns:
            sections.append("## 🔗 Pattern Analysis\n\n" + patterns)

        # Top profiles table (if applicable)
        if self._should_show_profiles_table(query_type, available_fields):
            profiles_table = self._generate_profiles_table(results)
            if profiles_table:
                sections.append("## 👥 Top Profiles\n\n" + profiles_table)

        # Risk analysis (if risk scores present)
        if any(field in available_fields for field in
               ['risk_score', 'drug_dealing_score', 'drug_addict_score', 'murder_score']):
            risk_analysis = self._generate_risk_analysis(results, available_fields)
            if risk_analysis:
                sections.append("## ⚠️ Risk Analysis\n\n" + risk_analysis)

        return "\n\n".join(sections)

    def _is_aggregate_result(self, row: Dict) -> bool:
        """Check if this is a single-row aggregate result"""
        # Look for aggregate function patterns in field names
        aggregate_patterns = ['COUNT(', 'AVG(', 'SUM(', 'MIN(', 'MAX(', 'COUNTIf(']

        aggregate_fields = 0
        total_fields = len(row)

        for field in row.keys():
            if any(pattern in str(field).upper() for pattern in aggregate_patterns):
                aggregate_fields += 1

        # If more than 50% of fields are aggregates, treat as aggregate result
        return aggregate_fields > total_fields * 0.5

    def _generate_aggregate_summary(self, query: str, result: Dict, sql_query: str = None) -> str:
        """Generate smart summary for single-row aggregate results"""
        sections = []

        # Parse the aggregate results
        parsed_data = self._parse_aggregate_results(result)

        # Header
        total_count = parsed_data.get('totals', {}).get('total', 0)
        sections.append(f"# 📊 Query Summary\n\n**Query:** {query}\n")

        # Determine what type of analysis this is
        analysis_type = self._determine_aggregate_analysis_type(parsed_data, query)
        sections.append(f"**Analysis Type:** {analysis_type}")

        if total_count > 0:
            sections.append(f"**Total Records:** {total_count:,}")

        # Key Insights
        insights = self._generate_aggregate_insights(parsed_data, query)
        if insights:
            sections.append("\n## 🔍 Key Insights\n")
            sections.extend(insights)

        # Demographic Breakdown
        demographics = self._format_demographic_breakdown(parsed_data)
        if demographics:
            sections.append("\n## 📊 Demographic Breakdown\n")
            sections.append(demographics)

        # Statistical Summary
        stats = self._format_aggregate_statistics(parsed_data)
        if stats:
            sections.append("\n## 📈 Statistical Summary\n")
            sections.append(stats)

        # Risk Analysis (if applicable)
        if 'scores' in parsed_data and parsed_data['scores']:
            risk = self._format_aggregate_risk_analysis(parsed_data)
            if risk:
                sections.append("\n## ⚠️ Risk Analysis\n")
                sections.append(risk)

        # Recommendations
        recommendations = self._generate_recommendations(parsed_data, analysis_type)
        if recommendations:
            sections.append("\n## 💡 Recommendations\n")
            sections.extend(recommendations)

        return "\n".join(sections)

    def _parse_aggregate_results(self, result: Dict) -> Dict:
        """Parse single-row aggregate results into structured data"""

        parsed = {
            'totals': {},
            'demographics': {
                'gender': {},
                'age_groups': {},
                'marital_status': {},
                'nationality': {}
            },
            'scores': {},
            'other': {}
        }

        for key, value in result.items():
            key_upper = key.upper()

            # Total counts
            if 'TOTAL' in key_upper or key_upper == 'COUNT(*)':
                parsed['totals']['total'] = value

            # Average scores
            elif 'AVG(' in key_upper and 'SCORE' in key_upper:
                score_type = self._extract_score_type(key)
                parsed['scores'][score_type] = value

            # Gender counts
            elif 'GENDER' in key_upper or ('MALE' in key_upper or 'FEMALE' in key_upper):
                if 'MALE' in key_upper and 'FEMALE' not in key_upper:
                    parsed['demographics']['gender']['male'] = value
                elif 'FEMALE' in key_upper:
                    parsed['demographics']['gender']['female'] = value

            # Age group counts
            elif 'AGE_GROUP' in key_upper or re.search(r'\d{2}-\d{2}', key):
                age_match = re.search(r'(\d{2}-\d{2})', key)
                if age_match:
                    age_group = age_match.group(1)
                    parsed['demographics']['age_groups'][age_group] = value

            # Marital status counts
            elif any(status in key_upper for status in ['MARRIED', 'SINGLE', 'DIVORCED', 'WIDOWED']):
                if 'MARRIED' in key_upper and 'SINGLE' not in key_upper and 'DIVORCED' not in key_upper:
                    parsed['demographics']['marital_status']['married'] = value
                elif 'SINGLE' in key_upper:
                    parsed['demographics']['marital_status']['single'] = value
                elif 'DIVORCED' in key_upper:
                    parsed['demographics']['marital_status']['divorced'] = value
                elif 'WIDOWED' in key_upper:
                    parsed['demographics']['marital_status']['widowed'] = value

            # Nationality counts
            elif 'NATIONAL' in key_upper or 'UAE' in key_upper or 'ARE' in key_upper:
                if 'UAE' in key_upper or ('ARE' in key_upper and '!=' not in key):
                    parsed['demographics']['nationality']['uae'] = value
                elif 'NON' in key_upper or '!=' in key:
                    parsed['demographics']['nationality']['non_uae'] = value

            # Other fields
            else:
                parsed['other'][key] = value

        return parsed

    def _extract_score_type(self, field_name: str) -> str:
        """Extract score type from field name"""
        if 'drug_dealing' in field_name.lower():
            return 'drug_dealing'
        elif 'drug_addict' in field_name.lower():
            return 'drug_addict'
        elif 'murder' in field_name.lower():
            return 'murder'
        elif 'risk' in field_name.lower():
            return 'risk'
        return 'unknown'

    def _determine_aggregate_analysis_type(self, parsed_data: Dict, query: str) -> str:
        """Determine the type of analysis from parsed data and query"""
        query_lower = query.lower()

        if 'drug' in query_lower:
            if 'dealer' in query_lower or 'dealing' in query_lower:
                return "Drug Dealing Risk Analysis"
            elif 'addict' in query_lower:
                return "Drug Addiction Risk Analysis"
            else:
                return "Drug-Related Risk Analysis"
        elif 'risk' in query_lower:
            return "Comprehensive Risk Analysis"
        elif 'crime' in query_lower or 'criminal' in query_lower:
            return "Criminal Activity Analysis"
        elif 'demographic' in query_lower:
            return "Demographic Analysis"
        else:
            return "Profile Analysis"

    def _generate_aggregate_insights(self, parsed_data: Dict, query: str) -> List[str]:
        """Generate insights from parsed aggregate data"""
        insights = []

        total = parsed_data.get('totals', {}).get('total', 0)

        if total > 0:
            # Total count insight
            if 'drug' in query.lower() and 'dealer' in query.lower():
                insights.append(f"- 💊 **{total:,} individuals identified as high-risk drug dealers**")
            else:
                insights.append(f"- 📊 **Total individuals analyzed: {total:,}**")

            # Score insights
            for score_type, value in parsed_data.get('scores', {}).items():
                if value is not None:
                    score_label = score_type.replace('_', ' ').title()
                    if value > 0.7:
                        insights.append(f"- 🚨 **Average {score_label} Score: {value:.3f}** (Very High Risk)")
                    elif value > 0.5:
                        insights.append(f"- ⚠️ **Average {score_label} Score: {value:.3f}** (High Risk)")
                    elif value > 0.3:
                        insights.append(f"- 📌 **Average {score_label} Score: {value:.3f}** (Moderate Risk)")
                    else:
                        insights.append(f"- ✓ **Average {score_label} Score: {value:.3f}** (Low Risk)")

            # Gender insights
            gender_data = parsed_data['demographics']['gender']
            if gender_data:
                male_count = gender_data.get('male', 0)
                female_count = gender_data.get('female', 0)
                if male_count or female_count:
                    male_pct = (male_count / total * 100) if total > 0 else 0
                    female_pct = (female_count / total * 100) if total > 0 else 0
                    insights.append(
                        f"- 👤 **Gender Distribution:** Male: {male_count:,} ({male_pct:.1f}%), Female: {female_count:,} ({female_pct:.1f}%)")

            # Age insights
            age_groups = parsed_data['demographics']['age_groups']
            if age_groups:
                # Find dominant age group
                dominant_age = max(age_groups.items(), key=lambda x: x[1])
                if dominant_age[1] > 0:
                    age_pct = (dominant_age[1] / total * 100) if total > 0 else 0
                    insights.append(
                        f"- 👥 **Dominant Age Group:** {dominant_age[0]} years ({dominant_age[1]:,} individuals, {age_pct:.1f}%)")

                # Check for youth involvement
                youth_count = sum(v for k, v in age_groups.items() if k in ['20-30', '30-40'])
                if youth_count > total * 0.7:
                    youth_pct = (youth_count / total * 100) if total > 0 else 0
                    insights.append(
                        f"- 🎯 **Youth Concentration:** {youth_count:,} individuals under 40 ({youth_pct:.1f}%)")

            # Marital status insights
            marital_data = parsed_data['demographics']['marital_status']
            if marital_data:
                single_count = marital_data.get('single', 0)
                if single_count > total * 0.5:
                    single_pct = (single_count / total * 100) if total > 0 else 0
                    insights.append(f"- 💑 **Marital Status:** {single_count:,} are single ({single_pct:.1f}%)")

            # Nationality insights
            nat_data = parsed_data['demographics']['nationality']
            if nat_data:
                non_uae = nat_data.get('non_uae', 0)
                uae = nat_data.get('uae', 0)
                if non_uae > 0:
                    if uae == 0:
                        insights.append(f"- 🌍 **Nationality:** 100% are foreign nationals (no UAE citizens)")
                    else:
                        non_uae_pct = (non_uae / total * 100) if total > 0 else 0
                        insights.append(f"- 🌍 **Foreign Nationals:** {non_uae:,} ({non_uae_pct:.1f}%)")

        return insights

    def _format_demographic_breakdown(self, parsed_data: Dict) -> str:
        """Format demographic data into tables"""
        sections = []
        total = parsed_data.get('totals', {}).get('total', 1)  # Avoid division by zero

        # Age Distribution
        age_groups = parsed_data['demographics']['age_groups']
        if age_groups:
            sections.append("### Age Distribution")
            sections.append("| Age Group | Count | Percentage |")
            sections.append("|-----------|-------|------------|")

            # Sort age groups
            sorted_ages = sorted(age_groups.items(), key=lambda x: x[0])
            for age_group, count in sorted_ages:
                pct = (count / total * 100) if total > 0 else 0
                sections.append(f"| {age_group} | {count:,} | {pct:.1f}% |")
            sections.append("")

        # Gender Distribution
        gender_data = parsed_data['demographics']['gender']
        if gender_data:
            sections.append("### Gender Distribution")
            sections.append("| Gender | Count | Percentage |")
            sections.append("|--------|-------|------------|")

            if 'male' in gender_data:
                count = gender_data['male']
                pct = (count / total * 100) if total > 0 else 0
                sections.append(f"| Male | {count:,} | {pct:.1f}% |")

            if 'female' in gender_data:
                count = gender_data['female']
                pct = (count / total * 100) if total > 0 else 0
                sections.append(f"| Female | {count:,} | {pct:.1f}% |")
            sections.append("")

        # Marital Status
        marital_data = parsed_data['demographics']['marital_status']
        if marital_data:
            sections.append("### Marital Status")
            sections.append("| Status | Count | Percentage |")
            sections.append("|--------|-------|------------|")

            status_order = ['single', 'married', 'divorced', 'widowed']
            for status in status_order:
                if status in marital_data:
                    count = marital_data[status]
                    pct = (count / total * 100) if total > 0 else 0
                    sections.append(f"| {status.capitalize()} | {count:,} | {pct:.1f}% |")
            sections.append("")

        # Nationality
        nat_data = parsed_data['demographics']['nationality']
        if nat_data:
            sections.append("### Nationality Breakdown")
            sections.append("| Category | Count | Percentage |")
            sections.append("|----------|-------|------------|")

            if 'uae' in nat_data:
                count = nat_data['uae']
                pct = (count / total * 100) if total > 0 else 0
                sections.append(f"| UAE Citizens | {count:,} | {pct:.1f}% |")

            if 'non_uae' in nat_data:
                count = nat_data['non_uae']
                pct = (count / total * 100) if total > 0 else 0
                sections.append(f"| Foreign Nationals | {count:,} | {pct:.1f}% |")

        return "\n".join(sections)

    def _format_aggregate_statistics(self, parsed_data: Dict) -> str:
        """Format statistical summary"""
        stats = []

        # Add score statistics
        scores = parsed_data.get('scores', {})
        if scores:
            for score_type, value in scores.items():
                if value is not None:
                    score_label = score_type.replace('_', ' ').title()
                    stats.append(f"- **{score_label} Score:** {value:.3f}")

        # Add other statistics from 'other' fields
        other_data = parsed_data.get('other', {})
        for key, value in other_data.items():
            if isinstance(value, (int, float)) and value != 0:
                # Clean up the key name
                clean_key = key.replace('_', ' ').title()
                if isinstance(value, float):
                    stats.append(f"- **{clean_key}:** {value:.3f}")
                else:
                    stats.append(f"- **{clean_key}:** {value:,}")

        return "\n".join(stats) if stats else ""

    def _format_aggregate_risk_analysis(self, parsed_data: Dict) -> str:
        """Format risk analysis section"""
        lines = []

        scores = parsed_data.get('scores', {})
        total = parsed_data.get('totals', {}).get('total', 0)

        if scores and total > 0:
            # Analyze risk levels
            for score_type, avg_score in scores.items():
                if avg_score is not None:
                    score_label = score_type.replace('_', ' ').title()

                    # Risk categorization
                    if avg_score > 0.7:
                        risk_level = "Very High"
                        emoji = "🚨"
                    elif avg_score > 0.5:
                        risk_level = "High"
                        emoji = "⚠️"
                    elif avg_score > 0.3:
                        risk_level = "Moderate"
                        emoji = "📌"
                    else:
                        risk_level = "Low"
                        emoji = "✓"

                    lines.append(f"{emoji} **{score_label} Score Risk Level:** {risk_level} (Average: {avg_score:.3f})")

            # Key risk factors based on demographics
            lines.append("\n### Risk Factors Identified:")

            # Youth risk
            age_groups = parsed_data['demographics']['age_groups']
            youth_count = sum(v for k, v in age_groups.items() if k in ['20-30', '30-40'])
            if youth_count > total * 0.7:
                lines.append(f"- **Youth Vulnerability:** High concentration in 20-40 age range")

            # Single status risk
            marital_data = parsed_data['demographics']['marital_status']
            single_count = marital_data.get('single', 0)
            if single_count > total * 0.6:
                lines.append(f"- **Social Stability:** Majority are single, potentially less family support")

            # Foreign national risk
            nat_data = parsed_data['demographics']['nationality']
            if nat_data.get('non_uae', 0) > total * 0.8:
                lines.append(f"- **Immigration Status:** High proportion of foreign nationals")

            # Gender concentration
            gender_data = parsed_data['demographics']['gender']
            male_count = gender_data.get('male', 0)
            if male_count > total * 0.7:
                lines.append(f"- **Gender Concentration:** Predominantly male population")

        return "\n".join(lines)

    def _generate_recommendations(self, parsed_data: Dict, analysis_type: str) -> List[str]:
        """Generate actionable recommendations based on the analysis"""
        recommendations = []

        total = parsed_data.get('totals', {}).get('total', 0)
        if total == 0:
            return recommendations

        # Drug-related recommendations
        if 'Drug' in analysis_type:
            scores = parsed_data.get('scores', {})
            avg_score = scores.get('drug_dealing', 0) or scores.get('drug_addict', 0)

            if avg_score > 0.7:
                recommendations.append(
                    "1. **Immediate Action Required:** Deploy targeted enforcement in identified areas")
                recommendations.append(
                    "2. **Intelligence Gathering:** Investigate networks among high-risk individuals")

            # Youth-focused recommendations
            age_groups = parsed_data['demographics']['age_groups']
            youth_count = sum(v for k, v in age_groups.items() if k in ['20-30'])
            if youth_count > total * 0.5:
                recommendations.append(
                    "3. **Youth Prevention:** Implement targeted intervention programs for 20-30 age group")

            # Foreign national recommendations
            nat_data = parsed_data['demographics']['nationality']
            if nat_data.get('non_uae', 0) == total:
                recommendations.append(
                    "4. **Immigration Coordination:** Review visa status and coordinate with immigration authorities")

            # Gender-specific recommendations
            gender_data = parsed_data['demographics']['gender']
            if gender_data.get('female', 0) > total * 0.25:
                recommendations.append(
                    "5. **Gender-Sensitive Approach:** Develop gender-specific intervention strategies")

        # General risk recommendations
        else:
            recommendations.append("1. **Risk Monitoring:** Establish continuous monitoring for high-risk individuals")
            recommendations.append("2. **Data Quality:** Investigate uniform risk scores if all values are identical")
            recommendations.append("3. **Cross-Agency Coordination:** Share findings with relevant security agencies")

        # Location-specific recommendations
        if 'abu dhabi' in analysis_type.lower():
            recommendations.append(
                f"6. **Geographic Focus:** Concentrate resources in Abu Dhabi with {total:,} identified individuals")

        return recommendations

    def _classify_query_type(self, query: str, sql_query: str = None, results: List[Dict] = None) -> str:
        """Classify the type of query for appropriate summarization"""

        query_lower = query.lower()
        sql_lower = sql_query.lower() if sql_query else ""

        # Check for specific patterns
        if any(word in query_lower for word in ['top', 'highest', 'most', 'maximum']):
            return 'ranking_high'
        elif any(word in query_lower for word in ['bottom', 'lowest', 'least', 'minimum']):
            return 'ranking_low'
        elif 'distribution' in query_lower or 'breakdown' in query_lower:
            return 'distribution'
        elif any(word in query_lower for word in ['average', 'mean', 'statistics', 'stats']):
            return 'statistics'
        elif 'group by' in sql_lower:
            return 'aggregation'
        elif any(word in query_lower for word in ['risk', 'criminal', 'crime', 'score']):
            return 'risk_analysis'
        elif any(word in query_lower for word in ['demographic', 'age', 'gender', 'nationality']):
            return 'demographic'
        elif results and len(results) == 1:
            # If we have exactly one result, it might be an individual lookup
            return 'individual'
        else:
            return 'general'

    def _detect_fields(self, results: List[Dict]) -> set:
        """Detect all fields present in the results"""

        fields = set()
        normalized_fields = set()

        for row in results[:10]:  # Sample first 10 rows
            for key in row.keys():
                fields.add(key)

                # Handle SQL aliases like "gender_en AS Gender"
                if ' AS ' in key:
                    # Extract the original column name
                    original_col = key.split(' AS ')[0].strip()
                    normalized_fields.add(original_col)
                else:
                    normalized_fields.add(key)

        # Return both original keys and normalized field names
        return fields.union(normalized_fields)

    def _generate_header(self, query: str, results: List[Dict], query_type: str) -> str:
        """Generate header section with overview"""

        header = "# 📊 Query Summary\n\n"
        header += f"**Query:** {query}\n\n"
        header += f"**Results Found:** {len(results):,} records\n"
        header += f"**Analysis Type:** {query_type.replace('_', ' ').title()}\n"

        return header

    def _generate_insights(self, results: List[Dict], fields: set, query_type: str) -> str:
        """Generate key insights based on patterns"""

        insights = []

        # Check if this is an aggregated result
        has_count_field = any('COUNT' in str(f).upper() for f in fields)
        has_avg_field = any('AVG' in str(f).upper() for f in fields)

        # For aggregated results, show different insights
        if has_count_field:
            # Total records represented
            total_count = sum(self._get_field_value(r, 'count') or 0 for r in results)
            if total_count > 0:
                insights.append(f"📊 **Total records aggregated: {total_count:,}**")

            # Top group
            if results:
                top_group = results[0]
                count_val = self._get_field_value(top_group, 'count')
                group_desc = []

                # Build description of top group
                for field in ['gender_en', 'age_group', 'nationality_name_en', 'marital_status_en']:
                    val = self._get_field_value(top_group, field)
                    if val:
                        group_desc.append(str(val))

                if group_desc and count_val:
                    insights.append(f"🎯 **Top group:** {' / '.join(group_desc)} ({count_val:,} records)")

        # Check for drug dealing specific insights
        for key in fields:
            if 'drug_dealing_score' in key.lower() and 'AVG' in key.upper():
                # Calculate average across all results
                total_count = sum(self._get_field_value(r, 'count') or 0 for r in results)
                weighted_sum = sum((self._get_field_value(r, 'count') or 0) *
                                   (self._get_field_value(r, key) or 0) for r in results)
                if total_count > 0:
                    overall_avg = weighted_sum / total_count
                    insights.append(f"💊 **Average drug dealing score: {overall_avg:.3f}** (very high risk)")

        # Pattern 1: High-risk concentration
        if 'risk_score' in fields:
            high_risk = sum(1 for r in results if (self._get_field_value(r, 'risk_score') or 0) > 0.7)
            if high_risk > 0:
                percentage = (high_risk / len(results)) * 100
                insights.append(f"🚨 **{high_risk:,} profiles ({percentage:.1f}%)** have high risk scores (>0.7)")

        # Pattern 2: Criminal cases
        if 'has_crime_case' in fields:
            crime_cases = sum(1 for r in results if self._get_field_value(r, 'has_crime_case'))
            if crime_cases > 0:
                percentage = (crime_cases / len(results)) * 100
                insights.append(f"⚖️ **{crime_cases:,} profiles ({percentage:.1f}%)** have criminal cases")

        # Pattern 3: Investigation cases
        if 'has_investigation_case' in fields:
            investigation_cases = sum(1 for r in results if self._get_field_value(r, 'has_investigation_case'))
            if investigation_cases > 0:
                percentage = (investigation_cases / len(results)) * 100
                insights.append(f"🔍 **{investigation_cases:,} profiles ({percentage:.1f}%)** are under investigation")

        # Pattern 4: Prison status
        if 'is_in_prison' in fields:
            in_prison = sum(1 for r in results if self._get_field_value(r, 'is_in_prison'))
            if in_prison > 0:
                insights.append(f"🔒 **{in_prison:,} profiles** are currently in prison")

        # Pattern 5: Visitor overstay
        if 'residency_status' in fields and 'dwell_duration_tag' in fields:
            long_visitors = sum(1 for r in results
                                if self._get_field_value(r, 'residency_status') == 'VISITOR'
                                and 'more_than' in str(self._get_field_value(r, 'dwell_duration_tag') or '').lower())
            if long_visitors > 0:
                insights.append(f"✈️ **{long_visitors:,} visitors** may be overstaying")

        # Pattern 6: Drug-related risks
        drug_fields = ['drug_dealing_score', 'drug_addict_score']
        for field in drug_fields:
            if field in fields:
                high_drug_risk = sum(1 for r in results if (self._get_field_value(r, field) or 0) > 0.5)
                if high_drug_risk > 0:
                    risk_type = "dealing" if "dealing" in field else "addiction"
                    insights.append(f"💊 **{high_drug_risk:,} profiles** show high drug {risk_type} risk")

        # Pattern 7: Age demographics
        if 'age_group' in fields:
            age_groups = Counter(self._get_field_value(r, 'age_group') for r in results
                                 if self._get_field_value(r, 'age_group'))
            if age_groups:
                dominant_age = age_groups.most_common(1)[0]
                if has_count_field:
                    # For aggregated data, sum the counts
                    age_totals = defaultdict(int)
                    for r in results:
                        age = self._get_field_value(r, 'age_group')
                        count = self._get_field_value(r, 'count') or 1
                        if age:
                            age_totals[age] += count
                    if age_totals:
                        dominant_age = max(age_totals.items(), key=lambda x: x[1])
                        insights.append(
                            f"👥 **{dominant_age[0]}** is the dominant age group ({dominant_age[1]:,} profiles)")
                else:
                    insights.append(f"👥 **{dominant_age[0]}** is the dominant age group ({dominant_age[1]:,} profiles)")

        # Pattern 8: Nationality concentration
        if 'nationality_code' in fields or 'nationality_name_en' in fields:
            if has_count_field:
                # Aggregated data - sum by nationality
                nat_totals = defaultdict(int)
                for r in results:
                    nat = self._get_field_value(r, 'nationality_name_en') or self._get_field_value(r,
                                                                                                   'nationality_code')
                    count = self._get_field_value(r, 'count') or 1
                    if nat:
                        nat_totals[nat] += count

                if nat_totals:
                    top_nat = sorted(nat_totals.items(), key=lambda x: x[1], reverse=True)[:3]
                    nat_str = ", ".join([f"{n[0]} ({n[1]:,})" for n in top_nat])
                    insights.append(f"🌍 **Top nationalities:** {nat_str}")
            else:
                nationalities = Counter(self._get_field_value(r, 'nationality_code') or
                                        self._get_field_value(r, 'nationality_name_en')
                                        for r in results
                                        if self._get_field_value(r, 'nationality_code') or
                                        self._get_field_value(r, 'nationality_name_en'))
                if nationalities:
                    top_nat = nationalities.most_common(3)
                    nat_str = ", ".join([f"{n[0]} ({n[1]:,})" for n in top_nat])
                    insights.append(f"🌍 **Top nationalities:** {nat_str}")

        # Pattern 9: Application usage
        if 'applications_used' in fields:
            all_apps = []
            for r in results:
                apps = self._get_field_value(r, 'applications_used') or []
                if isinstance(apps, str):
                    apps = [app.strip() for app in apps.split(',') if app.strip()]
                elif isinstance(apps, list):
                    all_apps.extend(apps)

            if all_apps:
                app_counter = Counter(all_apps)
                top_apps = app_counter.most_common(3)
                if top_apps:
                    app_str = ", ".join([f"{app[0]} ({app[1]:,} users)" for app in top_apps])
                    insights.append(f"📱 **Most used apps:** {app_str}")

        # Pattern 10: Geographic concentration
        if 'home_city' in fields:
            if has_count_field:
                # Aggregated data
                city_totals = defaultdict(int)
                for r in results:
                    city = self._get_field_value(r, 'home_city')
                    count = self._get_field_value(r, 'count') or 1
                    if city:
                        city_totals[city] += count

                if city_totals:
                    total = sum(city_totals.values())
                    top_city = max(city_totals.items(), key=lambda x: x[1])
                    percentage = (top_city[1] / total) * 100
                    insights.append(f"🏙️ **{percentage:.1f}%** are from {top_city[0]}")
            else:
                cities = Counter(self._get_field_value(r, 'home_city') for r in results
                                 if self._get_field_value(r, 'home_city'))
                if cities:
                    top_city = cities.most_common(1)[0]
                    percentage = (top_city[1] / len(results)) * 100
                    insights.append(f"🏙️ **{percentage:.1f}%** are from {top_city[0]}")

        # Pattern 11: Gender distribution for aggregated data
        if 'gender_en' in fields and has_count_field:
            gender_totals = defaultdict(int)
            for r in results:
                gender = self._get_field_value(r, 'gender_en')
                count = self._get_field_value(r, 'count') or 1
                if gender:
                    gender_totals[gender] += count

            if gender_totals:
                total = sum(gender_totals.values())
                gender_str = ", ".join([f"{g}: {c:,} ({c / total * 100:.1f}%)"
                                        for g, c in sorted(gender_totals.items(),
                                                           key=lambda x: x[1],
                                                           reverse=True)])
                insights.append(f"👤 **Gender distribution:** {gender_str}")

        # Pattern 12: Marital status for aggregated data
        if 'marital_status_en' in fields and has_count_field:
            marital_totals = defaultdict(int)
            for r in results:
                status = self._get_field_value(r, 'marital_status_en')
                count = self._get_field_value(r, 'count') or 1
                if status:
                    marital_totals[status] += count

            if marital_totals:
                total = sum(marital_totals.values())
                top_status = max(marital_totals.items(), key=lambda x: x[1])
                insights.append(
                    f"💑 **Most common marital status:** {top_status[0]} ({top_status[1]:,}, {top_status[1] / total * 100:.1f}%)")

        return "\n".join(f"- {insight}" for insight in insights)

    def _generate_statistics(self, results: List[Dict], fields: set) -> str:
        """Generate statistical analysis for numeric fields"""

        stats_lines = []

        # Check for aggregate functions first (AVG, SUM, etc.)
        for field in fields:
            field_str = str(field)

            # Skip if it's an aggregate function result
            if any(agg in field_str.upper() for agg in ['AVG(', 'SUM(', 'COUNT(', 'MAX(', 'MIN(']):
                # Extract the metric name
                metric_name = field_str
                if ' AS ' in field_str:
                    metric_name = field_str.split(' AS ')[-1].strip()

                values = []
                for r in results:
                    val = r.get(field)
                    if val is not None:
                        try:
                            values.append(float(val))
                        except (ValueError, TypeError):
                            continue

                if values and 'COUNT' not in field_str.upper():
                    stats_lines.append(f"### {metric_name}")
                    stats_lines.append(f"- **Range:** {min(values):.3f} - {max(values):.3f}")
                    if len(values) > 1:
                        stats_lines.append(f"- **Average:** {statistics.mean(values):.3f}")
                        stats_lines.append(f"- **Std Dev:** {statistics.stdev(values):.3f}")
                    stats_lines.append("")
                continue

        # Analyze each numeric field in the original data
        for field in self.numeric_fields:
            values = []
            for r in results:
                val = self._get_field_value(r, field)
                if val is not None:
                    try:
                        values.append(float(val))
                    except (ValueError, TypeError):
                        continue

            if values:
                stats = self._calculate_statistics(values)
                stats_lines.append(f"### {field.replace('_', ' ').title()}")
                stats_lines.append(f"- **Mean:** {stats['mean']:.3f}")
                stats_lines.append(f"- **Median:** {stats['median']:.3f}")
                stats_lines.append(f"- **Std Dev:** {stats['std_dev']:.3f}")
                stats_lines.append(f"- **Min:** {stats['min']:.3f}")
                stats_lines.append(f"- **Max:** {stats['max']:.3f}")

                # Add percentiles for risk scores
                if 'score' in field:
                    stats_lines.append(f"- **90th Percentile:** {stats['p90']:.3f}")
                    stats_lines.append(f"- **95th Percentile:** {stats['p95']:.3f}")

                stats_lines.append("")

        return "\n".join(stats_lines)

    def _calculate_statistics(self, values: List[float]) -> Dict[str, float]:
        """Calculate comprehensive statistics for a numeric field"""

        if not values:
            return {}

        sorted_values = sorted(values)
        n = len(sorted_values)

        return {
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std_dev': statistics.stdev(values) if n > 1 else 0,
            'min': min(values),
            'max': max(values),
            'p90': sorted_values[int(n * 0.9)] if n > 0 else 0,
            'p95': sorted_values[int(n * 0.95)] if n > 0 else 0
        }

    def _generate_distributions(self, results: List[Dict], fields: set) -> str:
        """Generate distribution analysis for categorical fields"""

        dist_lines = []

        # Check if this is aggregated data
        has_count_field = any('COUNT' in str(f).upper() for f in fields)

        # For aggregated data, show the actual aggregation as a table
        if has_count_field:
            # Find grouping fields (non-aggregate fields)
            grouping_fields = []
            value_fields = []

            for field in fields:
                field_str = str(field)
                if any(agg in field_str.upper() for agg in ['AVG(', 'SUM(', 'COUNT(', 'MAX(', 'MIN(']):
                    value_fields.append(field)
                else:
                    grouping_fields.append(field)

            if grouping_fields and results:
                dist_lines.append("### Aggregated Results by Demographics")

                # Build header
                headers = []
                for gf in grouping_fields:
                    if ' AS ' in gf:
                        headers.append(gf.split(' AS ')[-1].strip())
                    else:
                        headers.append(gf.replace('_', ' ').title())

                for vf in value_fields:
                    if ' AS ' in vf:
                        headers.append(vf.split(' AS ')[-1].strip())
                    else:
                        headers.append(vf)

                dist_lines.append("| " + " | ".join(headers) + " |")
                dist_lines.append("|" + "|".join(["-" * (len(h) + 2) for h in headers]) + "|")

                # Add rows (limit to top 15)
                for row in results[:15]:
                    values = []
                    for gf in grouping_fields:
                        val = row.get(gf, "")
                        if val is None:
                            val = "N/A"
                        values.append(str(val))

                    for vf in value_fields:
                        val = row.get(vf, "")
                        if isinstance(val, float):
                            val = f"{val:.3f}"
                        elif val is None:
                            val = "0"
                        values.append(str(val))

                    dist_lines.append("| " + " | ".join(values) + " |")

                if len(results) > 15:
                    dist_lines.append(f"| *...and {len(results) - 15} more rows* |")

                dist_lines.append("")

        # Analyze categorical fields for non-aggregated data
        if not has_count_field:
            for field in self.categorical_fields:
                values = []
                for r in results:
                    val = self._get_field_value(r, field)
                    if val:
                        values.append(val)

                if values:
                    distribution = Counter(values)
                    total = sum(distribution.values())

                    dist_lines.append(f"### {field.replace('_', ' ').title()} Distribution")
                    dist_lines.append("| Value | Count | Percentage |")
                    dist_lines.append("|-------|-------|------------|")

                    for value, count in distribution.most_common(10):
                        percentage = (count / total) * 100
                        dist_lines.append(f"| {value} | {count:,} | {percentage:.1f}% |")

                    if len(distribution) > 10:
                        dist_lines.append(f"| *...and {len(distribution) - 10} more* | | |")

                    dist_lines.append("")

        # Analyze boolean fields
        bool_summary = []
        for field in self.boolean_fields:
            true_count = sum(1 for r in results if self._get_field_value(r, field))
            if true_count > 0:
                percentage = (true_count / len(results)) * 100
                bool_summary.append(f"- **{field.replace('_', ' ').title()}:** {true_count:,} ({percentage:.1f}%)")

        if bool_summary:
            dist_lines.append("### Boolean Flags")
            dist_lines.extend(bool_summary)

        return "\n".join(dist_lines)

    def _analyze_patterns(self, results: List[Dict], fields: set) -> str:
        """Analyze complex patterns in the data"""

        patterns = []

        # Pattern 11: Risk correlations
        if all(f in fields for f in ['risk_score', 'nationality_code']):
            nat_risks = defaultdict(list)
            for r in results:
                if r.get('nationality_code') and r.get('risk_score') is not None:
                    nat_risks[r['nationality_code']].append(r['risk_score'])

            # Find nationalities with highest average risk
            nat_avg_risk = {nat: statistics.mean(scores) for nat, scores in nat_risks.items() if scores}
            if nat_avg_risk:
                top_risk_nats = sorted(nat_avg_risk.items(), key=lambda x: x[1], reverse=True)[:5]
                if top_risk_nats:
                    patterns.append("### Risk by Nationality")
                    patterns.append("| Nationality | Avg Risk Score |")
                    patterns.append("|-------------|----------------|")
                    for nat, avg_risk in top_risk_nats:
                        patterns.append(f"| {nat} | {avg_risk:.3f} |")
                    patterns.append("")

        # Pattern 12: Crime category analysis
        if 'crime_categories_en' in fields:
            all_crimes = []
            for r in results:
                crimes = r.get('crime_categories_en', [])
                if isinstance(crimes, str):
                    crimes = [c.strip() for c in crimes.split(',') if c.strip()]
                elif isinstance(crimes, list):
                    all_crimes.extend(crimes)

            if all_crimes:
                crime_dist = Counter(all_crimes)
                patterns.append("### Crime Categories")
                patterns.append("| Category | Occurrences |")
                patterns.append("|----------|-------------|")
                for crime, count in crime_dist.most_common(10):
                    patterns.append(f"| {crime} | {count:,} |")
                patterns.append("")

        # Pattern 13: Travel patterns
        if 'travelled_country_codes' in fields:
            travel_freq = defaultdict(int)
            for r in results:
                countries = r.get('travelled_country_codes', [])
                if isinstance(countries, str):
                    countries = [c.strip() for c in countries.split(',') if c.strip()]
                for country in countries:
                    travel_freq[country] += 1

            if travel_freq:
                patterns.append("### Travel Destinations")
                patterns.append("| Country | Travelers |")
                patterns.append("|---------|-----------|")
                for country, count in sorted(travel_freq.items(), key=lambda x: x[1], reverse=True)[:10]:
                    patterns.append(f"| {country} | {count:,} |")
                patterns.append("")

        # Pattern 14: Communication patterns
        if 'communicated_country_codes' in fields:
            comm_freq = defaultdict(int)
            for r in results:
                countries = r.get('communicated_country_codes', [])
                if isinstance(countries, str):
                    countries = [c.strip() for c in countries.split(',') if c.strip()]
                for country in countries:
                    comm_freq[country] += 1

            if comm_freq:
                patterns.append("### Communication Countries")
                patterns.append("| Country | Users |")
                patterns.append("|---------|-------|")
                for country, count in sorted(comm_freq.items(), key=lambda x: x[1], reverse=True)[:10]:
                    patterns.append(f"| {country} | {count:,} |")
                patterns.append("")

        # Pattern 15: Employment patterns
        if 'latest_job_title_en' in fields:
            jobs = [r.get('latest_job_title_en') for r in results if r.get('latest_job_title_en')]
            if jobs:
                # Extract common job keywords
                job_words = []
                for job in jobs:
                    words = re.findall(r'\b[A-Za-z]+\b', job.lower())
                    job_words.extend(words)

                # Filter out common words
                common_words = {'and', 'or', 'the', 'of', 'in', 'at', 'to', 'for', 'a', 'an'}
                job_words = [w for w in job_words if w not in common_words and len(w) > 3]

                if job_words:
                    word_freq = Counter(job_words).most_common(10)
                    patterns.append("### Common Job Keywords")
                    patterns.append("| Keyword | Frequency |")
                    patterns.append("|---------|-----------|")
                    for word, count in word_freq:
                        patterns.append(f"| {word.title()} | {count:,} |")
                    patterns.append("")

    def _should_show_profiles_table(self, query_type: str, fields: set) -> bool:
        """Determine if we should show a profiles table"""

        # Show for ranking queries or when we have profile identifiers
        return (query_type in ['ranking_high', 'ranking_low', 'risk_analysis'] or
                any(f in fields for f in self.identifier_fields))

    def _generate_profiles_table(self, results: List[Dict]) -> str:
        """Generate a table of top profiles"""

        if not results:
            return ""

        # Limit to top 10
        top_results = results[:10]

        # Determine which columns to show
        columns = []
        headers = []

        # Priority columns
        priority_cols = [
            ('fullname_en', 'Name'),
            ('phone_no', 'Phone'),
            ('nationality_code', 'Nat'),
            ('age_group', 'Age'),
            ('risk_score', 'Risk'),
            ('has_crime_case', 'Crime'),
            ('residency_status', 'Status'),
            ('home_city', 'City')
        ]

        for col, header in priority_cols:
            if col in results[0]:
                columns.append(col)
                headers.append(header)

        if not columns:
            return ""

        # Build table
        table_lines = []
        table_lines.append("| " + " | ".join(headers) + " |")
        table_lines.append("|" + "|".join(["-" * (len(h) + 2) for h in headers]) + "|")

        for row in top_results:
            values = []
            for col in columns:
                value = row.get(col, "")

                # Format specific fields
                if col == 'risk_score' and value is not None:
                    value = f"{float(value):.3f}"
                elif col == 'has_crime_case':
                    value = "✓" if value else ""
                elif col == 'fullname_en' and value:
                    # Truncate long names
                    value = value[:20] + "..." if len(value) > 20 else value
                elif value is None:
                    value = ""
                else:
                    value = str(value)

                values.append(value)

            table_lines.append("| " + " | ".join(values) + " |")

        return "\n".join(table_lines)

    def _generate_risk_analysis(self, results: List[Dict], fields: set) -> str:
        """Generate specific risk analysis section"""

        risk_lines = []

        # Risk score distribution
        risk_fields = ['risk_score', 'drug_dealing_score', 'drug_addict_score', 'murder_score']
        available_risk_fields = [f for f in risk_fields if f in fields]

        if available_risk_fields:
            risk_lines.append("### Risk Score Distribution")

            for field in available_risk_fields:
                scores = [r.get(field, 0) for r in results if r.get(field) is not None]
                if scores:
                    # Categorize scores
                    high = sum(1 for s in scores if s > 0.7)
                    medium = sum(1 for s in scores if 0.3 < s <= 0.7)
                    low = sum(1 for s in scores if s <= 0.3)

                    risk_lines.append(f"\n**{field.replace('_', ' ').title()}:**")
                    risk_lines.append(f"- High Risk (>0.7): {high:,} profiles")
                    risk_lines.append(f"- Medium Risk (0.3-0.7): {medium:,} profiles")
                    risk_lines.append(f"- Low Risk (≤0.3): {low:,} profiles")

            # Analyze risk rules if present
            rule_fields = ['risk_rules', 'drug_dealing_rules', 'drug_addict_rules', 'murder_rules']
            for rule_field in rule_fields:
                if rule_field in fields:
                    all_rules = []
                    for r in results:
                        rules = r.get(rule_field, [])
                        if isinstance(rules, str):
                            rules = [rule.strip() for rule in rules.split(',') if rule.strip()]
                        elif isinstance(rules, list):
                            all_rules.extend(rules)

                    if all_rules:
                        rule_counter = Counter(all_rules)
                        top_rules = rule_counter.most_common(5)
                        if top_rules:
                            risk_lines.append(f"\n### Top {rule_field.replace('_', ' ').title()}")
                            for rule, count in top_rules:
                                risk_lines.append(f"- {rule}: {count:,} occurrences")

        return "\n".join(risk_lines)
