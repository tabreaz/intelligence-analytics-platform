# src/agents/profile_analyzer/agent.py
import re
from typing import Dict, List, Any, Tuple

from src.agents.base_agent import BaseAgent, AgentRequest, AgentResponse, AgentStatus
from src.agents.profile_analyzer.query_analyzer import IntelligentResultSummarizer
from src.agents.profile_analyzer.result_summarizer import ResultSummarizer
from src.core.config_manager import ConfigManager
from src.core.database.clickhouse_client import ClickHouseClient
from src.core.llm.base_llm import LLMClientFactory
from src.core.logger import get_logger

logger = get_logger(__name__)


class ProfileAnalyzerAgent(BaseAgent):
    """Agent for analyzing user profiles and risk scores using natural language queries"""

    def __init__(self, name: str, config: dict, config_manager: ConfigManager):
        super().__init__(name, config)
        self.config_manager = config_manager

        # Initialize LLM client
        llm_config = config_manager.get_llm_config()
        self.llm_client = LLMClientFactory.create_client(llm_config)

        # Initialize ClickHouse client (use single connection for initialization)
        ch_config = config_manager.get_database_config('clickhouse')
        self.ch_client = ClickHouseClient(ch_config, use_pool=False)

        # Initialize the smart summarizer
        self.summarizer = ResultSummarizer()

        # Initialize the intelligent LLM-based summarizer
        self.intelligent_summarizer = IntelligentResultSummarizer(self.llm_client)

        # Load all metadata at initialization
        self.metadata = self._load_all_metadata()

        # Build complete schema context with metadata
        self.enhanced_schema_context = self._build_enhanced_schema_context()

        logger.info(f"ProfileAnalyzerAgent initialized with metadata loaded and smart summarizer")

    def _load_all_metadata(self) -> Dict[str, Any]:
        """Load all metadata from database at initialization"""

        metadata = {}

        try:
            # Load score ranges
            score_ranges_query = """
                                 SELECT 'risk_score'    as score_type, \
                                        MIN(risk_score) as min_val, \
                                        MAX(risk_score) as max_val
                                 FROM telecom_db.phone_imsi_uid_latest
                                 UNION ALL
                                 SELECT 'drug_dealing_score', MIN(drug_dealing_score), MAX(drug_dealing_score)
                                 FROM telecom_db.phone_imsi_uid_latest
                                 UNION ALL
                                 SELECT 'drug_addict_score', MIN(drug_addict_score), MAX(drug_addict_score)
                                 FROM telecom_db.phone_imsi_uid_latest
                                 UNION ALL
                                 SELECT 'murder_score', MIN(murder_score), MAX(murder_score)
                                 FROM telecom_db.phone_imsi_uid_latest \
                                 """

            score_results = self.ch_client.execute(score_ranges_query)
            metadata['score_ranges'] = {row[0]: {'min': row[1], 'max': row[2]} for row in score_results}
            logger.info(f"Loaded score ranges: {metadata['score_ranges']}")

            # Load enum values
            enum_queries = {
                'gender': "SELECT DISTINCT gender_en FROM telecom_db.phone_imsi_uid_latest WHERE gender_en IS NOT NULL",
                'age_group': "SELECT DISTINCT age_group FROM telecom_db.phone_imsi_uid_latest WHERE age_group IS NOT NULL ORDER BY age_group",
                'residency_status': "SELECT DISTINCT residency_status FROM telecom_db.phone_imsi_uid_latest WHERE residency_status IS NOT NULL",
                'marital_status': "SELECT DISTINCT marital_status_en FROM telecom_db.phone_imsi_uid_latest WHERE marital_status_en IS NOT NULL",
                'dwell_duration': "SELECT DISTINCT dwell_duration_tag FROM telecom_db.phone_imsi_uid_latest WHERE dwell_duration_tag IS NOT NULL"
            }

            for key, query in enum_queries.items():
                results = self.ch_client.execute(query)
                metadata[key] = [row[0] for row in results]
                logger.info(f"Loaded {key}: {metadata[key]}")

            # Load cities
            cities_query = """
                           SELECT DISTINCT home_city
                           FROM telecom_db.phone_imsi_uid_latest
                           WHERE home_city != ''
                           ORDER BY home_city \
                           """
            cities_results = self.ch_client.execute(cities_query)
            metadata['cities'] = [row[0] for row in cities_results]
            logger.info(f"Loaded {len(metadata['cities'])} cities")

            # Load applications with usage count
            apps_query = """
                         SELECT arrayJoin(applications_used) as app, \
                                COUNT(*)                     as usage_count
                         FROM telecom_db.phone_imsi_uid_latest
                         WHERE notEmpty(applications_used)
                         GROUP BY app
                         ORDER BY usage_count DESC \
                         """
            apps_results = self.ch_client.execute(apps_query)
            metadata['applications'] = [(row[0], row[1]) for row in apps_results]
            logger.info(f"Loaded {len(metadata['applications'])} applications")

            # Load crime categories
            crime_cat_query = """
                              SELECT DISTINCT arrayJoin(crime_categories_en) as category
                              FROM telecom_db.phone_imsi_uid_latest
                              WHERE notEmpty(crime_categories_en)
                              ORDER BY category \
                              """
            crime_cat_results = self.ch_client.execute(crime_cat_query)
            metadata['crime_categories'] = [row[0] for row in crime_cat_results]
            logger.info(f"Loaded {len(metadata['crime_categories'])} crime categories")

            # Load crime subcategories
            crime_subcat_query = """
                                 SELECT DISTINCT arrayJoin(crime_sub_categories_en) as subcategory
                                 FROM telecom_db.phone_imsi_uid_latest
                                 WHERE notEmpty(crime_sub_categories_en)
                                 ORDER BY subcategory \
                                 """
            crime_subcat_results = self.ch_client.execute(crime_subcat_query)
            metadata['crime_sub_categories'] = [row[0] for row in crime_subcat_results]
            logger.info(f"Loaded {len(metadata['crime_sub_categories'])} crime subcategories")

            # Load driving license types
            license_query = """
                            SELECT DISTINCT arrayJoin(driving_license_type) as license
                            FROM telecom_db.phone_imsi_uid_latest
                            WHERE notEmpty(driving_license_type) \
                            """
            license_results = self.ch_client.execute(license_query)
            metadata['driving_licenses'] = [row[0] for row in license_results]
            logger.info(f"Loaded {len(metadata['driving_licenses'])} license types")

            # Load sample nationality codes with names
            nationality_query = """
                                SELECT DISTINCT nationality_code, nationality_name_en
                                FROM telecom_db.phone_imsi_uid_latest
                                WHERE nationality_code != '' \
                                  AND nationality_name_en != ''
                                ORDER BY nationality_code
                                LIMIT 50 \
                                """
            nationality_results = self.ch_client.execute(nationality_query)
            metadata['nationalities'] = {row[0]: row[1] for row in nationality_results}
            logger.info(f"Loaded {len(metadata['nationalities'])} nationality codes")

        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            # Provide fallback values
            metadata = self._get_fallback_metadata()

        return metadata

    def _get_fallback_metadata(self) -> Dict[str, Any]:
        """Provide fallback metadata if database loading fails"""

        return {
            'score_ranges': {
                'risk_score': {'min': 0, 'max': 0.27},
                'drug_dealing_score': {'min': 0, 'max': 0.9},
                'drug_addict_score': {'min': 0, 'max': 0.8},
                'murder_score': {'min': 0, 'max': 0}
            },
            'gender': ['Male', 'Female'],
            'age_group': ['20-30', '30-40', '40-50', '50-60', '60-70'],
            'residency_status': ['CITIZEN', 'RESIDENT', 'VISITOR', 'INACTIVE'],
            'marital_status': ['SINGLE', 'MARRIED', 'DIVORCED', 'WIDOWED'],
            'cities': ['Dubai', 'AbuDhabi', 'Sharjah', 'Ajman'],
            'applications': [],
            'crime_categories': [],
            'crime_sub_categories': [],
            'driving_licenses': ['Light Vehicle'],
            'nationalities': {}
        }

    def _build_enhanced_schema_context(self) -> str:
        """Build complete schema context with all metadata"""

        # Format score ranges
        score_info = []
        for score_type, ranges in self.metadata.get('score_ranges', {}).items():
            score_info.append(f"  - {score_type}: {ranges['min']:.2f} to {ranges['max']:.2f}")

        # Format applications (top 30)
        apps = self.metadata.get('applications', [])
        top_apps = [app[0] for app in apps[:30]]

        # Format nationalities sample
        nationalities = self.metadata.get('nationalities', {})
        nat_sample = list(nationalities.items())[:10]
        nat_str = ", ".join([f"{code} ({name})" for code, name in nat_sample])

        context = f"""
Table: telecom_db.phone_imsi_uid_latest
Description: User profile data with risk scores and demographics

Complete Column List:
- imsi (String): Primary identifier
- phone_no (String): Phone number
- uid (String): Unique user ID
- eid (Array[String]): Emirates IDs
- fullname_en (String): Full name in English
- gender_en (Enum): {', '.join(self.metadata.get('gender', ['Male', 'Female']))}
- date_of_birth (Date): Birth date
- age (UInt8): Age in years
- age_group (Enum): {', '.join(self.metadata.get('age_group', []))}
- marital_status_en (Enum): {', '.join(self.metadata.get('marital_status', []))}
- nationality_code (FixedString(3)): ISO 3-letter country code
- nationality_name_en (String): Country name
- previous_nationality_code (FixedString(3)): Previous nationality
- previous_nationality_en (String): Previous nationality name
- residency_status (Enum): {', '.join(self.metadata.get('residency_status', []))}
- dwell_duration_tag (Enum): {', '.join(self.metadata.get('dwell_duration', []))}
- home_city (String): {', '.join(self.metadata.get('cities', [])[:7])} (total: {len(self.metadata.get('cities', []))})
- home_location (String): Geohash of home
- work_location (String): Geohash of work
- latest_sponsor_name_en (String): Sponsor name
- latest_job_title_en (String): Job title
- last_travelled_country_code (FixedString(3)): Last travel destination
- travelled_country_codes (Array[FixedString(3)]): Countries visited
- communicated_country_codes (Array[FixedString(3)]): Countries called
- applications_used (Array[String]): Apps used
- driving_license_type (Array[String]): License types
- has_investigation_case (Bool): Under investigation flag
- has_crime_case (Bool): Has criminal record
- is_in_prison (Bool): Currently in prison
- crime_categories_en (Array[String]): Crime categories
- crime_sub_categories_en (Array[String]): Crime subcategories
- drug_addict_score (Float32): Drug addiction risk
- drug_dealing_score (Float32): Drug dealing risk
- murder_score (Float32): Murder risk
- risk_score (Float32): Overall risk score
- drug_addict_rules (Array[String]): Rules triggering drug addiction score
- drug_dealing_rules (Array[String]): Rules triggering drug dealing score
- murder_rules (Array[String]): Rules triggering murder score
- risk_rules (Array[String]): General risk rules

Score Ranges in Database:
{chr(10).join(score_info)}

Risk Categories (based on your data):
- High Risk: score > 0.2
- Medium Risk: score 0.1-0.2
- Low Risk: score < 0.1

Available Values:
- Applications (top 30): {', '.join(top_apps)}
- Crime Categories: {', '.join(self.metadata.get('crime_categories', []))}
- Crime Subcategories: {', '.join(self.metadata.get('crime_sub_categories', [])[:20])}...
- Driving License Types: {', '.join(self.metadata.get('driving_licenses', []))}
- Sample Nationalities: {nat_str} (and more...)

ClickHouse Syntax Reminders:
- Search in array: has(array_column, 'value')
- Array length: length(array_column)
- Array to string: arrayStringConcat(array_column, ', ')
- Case-sensitive matching for array values
"""

        return context

    async def validate_request(self, request: AgentRequest) -> bool:
        """Accept all requests for now - let other agents handle if not profile-related"""
        return True

    async def process(self, request: AgentRequest) -> AgentResponse:
        """Process profile analysis request"""

        try:
            # Generate SQL query using LLM
            sql_query = await self._generate_sql(request.prompt)

            if not sql_query:
                return self._create_error_response(
                    request.request_id,
                    "Could not generate SQL query from request"
                )

            # Start SQL execution and LLM analysis in PARALLEL
            import asyncio
            import time

            start_time = time.time()

            # Create tasks for parallel execution
            sql_task = asyncio.create_task(self._execute_sql_query(sql_query))

            # Only analyze if intelligent summary is enabled
            use_intelligent_summary = self.config.get('use_intelligent_summary', True)
            if use_intelligent_summary:
                analysis_task = asyncio.create_task(
                    self.intelligent_summarizer.query_analyzer.analyze_query_and_sql(
                        request.prompt, sql_query
                    )
                )

                # Wait for both to complete
                task_results = await asyncio.gather(sql_task, analysis_task)
                sql_result = task_results[0]  # This is a tuple (results, execution_time)
                analysis = task_results[1]

                results, execution_time = sql_result
            else:
                # Just wait for SQL if not using intelligent summary
                results, execution_time = await sql_task
                analysis = None

            logger.info(
                f"Parallel execution completed - SQL: {execution_time:.2f}s, Total: {time.time() - start_time:.2f}s")

            # Format results
            formatted_results = self._format_results(results, sql_query)

            # Generate summary with pre-computed analysis
            if use_intelligent_summary and analysis:
                # Use LLM-based intelligent summarizer with pre-computed analysis
                try:
                    summary = await self.intelligent_summarizer.generate_summary_with_analysis(
                        user_query=request.prompt,
                        sql_query=sql_query,
                        results=formatted_results,
                        pre_computed_analysis=analysis
                    )
                    summary_method = "llm_intelligent_parallel"
                except Exception as e:
                    logger.warning(f"Intelligent summary failed, falling back to pattern-based: {e}")
                    summary = self.summarizer.generate_summary(
                        query=request.prompt,
                        results=formatted_results,
                        sql_query=sql_query
                    )
                    summary_method = "pattern_based_fallback"
            else:
                # Use fast pattern-based summarizer
                summary = self.summarizer.generate_summary(
                    query=request.prompt,
                    results=formatted_results,
                    sql_query=sql_query
                )
                summary_method = "pattern_based"

            return AgentResponse(
                request_id=request.request_id,
                agent_name=self.name,
                status=AgentStatus.COMPLETED,
                result={
                    "query": request.prompt,
                    "sql": sql_query,
                    "data": formatted_results,
                    "summary": summary,
                    "record_count": len(results) if results else 0,
                    "execution_time_seconds": round(execution_time, 3)
                },
                metadata={
                    "agent_role": "profile_analysis",
                    "query_type": self._classify_query_type(request.prompt),
                    "used_metadata": True,
                    "summary_method": summary_method
                }
            )

        except Exception as e:
            logger.error(f"Profile analysis failed: {str(e)}", exc_info=True)
            return self._create_error_response(request.request_id, str(e))

    async def _execute_sql_query(self, sql_query: str) -> Tuple[List, float]:
        """Execute SQL query and return results with execution time"""
        logger.info(f"Executing SQL: {sql_query[:200]}...")

        import time
        start_time = time.time()

        try:
            results = self.ch_client.execute(sql_query)
            execution_time = time.time() - start_time
            logger.info(f"Query executed in {execution_time:.2f}s, returned {len(results) if results else 0} rows")
            return results, execution_time

        except Exception as e:
            logger.error(f"SQL execution error: {e}")
            raise

    async def _generate_sql(self, user_query: str) -> str:
        """Generate ClickHouse SQL from natural language query with full context"""

        system_prompt = f"""
        You are a ClickHouse SQL expert. Generate efficient SQL queries for profile analysis.

        {self.enhanced_schema_context}

        Guidelines:
        1. Use EXACT values from the "Available Values" section when filtering arrays
        2. Always use proper LIMIT for detail queries (default 100 unless specified)
        3. Handle NULL values with isNotNull or coalesce()
        4. Use arrayStringConcat(array, ', ') to display arrays nicely
        5. For risk categorization, use the score thresholds provided
        6. Common patterns:
           - Top N by score: ORDER BY score DESC LIMIT N
           - Statistics: COUNT(*), AVG(score), etc. with GROUP BY
           - Array search: has(applications_used, 'WhatsApp')
           - Boolean counts: COUNTIf(has_crime_case) or SUM(CASE WHEN has_crime_case THEN 1 ELSE 0 END)
        7. Use CTEs for complex queries to improve readability
        8. For comparisons, use meaningful aliases that include group identifiers (e.g., _syr, _irn)

        Return ONLY the SQL query, no explanations or markdown.
        """

        user_prompt = f"""
        Generate ClickHouse SQL for this request: "{user_query}"

        Remember:
        - Use exact application names from the list (case-sensitive)
        - Use exact crime categories/subcategories from the lists
        - Country codes are 3-letter ISO codes (UAE, IND, PAK, etc.)
        """

        response = await self.llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )

        # Clean up the response
        sql = response.strip()

        # Remove markdown code blocks if present
        sql = re.sub(r'```sql\s*', '', sql)
        sql = re.sub(r'```\s*$', '', sql)

        # SQL injection prevention
        dangerous_keywords = ['drop', 'truncate', 'delete', 'insert', 'update', 'alter', 'create', 'grant', 'revoke']
        sql_lower = sql.lower()

        for keyword in dangerous_keywords:
            if re.search(r'\b' + keyword + r'\b', sql_lower):
                logger.warning(f"Dangerous SQL keyword detected: {keyword}")
                return None

        # Ensure it's a SELECT query
        if not re.match(r'^\s*(WITH|SELECT)', sql, re.IGNORECASE):
            logger.warning(f"Not a SELECT query: {sql[:100]}")
            return None

        # Remove trailing semicolon if present
        sql = sql.rstrip(';')

        # Check for multiple statements
        if ';' in sql:
            logger.warning("Multiple SQL statements detected")
            return None

        return sql

    def _format_results(self, results: list, sql_query: str) -> list:
        """Format query results for presentation"""

        if not results:
            return []

        # Try to extract column names from SQL
        column_names = self._extract_column_names(sql_query)

        # Limit results for memory efficiency
        max_results = 1000  # Increased from 100 for better analysis
        if len(results) > max_results:
            logger.info(f"Truncating results from {len(results)} to {max_results}")
            results = results[:max_results]

        # Convert to list of dicts
        formatted = []
        for row in results:
            if isinstance(row, dict):
                formatted.append(row)
            elif isinstance(row, (tuple, list)):
                # Create dict with extracted or generated column names
                row_dict = {}
                for i, value in enumerate(row):
                    if i < len(column_names):
                        col_name = column_names[i]
                    else:
                        col_name = f"col_{i}"

                    # Format arrays nicely
                    if isinstance(value, list):
                        value = ", ".join(str(v) for v in value)

                    row_dict[col_name] = value

                formatted.append(row_dict)

        return formatted

    def _extract_column_names(self, sql: str) -> list:
        """Extract column names from SELECT statement"""

        # Clean SQL
        sql_clean = ' '.join(sql.split())

        # Handle WITH clause
        if sql_clean.upper().startswith('WITH'):
            # Find the main SELECT after CTEs
            main_select_match = re.search(r'\)\s+SELECT\s+(.*?)\s+FROM', sql_clean, re.IGNORECASE)
        else:
            # Direct SELECT
            main_select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql_clean, re.IGNORECASE)

        if not main_select_match:
            logger.warning("Could not extract columns from SQL")
            return []

        columns_str = main_select_match.group(1)
        columns = []

        # Simple parser for column names and aliases
        # This is a simplified version - in production, use a proper SQL parser
        parts = columns_str.split(',')

        for part in parts:
            part = part.strip()

            # Extract alias if present
            if ' as ' in part.lower():
                alias = part.split(' as ')[-1].strip().strip('"\'`')
                columns.append(alias)
            else:
                # Try to extract the column name
                # Remove function calls, just get the core identifier
                clean_part = re.sub(r'\([^)]*\)', '', part)
                identifiers = re.findall(r'\b\w+\b', clean_part)
                if identifiers:
                    columns.append(identifiers[-1])

        return columns

    def _classify_query_type(self, query: str) -> str:
        """Classify the type of profile query"""

        query_lower = query.lower()

        if any(word in query_lower for word in ['top', 'highest', 'lowest', 'best', 'worst']):
            return "ranking"
        elif any(word in query_lower for word in ['statistics', 'count', 'average', 'distribution', 'breakdown']):
            return "statistics"
        elif any(word in query_lower for word in ['specific', 'individual', 'person', 'imsi', 'phone']):
            return "individual_lookup"
        elif any(word in query_lower for word in ['risk', 'score', 'criminal', 'crime']):
            return "risk_analysis"
        elif any(word in query_lower for word in ['demographic', 'nationality', 'age', 'gender']):
            return "demographic_analysis"
        else:
            return "general_query"

    def _create_error_response(self, request_id: str, error_msg: str) -> AgentResponse:
        """Create error response"""

        return AgentResponse(
            request_id=request_id,
            agent_name=self.name,
            status=AgentStatus.FAILED,
            result={
                "error": error_msg,
                "query": "Failed to process",
                "suggestion": "Please try rephrasing your query or check the error message."
            }
        )
