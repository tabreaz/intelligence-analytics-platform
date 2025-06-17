# src/agents/time_parser/agent.py
"""
Time Parser Agent - Extracts and parses time expressions from queries
"""
import re
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

from src.agents.base_agent import BaseAgent, AgentRequest, AgentResponse, AgentStatus
from src.core.activity_logger import ActivityLogger
from src.core.config_manager import ConfigManager
from src.core.llm.base_llm import LLMClientFactory
from src.core.logger import get_logger
from src.core.session_manager_models import QueryContext
from src.core.training_logger import TrainingLogger
from .constants import (
    TIME_INDICATORS, DEFAULT_TIME_RANGE_DAYS,
    DATE_PATTERNS, TIME_PATTERNS
)
from .date_expander import DateExpander
from .exceptions import TimeParserError
from .models import (
    DateRange, HourConstraint, TimeParsingResult, TimeContext,
    DayConstraint, ExpandedDates, EventMapping, ConstraintType
)
from .response_parser import TimeParserResponseParser

logger = get_logger(__name__)


class TimeParserAgent(BaseAgent):
    """
    Agent for parsing time expressions from natural language queries
    
    Features:
    - Extracts absolute and relative time expressions
    - Handles hour-of-day constraints (morning, evening, etc.)
    - Supports day-of-week constraints (weekend, weekday, etc.)
    - Provides structured time data for SQL generation
    - Context-aware parsing using previous time ranges
    """

    def __init__(self, name: str, config: dict, config_manager: ConfigManager, session_manager=None):
        super().__init__(name, config)
        self.config_manager = config_manager
        self.session_manager = session_manager

        # Initialize LLM client
        llm_model = config.get('llm_model', 'openai')
        llm_config = config_manager.get_llm_config(llm_model)
        self.llm_client = LLMClientFactory.create_client(llm_config)

        # Configuration
        self.default_time_range = config.get('default_time_range_days', DEFAULT_TIME_RANGE_DAYS)
        self.timezone = config.get('timezone', 'UTC')
        self.enable_context = config.get('enable_context', True)
        self.enable_training_data = config.get('enable_training_data', True)

        # Initialize training logger
        self.training_logger = TrainingLogger(session_manager) if self.enable_training_data else None

        # Initialize activity logger
        self.activity_logger = ActivityLogger(agent_name=self.name)

        logger.info(f"TimeParserAgent initialized with timezone: {self.timezone}")

    async def validate_request(self, request: AgentRequest) -> bool:
        """
        Quick validation to check if query contains time expressions
        Uses heuristics to avoid unnecessary LLM calls
        """
        if not request.prompt:
            return False

        query_lower = request.prompt.lower()

        # Check for time indicators
        has_time_indicator = any(indicator in query_lower for indicator in TIME_INDICATORS)

        # Check for date patterns
        has_date_pattern = any(
            re.search(pattern, request.prompt)
            for pattern in DATE_PATTERNS
        )

        # Check for time patterns
        has_time_pattern = any(
            re.search(pattern, request.prompt, re.IGNORECASE)
            for pattern in TIME_PATTERNS
        )

        should_parse = has_time_indicator or has_date_pattern or has_time_pattern

        logger.debug(f"Time validation for '{request.prompt[:50]}...': {should_parse}")
        return should_parse

    async def process(self, request: AgentRequest) -> AgentResponse:
        """
        Process request to extract and parse time expressions
        """
        start_time = datetime.now()
        llm_start_time = None
        llm_response = None

        # Set query context for activity logging
        query_context = request.context.get('query_context')
        if query_context and isinstance(query_context, QueryContext):
            self.activity_logger.set_query_context(query_context)

        try:
            # Log analyzing
            self.activity_logger.action("Analyzing query for time expressions...")

            # Create time context
            context = self._create_time_context(request.context)

            # Extract and parse time expressions
            llm_start_time = datetime.now()
            time_data, llm_response, prompts = await self._parse_time_expressions_with_tracking(
                request.prompt,
                context
            )

            # Log key findings
            if time_data.get('date_ranges') or time_data.get('hour_constraints') or time_data.get('day_constraints'):
                summary_parts = []
                if time_data.get('date_ranges'):
                    summary_parts.append(f"{len(time_data['date_ranges'])} date range(s)")
                if time_data.get('hour_constraints'):
                    summary_parts.append(f"{len(time_data['hour_constraints'])} hour constraint(s)")
                if time_data.get('day_constraints'):
                    summary_parts.append(f"{len(time_data['day_constraints'])} day constraint(s)")

                self.activity_logger.identified(
                    "time constraints",
                    {"summary": ", ".join(summary_parts)}
                )
            elif time_data.get('default_range'):
                self.activity_logger.decision(
                    f"No explicit time found, applying semantic default: {time_data['default_range'].get('reason', 'default range')}"
                )

            # Build result
            result = self._build_result(time_data, request.prompt)

            execution_time = (datetime.now() - start_time).total_seconds()

            # Log training data if enabled (fire-and-forget)
            if self.training_logger and llm_response:
                # Get model config
                model_config = {
                    'model': getattr(self.llm_client.config, 'model', 'unknown') if hasattr(self.llm_client,
                                                                                            'config') else 'unknown',
                    'temperature': getattr(self.llm_client.config, 'temperature', 0.1) if hasattr(self.llm_client,
                                                                                                  'config') else 0.1,
                    'max_tokens': getattr(self.llm_client.config, 'max_tokens', 1000) if hasattr(self.llm_client,
                                                                                                 'config') else 1000
                }

                # Prepare extracted params for time parsing
                extracted_params = {
                    "date_ranges": [dr.__dict__ if hasattr(dr, '__dict__') else dr for dr in result.date_ranges],
                    "hour_constraints": [hc.__dict__ if hasattr(hc, '__dict__') else hc for hc in
                                         result.hour_constraints],
                    "day_constraints": [dc.__dict__ if hasattr(dc, '__dict__') else dc for dc in
                                        result.day_constraints],
                    "has_time_expressions": result.has_time_expressions,
                    "default_range": result.default_range
                }

                # Log in background
                self.training_logger.log_llm_interaction_background(
                    session_id=request.context.get('session_id'),
                    query_id=request.context.get('query_id', request.request_id),
                    query_text=request.prompt,
                    event_type="time_parsing",
                    llm_response=llm_response,
                    llm_start_time=llm_start_time,
                    prompts=prompts,
                    result=asdict(result) if hasattr(result, '__dict__') else result,
                    model_config=model_config,
                    extracted_params=extracted_params,
                    category="time_parsing",
                    confidence=result.parsing_confidence,
                    success=result.has_time_expressions
                )

            return AgentResponse(
                request_id=request.request_id,
                agent_name=self.name,
                status=AgentStatus.COMPLETED,
                result=asdict(result) if hasattr(result, '__dict__') else result,
                execution_time=execution_time
            )

        except Exception as e:
            logger.error(f"Time parsing failed: {str(e)}", exc_info=True)
            self.activity_logger.error(f"Failed to parse time expressions", error=e)

            return AgentResponse(
                request_id=request.request_id,
                agent_name=self.name,
                status=AgentStatus.FAILED,
                result={"error": str(e)},
                error=str(e),
                execution_time=(datetime.now() - start_time).total_seconds()
            )

    def _create_time_context(self, request_context: Dict[str, Any]) -> TimeContext:
        """Create time context from request context"""
        return TimeContext(
            current_datetime=datetime.now(),
            timezone=self.timezone,
            previous_time_ranges=request_context.get('previous_time_ranges'),
            query_context=request_context
        )

    async def _parse_time_expressions(
            self,
            prompt: str,
            context: TimeContext
    ) -> Dict[str, Any]:
        """Parse time expressions using LLM"""

        # Build system prompt with context
        system_prompt = self._build_system_prompt(context)
        user_prompt = f"""Extract time expressions from: {prompt}

IMPORTANT: Even when NO explicit time is mentioned, analyze if the query has intrinsic time requirements:
- Does the query imply time-bounded calculations? (e.g., "commuting" implies regular daily patterns)
- Does it require historical data to be meaningful? (e.g., "frequent visitors" needs time to measure frequency)
- Does the activity type suggest specific time windows? (e.g., "nightlife" implies evening/night hours)

Apply semantic time constraints when the query's meaning inherently requires them."""

        try:
            # Call LLM
            response = await self.llm_client.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )

            # Parse response
            parsed = TimeParserResponseParser.parse(response)

            # Validate and enhance parsed data
            validated = self._validate_time_data(parsed, context)

            return validated

        except Exception as e:
            logger.error(f"LLM time parsing failed: {e}")
            raise TimeParserError(f"Failed to parse time expressions: {str(e)}")

    async def _parse_time_expressions_with_tracking(
            self,
            prompt: str,
            context: TimeContext
    ) -> Tuple[Dict[str, Any], str, Dict[str, str]]:
        """Parse time expressions using LLM with response tracking"""

        # Build system prompt with context
        system_prompt = self._build_system_prompt(context)
        user_prompt = f"Extract time expressions from: {prompt}"

        try:
            # Call LLM
            response = await self.llm_client.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )

            # Parse response
            parsed = TimeParserResponseParser.parse(response)

            # Validate and enhance parsed data
            validated = self._validate_time_data(parsed, context)

            # Return validated data, response, and prompts
            prompts = {"system": system_prompt, "user": user_prompt}
            return validated, response, prompts

        except Exception as e:
            logger.error(f"LLM time parsing failed: {e}")
            raise TimeParserError(f"Failed to parse time expressions: {str(e)}")

    def _build_system_prompt(self, context: TimeContext) -> str:
        """Build context-aware system prompt"""
        current = context.current_datetime
        yesterday = current - timedelta(days=1)
        last_week_start = current - timedelta(days=7)

        prompt = f"""You are an advanced time expression parser for database queries.

Current datetime: {current.strftime('%Y-%m-%d %H:%M:%S')} {context.timezone}
Today is: {current.strftime('%A, %B %d, %Y')}

Time mappings:
- "yesterday" → {yesterday.strftime('%Y-%m-%d')}
- "today" → {current.strftime('%Y-%m-%d')}
- "last week" → {last_week_start.strftime('%Y-%m-%d')} to {yesterday.strftime('%Y-%m-%d')}
- "morning" → hours 6-12
- "afternoon" → hours 12-18
- "evening" → hours 18-22
- "night" → hours 22-6
- "weekend" → Saturday, Sunday
- "weekday" → Monday through Friday

Activity context (use when relevant to query):
- **Work/Commute**: 07-17 (office hours), 06-09 & 16-19 (commute times), weekdays
- **Social/Entertainment**: 18-22 (evening), 23-05 (late night)
- **Pattern Analysis**: Requires minimum 2 weeks, ideally 1 month of data

IMPORTANT: Support EXCLUSIONS and SPECIFIC SELECTIONS:
- "except 10th hour" → exclude hour 10
- "excluding weekends" → exclude Saturday and Sunday
- "only Fridays" → include only Friday
- "Jun 1st till 5th" → expand to individual dates

When NO explicit time is mentioned, check if query needs semantic time:
- Pattern/Behavior/Trend queries → Use 30 days default (instead of 2 days)
- Commuting/Work queries → Add work hours (07-17) and weekdays AND 30 days range
- Social/Entertainment queries → Add evening/night hours (18-05)

ALWAYS set default_range when no date_ranges are specified, especially for pattern analysis!

When NO time is specified but query implies pattern analysis, set appropriate default_range:
{{
    "default_range": {{
        "type": "semantic_requirement",
        "days_back": 30,
        "reason": "Commuting pattern analysis requires at least 1 month of data",
        "minimum_days": 14
    }}
}}

Return ONLY valid JSON:
{{
    "date_ranges": [
        {{
            "type": "absolute",
            "start": "2024-01-15T00:00:00",
            "end": "2024-01-15T23:59:59",
            "original_text": "yesterday",
            "confidence": 0.95,
            "constraint_type": "include",
            "expand_to_dates": false
        }}
    ],
    "excluded_date_ranges": [],
    "hour_constraints": [
        {{
            "start_hour": 9,
            "end_hour": 17,
            "original_text": "9 to 5",
            "constraint_type": "include",
            "excluded_hours": null,
            "days_applicable": null
        }}
    ],
    "day_constraints": [
        {{
            "days": ["friday"],
            "constraint_type": "include",
            "original_text": "only Fridays"
        }}
    ],
    "composite_constraints": {{
        "day_hour_combinations": [
            {{
                "days": ["friday"],
                "start_hour": 12,
                "end_hour": 14
            }}
        ]
    }},
    "event_mappings": [],
    "raw_expressions": ["yesterday", "9 to 5", "only Fridays"],
    "default_range": null
}}

For complex queries like "only Fridays between 12-14 hrs for last 3 months":
1. Create date_ranges for "last 3 months"
2. Add day_constraints for "only Fridays" with constraint_type="include"
3. Add hour_constraints for "12-14 hrs"
4. Set composite_constraints.day_hour_combinations

If NO time expressions found, check if semantic time is needed based on query intent.

Example for pattern query without explicit time:
Query: "Show all IRNs who are commuting for work between DXB and AUH"
{{
    "date_ranges": [],
    "excluded_date_ranges": [],
    "hour_constraints": [],
    "day_constraints": [],
    "composite_constraints": null,
    "event_mappings": [],
    "raw_expressions": [],
    "default_range": {{
        "type": "semantic_requirement",
        "days_back": 30,
        "reason": "Commuting pattern analysis requires at least 1 month of data to identify regular travelers",
        "minimum_days": 14,
        "pattern_type": "commuting"
    }}
}}
"""

        # Add context if enabled
        if self.enable_context and context.previous_time_ranges:
            prompt += f"\n\nPrevious time context: {context.previous_time_ranges}"

        return prompt

    def _validate_time_data(
            self,
            parsed: Dict[str, Any],
            context: TimeContext
    ) -> Dict[str, Any]:
        """Validate and enhance parsed time data"""

        # Ensure all required fields exist
        validated = {
            'date_ranges': [],
            'hour_constraints': [],
            'day_constraints': [],
            'raw_expressions': [],
            'default_range': None
        }

        # Validate date ranges
        for date_range in parsed.get('date_ranges', []):
            try:
                # Parse dates to ensure they're valid
                start = datetime.fromisoformat(date_range['start'].replace('Z', '+00:00'))
                end = datetime.fromisoformat(date_range['end'].replace('Z', '+00:00'))

                if end < start:
                    logger.warning(f"Invalid date range: end before start")
                    continue

                validated['date_ranges'].append(date_range)

            except Exception as e:
                logger.warning(f"Invalid date range format: {e}")
                continue

        # Validate hour constraints
        for hour in parsed.get('hour_constraints', []):
            if 0 <= hour.get('start_hour', 0) <= 23 and 0 <= hour.get('end_hour', 0) <= 23:
                validated['hour_constraints'].append(hour)

        # Copy other fields
        validated['day_constraints'] = parsed.get('day_constraints', [])
        validated['raw_expressions'] = parsed.get('raw_expressions', [])

        # Add default range if no specific time found
        if not validated['date_ranges'] and not parsed.get('default_range'):
            validated['default_range'] = {
                'type': 'relative',
                'days_back': self.default_time_range,
                'reason': 'No specific time mentioned, using default range'
            }
        else:
            validated['default_range'] = parsed.get('default_range')

        return validated

    def _build_result(
            self,
            time_data: Dict[str, Any],
            original_prompt: str
    ) -> TimeParsingResult:
        """Build structured result with enhanced features"""

        # Convert to model objects
        date_ranges = [
            DateRange(**dr) if isinstance(dr, dict) else dr
            for dr in time_data.get('date_ranges', [])
        ]

        excluded_date_ranges = [
            DateRange(**dr) if isinstance(dr, dict) else dr
            for dr in time_data.get('excluded_date_ranges', [])
        ]

        hour_constraints = [
            HourConstraint(**hc) if isinstance(hc, dict) else hc
            for hc in time_data.get('hour_constraints', [])
        ]

        # Convert day constraints to new format
        day_constraints = []
        for dc in time_data.get('day_constraints', []):
            if isinstance(dc, dict):
                day_constraints.append(DayConstraint(**dc))
            elif isinstance(dc, str):
                # Legacy format - convert string to DayConstraint
                day_constraints.append(DayConstraint(
                    days=[dc],
                    constraint_type=ConstraintType.INCLUDE,
                    original_text=dc
                ))

        # Handle event mappings
        event_mappings = [
            EventMapping(**em) if isinstance(em, dict) else em
            for em in time_data.get('event_mappings', [])
        ]

        # Get composite constraints
        composite_constraints = time_data.get('composite_constraints')

        # Expand dates if requested
        expanded_dates = None
        if any(dr.expand_to_dates for dr in date_ranges):
            expanded_dates = self._expand_dates(
                date_ranges, day_constraints, hour_constraints
            )

        # Calculate confidence
        has_expressions = bool(
            date_ranges or excluded_date_ranges or hour_constraints or
            day_constraints or event_mappings or composite_constraints
        )
        confidence = 0.95 if has_expressions else 0.5

        # Generate SQL hints
        sql_hints = DateExpander.generate_sql_hints(expanded_dates, composite_constraints)

        # Generate summary
        summary = self._generate_enhanced_summary(
            date_ranges, excluded_date_ranges, hour_constraints,
            day_constraints, composite_constraints
        )

        return TimeParsingResult(
            has_time_expressions=has_expressions,
            date_ranges=date_ranges,
            excluded_date_ranges=excluded_date_ranges,
            expanded_dates=expanded_dates,
            hour_constraints=hour_constraints,
            day_constraints=day_constraints,
            event_mappings=event_mappings,
            composite_constraints=composite_constraints,
            default_range=time_data.get('default_range'),
            summary=summary,
            raw_expressions=time_data.get('raw_expressions', []),
            parsing_confidence=confidence,
            sql_hints=sql_hints
        )

    def _expand_dates(
            self,
            date_ranges: List[DateRange],
            day_constraints: List[DayConstraint],
            hour_constraints: List[HourConstraint]
    ) -> List[ExpandedDates]:
        """Expand date ranges to individual dates"""
        expanded_list = []

        for dr in date_ranges:
            if dr.expand_to_dates:
                expanded = DateExpander.expand_date_range(
                    dr,
                    granularity=self._determine_granularity(hour_constraints),
                    day_constraints=day_constraints,
                    hour_constraints=hour_constraints
                )
                expanded_list.append(expanded)

        return expanded_list

    def _determine_granularity(self, hour_constraints: List[HourConstraint]) -> str:
        """Determine appropriate granularity based on constraints"""
        if hour_constraints:
            # If we have specific hour constraints, use hour granularity
            return "hour"
        return "day"

    def _generate_enhanced_summary(
            self,
            date_ranges: List[DateRange],
            excluded_date_ranges: List[DateRange],
            hour_constraints: List[HourConstraint],
            day_constraints: List[DayConstraint],
            composite_constraints: Optional[Dict[str, Any]]
    ) -> str:
        """Generate enhanced human-readable summary"""
        parts = []

        if date_ranges:
            parts.append(f"{len(date_ranges)} date range(s)")

        if excluded_date_ranges:
            parts.append(f"excluding {len(excluded_date_ranges)} range(s)")

        if hour_constraints:
            include_hours = [hc for hc in hour_constraints if hc.constraint_type == ConstraintType.INCLUDE]
            exclude_hours = [hc for hc in hour_constraints if hc.constraint_type == ConstraintType.EXCLUDE]

            if include_hours:
                hours = [f"{hc.original_text}" for hc in include_hours[:2]]
                parts.append(f"Time: {', '.join(hours)}")
            if exclude_hours:
                hours = [f"{hc.original_text}" for hc in exclude_hours[:2]]
                parts.append(f"Excluding: {', '.join(hours)}")

        if day_constraints:
            include_days = [dc for dc in day_constraints if dc.constraint_type == ConstraintType.INCLUDE]
            exclude_days = [dc for dc in day_constraints if dc.constraint_type == ConstraintType.EXCLUDE]

            if include_days:
                days = [', '.join(dc.days) for dc in include_days]
                parts.append(f"Days: {', '.join(days)}")
            if exclude_days:
                days = [', '.join(dc.days) for dc in exclude_days]
                parts.append(f"Excluding days: {', '.join(days)}")

        if composite_constraints:
            parts.append("Complex constraints applied")

        return " | ".join(parts) if parts else "No specific time constraints"
