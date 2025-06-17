# src/agents/time_parser/agent.py
"""
Time Parser Agent - Clean implementation with modern BaseAgent
This serves as a template for updating all other agents
"""
from datetime import datetime
from typing import Dict, Any, Optional

from src.agents.base_agent import BaseAgent, AgentRequest, AgentResponse
from src.core.logger import get_logger
from .constants import DEFAULT_TIME_RANGE_DAYS, TIME_INDICATORS
from .prompt import build_time_parser_prompt
from .response_parser import TimeParserResponseParser

logger = get_logger(__name__)


class TimeParserAgent(BaseAgent):
    """
    Extracts and parses time expressions from natural language queries
    
    All common functionality is handled by BaseAgent:
    - Resource management
    - Activity logging
    - Training data collection
    - Error handling
    - LLM retry logic
    """
    
    def __init__(self, name: str, config: dict, resource_manager):
        """Initialize with agent-specific components only"""
        super().__init__(name, config, resource_manager)
        
        # Agent-specific components
        self.response_parser = TimeParserResponseParser()
        
        # Agent-specific config
        self.default_time_range_days = config.get('default_time_range_days', DEFAULT_TIME_RANGE_DAYS)
        self.timezone = config.get('timezone', 'UTC')
        self.enable_context = config.get('enable_context', True)
        
    async def validate_request(self, request: AgentRequest) -> bool:
        """Check if request has a valid prompt"""
        return bool(request.prompt and request.prompt.strip())
    
    async def process_internal(self, request: AgentRequest) -> AgentResponse:
        """
        Core time parsing logic
        Focus only on business logic - all infrastructure concerns handled by BaseAgent
        """
        query_text = request.prompt
        start_time = datetime.now()
        
        # Quick check for time indicators
        if not self._has_time_indicators(query_text):
            self.activity_logger.info("No time expressions detected")
            return self._create_default_response(request, start_time)
        
        # Get previous context if available
        previous_context = self._get_previous_context(request.context)
        
        # Build prompt with context
        current_datetime = datetime.now()
        system_prompt = build_time_parser_prompt(
            current_datetime=current_datetime,
            timezone=self.timezone,
            previous_context=previous_context
        )
        
        user_prompt = f"""Extract time expressions from: {query_text}

IMPORTANT: Even when NO explicit time is mentioned, analyze if the query has intrinsic time requirements:
- Does the query imply time-bounded calculations? (e.g., "commuting" implies regular daily patterns)
- Does it require historical data to be meaningful? (e.g., "frequent visitors" needs time to measure frequency)
- Does the activity type suggest specific time windows? (e.g., "nightlife" implies evening/night hours)

Apply semantic time constraints when the query's meaning inherently requires them."""
        
        # Call LLM (retry and logging handled by base class)
        llm_response = await self.call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            request=request,
            event_type="time_parsing"
        )
        
        if not llm_response:
            raise ValueError("Failed to get LLM response")
        
        # Parse and validate response
        parsed_data = self.response_parser.parse(llm_response)
        validated_data = self._validate_time_data(parsed_data)
        
        # Build structured result
        result = self._build_result(validated_data, query_text)
        
        # Log key findings
        self._log_findings(result)
        
        # Build response
        return self._create_success_response(
            request=request,
            result=result.to_dict() if hasattr(result, 'to_dict') else result,
            start_time=start_time,
            has_time_expressions=result.get('has_time_expressions', False),
            confidence=result.get('parsing_confidence', 0.5),
            extraction_method=result.get('extraction_method', 'llm')
        )
    
    def _has_time_indicators(self, text: str) -> bool:
        """Check if text contains time expressions"""
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in TIME_INDICATORS)
    
    def _get_previous_context(self, context: Dict[str, Any]) -> Optional[str]:
        """Extract previous time context from request"""
        if not context or 'previous_time_ranges' not in context:
            return None
            
        try:
            # Convert to string representation for prompt
            return str(context['previous_time_ranges'])
        except Exception as e:
            logger.warning(f"Failed to parse previous context: {e}")
            return None
    
    def _validate_time_data(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and enhance parsed time data"""
        # Ensure all required fields exist
        validated = {
            'date_ranges': [],
            'hour_constraints': [],
            'day_constraints': [],
            'raw_expressions': [],
            'default_range': None,
            'excluded_date_ranges': [],
            'event_mappings': [],
            'composite_constraints': None
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
        validated['excluded_date_ranges'] = parsed.get('excluded_date_ranges', [])
        validated['event_mappings'] = parsed.get('event_mappings', [])
        validated['composite_constraints'] = parsed.get('composite_constraints')
        
        # Handle default range
        if not validated['date_ranges'] and not parsed.get('default_range'):
            validated['default_range'] = {
                'type': 'relative',
                'days_back': self.default_time_range_days,
                'reason': 'No specific time mentioned, using default range'
            }
        else:
            validated['default_range'] = parsed.get('default_range')
        
        return validated
    
    def _build_result(self, time_data: Dict[str, Any], original_prompt: str) -> Dict[str, Any]:
        """Build structured result"""
        # Determine if we have time expressions
        has_expressions = bool(
            time_data['date_ranges'] or 
            time_data['hour_constraints'] or 
            time_data['day_constraints'] or
            time_data['excluded_date_ranges'] or
            time_data['event_mappings'] or
            time_data['composite_constraints']
        )
        
        # Generate summary
        summary = self._generate_summary(time_data)
        
        return {
            'has_time_expressions': has_expressions,
            'date_ranges': time_data['date_ranges'],
            'excluded_date_ranges': time_data['excluded_date_ranges'],
            'hour_constraints': time_data['hour_constraints'],
            'day_constraints': time_data['day_constraints'],
            'event_mappings': time_data['event_mappings'],
            'composite_constraints': time_data['composite_constraints'],
            'default_range': time_data['default_range'],
            'raw_expressions': time_data['raw_expressions'],
            'summary': summary,
            'parsing_confidence': 0.95 if has_expressions else 0.5,
            'extraction_method': 'llm'
        }
    
    def _generate_summary(self, time_data: Dict[str, Any]) -> str:
        """Generate human-readable summary"""
        parts = []
        
        if time_data['date_ranges']:
            parts.append(f"{len(time_data['date_ranges'])} date range(s)")
            
        if time_data['excluded_date_ranges']:
            parts.append(f"excluding {len(time_data['excluded_date_ranges'])} range(s)")
            
        if time_data['hour_constraints']:
            parts.append(f"{len(time_data['hour_constraints'])} hour constraint(s)")
            
        if time_data['day_constraints']:
            parts.append(f"{len(time_data['day_constraints'])} day constraint(s)")
            
        if time_data['composite_constraints']:
            parts.append("complex constraints")
            
        return " | ".join(parts) if parts else "No specific time constraints"
    
    def _log_findings(self, result: Dict[str, Any]):
        """Log key findings for activity tracking"""
        if result['date_ranges'] or result['hour_constraints'] or result['day_constraints']:
            summary_parts = []
            if result['date_ranges']:
                summary_parts.append(f"{len(result['date_ranges'])} date range(s)")
            if result['hour_constraints']:
                summary_parts.append(f"{len(result['hour_constraints'])} hour constraint(s)")
            if result['day_constraints']:
                summary_parts.append(f"{len(result['day_constraints'])} day constraint(s)")
            
            self.activity_logger.info(
                f"Identified time constraints: {', '.join(summary_parts)}"
            )
        elif result['default_range']:
            self.activity_logger.decision(
                f"No explicit time found, applying semantic default: {result['default_range'].get('reason', 'default range')}"
            )
    
    def _create_default_response(self, request: AgentRequest, start_time: datetime) -> AgentResponse:
        """Create response with default time range when no expressions found"""
        result = {
            "has_time_expressions": False,
            "date_ranges": [{
                "start": datetime.now().isoformat(),
                "end": datetime.now().isoformat(),
                "type": "default",
                "days": self.default_time_range_days
            }],
            "hour_constraints": [],
            "day_constraints": [],
            "excluded_date_ranges": [],
            "event_mappings": [],
            "composite_constraints": None,
            "default_range": {
                "type": "relative",
                "days_back": self.default_time_range_days,
                "reason": "No time expressions detected"
            },
            "raw_expressions": [],
            "summary": "Using default time range",
            "parsing_confidence": 1.0,
            "extraction_method": "default"
        }
        
        return self._create_success_response(
            request=request,
            result=result,
            start_time=start_time,
            used_default=True,
            reason="no_time_expressions"
        )