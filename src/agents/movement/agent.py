# src/agents/movement/agent.py
"""
Movement Analysis Agent - Extracts spatial-temporal patterns and movement filters
Modernized version using BaseAgent with resource management
"""
import json
from datetime import datetime
from typing import Optional

from .constants import MAX_RETRIES
from .prompt import MOVEMENT_PROMPT
from .response_parser import MovementResponseParser
from .utils import parse_json_response
from ..base_agent import BaseAgent, AgentResponse, AgentRequest
from ...core.logger import get_logger

logger = get_logger(__name__)


class MovementAgent(BaseAgent):
    """Agent for extracting movement and location-based filters from queries"""

    def __init__(self, name: str, config: dict, resource_manager):
        """Initialize with agent-specific components only"""
        super().__init__(name, config, resource_manager)
        
        # Agent-specific components
        self.response_parser = MovementResponseParser()
        
        # Store last LLM response if needed for debugging
        self._last_llm_response = None

    async def validate_request(self, request: AgentRequest) -> bool:
        """Validate if the agent can handle this request"""
        # Movement agent handles queries about locations, movements, presence, etc.
        movement_keywords = [
            'location', 'movement', 'where', 'visit', 'went', 'travel',
            'present', 'at', 'in', 'near', 'around', 'between',
            'commute', 'pattern', 'frequent', 'stay', 'dwell',
            'meet', 'together', 'density', 'crowd', 'heatmap',
            'predict', 'forecast', 'anomaly', 'unusual'
        ]
        
        prompt_lower = request.prompt.lower()
        return any(keyword in prompt_lower for keyword in movement_keywords)

    async def process_internal(self, request: AgentRequest) -> AgentResponse:
        """
        Core movement filter extraction logic
        All infrastructure concerns handled by BaseAgent
        """
        start_time = datetime.now()
        
        # Log activity
        self.activity_logger.action("Extracting movement and location filters from query")
        
        # Build prompts
        system_prompt = MOVEMENT_PROMPT
        user_prompt = f"Extract movement patterns and location filters from: {request.prompt}"
        
        # Get LLM response with retry logic
        llm_response = await self._call_llm_with_schema_retry(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            request=request
        )
        
        if not llm_response:
            raise ValueError("Failed to get valid LLM response after retries")
        
        # Store LLM response if capture is requested
        if request.context.get('capture_llm_response', False):
            self._last_llm_response = llm_response
        
        # Parse response
        self.activity_logger.action("Parsing movement filter response")
        result = self.response_parser.parse(llm_response)
        
        # Log reasoning if available
        if result.reasoning:
            self.activity_logger.info(f"Reasoning: {result.reasoning[:200]}...")
        
        # Log what was identified
        self.activity_logger.identified(
            "movement analysis",
            {
                "query_type": result.query_type,
                "geofences": len(result.geofences),
                "identity_filters": len(result.identity_filters),
                "has_co_presence": result.co_presence is not None,
                "has_patterns": bool(result.sequence_patterns or result.pattern_detection),
                "ambiguities": len(result.ambiguities),
                "confidence": result.confidence
            }
        )
        
        # Log key components
        if result.geofences:
            for gf in result.geofences:
                self.activity_logger.info(
                    f"Geofence '{gf.reference}': {gf.spatial_filter.method} "
                    f"{'with time constraints' if gf.time_constraints else 'no time constraints'}"
                )
        
        if result.co_presence:
            self.activity_logger.info(
                f"Co-presence analysis: {len(result.co_presence.get('target_ids', []))} targets, "
                f"{result.co_presence.get('match_granularity', 'unknown')} granularity"
            )
        
        if result.ambiguities:
            self.activity_logger.warning(f"Found {len(result.ambiguities)} ambiguous references")
            for amb in result.ambiguities:
                self.activity_logger.warning(f"- {amb.get('parameter')}: {amb.get('issue')}")
        
        # Create response with metadata
        return self._create_success_response(
            request=request,
            result=result.to_dict(),
            start_time=start_time,
            confidence=result.confidence,
            extraction_method=result.extraction_method,
            validation_warnings=result.validation_warnings,
            component_count=len(result.geofences) + len(result.sequence_patterns)
        )

    async def _call_llm_with_schema_retry(
        self, 
        system_prompt: str, 
        user_prompt: str,
        request: AgentRequest,
        retry_count: int = 0
    ) -> Optional[str]:
        """
        Call LLM with retry logic for JSON validation
        This is agent-specific because it needs schema hints
        """
        try:
            # First attempt or retry
            current_user_prompt = user_prompt
            
            # If this is a retry, add schema hint
            if retry_count > 0:
                self.activity_logger.action(f"Retrying with schema hint (attempt {retry_count + 1})")
                current_user_prompt = self._enhance_prompt_with_schema(user_prompt)
            
            # Use base class helper for LLM call with automatic retry and training logging
            response = await self.call_llm(
                system_prompt=system_prompt,
                user_prompt=current_user_prompt,
                request=request,
                event_type="movement_filter_extraction"
            )
            
            if not response:
                return None
            
            # Validate JSON format
            try:
                # Validate that response can be parsed
                parse_json_response(response)
                return response  # Return original for further processing
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON response from LLM: {e}")
                
                # Retry with schema hint if we haven't exceeded retries
                if retry_count < MAX_RETRIES:
                    return await self._call_llm_with_schema_retry(
                        system_prompt,
                        user_prompt,
                        request,
                        retry_count + 1
                    )
                return None
                
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            if retry_count < MAX_RETRIES:
                return await self._call_llm_with_schema_retry(
                    system_prompt,
                    user_prompt,
                    request,
                    retry_count + 1
                )
            return None

    def _enhance_prompt_with_schema(self, user_prompt: str) -> str:
        """Add schema hint to prompt for retry"""
        schema_hint = """
        
IMPORTANT: Return ONLY valid JSON matching this structure:
{
  "reasoning": "string",
  "query_type": "string",
  "identity_filters": {},
  "geofences": [],
  "confidence": 0.95
}

Only include fields that are relevant to the query. Omit empty/null fields.
"""
        return f"{user_prompt}\n{schema_hint}"

    def _count_components(self, result) -> int:
        """Count total components extracted"""
        count = 0
        if result.geofences:
            count += len(result.geofences)
        if result.sequence_patterns:
            count += len(result.sequence_patterns)
        if result.co_presence:
            count += 1
        if result.heatmap:
            count += 1
        if result.clustering:
            count += 1
        if result.pattern_detection:
            count += 1
        if result.anomaly_detection:
            count += 1
        if result.predictive_modeling:
            count += 1
        return count