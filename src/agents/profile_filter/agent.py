# src/agents/profile_filter/agent.py
"""
Profile Filter Agent - Extracts demographic, identity, and profile-related filters
Modernized version using BaseAgent with resource management
"""
import json
from datetime import datetime
from typing import Optional, Dict, Any

from .constants import MAX_RETRIES
from .prompt import PROFILE_FILTER_PROMPT
from .response_parser import ProfileFilterResponseParser
from ..base_agent import BaseAgent, AgentResponse, AgentRequest
from ...core.logger import get_logger

logger = get_logger(__name__)


class ProfileFilterAgent(BaseAgent):
    """Agent for extracting profile-related filters from queries"""

    def __init__(self, name: str, config: dict, resource_manager):
        """Initialize with agent-specific components only"""
        super().__init__(name, config, resource_manager)
        
        # Agent-specific components
        self.response_parser = ProfileFilterResponseParser()
        
        # Agent metadata handled by base class
        # self.agent_id, self.description, self.capabilities are in base class

    async def validate_request(self, request: AgentRequest) -> bool:
        """Validate if the agent can handle this request"""
        # Profile filter agent can handle most queries that mention people or demographics
        return True

    async def process_internal(self, request: AgentRequest) -> AgentResponse:
        """
        Core profile filter extraction logic
        All infrastructure concerns handled by BaseAgent
        """
        start_time = datetime.now()
        
        # Log activity
        self.activity_logger.action("Extracting profile filters from query")
        
        # Build prompts
        system_prompt = PROFILE_FILTER_PROMPT
        user_prompt = f"Extract profile filters from: {request.prompt}"
        
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
        self.activity_logger.action("Parsing profile filter response")
        result = self.response_parser.parse(llm_response)
        
        # Log reasoning if available
        if result.raw_extractions.get('reasoning'):
            self.activity_logger.info(result.raw_extractions['reasoning'])
        
        # Log what was identified
        filter_count = self._count_filters(result.filter_tree)
        exclusion_count = self._count_filters(result.exclusions) if result.exclusions else 0
        ambiguity_count = len(result.ambiguities)
        
        self.activity_logger.identified(
            "profile filters",
            {
                "filters": filter_count,
                "exclusions": exclusion_count,
                "ambiguities": ambiguity_count,
                "confidence": result.confidence
            }
        )
        
        # Log filter tree structure (truncated for readability)
        if result.filter_tree:
            self.activity_logger.info(f"Filter tree: {json.dumps(result.filter_tree, indent=2)[:200]}...")
        if result.exclusions:
            self.activity_logger.info(f"Exclusions: {json.dumps(result.exclusions, indent=2)[:200]}...")
        if result.ambiguities:
            self.activity_logger.warning(f"Found {ambiguity_count} ambiguous references")
        
        # Create response with metadata
        return self._create_success_response(
            request=request,
            result=result.to_dict(),
            start_time=start_time,
            confidence=result.confidence,
            extraction_method=result.extraction_method,
            validation_warnings=result.validation_warnings,
            field_count=filter_count + exclusion_count
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
                event_type="profile_filter_extraction"
            )
            
            if not response:
                return None
            
            # Validate JSON format
            try:
                json.loads(response)
                return response
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
            logger.error(f"Error in LLM call: {e}")
            return None

    def _enhance_prompt_with_schema(self, user_prompt: str) -> str:
        """Add schema hint to prompt for retry attempts"""
        return f"""{user_prompt}

IMPORTANT: Return ONLY valid JSON using the filter_tree format. Example:
{{
    "reasoning": "User wants Indian males under 35 years old",
    "filter_tree": {{
        "AND": [
            {{"field": "nationality_code", "operator": "IN", "value": ["IND"]}},
            {{"field": "gender_en", "operator": "=", "value": "Male"}},
            {{"field": "age", "operator": "<", "value": 35}}
        ]
    }},
    "exclusions": {{
        "AND": []
    }},
    "ambiguities": [],
    "confidence": 0.9
}}

Use logical operators (AND, OR) to combine conditions.
Each condition must have: field, operator, value."""

    def _count_filters(self, filter_tree: Dict[str, Any]) -> int:
        """Count the number of filter conditions in a filter tree"""
        if not filter_tree:
            return 0
        
        count = 0
        
        # Handle different tree structures
        if isinstance(filter_tree, dict):
            for key, value in filter_tree.items():
                if key in ["AND", "OR", "NOT"]:
                    # Logical operator with list of conditions
                    if isinstance(value, list):
                        for condition in value:
                            if isinstance(condition, dict) and "field" in condition:
                                count += 1
                            else:
                                count += self._count_filters(condition)
                    else:
                        count += self._count_filters(value)
                elif key == "field":
                    # Direct condition
                    count += 1
        elif isinstance(filter_tree, list):
            # List of conditions
            for item in filter_tree:
                count += self._count_filters(item)
        
        return count