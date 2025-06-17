# src/agents/risk_filter/agent.py
"""
Risk Filter Agent - Extracts risk and security filters from queries
Modernized version using BaseAgent with resource management
"""
import json
from datetime import datetime
from typing import Optional

from .constants import MAX_RETRIES
from .models import RiskFilterResult
from .prompt import RISK_FILTER_PROMPT
from .response_parser import RiskFilterResponseParser
from .validator import RiskFilterValidator
from ..base_agent import BaseAgent, AgentResponse, AgentRequest
from ...core.logger import get_logger

logger = get_logger(__name__)


class RiskFilterAgent(BaseAgent):
    """Agent for extracting risk and security filters from queries"""

    def __init__(self, name: str, config: dict, resource_manager):
        """Initialize with agent-specific components only"""
        super().__init__(name, config, resource_manager)
        
        # Agent-specific components
        self.response_parser = RiskFilterResponseParser()
        self.validator = RiskFilterValidator()
        
        # Agent metadata - now part of base class
        # self.agent_id, self.description, self.capabilities are in base class

    async def validate_request(self, request: AgentRequest) -> bool:
        """Validate if the agent can handle this request"""
        # Risk filter agent can handle any query to check for risk filters
        return True

    async def process_internal(self, request: AgentRequest) -> AgentResponse:
        """
        Core risk filter extraction logic
        All infrastructure concerns handled by BaseAgent
        """
        start_time = datetime.now()
        
        # Log activity
        self.activity_logger.action("Extracting risk and security filters...")
        
        # Build prompts
        system_prompt = RISK_FILTER_PROMPT
        user_prompt = f"Extract risk filters from: {request.prompt}"
        
        # Get LLM response with retry logic
        llm_response = await self._call_llm_with_schema_retry(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            request=request
        )
        
        if not llm_response:
            raise ValueError("Failed to get valid LLM response after retries")
        
        # Parse response
        self.activity_logger.action("Parsing LLM response")
        result = self.response_parser.parse(llm_response)
        
        # Log the reasoning from LLM
        if result.raw_extractions.get('reasoning'):
            self.activity_logger.info(result.raw_extractions['reasoning'])
        
        # Validate against schema
        self.activity_logger.action("Validating against database schema")
        cleaned_data, warnings = self.validator.validate_and_clean(
            result.risk_scores,
            result.flags,
            result.exclude_scores,
            result.exclude_flags
        )
        
        # Update result with validated data
        result.risk_scores = cleaned_data['risk_scores']
        result.flags = cleaned_data['flags']
        result.exclude_scores = cleaned_data['exclude_scores']
        result.exclude_flags = cleaned_data['exclude_flags']
        result.validation_warnings.extend(warnings)
        
        # Log extraction summary
        summary_details = {
            "risk_scores": len(result.risk_scores) + len(result.exclude_scores),
            "flags": len(result.flags) + len(result.exclude_flags),
            "crime_categories": len(result.crime_categories.include) if result.crime_categories else 0,
            "warnings": len(result.validation_warnings)
        }
        self.activity_logger.identified("risk filters", summary_details)
        
        # Create response with metadata
        return self._create_success_response(
            request=request,
            result=result.to_dict(),
            start_time=start_time,
            confidence=result.confidence,
            extraction_method=result.extraction_method,
            validation_warnings=result.validation_warnings,
            schema_info=self.validator.get_schema_info()
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
                event_type="risk_filter_extraction"
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

IMPORTANT: Return ONLY valid JSON. Example format:
{{
  "inclusions": {{
    "risk_scores": {{"risk_score": {{"operator": ">", "value": 0.7}}}},
    "flags": {{"has_crime_case": true}},
    "crime_categories": {{"categories": ["drug_dealing"], "severity": null}}
  }},
  "exclusions": {{
    "risk_scores": {{}},
    "flags": {{}},
    "crime_categories": {{}}
  }}
}}

Valid fields: {self.validator.get_schema_info()}"""

    def _create_extraction_summary(self, result: RiskFilterResult) -> str:
        """Create a summary of extracted risk filters"""
        parts = []
        
        # Count filters
        score_count = len(result.risk_scores) + len(result.exclude_scores)
        flag_count = len(result.flags) + len(result.exclude_flags)
        category_count = len(result.crime_categories.include) if result.crime_categories else 0
        
        if score_count > 0:
            parts.append(f"{score_count} risk scores")
        if flag_count > 0:
            parts.append(f"{flag_count} flags")
        if category_count > 0:
            parts.append(f"{category_count} crime categories")
        
        if not parts:
            return "No risk filters found"
        
        summary = f"Extracted: {', '.join(parts)}"
        if result.validation_warnings:
            summary += f" ({len(result.validation_warnings)} warnings)"
        
        return summary