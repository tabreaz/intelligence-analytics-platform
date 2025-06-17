# src/agents/risk_filter/agent.py
"""
Risk Filter Agent - Extracts risk and security filters from queries
"""
import json
from datetime import datetime
from typing import Optional

from .constants import MAX_RETRIES
from .models import RiskFilterResult
from .prompt import RISK_FILTER_PROMPT
from .response_parser import RiskFilterResponseParser
from .validator import RiskFilterValidator
from ..base_agent import BaseAgent, AgentResponse, AgentStatus, AgentRequest
from ...core.activity_logger import ActivityLogger
from ...core.config_manager import ConfigManager
from ...core.llm.base_llm import LLMClientFactory
from ...core.logger import get_logger
from ...core.session_manager_models import QueryContext as QueryContextModel
from ...core.training_logger import TrainingLogger

logger = get_logger(__name__)


class RiskFilterAgent(BaseAgent):
    """Agent for extracting risk and security filters from queries"""

    def __init__(self, name: str, config: dict, config_manager: ConfigManager, session_manager=None):
        super().__init__(name, config)
        self.config_manager = config_manager
        self.session_manager = session_manager

        # Initialize LLM client
        llm_model = config.get('llm_model', 'openai')
        llm_config = config_manager.get_llm_config(llm_model)
        self.llm_client = LLMClientFactory.create_client(llm_config)

        # Agent properties
        self.agent_id = "risk_filter"
        self.description = "Extracts risk scores, security flags, and crime categories from queries"
        self.capabilities = ["risk_extraction", "security_filtering", "crime_categorization"]

        # Initialize components
        self.response_parser = RiskFilterResponseParser()
        self.validator = RiskFilterValidator()

        # Enable training data logging
        self.enable_training_data = config.get('enable_training_data', True)

        # Initialize training logger
        self.training_logger = TrainingLogger(session_manager) if self.enable_training_data else None

        # Initialize activity logger
        self.activity_logger = ActivityLogger(agent_name=self.name)

    async def process(self, request: AgentRequest) -> AgentResponse:
        """
        Process request to extract risk and security filters
        """
        start_time = datetime.now()

        try:
            # Set query context for activity logger if available
            query_context = request.context.get('query_context') if hasattr(request, 'context') and isinstance(
                request.context, dict) else None
            if query_context and isinstance(query_context, QueryContextModel):
                self.activity_logger.set_query_context(query_context)

            # Log activity
            self.activity_logger.action("Extracting risk and security filters...")

            # Build prompts
            system_prompt = RISK_FILTER_PROMPT
            user_prompt = f"Extract risk filters from: {request.prompt}"

            # Get LLM response
            llm_response = await self._call_llm_with_retry(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                request=request
            )

            if not llm_response:
                return self._create_error_response(
                    request,
                    "Failed to get LLM response",
                    start_time
                )

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
            summary = self._create_extraction_summary(result)
            summary_details = {
                "risk_scores": len(result.risk_scores) + len(result.exclude_scores),
                "flags": len(result.flags) + len(result.exclude_flags),
                "crime_categories": len(result.crime_categories.include) if result.crime_categories else 0,
                "warnings": len(result.validation_warnings)
            }
            self.activity_logger.identified("risk filters", summary_details)

            # Create response
            return AgentResponse(
                request_id=request.request_id,
                agent_name=self.name,
                status=AgentStatus.COMPLETED,
                result=result.to_dict(),
                execution_time=(datetime.now() - start_time).total_seconds(),
                metadata={
                    "confidence": result.confidence,
                    "extraction_method": result.extraction_method,
                    "validation_warnings": result.validation_warnings,
                    "schema_info": self.validator.get_schema_info()
                }
            )

        except Exception as e:
            logger.error(f"Error in risk filter extraction: {e}")
            self.activity_logger.error("Failed to extract risk filters", error=e)
            return self._create_error_response(
                request,
                f"Risk filter extraction failed: {str(e)}",
                start_time
            )

    async def _call_llm_with_retry(self, system_prompt: str, user_prompt: str,
                                   request: AgentRequest, retry_count: int = 0) -> Optional[str]:
        """Call LLM with retry logic and schema validation
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            request: Agent request
            retry_count: Current retry attempt
            
        Returns:
            LLM response string or None
        """
        try:
            # Record start time for timing
            llm_start_time = datetime.now()

            # Call LLM
            response = await self.llm_client.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )

            # Try to parse response to validate it's JSON
            parsed_result = json.loads(response)

            # Prepare prompts dict
            prompts = {
                "system": system_prompt,
                "user": user_prompt
            }

            # Prepare model config
            model_config = {
                'model': getattr(self.llm_client.config, 'model', 'unknown') if hasattr(self.llm_client,
                                                                                        'config') else 'unknown',
                'temperature': getattr(self.llm_client.config, 'temperature', 0.1) if hasattr(self.llm_client,
                                                                                              'config') else 0.1,
                'max_tokens': getattr(self.llm_client.config, 'max_tokens', 1000) if hasattr(self.llm_client,
                                                                                             'config') else 1000
            }

            # Log successful response in background
            if self.training_logger:
                self.training_logger.log_llm_interaction_background(
                    session_id=request.context.get('session_id'),
                    query_id=request.request_id,
                    query_text=request.prompt,
                    event_type="risk_filter_extraction",
                    llm_response=response,
                    llm_start_time=llm_start_time,
                    prompts=prompts,
                    result=parsed_result,
                    model_config=model_config,
                    success=True
                )

            return response

        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON response from LLM: {e}")

            if retry_count < MAX_RETRIES:
                # Retry with schema hint
                self.activity_logger.action(f"Retrying with schema hint (attempt {retry_count + 1})")

                enhanced_prompt = f"""{user_prompt}

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

                return await self._call_llm_with_retry(
                    system_prompt,
                    enhanced_prompt,
                    request,
                    retry_count + 1
                )

            # No need to log failed JSON parsing - we'll retry with hints
            return None

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None

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

    def _log_activity_to_queue(self, request: AgentRequest, message: str, is_error: bool = False) -> None:
        """Log activity to the request's activity queue if available"""
        # This is kept for compatibility but we use ActivityLogger directly now
        if is_error:
            self.activity_logger.error(message)
        else:
            self.activity_logger.info(message)

    def _create_error_response(self, request: AgentRequest, error_message: str,
                               start_time: datetime) -> AgentResponse:
        """Create an error response"""
        return AgentResponse(
            request_id=request.request_id,
            agent_name=self.name,
            status=AgentStatus.FAILED,
            result={},
            error=error_message,
            execution_time=(datetime.now() - start_time).total_seconds()
        )

    async def validate_request(self, request: AgentRequest) -> bool:
        """Validate if the agent can handle this request"""
        # Risk filter agent can handle any query to check for risk filters
        return True
