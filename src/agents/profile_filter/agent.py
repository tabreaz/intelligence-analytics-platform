# src/agents/profile_filter/agent.py
"""
Profile Filter Agent - Extracts demographic, identity, and profile-related filters
"""
import json
from datetime import datetime
from typing import Optional, Dict, Any

from .constants import MAX_RETRIES
from .prompt import PROFILE_FILTER_PROMPT
from .response_parser import ProfileFilterResponseParser
from ..base_agent import BaseAgent, AgentResponse, AgentStatus, AgentRequest
from ...core.activity_logger import ActivityLogger
from ...core.config_manager import ConfigManager
from ...core.llm.base_llm import LLMClientFactory
from ...core.logger import get_logger
from ...core.session_manager_models import QueryContext as QueryContextModel
from ...core.training_logger import TrainingLogger

logger = get_logger(__name__)


class ProfileFilterAgent(BaseAgent):
    """Agent for extracting profile-related filters from queries"""

    def __init__(self, name: str, config: dict, config_manager: ConfigManager, session_manager=None):
        """
        Initialize Profile Filter Agent
        """
        super().__init__(name, config)
        self.config_manager = config_manager
        self.session_manager = session_manager

        # Initialize LLM client
        llm_model = config.get('llm_model', 'openai')
        llm_config = config_manager.get_llm_config(llm_model)
        self.llm_client = LLMClientFactory.create_client(llm_config)

        # Agent properties
        self.agent_id = "profile_filter"
        self.description = "Extracts demographic, identity, and profile-based filters"
        self.capabilities = [
            "identity_extraction",
            "demographic_filtering",
            "nationality_processing",
            "work_sponsor_extraction",
            "lifestyle_filtering"
        ]

        # Initialize components
        self.response_parser = ProfileFilterResponseParser()

        # Enable training data logging
        self.enable_training_data = config.get('enable_training_data', True)

        # Initialize loggers
        self.training_logger = TrainingLogger(session_manager) if self.enable_training_data else None
        self.activity_logger = ActivityLogger(agent_name=self.name)

    async def process(self, request: AgentRequest) -> AgentResponse:
        """
        Process request to extract profile filters
        """
        start_time = datetime.now()

        try:
            # Set query context for activity logger
            query_context = request.context.get('query_context') if hasattr(request, 'context') and isinstance(
                request.context, dict) else None
            if query_context and isinstance(query_context, QueryContextModel):
                self.activity_logger.set_query_context(query_context)

            # Log activity
            self.activity_logger.action("Extracting profile filters from query")

            # Build prompts
            system_prompt = PROFILE_FILTER_PROMPT
            user_prompt = f"Extract profile filters from: {request.prompt}"

            # Get LLM response with retry
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

            # Log filter tree structure
            if result.filter_tree:
                self.activity_logger.info(f"Filter tree: {json.dumps(result.filter_tree, indent=2)[:200]}...")
            if result.exclusions:
                self.activity_logger.info(f"Exclusions: {json.dumps(result.exclusions, indent=2)[:200]}...")
            if result.ambiguities:
                self.activity_logger.warning(f"Found {ambiguity_count} ambiguous references")

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
                    "field_count": filter_count + exclusion_count
                }
            )

        except Exception as e:
            logger.error(f"Error in profile filter processing: {e}")
            self.activity_logger.error("Failed to process profile filters", error=e)
            return self._create_error_response(
                request,
                f"Processing failed: {str(e)}",
                start_time
            )

    async def _call_llm_with_retry(self, system_prompt: str, user_prompt: str,
                                   request: AgentRequest, retry_count: int = 0) -> Optional[str]:
        """
        Call LLM with retry logic
        """
        try:
            # Record start time
            llm_start_time = datetime.now()

            # Call LLM
            response = await self.llm_client.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )

            # Try to parse to validate JSON
            parsed_result = json.loads(response)

            # Prepare data for training logger
            prompts = {
                "system": system_prompt,
                "user": user_prompt
            }

            model_config = {
                'model': getattr(self.llm_client.config, 'model', 'unknown') if hasattr(self.llm_client,
                                                                                        'config') else 'unknown',
                'temperature': getattr(self.llm_client.config, 'temperature', 0.1) if hasattr(self.llm_client,
                                                                                              'config') else 0.1,
                'max_tokens': getattr(self.llm_client.config, 'max_tokens', 1000) if hasattr(self.llm_client,
                                                                                             'config') else 1000
            }

            # Log to PostgreSQL in background
            if self.training_logger:
                self.training_logger.log_llm_interaction_background(
                    session_id=request.context.get('session_id'),
                    query_id=request.request_id,
                    query_text=request.prompt,
                    event_type="profile_filter_extraction",
                    llm_response=response,
                    llm_start_time=llm_start_time,
                    prompts=prompts,
                    result=parsed_result,
                    model_config=model_config,
                    success=True
                )

            return response

        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON from LLM: {e}")

            if retry_count < MAX_RETRIES:
                # Retry with hint
                self.activity_logger.action(f"Retrying with schema hint (attempt {retry_count + 1})")

                enhanced_prompt = f"""{user_prompt}

IMPORTANT: Return ONLY valid JSON. Example:
{{
    "reasoning": "User wants Indian males under 35",
    "inclusions": {{
        "nationality_code": ["IND"],
        "gender_en": "Male",
        "age": {{"operator": "<", "value": 35}}
    }},
    "exclusions": {{}},
    "ambiguities": [],
    "confidence": 0.9
}}"""

                return await self._call_llm_with_retry(
                    system_prompt,
                    enhanced_prompt,
                    request,
                    retry_count + 1
                )

            return None

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None

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

    def _count_filters(self, filter_tree: Dict[str, Any]) -> int:
        """Count the number of filter conditions in a filter tree"""
        if not filter_tree:
            return 0
        
        count = 0
        for key, value in filter_tree.items():
            if key in ["AND", "OR", "NOT"]:
                if isinstance(value, list):
                    for condition in value:
                        if "field" in condition:
                            count += 1
                        else:
                            count += self._count_filters(condition)
                else:
                    count += self._count_filters(value)
            elif key == "field":
                count += 1
        
        return count
    
    async def validate_request(self, request: AgentRequest) -> bool:
        """Validate if the agent can handle this request"""
        # Profile filter agent can handle most queries that mention people or demographics
        return True
