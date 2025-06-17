# src/agents/entity_annotator/agent.py
"""
Entity Annotator Agent - Annotates entities in queries for intelligence analysis
"""
import json
from datetime import datetime
from typing import Optional

from .constants import MAX_RETRIES
from .prompt import ENTITY_ANNOTATOR_PROMPT
from .response_parser import EntityAnnotatorResponseParser
from ..base_agent import BaseAgent, AgentResponse, AgentStatus, AgentRequest
from ...core.activity_logger import ActivityLogger
from ...core.annotated_query_logger import AnnotatedQueryLogger
from ...core.config_manager import ConfigManager
from ...core.llm.base_llm import LLMClientFactory
from ...core.logger import get_logger
from ...core.session_manager_models import QueryContext as QueryContextModel
from ...core.training_logger import TrainingLogger

logger = get_logger(__name__)


class EntityAnnotatorAgent(BaseAgent):
    """Agent for annotating entities in queries"""

    def __init__(self, name: str, config: dict, config_manager: ConfigManager, session_manager=None):
        """
        Initialize Entity Annotator Agent
        """
        super().__init__(name, config)
        self.config_manager = config_manager
        self.session_manager = session_manager

        # Initialize LLM client
        llm_model = config.get('llm_model', 'openai')
        llm_config = config_manager.get_llm_config(llm_model)
        self.llm_client = LLMClientFactory.create_client(llm_config)

        # Agent properties
        self.agent_id = "entity_annotator"
        self.description = "Annotates entities in queries using [ENTITY_TYPE:value] format"
        self.capabilities = [
            "entity_recognition",
            "entity_annotation",
            "custom_entity_types",
            "position_tracking"
        ]

        # Initialize components
        self.response_parser = EntityAnnotatorResponseParser()

        # Enable training data logging
        self.enable_training_data = config.get('enable_training_data', True)

        # Initialize loggers
        self.training_logger = TrainingLogger(session_manager) if self.enable_training_data else None
        self.activity_logger = ActivityLogger(agent_name=self.name)

    async def process(self, request: AgentRequest) -> AgentResponse:
        """
        Process request to annotate entities
        """
        start_time = datetime.now()

        try:
            # Set query context for activity logger
            query_context = request.context.get('query_context') if hasattr(request, 'context') and isinstance(
                request.context, dict) else None
            if query_context and isinstance(query_context, QueryContextModel):
                self.activity_logger.set_query_context(query_context)

            # Get the query to annotate
            query_to_annotate = request.prompt

            # Check if we have a resolved query from query_classifier
            if request.context and 'context_aware_query' in request.context:
                query_to_annotate = request.context['context_aware_query']
                self.activity_logger.info(f"Using context-aware query: {query_to_annotate}")

            # Log activity
            self.activity_logger.action("Annotating entities in query")

            # Build prompts
            system_prompt = ENTITY_ANNOTATOR_PROMPT
            user_prompt = f"Annotate entities in this query: {query_to_annotate}"

            # Get LLM response with retry
            llm_response = await self._call_llm_with_retry(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                request=request,
                query=query_to_annotate
            )

            if not llm_response:
                return self._create_error_response(
                    request,
                    "Failed to get LLM response",
                    start_time
                )

            # Parse response
            self.activity_logger.action("Parsing entity annotations")
            result = self.response_parser.parse(llm_response, query_to_annotate)

            # Log what was identified
            entity_count = len(result.entities)
            unique_types = len(result.entity_types)

            self.activity_logger.identified(
                "entities",
                {
                    "total_entities": entity_count,
                    "unique_types": unique_types,
                    "confidence": result.confidence
                }
            )

            # Log entity types found
            if result.entity_types:
                self.activity_logger.info(f"Entity types found: {', '.join(result.entity_types)}")

            # Log any new/custom entity types
            custom_types = [t for t in result.entity_types if
                            t not in ['PERSON', 'LOCATION', 'TIME', 'PHONE', 'NATIONALITY']]
            if custom_types:
                self.activity_logger.info(f"Custom entity types created: {', '.join(custom_types)}")

            # Log annotated query to special log file
            if result.annotated_query and result.annotated_query != query_to_annotate:
                AnnotatedQueryLogger.log_annotation(
                    original_query=query_to_annotate,
                    annotated_query=result.annotated_query,
                    query_id=request.request_id,
                    session_id=request.context.get('session_id') if request.context else None
                )
                self.activity_logger.info("Logged annotated query to file")

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
                    "entity_count": entity_count,
                    "unique_types": unique_types
                }
            )

        except Exception as e:
            logger.error(f"Error in entity annotation: {e}")
            self.activity_logger.error("Failed to annotate entities", error=e)
            return self._create_error_response(
                request,
                f"Processing failed: {str(e)}",
                start_time
            )

    async def _call_llm_with_retry(self, system_prompt: str, user_prompt: str,
                                   request: AgentRequest, query: str, retry_count: int = 0) -> Optional[str]:
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

            # Log to PostgresSQL in background
            if self.training_logger:
                self.training_logger.log_llm_interaction_background(
                    session_id=request.context.get('session_id'),
                    query_id=request.request_id,
                    query_text=query,
                    event_type="entity_annotation",
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
    "query": "{query}",
    "annotated_query": "Show [RISK:high risk] [NATIONALITY:Syrians] in [CITY:Dubai]",
    "entities": [
        {{"type": "RISK", "value": "high risk", "start_pos": 5, "end_pos": 14}}
    ],
    "entity_types": ["RISK", "NATIONALITY", "CITY"],
    "confidence": 0.9
}}"""

                return await self._call_llm_with_retry(
                    system_prompt,
                    enhanced_prompt,
                    request,
                    query,
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

    async def validate_request(self, request: AgentRequest) -> bool:
        """Validate if the agent can handle this request"""
        # Entity annotator can handle any text query
        return True
