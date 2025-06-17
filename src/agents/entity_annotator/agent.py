# src/agents/entity_annotator/agent.py
"""
Entity Annotator Agent - Annotates entities in queries for intelligence analysis
Modernized version using BaseAgent with resource management
"""
import json
from datetime import datetime
from typing import Optional

from .constants import MAX_RETRIES
from .prompt import ENTITY_ANNOTATOR_PROMPT
from .response_parser import EntityAnnotatorResponseParser
from ..base_agent import BaseAgent, AgentResponse, AgentRequest
from ...core.annotated_query_logger import AnnotatedQueryLogger
from ...core.logger import get_logger

logger = get_logger(__name__)


class EntityAnnotatorAgent(BaseAgent):
    """Agent for annotating entities in queries"""

    def __init__(self, name: str, config: dict, resource_manager):
        """Initialize with agent-specific components only"""
        super().__init__(name, config, resource_manager)
        
        # Agent-specific components
        self.response_parser = EntityAnnotatorResponseParser()
        
        # Initialize annotated query logger (fire-and-forget)
        self.annotated_query_logger = AnnotatedQueryLogger()
        
        # Agent metadata handled by base class
        # self.agent_id, self.description, self.capabilities are in base class

    async def validate_request(self, request: AgentRequest) -> bool:
        """Entity annotator can handle any request"""
        return True

    async def process_internal(self, request: AgentRequest) -> AgentResponse:
        """
        Core entity annotation logic
        All infrastructure concerns handled by BaseAgent
        """
        start_time = datetime.now()
        
        # Log activity
        self.activity_logger.action("Starting entity annotation")
        
        # Build prompts
        system_prompt = ENTITY_ANNOTATOR_PROMPT
        user_prompt = f"Annotate entities in this query: {request.prompt}"
        
        # Get LLM response with retry logic
        llm_response = await self._call_llm_with_retry(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            request=request
        )
        
        if not llm_response:
            raise ValueError("Failed to get valid LLM response after retries")
        
        # Parse response
        self.activity_logger.action("Parsing entity annotation response")
        result = self.response_parser.parse(llm_response, request.prompt)
        
        # Initialize entity_type_counts
        entity_type_counts = {}
        
        # Log what was identified
        entity_count = len(result.entities)
        if entity_count > 0:
            # Collect entity types from Entity objects
            entity_types = []
            for entity in result.entities:
                # Entity is an object with attributes, not a dict
                entity_types.append(entity.type)
                entity_type_counts[entity.type] = entity_type_counts.get(entity.type, 0) + 1
            
            self.activity_logger.identified(
                "entities",
                {
                    "count": entity_count,
                    "types": list(set(entity_types))
                }
            )
            
            for entity_type, count in entity_type_counts.items():
                self.activity_logger.info(f"Found {count} {entity_type} entities")
        else:
            self.activity_logger.info("No entities found to annotate")
        
        # Log confidence
        self.activity_logger.info(f"Annotation confidence: {result.confidence:.2f}")
        
        # Fire-and-forget: Log annotated query
        # This is non-blocking and doesn't affect the response
        if result.annotated_query and entity_count > 0:
            try:
                query_context = {
                    'query_id': request.context.get('query_id') if request.context else None,
                    'session_id': request.context.get('session_id') if request.context else None,
                    'user_id': request.context.get('user_id') if request.context else None
                }
                self.annotated_query_logger.log_annotation(
                    original_query=request.prompt,
                    annotated_query=result.annotated_query,
                    query_id=query_context.get('query_id'),
                    session_id=query_context.get('session_id')
                )
            except Exception as e:
                # Log error but don't fail the request
                logger.warning(f"Failed to log annotated query: {e}")
        
        # Create response with metadata
        return self._create_success_response(
            request=request,
            result=result.to_dict(),
            start_time=start_time,
            entity_count=entity_count,
            confidence=result.confidence,
            unique_entity_types=len(entity_type_counts)
        )

    async def _call_llm_with_retry(
        self, 
        system_prompt: str, 
        user_prompt: str,
        request: AgentRequest,
        retry_count: int = 0
    ) -> Optional[str]:
        """
        Call LLM with retry logic specific to entity annotation
        """
        max_retries = self.config.get('max_retries', MAX_RETRIES)
        
        # Use base class helper for LLM call
        response = await self.call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            request=request,
            event_type="entity_annotation",
            max_retries=0  # We handle retry ourselves for JSON validation
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
            if retry_count < max_retries:
                self.activity_logger.retry(f"Retrying with schema hint (attempt {retry_count + 2}/{max_retries + 1})")
                enhanced_prompt = EntityAnnotatorAgent._enhance_prompt_with_schema(user_prompt)
                
                return await self._call_llm_with_retry(
                    system_prompt,
                    enhanced_prompt,
                    request,
                    retry_count + 1
                )
            return None

    @staticmethod
    def _enhance_prompt_with_schema(user_prompt: str) -> str:
        """Add schema hint to prompt for retry attempts"""
        return f"""{user_prompt}

IMPORTANT: Return ONLY valid JSON in this exact format:
{{
    "entities": [
        {{
            "text": "Dubai",
            "type": "LOCATION",
            "start_pos": 15,
            "end_pos": 20,
            "confidence": 0.95
        }}
    ],
    "annotated_query": "Show me people in [LOCATION:Dubai]",
    "confidence": 0.9
}}

Ensure all JSON syntax is correct with proper quotes and commas."""