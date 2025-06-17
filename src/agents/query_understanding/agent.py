# src/agents/query_understanding/agent.py
"""
Query Understanding Agent - Main orchestrator
Shows the complete flow from query receipt to classification
"""
import json
from datetime import datetime
from typing import Dict, List, Optional, Any

from src.core.logger import get_logger
from src.core.session_manager import EnhancedSessionManager
from src.core.session_manager_models import (
    QueryContext, QueryStatus, Session
)
from .classifiers.llm_classifier import LLMClassifier
from .models import (
    QueryCategory, QueryResult
)
from .prompts.classification_prompt_builder import (
    ClassificationPromptBuilder, ContextExtractor
)
from .status_publisher import QueryStatusPublisher

logger = get_logger(__name__)


class QueryUnderstandingAgent:
    """
    Main Query Understanding Agent
    This shows the complete flow including the first LLM interaction
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Initialize components
        self.session_manager = EnhancedSessionManager(config['database'])
        self.prompt_builder = ClassificationPromptBuilder()
        self.context_extractor = ContextExtractor()
        self.llm_classifier = LLMClassifier(config['llm'])
        self.status_publisher = QueryStatusPublisher(config['redis'])

    async def initialize(self):
        """Initialize all components"""
        await self.session_manager.initialize()
        await self.status_publisher.initialize()
        logger.info("Query Understanding Agent initialized")

    async def process_query(self,
                            query: str,
                            session_id: Optional[str] = None) -> QueryResult:
        """
        Main entry point - processes a user query
        This is where the first LLM interaction happens
        """
        start_time = datetime.utcnow()

        # Step 1: Session Management
        await self.status_publisher.publish_status(
            session_id, "STARTED",
            {"message": "Retrieving session context..."}
        )

        session = await self.session_manager.get_or_create_session(session_id)

        # Step 2: Get Query History
        query_history = await self.session_manager.get_session_history(
            session.session_id, limit=5
        )

        # Step 3: Extract Available Context (Pre-LLM)
        await self.status_publisher.publish_status(
            session.session_id, "CONTEXT_EXTRACTION",
            {"message": "Extracting context from query and history..."}
        )

        available_context = self.context_extractor.extract_available_context(
            query, session, query_history
        )

        logger.info(f"Extracted context: {json.dumps(available_context, indent=2)}")

        # Step 4: Build First LLM Prompt
        await self.status_publisher.publish_status(
            session.session_id, "BUILDING_PROMPT",
            {"message": "Preparing analysis prompt..."}
        )

        prompt = self.prompt_builder.build_first_prompt(
            query, session, query_history, available_context
        )

        # Step 5: First LLM Call - Classification with Context
        await self.status_publisher.publish_status(
            session.session_id, "CLASSIFICATION",
            {"message": "Analyzing query intent and context..."}
        )

        classification_result = await self._classify_with_context(
            prompt, session.session_id
        )

        # Step 6: Process LLM Response
        query_context = await self._process_classification_result(
            query, classification_result, session, query_history
        )

        # Step 7: Update Session Context
        await self._update_session_context(session, query_context)

        # Step 8: Check if Clarification Needed
        if query_context.clarifications_requested:
            return await self._handle_clarification(query_context, session)

        # Step 9: Route to Analysis Agent
        await self.status_publisher.publish_status(
            session.session_id, "ROUTING",
            {"message": f"Routing to {query_context.category.value} analysis..."}
        )

        # Record query in session
        query_id = await self.session_manager.add_query_to_session(
            session.session_id, query_context
        )

        execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)

        return QueryResult(
            request_id=query_id,
            session_id=session.session_id,
            category=query_context.category,
            parameters=query_context.extracted_params,
            execution_time_ms=execution_time,
            metadata={
                'confidence': query_context.confidence,
                'inherited_context': query_context.inherited_elements,
                'context_type': query_context.inheritance_type.value
            }
        )

    async def _classify_with_context(self,
                                     prompt: Dict[str, str],
                                     session_id: str) -> Dict[str, Any]:
        """
        Execute the first LLM call with full context
        This is where the magic happens
        """

        # Log the actual prompt being sent
        logger.info(f"LLM System Prompt:\n{prompt['system'][:500]}...")
        logger.info(f"LLM User Prompt:\n{prompt['user']}")

        # Track the LLM interaction
        interaction_start = datetime.utcnow()

        try:
            # Call LLM
            response = await self.llm_classifier.classify(
                system_prompt=prompt['system'],
                user_prompt=prompt['user']
            )

            # Calculate metrics
            latency_ms = int((datetime.utcnow() - interaction_start).total_seconds() * 1000)

            # Record the interaction for training
            await self.session_manager.record_interaction_event(
                session_id=session_id,
                query_id=None,  # Will be set later
                event_type="classification",
                llm_interaction={
                    'model': self.config['llm']['model'],
                    'prompt_template': 'classification_with_context',
                    'prompt_tokens': len(prompt['system'] + prompt['user']) // 4,
                    'completion_tokens': len(json.dumps(response)) // 4,
                    'latency_ms': latency_ms
                },
                input_data={'query': prompt['user']},
                output_data=response,
                success=True,
                confidence=response.get('classification', {}).get('confidence', 0)
            )

            return response

        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            # Record failure
            await self.session_manager.record_interaction_event(
                session_id=session_id,
                query_id=None,
                event_type="classification",
                input_data={'query': prompt['user']},
                success=False
            )
            raise

    async def _process_classification_result(self,
                                             query: str,
                                             llm_response: Dict[str, Any],
                                             session: Session,
                                             query_history: List[QueryContext]) -> QueryContext:
        """
        Process the LLM classification response into QueryContext
        """
        classification = llm_response.get('classification', {})

        # Create query context
        query_context = QueryContext(
            query_text=query,
            normalized_query=query.lower().strip(),
            category=QueryCategory(classification.get('category')),
            confidence=classification.get('confidence', 0),
            status=QueryStatus.PROCESSING
        )

        # Handle context inheritance if this is a continuation
        if llm_response.get('classification', {}).get('is_continuation') and query_history:
            inheritance_info = llm_response.get('context_inheritance', {})

            query_context.inherited_from_query = str(query_history[0].query_id)
            query_context.inheritance_type = self._determine_inheritance_type(
                inheritance_info
            )
            query_context.inherited_elements = inheritance_info.get('inherited_parameters', {})

            # Merge inherited and new parameters
            all_params = {}
            all_params.update(inheritance_info.get('inherited_parameters', {}))
            all_params.update(llm_response.get('new_parameters', {}))

            # Apply modifications
            for param, mod in llm_response.get('modified_parameters', {}).items():
                all_params[param] = mod.get('new_value')

            query_context.extracted_params = all_params
        else:
            # Standalone query
            query_context.extracted_params = llm_response.get('initial_parameters', {})

        # Extract entities
        query_context.entities_mentioned = llm_response.get('entities_detected', {})

        # Handle ambiguities
        ambiguities = llm_response.get('ambiguities', [])
        if ambiguities:
            query_context.clarifications_requested = [
                {
                    'parameter': amb['parameter'],
                    'question': amb['suggested_clarification'],
                    'issue': amb['issue']
                }
                for amb in ambiguities
            ]
            query_context.status = QueryStatus.CLARIFYING

        return query_context

    async def _update_session_context(self,
                                      session: Session,
                                      query_context: QueryContext):
        """Update session with new context"""

        # Build updated context
        updated_context = {
            'filters': query_context.extracted_params,
            'entities': query_context.entities_mentioned,
            'last_category': query_context.category.value if query_context.category else None,
            'time_range': query_context.extracted_params.get('time_range'),
            'locations': query_context.entities_mentioned.get('locations', [])
        }

        # Merge with existing context
        session.active_context.update(updated_context)

        # Save to database
        await self.session_manager.update_session_context(
            session.session_id,
            session.active_context
        )

    def _determine_inheritance_type(self, inheritance_info: Dict) -> str:
        """Determine the type of inheritance"""
        if not inheritance_info.get('inherit_from_previous'):
            return "none"

        inherited_params = inheritance_info.get('inherited_parameters', {})
        if len(inherited_params) > 3:  # Arbitrary threshold
            return "full"
        else:
            return "partial"

    async def _handle_clarification(self,
                                    query_context: QueryContext,
                                    session: Session) -> QueryResult:
        """Handle clarification requests"""
        # This would trigger the clarification flow
        # For now, return with clarification flag

        return QueryResult(
            request_id=str(session.session_id),
            session_id=session.session_id,
            category=query_context.category,
            parameters=query_context.extracted_params,
            requires_clarification=True,
            clarification_requests=query_context.clarifications_requested
        )
