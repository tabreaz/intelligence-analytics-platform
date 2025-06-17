# src/agents/query_orchestrator/agent.py
"""
Query Orchestrator Agent - Coordinates all other agents for query processing
"""
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from .constants import AGENT_MAPPING, PARALLEL_TIMEOUT_SECONDS
from .models import OrchestratorResult, AgentResult, OrchestratorStatus
from .result_merger import ResultMerger
from ..base_agent import BaseAgent, AgentResponse, AgentStatus, AgentRequest
from ...core.activity_logger import ActivityLogger
from ...core.config_manager import ConfigManager
from ...core.logger import get_logger
from ...core.session_manager_models import QueryContext as QueryContextModel

logger = get_logger(__name__)


class QueryOrchestratorAgent(BaseAgent):
    """Orchestrates query processing across multiple agents"""

    def __init__(self, name: str, config: dict, config_manager: ConfigManager,
                 agent_manager=None, session_manager=None):
        """
        Initialize Query Orchestrator Agent
        
        Args:
            agent_manager: AgentManager instance for accessing other agents
        """
        super().__init__(name, config)
        self.config_manager = config_manager
        self.session_manager = session_manager
        self.agent_manager = agent_manager

        # Agent properties
        self.agent_id = "query_orchestrator"
        self.description = "Orchestrates query processing across all agents"
        self.capabilities = [
            "query_classification",
            "ambiguity_resolution",
            "parallel_agent_execution",
            "result_merging",
            "deduplication"
        ]

        # Initialize activity logger
        self.activity_logger = ActivityLogger(agent_name=self.name)

        # Cache for initialized agents
        self._agent_cache = {}
        
        # Simple query history for context (session_id -> list of classification results)
        self._query_history = {}

    async def process(self, request: AgentRequest) -> AgentResponse:
        """
        Process request by orchestrating multiple agents
        """
        start_time = datetime.now()
        result = OrchestratorResult()

        try:
            # Set query context for activity logger
            query_context = request.context.get('query_context') if hasattr(request, 'context') and isinstance(
                request.context, dict) else None
            if query_context and isinstance(query_context, QueryContextModel):
                self.activity_logger.set_query_context(query_context)

            self.activity_logger.action("Starting query orchestration")

            # Get session_id for history tracking
            session_id = request.context.get('session_id') if request.context else None
            
            # Build context with history for classifier
            enhanced_request = self._build_context_aware_request(request, session_id)

            # Step 1: Classification with context
            result.orchestrator_status = OrchestratorStatus.CLASSIFYING
            classification_result = await self._run_classification(enhanced_request)

            if not classification_result:
                raise Exception("Classification failed")

            # Update result with classification data
            result.classification = classification_result.get('classification', {})
            result.context_aware_query = classification_result.get('context_aware_query', request.prompt)
            result.is_continuation = classification_result.get('is_continuation', False)
            result.domains = classification_result.get('domains', [])
            result.agents_required = classification_result.get('agents_required', [])
            result.ambiguities = classification_result.get('ambiguities', [])
            result.has_ambiguities = len(result.ambiguities) > 0

            # Step 2: Handle ambiguities if any
            if result.has_ambiguities:
                result.orchestrator_status = OrchestratorStatus.RESOLVING_AMBIGUITIES
                self.activity_logger.warning(f"Found {len(result.ambiguities)} ambiguities")

                # For now, we'll return the ambiguities to the user
                # In a real implementation, this would involve user interaction
                result.ambiguities_resolved = False

                # Return early with ambiguities
                result.total_execution_time = (datetime.now() - start_time).total_seconds()

                return AgentResponse(
                    request_id=request.request_id,
                    agent_name=self.name,
                    status=AgentStatus.COMPLETED,
                    result=result.to_dict(),
                    execution_time=result.total_execution_time,
                    metadata={
                        "has_ambiguities": True,
                        "ambiguity_count": len(result.ambiguities)
                    }
                )

            # Step 3: Check if domain is supported
            domain_check = classification_result.get('domain_check', {})
            if not domain_check.get('is_supported', True):
                self.activity_logger.warning("Query domain not supported")
                result.error = domain_check.get('message', 'Query domain not supported')
                result.orchestrator_status = OrchestratorStatus.FAILED

                return AgentResponse(
                    request_id=request.request_id,
                    agent_name=self.name,
                    status=AgentStatus.FAILED,
                    result=result.to_dict(),
                    error=result.error,
                    execution_time=(datetime.now() - start_time).total_seconds()
                )

            # Step 4: Fire entity annotator (truly non-blocking)
            self.activity_logger.action("Starting entity annotator in background")
            # Fire and forget - don't even store the task
            asyncio.create_task(self._run_entity_annotator_fire_forget(request, result.context_aware_query))
            
            # Step 5: Run required agents in parallel
            result.orchestrator_status = OrchestratorStatus.PROCESSING_FILTERS
            self.activity_logger.action(f"Running {len(result.agents_required)} required agents")

            # Prepare agent requests
            agent_requests = self._prepare_agent_requests(
                request,
                result.context_aware_query,
                result.agents_required
            )

            # Execute agents in parallel
            agent_results = await self._execute_agents_parallel(agent_requests)

            # Step 6: Process and merge results (no entity annotator results)
            result = self._merge_agent_results(result, agent_results)

            # Step 8: Mark as completed
            result.orchestrator_status = OrchestratorStatus.COMPLETED
            result.total_execution_time = (datetime.now() - start_time).total_seconds()

            self.activity_logger.info(f"Orchestration completed in {result.total_execution_time:.2f}s")
            
            # Store classification result for future context
            if session_id:
                self._store_classification_for_context(session_id, request.prompt, classification_result)

            # Create response
            return AgentResponse(
                request_id=request.request_id,
                agent_name=self.name,
                status=AgentStatus.COMPLETED,
                result=result.to_dict(),
                execution_time=result.total_execution_time,
                metadata={
                    "agents_executed": len(agent_results),
                    "has_errors": any(r.error for r in agent_results.values())
                }
            )

        except Exception as e:
            logger.error(f"Orchestration failed: {e}")
            self.activity_logger.error("Orchestration failed", error=e)

            result.orchestrator_status = OrchestratorStatus.FAILED
            result.error = str(e)
            result.total_execution_time = (datetime.now() - start_time).total_seconds()

            return AgentResponse(
                request_id=request.request_id,
                agent_name=self.name,
                status=AgentStatus.FAILED,
                result=result.to_dict(),
                error=str(e),
                execution_time=result.total_execution_time
            )

    async def _run_classification(self, request: AgentRequest) -> Optional[Dict[str, Any]]:
        """Run the query classifier agent"""
        try:
            classifier_agent = await self._get_agent("query_classifier")
            if not classifier_agent:
                raise Exception("Query classifier agent not available")

            response = await classifier_agent.process(request)

            if response.status == AgentStatus.COMPLETED:
                return response.result
            else:
                logger.error(f"Classification failed: {response.error}")
                return None

        except Exception as e:
            logger.error(f"Error running classifier: {e}")
            return None

    def _prepare_agent_requests(self, original_request: AgentRequest,
                                context_aware_query: str,
                                agents_required: List[str]) -> List[Tuple[str, AgentRequest]]:
        """Prepare requests for each required agent"""
        requests = []

        # Create enhanced context with classification results
        enhanced_context = original_request.context.copy() if original_request.context else {}
        enhanced_context['context_aware_query'] = context_aware_query
        enhanced_context['original_request_id'] = original_request.request_id

        for agent_short_name in agents_required:
            # Map short name to full agent name
            agent_full_name = AGENT_MAPPING.get(agent_short_name, agent_short_name)

            # Create request for this agent
            agent_request = AgentRequest(
                request_id=f"{original_request.request_id}_{agent_short_name}",
                prompt=context_aware_query,  # Use context-aware query
                context=enhanced_context,
                timestamp=original_request.timestamp
            )

            requests.append((agent_full_name, agent_request))

        return requests

    async def _execute_agents_parallel(self, agent_requests: List[Tuple[str, AgentRequest]]) -> Dict[str, AgentResult]:
        """Execute multiple agents in parallel"""
        results = {}

        async def run_agent(agent_name: str, request: AgentRequest) -> Tuple[str, AgentResult]:
            """Run a single agent and return result"""
            start_time = datetime.now()

            try:
                agent = await self._get_agent(agent_name)
                if not agent:
                    return agent_name, AgentResult(
                        agent_name=agent_name,
                        status="failed",
                        result={},
                        execution_time=0.0,
                        error=f"Agent {agent_name} not available"
                    )

                response = await agent.process(request)

                return agent_name, AgentResult(
                    agent_name=agent_name,
                    status=response.status.value,
                    result=response.result,
                    execution_time=response.execution_time,
                    error=response.error
                )

            except Exception as e:
                logger.error(f"Error running agent {agent_name}: {e}")
                return agent_name, AgentResult(
                    agent_name=agent_name,
                    status="failed",
                    result={},
                    execution_time=(datetime.now() - start_time).total_seconds(),
                    error=str(e)
                )

        # Create tasks for parallel execution
        tasks = [run_agent(name, req) for name, req in agent_requests]

        # Execute with timeout
        try:
            agent_results = await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=PARALLEL_TIMEOUT_SECONDS
            )

            # Convert to dictionary
            for agent_name, result in agent_results:
                results[agent_name] = result

        except asyncio.TimeoutError:
            logger.error("Parallel agent execution timed out")
            self.activity_logger.error("Some agents timed out during parallel execution")

        return results

    def _merge_agent_results(self, orchestrator_result: OrchestratorResult,
                             agent_results: Dict[str, AgentResult]) -> OrchestratorResult:
        """Merge results from all agents"""
        orchestrator_result.agent_results = agent_results

        # Collect filter results for merging
        filter_results = {}

        for agent_name, agent_result in agent_results.items():
            if agent_result.status != "completed":
                continue

            result_data = agent_result.result

            # Skip entity annotator results (fire-and-forget)
            if agent_name == "entity_annotator":
                continue

            # Store all filter agent results
            if agent_name in ["time_parser", "location_extractor", "profile_filter", "risk_filter"]:
                filter_results[agent_name] = result_data

        # New approach: Store raw results from each agent
        # Profile and risk filters now use filter_tree format
        if 'profile_filter' in filter_results:
            orchestrator_result.profile_filters = filter_results['profile_filter']
        
        if 'risk_filter' in filter_results:
            orchestrator_result.risk_filters = filter_results['risk_filter']
            
        if 'time_parser' in filter_results:
            orchestrator_result.time_filters = filter_results['time_parser']
            
        if 'location_extractor' in filter_results:
            orchestrator_result.location_filters = filter_results['location_extractor']

        # Create a unified filters object for backward compatibility
        orchestrator_result.filters = {
            'profile': orchestrator_result.profile_filters,
            'risk': orchestrator_result.risk_filters,
            'time': orchestrator_result.time_filters,
            'location': orchestrator_result.location_filters
        }

        return orchestrator_result


    async def _get_agent(self, agent_name: str):
        """Get an agent instance from cache or agent manager"""
        # Check cache first
        if agent_name in self._agent_cache:
            return self._agent_cache[agent_name]

        # Get from agent manager
        if self.agent_manager:
            agent = self.agent_manager.get_agent(agent_name)
            if agent:
                self._agent_cache[agent_name] = agent
            return agent

        logger.warning(f"Agent {agent_name} not found")
        return None

    async def validate_request(self, request: AgentRequest) -> bool:
        """Validate if the agent can handle this request"""
        # Orchestrator can handle any query
        return True
    
    def _build_context_aware_request(self, request: AgentRequest, session_id: str) -> AgentRequest:
        """Build request with previous query context for classifier"""
        # Start with original context
        enhanced_context = request.context.copy() if request.context else {}
        
        # Add previous query if available
        if session_id and session_id in self._query_history:
            history = self._query_history[session_id]
            if history:
                # Get the most recent classification
                last_classification = history[-1]
                
                # Add previous query info to context
                enhanced_context['previous_query'] = {
                    'query_text': last_classification.get('original_query', ''),
                    'category': last_classification.get('classification', {}).get('category'),
                    'context_aware_query': last_classification.get('context_aware_query', ''),
                    'extracted_params': {}  # Simplified - classifier will handle the context
                }
        
        # Return enhanced request
        return AgentRequest(
            request_id=request.request_id,
            prompt=request.prompt,
            context=enhanced_context,
            timestamp=request.timestamp
        )
    
    def _store_classification_for_context(self, session_id: str, original_query: str, classification_result: Dict[str, Any]):
        """Store classification result for future context"""
        if not session_id:
            return
            
        if session_id not in self._query_history:
            self._query_history[session_id] = []
        
        # Store key information from classification
        context_data = {
            'original_query': original_query,
            'classification': classification_result.get('classification', {}),
            'context_aware_query': classification_result.get('context_aware_query', original_query),
            'is_continuation': classification_result.get('is_continuation', False),
            'timestamp': datetime.now().isoformat()
        }
        
        # Keep last 5 queries per session
        self._query_history[session_id].append(context_data)
        if len(self._query_history[session_id]) > 5:
            self._query_history[session_id].pop(0)
    
    async def _run_entity_annotator_fire_forget(self, request: AgentRequest, context_aware_query: str):
        """Fire and forget entity annotator - runs completely in background"""
        try:
            agent = await self._get_agent("entity_annotator")
            if not agent:
                return
                
            # Create request for entity annotator
            entity_request = AgentRequest(
                request_id=f"{request.request_id}_entity",
                prompt=context_aware_query,
                context=request.context.copy() if request.context else {},
                timestamp=request.timestamp
            )
            
            # Process in background - we don't care about the result
            await agent.process(entity_request)
            logger.debug("Entity annotator completed in background")
            
        except Exception as e:
            logger.debug(f"Entity annotator failed in background: {e}")
            # Don't log as warning - this is truly fire-and-forget
