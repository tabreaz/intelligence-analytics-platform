# src/agents/agent_manager.py
import asyncio
import time
import uuid
from typing import Dict, List, Optional

from src.agents.base_agent import BaseAgent, AgentRequest, AgentResponse
from src.core.config_manager import ConfigManager
from src.core.logger import get_logger

logger = get_logger(__name__)


class AgentManager:
    """Manages all intelligence agents and workflows"""

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.agents: Dict[str, BaseAgent] = {}
        self.workflow_config = config_manager.get_workflow_config()
        self._initialize_agents()

    def _initialize_agents(self):
        """Initialize all enabled agents"""
        # Import agents dynamically
        from src.agents.location_extractor.agent import LocationExtractorAgent
        from src.agents.query_classifier.agent import QueryClassifierAgent
        from src.agents.time_parser.agent import TimeParserAgent
        from src.agents.risk_filter.agent import RiskFilterAgent
        from src.agents.profile_filter.agent import ProfileFilterAgent
        from src.agents.entity_annotator.agent import EntityAnnotatorAgent
        from src.agents.query_orchestrator.agent import QueryOrchestratorAgent
        from src.agents.unified_filter.agent import UnifiedFilterAgent
        from src.agents.query_executor.agent import QueryExecutorAgent

        # Placeholder for other agents
        class PlaceholderAgent(BaseAgent):
            async def validate_request(self, request):
                return False  # Disable placeholder agents

            async def process(self, request):
                pass

        agent_classes = {
            'location_extractor': LocationExtractorAgent,
            # New agents
            'query_classifier': QueryClassifierAgent,
            'time_parser': TimeParserAgent,
            'risk_filter': RiskFilterAgent,
            'profile_filter': ProfileFilterAgent,
            'entity_annotator': EntityAnnotatorAgent,
            'query_orchestrator': QueryOrchestratorAgent,
            'unified_filter': UnifiedFilterAgent,
            'query_executor': QueryExecutorAgent
        }

        for agent_name, agent_class in agent_classes.items():
            try:
                agent_config = self.config_manager.get_agent_config(agent_name)
                if agent_config.get('enabled', False) and agent_class != PlaceholderAgent:
                    # Special handling for query_orchestrator which needs agent_manager
                    if agent_name == 'query_orchestrator':
                        agent = agent_class(agent_name, agent_config, self.config_manager, 
                                          agent_manager=self)
                    else:
                        agent = agent_class(agent_name, agent_config, self.config_manager)
                    self.agents[agent_name] = agent
                    logger.info(f"Initialized agent: {agent_name}")
                elif agent_class == PlaceholderAgent:
                    logger.info(f"Skipped placeholder agent: {agent_name}")
            except Exception as e:
                logger.error(f"Failed to initialize agent {agent_name}: {e}")

    async def process_request(self, prompt: str, context: Dict = None,
                              classification: str = "UNCLASS") -> Dict[str, AgentResponse]:
        """Process request through relevant agents with proper workflow"""
        request_id = str(uuid.uuid4())
        context = context or {}

        request = AgentRequest(
            request_id=request_id,
            prompt=prompt,
            context=context,
            classification=classification,
            timestamp=time.time()
        )

        # Get relevant agents for this request
        relevant_agents = await self._get_relevant_agents(request)

        # Execute agents in priority order with context passing
        results = {}

        # Group agents by priority for proper workflow
        priority_groups = self._group_by_priority(relevant_agents)

        for priority in sorted(priority_groups.keys()):
            agents_in_group = priority_groups[priority]

            if len(agents_in_group) == 1:
                # Single agent - execute directly
                agent = agents_in_group[0]
                response = await agent.execute(request)
                results[agent.name] = response

                # Pass successful results to next priority level
                if response.status.value == "completed":
                    request.context[agent.name] = response.result

            else:
                # Multiple agents at same priority - execute in parallel
                tasks = [agent.execute(request) for agent in agents_in_group]
                responses = await asyncio.gather(*tasks, return_exceptions=True)

                for agent, response in zip(agents_in_group, responses):
                    if isinstance(response, Exception):
                        logger.error(f"Agent {agent.name} failed: {response}")
                        continue

                    results[agent.name] = response

                    # Pass successful results to context
                    if response.status.value == "completed":
                        request.context[agent.name] = response.result

        return results

    async def _get_relevant_agents(self, request: AgentRequest) -> List[BaseAgent]:
        """Determine which agents should process the request"""
        relevant_agents = []

        for agent in self.agents.values():
            if await agent.validate_request(request):
                relevant_agents.append(agent)

        return relevant_agents

    def _group_by_priority(self, agents: List[BaseAgent]) -> Dict[int, List[BaseAgent]]:
        """Group agents by priority level"""
        priority_groups = {}

        for agent in agents:
            priority = agent.priority
            if priority not in priority_groups:
                priority_groups[priority] = []
            priority_groups[priority].append(agent)

        return priority_groups

    def get_agent(self, agent_name: str) -> Optional[BaseAgent]:
        """Get specific agent by name"""
        return self.agents.get(agent_name)

    def list_agents(self) -> List[str]:
        """List all available agent names"""
        return list(self.agents.keys())
