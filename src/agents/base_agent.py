# src/agents/base_agent.py
"""
Modern Base Agent with centralized resource management and common functionality
No legacy support - all agents must use ResourceManager
"""
import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional

from src.core.activity_logger import ActivityLogger
from src.core.logger import get_logger
from src.core.resource_manager import ResourceManager
from src.core.session_manager_models import QueryContext
from src.core.training_logger import TrainingLogger

logger = get_logger(__name__)


class AgentStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class AgentRequest:
    request_id: str
    prompt: str
    context: Dict[str, Any] = field(default_factory=dict)
    classification: str = "UNCLASS"
    priority: int = 1
    timeout: int = 300
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentResponse:
    request_id: str
    agent_name: str
    status: AgentStatus
    result: Dict[str, Any]
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Modern base class for all intelligence agents
    Requires ResourceManager - no legacy support
    """

    def __init__(self, name: str, config: Dict[str, Any], resource_manager: ResourceManager):
        """
        Initialize base agent

        Args:
            name: Agent name
            config: Agent configuration
            resource_manager: Required centralized resource manager
        """
        if not resource_manager:
            raise ValueError(f"ResourceManager is required for agent {name}")

        self.name = name
        self.config = config
        self.resource_manager = resource_manager
        self.status = AgentStatus.IDLE
        self.logger = get_logger(f"agents.{name}")

        # Basic config
        self.enabled = config.get('enabled', True)
        self.priority = config.get('priority', 1)

        # Initialize components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all common components from ResourceManager"""
        # Activity logger
        self.activity_logger = ActivityLogger(agent_name=self.name)

        # Get resources based on agent configuration
        resources = self.resource_manager.get_resources_for_agent(self.name)

        # Core resources
        self.llm_client = resources.get('llm_client')
        self.session_manager = resources.get('session_manager')
        self.config_manager = resources.get('config_manager')
        self.clickhouse_client = resources.get('clickhouse_client')
        self.redis_client = resources.get('redis_client')

        # Training logger (conditional)
        self.enable_training_data = self.config.get('enable_training_data', True)
        self.training_logger = None
        if self.enable_training_data and self.session_manager:
            self.training_logger = TrainingLogger(self.session_manager)

        # Agent metadata
        self.agent_id = self.config.get('agent_id', self.name)
        self.description = self.config.get('description', f"{self.name} agent")
        self.capabilities = self.config.get('capabilities', [])

        # Common configurations
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delay = self.config.get('retry_delay', 1.0)
        self.default_confidence = self.config.get('default_confidence', 0.8)

    @abstractmethod
    async def process_internal(self, request: AgentRequest) -> AgentResponse:
        """
        Agent-specific processing logic
        Must be implemented by each agent
        """
        pass

    @abstractmethod
    async def validate_request(self, request: AgentRequest) -> bool:
        """
        Validate if the agent can handle this request
        Must be implemented by each agent
        """
        pass

    async def execute(self, request: AgentRequest) -> AgentResponse:
        """
        Execute agent with full lifecycle management
        Called by AgentManager
        """
        if not self.enabled:
            return self._create_error_response(request, "Agent is disabled")

        if not await self.validate_request(request):
            return self._create_error_response(request, "Request validation failed")

        self.status = AgentStatus.RUNNING
        start_time = datetime.now()

        try:
            # Execute with timeout
            response = await asyncio.wait_for(
                self.process(request),
                timeout=request.timeout
            )

            self.status = AgentStatus.COMPLETED
            return response

        except asyncio.TimeoutError:
            self.status = AgentStatus.TIMEOUT
            return self._create_timeout_response(request, start_time)
        except Exception as e:
            self.status = AgentStatus.FAILED
            logger.error(f"Agent {self.name} execution failed: {e}", exc_info=True)
            return self._create_error_response(request, str(e), start_time)

    async def process(self, request: AgentRequest) -> AgentResponse:
        """
        Main processing method with common functionality
        Wraps agent-specific process_internal()
        """
        start_time = datetime.now()

        # Use resource tracking
        async with self.resource_manager.track_agent_usage(self.name):
            try:
                # Setup query context
                query_context = BaseAgent._extract_query_context(request)
                if query_context:
                    self.activity_logger.set_query_context(query_context)

                # Log start
                self.activity_logger.action(f"Starting {self.name} processing")

                # Validate required resources are available
                self._validate_resources()

                # Call agent-specific processing
                response = await self.process_internal(request)

                # Ensure execution time is set
                if response.execution_time == 0:
                    response.execution_time = (datetime.now() - start_time).total_seconds()

                # Log success
                self.activity_logger.info(f"Completed successfully in {response.execution_time:.2f}s")

                return response

            except Exception as e:
                logger.error(f"Error in {self.name}: {e}", exc_info=True)
                self.activity_logger.error(f"Processing failed: {str(e)}")
                return self._create_error_response(request, str(e), start_time)

    # Resource validation

    def _validate_resources(self):
        """Validate required resources are available"""
        required = set(self.config.get('required_resources', []))

        resource_map = {
            'llm': self.llm_client,
            'clickhouse': self.clickhouse_client,
            'redis': self.redis_client,
            'session_manager': self.session_manager
        }

        for resource_name in required:
            if resource_name in resource_map and not resource_map[resource_name]:
                raise RuntimeError(f"{self.name}: Required resource '{resource_name}' not available")

    # LLM interaction helpers

    async def call_llm(self, system_prompt: str, user_prompt: str,
                       request: AgentRequest, event_type: str,
                       max_retries: Optional[int] = None,
                       **kwargs) -> Optional[str]:
        """
        Call LLM with automatic retry and logging

        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            request: Original request for context
            event_type: Event type for logging (e.g., "time_parsing")
            max_retries: Override default retries
            **kwargs: Additional data for training logs

        Returns:
            LLM response or None if failed
        """
        if not self.llm_client:
            raise RuntimeError(f"{self.name}: LLM client not available")

        max_retries = max_retries or self.max_retries
        last_error = None

        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    self.activity_logger.info(f"LLM retry attempt {attempt + 1}/{max_retries}")
                    await asyncio.sleep(self.retry_delay * attempt)

                # Track timing
                llm_start_time = datetime.now()

                # Call LLM
                response = await self.llm_client.generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt
                )

                # Log for training if enabled
                if self.training_logger and response:
                    self._schedule_training_log(
                        request=request,
                        event_type=event_type,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        llm_response=response,
                        llm_start_time=llm_start_time,
                        **kwargs
                    )

                return response

            except Exception as e:
                last_error = e
                logger.warning(f"LLM call failed (attempt {attempt + 1}): {e}")

        self.activity_logger.error(f"All LLM attempts failed: {last_error}")
        return None

    def _schedule_training_log(self, **kwargs):
        """Schedule training data logging in background"""
        if not self.training_logger:
            return

        try:
            # Get model info
            model_config = {
                'model': getattr(self.llm_client, 'model', 'unknown'),
                'temperature': getattr(self.llm_client, 'temperature', 0.1),
                'max_tokens': getattr(self.llm_client, 'max_tokens', 1000)
            }

            # Log in background (fire and forget)
            self.training_logger.log_llm_interaction_background(
                session_id=kwargs.get('request').context.get('session_id'),
                query_id=kwargs.get('request').context.get('query_id', kwargs.get('request').request_id),
                query_text=kwargs.get('request').prompt,
                event_type=kwargs.get('event_type'),
                llm_response=kwargs.get('llm_response'),
                llm_start_time=kwargs.get('llm_start_time'),
                prompts={
                    'system': kwargs.get('system_prompt'),
                    'user': kwargs.get('user_prompt')
                },
                result=kwargs.get('result', {}),
                model_config=model_config,
                **{k: v for k, v in kwargs.items() if k not in [
                    'request', 'event_type', 'system_prompt', 'user_prompt',
                    'llm_response', 'llm_start_time', 'result'
                ]}
            )
        except Exception as e:
            logger.error(f"Failed to schedule training log: {e}")

    # Helper methods

    @staticmethod
    def _extract_query_context(request: AgentRequest) -> Optional[QueryContext]:
        """Extract query context from request"""
        if not request.context:
            return None

        query_context = request.context.get('query_context')
        if query_context and isinstance(query_context, QueryContext):
            return query_context
        return None

    # Response creation helpers

    def _create_error_response(self, request: AgentRequest, error: str,
                               start_time: Optional[datetime] = None) -> AgentResponse:
        """Create standardized error response"""
        execution_time = 0.0
        if start_time:
            execution_time = (datetime.now() - start_time).total_seconds()

        return AgentResponse(
            request_id=request.request_id,
            agent_name=self.name,
            status=AgentStatus.FAILED,
            result={"error": error},
            error=error,
            execution_time=execution_time,
            metadata={"error_type": type(error).__name__ if isinstance(error, Exception) else "error"}
        )

    def _create_timeout_response(self, request: AgentRequest, start_time: datetime) -> AgentResponse:
        """Create standardized timeout response"""
        return AgentResponse(
            request_id=request.request_id,
            agent_name=self.name,
            status=AgentStatus.TIMEOUT,
            result={},
            error=f"Processing timeout after {request.timeout}s",
            execution_time=(datetime.now() - start_time).total_seconds(),
            metadata={"timeout": request.timeout}
        )

    def _create_success_response(self, request: AgentRequest, result: Dict[str, Any],
                                 start_time: datetime, **metadata) -> AgentResponse:
        """Create standardized success response"""
        return AgentResponse(
            request_id=request.request_id,
            agent_name=self.name,
            status=AgentStatus.COMPLETED,
            result=result,
            execution_time=(datetime.now() - start_time).total_seconds(),
            metadata=metadata
        )