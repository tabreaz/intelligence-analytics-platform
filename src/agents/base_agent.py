# src/agents/base_agent.py
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional

from src.core.logger import get_logger

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
    context: Dict[str, Any]
    classification: str = "UNCLASS"
    priority: int = 1
    timeout: int = 300
    timestamp: float = None


@dataclass
class AgentResponse:
    request_id: str
    agent_name: str
    status: AgentStatus
    result: Dict[str, Any]
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None


class BaseAgent(ABC):
    """Base class for all intelligence agents"""

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.status = AgentStatus.IDLE
        self.logger = logging.getLogger(f"agents.{name}")
        self.enabled = config.get('enabled', True)
        self.priority = config.get('priority', 1)

    @abstractmethod
    async def process(self, request: AgentRequest) -> AgentResponse:
        """Process the agent request"""
        pass

    @abstractmethod
    async def validate_request(self, request: AgentRequest) -> bool:
        """Validate if the agent can handle this request"""
        pass

    async def execute(self, request: AgentRequest) -> AgentResponse:
        """Execute agent with monitoring and error handling"""
        if not self.enabled:
            return AgentResponse(
                request_id=request.request_id,
                agent_name=self.name,
                status=AgentStatus.FAILED,
                result={},
                error="Agent is disabled"
            )

        if not await self.validate_request(request):
            return AgentResponse(
                request_id=request.request_id,
                agent_name=self.name,
                status=AgentStatus.FAILED,
                result={},
                error="Request validation failed"
            )

        start_time = time.time()
        self.status = AgentStatus.RUNNING

        try:
            self.logger.info(f"Starting processing request {request.request_id}")

            # Execute with timeout
            response = await self._execute_with_timeout(request)

            execution_time = time.time() - start_time
            response.execution_time = execution_time

            self.logger.info(f"Completed request {request.request_id} in {execution_time:.2f}s")
            self.status = AgentStatus.COMPLETED

            return response

        except TimeoutError:
            self.status = AgentStatus.TIMEOUT
            return AgentResponse(
                request_id=request.request_id,
                agent_name=self.name,
                status=AgentStatus.TIMEOUT,
                result={},
                error="Agent execution timeout",
                execution_time=time.time() - start_time
            )

        except Exception as e:
            self.status = AgentStatus.FAILED
            self.logger.error(f"Agent execution failed: {str(e)}", exc_info=True)
            return AgentResponse(
                request_id=request.request_id,
                agent_name=self.name,
                status=AgentStatus.FAILED,
                result={},
                error=str(e),
                execution_time=time.time() - start_time
            )

    async def _execute_with_timeout(self, request: AgentRequest) -> AgentResponse:
        """Execute with timeout handling"""
        import asyncio

        try:
            return await asyncio.wait_for(
                self.process(request),
                timeout=request.timeout
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"Agent {self.name} execution timeout after {request.timeout}s")
