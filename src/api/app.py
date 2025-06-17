# src/api/app.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging
from src.core.logger import get_logger

from src.core.config_manager import ConfigManager
from src.agents.agent_manager import AgentManager
from src.agents.base_agent import AgentRequest

logger = get_logger(__name__)


class AnalysisRequest(BaseModel):
    prompt: str
    classification: str = "UNCLASS"
    context: Optional[Dict[str, Any]] = None


class AnalysisResponse(BaseModel):
    request_id: str
    results: Dict[str, Any]
    status: str
    execution_time: float


def create_app(config_manager: ConfigManager, agent_manager: AgentManager) -> FastAPI:
    """Create FastAPI application"""

    app = FastAPI(
        title="Intelligence Analytics Platform",
        description="AI-powered intelligence analytics with location extraction",
        version="1.0.0"
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "agents": agent_manager.list_agents(),
            "version": "1.0.0"
        }

    @app.post("/analyze", response_model=AnalysisResponse)
    async def analyze_prompt(request: AnalysisRequest):
        """Analyze prompt through agent pipeline"""

        try:
            results = await agent_manager.process_request(
                prompt=request.prompt,
                context=request.context or {},
                classification=request.classification
            )

            # Calculate total execution time
            total_time = sum(
                response.execution_time
                for response in results.values()
            )

            # Check if any agent failed
            status = "success"
            failed_agents = [
                name for name, response in results.items()
                if response.status.value in ["failed", "timeout"]
            ]

            if failed_agents:
                status = f"partial_failure: {', '.join(failed_agents)}"

            return AnalysisResponse(
                request_id=list(results.values())[0].request_id if results else "unknown",
                results={name: response.result for name, response in results.items()},
                status=status,
                execution_time=total_time
            )

        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/agents")
    async def list_agents():
        """List available agents"""
        agents = {}
        for name in agent_manager.list_agents():
            agent = agent_manager.get_agent(name)
            agents[name] = {
                "enabled": agent.enabled,
                "priority": agent.priority,
                "status": agent.status.value
            }
        return agents



    return app