# src/api/routes/agents.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uuid
import time

from src.api.dependencies import get_agent_manager
from src.agents.agent_manager import AgentManager
from src.agents.base_agent import AgentRequest

router = APIRouter(tags=["agents"])


class AnalysisRequest(BaseModel):
    prompt: str
    classification: str = "UNCLASS"
    context: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None




@router.get("/agents")
async def list_agents(agent_manager: AgentManager = Depends(get_agent_manager)):
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


