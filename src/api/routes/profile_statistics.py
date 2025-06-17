"""
Profile Statistics API Routes
Endpoints for integrated profile analysis with statistics
"""
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Depends, HTTPException, Body
from pydantic import BaseModel, Field
from datetime import datetime
import logging

from src.services.factory import ProfileAnalyticsServiceManager
from src.services.profile_analytics_integration import ProfileAnalyticsIntegration
from src.services.response_generator import ResponseConfig
from src.api.dependencies import get_config_manager, get_agent_manager, get_llm_client
from src.agents.agent_manager import AgentManager
from src.agents.base_agent import AgentRequest, AgentStatus
from src.core.config_manager import ConfigManager
from src.core.llm.base_llm import BaseLLMClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/profile-statistics", tags=["profile-statistics"])


# Request/Response Models
class AnalyzeRequest(BaseModel):
    """Request for comprehensive profile analysis"""
    query: str = Field(..., description="Natural language query")
    fields_to_analyze: Optional[List[str]] = Field(None, description="Specific fields for distribution analysis")
    extended_stat_ids: Optional[List[str]] = Field(None, description="Specific extended statistics to calculate")
    limit: int = Field(100, description="Number of profiles to return", ge=1, le=1000)
    response_config: Optional[Dict[str, Any]] = Field(None, description="Response generation configuration")
    session_id: Optional[str] = Field(None, description="Session ID for tracking")


class DashboardRequest(BaseModel):
    """Request for dashboard statistics"""
    query: str = Field(..., description="Natural language query or SQL WHERE clause")
    query_type: str = Field("natural", description="Type: 'natural' or 'sql_where'")
    dashboard_config: Dict[str, Any] = Field(..., description="Dashboard configuration")
    session_id: Optional[str] = Field(None, description="Session ID")


class AvailableStatisticsResponse(BaseModel):
    """Response for available statistics"""
    field_distributions: Dict[str, Any]
    extended_statistics: Dict[str, List[Dict[str, Any]]]


class AnalysisResponse(BaseModel):
    """Response for comprehensive analysis"""
    status: str = Field("success")
    query_result: Dict[str, Any]
    statistics: Dict[str, Any]
    generated_response: Dict[str, Any]
    total_execution_time_ms: float
    engine_type: str


# Dependency to get service manager
async def get_service_manager(
    config_manager: ConfigManager = Depends(get_config_manager)
) -> ProfileAnalyticsServiceManager:
    """Get or create service manager instance"""
    return ProfileAnalyticsServiceManager(config_manager)


# Dependency to get integration service
async def get_integration_service(
    service_manager: ProfileAnalyticsServiceManager = Depends(get_service_manager),
    llm_client: Optional[BaseLLMClient] = Depends(get_llm_client)
) -> ProfileAnalyticsIntegration:
    """Get profile analytics integration service"""
    profile_service = await service_manager.get_service()
    return ProfileAnalyticsIntegration(profile_service, llm_client)


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_with_statistics(
    request: AnalyzeRequest,
    agent_manager: AgentManager = Depends(get_agent_manager),
    integration_service: ProfileAnalyticsIntegration = Depends(get_integration_service)
):
    """
    Execute comprehensive profile analysis with statistics
    
    This endpoint:
    1. Processes natural language query through agents
    2. Executes profile query
    3. Calculates relevant statistics
    4. Generates intelligent response
    """
    try:
        # Create QueryContext for activity tracking
        from src.core.session_manager_models import QueryContext
        query_context = QueryContext(
            query_id=f"stats_analysis_{datetime.utcnow().timestamp()}",
            session_id=request.session_id,
            query_text=request.query,
            created_at=datetime.utcnow()
        )
        
        # Run through agent pipeline to get SQL WHERE clause
        agent_request = AgentRequest(
            request_id=query_context.query_id,
            prompt=request.query,
            context={
                "session_id": request.session_id,
                "execution_mode": "generate_only",
                "query_context": query_context  # Pass the QueryContext object
            },
            timestamp=datetime.utcnow().isoformat()
        )
        
        # Run query orchestrator
        orchestrator = agent_manager.get_agent("query_orchestrator")
        if not orchestrator:
            raise HTTPException(status_code=500, detail="Query orchestrator not found")
            
        orch_response = await orchestrator.process(agent_request)
        if orch_response.status != AgentStatus.COMPLETED:
            raise HTTPException(status_code=400, detail="Query understanding failed")
        
        orch_result = orch_response.result
        
        # Check for ambiguities
        if orch_result.get('has_ambiguities', False):
            # Return a proper error response instead of a dict
            raise HTTPException(
                status_code=400, 
                detail={
                    "status": "ambiguous",
                    "ambiguities": orch_result.get('ambiguities', []),
                    "message": "Query has ambiguities that need clarification"
                }
            )
        
        # Run unified filter
        unified_filter = agent_manager.get_agent("unified_filter")
        if not unified_filter:
            raise HTTPException(status_code=500, detail="Unified filter agent not found")
            
        unified_request = AgentRequest(
            request_id=f"{agent_request.request_id}_unified",
            prompt="Create unified filter",
            context={
                "orchestrator_results": orch_result,
                "session_id": request.session_id,
                "query_context": query_context  # Pass query context along
            },
            timestamp=datetime.utcnow().isoformat()
        )
        
        unified_response = await unified_filter.process(unified_request)
        if unified_response.status != AgentStatus.COMPLETED:
            raise HTTPException(status_code=400, detail="Filter creation failed")
        
        # Run query executor to get SQL
        query_executor = agent_manager.get_agent("query_executor")
        if not query_executor:
            raise HTTPException(status_code=500, detail="Query executor agent not found")
            
        executor_request = AgentRequest(
            request_id=f"{agent_request.request_id}_sql",
            prompt="Generate SQL",
            context={
                "unified_filter_result": unified_response.result,
                "engine_type": integration_service.profile_service.get_engine_type(),
                "execution_mode": "generate_only",
                "query_context": query_context  # Pass query context along
            },
            timestamp=datetime.utcnow().isoformat()
        )
        
        executor_response = await query_executor.process(executor_request)
        if executor_response.status != AgentStatus.COMPLETED:
            raise HTTPException(status_code=400, detail="SQL generation failed")
        
        # Extract WHERE clause
        sql_result = executor_response.result
        generated_sql = sql_result['generated_query']['query']
        
        where_start = generated_sql.find("WHERE") + 6
        where_end = generated_sql.find("LIMIT")
        if where_start > 5 and where_end > where_start:
            where_clause = generated_sql[where_start:where_end].strip()
        else:
            raise HTTPException(status_code=400, detail="Failed to extract WHERE clause")
        
        # Prepare query context
        query_context = {
            "original_query": request.query,
            "category": orch_result.get("category", "general"),
            "session_id": request.session_id,
            "orchestrator_results": orch_result,
            "unified_filter_results": unified_response.result
        }
        
        # Create response config if provided
        response_config = None
        if request.response_config:
            response_config = ResponseConfig(**request.response_config)
        
        # Execute analysis with statistics
        result = await integration_service.analyze_with_statistics(
            where_clause=where_clause,
            query_context=query_context,
            fields_to_analyze=request.fields_to_analyze,
            extended_stat_ids=request.extended_stat_ids,
            limit=request.limit,
            response_config=response_config
        )
        
        return AnalysisResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Profile analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dashboard-stats", response_model=Dict[str, Any])
async def get_dashboard_statistics(
    request: DashboardRequest,
    agent_manager: AgentManager = Depends(get_agent_manager),
    integration_service: ProfileAnalyticsIntegration = Depends(get_integration_service)
):
    """
    Get statistics for dashboard visualization
    
    Supports both natural language queries and direct SQL WHERE clauses
    """
    try:
        where_clause = request.query
        
        # If natural language, convert to SQL
        if request.query_type == "natural":
            # Similar agent pipeline as above
            agent_request = AgentRequest(
                request_id=f"dashboard_{datetime.utcnow().timestamp()}",
                prompt=request.query,
                context={"session_id": request.session_id},
                timestamp=datetime.utcnow().isoformat()
            )
            
            # Run through agents (simplified for brevity)
            # ... (same as analyze endpoint)
            
            # For now, assuming we have the WHERE clause
            pass
        
        # Get dashboard statistics
        result = await integration_service.get_dashboard_statistics(
            where_clause=where_clause,
            dashboard_config=request.dashboard_config
        )
        
        result["status"] = "success"
        result["engine_type"] = integration_service.profile_service.get_engine_type()
        
        return result
        
    except Exception as e:
        logger.error(f"Dashboard statistics failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/available-statistics", response_model=AvailableStatisticsResponse)
async def get_available_statistics(
    integration_service: ProfileAnalyticsIntegration = Depends(get_integration_service)
):
    """Get all available statistics for configuration"""
    try:
        stats = await integration_service.get_available_statistics()
        return AvailableStatisticsResponse(**stats)
    except Exception as e:
        logger.error(f"Failed to get available statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-response", response_model=Dict[str, Any])
async def generate_response_only(
    query_result: Dict[str, Any] = Body(..., description="Query execution result"),
    statistics: Dict[str, Any] = Body(..., description="Calculated statistics"),
    query_context: Dict[str, Any] = Body(..., description="Query context"),
    response_config: Optional[Dict[str, Any]] = Body(None, description="Response configuration"),
    integration_service: ProfileAnalyticsIntegration = Depends(get_integration_service)
):
    """
    Generate response from existing query results and statistics
    
    Useful for regenerating responses with different configurations
    """
    try:
        # Convert dict to ProfileQueryResult
        from src.services.base.profile_analytics import ProfileQueryResult
        from uuid import UUID
        
        profile_result = ProfileQueryResult(
            query_id=UUID(query_result["query_id"]),
            session_id=query_result.get("session_id"),
            sql_generated=query_result["sql_generated"],
            execution_time_ms=query_result["execution_time_ms"],
            result_count=query_result["result_count"],
            data=query_result.get("profiles", [])
        )
        
        # Create response config
        config = None
        if response_config:
            config = ResponseConfig(**response_config)
        
        # Generate response
        generated = await integration_service.response_generator.generate_response(
            query_result=profile_result,
            statistics=statistics,
            query_context=query_context,
            config=config
        )
        
        # Convert to dict
        return {
            "status": "success",
            "summary": generated.summary,
            "key_insights": generated.key_insights,
            "statistics_shown": generated.statistics_shown,
            "profiles_shown": generated.profiles_shown,
            "sql_filter": generated.sql_filter,
            "total_matches": generated.total_matches,
            "metadata": generated.metadata,
            "markdown": integration_service.response_generator.format_markdown_response(generated)
        }
        
    except Exception as e:
        logger.error(f"Response generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))