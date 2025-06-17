"""
Profile Analytics API Routes
Endpoints for profile query execution and analytics
"""
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Depends, HTTPException, Query, Body
from pydantic import BaseModel, Field
from uuid import UUID
from datetime import datetime
import logging

from src.services.base.profile_analytics import BaseProfileAnalyticsService
from src.services.factory import ProfileAnalyticsServiceManager
from src.api.dependencies import get_config_manager, get_agent_manager
from src.agents.agent_manager import AgentManager
from src.agents.base_agent import AgentRequest, AgentStatus
from src.core.config_manager import ConfigManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/profile-analytics", tags=["profile-analytics"])


# Request/Response Models
class ProfileQueryRequest(BaseModel):
    """Request for profile query execution"""
    query: str = Field(..., description="Natural language query or SQL WHERE clause")
    query_type: str = Field("natural", description="Type: 'natural' or 'sql_where'")
    select_fields: Optional[List[str]] = Field(None, description="Fields to select")
    limit: int = Field(100, description="Result limit", ge=1, le=10000)
    offset: int = Field(0, description="Result offset", ge=0)
    order_by: Optional[List[Dict[str, str]]] = Field(None, description="Order by fields")
    session_id: Optional[str] = Field(None, description="Session ID for tracking")
    execute: bool = Field(True, description="Execute query or just generate SQL")
    engine_type: Optional[str] = Field(None, description="Query engine type (default: from config)")


class FieldDistributionRequest(BaseModel):
    """Request for field distribution analysis"""
    where_clause: str = Field(..., description="SQL WHERE clause")
    field_name: str = Field(..., description="Field to analyze")
    top_n: int = Field(20, description="Top N values to return", ge=1, le=100)
    include_others: bool = Field(True, description="Include 'Others' category")
    engine_type: Optional[str] = Field(None, description="Query engine type")


class CrossTabRequest(BaseModel):
    """Request for cross-tabulation analysis"""
    where_clause: str = Field(..., description="SQL WHERE clause")
    field1: str = Field(..., description="First field")
    field2: str = Field(..., description="Second field")
    limit: int = Field(100, description="Limit combinations", ge=1, le=1000)
    engine_type: Optional[str] = Field(None, description="Query engine type")


class BaseProfileResponse(BaseModel):
    """Base response model with consistent fields"""
    status: str = Field("success", description="Response status")
    execution_time_ms: float = Field(..., description="Execution time in milliseconds")
    engine_type: str = Field(..., description="Query engine type")


class ProfileDetailsResponse(BaseProfileResponse):
    """Response for profile details"""
    total_count: int
    page_count: int
    limit: int
    offset: int
    data: List[Dict[str, Any]]


class UniqueCountsResponse(BaseProfileResponse):
    """Response for unique counts"""
    counts: Dict[str, int]


class FieldStatisticsResponse(BaseProfileResponse):
    """Response for field statistics"""
    field_name: str
    total_count: int
    unique_count: Optional[int]
    null_count: int = 0
    distribution: Dict[str, int]
    percentiles: Optional[Dict[float, float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DemographicsResponse(BaseProfileResponse):
    """Response for demographics statistics"""
    demographics: Dict[str, Dict[str, Any]]
    total_profiles: int


class RiskStatisticsResponse(BaseProfileResponse):
    """Response for risk statistics"""
    risk_metrics: Dict[str, Dict[str, Any]]
    total_profiles: int


class CrossTabResponse(BaseProfileResponse):
    """Response for cross-tabulation"""
    field1: str
    field2: str
    data: Dict[str, Dict[str, int]]
    total_combinations: int


# Dependency to get service manager instance
_service_manager = None

async def get_service_manager(
    config_manager: ConfigManager = Depends(get_config_manager)
) -> ProfileAnalyticsServiceManager:
    """Get or create service manager instance"""
    global _service_manager
    if not _service_manager:
        _service_manager = ProfileAnalyticsServiceManager(config_manager)
    return _service_manager


async def get_profile_service(
    engine_type: Optional[str] = None,
    service_manager: ProfileAnalyticsServiceManager = Depends(get_service_manager)
) -> BaseProfileAnalyticsService:
    """Get profile analytics service for specified engine"""
    return await service_manager.get_service(engine_type)


@router.post("/query", response_model=Dict[str, Any])
async def execute_profile_query(
    request: ProfileQueryRequest,
    agent_manager: AgentManager = Depends(get_agent_manager),
    service_manager: ProfileAnalyticsServiceManager = Depends(get_service_manager)
):
    """
    Execute a profile query
    
    Supports both natural language queries and direct SQL WHERE clauses
    """
    try:
        # Get appropriate service
        service = await service_manager.get_service(request.engine_type)
        
        where_clause = request.query
        
        # If natural language, convert to SQL using agents
        if request.query_type == "natural":
            # Run through the agent pipeline
            agent_request = AgentRequest(
                request_id=f"api_profile_{datetime.utcnow().timestamp()}",
                prompt=request.query,
                context={
                    "session_id": request.session_id,
                    "execution_mode": "generate_only",
                    "limit": request.limit
                },
                timestamp=datetime.utcnow().isoformat()
            )
            
            # Run query orchestrator
            orchestrator = agent_manager.get_agent("query_orchestrator")
            if not orchestrator:
                raise HTTPException(status_code=500, detail="Query orchestrator agent not found")
            orch_response = await orchestrator.process(agent_request)
            
            if orch_response.status != AgentStatus.COMPLETED:
                raise HTTPException(status_code=400, detail="Query understanding failed")
            
            # Check for ambiguities
            orch_result = orch_response.result
            if orch_result.get('has_ambiguities', False):
                return {
                    "status": "ambiguous",
                    "ambiguities": orch_result.get('ambiguities', []),
                    "message": "Query has ambiguities that need clarification"
                }
            
            # Run unified filter
            unified_filter = agent_manager.get_agent("unified_filter")
            if not unified_filter:
                raise HTTPException(status_code=500, detail="Unified filter agent not found")
            unified_request = AgentRequest(
                request_id=f"{agent_request.request_id}_unified",
                prompt="Create unified filter",
                context={
                    "orchestrator_results": orch_result,
                    "session_id": request.session_id
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
                    "engine_type": service.get_engine_type(),
                    "execution_mode": "generate_only",
                    "limit": request.limit,
                    "session_id": request.session_id
                },
                timestamp=datetime.utcnow().isoformat()
            )
            
            executor_response = await query_executor.process(executor_request)
            
            if executor_response.status != AgentStatus.COMPLETED:
                raise HTTPException(status_code=400, detail="SQL generation failed")
            
            # Extract WHERE clause from generated SQL
            sql_result = executor_response.result
            generated_sql = sql_result['generated_query']['query']
            
            # Parse WHERE clause from SQL
            where_start = generated_sql.find("WHERE") + 6
            where_end = generated_sql.find("LIMIT")
            if where_start > 5 and where_end > where_start:
                where_clause = generated_sql[where_start:where_end].strip()
            else:
                raise HTTPException(status_code=400, detail="Failed to extract WHERE clause")
        
        # Execute the query if requested
        if request.execute:
            result = await service.execute_profile_query(
                where_clause=where_clause,
                select_fields=request.select_fields,
                limit=request.limit,
                offset=request.offset,
                order_by=request.order_by,
                session_id=request.session_id
            )
            
            return {
                "status": "success",
                "query_id": str(result.query_id),
                "sql_generated": result.sql_generated,
                "result_count": result.result_count,
                "data": result.data,
                "execution_time_ms": result.execution_time_ms,
                "engine_type": service.get_engine_type()
            }
        else:
            # Just return the generated SQL
            return {
                "status": "generated",
                "where_clause": where_clause,
                "engine_type": service.get_engine_type(),
                "message": "SQL generated but not executed"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Profile query execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/details", response_model=ProfileDetailsResponse)
async def get_profile_details(
    where_clause: str = Body(..., description="SQL WHERE clause"),
    fields: Optional[List[str]] = Body(None, description="Fields to return"),
    limit: int = Body(100, ge=1, le=1000),
    offset: int = Body(0, ge=0),
    engine_type: Optional[str] = Body(None, description="Query engine type"),
    service_manager: ProfileAnalyticsServiceManager = Depends(get_service_manager)
):
    """Get detailed profile information with pagination"""
    try:
        service = await service_manager.get_service(engine_type)
        result = await service.get_profile_details(
            where_clause=where_clause,
            fields=fields,
            limit=limit,
            offset=offset
        )
        result["engine_type"] = service.get_engine_type()
        result["status"] = "success"
        return ProfileDetailsResponse(**result)
    except Exception as e:
        logger.error(f"Failed to get profile details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/unique-counts", response_model=UniqueCountsResponse)
async def get_unique_counts(
    where_clause: str = Body(..., description="SQL WHERE clause"),
    identifier_fields: Optional[List[str]] = Body(None, description="Fields to count"),
    engine_type: Optional[str] = Body(None, description="Query engine type"),
    service_manager: ProfileAnalyticsServiceManager = Depends(get_service_manager)
):
    """Get unique counts for identifier fields"""
    try:
        service = await service_manager.get_service(engine_type)
        start_time = datetime.utcnow()
        counts = await service.get_unique_counts(
            where_clause=where_clause,
            identifier_fields=identifier_fields
        )
        execution_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return UniqueCountsResponse(
            status="success",
            counts=counts,
            execution_time_ms=execution_time_ms,
            engine_type=service.get_engine_type()
        )
    except Exception as e:
        logger.error(f"Failed to get unique counts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/field-distribution", response_model=FieldStatisticsResponse)
async def get_field_distribution(
    request: FieldDistributionRequest,
    service_manager: ProfileAnalyticsServiceManager = Depends(get_service_manager)
):
    """Get distribution of values for a specific field"""
    try:
        service = await service_manager.get_service(request.engine_type)
        start_time = datetime.utcnow()
        stats = await service.get_field_distribution(
            where_clause=request.where_clause,
            field_name=request.field_name,
            top_n=request.top_n,
            include_others=request.include_others
        )
        
        execution_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return FieldStatisticsResponse(
            status="success",
            field_name=stats.field_name,
            total_count=stats.total_count,
            unique_count=stats.unique_count,
            null_count=stats.null_count,
            distribution=stats.distribution,
            percentiles=stats.percentiles,
            metadata=stats.metadata,
            execution_time_ms=execution_time_ms,
            engine_type=service.get_engine_type()
        )
    except Exception as e:
        logger.error(f"Failed to get field distribution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/demographics", response_model=DemographicsResponse)
async def get_demographic_statistics(
    where_clause: str = Body(..., description="SQL WHERE clause"),
    engine_type: Optional[str] = Body(None, description="Query engine type"),
    service_manager: ProfileAnalyticsServiceManager = Depends(get_service_manager)
):
    """Get comprehensive demographic statistics"""
    try:
        service = await service_manager.get_service(engine_type)
        start_time = datetime.utcnow()
        stats = await service.get_demographic_statistics(where_clause)
        
        # Convert to dict format for response
        demographics_data = {}
        total_profiles = 0
        
        for field, field_stats in stats.items():
            demographics_data[field] = {
                "field_name": field_stats.field_name,
                "total_count": field_stats.total_count,
                "unique_count": field_stats.unique_count,
                "distribution": field_stats.distribution,
                "metadata": field_stats.metadata
            }
            if total_profiles == 0:
                total_profiles = field_stats.total_count
        
        execution_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return DemographicsResponse(
            status="success",
            demographics=demographics_data,
            total_profiles=total_profiles,
            execution_time_ms=execution_time_ms,
            engine_type=service.get_engine_type()
        )
    except Exception as e:
        logger.error(f"Failed to get demographic statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/risk-statistics", response_model=RiskStatisticsResponse)
async def get_risk_statistics(
    where_clause: str = Body(..., description="SQL WHERE clause"),
    engine_type: Optional[str] = Body(None, description="Query engine type"),
    service_manager: ProfileAnalyticsServiceManager = Depends(get_service_manager)
):
    """Get risk score and crime statistics"""
    try:
        service = await service_manager.get_service(engine_type)
        start_time = datetime.utcnow()
        stats = await service.get_risk_statistics(where_clause)
        
        # Convert to dict format for response
        risk_metrics = {}
        total_profiles = 0
        
        for field, field_stats in stats.items():
            risk_metrics[field] = {
                "field_name": field_stats.field_name,
                "total_count": field_stats.total_count,
                "unique_count": field_stats.unique_count,
                "distribution": field_stats.distribution,
                "percentiles": field_stats.percentiles,
                "metadata": field_stats.metadata
            }
            if total_profiles == 0:
                total_profiles = field_stats.total_count
        
        execution_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return RiskStatisticsResponse(
            status="success",
            risk_metrics=risk_metrics,
            total_profiles=total_profiles,
            execution_time_ms=execution_time_ms,
            engine_type=service.get_engine_type()
        )
    except Exception as e:
        logger.error(f"Failed to get risk statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/communication-stats", response_model=Dict[str, FieldStatisticsResponse])
async def get_communication_statistics(
    where_clause: str = Body(..., description="SQL WHERE clause"),
    engine_type: Optional[str] = Body(None, description="Query engine type"),
    service_manager: ProfileAnalyticsServiceManager = Depends(get_service_manager)
):
    """Get communication pattern statistics"""
    try:
        service = await service_manager.get_service(engine_type)
        stats = await service.get_communication_statistics(where_clause)
        
        return {
            field: FieldStatisticsResponse(
                field_name=field_stats.field_name,
                total_count=field_stats.total_count,
                unique_count=field_stats.unique_count,
                null_count=field_stats.null_count,
                distribution=field_stats.distribution,
                percentiles=field_stats.percentiles,
                metadata=field_stats.metadata
            )
            for field, field_stats in stats.items()
        }
    except Exception as e:
        logger.error(f"Failed to get communication statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/travel-stats", response_model=Dict[str, FieldStatisticsResponse])
async def get_travel_statistics(
    where_clause: str = Body(..., description="SQL WHERE clause"),
    engine_type: Optional[str] = Body(None, description="Query engine type"),
    service_manager: ProfileAnalyticsServiceManager = Depends(get_service_manager)
):
    """Get travel pattern statistics"""
    try:
        service = await service_manager.get_service(engine_type)
        stats = await service.get_travel_statistics(where_clause)
        
        return {
            field: FieldStatisticsResponse(
                field_name=field_stats.field_name,
                total_count=field_stats.total_count,
                unique_count=field_stats.unique_count,
                null_count=field_stats.null_count,
                distribution=field_stats.distribution,
                percentiles=field_stats.percentiles,
                metadata=field_stats.metadata
            )
            for field, field_stats in stats.items()
        }
    except Exception as e:
        logger.error(f"Failed to get travel statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cross-tabulation", response_model=CrossTabResponse)
async def get_cross_tabulation(
    request: CrossTabRequest,
    service_manager: ProfileAnalyticsServiceManager = Depends(get_service_manager)
):
    """Get cross-tabulation between two fields"""
    try:
        service = await service_manager.get_service(request.engine_type)
        start_time = datetime.utcnow()
        result = await service.get_cross_tabulation(
            where_clause=request.where_clause,
            field1=request.field1,
            field2=request.field2,
            limit=request.limit
        )
        execution_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return CrossTabResponse(
            status="success",
            field1=result["field1"],
            field2=result["field2"],
            data=result["data"],
            total_combinations=result["total_combinations"],
            execution_time_ms=execution_time_ms,
            engine_type=service.get_engine_type()
        )
    except Exception as e:
        logger.error(f"Failed to get cross-tabulation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/supported-fields", response_model=Dict[str, Any])
async def get_supported_fields(
    engine_type: Optional[str] = Query(None, description="Query engine type"),
    service_manager: ProfileAnalyticsServiceManager = Depends(get_service_manager)
):
    """Get list of supported fields by category"""
    try:
        service = await service_manager.get_service(engine_type)
        return {
            "status": "success",
            "engine_type": service.get_engine_type(),
            "fields": service.get_supported_fields()
        }
    except Exception as e:
        logger.error(f"Failed to get supported fields: {e}")
        raise HTTPException(status_code=500, detail=str(e))