"""
Query Executor Agent - Main agent implementation
Modernized version using BaseAgent with resource management
"""
import time
import logging
from typing import Dict, Any, Optional

from ..base_agent import BaseAgent, AgentRequest, AgentResponse, AgentStatus
from .models import (
    QueryExecutorRequest, QueryExecutorResult, EngineType, 
    QueryType, ExecutionMode, GeneratedQuery, QueryPlan
)
from .engines.clickhouse import ClickHouseEngine

logger = logging.getLogger(__name__)


class QueryExecutorAgent(BaseAgent):
    """
    Query Executor Agent
    
    Responsibilities:
    - Generate SQL queries from unified filter trees
    - Support multiple query engines (ClickHouse, Spark, etc.)
    - Validate and optimize queries
    - Execute queries (optional)
    """
    
    def __init__(self, name: str, config: Dict[str, Any], resource_manager):
        """Initialize with resource management"""
        super().__init__(name, config, resource_manager)
        
        # Initialize query engines
        self.engines = self._initialize_engines()
        
        # Default engine
        self.default_engine = EngineType(
            self.config.get('default_engine', 'clickhouse')
        )
        
        # Agent metadata handled by base class
        # self.agent_id, self.description, self.capabilities are in base class
    
    async def validate_request(self, request: AgentRequest) -> bool:
        """Validate the request has required context"""
        # Check for unified filter result in context
        if not request.context.get('unified_filter_result'):
            logger.warning("No unified filter result in request context")
            return False
        return True
    
    def _initialize_engines(self) -> Dict[EngineType, Any]:
        """Initialize available query engines"""
        engines = {}
        
        # Initialize ClickHouse engine
        if self.config.get('engines', {}).get('clickhouse', {}).get('enabled', True):
            clickhouse_config = self.config.get('engines', {}).get('clickhouse', {})
            engines[EngineType.CLICKHOUSE] = ClickHouseEngine(clickhouse_config)
            logger.info("Initialized ClickHouse query engine")
        
        # TODO: Initialize Spark engine when implemented
        # if self.config.get('engines', {}).get('spark', {}).get('enabled', False):
        #     spark_config = self.config.get('engines', {}).get('spark', {})
        #     engines[EngineType.SPARK] = SparkEngine(spark_config)
        
        return engines
    
    async def process_internal(self, request: AgentRequest) -> AgentResponse:
        """
        Core query execution logic
        All infrastructure concerns handled by BaseAgent
        """
        start_time = time.time()
        
        try:
            # Extract unified filter result from context
            unified_result = request.context.get('unified_filter_result')
            if not unified_result:
                return self._create_error_response(
                    request,
                    "No unified filter result found in context"
                )
            
            # Determine query type from unified result
            unified_tree = unified_result.get('unified_filter_tree', {})
            query_pattern = unified_tree.get('query_pattern', {})
            pattern_type = query_pattern.get('pattern_type', 'unknown')
            
            # Check if there are actual location contexts for movement analysis
            location_contexts = unified_tree.get('location_contexts', [])
            has_movement_locations = len(location_contexts) > 0
            
            # Map pattern type to query type
            # If pattern suggests location but no location contexts exist, treat as profile
            if pattern_type in ['profile_only', 'non_spatial']:
                query_type = QueryType.PROFILE_ONLY
            elif pattern_type in ['single_location', 'multi_location_union', 'multi_location_intersection']:
                # Check if this is actually a location-based query or just profile with city filters
                if has_movement_locations:
                    query_type = QueryType.LOCATION_BASED
                else:
                    # No actual location contexts, just profile filters with city fields
                    logger.info(f"Pattern type {pattern_type} but no location contexts - treating as profile query")
                    query_type = QueryType.PROFILE_ONLY
            elif pattern_type in ['location_sequence', 'co_location']:
                if has_movement_locations:
                    query_type = QueryType.MOVEMENT_PATTERN
                else:
                    logger.info(f"Pattern type {pattern_type} but no location contexts - treating as profile query")
                    query_type = QueryType.PROFILE_ONLY
            else:
                query_type = QueryType.PROFILE_ONLY  # Default
            
            # Get engine type from request or use default
            engine_type = EngineType(
                request.context.get('engine_type', self.default_engine.value)
            )
            
            # Check if engine is available
            if engine_type not in self.engines:
                return self._create_error_response(
                    request,
                    f"Query engine {engine_type.value} not available"
                )
            
            # Create query executor request
            executor_request = QueryExecutorRequest(
                unified_filter_tree=unified_result.get('unified_filter_tree', {}),
                engine_type=engine_type,
                query_type=query_type,
                execution_mode=ExecutionMode(
                    request.context.get('execution_mode', 'generate_only')
                ),
                select_fields=request.context.get('select_fields'),
                limit=request.context.get('limit', 1000),
                offset=request.context.get('offset', 0),
                order_by=request.context.get('order_by'),
                engine_options=request.context.get('engine_options', {}),
                session_id=request.context.get('session_id'),
                query_id=request.request_id,
                user_id=request.context.get('user_id')
            )
            
            # Get the appropriate engine
            engine = self.engines[engine_type]
            
            # Generate query
            generation_start = time.time()
            generated_query = engine.generate_query(executor_request)
            generation_time = (time.time() - generation_start) * 1000  # ms
            
            # Validate query
            is_valid, error_msg = engine.validate_query(generated_query)
            if not is_valid:
                return self._create_error_response(
                    request,
                    f"Query validation failed: {error_msg}"
                )
            
            # Estimate query plan
            query_plan = engine.estimate_query_plan(generated_query)
            
            # Create result
            result = QueryExecutorResult(
                generated_query=generated_query,
                query_plan=query_plan,
                execution_result=None,  # Not executing in generate_only mode
                success=True,
                generation_time_ms=generation_time,
                total_time_ms=(time.time() - start_time) * 1000,
                engine_type=engine_type,
                query_type=query_type,
                session_id=executor_request.session_id,
                query_id=executor_request.query_id
            )
            
            # Log the generated query
            logger.info(f"Generated {engine_type.value} query for {query_type.value}")
            logger.debug(f"Query: {generated_query.query}")
            
            return AgentResponse(
                request_id=request.request_id,
                agent_name=self.name,
                status=AgentStatus.COMPLETED,
                result=result.to_dict(),
                metadata={
                    "engine": engine_type.value,
                    "query_type": query_type.value,
                    "execution_mode": executor_request.execution_mode.value,
                    "generation_time_ms": generation_time
                },
                execution_time=(time.time() - start_time)
            )
            
        except Exception as e:
            logger.error(f"Error in query executor: {str(e)}", exc_info=True)
            return self._create_error_response(
                request,
                f"Query generation failed: {str(e)}"
            )
    
    def _create_error_response(
        self, 
        request: AgentRequest, 
        error_message: str
    ) -> AgentResponse:
        """Create error response"""
        return AgentResponse(
            request_id=request.request_id,
            agent_name=self.name,
            status=AgentStatus.FAILED,
            result={},  # Empty dict for error responses
            error=error_message,
            execution_time=0.0
        )