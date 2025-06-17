# src/agents/profiler/agent.py
"""
Profiler Agent - Generates SQL for profile queries based on orchestrator results
Focuses on profile and risk filters only, using phone_imsi_uid_latest table
"""
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from ..base_agent import BaseAgent, AgentResponse, AgentStatus, AgentRequest
from ...core.activity_logger import ActivityLogger
from ...core.config_manager import ConfigManager
from ...core.logger import get_logger
from .models import ProfilerSQLGenerated, ProfilerResult, QueryType, SQLGenerationMethod, QueryStats
from .sql_builder import ProfilerSQLBuilder

logger = get_logger(__name__)


class ProfilerAgent(BaseAgent):
    """
    Generates SQL scripts for profile-based queries
    
    Responsibilities:
    - Receive orchestrator results with profile/risk filters
    - Generate optimized ClickHouse SQL
    - Store SQL with metadata for analytics
    - Track query patterns and statistics
    """
    
    def __init__(self, name: str, config: dict, config_manager: ConfigManager):
        """Initialize Profiler Agent"""
        super().__init__(name, config)
        self.config_manager = config_manager
        
        # Agent properties
        self.agent_id = "profiler"
        self.description = "Generates SQL for profile and risk-based queries"
        self.capabilities = [
            "sql_generation",
            "profile_queries",
            "risk_queries",
            "query_analytics"
        ]
        
        # Initialize components
        self.activity_logger = ActivityLogger(agent_name=self.name)
        self.sql_builder = ProfilerSQLBuilder()
        
        # Configuration
        self.default_limit = config.get('default_limit', 1000)
        self.max_limit = config.get('max_limit', 10000)
        
    async def process(self, request: AgentRequest) -> AgentResponse:
        """
        Process orchestrator results to generate SQL
        
        Expected request.context to contain full orchestrator result
        """
        start_time = datetime.now()
        
        try:
            self.activity_logger.action("Starting SQL generation for profile query")
            
            # Extract orchestrator result from context
            orchestrator_result = request.context.get('orchestrator_result', {})
            if not orchestrator_result:
                raise ValueError("No orchestrator result found in request context")
            
            # Extract key information
            session_id = request.context.get('session_id', '')
            query_id = request.request_id
            
            # Parse orchestrator result
            parsed_data = self._parse_orchestrator_result(orchestrator_result)
            
            # Determine query type
            query_type = self._determine_query_type(parsed_data['filters'])
            self.activity_logger.info(f"Determined query type: {query_type.value}")
            
            # Set filter flags
            filter_flags = self._set_filter_flags(parsed_data['filters'])
            
            # Generate SQL
            sql_script, generation_method, warnings = self.sql_builder.build_sql_script(
                filters=parsed_data['filters'],
                query_type=query_type,
                context_aware_query=parsed_data['context_aware_query'],
                original_query=parsed_data['original_query']
            )
            
            self.activity_logger.action("SQL script generated successfully")
            
            # Create SQL metadata object
            sql_generated = ProfilerSQLGenerated(
                session_id=session_id,
                query_id=query_id,
                original_query=parsed_data['original_query'],
                context_aware_query=parsed_data['context_aware_query'],
                classification=parsed_data['classification'],
                domains=parsed_data['domains'],
                agents_required=parsed_data['agents_required'],
                query_type=query_type,
                entities_detected=parsed_data.get('entities_detected', {}),
                filters=parsed_data['filters'],
                sql_generated_script=sql_script,
                sql_generation_method=generation_method,
                validation_warnings=warnings,
                ambiguities=parsed_data.get('ambiguities', []),
                **filter_flags
            )
            
            # Calculate basic stats (without executing)
            stats = self._calculate_query_stats(parsed_data['filters'], query_type)
            sql_generated.stats = stats
            
            # Create result
            result = ProfilerResult(
                status="completed",
                sql_generated=sql_generated,
                execution_time_ms=int((datetime.now() - start_time).total_seconds() * 1000)
            )
            
            self.activity_logger.info(f"Profiler completed in {result.execution_time_ms}ms")
            
            return AgentResponse(
                request_id=request.request_id,
                agent_name=self.name,
                status=AgentStatus.COMPLETED,
                result=result.to_dict(),
                execution_time=result.execution_time_ms / 1000,
                metadata={
                    "query_type": query_type.value,
                    "filter_count": stats.filter_count if stats else 0,
                    "sql_length": len(sql_script)
                }
            )
            
        except Exception as e:
            logger.error(f"Profiler agent failed: {e}")
            self.activity_logger.error("SQL generation failed", error=e)
            
            result = ProfilerResult(
                status="failed",
                error=str(e),
                execution_time_ms=int((datetime.now() - start_time).total_seconds() * 1000)
            )
            
            return AgentResponse(
                request_id=request.request_id,
                agent_name=self.name,
                status=AgentStatus.FAILED,
                result=result.to_dict(),
                error=str(e),
                execution_time=result.execution_time_ms / 1000
            )
    
    def _parse_orchestrator_result(self, orchestrator_result: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and extract relevant data from orchestrator result"""
        # Extract filters - focusing only on profile and risk
        filters = orchestrator_result.get('filters', {})
        profile_filters = filters.get('profile', {})
        risk_filters = filters.get('risk', {})
        
        return {
            'original_query': orchestrator_result.get('context_aware_query', ''),  # Using context_aware as original for now
            'context_aware_query': orchestrator_result.get('context_aware_query', ''),
            'classification': orchestrator_result.get('classification', {}),
            'domains': orchestrator_result.get('domains', []),
            'agents_required': orchestrator_result.get('agents_required', []),
            'filters': {
                'profile': profile_filters,
                'risk': risk_filters
            },
            'ambiguities': orchestrator_result.get('ambiguities', []),
            'entities_detected': {}  # Not available from orchestrator (fire-and-forget)
        }
    
    def _determine_query_type(self, filters: Dict[str, Any]) -> QueryType:
        """Determine the type of query based on filters"""
        profile_filters = filters.get('profile', {})
        risk_filters = filters.get('risk', {})
        
        # Check if filters exist (new format uses filter_tree)
        has_profile = bool(profile_filters.get('filter_tree')) or bool(profile_filters.get('exclusions'))
        has_risk = bool(risk_filters.get('filter_tree')) or bool(risk_filters.get('exclusions'))
        
        # Check for specific risk indicators in filter_tree
        has_crime = False
        if risk_filters.get('filter_tree'):
            # Check if crime-related fields are mentioned
            filter_str = str(risk_filters['filter_tree'])
            has_crime = ('has_crime_case' in filter_str or 
                        'is_in_prison' in filter_str or 
                        'crime_categories' in filter_str)
        
        if has_crime:
            return QueryType.CRIME_INVESTIGATION
        elif has_profile and has_risk:
            return QueryType.COMPLEX_MULTI_FILTER
        elif has_risk:
            return QueryType.RISK_BASED
        elif has_profile:
            # Check profile details for more specific types
            if profile_filters.get('filter_tree'):
                filter_str = str(profile_filters['filter_tree'])
                if 'nationality_code' in filter_str or 'age_group' in filter_str:
                    return QueryType.DEMOGRAPHIC
                else:
                    return QueryType.DETAIL_RECORDS
            else:
                return QueryType.DETAIL_RECORDS
        else:
            return QueryType.DETAIL_RECORDS
    
    def _set_filter_flags(self, filters: Dict[str, Any]) -> Dict[str, bool]:
        """Set boolean flags based on filters present"""
        profile_filters = filters.get('profile', {})
        risk_filters = filters.get('risk', {})
        
        # Check profile filter details (new format uses filter_tree)
        has_profile = bool(profile_filters.get('filter_tree')) or bool(profile_filters.get('exclusions'))
        
        # Check risk filter details
        has_risk = bool(risk_filters.get('filter_tree')) or bool(risk_filters.get('exclusions'))
        
        # Check for crime-related filters in filter_tree
        has_crime = False
        if risk_filters.get('filter_tree'):
            filter_str = str(risk_filters['filter_tree'])
            has_crime = ('has_crime_case' in filter_str or 
                        'is_in_prison' in filter_str or
                        'crime_categories' in filter_str)
        
        # Check for applications filter in profile
        has_application = False
        if profile_filters.get('filter_tree'):
            filter_str = str(profile_filters['filter_tree'])
            has_application = 'applications_used' in filter_str
        
        return {
            'has_time_filter': False,  # Profiler doesn't handle time
            'has_location_filter': False,  # Profiler doesn't handle location
            'has_profile_filter': has_profile,
            'has_risk_filter': has_risk,
            'has_movement_filter': False,  # Not applicable
            'has_crime_filter': has_crime,
            'has_application_filter': has_application,
            'is_aggregate_query': False,  # Will be set based on query type
            'is_detail_query': True  # Most profile queries return details
        }
    
    def _calculate_query_stats(self, filters: Dict[str, Any], query_type: QueryType) -> QueryStats:
        """Calculate statistics about the query"""
        stats = QueryStats()
        
        # Count filters (new format uses filter_tree)
        profile_filters = filters.get('profile', {})
        risk_filters = filters.get('risk', {})
        
        # Count conditions in filter trees
        profile_count = self._count_conditions_in_tree(profile_filters.get('filter_tree', {}))
        profile_count += self._count_conditions_in_tree(profile_filters.get('exclusions', {}))
        
        risk_count = self._count_conditions_in_tree(risk_filters.get('filter_tree', {}))
        risk_count += self._count_conditions_in_tree(risk_filters.get('exclusions', {}))
        
        stats.filter_count = profile_count + risk_count
        stats.filter_complexity_score = self._calculate_complexity_score(profile_count, risk_count)
        
        # No joins needed for profiler queries (single table)
        stats.join_count = 0
        
        # Index usage hints
        stats.index_usage = []
        if 'nationality_code' in profile_filters.get('inclusions', {}):
            stats.index_usage.append('idx_nationality')
        if 'risk_score' in risk_filters.get('inclusions', {}):
            stats.index_usage.append('idx_risk_score')
        
        return stats
    
    def _count_conditions_in_tree(self, tree: Dict[str, Any]) -> int:
        """Count the number of conditions in a filter tree"""
        if not tree:
            return 0
            
        count = 0
        
        if 'AND' in tree:
            for condition in tree['AND']:
                if 'AND' in condition or 'OR' in condition:
                    count += self._count_conditions_in_tree(condition)
                else:
                    count += 1
        elif 'OR' in tree:
            for condition in tree['OR']:
                if 'AND' in condition or 'OR' in condition:
                    count += self._count_conditions_in_tree(condition)
                else:
                    count += 1
        elif 'field' in tree:
            # Single condition at root
            count = 1
            
        return count
    
    def _calculate_complexity_score(self, profile_count: int, risk_count: int) -> int:
        """Calculate a complexity score for the query"""
        # Simple scoring: 1 point per filter, 2 points for risk filters
        return profile_count + (risk_count * 2)
    
    async def validate_request(self, request: AgentRequest) -> bool:
        """Validate if this agent can handle the request"""
        # Profiler requires orchestrator result in context
        if not request.context or 'orchestrator_result' not in request.context:
            return False
        
        orchestrator_result = request.context['orchestrator_result']
        
        # Check if it's a profile-related query
        classification = orchestrator_result.get('classification', {})
        category = classification.get('category', '')
        
        # Profiler handles these categories
        valid_categories = [
            'profile_search',
            'risk_assessment',
            'demographic_analysis',
            'crime_investigation'
        ]
        
        return category in valid_categories