# src/agents/unified_filter/agent.py
"""
Unified Filter Agent - Combines filters from multiple agents into a unified structure
"""
import json
from typing import Dict, Any, Optional
from datetime import datetime

from ..base_agent import BaseAgent, AgentRequest, AgentResponse, AgentStatus
from ...core.logger import get_logger
from ...core.llm.base_llm import LLMClientFactory
from ...core.config_manager import ConfigManager
from ...core.training_logger import TrainingLogger
from ...core.activity_logger import ActivityLogger
from ...core.session_manager_models import QueryContext as QueryContextModel
from .models import UnifiedFilterTree, LocationContext, UnifiedFilterResult, FilterGenerationMethod, QueryPattern, QueryPatternType, SequenceConfig
from .prompt import UNIFIED_FILTER_TREE_PROMPT

logger = get_logger(__name__)


class UnifiedFilterAgent(BaseAgent):
    """
    Agent that unifies filters from multiple agents into a single filter tree structure
    """
    
    def __init__(self, name: str, config: dict, config_manager: ConfigManager, session_manager=None):
        """
        Initialize Unified Filter Agent
        
        Args:
            name: Agent name
            config: Agent configuration
            config_manager: Configuration manager instance
            session_manager: Session manager for training data
        """
        super().__init__(name, config)
        self.config_manager = config_manager
        self.session_manager = session_manager
        
        # Initialize LLM client
        llm_model = config.get('llm_model', 'openai')
        llm_config = config_manager.get_llm_config(llm_model)
        self.llm_client = LLMClientFactory.create_client(llm_config)
        
        # Agent properties
        self.agent_id = "unified_filter"
        self.description = "Combines filters from multiple agents into a unified structure"
        self.capabilities = ["filter_unification", "location_aware_filtering", "query_optimization"]
        
        # Enable training data logging
        self.enable_training_data = config.get('enable_training_data', True)
        
        # Initialize loggers
        self.training_logger = TrainingLogger(session_manager) if self.enable_training_data else None
        self.activity_logger = ActivityLogger(agent_name=self.name)
        
        # Cache for optimization
        self._cache = {}
        
        logger.info(f"UnifiedFilterAgent initialized")
    
    async def process(self, request: AgentRequest) -> AgentResponse:
        """
        Process the request to create a unified filter tree
        
        Args:
            request: Agent request containing orchestrator results
            
        Returns:
            AgentResponse with unified filter tree
        """
        start_time = datetime.now()
        
        try:
            # Set query context for activity logger
            query_context = request.context.get('query_context') if hasattr(request, 'context') and isinstance(
                request.context, dict) else None
            if query_context and isinstance(query_context, QueryContextModel):
                self.activity_logger.set_query_context(query_context)
            
            # Log activity
            self.activity_logger.action("Creating unified filter tree from orchestrator results")
            
            # Extract orchestrator results from context
            orchestrator_results = request.context.get('orchestrator_results', {})
            if not orchestrator_results:
                raise ValueError("No orchestrator results found in request context")
            
            # Create unified filter tree using LLM
            unified_tree, llm_response = await self._create_unified_filter_tree(orchestrator_results, request)
            
            # Log reasoning if available
            if unified_tree.reasoning:
                self.activity_logger.info(unified_tree.reasoning)
            
            # Create result object
            result = UnifiedFilterResult(
                unified_filter_tree=unified_tree,
                query_type=self._determine_query_type(unified_tree),
                estimated_complexity=self._estimate_complexity(unified_tree),
                generation_method=FilterGenerationMethod.UNIFIED_FILTERS,
                optimization_hints=self._generate_optimization_hints(unified_tree)
            )
            
            # Log what was identified
            self.activity_logger.identified(
                "unified filter structure",
                {
                    "location_count": len(unified_tree.location_contexts),
                    "has_city_filters": unified_tree.has_city_filters,
                    "has_facility_filters": unified_tree.has_facility_filters,
                    "complexity": result.estimated_complexity
                }
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResponse(
                request_id=request.request_id,
                agent_name=self.name,
                status=AgentStatus.COMPLETED,
                result=result.to_dict(),
                execution_time=execution_time,
                metadata={
                    "filter_generation_method": result.generation_method.value,
                    "location_count": len(unified_tree.location_contexts)
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to create unified filter tree: {e}")
            self.activity_logger.error("Failed to create unified filter tree", error=e)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResponse(
                request_id=request.request_id,
                agent_name=self.name,
                status=AgentStatus.FAILED,
                result={},
                error=str(e),
                execution_time=execution_time
            )
    
    async def _create_unified_filter_tree(self, orchestrator_results: Dict[str, Any], request: AgentRequest) -> tuple[UnifiedFilterTree, str]:
        """
        Create unified filter tree using LLM
        
        Args:
            orchestrator_results: Results from query orchestrator
            request: Original agent request for logging
            
        Returns:
            Tuple of (UnifiedFilterTree object, LLM response)
        """
        # Sanitize orchestrator results to remove geohashes
        sanitized_results = self._sanitize_orchestrator_results(orchestrator_results, request.request_id)
        
        # Prepare the prompt with sanitized results
        system_prompt = UNIFIED_FILTER_TREE_PROMPT
        user_prompt = f"""
Create a unified filter tree from these orchestrator results:

{json.dumps(sanitized_results, indent=2, default=str)}

Remember to:
1. Identify the query pattern (single_location, multi_location_union, etc.)
2. Each location context must have a complete_filter_tree with ALL conditions
3. Do NOT use global_filters - they create conflicts
4. Handle CITY/EMIRATE location types differently from FACILITY/ADDRESS types
5. Set radius_meters to null for CITY/EMIRATE types
6. For FACILITY/ADDRESS locations, use location_index to reference geohashes stored in query_location_geohashes table
"""
        
        # Record start time for training logger
        llm_start_time = datetime.now()
        
        # Call LLM
        self.activity_logger.action("Calling LLM to create unified filter structure")
        response = await self.llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
        
        # TODO: Remove this debugging code after fixing LLM response parsing issues
        # Temporary: Log the response for debugging empty response issues
        if self.config.get('debug_llm_responses', False):
            logger.info(f"LLM Response received, length: {len(response)}")
            
            # Save raw LLM response to file for debugging
            if request.context.get('session_id'):
                debug_file = f"unified_filter_llm_{request.context['session_id']}.txt"
                with open(debug_file, 'w') as f:
                    f.write(f"System Prompt:\n{system_prompt}\n\n")
                    f.write(f"User Prompt:\n{user_prompt}\n\n")
                    f.write(f"LLM Response:\n{response}")
                logger.info(f"Saved LLM response to {debug_file}")
        
        # Parse LLM response
        filter_tree_data = self._parse_llm_response(response)
        
        # Log to PostgreSQL in background if enabled
        if self.training_logger:
            prompts = {
                "system": system_prompt,
                "user": user_prompt
            }
            
            model_config = {
                'model': getattr(self.llm_client.config, 'model', 'unknown') if hasattr(self.llm_client, 'config') else 'unknown',
                'temperature': getattr(self.llm_client.config, 'temperature', 0.1) if hasattr(self.llm_client, 'config') else 0.1,
                'max_tokens': getattr(self.llm_client.config, 'max_tokens', 1000) if hasattr(self.llm_client, 'config') else 1000
            }
            
            self.training_logger.log_llm_interaction_background(
                session_id=request.context.get('session_id'),
                query_id=request.request_id,
                query_text=request.prompt,
                event_type="unified_filter_creation",
                llm_response=response,
                llm_start_time=llm_start_time,
                prompts=prompts,
                result=filter_tree_data,
                model_config=model_config,
                success=True
            )
        
        # Create UnifiedFilterTree from parsed data
        unified_tree = self._build_unified_filter_tree(filter_tree_data)
        
        return unified_tree, response
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response to extract filter tree structure
        
        Args:
            response: Raw LLM response
            
        Returns:
            Parsed filter tree data
        """
        try:
            # Log the raw response for debugging
            logger.debug(f"Raw LLM response length: {len(response)}")
            logger.debug(f"Raw LLM response first 500 chars: {response[:500]}")
            
            # Extract JSON from response
            # Handle potential markdown code blocks
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                json_str = response[start:end].strip()
            else:
                # Try to find JSON object directly
                start = response.find("{")
                if start == -1:
                    logger.error("No JSON object found in response")
                    logger.error(f"Full response: {response}")
                    raise ValueError("No JSON object found in LLM response")
                end = response.rfind("}") + 1
                json_str = response[start:end]
            
            logger.debug(f"Extracted JSON string length: {len(json_str)}")
            parsed_data = json.loads(json_str)
            logger.debug(f"Successfully parsed JSON with keys: {list(parsed_data.keys())}")
            
            return parsed_data
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            logger.error(f"JSON string that failed: {json_str[:500] if 'json_str' in locals() else 'Not extracted'}")
            raise ValueError(f"Invalid JSON in LLM response: {e}")
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.error(f"Raw response: {response}")
            raise ValueError(f"Invalid filter tree response from LLM: {e}")
    
    def _build_unified_filter_tree(self, data: Dict[str, Any]) -> UnifiedFilterTree:
        """
        Build UnifiedFilterTree object from parsed data
        
        Args:
            data: Parsed filter tree data
            
        Returns:
            UnifiedFilterTree object
        """
        # Extract query pattern
        pattern_data = data.get('query_pattern', {})
        if isinstance(pattern_data, str):
            # Handle case where query_pattern is just a string
            pattern_data = {'pattern_type': pattern_data}
        pattern_type_str = pattern_data.get('pattern_type', 'single_location')
        
        # Convert string to enum
        pattern_type_map = {
            'profile_only': QueryPatternType.NON_SPATIAL,  # Map profile_only to NON_SPATIAL enum
            'single_location': QueryPatternType.SINGLE_LOCATION,
            'multi_location_union': QueryPatternType.MULTI_LOCATION_UNION,
            'multi_location_intersection': QueryPatternType.MULTI_LOCATION_INTERSECTION,
            'location_sequence': QueryPatternType.LOCATION_SEQUENCE,
            'co_location': QueryPatternType.CO_LOCATION,
            'time_only': QueryPatternType.NON_SPATIAL  # Time-only queries are also non-spatial
        }
        pattern_type = pattern_type_map.get(pattern_type_str, QueryPatternType.SINGLE_LOCATION)
        
        # Handle sequence config if present
        sequence_config = None
        if 'sequence_config' in pattern_data:
            seq_data = pattern_data['sequence_config']
            sequence_config = SequenceConfig(
                window_hours=seq_data.get('window_hours', 24),
                order_matters=seq_data.get('order_matters', True),
                allow_intermediate_stops=seq_data.get('allow_intermediate_stops', False)
            )
        
        query_pattern = QueryPattern(
            pattern_type=pattern_type,
            combination_logic=pattern_data.get('combination_logic', ''),
            requires_movement_data=pattern_data.get('requires_movement_data', False),
            requires_profile_data=pattern_data.get('requires_profile_data', True),
            common_criteria=pattern_data.get('common_criteria'),
            sequence_config=sequence_config
        )
        
        # Extract unified filter tree (for non-spatial queries)
        unified_filter_tree = data.get('unified_filter_tree')
        
        # Extract location contexts (for spatial queries)
        location_contexts = []
        for loc_data in data.get('location_contexts', []):
            # Extract geohash reference if present
            geohash_ref = loc_data.get('geohash_reference')
            
            location_context = LocationContext(
                location_name=loc_data['location_name'],
                location_index=loc_data['location_index'],
                location_type=loc_data.get('location_type', 'FACILITY'),
                location_field=loc_data.get('location_field', 'geohash'),
                location_value=loc_data.get('location_value'),
                radius_meters=loc_data.get('radius_meters'),
                sequence_order=loc_data.get('sequence_order'),
                complete_filter_tree=loc_data.get('complete_filter_tree', {}),
                time_filters=loc_data.get('time_filters'),
                profile_filters=loc_data.get('profile_filters'),
                risk_filters=loc_data.get('risk_filters'),
                geohash_reference=geohash_ref  # Add geohash reference
            )
            location_contexts.append(location_context)
        
        # Extract metadata
        metadata = data.get('metadata', {})
        query_opt = data.get('query_optimization', {})
        
        # Determine if we have city/facility filters
        has_city = any(loc.location_type in ['CITY', 'EMIRATE'] for loc in location_contexts)
        has_facility = any(loc.location_type in ['FACILITY', 'ADDRESS'] for loc in location_contexts)
        
        # Create unified filter tree
        unified_tree = UnifiedFilterTree(
            query_pattern=query_pattern,
            unified_filter_tree=unified_filter_tree,  # For non-spatial queries
            location_contexts=location_contexts,
            has_time_filters=metadata.get('has_time_filters', False),
            has_location_filters=metadata.get('has_location_filters', False),
            has_profile_filters=metadata.get('has_profile_filters', False),
            has_risk_filters=metadata.get('has_risk_filters', False),
            has_city_filters=has_city,
            has_facility_filters=has_facility,
            uses_location_geohashes=metadata.get('requires_location_join', False),
            suggested_data_source=query_opt.get('suggested_data_source', ''),
            query_plan=query_opt,
            reasoning=data.get('reasoning', ''),
            confidence=data.get('confidence', 0.0),
            warnings=data.get('warnings', [])
        )
        
        return unified_tree
    
    def _determine_query_type(self, tree: UnifiedFilterTree) -> str:
        """
        Determine the query type based on filter tree
        
        Args:
            tree: UnifiedFilterTree object
            
        Returns:
            Query type string
        """
        # Use the query pattern type
        pattern_type = tree.query_pattern.pattern_type
        
        if pattern_type == QueryPatternType.SINGLE_LOCATION:
            if tree.has_city_filters:
                return "city_based"
            else:
                return "facility_based"
        elif pattern_type == QueryPatternType.MULTI_LOCATION_UNION:
            return "multi_location_union"
        elif pattern_type == QueryPatternType.MULTI_LOCATION_INTERSECTION:
            return "multi_location_intersection"
        elif pattern_type == QueryPatternType.LOCATION_SEQUENCE:
            return "location_sequence"
        elif pattern_type == QueryPatternType.CO_LOCATION:
            return "co_location"
        else:
            return "unknown"
    
    def _estimate_complexity(self, tree: UnifiedFilterTree) -> int:
        """
        Estimate query complexity (1-10 scale)
        
        Args:
            tree: UnifiedFilterTree object
            
        Returns:
            Complexity score
        """
        complexity = 1
        
        # Add complexity based on query pattern
        pattern_complexity = {
            QueryPatternType.SINGLE_LOCATION: 0,
            QueryPatternType.MULTI_LOCATION_UNION: 2,
            QueryPatternType.MULTI_LOCATION_INTERSECTION: 3,
            QueryPatternType.LOCATION_SEQUENCE: 4,
            QueryPatternType.CO_LOCATION: 4
        }
        complexity += pattern_complexity.get(tree.query_pattern.pattern_type, 0)
        
        # Add complexity for different filter types
        if tree.has_time_filters:
            complexity += 2
        if tree.has_location_filters:
            complexity += 2
        if tree.has_profile_filters:
            complexity += 1
        if tree.has_risk_filters:
            complexity += 1
        
        # Add complexity for multiple locations
        if len(tree.location_contexts) > 1:
            complexity += 1
        
        # Add complexity for mixed location types
        if tree.has_city_filters and tree.has_facility_filters:
            complexity += 2
        
        return min(complexity, 10)
    
    def _generate_optimization_hints(self, tree: UnifiedFilterTree) -> list:
        """
        Generate optimization hints based on filter tree
        
        Args:
            tree: UnifiedFilterTree object
            
        Returns:
            List of optimization hints
        """
        hints = []
        
        # Query pattern hints
        pattern_hints = {
            QueryPatternType.SINGLE_LOCATION: ["simple_filter"],
            QueryPatternType.MULTI_LOCATION_UNION: ["union_pattern", "parallel_execution"],
            QueryPatternType.MULTI_LOCATION_INTERSECTION: ["intersection_pattern", "multiple_joins"],
            QueryPatternType.LOCATION_SEQUENCE: ["sequence_pattern", "temporal_ordering"],
            QueryPatternType.CO_LOCATION: ["co_location_pattern", "time_window_join"]
        }
        hints.extend(pattern_hints.get(tree.query_pattern.pattern_type, []))
        
        # Location type hints
        if tree.has_city_filters and not tree.has_facility_filters:
            hints.append("profile_only_query")
            hints.append("use_city_indexes")
        elif tree.has_facility_filters:
            hints.append("requires_movement_data")
            hints.append("geohash_join_required")
        
        # Time hints
        if tree.has_time_filters:
            hints.append("partition_by_date")
            hints.append("time_based_filtering")
        
        # Multi-location hints
        if len(tree.location_contexts) > 1:
            hints.append("multi_location_processing")
            if tree.has_city_filters and tree.has_facility_filters:
                hints.append("mixed_location_types")
        
        # Data source hints
        if tree.suggested_data_source:
            hints.append(f"suggested_source_{tree.suggested_data_source}")
        
        return hints
    
    async def validate_request(self, request: AgentRequest) -> bool:
        """
        Validate if the request contains necessary orchestrator results
        
        Args:
            request: Agent request
            
        Returns:
            True if valid, False otherwise
        """
        orchestrator_results = request.context.get('orchestrator_results', {})
        
        # Check if we have the required filter data
        filters = orchestrator_results.get('filters', {})
        
        # At minimum, we need some filter data
        return bool(filters)
    
    def _sanitize_orchestrator_results(self, orchestrator_results: Dict[str, Any], query_id: str) -> Dict[str, Any]:
        """
        Remove unnecessary fields and geohashes from orchestrator results to avoid token limit issues.
        Only keep essential information for unified filter creation.
        
        Args:
            orchestrator_results: Raw orchestrator results
            query_id: Query ID for reference
            
        Returns:
            Minimal sanitized results
        """
        # Only keep essential fields
        sanitized = {
            'context_aware_query': orchestrator_results.get('context_aware_query', ''),
            'filters': {}
        }
        
        # Process each filter type
        filters = orchestrator_results.get('filters', {})
        
        # Process TIME filters - keep only essential fields
        if 'time' in filters and filters['time']:
            time_filter = filters['time']
            sanitized['filters']['time'] = {
                'date_ranges': time_filter.get('date_ranges', []),
                'hour_constraints': time_filter.get('hour_constraints', []),
                'day_constraints': time_filter.get('day_constraints', [])
            }
            # Remove unnecessary fields from date_ranges
            for dr in sanitized['filters']['time']['date_ranges']:
                dr.pop('original_text', None)
                dr.pop('confidence', None)
                dr.pop('expand_to_dates', None)
        
        # Process LOCATION filters - remove geohashes and unnecessary data
        if 'location' in filters and filters['location']:
            location_filter = filters['location']
            sanitized['filters']['location'] = {
                'locations': {}
            }
            
            # Process individual locations
            locations = location_filter.get('locations', {})
            for idx, (loc_key, loc_data) in enumerate(locations.items()):
                clean_loc = {
                    'name': loc_data.get('name'),
                    'type': loc_data.get('type'),
                    'field': loc_data.get('field'),
                    'value': loc_data.get('value')
                }
                
                # For FACILITY/ADDRESS types, add reference info
                if loc_data.get('type') in ['FACILITY', 'ADDRESS']:
                    clean_loc['radius_meters'] = loc_data.get('radius_meters', 500)
                    clean_loc['geohash_reference'] = {
                        'query_id': query_id,
                        'location_index': idx,
                        'table': 'telecom_db.query_location_geohashes'
                    }
                    if 'total_existing_geohashes' in loc_data:
                        clean_loc['geohash_count'] = len(loc_data.get('total_existing_geohashes', []))
                
                sanitized['filters']['location']['locations'][loc_key] = clean_loc
        
        # Process PROFILE filters - already minimal
        if 'profile' in filters and filters['profile']:
            profile_filter = filters['profile']
            sanitized['filters']['profile'] = {
                'filter_tree': profile_filter.get('filter_tree', {}),
                'exclusions': profile_filter.get('exclusions', {})
            }
        
        # Process RISK filters - already minimal
        if 'risk' in filters and filters['risk']:
            risk_filter = filters['risk']
            sanitized['filters']['risk'] = {
                'filter_tree': risk_filter.get('filter_tree', {}),
                'exclusions': risk_filter.get('exclusions', {})
            }
        
        self.activity_logger.action(
            f"Sanitized orchestrator results: reduced from {len(str(orchestrator_results))} to {len(str(sanitized))} chars"
        )
        
        return sanitized