# src/agents/unified_filter/agent.py
"""
Unified Filter Agent - Combines filters from multiple agents into a unified structure
Modernized version using BaseAgent with resource management
"""
import json
from datetime import datetime
from typing import Dict, Any, List

from .models import (
    UnifiedFilterTree, LocationContext, UnifiedFilterResult,
    FilterGenerationMethod, QueryPattern, QueryPatternType,
    SequenceConfig
)
from .prompt import UNIFIED_FILTER_TREE_PROMPT
from ..base_agent import BaseAgent, AgentResponse, AgentRequest
from ...core.logger import get_logger

logger = get_logger(__name__)


class UnifiedFilterAgent(BaseAgent):
    """
    Agent that unifies filters from multiple agents into a single filter tree structure
    """

    def __init__(self, name: str, config: dict, resource_manager):
        """Initialize with agent-specific components only"""
        super().__init__(name, config, resource_manager)
        
        # Agent metadata handled by base class
        # self.agent_id, self.description, self.capabilities are in base class

    async def validate_request(self, request: AgentRequest) -> bool:
        """Validate if the agent can handle this request"""
        # Check if we have orchestrator results to work with
        context = request.context or {}
        return 'orchestrator_results' in context

    async def process_internal(self, request: AgentRequest) -> AgentResponse:
        """
        Core filter unification logic
        All infrastructure concerns handled by BaseAgent
        """
        start_time = datetime.now()
        
        # Log activity
        self.activity_logger.action("Starting filter unification")
        
        # Extract context
        context = request.context or {}
        orchestrator_results = context.get('orchestrator_results', {})
        
        if not orchestrator_results:
            raise ValueError("No orchestrator results found in context")
        
        try:
            # Create unified filter tree
            unified_tree, llm_response = await self._create_unified_filter_tree(
                orchestrator_results, request
            )
            
            # Create result
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
            
            # Create response with metadata
            return self._create_success_response(
                request=request,
                result=result.to_dict(),
                start_time=start_time,
                filter_generation_method=result.generation_method.value,
                location_count=len(unified_tree.location_contexts)
            )
            
        except Exception as e:
            logger.error(f"Failed to create unified filter tree: {e}")
            self.activity_logger.error("Failed to create unified filter tree", error=e)
            raise

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
        
        # Call LLM
        self.activity_logger.action("Calling LLM to create unified filter structure")
        response = await self.call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            request=request,
            event_type="unified_filter_creation"
        )
        
        if not response:
            raise ValueError("No response from LLM")
        
        # Parse LLM response
        filter_tree_data = self._parse_llm_response(response)
        
        # Create UnifiedFilterTree from parsed data
        unified_tree = self._build_unified_filter_tree(filter_tree_data)
        
        return unified_tree, response
    
    def _sanitize_orchestrator_results(self, results: Dict[str, Any], query_id: str) -> Dict[str, Any]:
        """
        Sanitize orchestrator results to remove geohashes
        Replace them with location index references
        """
        sanitized = json.loads(json.dumps(results))  # Deep copy
        
        # Process location filters
        if 'location_filters' in sanitized and 'location_contexts' in sanitized['location_filters']:
            for idx, loc_context in enumerate(sanitized['location_filters']['location_contexts']):
                if 'geohashes' in loc_context:
                    # Replace geohashes with reference
                    loc_context['geohash_reference'] = {
                        'table': 'query_location_geohashes',
                        'query_id': query_id,
                        'location_index': idx,
                        'geohash_count': len(loc_context['geohashes'])
                    }
                    # Remove actual geohashes
                    del loc_context['geohashes']
        
        # Remove geohashes from top level location_filters
        if 'location_filters' in sanitized and 'geohashes' in sanitized['location_filters']:
            del sanitized['location_filters']['geohashes']
        
        return sanitized
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to extract filter tree structure"""
        try:
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
                    raise ValueError("No JSON object found in LLM response")
                end = response.rfind("}") + 1
                json_str = response[start:end]
            
            parsed_data = json.loads(json_str)
            return parsed_data
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            raise ValueError(f"Invalid JSON in LLM response: {e}")
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            raise ValueError(f"Invalid filter tree response from LLM: {e}")
    
    def _build_unified_filter_tree(self, data: Dict[str, Any]) -> UnifiedFilterTree:
        """Build UnifiedFilterTree object from parsed data"""
        # Extract query pattern
        pattern_data = data.get('query_pattern', {})
        if isinstance(pattern_data, str):
            # Handle case where query_pattern is just a string
            pattern_data = {'pattern_type': pattern_data}
        pattern_type_str = pattern_data.get('pattern_type', 'single_location')
        
        # Map string to enum
        pattern_type_mapping = {
            'non_spatial': QueryPatternType.NON_SPATIAL,
            'profile_only': QueryPatternType.NON_SPATIAL,  # Map profile_only to NON_SPATIAL enum
            'time_only': QueryPatternType.NON_SPATIAL,  # Time-only queries are also non-spatial
            'single_location': QueryPatternType.SINGLE_LOCATION,
            'multi_location_union': QueryPatternType.MULTI_LOCATION_UNION,
            'multi_location_intersection': QueryPatternType.MULTI_LOCATION_INTERSECTION,
            'location_sequence': QueryPatternType.LOCATION_SEQUENCE,
            'co_location': QueryPatternType.CO_LOCATION
        }
        pattern_type = pattern_type_mapping.get(pattern_type_str, QueryPatternType.SINGLE_LOCATION)
        
        # Build query pattern
        query_pattern = QueryPattern(
            pattern_type=pattern_type,
            combination_logic=pattern_data.get('combination_logic', 'OR') if isinstance(pattern_data, dict) else 'OR',
            requires_movement_data=pattern_data.get('requires_movement_data', False) if isinstance(pattern_data, dict) else False,
            requires_profile_data=pattern_data.get('requires_profile_data', True) if isinstance(pattern_data, dict) else True,
            common_criteria=pattern_data.get('common_criteria') if isinstance(pattern_data, dict) else None
        )
        
        # Handle sequence config if present (only if pattern_data is a dict)
        if isinstance(pattern_data, dict) and pattern_data.get('sequence_config'):
            seq_config = pattern_data['sequence_config']
            if isinstance(seq_config, dict):
                query_pattern.sequence_config = SequenceConfig(
                    window_hours=seq_config.get('window_hours', 24),
                    order_matters=seq_config.get('order_matters', True),
                    allow_intermediate_stops=seq_config.get('allow_intermediate_stops', False)
                )
        
        # Build location contexts
        location_contexts = []
        for loc_data in data.get('location_contexts', []):
            location_context = LocationContext(
                location_name=loc_data.get('location_name', ''),
                location_index=loc_data.get('location_index', 0),
                location_type=loc_data.get('location_type', 'CITY'),
                location_field=loc_data.get('location_field', 'visited_city'),
                location_value=loc_data.get('location_value'),
                radius_meters=loc_data.get('radius_meters'),
                sequence_order=loc_data.get('sequence_order'),
                geohash_reference=loc_data.get('geohash_reference'),
                complete_filter_tree=loc_data.get('complete_filter_tree', {}),
                time_filters=loc_data.get('time_filters'),
                profile_filters=loc_data.get('profile_filters'),
                risk_filters=loc_data.get('risk_filters')
            )
            location_contexts.append(location_context)
        
        # Create UnifiedFilterTree
        unified_tree = UnifiedFilterTree(
            query_pattern=query_pattern,
            unified_filter_tree=data.get('unified_filter_tree'),
            location_contexts=location_contexts,
            has_time_filters=data.get('metadata', {}).get('has_time_filters', False),
            has_location_filters=data.get('metadata', {}).get('has_location_filters', False),
            has_profile_filters=data.get('metadata', {}).get('has_profile_filters', False),
            has_risk_filters=data.get('metadata', {}).get('has_risk_filters', False),
            has_city_filters=data.get('metadata', {}).get('has_city_filters', False),
            has_facility_filters=data.get('metadata', {}).get('has_facility_filters', False),
            uses_location_geohashes=data.get('metadata', {}).get('uses_location_geohashes', False),
            suggested_data_source=data.get('query_optimization', {}).get('suggested_data_source', ''),
            query_plan=data.get('query_optimization', {}).get('query_plan', {}),
            reasoning=data.get('reasoning', ''),
            confidence=data.get('confidence', 0.0),
            warnings=data.get('warnings', [])
        )
        
        return unified_tree
    
    def _determine_query_type(self, unified_tree: UnifiedFilterTree) -> str:
        """Determine query type from unified tree"""
        if not unified_tree.has_location_filters:
            if unified_tree.has_risk_filters:
                return "risk_based"
            else:
                return "demographic"
        else:
            if len(unified_tree.location_contexts) > 1:
                return "complex_multi_location"
            else:
                return "location_based"
    
    def _estimate_complexity(self, unified_tree: UnifiedFilterTree) -> int:
        """Estimate query complexity on 1-10 scale"""
        complexity = 1
        
        # Add complexity for different filter types
        if unified_tree.has_profile_filters:
            complexity += 2
        if unified_tree.has_risk_filters:
            complexity += 2
        if unified_tree.has_time_filters:
            complexity += 1
        if unified_tree.has_location_filters:
            complexity += 2
        
        # Add complexity for multiple locations
        if len(unified_tree.location_contexts) > 1:
            complexity += len(unified_tree.location_contexts)
        
        # Cap at 10
        return min(complexity, 10)
    
    def _generate_optimization_hints(self, unified_tree: UnifiedFilterTree) -> List[str]:
        """Generate optimization hints for SQL generation"""
        hints = []
        
        # Add hints based on query pattern
        if unified_tree.query_pattern.pattern_type == QueryPatternType.NON_SPATIAL:
            hints.append("Use phone_imsi_uid_latest table directly")
        elif unified_tree.query_pattern.requires_movement_data:
            hints.append("Use movements table with proper time partitioning")
        
        # Add hints based on filters
        if unified_tree.has_time_filters:
            hints.append("Use time-based partitioning for better performance")
        
        if unified_tree.has_facility_filters:
            hints.append("Join with query_location_geohashes for facility filtering")
        
        return hints