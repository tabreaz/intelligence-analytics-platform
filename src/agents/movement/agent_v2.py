# src/agents/movement/agent_v2.py
"""
Movement Analysis Agent V2 - Two-step approach with intent classification
"""
import json
from datetime import datetime
import pytz
from typing import Optional, Dict, Any

from .constants import MAX_RETRIES
from .parsers import SingleLocationParser
from .prompt import MOVEMENT_INTENT_CLASSIFIER, MOVEMENT_PROMPT
from .response_parser import MovementResponseParser
from .templates.co_presence_prompt import CO_PRESENCE_PROMPT
from .templates.movement_pattern_prompt import MOVEMENT_PATTERN_PROMPT
from .utils import parse_json_response
from ..base_agent import BaseAgent, AgentResponse, AgentRequest
from ...core.logger import get_logger
from ...core.location_resolver import LocationResolver
from ...core.geohash_storage import GeohashStorageManager

logger = get_logger(__name__)


class MovementAgentV2(BaseAgent):
    """
    Movement Analysis Agent with two-step approach:
    1. Classify intent
    2. Route to specialized prompt
    """

    def __init__(self, name: str, config: dict, resource_manager):
        """Initialize with agent-specific components only"""
        super().__init__(name, config, resource_manager)
        
        # Initialize shared location resolver
        self.location_resolver = LocationResolver(resource_manager, config)
        
        # Initialize geohash storage with the ClickHouse client from BaseAgent
        # BaseAgent's __init__ has already initialized self.clickhouse_client
        self.geohash_storage = GeohashStorageManager(self.clickhouse_client)
        
        # Agent-specific components
        self.response_parser = MovementResponseParser()
        
        # Initialize specialized parsers
        self.single_location_parser = SingleLocationParser()
        
        # Store last responses for debugging
        self._last_llm_response = None
        self._last_intent_classification = None
        self._last_parsed_result = None
        
        # Map intents to specialized prompts and parsers
        self.intent_prompt_map = {
            "movement_pattern": MOVEMENT_PATTERN_PROMPT,
            "co_presence": CO_PRESENCE_PROMPT,
            "single_location": self.single_location_parser.get_prompt(),
            # For other intents, fall back to general MOVEMENT_PROMPT
        }
        
        self.intent_parser_map = {
            "single_location": self.single_location_parser,
            # Add other specialized parsers here as they're created
        }
        
        # UAE timezone for context
        self.uae_tz = pytz.timezone('Asia/Dubai')

    def _get_time_context(self) -> str:
        """Get current time context for UAE"""
        now_uae = datetime.now(self.uae_tz)
        return (
            f"Current time: {now_uae.strftime('%Y-%m-%d %H:%M:%S')} UAE Time\n"
            f"Current date: {now_uae.strftime('%Y-%m-%d')}\n"
            f"Current day: {now_uae.strftime('%A')}\n"
            f"Timezone: Asia/Dubai (UTC+4)"
        )

    async def validate_request(self, request: AgentRequest) -> bool:
        """Validate if the agent can handle this request"""
        movement_keywords = [
            'location', 'movement', 'where', 'visit', 'went', 'travel',
            'present', 'at', 'in', 'near', 'around', 'between',
            'commute', 'pattern', 'frequent', 'stay', 'dwell',
            'meet', 'together', 'density', 'crowd', 'heatmap',
            'predict', 'forecast', 'anomaly', 'unusual'
        ]
        
        prompt_lower = request.prompt.lower()
        return any(keyword in prompt_lower for keyword in movement_keywords)

    async def process_internal(self, request: AgentRequest) -> AgentResponse:
        """
        Core movement filter extraction logic with two-step approach
        """
        start_time = datetime.now()
        
        # Step 1: Classify Intent
        self.activity_logger.action("Step 1: Classifying movement query intent")
        
        intent_classification = await self._classify_intent(request)
        
        if not intent_classification:
            raise ValueError("Failed to classify query intent")
        
        self._last_intent_classification = intent_classification
        
        # Log intent classification
        primary_intent = intent_classification.get('primary_intent', 'unknown')
        confidence = intent_classification.get('confidence', 0)
        self.activity_logger.info(
            f"Intent classified as: {primary_intent} (confidence: {confidence:.2f})"
        )
        
        if intent_classification.get('secondary_intents'):
            self.activity_logger.info(
                f"Secondary intents: {intent_classification['secondary_intents']}"
            )
        
        # Step 2: Route to appropriate prompt
        self.activity_logger.action("Step 2: Processing with specialized prompt")
        
        system_prompt = self._get_prompt_for_intent(primary_intent)
        time_context = self._get_time_context()
        user_prompt = f"{time_context}\n\nExtract movement information from: {request.prompt}"
        
        # Get LLM response with specialized prompt
        llm_response = await self._call_llm_with_schema_retry(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            request=request,
            event_type=f"movement_{primary_intent}_extraction"
        )
        
        if not llm_response:
            raise ValueError("Failed to get valid LLM response after retries")
        
        # Store LLM response if capture is requested
        if request.context.get('capture_llm_response', False):
            self._last_llm_response = llm_response
        
        # Parse response based on intent type
        self.activity_logger.action("Parsing movement filter response")
        
        # Check if we have a specialized parser for this intent
        specialized_parser = self.intent_parser_map.get(primary_intent)
        
        if specialized_parser:
            # Use specialized parser (returns model object)
            parsed_result = specialized_parser.parse_response(llm_response)
            self._last_parsed_result = parsed_result
            
            # Convert to format expected by MovementFilterResult
            result = self._convert_specialized_result(parsed_result, primary_intent)
            
        elif primary_intent in ["movement_pattern", "co_presence"]:
            # Use existing specialized parsing
            result = self._parse_specialized_response(llm_response, primary_intent)
        else:
            # Use general parser
            result = self.response_parser.parse(llm_response)
        
        # Add intent classification to result
        result.raw_extractions['intent_classification'] = intent_classification
        result.query_type = primary_intent
        
        # Log reasoning if available
        if result.reasoning:
            self.activity_logger.info(f"Reasoning: {result.reasoning[:200]}...")
        
        # Log what was identified
        self.activity_logger.identified(
            f"movement analysis ({primary_intent})",
            {
                "intent": primary_intent,
                "intent_confidence": confidence,
                "geofences": len(result.geofences),
                "identity_filters": len(result.identity_filters),
                "has_co_presence": result.co_presence is not None,
                "ambiguities": len(result.ambiguities),
                "confidence": result.confidence
            }
        )
        
        # Enrich geofences with geohashes if requested
        if request.context.get('enrich_geofences', True) and result.geofences:
            self.activity_logger.action("Enriching geofences with geohash data")
            query_id = request.context.get('query_id', request.request_id)
            result.geofences = await self._enrich_and_persist_geofences(
                query_id=query_id,
                geofences=result.geofences
            )
        
        # Create response with metadata
        return self._create_success_response(
            request=request,
            result=result.to_dict(),
            start_time=start_time,
            confidence=result.confidence,
            extraction_method=f"two_step_{primary_intent}",
            validation_warnings=result.validation_warnings,
            intent_classification=intent_classification
        )

    async def _classify_intent(self, request: AgentRequest) -> Optional[Dict]:
        """Classify the query intent using MOVEMENT_INTENT_CLASSIFIER"""
        try:
            # Include time context
            time_context = self._get_time_context()
            user_prompt = f"{time_context}\n\nClassify this movement query: {request.prompt}"
            
            response = await self.call_llm(
                system_prompt=MOVEMENT_INTENT_CLASSIFIER,
                user_prompt=user_prompt,
                request=request,
                event_type="movement_intent_classification"
            )
            
            if not response:
                logger.error("LLM returned empty response for intent classification")
                return None
            
            # Log raw response for debugging
            logger.debug(f"Intent classification raw response: {response[:200]}...")
            
            return parse_json_response(response)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse intent classification: {e}")
            logger.error(f"Raw response was: {response if 'response' in locals() else 'No response'}")
            return None
        except Exception as e:
            logger.error(f"Error classifying intent: {e}")
            return None

    def _get_prompt_for_intent(self, intent: str) -> str:
        """Get the appropriate prompt for the classified intent"""
        # Use specialized prompts if available
        specialized_prompt = self.intent_prompt_map.get(intent)
        if specialized_prompt:
            logger.info(f"Using specialized prompt for intent: {intent}")
            return specialized_prompt
        
        # Fall back to general MOVEMENT_PROMPT
        logger.info(f"Using general prompt for intent: {intent}")
        return MOVEMENT_PROMPT

    def _parse_specialized_response(self, llm_response: str, intent: str) -> Any:
        """Parse response from specialized prompts"""
        try:
            data = parse_json_response(llm_response)
            
            # Create a result object compatible with MovementFilterResult
            from .models import MovementFilterResult, Geofence, SpatialFilter
            
            result = MovementFilterResult()
            result.reasoning = data.get('reasoning', '')
            result.query_type = intent
            result.confidence = data.get('confidence', 0.8)
            
            # Handle movement_pattern specific response
            if intent == "movement_pattern":
                # Extract identity filters
                if 'identity_filters' in data:
                    result.identity_filters = data['identity_filters']
                
                # Extract location filter as geofence
                if 'location_filter' in data and data['location_filter']:
                    loc = data['location_filter']
                    spatial_filter = SpatialFilter(
                        method=loc.get('method', 'name'),
                        value=loc.get('value'),
                        latitude=loc.get('latitude'),
                        longitude=loc.get('longitude'),
                        radius_meters=loc.get('radius_meters', 1000)
                    )
                    
                    geofence = Geofence(
                        id="movement_area",
                        reference=loc.get('value', 'movement area'),
                        spatial_filter=spatial_filter
                    )
                    result.geofences = [geofence]
                
                # Extract time constraints
                if 'time_constraints' in data:
                    result.global_time_filter = data['time_constraints']
                
                # Extract output options
                if 'output_options' in data:
                    result.output_options = data['output_options']
            
            # Handle co_presence specific response
            elif intent == "co_presence":
                # Log the raw data for debugging
                logger.debug(f"Co-presence raw data: {json.dumps(data, indent=2)}")
                
                # Extract reasoning
                result.reasoning = data.get('reasoning', '')
                
                # Extract identity filters
                if 'identity_filters' in data:
                    result.identity_filters = data['identity_filters']
                
                # Create co_presence configuration
                result.co_presence = {
                    "target_ids": [],
                    "match_granularity": "geohash7",
                    "time_window_days": 7
                }
                
                # Extract phone numbers and IMSIs as target IDs
                if 'identity_filters' in data:
                    id_filters = data['identity_filters']
                    target_ids = []
                    if 'phone_no' in id_filters and id_filters['phone_no']:
                        target_ids.extend(id_filters['phone_no'])
                    if 'imsi' in id_filters and id_filters['imsi']:
                        target_ids.extend(id_filters['imsi'])
                    result.co_presence['target_ids'] = target_ids
                
                # Initialize geofences list if not already
                if not result.geofences:
                    result.geofences = []
                    
                # Extract location filter as geofence
                if 'location_filter' in data and data['location_filter']:
                    # Handle array format from co_presence prompt
                    loc_filters = data['location_filter'] if isinstance(data['location_filter'], list) else [data['location_filter']]
                    for idx, loc in enumerate(loc_filters):
                        spatial_filter = SpatialFilter(
                            method=loc.get('method', 'name'),
                            value=loc.get('value'),
                            latitude=loc.get('latitude'),
                            longitude=loc.get('longitude'),
                            radius_meters=loc.get('radius_meters', 1000)
                        )
                        
                        geofence = Geofence(
                            id=f"co_presence_area_{idx}",
                            reference=loc.get('value', 'co-presence area'),
                            spatial_filter=spatial_filter
                        )
                        result.geofences.append(geofence)
                
                # Extract time constraints
                if 'time_constraints' in data:
                    result.global_time_filter = data['time_constraints']
            
            # Extract ambiguities
            if 'ambiguities' in data:
                result.ambiguities = data['ambiguities']
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing specialized response for {intent}: {e}")
            # Fall back to general parser
            return self.response_parser.parse(llm_response)

    async def _call_llm_with_schema_retry(
        self, 
        system_prompt: str, 
        user_prompt: str,
        request: AgentRequest,
        event_type: str = "movement_extraction",
        retry_count: int = 0
    ) -> Optional[str]:
        """Call LLM with retry logic for JSON validation"""
        try:
            current_user_prompt = user_prompt
            
            if retry_count > 0:
                self.activity_logger.action(f"Retrying with schema hint (attempt {retry_count + 1})")
                current_user_prompt = self._enhance_prompt_with_schema(user_prompt)
            
            response = await self.call_llm(
                system_prompt=system_prompt,
                user_prompt=current_user_prompt,
                request=request,
                event_type=event_type
            )
            
            if not response:
                return None
            
            # Validate JSON format
            try:
                # Validate that response can be parsed
                parse_json_response(response)
                return response  # Return original for further processing
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON response from LLM: {e}")
                
                if retry_count < MAX_RETRIES:
                    return await self._call_llm_with_schema_retry(
                        system_prompt,
                        user_prompt,
                        request,
                        event_type,
                        retry_count + 1
                    )
                return None
                
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            if retry_count < MAX_RETRIES:
                return await self._call_llm_with_schema_retry(
                    system_prompt,
                    user_prompt,
                    request,
                    event_type,
                    retry_count + 1
                )
            return None

    def _convert_specialized_result(self, parsed_result: Any, intent: str) -> Any:
        """Convert specialized parser result to MovementFilterResult"""
        from .models import MovementFilterResult, Geofence, SpatialFilter
        from .models.single_location import SingleLocationResult
        
        result = MovementFilterResult()
        
        if intent == "single_location" and isinstance(parsed_result, SingleLocationResult):
            # Convert SingleLocationResult to MovementFilterResult
            result.reasoning = parsed_result.reasoning
            result.query_type = intent
            result.confidence = parsed_result.confidence
            result.ambiguities = [amb.to_dict() for amb in parsed_result.ambiguities]
            
            # Map country code filter to identity filters
            if parsed_result.country_code_filter:
                result.identity_filters['country_code'] = parsed_result.country_code_filter.country_code
            
            # Store location scope as metadata
            if parsed_result.location_scope:
                result.raw_extractions['location_scope'] = parsed_result.location_scope.to_dict()
            
            # Map location filter to geofence
            if parsed_result.location_filter:
                loc = parsed_result.location_filter
                spatial_filter = SpatialFilter(
                    method=loc.method,
                    value=loc.value,
                    latitude=loc.latitude,
                    longitude=loc.longitude,
                    radius_meters=loc.radius_meters,
                    polygon=loc.polygon
                )
                
                geofence = Geofence(
                    id="single_location",
                    reference=loc.value or "location",
                    spatial_filter=spatial_filter
                )
                result.geofences = [geofence]
            
            # Map time constraints
            if parsed_result.time_constraints:
                result.global_time_filter = parsed_result.time_constraints.to_dict()
            
            # Map presence requirements
            if parsed_result.presence_requirements:
                result.raw_extractions['presence_requirements'] = parsed_result.presence_requirements.to_dict()
            
            # Map output options
            result.output_options = parsed_result.output_options.to_dict()
            
            # Store the original parsed model for reference
            result.raw_extractions['parsed_model'] = parsed_result
            
        return result
    
    async def resolve_locations(self, location_names: list[str]) -> Dict[str, list[str]]:
        """
        Resolve location names to geohashes using shared LocationResolver
        
        Args:
            location_names: List of location names to resolve
            
        Returns:
            Dict mapping location names to list of geohashes
        """
        location_to_geohashes = {}
        
        for name in location_names:
            try:
                # Resolve location using shared resolver
                resolved_locations = self.location_resolver.resolve_location(name)
                
                # Collect all geohashes from all resolved locations
                all_geohashes = []
                for location in resolved_locations:
                    if location.existing_geohashes:
                        all_geohashes.extend(location.existing_geohashes)
                
                # Remove duplicates
                unique_geohashes = list(set(all_geohashes))
                location_to_geohashes[name] = unique_geohashes
                
                if unique_geohashes:
                    self.activity_logger.info(
                        f"Resolved '{name}' to {len(unique_geohashes)} geohash(es)"
                    )
                else:
                    self.activity_logger.warning(f"No geohashes found for location: {name}")
                    
            except Exception as e:
                logger.error(f"Error resolving location '{name}': {e}")
                location_to_geohashes[name] = []
        
        return location_to_geohashes
    
    async def _enrich_and_persist_geofences(
        self,
        query_id: str,
        geofences: list
    ) -> list:
        """
        Enrich geofences with resolved geohashes and persist to ClickHouse
        
        Args:
            query_id: UUID for the query
            geofences: List of Geofence objects
            
        Returns:
            List of enriched geofences with geohash metadata
        """
        enriched_geofences = []
        
        for idx, geofence in enumerate(geofences):
            try:
                # Skip if no spatial filter
                if not geofence.spatial_filter:
                    enriched_geofences.append(geofence)
                    continue
                
                # Resolve location to get geohashes
                location_name = geofence.reference
                spatial_filter = geofence.spatial_filter
                
                # Check if we have coordinates
                if spatial_filter.latitude and spatial_filter.longitude:
                    # Use provided coordinates
                    resolved_locations = self.location_resolver.resolve_location(
                        location_name=location_name,
                        radius_meters=spatial_filter.radius_meters or 500,
                        coordinates=(spatial_filter.latitude, spatial_filter.longitude)
                    )
                else:
                    # Resolve by name
                    resolved_locations = self.location_resolver.resolve_location(
                        location_name=location_name,
                        radius_meters=spatial_filter.radius_meters or 500
                    )
                
                # Enrich spatial filter with all resolved locations
                if resolved_locations:
                    # Store all resolved locations with full details
                    spatial_filter.resolved_locations = []
                    for loc in resolved_locations:
                        resolved_loc = {
                            "name": loc.name,
                            "lat": loc.lat,
                            "lng": loc.lng,
                            "address": loc.address or "",
                            "place_id": loc.place_id,
                            "rating": loc.rating,
                            "radius_meters": loc.expanded_radius or loc.radius_meters,
                            "geohash_count": len(loc.existing_geohashes)
                        }
                        spatial_filter.resolved_locations.append(resolved_loc)
                    
                    # For backward compatibility, also set the first location's coordinates
                    if not (spatial_filter.latitude and spatial_filter.longitude):
                        first_loc = resolved_locations[0]
                        spatial_filter.latitude = first_loc.lat
                        spatial_filter.longitude = first_loc.lng
                        if first_loc.expanded_radius:
                            spatial_filter.radius_meters = first_loc.expanded_radius
                
                # Collect all geohashes
                all_geohashes = []
                for loc in resolved_locations:
                    if loc.existing_geohashes:
                        all_geohashes.extend(loc.existing_geohashes)
                
                # Remove duplicates
                unique_geohashes = list(set(all_geohashes))
                
                if unique_geohashes:
                    # Store geohashes to ClickHouse
                    # Use the first resolved location for coordinates
                    if resolved_locations:
                        first_loc = resolved_locations[0]
                        stored_count = await self.geohash_storage.store_location_geohashes(
                            query_id=query_id,
                            location_index=idx,
                            location_name=location_name,
                            geohashes=unique_geohashes,
                            latitude=first_loc.lat,
                            longitude=first_loc.lng,
                            radius_meters=first_loc.expanded_radius or first_loc.radius_meters,
                            confidence=getattr(geofence, 'confidence', 1.0)
                        )
                        
                        # Add metadata to geofence (without the actual geohash list)
                        geofence.geohash_metadata = {
                            'query_id': str(query_id),
                            'location_name': location_name,
                            'location_index': idx,
                            'total_geohashes_count': len(unique_geohashes),
                            'stored_count': stored_count,
                            'table_name': 'telecom_db.query_location_geohashes'
                        }
                        
                        self.activity_logger.info(
                            f"Enriched geofence '{location_name}' with {len(unique_geohashes)} geohashes"
                        )
                    else:
                        # No locations resolved
                        geofence.geohash_metadata = {
                            'query_id': str(query_id),
                            'location_name': location_name,
                            'location_index': idx,
                            'total_geohashes_count': 0,
                            'stored_count': 0,
                            'table_name': 'telecom_db.query_location_geohashes',
                            'error': 'No locations resolved'
                        }
                else:
                    # No geohashes found
                    geofence.geohash_metadata = {
                        'query_id': str(query_id),
                        'location_name': location_name,
                        'location_index': idx,
                        'total_geohashes_count': 0,
                        'stored_count': 0,
                        'table_name': 'telecom_db.query_location_geohashes',
                        'error': 'No geohashes found in area'
                    }
                
                enriched_geofences.append(geofence)
                
            except Exception as e:
                logger.error(f"Error enriching geofence {idx}: {e}")
                # Add error metadata
                geofence.geohash_metadata = {
                    'query_id': str(query_id),
                    'location_name': geofence.reference,
                    'location_index': idx,
                    'total_geohashes_count': 0,
                    'stored_count': 0,
                    'table_name': 'telecom_db.query_location_geohashes',
                    'error': str(e)
                }
                enriched_geofences.append(geofence)
        
        return enriched_geofences

    def _enhance_prompt_with_schema(self, user_prompt: str) -> str:
        """Add schema hint to prompt for retry"""
        # Ensure time context is maintained
        if self._get_time_context() not in user_prompt:
            user_prompt = f"{self._get_time_context()}\n\n{user_prompt}"
            
        schema_hint = """
        
IMPORTANT: Return ONLY valid JSON. No additional text or explanation.
Ensure all JSON syntax is correct (proper quotes, commas, brackets).
"""
        return f"{user_prompt}\n{schema_hint}"