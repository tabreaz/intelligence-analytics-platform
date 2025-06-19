# src/agents/location_extractor/agent.py
"""
Location Extractor Agent - Extracts locations from queries and converts to geohashes
Modernized version using BaseAgent with resource management
"""
import re
from datetime import datetime
from typing import List, Dict, Any

from .geohash_storage import GeohashStorageManager
from .location_processor import LocationProcessor
from .redis_manager import RedisGeohashManager
from ..base_agent import BaseAgent, AgentResponse, AgentRequest
from ...core.logger import get_logger
from ...core.location_resolver import LocationResolver

logger = get_logger(__name__)


class LocationExtractorAgent(BaseAgent):
    """Agent for extracting locations from prompts and returning geohashes"""

    def __init__(self, name: str, config: dict, resource_manager):
        """Initialize with agent-specific components only"""
        super().__init__(name, config, resource_manager)
        
        # Initialize shared location resolver
        self.location_resolver = LocationResolver(resource_manager, config)
        
        # Keep RedisGeohashManager for backward compatibility but use shared Redis client
        redis_client = resource_manager.get_redis_client()
        self.redis_manager = RedisGeohashManager({'client': redis_client})
        
        # Initialize geohash storage manager (ClickHouse)
        self.geohash_storage = None
        try:
            ch_pool = resource_manager.get_clickhouse_pool()
            # Create a ClickHouseClient with the pool
            from ...core.database.clickhouse_client import ClickHouseClient
            ch_config = resource_manager.config_manager.get_database_config('clickhouse')
            ch_client = ClickHouseClient(ch_config, use_pool=True)
            ch_client._pool = ch_pool  # Use the shared pool
            self.geohash_storage = GeohashStorageManager(ch_client)
        except Exception as e:
            logger.warning(f"Could not initialize geohash storage: {e}")
        
        # Initialize location processor with LLM client from resource manager
        llm_client = resource_manager.get_llm_client()
        self.location_processor = LocationProcessor(
            config, llm_client, self.redis_manager
        )
        
        # Log initialization status
        logger.info(f"LocationExtractorAgent initialized with {self.redis_manager.get_total_geohash_count()} geohashes")
        
        # Agent metadata handled by base class
        # self.agent_id, self.description, self.capabilities are in base class

    async def validate_request(self, request: AgentRequest) -> bool:
        """Fast validation using simple heuristics - no LLM call needed"""
        
        # Use lightweight heuristics for initial filtering
        prompt = request.prompt.lower()
        
        # Quick checks that don't require LLM
        has_location_indicators = any(word in prompt for word in [
            'near', 'at', 'in', 'around', 'visiting', 'went to', 'from', 'to',
            'mall', 'airport', 'restaurant', 'hotel', 'office', 'building',
            'street', 'road', 'city', 'country', 'area', 'place'
        ])
        
        # Check for coordinate patterns
        has_coordinates = bool(re.search(r'\d+\.\d+[,\s]+\d+\.\d+', request.prompt))
        
        # Check for proper nouns (potential place names) - capitalized words
        has_proper_nouns = bool(re.search(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', request.prompt))
        
        # Allow through if any indicator suggests locations might be present
        # The LLM extraction will do the real validation
        should_process = has_location_indicators or has_coordinates or has_proper_nouns
        
        logger.info(
            f"Fast validation: {should_process} (indicators: {has_location_indicators}, coords: {has_coordinates}, proper_nouns: {has_proper_nouns})")
        
        return should_process

    async def process_internal(self, request: AgentRequest) -> AgentResponse:
        """
        Core location extraction logic
        All infrastructure concerns handled by BaseAgent
        """
        start_time = datetime.now()
        
        # Log activity
        self.activity_logger.action("Searching for location mentions in query...")
        
        # Single LLM call for location extraction
        locations_data, llm_metadata = await self.location_processor.extract_locations_from_prompt(request.prompt)
        
        # Extract locations from the response
        # locations_data is a dict with keys like "location1", "location2", etc.
        locations = []
        if locations_data:
            for loc_key, loc_data in locations_data.items():
                if loc_key.startswith('location'):
                    # Convert to simpler format for processing
                    location = {
                        'location_name': loc_data.get('name', ''),
                        'type': loc_data.get('type', 'FACILITY'),
                        'field': loc_data.get('field', 'geohash'),
                        'value': loc_data.get('value', ''),
                        'confidence': loc_data.get('confidence', 0.9),
                        'geohashes': loc_data.get('total_existing_geohashes', [])
                    }
                    # Add coordinate info if available
                    if loc_data.get('coordinates'):
                        coord = loc_data['coordinates'][0]  # Take first coordinate
                        location['latitude'] = coord.get('lat')
                        location['longitude'] = coord.get('lng')
                    locations.append(location)
        
        if not locations:
            self.activity_logger.decision("No specific locations found")
            return self._create_success_response(
                request=request,
                result={
                    "locations": [],
                    "geohashes": [],
                    "location_contexts": [],
                    "extraction_status": "no_locations_found"
                },
                start_time=start_time
            )
        
        self.activity_logger.info(f"Extracted {len(locations)} location(s): {', '.join([loc['location_name'] for loc in locations])}")
        
        # Limit locations to prevent performance issues
        max_locations = self.config.get('max_locations', 200)
        if len(locations) > max_locations:
            self.activity_logger.warning(f"Found {len(locations)} locations, limiting to {max_locations}")
            locations = locations[:max_locations]
        
        # Extract unique geohashes from all locations
        all_geohashes = set()
        enriched_locations = []
        
        for location in locations:
            # Get geohashes for each location
            location_geohashes = await self._get_geohashes_for_location(location)
            
            if location_geohashes:
                all_geohashes.update(location_geohashes)
                enriched_location = location.copy()
                enriched_location['geohashes'] = list(location_geohashes)
                enriched_locations.append(enriched_location)
                
                self.activity_logger.info(
                    f"Location '{location['location_name']}' mapped to {len(location_geohashes)} geohash(es)"
                )
            else:
                self.activity_logger.warning(f"No geohashes found for location: {location['location_name']}")
        
        # Convert to sorted list
        final_geohashes = sorted(list(all_geohashes))
        
        self.activity_logger.identified(
            "location geohashes",
            {
                "total_locations": len(locations),
                "matched_locations": len(enriched_locations),
                "total_geohashes": len(final_geohashes)
            }
        )
        
        # Build response
        result = {
            "locations": [loc['location_name'] for loc in enriched_locations],
            "geohashes": final_geohashes,
            "location_contexts": enriched_locations,
            "extraction_status": "success"
        }
        
        return self._create_success_response(
            request=request,
            result=result,
            start_time=start_time,
            location_count=len(enriched_locations),
            geohash_count=len(final_geohashes)
        )

    async def _get_geohashes_for_location(self, location: Dict[str, Any]) -> List[str]:
        """Get geohashes for a single location"""
        # Check if location already has geohashes (from location processor)
        if location.get('geohashes'):
            return location['geohashes']
        
        geohashes = []
        
        # Priority 1: If location has coordinates, find nearby geohashes
        if location.get('latitude') and location.get('longitude'):
            try:
                # Get geohashes within radius
                radius = location.get('radius_meters', self.config.get('default_radius_meters', 200))
                nearby_geohashes = self.redis_manager.get_geohashes_in_radius(
                    location['latitude'],
                    location['longitude'], 
                    radius
                )
                geohashes.extend(nearby_geohashes)
            except Exception as e:
                logger.error(f"Error finding nearby geohashes: {e}")
        
        # Priority 2: Use value field if it's a geohash
        if not geohashes and location.get('value'):
            value = location['value']
            # Check if value looks like a geohash (7 chars alphanumeric)
            if isinstance(value, str) and len(value) == 7 and value.isalnum():
                geohashes.append(value)
        
        return geohashes

    async def load_geohashes_from_db(self) -> Dict[str, Any]:
        """Load geohashes from ClickHouse into Redis"""
        # This method would need to be implemented based on actual ClickHouse schema
        # For now, return a placeholder response
        try:
            self.activity_logger.action("Loading geohashes from Redis...")
            
            total_count = self.redis_manager.get_total_geohash_count()
            
            self.activity_logger.info(f"Redis contains {total_count} geohashes")
            
            return {
                "success": True,
                "total_geohashes": total_count,
                "message": "Geohashes already loaded in Redis"
            }
            
        except Exception as e:
            logger.error(f"Error checking geohashes: {e}")
            return {
                "success": False,
                "error": str(e)
            }