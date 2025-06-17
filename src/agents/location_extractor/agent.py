# src/agents/location_extractor/agent.py
from src.agents.base_agent import BaseAgent, AgentRequest, AgentResponse, AgentStatus
from src.agents.location_extractor.location_processor import LocationProcessor
from src.agents.location_extractor.redis_manager import RedisGeohashManager
from src.agents.location_extractor.geohash_storage import GeohashStorageManager
from src.core.activity_logger import ActivityLogger
from src.core.config_manager import ConfigManager
from src.core.llm.base_llm import LLMClientFactory
from src.core.logger import get_logger
from src.core.session_manager_models import QueryContext
from src.core.training_logger import TrainingLogger

logger = get_logger(__name__)


class LocationExtractorAgent(BaseAgent):
    """Agent for extracting locations from prompts and returning geohashes"""

    def __init__(self, name: str, config: dict, config_manager: ConfigManager, session_manager=None):
        super().__init__(name, config)
        self.config_manager = config_manager
        self.session_manager = session_manager

        # Initialize Redis manager
        redis_config = config_manager.get_database_config('redis').__dict__
        self.redis_manager = RedisGeohashManager(redis_config)
        
        # Initialize geohash storage manager (ClickHouse)
        self.geohash_storage = None
        try:
            from ...core.database.clickhouse_client import ClickHouseClient
            ch_config = config_manager.get_database_config('clickhouse')
            ch_client = ClickHouseClient(ch_config)
            self.geohash_storage = GeohashStorageManager(ch_client)
        except Exception as e:
            logger.warning(f"Could not initialize geohash storage: {e}")

        # Initialize LLM client
        llm_config = config_manager.get_llm_config()
        self.llm_client = LLMClientFactory.create_client(llm_config)

        # Initialize location processor
        self.location_processor = LocationProcessor(
            config, self.llm_client, self.redis_manager
        )

        # Enable training data logging
        self.enable_training_data = config.get('enable_training_data', True)

        # Initialize training logger
        self.training_logger = TrainingLogger(session_manager) if self.enable_training_data else None

        # Initialize activity logger
        self.activity_logger = ActivityLogger(agent_name=self.name)

        logger.info(f"LocationExtractorAgent initialized with {self.redis_manager.get_total_geohash_count()} geohashes")

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
        import re
        has_coordinates = bool(re.search(r'\d+\.\d+[,\s]+\d+\.\d+', request.prompt))

        # Check for proper nouns (potential place names) - capitalized words
        has_proper_nouns = bool(re.search(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', request.prompt))

        # Allow through if any indicator suggests locations might be present
        # The LLM extraction will do the real validation
        should_process = has_location_indicators or has_coordinates or has_proper_nouns

        logger.info(
            f"Fast validation: {should_process} (indicators: {has_location_indicators}, coords: {has_coordinates}, proper_nouns: {has_proper_nouns})")

        return should_process

    async def process(self, request: AgentRequest) -> AgentResponse:
        """Process location extraction request - pure location-to-geohashes conversion"""

        # Set query context for activity logging
        query_context = request.context.get('query_context')
        if query_context and isinstance(query_context, QueryContext):
            self.activity_logger.set_query_context(query_context)

        try:
            # Log start
            self.activity_logger.action("Searching for location mentions in query...")

            # Single LLM call for location extraction
            locations_data, llm_metadata = await self.location_processor.extract_locations_from_prompt(
                request.prompt
            )

            # Extract ambiguities from metadata
            ambiguities = llm_metadata.get('ambiguities', [])
            
            # If no locations found, return clean result with ambiguities
            if not locations_data:
                logger.info("No locations detected in prompt")
                if ambiguities:
                    self.activity_logger.info(f"No specific locations found, but {len(ambiguities)} ambiguous references detected")
                else:
                    self.activity_logger.info("No specific locations found in the query")
                    
                return AgentResponse(
                    request_id=request.request_id,
                    agent_name=self.name,
                    status=AgentStatus.COMPLETED,
                    result={
                        "locations": {},
                        "total_locations": 0,
                        "geohashes": [],
                        "geohash_count": 0,
                        "ambiguities": ambiguities,
                        "summary": f"No locations found in prompt. {len(ambiguities)} ambiguities detected." if ambiguities else "No locations found in prompt"
                    }
                )

            # Log locations found in user-friendly way
            location_names = [loc_data['name'] for loc_data in locations_data.values()]
            if len(location_names) == 1:
                self.activity_logger.info(f"Found location: {location_names[0]}")
            else:
                self.activity_logger.info(f"Found {len(location_names)} locations: {', '.join(location_names[:3])}" +
                                          (" and more..." if len(location_names) > 3 else ""))

            # Store geohashes in ClickHouse if storage is available
            query_id = request.request_id
            
            # Consolidate all geohashes from all locations
            all_geohashes = []
            total_confidence = 0.0
            location_count = 0

            for location_idx, location_data in enumerate(locations_data.values()):
                # Handle new format with type field
                if location_data.get('type') in ['CITY', 'EMIRATE']:
                    # City/Emirate locations don't have geohashes
                    total_confidence += location_data.get('confidence', 0.9)
                    location_count += 1
                else:
                    # Facility locations have geohashes
                    geohashes = location_data.get('total_existing_geohashes', [])
                    all_geohashes.extend(geohashes)
                    total_confidence += location_data.get('confidence', 0.9)
                    location_count += 1
                    
                    # Store geohashes in ClickHouse for this facility
                    if self.geohash_storage and geohashes:
                        try:
                            # Get first coordinate for center point
                            coords = location_data.get('coordinates', [])
                            if coords:
                                lat = coords[0].get('lat', 0)
                                lng = coords[0].get('lng', 0)
                            else:
                                lat, lng = 0, 0
                            
                            await self.geohash_storage.store_location_geohashes(
                                query_id=query_id,
                                location_index=location_idx,
                                location_name=location_data.get('name', ''),
                                geohashes=geohashes,
                                latitude=lat,
                                longitude=lng,
                                radius_meters=location_data.get('radius_meters', 500),
                                confidence=location_data.get('confidence', 0.9)
                            )
                            logger.info(f"Stored {len(geohashes)} geohashes for {location_data.get('name')}")
                        except Exception as e:
                            logger.error(f"Failed to store geohashes: {e}")

            # Remove duplicate geohashes
            unique_geohashes = list(set(all_geohashes))
            avg_confidence = total_confidence / location_count if location_count > 0 else 0.0

            # Log geohash mapping result
            if unique_geohashes:
                self.activity_logger.decision(f"Mapped to {len(unique_geohashes)} geographic areas for data retrieval")
            else:
                self.activity_logger.info("No matching geographic data found for these locations")

            # Pure location extraction result - no database assumptions
            result = {
                "locations": locations_data,
                "total_locations": len(locations_data),
                "geohashes": unique_geohashes,
                "geohash_count": len(unique_geohashes),
                "ambiguities": ambiguities,
                "average_confidence": round(avg_confidence, 2),
                "summary": f"Extracted {len(locations_data)} locations -> {len(unique_geohashes)} geohashes (confidence: {avg_confidence:.2f}). {len(ambiguities)} ambiguities detected." if ambiguities else f"Extracted {len(locations_data)} locations -> {len(unique_geohashes)} geohashes (confidence: {avg_confidence:.2f})",
                "processing_stats": {
                    "llm_calls": 1,
                    "redis_geohashes_available": self.redis_manager.get_total_geohash_count(),
                    "extraction_method": "single_llm_call",
                    "llm_metadata": llm_metadata  # Include for debugging
                }
            }

            logger.info(f"Location extraction completed: {result['summary']}")

            # Log training data if enabled (fire-and-forget)
            if self.training_logger and llm_metadata.get('llm_response'):
                # Get model config
                model_config = {
                    'model': getattr(self.llm_client.config, 'model', 'unknown') if hasattr(self.llm_client,
                                                                                            'config') else 'unknown',
                    'temperature': getattr(self.llm_client.config, 'temperature', 0.1) if hasattr(self.llm_client,
                                                                                                  'config') else 0.1,
                    'max_tokens': getattr(self.llm_client.config, 'max_tokens', 1000) if hasattr(self.llm_client,
                                                                                                 'config') else 1000
                }

                # Prepare extracted params
                extracted_params = {
                    "locations": locations_data,
                    "total_locations": len(locations_data),
                    "unique_geohashes": len(unique_geohashes),
                    "average_confidence": avg_confidence
                }

                # Log in background
                self.training_logger.log_llm_interaction_background(
                    session_id=request.context.get('session_id'),
                    query_id=request.context.get('query_id', request.request_id),
                    query_text=request.prompt,
                    event_type="location_extraction",
                    llm_response=llm_metadata['llm_response'],
                    llm_start_time=llm_metadata['llm_start_time'],
                    prompts=llm_metadata['prompts'],
                    result=result,
                    model_config=model_config,
                    extracted_params=extracted_params,
                    category="location_extraction",
                    confidence=avg_confidence,
                    success=bool(locations_data)
                )

            return AgentResponse(
                request_id=request.request_id,
                agent_name=self.name,
                status=AgentStatus.COMPLETED,
                result=result,
                metadata={
                    "agent_role": "location_extraction_only",
                    "next_agents": ["data_correlator", "intelligence_analyzer"],
                    "output_type": "geohash_list"
                }
            )

        except Exception as e:
            logger.error(f"Location extraction failed: {str(e)}", exc_info=True)
            self.activity_logger.error("Failed to extract locations", error=e)
            raise

    def load_geohashes(self, geohash_source: str, source_type: str = 'list') -> int:
        """Load geohashes into Redis from various sources"""

        try:
            if source_type == 'list':
                # Assume geohash_source is a list of geohashes
                geohash_list = geohash_source if isinstance(geohash_source, list) else [geohash_source]
                return self.redis_manager.load_geohashes_from_list(geohash_list)

            elif source_type == 'file':
                # Load from file
                with open(geohash_source, 'r') as f:
                    geohash_list = [line.strip() for line in f if line.strip()]
                return self.redis_manager.load_geohashes_from_list(geohash_list)

            elif source_type == 'clickhouse':
                # Load from ClickHouse query (only for geohash loading, not querying)
                from src.core.database.clickhouse_client import ClickHouseClient
                ch_config = self.config_manager.get_database_config('clickhouse')
                ch_client = ClickHouseClient(ch_config)

                # Execute query to get geohashes for loading into Redis
                result = ch_client.execute(geohash_source)
                geohash_list = [row[0] for row in result]
                return self.redis_manager.load_geohashes_from_list(geohash_list)

            else:
                raise ValueError(f"Unsupported source type: {source_type}")

        except Exception as e:
            logger.error(f"Failed to load geohashes: {e}")
            raise
