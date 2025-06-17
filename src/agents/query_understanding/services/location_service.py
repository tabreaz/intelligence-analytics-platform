# src/agents/query_understanding/services/location_service.py
"""
Location extraction service - handles communication with LocationExtractorAgent
"""
import time
from typing import Optional, Dict, Any, TYPE_CHECKING

import geohash2

from src.core.logger import get_logger
from src.core.session_manager_models import LLMInteraction, TrainingExample
from ..constants import (
    CONTINUATION_PHRASES, LOCATION_MODIFIERS,
    REPLACEMENT_INDICATORS, ADDITION_INDICATORS, MODIFICATION_INDICATORS,
    TOKEN_ESTIMATION_DIVISOR, MIN_CONTINUATION_QUERY_WORDS,
    GEOHASH7_LENGTH, GEOHASH6_LENGTH
)
from ...base_agent import AgentRequest
from ...location_extractor.agent import LocationExtractorAgent
from ....core.database.clickhouse_client import ClickHouseClient

if TYPE_CHECKING:
    from src.core.session_manager import EnhancedSessionManager

logger = get_logger(__name__)


class LocationExtractionService:
    """Service to handle location extraction from queries"""

    def __init__(self, location_extractor: LocationExtractorAgent, session_manager: 'EnhancedSessionManager' = None):
        self.location_extractor = location_extractor
        self.session_manager = session_manager

    async def extract_locations(self, query: str, query_id: str, session_id: str = None,
                                query_context: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        Extract locations from query using LocationExtractorAgent with context awareness
        
        Args:
            query: Current query text
            query_id: Unique query identifier
            session_id: Session identifier for tracking
            query_context: Previous query context including locations
            
        Returns:
            Location extraction results or None if no locations found
        """
        start_time = time.time()

        try:
            # Check if this is a continuation query
            enhanced_query = query
            is_continuation = False

            if query_context:
                is_continuation = self._is_continuation_query(query, query_context)
                if is_continuation:
                    enhanced_query = self._build_contextual_prompt(query, query_context)
                    logger.info(f"Detected continuation query, enhanced prompt for location extraction")

            # Create agent request with potentially enhanced query
            agent_request = AgentRequest(
                request_id=query_id,
                prompt=enhanced_query,
                context={
                    "source": "query_understanding",
                    "session_id": session_id,
                    "is_continuation": is_continuation,
                    "original_query": query if is_continuation else None
                }
            )

            # Quick validation - if agent says no locations, skip
            if not await self.location_extractor.validate_request(agent_request):
                logger.info(f"No location indicators found in query: {query_id}")

                # Still record this as a location extraction attempt
                if self.session_manager and session_id:
                    await self._record_location_extraction(
                        session_id=session_id,
                        query_id=query_id,
                        query_text=query,
                        location_result=None,
                        latency_ms=int((time.time() - start_time) * 1000),
                        success=True,
                        reason="no_locations_detected"
                    )

                return None

            # Process location extraction
            response = await self.location_extractor.process(agent_request)

            latency_ms = int((time.time() - start_time) * 1000)

            if response.status.value == "completed" and response.result:
                result = response.result
                # Add query_id to result for tracking
                result['query_id'] = query_id

                # Record successful extraction
                if self.session_manager and session_id:
                    await self._record_location_extraction(
                        session_id=session_id,
                        query_id=query_id,
                        query_text=query,
                        location_result=result,
                        latency_ms=latency_ms,
                        success=True
                    )

                return result

            # Record failed extraction
            if self.session_manager and session_id:
                await self._record_location_extraction(
                    session_id=session_id,
                    query_id=query_id,
                    query_text=query,
                    location_result=None,
                    latency_ms=latency_ms,
                    success=False,
                    reason=f"agent_status: {response.status.value}"
                )

            return None

        except Exception as e:
            from ..exceptions import LocationExtractionError
            logger.error(f"Location extraction failed for query {query_id}: {e}")

            # Record exception
            if self.session_manager and session_id:
                try:
                    await self._record_location_extraction(
                        session_id=session_id,
                        query_id=query_id,
                        query_text=query,
                        location_result=None,
                        latency_ms=int((time.time() - start_time) * 1000),
                        success=False,
                        reason=str(e)
                    )
                except Exception as record_error:
                    logger.error(f"Failed to record location extraction error: {record_error}")

            # Re-raise as our custom exception
            raise LocationExtractionError(f"Failed to extract locations: {str(e)}")

    async def _record_location_extraction(
            self,
            session_id: str,
            query_id: str,
            query_text: str,
            location_result: Optional[Dict[str, Any]],
            latency_ms: int,
            success: bool,
            reason: str = None
    ):
        """Record location extraction for training and metrics"""

        try:
            # Create LLM interaction record (even though it's the location agent)
            llm_interaction = LLMInteraction(
                model="location_extractor_agent",
                prompt_template="location_extraction",
                prompt_tokens=len(query_text) // TOKEN_ESTIMATION_DIVISOR,  # Rough estimate
                completion_tokens=len(str(location_result)) // TOKEN_ESTIMATION_DIVISOR if location_result else 0,
                latency_ms=latency_ms,
                temperature=0.0,  # Not applicable for location agent
                max_tokens=0
            )

            # Prepare input/output data
            input_data = {
                "query": query_text,
                "validation_passed": success and reason != "no_locations_detected"
            }

            output_data = {
                "location_result": location_result,
                "success": success,
                "reason": reason
            } if location_result else {"success": success, "reason": reason}

            # Record interaction event
            await self.session_manager.record_interaction_event(
                session_id=session_id,
                query_id=query_id,
                event_type="location_extraction",
                llm_interaction=llm_interaction,
                input_data=input_data,
                output_data=output_data,
                success=success,
                confidence=location_result.get('average_confidence', 0.0) if location_result else 0.0
            )

            # Create training example if we have results
            if location_result and location_result.get('locations'):
                training_example = TrainingExample(
                    query_id=query_id,
                    query_text=query_text,
                    normalized_query=self._normalize_query(query_text),
                    category="location_extraction",  # Special category for location training
                    extracted_params={
                        "locations": location_result.get('locations', {}),
                        "geohashes": location_result.get('geohashes', [])
                    },
                    confidence=location_result.get('average_confidence', 1.0),
                    event_type="location_extraction",
                    input_data=input_data,
                    output_data=output_data,
                    has_positive_feedback=success
                )

                await self.session_manager.record_training_example(training_example)

        except Exception as e:
            logger.error(f"Failed to record location extraction: {e}")
            # Don't fail the main process

    def _normalize_query(self, query: str) -> str:
        """Normalize query text for storage"""
        return " ".join(query.strip().lower().split())

    def _is_continuation_query(self, query: str, query_context: Dict[str, Any]) -> bool:
        """
        Detect if this is a continuation query that might need location context
        Reuses patterns from the controller for consistency
        """
        # Use shared constants for consistency

        query_lower = query.lower()

        # Check if query contains continuation patterns
        has_continuation = any(phrase in query_lower for phrase in CONTINUATION_PHRASES)
        has_modifier = any(modifier in query_lower for modifier in LOCATION_MODIFIERS)

        # Check if previous context had locations
        previous_locations = query_context.get('previous_locations', {})
        had_locations = bool(previous_locations)

        # It's a continuation if:
        # 1. Has continuation phrase and previous query had locations
        # 2. Has location modifier (implies modifying previous location)
        # 3. Very short query (< MIN_CONTINUATION_QUERY_WORDS) with previous locations
        word_count = len(query.split())
        is_short_query = word_count < MIN_CONTINUATION_QUERY_WORDS and had_locations

        return (has_continuation and had_locations) or has_modifier or is_short_query

    def _build_contextual_prompt(self, query: str, query_context: Dict[str, Any]) -> str:
        """
        Build an enhanced prompt with location context for the location extractor
        """
        previous_locations = query_context.get('previous_locations', {})
        previous_query = query_context.get('previous_query', '')

        # Determine the context action
        query_lower = query.lower()

        # Use shared constants to determine action
        if any(word in query_lower for word in REPLACEMENT_INDICATORS):
            action = "REPLACE"
        elif any(word in query_lower for word in ADDITION_INDICATORS):
            action = "ADD"
        elif any(word in query_lower for word in MODIFICATION_INDICATORS):
            action = "MODIFY"
        else:
            action = "INHERIT"

        # Build contextual prompt
        context_prompt = f"""Current query: "{query}"

This is a continuation of a previous query about locations.

Previous query: "{previous_query}"
Previous locations extracted: {self._format_previous_locations(previous_locations)}

Context action: {action}
- INHERIT: Reuse previous locations if no new ones mentioned
- REPLACE: Replace previous locations with new ones
- ADD: Add new locations to previous ones
- MODIFY: Adjust previous locations based on modifiers

Please extract locations considering this context."""

        return context_prompt

    def _format_previous_locations(self, locations: Dict[str, Any]) -> str:
        """Format previous locations for context prompt"""
        if not locations:
            return "None"

        formatted = []
        for loc_key, loc_data in locations.items():
            if isinstance(loc_data, dict) and 'name' in loc_data:
                formatted.append(f"- {loc_data['name']} (radius: {loc_data.get('radius_meters', 'default')}m)")

        return "\n".join(formatted) if formatted else str(locations)


class GeohashStorageService:
    """Service to store extracted geohashes in ClickHouse"""

    def __init__(self, clickhouse_client: ClickHouseClient):
        self.ch_client = clickhouse_client

    async def store_geohashes(self, query_id: str, location_result: Dict[str, Any]) -> bool:
        """
        Store extracted geohashes in ClickHouse for SQL generation
        
        Returns:
            True if successful, False otherwise
        """
        if not location_result or not location_result.get('locations'):
            logger.info(f"No locations to store for query {query_id}")
            return True

        try:
            insert_data = []

            # Process each location
            for loc_key, loc_data in location_result['locations'].items():
                # Validate location key
                from ..validators import LocationValidator
                is_valid, location_index = LocationValidator.validate_location_key(loc_key)
                if not is_valid:
                    logger.warning(f"Invalid location key {loc_key} - skipping")
                    continue
                location_name = loc_data['name']
                radius_meters = loc_data.get('radius_meters', 2000)

                # Store each geohash with its metadata
                for geohash in loc_data.get('total_existing_geohashes', []):
                    # Decode geohash to get coordinates
                    lat, lng = geohash2.decode(geohash)

                    # Validate geohash
                    from ..validators import LocationValidator
                    is_valid, error_msg = LocationValidator.validate_geohash(geohash, GEOHASH7_LENGTH)
                    if not is_valid:
                        logger.warning(f"Skipping geohash {geohash} - {error_msg}")
                        continue

                    # Generate geohash6 from geohash7
                    geohash6 = geohash[:GEOHASH6_LENGTH]

                    insert_data.append({
                        'query_id': str(query_id),  # Ensure UUID is string
                        'location_name': str(location_name),
                        'location_index': int(location_index),
                        'geohash7': str(geohash),  # Ensure string
                        'geohash6': str(geohash6),  # Ensure string
                        'latitude': float(lat),
                        'longitude': float(lng),
                        'radius_meters': int(radius_meters),
                        'confidence': float(loc_data.get('average_confidence', 1.0))
                        # Remove created_at and part_date - let ClickHouse use defaults
                    })

            if insert_data:
                logger.info(f"Attempting to insert {len(insert_data)} geohashes for query {query_id}")
                logger.debug(f"First row sample: {insert_data[0] if insert_data else 'No data'}")

                # Convert list of dicts to list of lists for clickhouse-connect
                column_names = ['query_id', 'location_name', 'location_index',
                                'geohash7', 'geohash6', 'latitude', 'longitude',
                                'radius_meters', 'confidence']

                # Convert dict data to list format
                data_lists = []
                for row in insert_data:
                    data_lists.append([row[col] for col in column_names])

                # Batch insert into ClickHouse
                self.ch_client.insert(
                    'telecom_db.query_location_geohashes',
                    data_lists,
                    column_names=column_names
                )
                logger.info(f"Successfully stored {len(insert_data)} geohashes for query {query_id}")
                return True

            return True

        except Exception as e:
            from ..exceptions import StorageError
            logger.error(f"Failed to store geohashes in ClickHouse: {e}")
            logger.error(f"Insert data sample: {insert_data[0] if insert_data else 'No data'}")
            # Don't raise - return False to indicate failure
            # This allows the query to continue even if storage fails
            return False
