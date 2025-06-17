# src/agents/query_understanding/controller.py
"""
Query Understanding Controller - Main orchestrator for query processing
"""
import asyncio
import json
import time
import uuid
from typing import Dict, Any, List, Optional, Tuple

from src.core.logger import get_logger
from src.core.session_manager import EnhancedSessionManager
from src.core.session_manager_models import (
    QueryContext, Session, TrainingExample, InheritanceType, QueryStatus, LLMInteraction
)
from .constants import CONTINUATION_PHRASES
from .exceptions import ValidationError, ClassificationError
from .models import QueryCategory
from .prompt_builder import UnifiedPromptBuilder
from .prompts.prompts import SCHEMA_AWARE_PROMPT
from ...core.llm.base_llm import BaseLLMClient

logger = get_logger(__name__)


class QueryUnderstandingController:
    """
    Main controller for the query understanding system.
    Coordinates session management, prompt building, and response handling.
    """

    def __init__(
            self,
            session_manager: EnhancedSessionManager,
            llm_client: BaseLLMClient,
            location_service,  # Required location service
            geohash_storage,  # Required geohash storage
            classification_builder=None,  # Deprecated, kept for compatibility
            contextual_builder=None  # Deprecated, kept for compatibility
    ):
        self._last_session_id = None
        self.session_manager = session_manager
        self.llm_client = llm_client
        self.location_service = location_service
        self.geohash_storage = geohash_storage
        self.prompt_builder = UnifiedPromptBuilder()

    def _get_category_value(self, category) -> Optional[str]:
        """Safely get category value whether it's a string or enum"""
        if category is None:
            return None
        if isinstance(category, str):
            return category
        # Assume it's an enum with .value attribute
        return category.value

    async def process_query(
            self,
            query: str,
            session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a user query through the full pipeline:
        1. Get or create session
        2. Extract available context
        3. Build classification prompt
        4. Send to LLM and get classification result
        5. Handle ambiguities (if any)
        6. Build final contextual prompt
        7. Send to LLM and return structured result
        8. Save interaction to session history
        9. Log for training
        """
        # Validate inputs
        from .validators import QueryValidator

        is_valid, error_msg = QueryValidator.validate_query(query)
        if not is_valid:
            raise ValidationError("query", error_msg, query)

        is_valid, error_msg = QueryValidator.validate_session_id(session_id)
        if not is_valid:
            raise ValidationError("session_id", error_msg, session_id)

        # Step 1: Get or create session and generate query_id
        session = await self.session_manager.get_or_create_session(session_id)
        self._last_session_id = session.session_id  # Store for context continuity
        query_id = str(uuid.uuid4())

        # Step 2: Create preliminary query entry to satisfy foreign key constraints
        preliminary_context = QueryContext(
            query_id=query_id,
            query_text=query,
            normalized_query=self._normalize_query(query),
            category=None,  # Will be updated later
            confidence=0.0,
            inherited_from_query=None,
            inheritance_type=InheritanceType.NONE,
            inherited_elements={},
            extracted_params={},
            active_filters={},
            entities_mentioned={},
            status=QueryStatus.PENDING,
            result_count=0,
            execution_time_ms=0
        )
        await self.session_manager.add_query_to_session(session.session_id, preliminary_context)

        # Step 3: Extract available context from session and history
        query_history = await self.session_manager.get_session_history(session.session_id)
        # Remove the current query from history to avoid self-reference
        query_history = [q for q in query_history if q.query_id != query_id]
        available_context = self._extract_available_context(query, session, query_history)

        # Step 3: Execute parallel tasks
        result, location_result, classification_result = await self._execute_parallel_tasks(
            query, query_id, session, query_history, available_context
        )

        # Step 4: Handle location results and ambiguities
        result = await self._handle_location_results(
            result, location_result, query_id
        )

        # Step 7: Build query context from result
        query_context = self._build_query_context(
            query=query,
            query_id=query_id,
            classified_query=result,
            execution_result=result,
            session=session
        )

        # Step 8: Update query in session history with final results
        await self.session_manager.update_query_in_session(session.session_id, query_context)

        # Step 9: Build and return final result
        return await self._build_final_result(
            session, query, query_id, query_context, result, location_result, classification_result
        )

    # -- Helper Methods --

    async def _execute_parallel_tasks(
            self,
            query: str,
            query_id: str,
            session: Session,
            query_history: List[QueryContext],
            available_context: Dict[str, Any]
    ) -> tuple[Dict[str, Any], Optional[Dict[str, Any]], Dict[str, Any]]:
        """
        Execute classification and location extraction in parallel
        
        Returns:
            (merged_result, location_result, classification_result)
        """
        # Prepare location context for context-aware extraction
        location_context = self._prepare_location_context(query_history, available_context)

        # Create parallel tasks
        classification_task = self._classify_and_extract(
            query, session, query_history, available_context
        )
        location_task = self.location_service.extract_locations(
            query, query_id, session.session_id, location_context
        )

        # Execute in parallel
        classification_result, location_result = await asyncio.gather(
            classification_task,
            location_task,
            return_exceptions=True
        )

        # Handle exceptions
        if isinstance(classification_result, Exception):
            raise ClassificationError(f"Classification failed: {str(classification_result)}")

        if isinstance(location_result, Exception):
            # Location extraction is optional, log warning but continue
            logger.warning(f"Location extraction failed: {location_result}")
            location_result = None

        # Merge results - use location extractor's data
        merged_result = self._merge_results(classification_result, location_result)

        return merged_result, location_result, classification_result

    async def _handle_location_results(
            self,
            result: Dict[str, Any],
            location_result: Optional[Dict[str, Any]],
            query_id: str
    ) -> Dict[str, Any]:
        """
        Handle location results including ambiguities and storage
        
        Returns:
            Updated result dictionary
        """
        # Check for location ambiguities (no geohashes found)
        location_ambiguities = self._check_location_ambiguities(location_result)
        if location_ambiguities:
            if 'ambiguities' not in result:
                result['ambiguities'] = []
            result['ambiguities'].extend(location_ambiguities)

        # Store geohashes in ClickHouse if locations were found
        if location_result and location_result.get('geohashes'):
            try:
                storage_success = await self.geohash_storage.store_geohashes(query_id, location_result)
                if not storage_success:
                    logger.warning(f"Failed to store geohashes for query {query_id}")
            except Exception as e:
                # Storage failure shouldn't break the whole query
                logger.error(f"Geohash storage error: {e}", exc_info=True)

        # If there are ambiguities, request clarification
        if result.get("ambiguities"):
            clarifications = self._generate_clarification_requests(result["ambiguities"])
            # In a real app, this would be an interactive step
            resolved_ambiguities = self._resolve_ambiguities(clarifications)
            # TODO: Make another LLM call with resolved ambiguities if needed

        return result

    async def _build_final_result(
            self,
            session: Session,
            query: str,
            query_id: str,
            query_context: QueryContext,
            result: Dict[str, Any],
            location_result: Optional[Dict[str, Any]],
            classification_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build final result and record training data
        
        Returns:
            Final result dictionary
        """
        # Record training data for parallel approach
        await self._log_parallel_training_data(
            session_id=session.session_id,
            query_id=query_id,
            query_context=query_context,
            classification_result=classification_result,
            location_result=location_result,
            merged_result=result
        )

        # Return final result
        return {
            "session_id": session.session_id,
            "query_id": query_id,
            "query": query,
            "result": result,
            "has_locations": bool(location_result and location_result.get('geohashes'))
        }

    async def _classify_and_extract(
            self, query: str, session: Session,
            query_history: List[QueryContext], available_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Classification and parameter extraction (without location details)"""

        # Build prompt that excludes location extraction
        prompt_data = self.prompt_builder.build_prompt(
            query=query,
            query_history=query_history,
            available_context=available_context,
            include_location_extraction=False  # Don't extract locations here
        )

        # Call LLM
        llm_response, llm_interaction = await self._call_llm(prompt_data)

        # Parse response
        result = self._parse_unified_response(query, llm_response)

        # Store LLM interaction for tracking
        result['_llm_interaction'] = llm_interaction

        return result

    def _merge_results(
            self, classification_result: Dict[str, Any],
            location_result: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Merge classification and location extraction results"""

        # Start with classification result
        merged = classification_result.copy()

        # Remove internal tracking
        merged.pop('_llm_interaction', None)

        # If we have location results, update the entities_detected
        if location_result and location_result.get('locations'):
            if 'entities_detected' not in merged:
                merged['entities_detected'] = {}

            # Replace location data with extractor results
            merged['entities_detected']['locations'] = {
                'extracted_by': 'location_extractor_agent',
                'total_locations': location_result['total_locations'],
                'location_details': location_result['locations'],
                'geohash_count': location_result['geohash_count'],
                'average_confidence': location_result.get('average_confidence', 1.0)
            }

            # Add location metadata to mapped_parameters for SQL generation
            if 'mapped_parameters' not in merged:
                merged['mapped_parameters'] = {}

            # Ensure include/exclude structure exists
            if 'include' not in merged['mapped_parameters']:
                merged['mapped_parameters'] = {
                    'include': merged.get('mapped_parameters', {}),
                    'exclude': {}
                }

            # Add location metadata
            merged['mapped_parameters']['_location_metadata'] = {
                'query_id': location_result.get('query_id'),
                'has_locations': True,
                'location_count': location_result['total_locations']
            }

        return merged

    async def _call_llm(self, prompt_data: Dict[str, str]) -> Tuple[str, LLMInteraction]:
        """Call LLM and track interaction details"""
        start_time = time.time()

        try:
            # Call the actual LLM
            response = await self.llm_client.generate(
                system_prompt=prompt_data["system"],
                user_prompt=prompt_data["user"]
            )
        except Exception as e:
            raise ClassificationError(f"LLM call failed: {str(e)}")

        # Calculate latency
        latency_ms = int((time.time() - start_time) * 1000)

        # Estimate tokens (rough approximation - actual implementation should use tokenizer)
        from .constants import TOKEN_ESTIMATION_DIVISOR
        prompt_tokens = len(prompt_data["system"]) // TOKEN_ESTIMATION_DIVISOR + len(
            prompt_data["user"]) // TOKEN_ESTIMATION_DIVISOR
        completion_tokens = len(response) // TOKEN_ESTIMATION_DIVISOR

        # Create interaction tracking object
        interaction = LLMInteraction(
            model=self.llm_client.config.model,
            prompt_template=prompt_data["system"][:100] + "...",  # First 100 chars
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            temperature=self.llm_client.config.temperature,
            max_tokens=self.llm_client.config.max_tokens
        )

        return response, interaction

    def _build_classification_prompt(self, query: str, session: Session,
                                     query_history: List[QueryContext],
                                     available_context: Dict[str, Any]) -> Dict[str, str]:
        """Build prompt for classification without location extraction"""

        # Check if this is a continuation query
        is_continuation = False
        previous_query = None
        if query_history and available_context.get("mapped_parameters"):
            # Use shared continuation phrases
            if any(phrase in query.lower() for phrase in CONTINUATION_PHRASES):
                is_continuation = True
                previous_query = query_history[0]

        # Build system prompt that excludes location details
        system_prompt = f"""{SCHEMA_AWARE_PROMPT}

## IMPORTANT: Location Handling

DO NOT extract detailed location information. Only identify that locations are mentioned.
Location details (coordinates, geohashes) will be handled by a specialized location service.

When you see location mentions:
- In entities_detected.locations, just note "location_mentioned": true
- Do NOT resolve to coordinates or geohashes
- Do NOT provide geohash lists in mapped_parameters

## CONTEXT-AWARE INSTRUCTIONS

You are processing a {"continuation" if is_continuation else "new"} query.

{"### Previous Query Context:" if is_continuation else ""}
{f"- Previous Query: '{previous_query.query_text}'" if is_continuation and previous_query else ""}
{f"- Previous Category: {self._get_category_value(previous_query.category) if previous_query else 'N/A'}" if is_continuation else ""}
{f"- Previous Parameters: {json.dumps(available_context.get('mapped_parameters', {}), indent=2)}" if is_continuation else ""}

### IMPORTANT RULES FOR CONTINUATION QUERIES:
1. If this is a continuation query (e.g., "how about X"), inherit ALL parameters from the previous query except what's explicitly changed
2. Keep the SAME category as the previous query unless the intent completely changes
3. Only modify the specific parameters mentioned in the new query
4. Preserve all other parameters (time ranges, filters, etc.)

CRITICAL OUTPUT REQUIREMENTS:
1. Return ONLY valid JSON - no markdown, no code blocks, no explanatory text
2. Do NOT wrap the JSON in ```json``` or ``` tags  
3. Do NOT include any text before or after the JSON
4. The response must start with {{ and end with }}
5. The response must be valid, parseable JSON
"""

        # Build user prompt
        user_prompt = f"""
Current Query: "{query}"

{"Previous Query Context:" if query_history else ""}
{json.dumps(available_context.get('previous_queries', []), indent=2) if query_history else ""}

Process this query for classification and parameter extraction.
Remember: Only identify location mentions, don't extract location details.
"""

        return {
            "system": system_prompt,
            "user": user_prompt
        }

    def _build_unified_prompt(self, query: str, session: Session,
                              query_history: List[QueryContext],
                              available_context: Dict[str, Any]) -> Dict[str, str]:
        """Build a unified prompt for both classification and extraction"""

        # Check if this is a continuation query
        is_continuation = False
        previous_query = None
        if query_history and available_context.get("mapped_parameters"):
            # Simple heuristics for continuation queries
            continuation_phrases = ["how about", "what about", "and for", "also show", "instead"]
            if any(phrase in query.lower() for phrase in continuation_phrases):
                is_continuation = True
                previous_query = query_history[0]

        from datetime import datetime, timedelta

        current_date = datetime.now()
        yesterday = current_date - timedelta(days=1)

        # Build system prompt
        system_prompt = f"""{SCHEMA_AWARE_PROMPT}

## CONTEXT-AWARE INSTRUCTIONS

You are processing a {"continuation" if is_continuation else "new"} query.

{"### Previous Query Context:" if is_continuation else ""}
{f"- Previous Query: '{previous_query.query_text}'" if is_continuation and previous_query else ""}
{f"- Previous Category: {previous_query.category.value if previous_query and previous_query.category else 'N/A'}" if is_continuation else ""}
{f"- Previous Parameters: {json.dumps(available_context.get('mapped_parameters', {}), indent=2)}" if is_continuation else ""}

### IMPORTANT RULES FOR CONTINUATION QUERIES:
1. If this is a continuation query (e.g., "how about X"), inherit ALL parameters from the previous query except what's explicitly changed
2. Keep the SAME category as the previous query unless the intent completely changes
3. Only modify the specific parameters mentioned in the new query
4. Preserve all other parameters (time ranges, locations, filters, etc.)

CRITICAL OUTPUT REQUIREMENTS:
1. Return ONLY valid JSON - no markdown, no code blocks, no explanatory text
2. Do NOT wrap the JSON in ```json``` or ``` tags  
3. Do NOT include any text before or after the JSON
4. The response must start with {{ and end with }}
5. The response must be valid, parseable JSON

## CURRENT DATE AND TIME
Today is: {current_date.strftime('%Y-%m-%d')} ({current_date.strftime('%A')})
Current time: {current_date.strftime('%H:%M:%S')}

When converting relative time expressions:
- "yesterday" = {yesterday.strftime('%Y-%m-%d')}
- "today" = {current_date.strftime('%Y-%m-%d')}
- "last week" = {(current_date - timedelta(days=7)).strftime('%Y-%m-%d')} to {yesterday.strftime('%Y-%m-%d')}

Use these actual dates when creating event_timestamp ranges.
"""

        # Build user prompt
        user_prompt = f"""
Current Query: "{query}"

{"Previous Query Context:" if query_history else ""}
{json.dumps(available_context.get('previous_queries', []), indent=2) if query_history else ""}

Process this query and return the complete classification and parameter extraction.
"""

        return {
            "system": system_prompt,
            "user": user_prompt
        }

    def _parse_unified_response(self, original_query: str, llm_response: str) -> Dict[str, Any]:
        """Parse the unified LLM response"""
        from .response_parser import QueryUnderstandingResponseParser
        return QueryUnderstandingResponseParser.parse(llm_response)

    def _parse_classification_result(self, original_query: str, llm_response: str) -> Dict[str, Any]:
        """Parse raw LLM response into structured classification"""
        from .response_parser import QueryUnderstandingResponseParser
        return QueryUnderstandingResponseParser.parse(llm_response)

    def _generate_clarification_requests(self, ambiguities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate clarification requests from detected ambiguities"""
        return [{
            "parameter": item["parameter"],
            "question": item["suggested_clarification"],
            "options": item["options"]
        } for item in ambiguities]

    def _resolve_ambiguities(self, clarifications: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve ambiguities based on user feedback"""
        return {
            item["parameter"]: item["options"][0] if item["options"] else None
            for item in clarifications
        }

    async def _log_parallel_training_data(
            self,
            session_id: str,
            query_id: str,
            query_context: QueryContext,
            classification_result: dict,
            location_result: Optional[dict],
            merged_result: dict
    ):
        """Log training data for the parallel approach"""

        # Get LLM interaction from classification
        llm_interaction = classification_result.get('_llm_interaction')

        # Prepare training data
        training_data = {
            "classification_result": {k: v for k, v in classification_result.items() if k != '_llm_interaction'},
            "location_result": location_result,
            "merged_result": merged_result,
            "parallel_execution": True
        }

        # Remove LLMInteraction from classification_result for JSON serialization
        classification_output = {k: v for k, v in classification_result.items() if k != '_llm_interaction'}

        # Record the classification event
        if llm_interaction:
            await self.session_manager.record_interaction_event(
                session_id=session_id,
                query_id=query_id,
                event_type="parallel_classification",
                llm_interaction=llm_interaction,
                input_data={"query": query_context.query_text},
                output_data=classification_output,
                success=bool(classification_result.get("classification", {}).get("category")),
                confidence=classification_result.get("classification", {}).get("confidence", 0.0)
            )

        # Create training example
        training_example = TrainingExample(
            query_id=query_id,
            query_text=query_context.query_text,
            normalized_query=query_context.normalized_query,
            category=merged_result.get("classification", {}).get("category"),
            extracted_params=merged_result.get("mapped_parameters", {}),
            confidence=merged_result.get("classification", {}).get("confidence", 0.0),
            event_type="parallel_classification_extraction",
            input_data=training_data,
            output_data=merged_result,
            has_positive_feedback=True
        )

        await self.session_manager.record_training_example(training_example)

    def _build_query_context(self, query: str, query_id: str, classified_query: dict,
                             execution_result: dict, session: Session) -> QueryContext:
        """Convert results into QueryContext object"""
        classification_info = classified_query.get("classification", {})

        # Handle new include/exclude format in mapped_parameters
        mapped_params = execution_result.get("mapped_parameters", {})
        if isinstance(mapped_params, dict) and "include" in mapped_params:
            # New format with include/exclude structure
            # For now, merge them for backward compatibility
            # In a real implementation, you'd handle these separately
            merged_params = mapped_params.get("include", {})
            # Note: exclude params would be handled differently in actual query execution
        else:
            # Old format - use as is
            merged_params = mapped_params

        return QueryContext(
            query_id=query_id,
            query_text=query,
            normalized_query=self._normalize_query(query),
            category=QueryCategory(classification_info.get("category")) if classification_info.get(
                "category") else None,
            confidence=classification_info.get("confidence", 0.0),
            inherited_from_query=None,
            inheritance_type=InheritanceType.NONE,
            inherited_elements={},
            extracted_params=mapped_params,  # Store full structure
            active_filters=merged_params,  # Simplified view for filters
            entities_mentioned=classified_query.get("entities_detected", {}),
            status=QueryStatus.COMPLETED,
            result_count=len(merged_params),
            result_entities=[],  # Would come from actual DB results
            execution_time_ms=0
        )

    def _normalize_query(self, query: str) -> str:
        """Normalize query text for storage/training"""
        return " ".join(query.strip().lower().split())

    def _check_location_ambiguities(self, location_result: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Check if location extraction found places but no geohashes"""
        ambiguities = []

        if not location_result or 'locations' not in location_result:
            return ambiguities

        # Check each location for missing geohashes
        for loc_key, loc_data in location_result.get('locations', {}).items():
            if isinstance(loc_data, dict):
                location_name = loc_data.get('name', 'Unknown location')
                geohash_count = loc_data.get('total_geohash_count', 0)
                current_radius = loc_data.get('radius_meters', 0)

                # If location was found but no geohashes, suggest larger radius
                if geohash_count == 0 and len(loc_data.get('coordinates', [])) > 0:
                    from .constants import (
                        SMALL_RADIUS_SUGGESTIONS,
                        MEDIUM_RADIUS_SUGGESTIONS,
                        LARGE_RADIUS_MULTIPLIERS
                    )
                    # Suggest radius options based on current radius
                    if current_radius <= 300:
                        suggested_radii = SMALL_RADIUS_SUGGESTIONS
                    elif current_radius <= 500:
                        suggested_radii = MEDIUM_RADIUS_SUGGESTIONS
                    else:
                        suggested_radii = [current_radius * mult for mult in LARGE_RADIUS_MULTIPLIERS]

                    ambiguities.append({
                        "parameter": f"location_radius_{loc_key}",
                        "issue": f"No data found within {current_radius}m of {location_name}",
                        "suggested_clarification": f"Would you like to search with a larger radius around {location_name}?",
                        "options": [f"{radius}m" for radius in suggested_radii],
                        "location_details": {
                            "name": location_name,
                            "current_radius": current_radius,
                            "coordinates": loc_data.get('coordinates', [])
                        }
                    })

        return ambiguities

    async def _log_unified_training_data(
            self,
            session_id: str,
            query_id: str,
            query_context: QueryContext,
            prompt_data: dict,
            llm_response: str,
            llm_interaction: LLMInteraction,
            result: dict
    ):
        """Log training data for the unified approach"""

        try:
            llm_response_dict = json.loads(llm_response) if llm_response else {}
        except json.JSONDecodeError:
            llm_response_dict = {"raw_response": llm_response[:500]}

        # Log the single interaction event
        await self.session_manager.record_interaction_event(
            session_id=session_id,
            query_id=query_id,
            event_type="unified_classification_extraction",
            llm_interaction=llm_interaction,
            input_data=prompt_data,
            output_data=llm_response_dict,
            success=bool(result.get("classification", {}).get("category")),
            confidence=result.get("classification", {}).get("confidence", 0.0)
        )

        # Save training example
        training_example = TrainingExample(
            query_id=query_id,
            query_text=query_context.query_text,
            normalized_query=query_context.normalized_query,
            category=result.get("classification", {}).get("category"),
            extracted_params=result.get("mapped_parameters", {}),
            confidence=result.get("classification", {}).get("confidence", 0.0),
            event_type="unified_classification_extraction",
            input_data=prompt_data,
            output_data=llm_response_dict,
            has_positive_feedback=True
        )

        await self.session_manager.record_training_example(training_example)

    async def _log_training_data(
            self,
            session_id: str,
            query_id: str,
            query_context: QueryContext,
            prompt_data: dict,
            llm_classification_response: str,
            llm_classification_interaction: LLMInteraction,
            classified_query: dict,
            final_prompt_data: dict,
            llm_execution_response: str,
            llm_execution_interaction: LLMInteraction,
            execution_result: dict
    ):
        """Log all relevant information for future training"""

        try:
            llm_classification_dict = json.loads(llm_classification_response)
        except json.JSONDecodeError:
            llm_classification_dict = {}

        try:
            llm_execution_dict = json.loads(llm_execution_response)
        except json.JSONDecodeError:
            llm_execution_dict = {}

        try:
            llm_classification_dict = json.loads(llm_classification_response)
        except json.JSONDecodeError:
            llm_classification_dict = {}

        # Log classification event
        await self.session_manager.record_interaction_event(
            session_id=session_id,
            query_id=query_id,
            event_type="classification",
            llm_interaction=llm_classification_interaction,
            input_data=prompt_data,
            output_data=llm_classification_dict,
            success=True,
            confidence=classified_query.get("classification", {}).get("confidence", 0.0)
        )

        # Log execution event
        await self.session_manager.record_interaction_event(
            session_id=session_id,
            query_id=query_id,
            event_type="execution",
            llm_interaction=llm_execution_interaction,
            input_data=final_prompt_data,
            output_data=llm_execution_dict,
            success=True
        )

        # Optionally save training example
        training_example = TrainingExample(
            query_id=query_id,
            query_text=query_context.query_text,
            normalized_query=query_context.normalized_query,
            category=classified_query.get("classification", {}).get("category"),
            extracted_params=execution_result.get("mapped_parameters", {}),  # Use execution result params
            confidence=classified_query.get("classification", {}).get("confidence", 0.0),
            event_type="classification",
            input_data=prompt_data,
            output_data=llm_classification_dict,  # Use parsed dict, not string
            has_positive_feedback=True
        )

        # Save training example to DB
        await self.session_manager.record_training_example(training_example)

    async def get_category_performance(self):
        """Get performance metrics by category"""
        return await self.session_manager.get_category_performance()

    async def get_clarification_effectiveness(self):
        """Get metrics about clarification quality"""
        return await self.session_manager.get_clarification_effectiveness()

    def _prepare_location_context(self, query_history: List[QueryContext], available_context: Dict[str, Any]) -> Dict[
        str, Any]:
        """
        Prepare context for location extraction service
        """
        location_context = {}

        if query_history and len(query_history) > 0:
            # Get the most recent query
            recent_query = query_history[0]
            location_context['previous_query'] = recent_query.query_text

            # Extract previous locations from entities_detected
            if recent_query.entities_mentioned and 'locations' in recent_query.entities_mentioned:
                prev_locations = recent_query.entities_mentioned.get('locations', {})

                # Handle both old and new location format
                if isinstance(prev_locations, dict) and 'location_details' in prev_locations:
                    # New format from location extractor
                    location_context['previous_locations'] = prev_locations.get('location_details', {})
                else:
                    # Old format or simple location list
                    location_context['previous_locations'] = prev_locations

            # Also check available_context for recent locations
            if 'entities_detected' in available_context and 'locations' in available_context['entities_detected']:
                # This might have more recent location data
                recent_locations = available_context['entities_detected'].get('locations', {})
                if isinstance(recent_locations, dict) and 'location_details' in recent_locations:
                    location_context['previous_locations'] = recent_locations.get('location_details', {})

        return location_context

    def _extract_available_context(self, query: str, session: Session, query_history: List[QueryContext]) -> Dict[
        str, Any]:
        """Extract available context from session and history"""
        context = {
            "entities_detected": {},
            "mapped_parameters": {},
            "previous_queries": []
        }

        # Extract from session active context
        if session.active_context and isinstance(session.active_context, dict):
            context.update(session.active_context)

        # Extract from recent queries
        if query_history:
            from .constants import MAX_QUERY_HISTORY_SIZE
            for i, qc in enumerate(query_history[:MAX_QUERY_HISTORY_SIZE]):  # Last N queries
                context["previous_queries"].append({
                    "query": qc.query_text,
                    "category": self._get_category_value(qc.category),
                    "entities_detected": qc.entities_mentioned,
                    "mapped_parameters": qc.extracted_params
                })

                # Use most recent query's entities and parameters as active context
                if i == 0:
                    context["entities_detected"] = qc.entities_mentioned or {}
                    context["mapped_parameters"] = qc.extracted_params or {}

                    # Handle both old and new parameter formats
                    if isinstance(qc.extracted_params, dict):
                        if "include" in qc.extracted_params and "exclude" in qc.extracted_params:
                            # New format - pass as is
                            context["mapped_parameters"] = qc.extracted_params
                        else:
                            # Old format - convert to new format for consistency
                            context["mapped_parameters"] = {
                                "include": qc.extracted_params,
                                "exclude": {}
                            }

        return context
