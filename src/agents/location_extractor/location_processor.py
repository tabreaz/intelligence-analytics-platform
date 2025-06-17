# src/agents/location_extractor/location_processor.py
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Dict

import requests

from src.core.logger import get_logger
from .constants import DEFAULT_RADIUS_METERS, DEFAULT_MAX_RESULTS, MAX_RESPONSE_LOG_LENGTH

logger = get_logger(__name__)


@dataclass
class LocationResult:
    name: str
    lat: float
    lng: float
    radius_meters: int
    existing_geohashes: List[str]
    address: str = ""
    rating: float = 0.0


class LocationProcessor:
    """Process location queries and extract geohashes"""

    def __init__(self, config: Dict, llm_client, redis_manager):
        self.config = config
        self.llm_client = llm_client
        self.redis_manager = redis_manager
        self.google_api_key = config.get('google_places', {}).get('api_key')
        self.max_results = config.get('google_places', {}).get('max_results', DEFAULT_MAX_RESULTS)
        self.default_radius = config.get('default_radius_meters', DEFAULT_RADIUS_METERS)

    async def extract_locations_from_prompt(self, prompt: str) -> tuple[Dict, Dict]:
        """Extract N locations and return existing geohashes from Redis
        
        Returns:
            tuple: (locations_data, llm_metadata) where llm_metadata contains response details
        """

        # Parse locations from prompt using LLM
        locations, llm_metadata = await self._parse_locations_with_llm(prompt)

        if not locations:
            logger.warning("No locations extracted from prompt")
            return {}, llm_metadata

        # Separate locations by type
        city_emirate_locations = []
        facility_locations = []
        
        for i, loc_data in enumerate(locations):
            loc_type = loc_data.get('type', 'FACILITY')
            if loc_type in ['CITY', 'EMIRATE']:
                city_emirate_locations.append((i + 1, loc_data))
            else:
                facility_locations.append((i + 1, loc_data))
        
        results = {}
        
        # Process CITY/EMIRATE locations (no geohash lookup needed)
        for location_num, location_data in city_emirate_locations:
            location_key = f"location{location_num}"
            results[location_key] = {
                "name": location_data['location'],
                "type": location_data['type'],
                "field": location_data.get('field', 'home_city'),
                "value": location_data.get('value', location_data['location']).upper(),
                "confidence": location_data.get('confidence', 0.9),
                "coordinates": [],  # No coordinates for city/emirate
                "total_existing_geohashes": [],  # No geohashes for city/emirate
                "total_geohash_count": 0
            }
            logger.info(f"Processed {location_data['type']}: {location_data['location']} -> field: {location_data.get('field')}")
        
        # Process FACILITY/ADDRESS locations in parallel
        if facility_locations:
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_location = {
                    executor.submit(self._process_location, loc_data): (num, loc_data)
                    for num, loc_data in facility_locations
                }

                for future in future_to_location:
                    location_num, location_data = future_to_location[future]
                    location_key = f"location{location_num}"

                    try:
                        location_results = future.result()

                        if location_results:
                            # Collect all existing geohashes
                            all_existing_geohashes = []
                            for result in location_results:
                                all_existing_geohashes.extend(result.existing_geohashes)

                            # Remove duplicates
                            unique_geohashes = list(set(all_existing_geohashes))

                            results[location_key] = {
                                "name": location_data['location'],
                                "type": location_data.get('type', 'FACILITY'),
                                "field": location_data.get('field', 'geohash'),
                                "value": location_data.get('value', location_data['location']),
                                "radius_meters": location_data.get('radius_meters', self.default_radius),
                                "confidence": location_data.get('confidence', 0.9),
                                "coordinates": [
                                    {
                                        "name": result.name,
                                        "lat": result.lat,
                                        "lng": result.lng,
                                        "radius_meters": result.radius_meters,
                                        "address": result.address,
                                        "rating": result.rating,
                                        "existing_geohash_count": len(result.existing_geohashes)
                                    }
                                    for result in location_results
                                ],
                                "total_existing_geohashes": unique_geohashes,
                                "total_geohash_count": len(unique_geohashes)
                            }

                    except Exception as e:
                        logger.error(f"Error processing {location_data}: {e}")

        return results, llm_metadata

    async def _parse_locations_with_llm(self, prompt: str) -> tuple[List[Dict], Dict]:
        """Extract location information with meter-level radius hints - SINGLE LLM CALL
        
        Returns:
            tuple: (locations, llm_metadata)
        """
        from datetime import datetime
        from .prompt import LOCATION_EXTRACTION_PROMPT

        # Initialize metadata early to avoid reference error
        llm_metadata = {'error': None}
        llm_start_time = datetime.now()
        user_prompt = f"Extract locations from: {prompt}"

        try:
            logger.info(f"Making SINGLE LLM call for extraction and validation: {prompt[:100]}...")

            response = await self.llm_client.generate(
                system_prompt=LOCATION_EXTRACTION_PROMPT,
                user_prompt=user_prompt
            )

            # Update metadata with successful response
            llm_metadata = {
                'llm_response': response,
                'llm_start_time': llm_start_time,
                'prompts': {
                    'system': LOCATION_EXTRACTION_PROMPT,
                    'user': user_prompt
                }
            }

            logger.info(f"LLM response received: {response[:MAX_RESPONSE_LOG_LENGTH]}...")
            print(f"DEBUG - Raw LLM Response: {response}")  # Debug output

            # Use the response parser
            from .response_parser import LocationExtractorResponseParser
            parsed_result = LocationExtractorResponseParser.parse(response)
            print(f"DEBUG - Parsed result: {parsed_result}")  # Debug output

            # Extract locations and ambiguities
            parsed_locations = parsed_result.get('locations', [])
            parsed_ambiguities = parsed_result.get('ambiguities', [])
            
            # Add ambiguities to metadata
            llm_metadata['ambiguities'] = parsed_ambiguities

            if not parsed_locations and not parsed_ambiguities:
                logger.info("No locations or ambiguities found by LLM - this is valid (not an error)")
                return [], llm_metadata

            logger.info(f"Successfully parsed {len(parsed_locations)} locations and {len(parsed_ambiguities)} ambiguities")

            # Log confidence scores
            for loc in parsed_locations:
                confidence = loc.get('confidence', 0.0)
                logger.info(f"Location: {loc.get('location', 'Unknown')} - Confidence: {confidence}")
            
            # Log ambiguities
            for amb in parsed_ambiguities:
                logger.info(f"Ambiguity: {amb.get('reference')} - Type: {amb.get('ambiguity_type')} - Severity: {amb.get('severity')}")

            return parsed_locations, llm_metadata

        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            # Update metadata with error information
            llm_metadata['error'] = str(e)
            llm_metadata['llm_start_time'] = llm_start_time
            llm_metadata['prompts'] = {
                'system': LOCATION_EXTRACTION_PROMPT,
                'user': user_prompt
            }
            return [], llm_metadata

    def _process_location(self, location_data: Dict) -> List[LocationResult]:
        """Process location and get existing geohashes from Redis"""

        location_name = location_data['location']
        radius_meters = location_data.get('radius_meters', self.default_radius)

        # Get all matching places from Google
        places = self._google_search_all_locations(location_name)

        results = []
        for place in places:
            # Get existing geohashes from Redis within radius
            existing_geohashes = self.redis_manager.get_geohashes_in_radius(
                place['lat'], place['lng'], radius_meters
            )

            result = LocationResult(
                name=place['name'],
                lat=place['lat'],
                lng=place['lng'],
                radius_meters=radius_meters,
                existing_geohashes=existing_geohashes,
                address=place.get('address', ''),
                rating=place.get('rating', 0.0)
            )
            results.append(result)

        return results

    def _google_search_all_locations(self, query: str) -> List[Dict]:
        """Search Google Places"""

        if not self.google_api_key:
            logger.error("Google Places API key not configured")
            return []

        url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        params = {
            'query': query,
            'key': self.google_api_key
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            places = []
            for place in data.get('results', [])[:self.max_results]:
                places.append({
                    'name': place['name'],
                    'lat': place['geometry']['location']['lat'],
                    'lng': place['geometry']['location']['lng'],
                    'place_id': place['place_id'],
                    'address': place.get('formatted_address', ''),
                    'rating': place.get('rating', 0.0)
                })

            return places

        except Exception as e:
            logger.error(f"Google Places API error: {e}")
            return []
