# src/core/location_resolver.py
"""
Improved location resolver with proper async support
"""
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import aiohttp
import asyncio
import geohash2
import logging

logger = logging.getLogger(__name__)

# Default constants for location resolution
DEFAULT_RADIUS_METERS = 500
MIN_RADIUS_METERS = 100
MAX_RADIUS_METERS = 5000
DEFAULT_MAX_RESULTS = 15
GEOHASH7_LENGTH = 7
GEOHASH6_LENGTH = 6

# Radius expansion settings
RADIUS_EXPANSION_FACTOR = 2.0  # Double the radius each time
MAX_EXPANDED_RADIUS = 2000  # Maximum 2km for expansion


@dataclass
class ResolvedLocation:
    """Result of location resolution"""
    name: str
    lat: float
    lng: float
    radius_meters: int
    place_id: Optional[str] = None
    address: Optional[str] = ""
    rating: Optional[float] = 0.0
    existing_geohashes: List[str] = None
    expanded_radius: Optional[int] = None  # Track if radius was expanded

    def __post_init__(self):
        if self.existing_geohashes is None:
            self.existing_geohashes = []


class LocationResolver:
    """
    Async location resolver with improved performance
    """

    def __init__(self, resource_manager, config: Optional[Dict] = None):
        """Initialize location resolver"""
        self.resource_manager = resource_manager
        self.config = config or {}

        # Simplified config lookup
        self.google_api_key = self._get_google_api_key()
        self.max_results = self.config.get('max_results', DEFAULT_MAX_RESULTS)
        self.default_radius = self.config.get('default_radius_meters', DEFAULT_RADIUS_METERS)
        self.geohash_key = "system:geohashes"

        # Radius expansion settings
        self.enable_radius_expansion = self.config.get('enable_radius_expansion', True)
        self.radius_expansion_factor = self.config.get('radius_expansion_factor', RADIUS_EXPANSION_FACTOR)
        self.max_expanded_radius = self.config.get('max_expanded_radius', MAX_EXPANDED_RADIUS)

        # Log configuration
        if self.google_api_key:
            logger.info(f"Google Places API key configured (length: {len(self.google_api_key)})")
        else:
            logger.warning("Google Places API key not found in config")

    def _get_google_api_key(self) -> Optional[str]:
        """Simplified Google API key lookup"""
        # Try multiple config sources
        sources = [
            self.config.get('google_places', {}).get('api_key'),
            self.resource_manager.config_manager.get_agent_config('location_extractor').get('google_places', {}).get(
                'api_key'),
            self.resource_manager.config_manager.get('google_places', {}).get('api_key')
        ]

        for api_key in sources:
            if api_key:
                return api_key
        return None

    @property
    def redis_client(self):
        """Get Redis client from ResourceManager"""
        return self.resource_manager.get_redis_client()

    async def resolve_location(self,
                               location_name: str,
                               radius_meters: Optional[int] = None,
                               coordinates: Optional[Tuple[float, float]] = None) -> List[ResolvedLocation]:
        """
        Async resolve location to coordinates and geohashes
        """
        if radius_meters is None:
            radius_meters = self.default_radius

        results = []

        # If coordinates provided, use them directly
        if coordinates:
            lat, lng = coordinates
            location = ResolvedLocation(
                name=location_name,
                lat=lat,
                lng=lng,
                radius_meters=radius_meters
            )

            # Get geohashes if Redis available
            if self.redis_client:
                geohashes, expanded_radius = await self._get_geohashes_with_expansion(
                    lat, lng, radius_meters
                )
                location.existing_geohashes = geohashes
                if expanded_radius and expanded_radius > radius_meters:
                    location.expanded_radius = expanded_radius

            results.append(location)

        # Otherwise, search using Google Places
        elif self.google_api_key:
            places = await self._search_google_places(location_name)

            for place in places:
                location = ResolvedLocation(
                    name=place['name'],
                    lat=place['lat'],
                    lng=place['lng'],
                    radius_meters=radius_meters,
                    place_id=place.get('place_id'),
                    address=place.get('address', ''),
                    rating=place.get('rating', 0.0)
                )

                # Get geohashes if Redis available
                if self.redis_client:
                    geohashes, expanded_radius = await self._get_geohashes_with_expansion(
                        place['lat'], place['lng'], radius_meters
                    )
                    location.existing_geohashes = geohashes
                    if expanded_radius and expanded_radius > radius_meters:
                        location.expanded_radius = expanded_radius

                results.append(location)
        else:
            logger.warning(f"No resolution method available for location: {location_name}")

        return results

    async def _search_google_places(self, query: str) -> List[Dict]:
        """Async Google Places API search"""
        if not self.google_api_key:
            logger.error("Google Places API key not configured")
            return []

        url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        params = {
            'query': query,
            'key': self.google_api_key
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()

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

                    logger.info(f"Found {len(places)} places for query: {query}")
                    return places

        except Exception as e:
            logger.error(f"Google Places API error for query '{query}': {e}")
            return []

    async def _get_geohashes_in_radius(self, lat: float, lng: float, radius_meters: int) -> List[str]:
        """Async get geohashes from Redis within radius"""
        if not self.redis_client:
            return []

        radius_km = radius_meters / 1000.0

        try:
            # Run Redis operation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: list(self.redis_client.georadius(
                    self.geohash_key, lng, lat, radius_km, unit='km'
                ))
            )

            logger.debug(f"Found {len(results)} geohashes within {radius_meters}m of ({lat}, {lng})")
            return results

        except Exception as e:
            logger.error(f"Redis geohash lookup error: {e}")
            return []

    async def _get_geohashes_with_expansion(self, lat: float, lng: float, radius_meters: int) -> Tuple[
        List[str], Optional[int]]:
        """
        Async get geohashes with smart radius expansion
        """
        # First try with the original radius
        geohashes = await self._get_geohashes_in_radius(lat, lng, radius_meters)

        if geohashes or not self.enable_radius_expansion:
            return geohashes, radius_meters

        # If no geohashes found and expansion is enabled, try expanding radius
        current_radius = radius_meters
        expansion_attempts = 0
        max_attempts = 3  # Limit expansion attempts

        while (not geohashes and
               expansion_attempts < max_attempts and
               current_radius < self.max_expanded_radius):

            # Double the radius
            current_radius = min(
                int(current_radius * self.radius_expansion_factor),
                self.max_expanded_radius
            )

            logger.info(
                f"Expanding search radius from {radius_meters}m to {current_radius}m "
                f"for location ({lat}, {lng})"
            )

            geohashes = await self._get_geohashes_in_radius(lat, lng, current_radius)
            expansion_attempts += 1

            if geohashes:
                logger.info(
                    f"Found {len(geohashes)} geohashes after expanding radius to {current_radius}m"
                )

        if not geohashes:
            logger.warning(
                f"No geohashes found even after expanding radius to {current_radius}m "
                f"for location ({lat}, {lng})"
            )

        return geohashes, current_radius if current_radius != radius_meters else None

    async def get_all_geohashes_for_locations(self, locations: List[Dict]) -> Dict[str, List[str]]:
        """
        Async get all unique geohashes for a list of locations
        """
        if not self.redis_client:
            logger.warning("Redis client not available for geohash lookups")
            return {}

        result = {}

        # Process locations concurrently
        tasks = []
        for loc in locations:
            task = self._get_geohashes_with_expansion(
                loc['lat'],
                loc['lng'],
                loc.get('radius_meters', self.default_radius)
            )
            tasks.append((loc['name'], task))

        # Wait for all tasks to complete
        for name, task in tasks:
            try:
                geohashes, _ = await task
                result[name] = geohashes
            except Exception as e:
                logger.error(f"Failed to get geohashes for {name}: {e}")
                result[name] = []

        return result

    def encode_geohash(self, lat: float, lng: float, precision: int = 7) -> str:
        """Encode coordinates to geohash"""
        return geohash2.encode(lat, lng, precision)

    def decode_geohash(self, geohash: str) -> Tuple[float, float]:
        """Decode geohash to coordinates"""
        return geohash2.decode(geohash)

    async def load_geohashes_from_list(self, geohash_list: List[str], clear_existing: bool = False) -> int:
        """
        Async load geohashes into Redis (for initialization)
        """
        if not self.redis_client:
            logger.error("Redis client not available")
            return 0

        if not geohash_list:
            return 0

        loop = asyncio.get_event_loop()

        def _load_batch():
            # Clear existing data if requested
            if clear_existing:
                self.redis_client.delete(self.geohash_key)

            # Batch insert geohashes with their coordinates
            pipe = self.redis_client.pipeline()
            loaded_count = 0

            for geohash in geohash_list:
                try:
                    # Decode bytes if necessary
                    if isinstance(geohash, bytes):
                        geohash = geohash.decode('utf-8')

                    lat, lng = geohash2.decode(geohash)
                    pipe.geoadd(self.geohash_key, (lng, lat, geohash))
                    loaded_count += 1

                    if loaded_count % 5000 == 0:
                        pipe.execute()
                        pipe = self.redis_client.pipeline()
                        if loaded_count % 50000 == 0:
                            logger.info(f"Loaded {loaded_count:,} geohashes...")

                except Exception as e:
                    logger.warning(f"Invalid geohash {geohash}: {e}")
                    continue

            pipe.execute()
            return loaded_count

        # Run in thread pool
        loaded_count = await loop.run_in_executor(None, _load_batch)
        logger.info(f"Successfully loaded {loaded_count} total geohashes")
        return loaded_count

    async def get_total_geohash_count(self) -> int:
        """Async get total number of loaded geohashes"""
        if not self.redis_client:
            return 0
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.redis_client.zcard, self.geohash_key)
        except Exception as e:
            logger.error(f"Failed to get geohash count: {e}")
            return 0

    async def store_geohash(self, geohash7: str, latitude: float, longitude: float, metadata: Dict = None) -> bool:
        """Async store a single geohash with its coordinates"""
        if not self.redis_client:
            logger.error("Redis client not available")
            return False

        loop = asyncio.get_event_loop()

        def _store():
            try:
                # Add to geo index
                self.redis_client.geoadd(self.geohash_key, (longitude, latitude, geohash7))

                # Store metadata separately if provided
                if metadata:
                    meta_key = f"geohash:meta:{geohash7}"
                    self.redis_client.hset(meta_key, mapping=metadata)
                    self.redis_client.expire(meta_key, 86400 * 30)  # 30 days expiry

                return True
            except Exception as e:
                logger.error(f"Failed to store geohash {geohash7}: {e}")
                return False

        return await loop.run_in_executor(None, _store)