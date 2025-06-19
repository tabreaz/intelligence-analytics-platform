# src/core/location_resolver.py
"""
Shared utility for resolving location names to coordinates and geohashes
Used by both LocationExtractorAgent and MovementAgent
"""
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import requests
import geohash2
import logging

logger = logging.getLogger(__name__)


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
    
    def __post_init__(self):
        if self.existing_geohashes is None:
            self.existing_geohashes = []


class LocationResolver:
    """
    Resolves location names to coordinates and geohashes
    Supports multiple resolution methods:
    1. Google Places API
    2. Local database/cache
    3. Fallback to coordinates if provided
    """
    
    def __init__(self, resource_manager, config: Optional[Dict] = None):
        """
        Initialize location resolver using ResourceManager
        
        Args:
            resource_manager: ResourceManager instance for accessing shared resources
            config: Optional configuration override (defaults to resource_manager's config)
        """
        self.resource_manager = resource_manager
        
        # Use provided config or get from resource manager
        if config:
            self.config = config
        else:
            # Get config from resource manager's config manager
            self.config = resource_manager.config_manager.get('location_resolver', {})
        
        # Get Google Places config
        google_config = self.config.get('google_places', {})
        if not google_config:
            # Fallback to main google places config
            google_config = resource_manager.config_manager.get('google_places', {})
            
        self.google_api_key = google_config.get('api_key')
        self.max_results = google_config.get('max_results', 5)
        self.default_radius = self.config.get('default_radius_meters', 500)
        self.geohash_key = "system:geohashes"
        
    @property
    def redis_client(self):
        """Get Redis client from ResourceManager"""
        return self.resource_manager.get_redis_client()
        
    def resolve_location(self, 
                        location_name: str,
                        radius_meters: Optional[int] = None,
                        coordinates: Optional[Tuple[float, float]] = None) -> List[ResolvedLocation]:
        """
        Resolve a location name to coordinates and optionally get geohashes
        
        Args:
            location_name: Name of the location to resolve
            radius_meters: Radius for geohash search (uses default if None)
            coordinates: Optional (lat, lng) tuple if already known
            
        Returns:
            List of resolved locations with coordinates and geohashes
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
                location.existing_geohashes = self._get_geohashes_in_radius(
                    lat, lng, radius_meters
                )
            
            results.append(location)
            
        # Otherwise, search using Google Places
        elif self.google_api_key:
            places = self._search_google_places(location_name)
            
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
                    location.existing_geohashes = self._get_geohashes_in_radius(
                        place['lat'], place['lng'], radius_meters
                    )
                
                results.append(location)
        else:
            logger.warning(f"No resolution method available for location: {location_name}")
            
        return results
    
    def _search_google_places(self, query: str) -> List[Dict]:
        """Search Google Places API"""
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
                
            logger.info(f"Found {len(places)} places for query: {query}")
            return places
            
        except Exception as e:
            logger.error(f"Google Places API error for query '{query}': {e}")
            return []
    
    def _get_geohashes_in_radius(self, lat: float, lng: float, radius_meters: int) -> List[str]:
        """Get existing geohashes from Redis within radius"""
        if not self.redis_client:
            return []
            
        radius_km = radius_meters / 1000.0
        
        try:
            # Get geohashes within radius
            results = list(self.redis_client.georadius(
                self.geohash_key,
                lng, lat,
                radius_km,
                unit='km'
            ))
            
            logger.debug(f"Found {len(results)} geohashes within {radius_meters}m of ({lat}, {lng})")
            return results
            
        except Exception as e:
            logger.error(f"Redis geohash lookup error: {e}")
            return []
    
    def get_all_geohashes_for_locations(self, locations: List[Dict]) -> Dict[str, List[str]]:
        """
        Get all unique geohashes for a list of locations
        
        Args:
            locations: List of location dicts with 'name', 'lat', 'lng', 'radius_meters'
            
        Returns:
            Dict mapping location names to their geohashes
        """
        if not self.redis_client:
            logger.warning("Redis client not available for geohash lookups")
            return {}
            
        result = {}
        
        for loc in locations:
            geohashes = self._get_geohashes_in_radius(
                loc['lat'], 
                loc['lng'], 
                loc.get('radius_meters', self.default_radius)
            )
            result[loc['name']] = geohashes
            
        return result
    
    def encode_geohash(self, lat: float, lng: float, precision: int = 7) -> str:
        """Encode coordinates to geohash"""
        return geohash2.encode(lat, lng, precision)
    
    def decode_geohash(self, geohash: str) -> Tuple[float, float]:
        """Decode geohash to coordinates"""
        return geohash2.decode(geohash)
    
    def load_geohashes_from_list(self, geohash_list: List[str], clear_existing: bool = False) -> int:
        """
        Load geohashes into Redis (for initialization)
        
        Args:
            geohash_list: List of geohashes to load
            clear_existing: Whether to clear existing geohashes first
            
        Returns:
            Number of geohashes loaded
        """
        if not self.redis_client:
            logger.error("Redis client not available")
            return 0
            
        if not geohash_list:
            return 0
            
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
        logger.info(f"Successfully loaded {loaded_count} total geohashes")
        return loaded_count
    
    def get_total_geohash_count(self) -> int:
        """Get total number of loaded geohashes"""
        if not self.redis_client:
            return 0
        try:
            return self.redis_client.zcard(self.geohash_key)
        except Exception as e:
            logger.error(f"Failed to get geohash count: {e}")
            return 0
    
    def store_geohash(self, geohash7: str, latitude: float, longitude: float, metadata: Dict = None) -> bool:
        """Store a single geohash with its coordinates"""
        if not self.redis_client:
            logger.error("Redis client not available")
            return False
            
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