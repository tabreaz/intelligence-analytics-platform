# src/agents/location_extractor/redis_manager.py
from typing import List, Dict

import geohash2
import redis

from src.core.logger import get_logger

logger = get_logger(__name__)


class RedisGeohashManager:
    """Utility for loading and searching geohashes in Redis"""

    def __init__(self, redis_config: Dict):
        self.redis_client = redis.Redis(
            host=redis_config['host'],
            port=redis_config['port'],
            db=redis_config['db'],
            password=redis_config.get('password'),
            ssl=redis_config.get('ssl', False),
            decode_responses=True
        )
        self.geohash_key = "system:geohashes"

    def load_geohashes_from_list(self, geohash_list: List[str], clear_existing: bool = False) -> int:
        """Load geohashes from a list into Redis"""
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

    def get_geohashes_in_radius(self, center_lat: float, center_lng: float,
                                radius_meters: int) -> List[str]:
        """Get existing geohashes within radius"""
        radius_km = radius_meters / 1000.0

        try:
            # Cast to list to ensure it's iterable
            results = list(self.redis_client.georadius(
                self.geohash_key,
                center_lng, center_lat,
                radius_km,
                unit='km'
            ))

            if not results:
                logger.debug(f"No geohashes found within {radius_km}km of ({center_lat}, {center_lng})")
                return []

            # Ensure all results are strings, not bytes
            decoded_results = []
            for result in results:
                if isinstance(result, bytes):
                    decoded_results.append(result.decode('utf-8'))
                else:
                    decoded_results.append(str(result))

            logger.debug(f"Found {len(decoded_results)} geohashes within {radius_km}km")
            return decoded_results
        except Exception as e:
            logger.error(f"Redis georadius query failed: {e}")
            return []

    def get_total_geohash_count(self) -> int:
        """Get total number of loaded geohashes"""
        return self.redis_client.zcard(self.geohash_key)

    def store_geohash(self, geohash7: str, latitude: float, longitude: float, metadata: Dict = None) -> bool:
        """Store a single geohash with its coordinates"""
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
