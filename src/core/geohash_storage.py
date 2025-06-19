# src/core/geohash_storage.py
"""
Clean geohash storage manager using HTTP interface
Removed all broken async client fallbacks
"""
import aiohttp
from typing import List, Dict
from datetime import datetime
from src.core.logger import get_logger

logger = get_logger(__name__)


class GeohashStorageManager:
    """Simple, reliable geohash storage using ClickHouse HTTP interface"""

    def __init__(self, clickhouse_client=None, ch_host='localhost', ch_port=8123):
        """
        Initialize storage manager

        Args:
            clickhouse_client: Not used (kept for compatibility)
            ch_host: ClickHouse host
            ch_port: ClickHouse HTTP port
        """
        self.table_name = "telecom_db.query_location_geohashes"
        self.ch_url = f'http://{ch_host}:{ch_port}'

    async def store_location_geohashes(
            self,
            query_id: str,
            location_index: int,
            location_name: str,
            geohashes: List[str],
            latitude: float,
            longitude: float,
            radius_meters: int,
            confidence: float = 1.0
    ) -> int:
        """Store geohashes in batch via HTTP"""
        if not geohashes:
            return 0

        # Prepare batch insert data
        created_at = datetime.now()
        values = []

        for geohash7 in geohashes:
            geohash6 = geohash7[:6] if len(geohash7) >= 6 else geohash7
            values.append(f"""(
                '{query_id}',
                '{location_name.replace("'", "''")}',
                {location_index},
                '{geohash7}',
                '{geohash6}',
                {latitude},
                {longitude},
                {radius_meters},
                {confidence},
                '{created_at.strftime('%Y-%m-%d %H:%M:%S')}',
                '{created_at.date()}'
            )""")

        # Single batch insert
        query = f"""
        INSERT INTO {self.table_name} 
        (query_id, location_name, location_index, geohash7, geohash6, 
         latitude, longitude, radius_meters, confidence, created_at, part_date)
        VALUES {', '.join(values)}
        """

        # Execute via HTTP
        async with aiohttp.ClientSession() as session:
            async with session.post(self.ch_url, data=query) as response:
                if response.status != 200:
                    error = await response.text()
                    raise Exception(f"ClickHouse insert failed: {error}")

        logger.info(f"Stored {len(geohashes)} geohashes for {location_name}")
        return len(geohashes)

    async def store_multiple_locations_batch(self, query_id: str, locations: List[Dict]) -> int:
        """Store multiple locations in single batch"""
        if not locations:
            return 0

        created_at = datetime.now()
        all_values = []
        total_count = 0

        for loc in locations:
            for geohash7 in loc.get('geohashes', []):
                geohash6 = geohash7[:6] if len(geohash7) >= 6 else geohash7
                all_values.append(f"""(
                    '{query_id}',
                    '{loc["location_name"].replace("'", "''")}',
                    {loc['location_index']},
                    '{geohash7}',
                    '{geohash6}',
                    {loc['latitude']},
                    {loc['longitude']},
                    {loc['radius_meters']},
                    {loc.get('confidence', 1.0)},
                    '{created_at.strftime('%Y-%m-%d %H:%M:%S')}',
                    '{created_at.date()}'
                )""")
                total_count += 1

        if not all_values:
            return 0

        query = f"""
        INSERT INTO {self.table_name} 
        (query_id, location_name, location_index, geohash7, geohash6, 
         latitude, longitude, radius_meters, confidence, created_at, part_date)
        VALUES {', '.join(all_values)}
        """

        async with aiohttp.ClientSession() as session:
            async with session.post(self.ch_url, data=query) as response:
                if response.status != 200:
                    error = await response.text()
                    raise Exception(f"ClickHouse batch insert failed: {error}")

        logger.info(f"Batch stored {total_count} geohashes for {len(locations)} locations")
        return total_count

    async def get_location_geohashes(self, query_id: str, location_index: int) -> List[str]:
        """Get geohashes for specific location"""
        query = f"""
        SELECT DISTINCT geohash7 
        FROM {self.table_name} 
        WHERE query_id = '{query_id}' AND location_index = {location_index}
        FORMAT TabSeparated
        """

        async with aiohttp.ClientSession() as session:
            async with session.post(self.ch_url, data=query) as response:
                if response.status != 200:
                    error = await response.text()
                    raise Exception(f"ClickHouse query failed: {error}")

                result = await response.text()
                geohashes = [line.strip() for line in result.strip().split('\n') if line.strip()]

                logger.info(f"Retrieved {len(geohashes)} geohashes for query_id={query_id}")
                return geohashes

    async def cleanup_old_geohashes(self, days: int = 30) -> int:
        """Clean up old geohashes"""
        query = f"""
        ALTER TABLE {self.table_name} 
        DROP PARTITION WHERE part_date < today() - {days}
        """

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.ch_url, data=query) as response:
                    if response.status == 200:
                        logger.info(f"Cleaned up geohashes older than {days} days")
                        return 1
                    else:
                        error = await response.text()
                        logger.warning(f"Cleanup failed: {error}")
                        return 0
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
            return 0