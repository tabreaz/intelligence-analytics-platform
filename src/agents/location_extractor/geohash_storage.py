# src/agents/location_extractor/geohash_storage.py
"""
Storage manager for geohashes in ClickHouse query_location_geohashes table
"""
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

from ...core.logger import get_logger
from ...core.database.clickhouse_client import ClickHouseClient

logger = get_logger(__name__)


class GeohashStorageManager:
    """
    Manages storage of geohashes in ClickHouse for later reference
    """
    
    def __init__(self, clickhouse_client: ClickHouseClient):
        """
        Initialize storage manager
        
        Args:
            clickhouse_client: ClickHouse client instance
        """
        self.ch_client = clickhouse_client
        self.table_name = "telecom_db.query_location_geohashes"
    
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
        """
        Store geohashes for a location in ClickHouse
        
        Args:
            query_id: UUID for the query
            location_index: Index of this location (0, 1, 2...)
            location_name: Name of the location
            geohashes: List of geohash7 strings
            latitude: Center latitude
            longitude: Center longitude
            radius_meters: Radius in meters
            confidence: Confidence score
            
        Returns:
            Number of geohashes stored
        """
        if not geohashes:
            logger.warning(f"No geohashes to store for {location_name}")
            return 0
        
        try:
            # Prepare batch insert data
            rows = []
            created_at = datetime.now()
            
            for geohash7 in geohashes:
                # Extract geohash6 from geohash7
                geohash6 = geohash7[:6] if len(geohash7) >= 6 else geohash7
                
                row = {
                    'query_id': query_id,
                    'location_name': location_name,
                    'location_index': location_index,
                    'geohash7': geohash7,
                    'geohash6': geohash6,
                    'latitude': latitude,
                    'longitude': longitude,
                    'radius_meters': radius_meters,
                    'confidence': confidence,
                    'created_at': created_at,
                    'part_date': created_at.date()
                }
                rows.append(row)
            
            # Batch insert
            query = f"""
            INSERT INTO {self.table_name} 
            (query_id, location_name, location_index, geohash7, geohash6, 
             latitude, longitude, radius_meters, confidence, created_at, part_date)
            VALUES
            """
            
            self.ch_client.insert(self.table_name, rows)
            
            logger.info(
                f"Stored {len(rows)} geohashes for location '{location_name}' "
                f"(index={location_index}) with query_id={query_id}"
            )
            
            return len(rows)
            
        except Exception as e:
            logger.error(f"Failed to store geohashes: {e}")
            raise
    
    async def get_location_geohashes(
        self, 
        query_id: str, 
        location_index: int
    ) -> List[str]:
        """
        Retrieve geohashes for a specific location
        
        Args:
            query_id: Query UUID
            location_index: Location index
            
        Returns:
            List of geohash7 strings
        """
        try:
            query = f"""
            SELECT DISTINCT geohash7
            FROM {self.table_name}
            WHERE query_id = %(query_id)s
              AND location_index = %(location_index)s
            """
            
            params = {
                'query_id': query_id,
                'location_index': location_index
            }
            
            result = self.ch_client.execute(query, params)
            geohashes = [row[0] for row in result]
            
            logger.info(
                f"Retrieved {len(geohashes)} geohashes for "
                f"query_id={query_id}, location_index={location_index}"
            )
            
            return geohashes
            
        except Exception as e:
            logger.error(f"Failed to retrieve geohashes: {e}")
            return []
    
    async def cleanup_old_geohashes(self, days: int = 30) -> int:
        """
        Clean up old geohashes (handled by TTL, but can force cleanup)
        
        Args:
            days: Number of days to keep
            
        Returns:
            Number of partitions dropped
        """
        try:
            # TTL handles this automatically, but we can force cleanup
            query = f"""
            ALTER TABLE {self.table_name} 
            DROP PARTITION WHERE part_date < today() - {days}
            """
            
            self.ch_client.execute(query)
            logger.info(f"Cleaned up geohashes older than {days} days")
            return 1
            
        except Exception as e:
            logger.warning(f"Cleanup failed (may be no old partitions): {e}")
            return 0