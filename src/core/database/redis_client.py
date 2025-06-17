# src/core/database/redis_client.py
import redis
from typing import Any, Optional
import logging
from src.core.logger import get_logger
from ..config_manager import DatabaseConfig

logger = get_logger(__name__)


class RedisClient:
    """Redis database client"""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.client = None
        self._connect()

    def _connect(self):
        """Connect to Redis"""
        try:
            self.client = redis.Redis(
                host=self.config.host,
                port=self.config.port,
                db=self.config.database,
                password=self.config.password,
                decode_responses=True,
                socket_timeout=self.config.timeout
            )

            # Test connection
            self.client.ping()
            logger.info(f"Connected to Redis: {self.config.host}:{self.config.port}")

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def get(self, key: str) -> Optional[str]:
        """Get value by key"""
        return self.client.get(key)

    def set(self, key: str, value: Any, ex: Optional[int] = None):
        """Set key-value with optional expiration"""
        return self.client.set(key, value, ex=ex)

    def delete(self, *keys):
        """Delete keys"""
        return self.client.delete(*keys)