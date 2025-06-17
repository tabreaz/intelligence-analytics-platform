"""
Query engine implementations
"""

from .base import QueryEngine, FilterTranslator
from .clickhouse import ClickHouseEngine
# TODO: Add Spark engine when implemented
# from .spark import SparkEngine

__all__ = [
    'QueryEngine',
    'FilterTranslator',
    'ClickHouseEngine'
    # 'SparkEngine'
]