"""
Query Executor Agent - Generates and executes queries across multiple engines
Supports: ClickHouse, PySpark/Spark SQL (future), PostgreSQL (future), Elasticsearch (future)
"""

from .agent import QueryExecutorAgent
from .models import (
    QueryExecutorRequest,
    QueryExecutorResult,
    EngineType,
    QueryType,
    ExecutionMode
)

__all__ = [
    'QueryExecutorAgent',
    'QueryExecutorRequest',
    'QueryExecutorResult',
    'EngineType',
    'QueryType',
    'ExecutionMode'
]