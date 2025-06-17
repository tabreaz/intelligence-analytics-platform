# src/core/annotated_query_logger.py
"""
Logger for annotated queries - logs original and annotated queries as JSON
"""
import json
import logging
from datetime import datetime
from typing import Optional
from src.core.logger import get_logger

# Get the special logger for annotated queries
annotated_logger = logging.getLogger('annotated_queries')

# Ensure we have a default handler if logging isn't configured yet
if not annotated_logger.handlers:
    # Add a simple file handler as fallback
    handler = logging.FileHandler('logs/annotated_queries.log')
    handler.setFormatter(logging.Formatter('%(message)s'))
    annotated_logger.addHandler(handler)
    annotated_logger.setLevel(logging.INFO)


class AnnotatedQueryLogger:
    """
    Logger specifically for annotated queries
    Logs only the input and annotated query as clean JSON
    """
    
    @staticmethod
    def log_annotation(
        original_query: str,
        annotated_query: str,
        query_id: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """
        Log the original and annotated query as JSON
        
        Args:
            original_query: The original user query
            annotated_query: The annotated version with [ENTITY_TYPE:value] tags
            query_id: Optional query ID for tracking
            session_id: Optional session ID for tracking
        """
        try:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "query_id": query_id,
                "session_id": session_id,
                "original": original_query,
                "annotated": annotated_query
            }
            
            # Log as clean JSON (formatter will only output the message)
            annotated_logger.info(json.dumps(log_entry))
            
        except Exception as e:
            # Don't let logging errors break the flow
            logging.getLogger(__name__).error(f"Failed to log annotated query: {e}")