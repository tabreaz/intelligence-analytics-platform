# src/core/logger.py
import logging
import logging.config
from typing import Dict, Any


def setup_logging(logging_config: Dict[str, Any]):
    """Setup logging configuration"""

    # Create logs directory if it doesn't exist
    import os
    os.makedirs('logs', exist_ok=True)

    # Configure logging
    logging.config.dictConfig(logging_config)

    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info("Logging system initialized")


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance

    Args:
        name (str): Name of the logger (typically __name__)

    Returns:
        logging.Logger: Configured logger
    """
    return logging.getLogger(name)