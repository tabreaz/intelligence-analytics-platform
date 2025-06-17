# src/core/activity_logger.py
"""
Activity logging utilities for agents to track their processing steps
"""
from typing import Optional, Dict, Any
import asyncio
from datetime import datetime
from src.core.session_manager_models import QueryContext
from src.core.logger import get_logger
import logging

logger = get_logger(__name__)

# WebSocket manager will be injected
_websocket_manager = None

def set_websocket_manager(manager):
    """Set the global WebSocket manager for activity broadcasting"""
    global _websocket_manager
    _websocket_manager = manager


class ActivityLogger:
    """
    Helper class for agents to log their activities to QueryContext
    """
    
    def __init__(self, agent_name: str, query_context: Optional[QueryContext] = None):
        self.agent_name = agent_name
        self.query_context = query_context
    
    def set_query_context(self, query_context: QueryContext):
        """Update the query context for logging"""
        self.query_context = query_context
    
    def info(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Log an informational activity"""
        if self.query_context:
            self.query_context.add_activity(
                agent_name=self.agent_name,
                message=message,
                is_error=False,
                activity_type="info",
                metadata=metadata
            )
            # Broadcast to WebSocket if available
            self._broadcast_activity(message, "info", False, metadata)
        logger.info(f"[{self.agent_name}] {message}")
    
    def decision(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Log a decision made by the agent"""
        if self.query_context:
            self.query_context.add_activity(
                agent_name=self.agent_name,
                message=message,
                is_error=False,
                activity_type="decision",
                metadata=metadata
            )
            self._broadcast_activity(message, "decision", False, metadata)
        logger.info(f"[{self.agent_name}] Decision: {message}")
    
    def action(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Log an action being performed"""
        if self.query_context:
            self.query_context.add_activity(
                agent_name=self.agent_name,
                message=message,
                is_error=False,
                activity_type="action",
                metadata=metadata
            )
            self._broadcast_activity(message, "action", False, metadata)
        logger.info(f"[{self.agent_name}] Action: {message}")
    
    def warning(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Log a warning"""
        if self.query_context:
            self.query_context.add_activity(
                agent_name=self.agent_name,
                message=message,
                is_error=False,
                activity_type="warning",
                metadata=metadata
            )
            self._broadcast_activity(message, "warning", False, metadata)
        logger.warning(f"[{self.agent_name}] Warning: {message}")
    
    def retry(self, message: str, attempt: int, metadata: Optional[Dict[str, Any]] = None):
        """Log a retry attempt"""
        full_metadata = {"attempt": attempt}
        if metadata:
            full_metadata.update(metadata)
            
        if self.query_context:
            self.query_context.add_activity(
                agent_name=self.agent_name,
                message=f"Retry attempt {attempt}: {message}",
                is_error=False,
                activity_type="retry",
                metadata=full_metadata
            )
            self._broadcast_activity(f"Retry attempt {attempt}: {message}", "retry", False, full_metadata)
        logger.warning(f"[{self.agent_name}] Retry {attempt}: {message}")
    
    def error(self, message: str, error: Optional[Exception] = None, metadata: Optional[Dict[str, Any]] = None):
        """Log an error"""
        full_metadata = {}
        if error:
            full_metadata["error_type"] = type(error).__name__
            full_metadata["error_details"] = str(error)
        if metadata:
            full_metadata.update(metadata)
            
        if self.query_context:
            self.query_context.add_activity(
                agent_name=self.agent_name,
                message=message,
                is_error=True,
                activity_type="error",
                metadata=full_metadata
            )
            self._broadcast_activity(message, "error", True, full_metadata)
        logger.error(f"[{self.agent_name}] Error: {message}", exc_info=error)
    
    def identified(self, what: str, details: Dict[str, Any]):
        """Log what the agent identified/extracted"""
        message = f"Identified {what}"
        if self.query_context:
            self.query_context.add_activity(
                agent_name=self.agent_name,
                message=message,
                is_error=False,
                activity_type="identified",
                metadata={"what": what, "details": details}
            )
            self._broadcast_activity(message, "identified", False, {"what": what, "details": details})
        logger.info(f"[{self.agent_name}] {message}: {details}")
    
    def _broadcast_activity(self, message: str, activity_type: str, is_error: bool, metadata: Optional[Dict[str, Any]] = None):
        """Broadcast activity to WebSocket connections (fire and forget)"""
        global _websocket_manager
        if _websocket_manager and self.query_context and hasattr(self.query_context, 'session_id'):
            activity = {
                "agent_name": self.agent_name,
                "message": message,
                "is_error": is_error,
                "activity_type": activity_type,
                "metadata": metadata or {},
                "timestamp": datetime.utcnow().isoformat()
            }
            # Fire and forget - create task without awaiting
            try:
                asyncio.create_task(
                    _websocket_manager.send_activity(self.query_context.session_id, activity)
                )
            except RuntimeError:
                # If no event loop is running, we can't create a task
                logger.debug("No event loop for WebSocket broadcast - activity not sent")