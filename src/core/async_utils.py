# src/core/async_utils.py
"""
Async utilities for fire-and-forget patterns and background tasks
"""
import asyncio
import functools
from typing import Callable, Any, Optional
from src.core.logger import get_logger

logger = get_logger(__name__)


def fire_and_forget(func: Callable) -> Callable:
    """
    Decorator to run an async function as a fire-and-forget background task
    
    Usage:
        @fire_and_forget
        async def log_data(...):
            await database.save(...)
            
        # Call it normally - won't block
        await log_data(...)  # Returns immediately
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        async def safe_execute():
            try:
                await func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"Background task {func.__name__} failed: {e}", 
                    exc_info=True
                )
        
        # Create task and don't wait for it
        asyncio.create_task(safe_execute())
        # Return immediately
        return None
    
    return wrapper


class BackgroundTaskManager:
    """
    Manager for background tasks with proper cleanup
    """
    def __init__(self):
        self.tasks = set()
        
    def create_task(self, coro, name: Optional[str] = None):
        """
        Create a background task and track it
        """
        task = asyncio.create_task(coro)
        if name:
            task.set_name(name)
        
        # Add to set and remove when done
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        
        return task
    
    async def shutdown(self, timeout: float = 10.0):
        """
        Gracefully shutdown all pending tasks
        """
        if not self.tasks:
            return
            
        logger.info(f"Shutting down {len(self.tasks)} background tasks...")
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
            
        # Wait for all tasks to complete with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*self.tasks, return_exceptions=True),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"Some tasks didn't complete within {timeout}s timeout")


# Global task manager instance
background_tasks = BackgroundTaskManager()


def create_background_task(coro, name: Optional[str] = None):
    """
    Create a tracked background task
    
    Args:
        coro: Coroutine to run
        name: Optional name for the task
        
    Returns:
        asyncio.Task: The created task
    """
    return background_tasks.create_task(coro, name)