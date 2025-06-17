# src/core/training_logger.py
"""
Centralized training data and interaction logging utilities
"""
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
from src.core.session_manager_models import LLMInteraction, TrainingExample
from src.core.logger import get_logger

logger = get_logger(__name__)


class TrainingLogger:
    """
    Centralized logger for LLM interactions and training data
    """
    
    def __init__(self, session_manager=None):
        self.session_manager = session_manager
        self.background_tasks = set()
    
    async def log_llm_interaction(
        self,
        session_id: Optional[str],
        query_id: str,
        query_text: str,
        event_type: str,
        llm_response: str,
        llm_start_time: datetime,
        prompts: Dict[str, str],
        result: Dict[str, Any],
        model_config: Dict[str, Any],
        extracted_params: Optional[Dict[str, Any]] = None,
        category: Optional[str] = None,
        confidence: Optional[float] = None,
        success: Optional[bool] = None
    ):
        """
        Log LLM interaction and training data in one call
        
        Args:
            session_id: Session ID (optional)
            query_id: Query ID
            query_text: Original query text
            event_type: Type of event (e.g., "query_classification", "time_parsing")
            llm_response: Raw LLM response
            llm_start_time: When LLM call started
            prompts: Dict with "system" and "user" prompts
            result: Complete result dict
            model_config: Dict with model configuration (model, temperature, max_tokens)
            extracted_params: Parameters extracted (optional, defaults to result)
            category: Category for training (optional, extracted from result)
            confidence: Confidence score (optional, extracted from result)
            success: Success flag (optional, defaults to True)
        """
        if not self.session_manager:
            return
            
        try:
            # Calculate timing
            llm_end_time = datetime.now()
            llm_duration = (llm_end_time - llm_start_time).total_seconds() * 1000
            
            # Estimate tokens
            TOKEN_ESTIMATION_DIVISOR = 4
            system_prompt = prompts.get("system", "")
            user_prompt = prompts.get("user", "")
            prompt_tokens = (len(system_prompt) + len(user_prompt)) // TOKEN_ESTIMATION_DIVISOR
            completion_tokens = len(llm_response) // TOKEN_ESTIMATION_DIVISOR
            
            # Combine full prompts
            full_prompt = f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}"
            
            # Create LLM interaction
            llm_interaction = LLMInteraction(
                model=model_config.get('model', 'unknown'),
                prompt_template=full_prompt,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=int(llm_duration),
                temperature=model_config.get('temperature', 0.1),
                max_tokens=model_config.get('max_tokens', 1000)
            )
            
            # Prepare input data
            input_data = {
                "query": query_text,
                "prompts": prompts,
                "event_type": event_type
            }
            
            # Extract category and confidence if not provided
            if category is None and 'classification' in result:
                category = result['classification'].get('category')
            if confidence is None:
                if 'classification' in result:
                    confidence = result['classification'].get('confidence', 0.5)
                elif 'parsing_confidence' in result:
                    confidence = result.get('parsing_confidence', 0.5)
                else:
                    confidence = 0.5
            
            # Default success if not provided
            if success is None:
                if 'domain_check' in result:
                    success = result['domain_check'].get('is_supported', True)
                elif 'has_time_expressions' in result:
                    success = result.get('has_time_expressions', True)
                else:
                    success = True
            
            # Log interaction event
            if session_id:
                await self.session_manager.record_interaction_event(
                    session_id=session_id,
                    query_id=query_id,
                    event_type=event_type,
                    llm_interaction=llm_interaction,
                    input_data=input_data,
                    output_data=result,
                    success=success,
                    confidence=confidence
                )
            
            # Create training example
            training_example = TrainingExample(
                query_id=query_id,
                query_text=query_text,
                normalized_query=query_text.lower().strip(),
                category=category or event_type,
                extracted_params=extracted_params or result,
                confidence=confidence,
                event_type=event_type,
                input_data=input_data,
                output_data=result,
                has_positive_feedback=success
            )
            
            # Save training example
            await self.session_manager.record_training_example(training_example)
            
            logger.debug(f"Training data logged for query: {query_id}")
            
        except Exception as e:
            logger.error(f"Failed to log training data: {e}", exc_info=True)
    
    def log_llm_interaction_background(
        self,
        session_id: Optional[str],
        query_id: str,
        query_text: str,
        event_type: str,
        llm_response: str,
        llm_start_time: datetime,
        prompts: Dict[str, str],
        result: Dict[str, Any],
        model_config: Dict[str, Any],
        **kwargs
    ):
        """
        Log LLM interaction in background (fire-and-forget)
        
        Same parameters as log_llm_interaction
        """
        # Create background task
        task = asyncio.create_task(
            self.log_llm_interaction(
                session_id, query_id, query_text, event_type,
                llm_response, llm_start_time, prompts, result,
                model_config, **kwargs
            )
        )
        
        # Track task
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
        
        # Set task name for debugging
        task.set_name(f"log_{event_type}_{query_id}")
        
        return task
    
    async def wait_for_background_tasks(self, timeout: float = 10.0):
        """
        Wait for all background logging tasks to complete
        
        Args:
            timeout: Maximum time to wait in seconds
        """
        if not self.background_tasks:
            return
            
        try:
            await asyncio.wait_for(
                asyncio.gather(*self.background_tasks, return_exceptions=True),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"Some logging tasks didn't complete within {timeout}s")