# src/agents/query_understanding/status_publisher.py
"""
Status Publisher for real-time query processing updates
"""
import asyncio
import json
from datetime import datetime
from typing import Dict, Optional, Any

import redis.asyncio as redis

from src.core.logger import get_logger

logger = get_logger(__name__)


class QueryStatusPublisher:
    """
    Publishes real-time status updates via Redis pub/sub
    This allows the frontend to show what's happening during query processing
    """

    def __init__(self, redis_config: Dict[str, Any]):
        self.redis_config = redis_config
        self.redis_client: Optional[redis.Redis] = None

        # Status types with user-friendly messages
        self.status_messages = {
            'STARTED': 'Starting query analysis...',
            'CONTEXT_EXTRACTION': 'Extracting context from your query...',
            'BUILDING_PROMPT': 'Preparing intelligent analysis...',
            'CLASSIFICATION': 'Understanding your intent...',
            'CONTEXT_ANALYSIS': 'Analyzing conversation context...',
            'PARAMETER_EXTRACTION': 'Extracting query parameters...',
            'VALIDATION': 'Validating parameters...',
            'CLARIFICATION': 'Preparing clarification questions...',
            'ROUTING': 'Routing to specialized analysis...',
            'EXECUTING': 'Executing analysis...',
            'COMPLETED': 'Analysis complete!',
            'FAILED': 'Analysis failed',

            # Detailed reasoning steps
            'REASONING_CONTEXT': 'Checking if this relates to previous queries...',
            'REASONING_CATEGORY': 'Determining query type...',
            'REASONING_INHERITANCE': 'Applying context from previous query...',
            'REASONING_ENTITIES': 'Identifying entities and references...'
        }

    async def initialize(self):
        """Initialize Redis connection"""
        self.redis_client = await redis.from_url(
            f"redis://{self.redis_config.get('host', 'localhost')}:"
            f"{self.redis_config.get('port', 6379)}",
            decode_responses=True
        )
        logger.info("Redis status publisher initialized")

    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()

    async def publish_status(self,
                             session_id: str,
                             status_type: str,
                             details: Optional[Dict] = None,
                             progress: Optional[int] = None):
        """
        Publish status update to session channel

        Args:
            session_id: User session ID
            status_type: Type of status (from status_messages)
            details: Additional details about the status
            progress: Progress percentage (0-100)
        """
        channel = f"query_status:{session_id}"

        message = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': status_type,
            'message': self.status_messages.get(status_type, status_type),
            'details': details or {},
            'progress': progress
        }

        # Publish to Redis
        await self.redis_client.publish(channel, json.dumps(message))

        # Also store in history for debugging
        history_key = f"query_status_history:{session_id}"
        await self.redis_client.lpush(history_key, json.dumps(message))
        await self.redis_client.ltrim(history_key, 0, 99)  # Keep last 100
        await self.redis_client.expire(history_key, 3600)  # 1 hour TTL

        logger.debug(f"Published status: {status_type} for session {session_id}")

    async def publish_reasoning(self,
                                session_id: str,
                                reasoning_step: str,
                                decision: str,
                                confidence: Optional[float] = None):
        """
        Publish reasoning transparency
        Shows users how the system is thinking
        """
        details = {
            'step': reasoning_step,
            'decision': decision,
            'confidence': confidence,
            'explanation': self._get_explanation(reasoning_step, decision)
        }

        await self.publish_status(
            session_id,
            f'REASONING_{reasoning_step.upper()}',
            details
        )

    def _get_explanation(self, step: str, decision: str) -> str:
        """Get user-friendly explanation for reasoning step"""
        explanations = {
            ('CONTEXT', 'continuation'): "I detected this is a follow-up to your previous question",
            ('CONTEXT', 'standalone'): "I'm treating this as a new question",
            ('CATEGORY', 'location_time'): "You're asking about who visited specific locations",
            ('CATEGORY', 'co_location'): "You're looking for meetings or co-locations",
            ('CATEGORY', 'profile_search'): "You're searching for people by their characteristics",
            ('INHERITANCE', 'full'): "I'm keeping all the context from your previous query",
            ('INHERITANCE', 'partial'): "I'm keeping some context and updating what you specified",
            ('INHERITANCE', 'none'): "Starting fresh with this new query",
            ('ENTITIES', 'pronoun_resolved'): "I understood 'them' refers to the previous results"
        }

        return explanations.get((step, decision), f"{step}: {decision}")

    async def publish_llm_prompt_preview(self,
                                         session_id: str,
                                         prompt_type: str,
                                         context_summary: Dict):
        """
        Publish a preview of what we're sending to the LLM
        This helps with transparency and debugging
        """
        await self.publish_status(
            session_id,
            'BUILDING_PROMPT',
            {
                'prompt_type': prompt_type,
                'context_included': list(context_summary.keys()),
                'has_history': context_summary.get('has_history', False),
                'detected_patterns': context_summary.get('patterns', [])
            }
        )


# Example usage showing real-time updates
async def example_real_time_flow():
    """
    Shows how status updates flow during query processing
    """
    publisher = QueryStatusPublisher({'host': 'localhost', 'port': 6379})
    await publisher.initialize()

    session_id = "123e4567-e89b-12d3-a456-426614174000"

    # Simulate query processing with status updates
    print("=== Real-time Status Updates Flow ===\n")

    # Start
    await publisher.publish_status(session_id, 'STARTED')
    print("→ User sees: 'Starting query analysis...'")
    await asyncio.sleep(0.5)

    # Context extraction
    await publisher.publish_status(
        session_id, 'CONTEXT_EXTRACTION',
        {'found_entities': 3, 'has_history': True},
        progress=20
    )
    print("→ User sees: 'Extracting context from your query...' [20%]")
    await asyncio.sleep(0.5)

    # Reasoning about context
    await publisher.publish_reasoning(
        session_id, 'CONTEXT', 'continuation', confidence=0.85
    )
    print("→ User sees: 'I detected this is a follow-up to your previous question'")
    await asyncio.sleep(0.5)

    # Building prompt
    await publisher.publish_llm_prompt_preview(
        session_id, 'contextual_classification',
        {
            'has_history': True,
            'patterns': ['what_about_pattern'],
            'previous_category': 'location_time',
            'inherited_params': ['location', 'time_range']
        }
    )
    print("→ User sees: 'Preparing intelligent analysis...' [context details]")
    await asyncio.sleep(0.5)

    # Classification
    await publisher.publish_status(
        session_id, 'CLASSIFICATION',
        {'model': 'gpt-4', 'confidence_threshold': 0.7},
        progress=40
    )
    print("→ User sees: 'Understanding your intent...' [40%]")
    await asyncio.sleep(1.0)

    # Reasoning about category
    await publisher.publish_reasoning(
        session_id, 'CATEGORY', 'location_time', confidence=0.95
    )
    print("→ User sees: 'You're asking about who visited specific locations'")
    await asyncio.sleep(0.5)

    # Parameter extraction
    await publisher.publish_status(
        session_id, 'PARAMETER_EXTRACTION',
        {'extracted': 4, 'inherited': 2},
        progress=60
    )
    print("→ User sees: 'Extracting query parameters...' [60%]")
    await asyncio.sleep(0.5)

    # Routing
    await publisher.publish_status(
        session_id, 'ROUTING',
        {'target_agent': 'location_time_analyzer'},
        progress=80
    )
    print("→ User sees: 'Routing to specialized analysis...' [80%]")

    await publisher.close()


if __name__ == "__main__":
    asyncio.run(example_real_time_flow())
