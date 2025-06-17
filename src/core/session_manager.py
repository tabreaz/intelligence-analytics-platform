# src/core/session_manager.py
"""
Session management with PostgreSQL backend
"""
import json
import logging
from src.core.logger import get_logger
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import UUID

import asyncpg
from asyncpg.pool import Pool

from .session_manager_models import (
    Session, QueryContext, LLMInteraction, TrainingExample,
    QueryStatus, InheritanceType
)

logger = get_logger(__name__)


class PostgreSQLSessionManager:
    """
    Manages user sessions with PostgreSQL backend
    Note: Updated to use public schema instead of intelligence_sessions
    """
    
    def __init__(self, db_config: Dict):
        self.db_config = db_config
        self.db_pool: Optional[Pool] = None
        self.cache = {}  # Local cache for active sessions
        
    async def initialize(self):
        """Initialize database connection pool"""
        self.db_pool = await asyncpg.create_pool(
            host=self.db_config.get('host', 'localhost'),
            port=self.db_config.get('port', 5432),
            user=self.db_config.get('user', 'tabreaz'),
            password=self.db_config.get('password', 'admin'),
            database=self.db_config.get('database', 'sigint'),
            min_size=5,
            max_size=20,
            command_timeout=60,
            init=self._init_connection
        )
        logger.info("PostgreSQL connection pool initialized")
        
        # Run maintenance on startup
        await self.update_session_active_status()
    
    async def _init_connection(self, conn):
        """Initialize each connection with JSON codec"""
        await conn.set_type_codec(
            'jsonb',
            encoder=json.dumps,
            decoder=json.loads,
            schema='pg_catalog'
        )
    
    async def close(self):
        """Close database connection pool"""
        if self.db_pool:
            await self.db_pool.close()
            logger.info("PostgreSQL connection pool closed")
    
    async def get_or_create_session(self, session_id: Optional[str] = None) -> Session:
        """Get existing session or create new one"""
        if not session_id:
            return await self.create_session()
        
        # Check cache first
        if session_id in self.cache:
            session = self.cache[session_id]
            if not session.is_expired():
                await self.touch_session(session_id)
                return session
            else:
                # Remove expired session from cache
                del self.cache[session_id]
        
        # Load from database
        session = await self.load_session(session_id)
        if session and not session.is_expired():
            self.cache[session_id] = session
            await self.touch_session(session_id)
            return session
        
        # Create new if not found or expired
        return await self.create_session()
    
    async def create_session(self) -> Session:
        """Create new session"""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                INSERT INTO public.sessions (
                    created_at, last_activity, expires_at, is_active
                ) VALUES (
                    NOW(), NOW(), NOW() + INTERVAL '2 hours', true
                ) RETURNING *
            """)
            
            session = Session(
                session_id=str(row['session_id']),
                created_at=row['created_at'],
                last_activity=row['last_activity'],
                expires_at=row['expires_at'],
                total_queries=row['total_queries'],
                active_context=row['active_context'],
                user_preferences=row['user_preferences'],
                session_metadata=row['session_metadata'],
                is_active=row['is_active']
            )
            
            self.cache[session.session_id] = session
            logger.info(f"Created new session: {session.session_id}")
            return session
    
    async def load_session(self, session_id: str) -> Optional[Session]:
        """Load session from database"""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM public.sessions 
                WHERE session_id = $1 AND is_active = true
            """, session_id)
            
            if not row:
                return None
            
            return Session(
                session_id=str(row['session_id']),
                created_at=row['created_at'],
                last_activity=row['last_activity'],
                expires_at=row['expires_at'],
                total_queries=row['total_queries'],
                active_context=row['active_context'],
                user_preferences=row['user_preferences'],
                session_metadata=row['session_metadata'],
                is_active=row['is_active']
            )
    
    async def touch_session(self, session_id: str):
        """Update session last activity"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE public.sessions 
                SET last_activity = NOW()
                WHERE session_id = $1
            """, session_id)
    
    async def update_session_context(self, session_id: str, context: Dict[str, Any]):
        """Update session active context"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE public.sessions 
                SET active_context = $2,
                    last_activity = NOW()
                WHERE session_id = $1
            """, session_id, json.dumps(context))
        
        # Update cache
        if session_id in self.cache:
            self.cache[session_id].active_context = context
    
    async def add_query_to_session(self, 
                                  session_id: str,
                                  query_context: QueryContext) -> str:
        """Add query to session history and return query_id"""
        async with self.db_pool.acquire() as conn:
            # Get sequence number
            seq_num = await conn.fetchval("""
                SELECT COALESCE(MAX(sequence_number), 0) + 1
                FROM public.query_history
                WHERE session_id = $1
            """, session_id)
            
            # Insert query with provided query_id
            await conn.execute("""
                INSERT INTO public.query_history (
                    query_id, session_id, sequence_number, query_text, normalized_query,
                    category, confidence, inherited_from_query, inheritance_type,
                    inherited_elements, extracted_params, active_filters,
                    entities_mentioned, status, result_count, execution_time_ms
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
            """, UUID(query_context.query_id), session_id, seq_num, query_context.query_text,
                query_context.normalized_query, 
                query_context.category.value if query_context.category else None,
                query_context.confidence, query_context.inherited_from_query,
                query_context.inheritance_type.value, 
                json.dumps(query_context.inherited_elements),
                json.dumps(query_context.extracted_params),
                json.dumps(query_context.active_filters),
                json.dumps(query_context.entities_mentioned),
                query_context.status.value,
                query_context.result_count,
                query_context.execution_time_ms)
            
            logger.info(f"Added query {query_context.query_id} to session {session_id}")
            return query_context.query_id
    
    async def update_query_in_session(self, 
                                    session_id: str,
                                    query_context: QueryContext):
        """Update existing query in session history"""
        async with self.db_pool.acquire() as conn:
            # Update query
            await conn.execute("""
                UPDATE public.query_history SET
                    category = $2,
                    confidence = $3,
                    inherited_from_query = $4,
                    inheritance_type = $5,
                    inherited_elements = $6,
                    extracted_params = $7,
                    active_filters = $8,
                    entities_mentioned = $9,
                    status = $10,
                    result_count = $11,
                    execution_time_ms = $12
                WHERE query_id = $1
            """, UUID(query_context.query_id),
                query_context.category.value if query_context.category else None,
                query_context.confidence, 
                UUID(query_context.inherited_from_query) if query_context.inherited_from_query else None,
                query_context.inheritance_type.value, 
                json.dumps(query_context.inherited_elements),
                json.dumps(query_context.extracted_params),
                json.dumps(query_context.active_filters),
                json.dumps(query_context.entities_mentioned),
                query_context.status.value,
                query_context.result_count,
                query_context.execution_time_ms)
            
            logger.info(f"Updated query {query_context.query_id} in session {session_id}")
    
    async def get_session_history(self, session_id: str, limit: int = 10) -> List[QueryContext]:
        """Get recent query history for session"""
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM public.query_history
                WHERE session_id = $1
                ORDER BY sequence_number DESC
                LIMIT $2
            """, session_id, limit)
            
            history = []
            for row in rows:
                # QueryCategory is agent-specific, so we store as string
                context = QueryContext(
                    query_id=str(row['query_id']),
                    query_text=row['query_text'],
                    normalized_query=row['normalized_query'],
                    category=row['category'],  # Keep as string, not enum
                    confidence=row['confidence'],
                    inherited_from_query=str(row['inherited_from_query']) if row['inherited_from_query'] else None,
                    inheritance_type=InheritanceType(row['inheritance_type']) if row['inheritance_type'] else InheritanceType.NONE,
                    inherited_elements=row['inherited_elements'] if isinstance(row['inherited_elements'], dict) else {},
                    extracted_params=row['extracted_params'] if isinstance(row['extracted_params'], dict) else {},
                    active_filters=row['active_filters'] if isinstance(row['active_filters'], dict) else {},
                    entities_mentioned=row['entities_mentioned'] if isinstance(row['entities_mentioned'], dict) else {},
                    status=QueryStatus(row['status']) if row['status'] else QueryStatus.PENDING,
                    result_count=row.get('result_count', 0),
                    execution_time_ms=row.get('execution_time_ms', 0)
                )
                history.append(context)
            
            return history
    
    async def update_session_active_status(self):
        """Update active status for all sessions based on expiration"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("SELECT public.update_session_active_status()")
            logger.info("Updated session active status")
    
    async def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        async with self.db_pool.acquire() as conn:
            deleted = await conn.fetchval("SELECT public.cleanup_expired_sessions()")
            logger.info(f"Cleaned up expired sessions")


class EnhancedSessionManager(PostgreSQLSessionManager):
    """Extended session manager with training data collection"""
    
    async def record_interaction_event(self,
                                     session_id: str,
                                     query_id: str,
                                     event_type: str,
                                     llm_interaction: Optional[LLMInteraction] = None,
                                     input_data: Dict = None,
                                     output_data: Dict = None,
                                     success: bool = True,
                                     confidence: float = None):
        """Record detailed interaction for training"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO public.interaction_events (
                    session_id, query_id, event_type,
                    llm_model_used, prompt_template_used,
                    prompt_tokens, completion_tokens, latency_ms,
                    input_data, output_data, confidence_score,
                    was_successful
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            """, session_id, query_id, event_type,
                llm_interaction.model if llm_interaction else None,
                llm_interaction.prompt_template if llm_interaction else None,
                llm_interaction.prompt_tokens if llm_interaction else None,
                llm_interaction.completion_tokens if llm_interaction else None,
                llm_interaction.latency_ms if llm_interaction else None,
                json.dumps(input_data) if input_data else None,
                json.dumps(output_data) if output_data else None,
                confidence, success)
    
    async def record_user_feedback(self,
                                  query_id: str,
                                  feedback_type: str,
                                  rating: Optional[int] = None,
                                  corrected_params: Optional[Dict] = None):
        """Record user feedback for reinforcement learning"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO public.user_feedback (
                    query_id, feedback_type, rating, corrected_params
                ) VALUES ($1, $2, $3, $4)
            """, query_id, feedback_type, rating, 
                json.dumps(corrected_params) if corrected_params else None)
    
    async def record_entity_resolution(self,
                                     query_id: str,
                                     entity_text: str,
                                     entity_type: str,
                                     resolved_value: str,
                                     method: str,
                                     confidence: float,
                                     alternatives: List[Dict] = None):
        """Track entity resolution for NER training"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO public.entity_resolutions (
                    query_id, entity_text, entity_type,
                    resolved_value, resolution_method, confidence, alternatives
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            """, query_id, entity_text, entity_type,
                resolved_value, method, confidence,
                json.dumps(alternatives) if alternatives else None)
    
    async def record_clarification(self,
                                  query_id: str,
                                  clarification_round: int,
                                  options_presented: List[str],
                                  user_selection: Any,
                                  free_text: Optional[str] = None,
                                  led_to_success: bool = True,
                                  time_to_resolve_ms: int = 0):
        """Track clarification effectiveness"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO public.clarification_tracking (
                    query_id, clarification_round, options_presented,
                    user_selection, free_text_provided, led_to_success,
                    time_to_resolve_ms
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            """, query_id, clarification_round, json.dumps(options_presented),
                json.dumps(user_selection), free_text, led_to_success,
                time_to_resolve_ms)
    
    async def get_training_data(self,
                               start_date: datetime,
                               end_date: datetime,
                               min_confidence: float = 0.7,
                               only_successful: bool = True) -> List[TrainingExample]:
        """Extract training data for model improvement"""
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT 
                    qh.query_text,
                    qh.normalized_query,
                    qh.category,
                    qh.extracted_params,
                    qh.confidence,
                    ie.event_type,
                    ie.input_data,
                    ie.output_data,
                    uf.feedback_type,
                    uf.corrected_params,
                    uf.rating
                FROM public.query_history qh
                LEFT JOIN public.interaction_events ie 
                    ON qh.query_id = ie.query_id
                LEFT JOIN public.user_feedback uf 
                    ON qh.query_id = uf.query_id
                WHERE qh.created_at BETWEEN $1 AND $2
                    AND qh.confidence >= $3
                    AND ($4 = FALSE OR qh.status = 'completed')
                ORDER BY qh.created_at
            """, start_date, end_date, min_confidence, only_successful)
            
            return [TrainingExample.from_row(dict(row)) for row in rows]

    async def record_training_example(self, training_example: TrainingExample):
        """Record a training example for model improvement"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                               INSERT INTO public.training_examples (query_id, query_text, normalized_query, category,
                                                                     extracted_params, confidence, event_type,
                                                                     input_data,
                                                                     output_data, feedback_type, corrected_params,
                                                                     rating)
                               VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                               """,
                               training_example.query_id,
                               training_example.query_text,
                               training_example.normalized_query,
                               training_example.category,
                               json.dumps(training_example.extracted_params),
                               training_example.confidence,
                               training_example.event_type,
                               json.dumps(training_example.input_data),
                               json.dumps(training_example.output_data),
                               training_example.feedback_type,
                               json.dumps(training_example.corrected_params),
                               training_example.rating
                               )

    
    async def get_clarification_effectiveness(self) -> Dict:
        """Analyze clarification strategy effectiveness"""
        async with self.db_pool.acquire() as conn:
            result = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_clarifications,
                    AVG(time_to_resolve_ms) as avg_resolution_time,
                    SUM(CASE WHEN led_to_success THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as success_rate,
                    AVG(clarification_round) as avg_rounds_needed
                FROM public.clarification_tracking
                WHERE created_at > NOW() - INTERVAL '30 days'
            """)
            
            return dict(result) if result else {
                'total_clarifications': 0,
                'avg_resolution_time': 0,
                'success_rate': 0,
                'avg_rounds_needed': 0
            }
    
    async def get_category_performance(self) -> Dict:
        """Get performance metrics by category"""
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM public.query_performance_view
            """)
            
            return {row['category']: dict(row) for row in rows}