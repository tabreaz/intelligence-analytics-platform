# src/core/session_manager_models.py
"""
Data models for Session Manager - used across multiple agents
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from enum import Enum
from collections import deque


class InheritanceType(Enum):
    """Types of context inheritance"""
    FULL = "full"
    PARTIAL = "partial"
    NONE = "none"


class QueryStatus(Enum):
    """Query processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    CLARIFYING = "clarifying"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentActivityMessage:
    """Activity message from an agent during query processing"""
    agent_name: str
    timestamp: datetime
    message: str
    is_error: bool = False
    activity_type: str = "info"  # info, decision, action, retry, error
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "agent_name": self.agent_name,
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
            "is_error": self.is_error,
            "activity_type": self.activity_type,
            "metadata": self.metadata
        }


@dataclass
class Session:
    """User session model"""
    session_id: str
    created_at: datetime
    last_activity: datetime
    expires_at: datetime
    total_queries: int = 0
    active_context: Dict[str, Any] = field(default_factory=dict)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    session_metadata: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    
    def is_expired(self) -> bool:
        """Check if session has expired"""
        return datetime.now(timezone.utc) > self.expires_at


@dataclass
class QueryContext:
    """Context for a query - aligned with query_history table"""
    # Primary fields (added to match DB)
    query_id: Optional[str] = None  # UUID in DB, str in Python
    session_id: Optional[str] = None
    sequence_number: Optional[int] = None
    
    # Query data
    query_text: str = ""
    normalized_query: Optional[str] = None
    category: Optional[Any] = None  # Will be QueryCategory from query_understanding
    subcategory: Optional[str] = None
    confidence: float = 0.0
    
    # Context inheritance
    inherited_from_query: Optional[str] = None
    inheritance_type: InheritanceType = InheritanceType.NONE
    inherited_elements: Dict[str, Any] = field(default_factory=dict)
    
    # Extracted data
    extracted_params: Dict[str, Any] = field(default_factory=dict)
    active_filters: Dict[str, Any] = field(default_factory=dict)
    entities_mentioned: Dict[str, List[str]] = field(default_factory=dict)
    
    # Results
    result_count: Optional[int] = None
    result_entities: List[Dict[str, Any]] = field(default_factory=list)
    execution_time_ms: Optional[int] = None
    
    # Status
    status: QueryStatus = QueryStatus.PENDING
    clarifications_requested: List[Dict[str, Any]] = field(default_factory=list)
    clarifications_received: List[Dict[str, Any]] = field(default_factory=list)
    
    # Timestamp (added to match DB)
    created_at: Optional[datetime] = None
    
    # Activity tracking for real-time monitoring
    activity_queue: deque = field(default_factory=lambda: deque(maxlen=200))
    
    def add_activity(self, agent_name: str, message: str, is_error: bool = False,
                     activity_type: str = "info", metadata: Optional[Dict[str, Any]] = None):
        """Add an activity message to the query's activity queue"""
        activity = AgentActivityMessage(
            agent_name=agent_name,
            timestamp=datetime.now(timezone.utc),
            message=message,
            is_error=is_error,
            activity_type=activity_type,
            metadata=metadata or {}
        )
        self.activity_queue.append(activity)
    
    def get_activities(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all or limited activities as list of dicts"""
        activities = list(self.activity_queue)
        if limit:
            activities = activities[-limit:]
        return [activity.to_dict() for activity in activities]


@dataclass
class LLMInteraction:
    """LLM interaction details for tracking"""
    model: str
    prompt_template: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    temperature: float = 0.1
    max_tokens: int = 1000


@dataclass
class TrainingExample:
    """Training example for model improvement - aligned with training_examples table"""
    # Primary key (added)
    example_id: Optional[str] = None  # UUID in DB, str in Python
    
    # Core fields
    query_id: Optional[str] = None  # References query_history
    query_text: str = ""
    normalized_query: Optional[str] = None
    category: Optional[str] = None
    extracted_params: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    
    # Training metadata
    event_type: Optional[str] = None  # 'classification', 'extraction', 'execution'
    input_data: Optional[Dict[str, Any]] = None  # The prompt sent to LLM
    output_data: Optional[Dict[str, Any]] = None  # The LLM response
    
    # Feedback data
    feedback_type: Optional[str] = None  # 'correct', 'incorrect', 'partial'
    corrected_params: Optional[Dict[str, Any]] = None  # User corrections if any
    rating: Optional[int] = None  # 1-5
    has_positive_feedback: bool = False
    
    # Training status (added to match DB)
    used_in_training: bool = False
    model_version_id: Optional[str] = None  # References model_versions
    
    # Timestamps (added to match DB)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    @classmethod
    def from_row(cls, row: Dict) -> 'TrainingExample':
        """Create TrainingExample from database row"""
        return cls(
            example_id=str(row.get('example_id')) if row.get('example_id') else None,
            query_id=str(row.get('query_id')) if row.get('query_id') else None,
            query_text=row.get('query_text', ''),
            normalized_query=row.get('normalized_query'),
            category=row.get('category'),
            extracted_params=row.get('extracted_params', {}),
            confidence=row.get('confidence', 0.0),
            event_type=row.get('event_type'),
            input_data=row.get('input_data'),
            output_data=row.get('output_data'),
            feedback_type=row.get('feedback_type'),
            corrected_params=row.get('corrected_params'),
            rating=row.get('rating'),
            has_positive_feedback=row.get('rating', 0) >= 4 if row.get('rating') else False,
            used_in_training=row.get('used_in_training', False),
            model_version_id=str(row.get('model_version_id')) if row.get('model_version_id') else None,
            created_at=row.get('created_at'),
            updated_at=row.get('updated_at')
        )