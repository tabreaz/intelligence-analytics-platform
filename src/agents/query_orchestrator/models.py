# src/agents/query_orchestrator/models.py
"""
Data models for Query Orchestrator Agent
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional


class OrchestratorStatus(str, Enum):
    """Status of orchestration process"""
    CLASSIFYING = "classifying"
    RESOLVING_AMBIGUITIES = "resolving_ambiguities"
    PROCESSING_FILTERS = "processing_filters"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentResult:
    """Result from an individual agent"""
    agent_name: str
    status: str
    result: Dict[str, Any]
    execution_time: float
    error: Optional[str] = None


@dataclass
class OrchestratorResult:
    """Combined result from orchestrator"""
    # Classification results
    classification: Dict[str, Any] = field(default_factory=dict)
    context_aware_query: str = ""
    is_continuation: bool = False
    domains: List[str] = field(default_factory=list)
    agents_required: List[str] = field(default_factory=list)

    # Ambiguity handling
    has_ambiguities: bool = False
    ambiguities: List[Dict[str, Any]] = field(default_factory=list)
    ambiguities_resolved: bool = False

    # Filter results (merged and deduplicated)
    time_filters: Dict[str, Any] = field(default_factory=dict)
    location_filters: Dict[str, Any] = field(default_factory=dict)
    profile_filters: Dict[str, Any] = field(default_factory=dict)
    risk_filters: Dict[str, Any] = field(default_factory=dict)

    # Entity annotation removed - fire and forget pattern

    # Metadata
    total_execution_time: float = 0.0
    agent_results: Dict[str, AgentResult] = field(default_factory=dict)
    orchestrator_status: OrchestratorStatus = OrchestratorStatus.CLASSIFYING
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response"""
        return {
            "classification": self.classification,
            "context_aware_query": self.context_aware_query,
            "is_continuation": self.is_continuation,
            "domains": self.domains,
            "agents_required": self.agents_required,
            "has_ambiguities": self.has_ambiguities,
            "ambiguities": self.ambiguities,
            "ambiguities_resolved": self.ambiguities_resolved,
            "filters": {
                "time": self.time_filters,
                "location": self.location_filters,
                "profile": self.profile_filters,
                "risk": self.risk_filters
            },
            "metadata": {
                "total_execution_time": self.total_execution_time,
                "orchestrator_status": self.orchestrator_status.value,
                "agent_execution_times": {
                    name: result.execution_time
                    for name, result in self.agent_results.items()
                }
            },
            "error": self.error
        }
