# src/agents/entity_annotator/models.py
"""
Data models for Entity Annotator Agent
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class Entity:
    """Represents an annotated entity"""
    type: str
    value: str
    start_pos: int
    end_pos: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "value": self.value,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos
        }


@dataclass
class EntityAnnotatorResult:
    """Result model for entity annotator agent"""
    query: str
    annotated_query: str
    entities: List[Entity] = field(default_factory=list)
    entity_types: List[str] = field(default_factory=list)
    confidence: float = 0.0
    extraction_method: str = "llm"
    raw_response: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response"""
        return {
            "query": self.query,
            "annotated_query": self.annotated_query,
            "entities": [e.to_dict() for e in self.entities],
            "entity_types": self.entity_types,
            "confidence": self.confidence,
            "extraction_method": self.extraction_method,
            "error": self.error
        }
