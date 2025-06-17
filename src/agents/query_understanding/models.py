# src/agents/query_understanding/models.py
"""
Data models for Query Understanding Agent
"""
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml

# Import session manager models that are referenced here
from src.core.session_manager_models import InheritanceType


def load_query_categories() -> Dict[str, Any]:
    """Load query categories from configuration file"""
    config_path = Path(__file__).parent.parent.parent.parent / "config" / "query_categories.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


# Load categories from config
QUERY_CATEGORIES_CONFIG = load_query_categories()


class QueryCategory(Enum):
    """Query categories for classification - dynamically loaded from config"""
    pass


# Dynamically create enum values from config
def _create_query_categories():
    """Create QueryCategory enum values from configuration"""
    categories = {}
    for group_name, group_categories in QUERY_CATEGORIES_CONFIG['query_categories'].items():
        for category_key, category_info in group_categories.items():
            # Convert to uppercase for enum
            enum_key = category_key.upper()
            categories[enum_key] = category_key

    # Create the enum class with dynamic values
    return Enum('QueryCategory', categories)


# Replace the class with dynamically created one
QueryCategory = _create_query_categories()


@dataclass
class PromptOutput:
    """Structured prompt output for LLM"""
    system_prompt: str
    user_prompt: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CategoryInfo:
    """Detailed information about a query category"""
    key: str
    name: str
    description: str
    examples: List[str]
    required_params: List[str]
    optional_params: List[str]
    group: str

    @classmethod
    def from_config(cls, key: str, config: Dict, group: str) -> 'CategoryInfo':
        """Create CategoryInfo from configuration"""
        return cls(
            key=key,
            name=config['name'],
            description=config['description'],
            examples=config['examples'],
            required_params=config.get('required_params', []),
            optional_params=config.get('optional_params', []),
            group=group
        )


class QueryCategoryManager:
    """Manager for query category operations"""

    def __init__(self):
        self.config = QUERY_CATEGORIES_CONFIG
        self._category_info_cache = {}
        self._load_category_info()

    def _load_category_info(self):
        """Load all category information"""
        for group_name, group_categories in self.config['query_categories'].items():
            for category_key, category_config in group_categories.items():
                self._category_info_cache[category_key] = CategoryInfo.from_config(
                    category_key, category_config, group_name
                )

    def get_category_info(self, category: QueryCategory) -> CategoryInfo:
        """Get detailed information about a category"""
        return self._category_info_cache.get(category.value)

    def get_examples_for_category(self, category: QueryCategory) -> List[str]:
        """Get example queries for a category"""
        info = self.get_category_info(category)
        return info.examples if info else []

    def get_required_params(self, category: QueryCategory) -> List[str]:
        """Get required parameters for a category"""
        info = self.get_category_info(category)
        return info.required_params if info else []

    def get_all_categories_by_group(self) -> Dict[str, List[QueryCategory]]:
        """Get all categories organized by group"""
        grouped = {}
        for group_name in self.config['query_categories']:
            grouped[group_name] = []
            for category_key in self.config['query_categories'][group_name]:
                enum_key = category_key.upper()
                grouped[group_name].append(QueryCategory[enum_key])
        return grouped


@dataclass
class ContextDecision:
    """Decision about context inheritance"""
    is_continuation: bool
    confidence: float
    inherited_from: Optional[str] = None
    inheritance_type: InheritanceType = InheritanceType.NONE
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_continuation": self.is_continuation,
            "confidence": self.confidence,
            "inherited_from": self.inherited_from,
            "inheritance_type": self.inheritance_type.value,
            "reason": self.reason
        }


@dataclass
class ContextRule:
    """Rule for context detection"""
    name: str
    pattern: str
    action: str
    confidence: float
    description: Optional[str] = None


@dataclass
class ContextMatch:
    """Match result from context rule"""
    rule: ContextRule
    matched_text: str
    groups: tuple
    confidence: float


@dataclass
class EntityOverlap:
    """Entity overlap analysis result"""
    score: float
    entities: List[str]
    entity_types: Dict[str, int] = field(default_factory=dict)


@dataclass
class CategoryClassification:
    """Query category classification result"""
    category: QueryCategory
    confidence: float
    reasoning: Optional[str] = None
    alternative_categories: List[tuple[QueryCategory, float]] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Parameter validation result"""
    is_complete: bool
    missing_params: List[str] = field(default_factory=list)
    invalid_params: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ClarificationRequest:
    """Request for user clarification"""
    query_id: str
    clarification_type: str
    question: str
    options: List[str] = field(default_factory=list)
    allow_free_text: bool = True
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    """Final query processing result"""
    request_id: str
    session_id: str
    category: QueryCategory
    parameters: Dict[str, Any]
    results: Optional[Any] = None
    summary: Optional[str] = None
    execution_time_ms: int = 0
    requires_clarification: bool = False
    clarification_requests: List[ClarificationRequest] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
