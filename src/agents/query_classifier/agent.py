# src/agents/query_classifier/agent.py
"""
Query Classifier Agent - Resolves context and classifies queries
Modernized version using BaseAgent with resource management
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

from .prompt import build_system_prompt
from .constants import (
    GENERAL_INQUIRY_KEY,
    VALID_DOMAINS, VALID_AGENTS,
    DOMAIN_PROFILE, DOMAIN_MOVEMENT, DOMAIN_RISK_PROFILES
)
from ..base_agent import BaseAgent, AgentResponse, AgentRequest
from ...core.logger import get_logger

logger = get_logger(__name__)


class QueryClassifierAgent(BaseAgent):
    """
    Agent for resolving context and classifying queries
    
    Features:
    - Detects and resolves continuation queries
    - Classifies queries into predefined categories
    - Identifies unsupported domains
    - Maintains conversation continuity
    """

    def __init__(self, name: str, config: dict, resource_manager):
        """Initialize with agent-specific components only"""
        super().__init__(name, config, resource_manager)
        
        # Load query categories from YAML
        self.categories = self._load_categories()
        
        # Agent metadata handled by base class
        # self.agent_id, self.description, self.capabilities are in base class

    async def validate_request(self, request: AgentRequest) -> bool:
        """Query classifier can handle any request"""
        return True

    async def process_internal(self, request: AgentRequest) -> AgentResponse:
        """
        Core query classification logic
        All infrastructure concerns handled by BaseAgent
        """
        start_time = datetime.now()
        
        # Log activity
        self.activity_logger.action("Classifying query and resolving context")
        
        # Extract context from request
        context = request.context or {}
        previous_query = context.get('previous_query', {})
        
        # Build prompts
        system_prompt = build_system_prompt(self.categories)
        user_prompt = self._build_user_prompt(request.prompt, previous_query)
        
        # Get LLM response with retry logic
        llm_response = await self.call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            request=request,
            event_type="query_classification"
        )
        
        if not llm_response:
            raise ValueError("Failed to get valid LLM response")
        
        # Parse response
        try:
            result = json.loads(llm_response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.debug(f"Raw response: {llm_response[:500]}...")
            raise ValueError("Invalid JSON response from LLM")
        
        # Validate and enhance result
        result = self._validate_and_enhance_result(result, request.prompt)
        
        # Log classification results
        classification = result.get('classification', {})
        self.activity_logger.info(
            f"Query classified as '{classification.get('category')}' "
            f"with confidence {classification.get('confidence', 0):.2f}"
        )
        
        # Log if continuation query
        if result.get('is_continuation'):
            self.activity_logger.info("Detected as continuation query")
        
        # Log domain check
        domain_check = result.get('domain_check', {})
        if not domain_check.get('is_supported', True):
            self.activity_logger.warning("Query is outside supported domain")
        
        # Log ambiguities if any
        ambiguities = result.get('ambiguities', [])
        if ambiguities:
            self.activity_logger.warning(f"Found {len(ambiguities)} ambiguities in query")
            for amb in ambiguities:
                self.activity_logger.info(f"Ambiguity: {amb.get('parameter')} - {amb.get('issue')}")
        
        # Create response with metadata
        return self._create_success_response(
            request=request,
            result=result,
            start_time=start_time,
            category=classification.get('category'),
            confidence=classification.get('confidence', 0),
            is_continuation=result.get('is_continuation', False),
            agents_required=result.get('agents_required', []),
            ambiguity_count=len(ambiguities)
        )

    def _load_categories(self) -> Dict[str, Any]:
        """Load query categories from YAML configuration"""
        config_path = Path(__file__).parent.parent.parent.parent / "config" / "query_categories.yaml"
        
        try:
            with open(config_path, 'r') as f:
                categories = yaml.safe_load(f)
                return categories
        except Exception as e:
            logger.error(f"Failed to load query categories: {e}")
            # Return minimal categories for fallback
            return {
                "query_categories": {
                    "other": {
                        "general_inquiry": {
                            "name": "General Inquiry",
                            "description": "General queries"
                        },
                        "unsupported_domain": {
                            "name": "Unsupported Domain",
                            "description": "Queries outside our domain"
                        }
                    }
                }
            }

    def _build_user_prompt(self, current_query: str, previous_query: Dict[str, Any]) -> str:
        """Build user prompt with context"""
        if not previous_query:
            return f"Classify this query: {current_query}"
        
        # Include previous context for continuation detection
        prev_text = previous_query.get('query_text', '')
        prev_category = previous_query.get('category', '')
        prev_context_aware = previous_query.get('context_aware_query', '')
        
        prompt_parts = [
            f"Previous query: {prev_text}",
            f"Previous classification: {prev_category}",
            f"Previous context-aware query: {prev_context_aware}",
            "",
            f"Current query: {current_query}",
            "",
            "Classify the current query. If it's a continuation, resolve the full context."
        ]
        
        return "\n".join(prompt_parts)

    def _validate_and_enhance_result(self, result: Dict[str, Any], original_query: str) -> Dict[str, Any]:
        """Validate and enhance classification result"""
        # Ensure required fields exist
        if 'original_query' not in result:
            result['original_query'] = original_query
        
        if 'context_aware_query' not in result:
            result['context_aware_query'] = original_query
        
        if 'is_continuation' not in result:
            result['is_continuation'] = False
        
        # Validate classification
        if 'classification' not in result:
            result['classification'] = {
                'category': GENERAL_INQUIRY_KEY,
                'confidence': 0.5,
                'reasoning': 'No classification provided'
            }
        
        # Validate domains
        if 'domains' not in result:
            result['domains'] = []
        else:
            # Filter to valid domains only
            result['domains'] = [d for d in result['domains'] if d in VALID_DOMAINS]
        
        # Validate domain check
        if 'domain_check' not in result:
            # Determine if supported based on category
            category = result['classification']['category']
            is_unsupported = category == 'unsupported_domain'
            result['domain_check'] = {
                'is_supported': not is_unsupported,
                'message': 'Query is outside our domain' if is_unsupported else ''
            }
        
        # Validate agents required
        if 'agents_required' not in result:
            result['agents_required'] = []
        else:
            # Filter to valid agents only
            result['agents_required'] = [a for a in result['agents_required'] if a in VALID_AGENTS]
        
        # Ensure ambiguities is a list
        if 'ambiguities' not in result:
            result['ambiguities'] = []
        
        return result