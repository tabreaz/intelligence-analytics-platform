# src/agents/query_classifier/agent.py
"""
Query Classifier Agent - Resolves context and classifies queries
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import yaml

from src.agents.base_agent import BaseAgent, AgentRequest, AgentResponse, AgentStatus
from src.core.activity_logger import ActivityLogger
from src.core.config_manager import ConfigManager
from src.core.llm.base_llm import LLMClientFactory
from src.core.logger import get_logger
from src.core.session_manager_models import QueryContext as QueryContextModel
from src.core.training_logger import TrainingLogger
from .constants import (
    GENERAL_INQUIRY_KEY,
    VALID_DOMAINS, VALID_AGENTS,
    DOMAIN_PROFILE, DOMAIN_MOVEMENT, DOMAIN_RISK_PROFILES
)

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

    def __init__(self, name: str, config: dict, config_manager: ConfigManager, session_manager=None):
        super().__init__(name, config)
        self.config_manager = config_manager
        self.session_manager = session_manager

        # Initialize LLM client
        llm_model = config.get('llm_model', 'openai')
        llm_config = config_manager.get_llm_config(llm_model)
        self.llm_client = LLMClientFactory.create_client(llm_config)

        # Load query categories from YAML
        self.categories = self._load_categories()
        self.category_metadata = self.categories.get('category_metadata', {})

        # Configuration
        self.enable_training_data = config.get('enable_training_data', True)
        self.confidence_threshold = self.category_metadata.get(
            'default_confidence_threshold', 0.7
        )

        # Initialize training logger
        self.training_logger = TrainingLogger(session_manager) if self.enable_training_data else None

        # Initialize activity logger
        self.activity_logger = ActivityLogger(agent_name=self.name)

        logger.info(f"QueryClassifierAgent initialized with {self._count_categories()} categories")

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

    def _count_categories(self) -> int:
        """Count total number of categories"""
        count = 0
        for group in self.categories.get('query_categories', {}).values():
            count += len(group)
        return count

    async def validate_request(self, request: AgentRequest) -> bool:
        """
        Always validate - all queries need classification
        """
        return bool(request.prompt)

    async def process(self, request: AgentRequest) -> AgentResponse:
        """
        Process request to resolve context and classify query
        """
        start_time = datetime.now()

        # Set query context for activity logging
        query_context = request.context.get('query_context')
        if query_context and isinstance(query_context, QueryContextModel):
            self.activity_logger.set_query_context(query_context)

        try:
            # Log start
            self.activity_logger.action("Analyzing query intent and context...")

            # Extract previous context
            previous_query = request.context.get('previous_query', {})

            # Resolve context and classify
            llm_start_time = datetime.now()
            result, prompts, llm_response = await self._classify_query(
                current_query=request.prompt,
                previous_query=previous_query
            )

            # Log the reasoning from LLM
            if result.get('classification', {}).get('reasoning'):
                self.activity_logger.info(result['classification']['reasoning'])

            # Log ambiguities if found
            ambiguities = result.get('ambiguities', [])
            if ambiguities:
                self.activity_logger.warning(f"Found {len(ambiguities)} ambiguities in query")
                for amb in ambiguities:
                    self.activity_logger.info(
                        f"Ambiguity: {amb.get('parameter', 'Unknown')} - {amb.get('issue', 'No description')}")

            # Log key decisions
            if result.get('is_continuation'):
                self.activity_logger.decision("This is a continuation of the previous query")

            if result.get('domain_check', {}).get('is_supported'):
                category = result.get('classification', {}).get('category', 'unknown')
                confidence = result.get('classification', {}).get('confidence', 0)
                domains = result.get('domains', [])

                # Log classification
                self.activity_logger.decision(f"Classified as '{category}' (confidence: {confidence:.0%})")

                # Log domains in user-friendly way
                if domains:
                    if len(domains) == 1:
                        self.activity_logger.info(f"Query relates to {domains[0]} analysis")
                    else:
                        self.activity_logger.info(f"Query spans multiple domains: {', '.join(domains)}")
            else:
                self.activity_logger.info(
                    result.get('domain_check', {}).get('message', 'Query outside supported domain'))

            execution_time = (datetime.now() - start_time).total_seconds()

            # Log training data if enabled (fire-and-forget)
            if self.training_logger and llm_response:
                # Get model config
                model_config = {
                    'model': getattr(self.llm_client.config, 'model', 'unknown') if hasattr(self.llm_client,
                                                                                            'config') else 'unknown',
                    'temperature': getattr(self.llm_client.config, 'temperature', 0.1) if hasattr(self.llm_client,
                                                                                                  'config') else 0.1,
                    'max_tokens': getattr(self.llm_client.config, 'max_tokens', 1000) if hasattr(self.llm_client,
                                                                                                 'config') else 1000
                }

                # Log in background
                self.training_logger.log_llm_interaction_background(
                    session_id=request.context.get('session_id'),
                    query_id=request.context.get('query_id', request.request_id),
                    query_text=request.prompt,
                    event_type="query_classification",
                    llm_response=llm_response,
                    llm_start_time=llm_start_time,
                    prompts=prompts,
                    result=result,
                    model_config=model_config,
                    category=result['classification'].get('category'),
                    confidence=result['classification'].get('confidence', 0.5),
                    success=result['domain_check'].get('is_supported', True)
                )

            return AgentResponse(
                request_id=request.request_id,
                agent_name=self.name,
                status=AgentStatus.COMPLETED,
                result=result,
                execution_time=execution_time
            )

        except Exception as e:
            logger.error(f"Query classification failed: {str(e)}", exc_info=True)
            self.activity_logger.error("Failed to classify query", error=e)

            return AgentResponse(
                request_id=request.request_id,
                agent_name=self.name,
                status=AgentStatus.FAILED,
                result={"error": str(e)},
                error=str(e),
                execution_time=(datetime.now() - start_time).total_seconds()
            )

    async def _classify_query(
            self,
            current_query: str,
            previous_query: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, str], str]:
        """Classify query and resolve context using LLM"""

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(current_query, previous_query)
        prompts = {"system": system_prompt, "user": user_prompt}

        try:
            # Call LLM
            response = await self.llm_client.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )

            # Check if response is empty
            if not response or not response.strip():
                logger.error("LLM returned empty response")
                raise ValueError("Empty LLM response")

            # Parse response
            try:
                parsed = json.loads(response)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {e}")
                logger.debug(f"Raw response: {response[:500]}")
                # Return fallback result with the actual response for training data
                result = self._create_fallback_result(current_query, f"JSON parse error: {str(e)}")
                return result, prompts, response

            # Validate and enhance result
            result = self._validate_result(parsed, current_query)

            return result, prompts, response

        except ValueError as e:
            # This catches "Empty LLM response" or other ValueError
            logger.error(f"LLM response error: {e}")
            result = self._create_fallback_result(current_query, str(e))
            return result, prompts, str(e)
        except Exception as e:
            # Catch any other exceptions (e.g., from LLM client)
            logger.error(f"Unexpected error during classification: {e}", exc_info=True)
            result = self._create_fallback_result(current_query, f"Unexpected error: {str(e)}")
            return result, prompts, str(e)

    def _create_fallback_result(self, original_query: str, error_reason: str) -> Dict[str, Any]:
        """Create a fallback result when classification fails"""
        return {
            'original_query': original_query,
            'context_aware_query': original_query,
            'is_continuation': False,
            'classification': {
                'category': GENERAL_INQUIRY_KEY,
                'confidence': 0.0,
                'reasoning': error_reason
            },
            'domains': [],
            'agents_required': [],
            'domain_check': {
                'is_supported': True,
                'message': None
            },
            'ambiguities': []
        }

    def _build_system_prompt(self) -> str:
        """Build system prompt for classification"""
        # Build categories section from loaded YAML
        categories_text = self._format_categories_for_prompt()

        return f"""You are a query classifier for telecom movement and SIGINT data analysis.

Your tasks:
1. Resolve context for continuation queries
2. Classify queries into appropriate categories
3. Identify unsupported domains

## Query Categories

{categories_text}

## Special Categories
- **unsupported_domain**: Query is NOT related to telecom/movement/SIGINT data
- **general_inquiry**: Related to our domain but doesn't fit specific categories

## Context Resolution
For continuation queries (e.g., "how about X", "what about Y"):
1. Inherit ALL context from previous query
2. Modify only what's explicitly changed
3. Create complete self-contained query

IMPORTANT: If the current query appears to be a continuation but there's no valid previous context (e.g., previous was unsupported_domain), try to infer the intended context or mark as general_inquiry

## Agents Required

Determine which processing agents are necessary to answer the query.  
Include any agent whose function is either directly referenced or implicitly required by the query context.  
Multiple agents may be required.

Available agents:

- **profile**: Extracts demographic, identity, and profile-based filters (age, gender, nationality, names, residency status, occupation, sponsor, EID, UID, etc.)
- **time**: Extracts and processes any temporal information—explicit (dates, time ranges, durations, hours) or implicit (e.g., "yesterday", "last week").
- **location**: Extracts, resolves, and processes location information (places, addresses, geohashes, cities, emirates, coordinates), including proximity.
- **risk**: Extracts or processes risk-related criteria, including risk scores, crime types, criminal flags, diplomatic status, or any threat/suspicion filter.
- **communication**: Required if the query involves calls, messaging, or network interactions.
- **movement**: Required if the query focuses on travel history, movement patterns, or physical presence.

**Choose all agents required to satisfy the prompt.**  
If the query includes both profile and risk filters, include both.  
If in doubt, err on the side of inclusion.

Return `agents_required` as an array of agent keys, e.g.  
`["profile", "location", "time", "risk"]`

## Domain Resolution
Identify which domains the query relates to. Multiple domains can apply:
- **profile**: Individual person analysis, demographics, behavior patterns
- **movement**: Location tracking, travel patterns, geo-spatial analysis
- **communication**: Call records, messaging, network interactions
- **risk_profiles**: Security analysis, threat assessment, suspicious activities

## Domain Check
If query is about:
- Weather, sports, news, entertainment → unsupported_domain
- General knowledge, math, coding → unsupported_domain
- Anything NOT related to people movement, telecom, security → unsupported_domain

## Ambiguity Detection
If the query has unclear references or missing information that prevents accurate processing, list them in the ambiguities array. Common ambiguities:
- Unclear entity references (e.g., "managers" - what type of managers?)
- Missing location context (e.g., "near the mall" - which mall?)
- Vague time references (e.g., "recently" - what time period?)
- Incomplete identifiers (e.g., partial phone numbers)
- Unclear nationality groups (e.g., "Asians" - which countries?)

## Output Format
Return ONLY valid JSON:
{{
    "original_query": "The user's original query",
    "context_aware_query": "Fully resolved query with all context",
    "is_continuation": true/false,
    "classification": {{
        "category": "category_key",
        "confidence": 0.0-1.0,
        "reasoning": "Brief explanation"
    }},
    "domains": ["profile", "movement"],  // Array of applicable domains
    "domain_check": {{
        "is_supported": true/false,
        "message": "User-friendly message if unsupported"
    }},
    "agents_required": ["profile", "time", "location", "risk"],
    "ambiguities": [
        {{
            "parameter": "field_name",
            "issue": "What is unclear about this parameter",
            "suggested_clarification": "Question to ask the user for clarification",
            "options": ["option1", "option2", "option3"]  // Optional suggested values
        }}
    ]
}}"""

    def _format_categories_for_prompt(self) -> str:
        """Format categories from YAML into prompt text"""
        lines = []
        query_categories = self.categories.get('query_categories', {})

        for group_name, group_categories in query_categories.items():
            # Format group name
            group_title = group_name.replace('_', ' ').title()
            lines.append(f"### {group_title}")

            for category_key, category_info in group_categories.items():
                # name = category_info.get('name', category_key)
                description = category_info.get('description', '')
                examples = category_info.get('examples', [])

                lines.append(f"- **{category_key}**: {description}")
                if examples:
                    example_text = ", ".join([f'"{ex}"' for ex in examples[:2]])
                    lines.append(f"  Examples: {example_text}")

            lines.append("")  # Empty line between groups

        return "\n".join(lines)

    def _build_user_prompt(
            self,
            current_query: str,
            previous_query: Dict[str, Any]
    ) -> str:
        """Build user prompt with context"""

        prompt = f'Current Query: "{current_query}"'

        if previous_query and previous_query.get('query_text'):
            prompt += f'\n\nPrevious Query: "{previous_query["query_text"]}"'
            prompt += f'\nPrevious Category: {previous_query.get("category", "unknown")}'
            if previous_query.get('extracted_params'):
                prompt += f'\nPrevious Parameters: {json.dumps(previous_query["extracted_params"], indent=2)}'
        else:
            prompt += "\n\nNo previous query context available."

        prompt += "\n\nClassify this query and resolve any context references."

        return prompt

    def _validate_result(self, parsed: Dict[str, Any], original_query: str) -> Dict[str, Any]:
        """Validate and enhance classification result"""

        # Ensure required fields
        result = {
            'original_query': original_query,
            'context_aware_query': parsed.get('context_aware_query', original_query),
            'is_continuation': parsed.get('is_continuation', False),
            'classification': parsed.get('classification', {}),
            'domains': parsed.get('domains', []),
            'domain_check': parsed.get('domain_check', {}),
            'agents_required': parsed.get('agents_required', []),
            'ambiguities': parsed.get('ambiguities', [])
        }

        # Validate domains
        domains = result['domains']

        # Ensure domains is a list
        if not isinstance(domains, list):
            domains = []

        # Filter valid domains using constants
        domains = [d for d in domains if d in VALID_DOMAINS]
        result['domains'] = domains

        # Validate agents_required
        agents_required = result['agents_required']

        # Ensure agents_required is a list
        if not isinstance(agents_required, list):
            agents_required = []

        # Filter valid agents using constants
        agents_required = [a for a in agents_required if a in VALID_AGENTS]
        result['agents_required'] = agents_required

        # Validate classification
        classification = result['classification']
        category = classification.get('category', 'general_inquiry')

        # Check if category exists in loaded categories
        if not self._is_valid_category(category):
            classification['reasoning'] = f"Invalid category '{category}' defaulted to general_inquiry"
            category = 'general_inquiry'

        classification['category'] = category
        classification['confidence'] = float(classification.get('confidence', 0.5))

        # Validate domain check
        domain_check = result['domain_check']
        if 'is_supported' not in domain_check:
            domain_check['is_supported'] = category != 'unsupported_domain'

        # If unsupported domain, ensure domains array is empty
        if category == 'unsupported_domain':
            result['domains'] = []
            if not domain_check.get('message'):
                domain_check[
                    'message'] = "I can only help with telecom movement and security-related queries. Please ask about people movements, locations, risk analysis, or similar topics."
        elif not domains and domain_check.get('is_supported', True):
            # If supported but no domains identified, add a default based on category
            if 'movement' in category or 'location' in category:
                result['domains'] = [DOMAIN_MOVEMENT]
            elif 'profile' in category or 'person' in category:
                result['domains'] = [DOMAIN_PROFILE]
            elif 'risk' in category or 'security' in category:
                result['domains'] = [DOMAIN_RISK_PROFILES]
            else:
                result['domains'] = [DOMAIN_PROFILE]  # Default to profile for general queries

        # Add flag to indicate if this should be stored in history
        result['store_in_history'] = category != 'unsupported_domain'

        return result

    def _is_valid_category(self, category: str) -> bool:
        """Check if category exists in loaded configuration"""
        query_categories = self.categories.get('query_categories', {})

        # Check all groups for the category
        for group_categories in query_categories.values():
            if category in group_categories:
                return True

        # Check special categories
        return category in ['unsupported_domain', 'general_inquiry']
