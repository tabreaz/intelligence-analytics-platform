# src/agents/query_understanding/prompt_builder.py
"""
Unified prompt builder for query understanding
"""
import json
from typing import Dict, Any, List, Optional

from src.core.session_manager_models import QueryContext
from .constants import CONTINUATION_PHRASES
from .prompts.prompts import SCHEMA_AWARE_PROMPT


class UnifiedPromptBuilder:
    """Build prompts for classification and extraction with configurable options"""

    def __init__(self):
        self.base_prompt = SCHEMA_AWARE_PROMPT

    def build_prompt(
            self,
            query: str,
            query_history: List[QueryContext],
            available_context: Dict[str, Any],
            include_location_extraction: bool = False
    ) -> Dict[str, str]:
        """
        Build unified prompt with configurable location handling
        
        Args:
            query: Current user query
            query_history: Previous queries in session
            available_context: Available context from session
            include_location_extraction: Whether to include location extraction instructions
            
        Returns:
            Dictionary with 'system' and 'user' prompts
        """
        # Determine if this is a continuation query
        is_continuation = self._is_continuation_query(query, query_history, available_context)
        previous_query = query_history[0] if query_history and is_continuation else None

        # Build system prompt
        system_prompt = self._build_system_prompt(
            is_continuation=is_continuation,
            previous_query=previous_query,
            available_context=available_context,
            include_location_extraction=include_location_extraction
        )

        # Build user prompt
        user_prompt = self._build_user_prompt(
            query=query,
            query_history=query_history,
            available_context=available_context,
            include_location_extraction=include_location_extraction,
            is_continuation=is_continuation
        )

        return {
            "system": system_prompt,
            "user": user_prompt
        }

    def _is_continuation_query(
            self,
            query: str,
            query_history: List[QueryContext],
            available_context: Dict[str, Any]
    ) -> bool:
        """Check if query is a continuation of previous queries"""
        if not query_history or not available_context.get("mapped_parameters"):
            return False

        # Check for continuation phrases
        query_lower = query.lower()
        return any(phrase in query_lower for phrase in CONTINUATION_PHRASES)

    def _build_system_prompt(
            self,
            is_continuation: bool,
            previous_query: Optional[QueryContext],
            available_context: Dict[str, Any],
            include_location_extraction: bool
    ) -> str:
        """Build system prompt with appropriate instructions"""

        # Start with base prompt
        system_prompt = self.base_prompt

        # Add location handling instructions
        if not include_location_extraction:
            system_prompt += """

## IMPORTANT: Location Handling

DO NOT extract detailed location information. Only identify that locations are mentioned.
Location details (coordinates, geohashes) will be handled by a specialized location service.

When you see location mentions:
- In entities_detected.locations, just note "location_mentioned": true
- Do NOT resolve to coordinates or geohashes
- Do NOT provide geohash lists in mapped_parameters
"""

        # Add context-aware instructions
        system_prompt += f"""

## CONTEXT-AWARE INSTRUCTIONS

You are processing a {"continuation" if is_continuation else "new"} query.

{"### Previous Query Context:" if is_continuation else ""}
{f"- Previous Query: '{previous_query.query_text}'" if is_continuation and previous_query else ""}
{f"- Previous Category: {previous_query.category if isinstance(previous_query.category, str) else (previous_query.category.value if previous_query and previous_query.category else 'N/A')}" if is_continuation else ""}
{f"- Previous Parameters: {json.dumps(available_context.get('mapped_parameters', {}), indent=2)}" if is_continuation else ""}

### IMPORTANT RULES FOR CONTINUATION QUERIES:
1. If this is a continuation query (e.g., "how about X"), inherit ALL parameters from the previous query except what's explicitly changed
2. Keep the SAME category as the previous query unless the intent completely changes
3. Only modify the specific parameters mentioned in the new query
4. Preserve all other parameters (time ranges, filters, etc.)
5. ALWAYS generate a complete "context_aware_query" that includes ALL context from previous queries
   - For example: If previous was "People at Dubai Mall yesterday" and current is "how about Iranians"
   - The context_aware_query should be: "Iranian people at Dubai Mall yesterday"

CRITICAL OUTPUT REQUIREMENTS:
1. Return ONLY valid JSON - no markdown, no code blocks, no explanatory text
2. Do NOT wrap the JSON in ```json``` or ``` tags  
3. Do NOT include any text before or after the JSON
4. The response must start with {{ and end with }}
5. The response must be valid, parseable JSON
"""

        return system_prompt

    def _build_user_prompt(
            self,
            query: str,
            query_history: List[QueryContext],
            available_context: Dict[str, Any],
            include_location_extraction: bool,
            is_continuation: bool
    ) -> str:
        """Build user prompt with query and context"""

        user_prompt = f"""
Current Query: "{query}"

{"Previous Query Context:" if query_history else ""}
{json.dumps(available_context.get('previous_queries', []), indent=2) if query_history else ""}

{f'''For continuation queries, generate a complete "context_aware_query" that combines:
- Previous query: "{query_history[0].query_text if query_history else ''}"
- Current modification: "{query}"
- Result should be a self-contained query with all context resolved
''' if is_continuation else 'For new queries, "context_aware_query" should be the same as the current query.'}

Process this query {"and return the complete classification and parameter extraction" if include_location_extraction else "for classification and parameter extraction"}.
{"Remember: Only identify location mentions, don't extract location details." if not include_location_extraction else ""}
"""

        return user_prompt.strip()
