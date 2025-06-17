# src/agents/query_understanding/prompts/contextual_prompt_builder.py

import json
from typing import Dict, List, Any, Optional

from src.core.session_manager_models import QueryContext, Session
from .prompts import SCHEMA_AWARE_PROMPT


class ContextualPromptBuilder:
    """
    Builds final contextual prompts after classification and ambiguity resolution.
    This is used for actual query execution or API mapping.
    """

    def build_final_prompt(
            self,
            classified_query: QueryContext,
            session: Session,
            query_history: List[QueryContext],
            resolved_ambiguities: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Build the final LLM prompt for query execution.
        Uses:
        - Classified query
        - Session context
        - Query history
        - Resolved ambiguities
        """
        # Merge resolved ambiguities if provided
        resolved_ambiguities = resolved_ambiguities or {}

        system_prompt = f"""{SCHEMA_AWARE_PROMPT}

        ## FINAL PROMPT INSTRUCTIONS

        You are now processing a query that has already been classified.
        Your task is to finalize parameter mapping and ensure it's valid for execution.

        ### Tasks:
        1. PRESERVE all inherited parameters from the classification phase
        2. Apply any resolved ambiguities provided
        3. Map all detected entities to database fields
        4. Validate input formats (IMSI, geohash, ISO codes, etc.)
        5. Flag any remaining validation issues
        6. Return structured JSON with final parameters
        
        IMPORTANT: If parameters were already mapped in the classification phase (shown below),
        you MUST include them in your final output unless explicitly overridden by the current query.

        ### Resolved Ambiguities:
        {json.dumps(resolved_ambiguities, indent=2) if resolved_ambiguities else 'None'}

        ### Session Info:
        - Session ID: {session.session_id}
        - Total Queries: {session.total_queries}
        - Active Filters: {json.dumps(session.active_context.get('filters', {}) if session.active_context else {}, indent=2)}
        """

        user_prompt = f"""
Current Query: "{classified_query.query_text}"

## Classification Results:
- Category: {classified_query.category.value if classified_query.category else 'unknown'}
- Confidence: {classified_query.confidence}
- Is Continuation: {"Yes" if classified_query.inherited_from_query else "No"}
- Inherited Parameters: 
{json.dumps(classified_query.inherited_elements, indent=2)}

## Extracted Entities:
{json.dumps(classified_query.entities_mentioned, indent=2)}

## Mapped Parameters:
{json.dumps(classified_query.extracted_params, indent=2)}

## Recent Interaction Summary:
{self._inject_chat_history(query_history[:3])}

IMPORTANT:
1. Start with the "Mapped Parameters" shown above as your base
2. Only modify parameters that are explicitly changed in the current query
3. Apply all validation rules from schema
4. Ensure proper mapping to database fields
5. If any ambiguity remains, flag it clearly
6. Return the complete set of parameters (inherited + new) in the final JSON response

CRITICAL: You MUST return ONLY valid JSON following the exact response format specified in the schema.
Do NOT include any explanatory text before or after the JSON.
"""

        return {
            "system": system_prompt,
            "user": user_prompt
        }

    def _inject_chat_history(self, query_history: List[QueryContext]) -> str:
        """Converts recent query history into natural language summary"""
        lines = []
        for i, qc in enumerate(query_history):
            lines.append(f"User: {qc.query_text}")
            if qc.extracted_params:
                lines.append(f"Assistant: {json.dumps(qc.extracted_params)}")
        return "\n".join(lines) if lines else "None"

    def postprocess_llm_response(self, llm_response: str) -> Dict[str, Any]:
        """
        Post-process raw LLM response into structured format
        """
        if not llm_response or not llm_response.strip():
            raise ValueError("Empty LLM response received")

        try:
            result = json.loads(llm_response)
            return result
        except json.JSONDecodeError as e:
            # Log the actual response for debugging
            print(f"Failed to parse LLM response: {llm_response[:200]}...")
            raise ValueError(f"Invalid LLM response format: {e}")
