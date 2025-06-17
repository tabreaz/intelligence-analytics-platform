# src/agents/query_understanding/prompts/classification_prompt_builder.py

import json
from typing import Dict, List, Any

from src.core.session_manager_models import QueryContext, Session
from .prompts import SCHEMA_AWARE_PROMPT


class ClassificationPromptBuilder:
    """
    Builds classification prompts using SCHEMA_AWARE_PROMPT and chat history injection.
    No hardcoded regex or manual pattern matching.
    """

    def build_first_prompt(
            self,
            query: str,
            session: Session,
            query_history: List[QueryContext],
            available_context: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Build prompt based on whether this is a standalone or continuation query.
        """
        has_recent_history = len(query_history) > 0 and query_history[0].status.value == 'completed'

        if has_recent_history:
            return self._build_contextual_classification_prompt(
                query=query,
                session=session,
                query_history=query_history,
                available_context=available_context
            )
        else:
            return self._build_standalone_classification_prompt(
                query=query,
                session=session
            )

    def _build_standalone_classification_prompt(self, query: str, session: Session) -> Dict[str, str]:
        """Build prompt for initial query without prior context"""
        system_prompt = SCHEMA_AWARE_PROMPT
        user_prompt = f"""
Classify and analyze this query: "{query}"

Session Info:
- Session ID: {session.session_id}
- User Preferences: {json.dumps(session.user_preferences)}
- This is the first interaction in the session
"""

        return {
            "system": system_prompt,
            "user": user_prompt
        }

    def _build_contextual_classification_prompt(
            self,
            query: str,
            session: Session,
            query_history: List[QueryContext],
            available_context: Dict[str, Any]
    ) -> Dict[str, str]:
        """Build prompt for follow-up queries with context injection"""

        # Get most recent completed query
        last_query = query_history[0]

        # Build system prompt with schema + continuation rules
        system_prompt = f"""{SCHEMA_AWARE_PROMPT}

### CONTEXTUAL UNDERSTANDING INSTRUCTIONS

You are analyzing a follow-up query. Your tasks:
1. Determine if this is a continuation of the previous query
2. Resolve pronouns like “them”, “there”, “same time” using prior results
3. Identify which parameters to inherit vs. modify
4. Apply validation rules and parameter mapping
5. Flag any ambiguities needing clarification

### Previous Query Summary:
Category: {last_query.category.value if last_query.category else 'unknown'}
Mapped Parameters:
{json.dumps(last_query.extracted_params, indent=2)}
Sample Results:
{json.dumps(last_query.result_entities[:3], indent=2) if last_query.result_entities else 'No results'}
"""

        # Build user prompt with current input + context
        user_prompt = f"""
Current Query: "{query}"

## Recent Interaction Summary:
{self._inject_chat_history(query_history[:3])}

## Previous Query Context:
- Previous Entities Detected: {json.dumps(available_context.get('entities_detected', {}), indent=2)}
- Previous Mapped Parameters: {json.dumps(available_context.get('mapped_parameters', {}), indent=2)}

## Query History:
{json.dumps(available_context.get('previous_queries', []), indent=2)}

IMPORTANT:
1. Apply all entity extraction rules from schema
2. Validate formats (e.g., IMSI must be 15 digits)
3. Map to correct database fields
4. Flag any ambiguities or validation issues

CRITICAL: Return ONLY valid JSON following the response format in the schema.
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
