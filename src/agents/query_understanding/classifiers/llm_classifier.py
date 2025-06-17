# src/agents/query_understanding/classifiers/llm_classifier.py
"""
LLM Classifier for Query Understanding
"""
import json
from typing import Dict, Any

import openai
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.logger import get_logger

logger = get_logger(__name__)


class LLMClassifier:
    """
    Handles LLM interactions for classification
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = config.get('model', 'gpt-4')
        self.temperature = config.get('temperature', 0.1)
        self.max_tokens = config.get('max_tokens', 1000)

        # Initialize OpenAI client (or your LLM of choice)
        openai.api_key = config.get('api_key')

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def classify(self,
                       system_prompt: str,
                       user_prompt: str) -> Dict[str, Any]:
        """
        Call LLM for classification
        Returns parsed JSON response
        """
        try:
            # Make the API call
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}  # If using GPT-4 Turbo
            )

            # Extract the response
            content = response.choices[0].message.content

            # Parse JSON response
            try:
                result = json.loads(content)
                logger.info(f"LLM Classification successful: {result.get('classification', {}).get('category')}")
                return result
            except json.JSONDecodeError:
                logger.error(f"Failed to parse LLM response as JSON: {content}")
                # Attempt to extract JSON from the response
                return self._extract_json_from_text(content)

        except Exception as e:
            logger.error(f"LLM classification error: {e}")
            # Return a fallback classification
            return self._fallback_classification(user_prompt)

    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """
        Try to extract JSON from text that might have extra content
        """
        import re

        # Look for JSON-like structure
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass

        # If all else fails, return minimal structure
        return {
            "classification": {
                "category": "multi_criteria",
                "confidence": 0.3,
                "reasoning": "Failed to parse LLM response"
            },
            "ambiguities": [{
                "parameter": "query",
                "issue": "Could not understand the query",
                "suggested_clarification": "Could you please rephrase your question?"
            }]
        }

    def _fallback_classification(self, query: str) -> Dict[str, Any]:
        """
        Fallback classification when LLM fails
        Uses simple keyword matching
        """
        query_lower = query.lower()

        # Simple keyword-based classification
        if any(word in query_lower for word in ['who visited', 'who was at', 'visited']):
            category = 'location_time'
        elif any(word in query_lower for word in ['meets', 'meeting', 'co-location']):
            category = 'co_location'
        elif any(word in query_lower for word in ['syrian', 'iranian', 'risk', 'diplomat']):
            category = 'profile_search'
        elif any(word in query_lower for word in ['movement', 'track', 'path']):
            category = 'movement_tracking'
        else:
            category = 'multi_criteria'

        return {
            "classification": {
                "category": category,
                "confidence": 0.5,
                "reasoning": "Fallback classification based on keywords"
            },
            "entities_detected": {
                "locations": [],
                "identifiers": [],
                "time_references": [],
                "nationalities": [],
                "risk_indicators": []
            },
            "initial_parameters": {},
            "ambiguities": []
        }


# Example showing actual prompt and response
def example_first_llm_interaction():
    """
    This shows exactly what the first LLM sees and returns
    """

    # Example 1: Standalone Query
    print("=== EXAMPLE 1: Standalone Query ===")
    print("\nQUERY: 'Find all Syrian diplomats who visited Dubai Mall last week'\n")

    print("SYSTEM PROMPT:")
    print("""You are a query understanding system for telecom movement analysis.
Your task is to classify user queries and extract key information.

## Available Query Categories:

### Location Based
- **location_time**: Queries about who visited specific locations at specific times
  - Examples: Who visited Dubai Mall yesterday?, Show everyone at the airport last week
  - Required: location, time_range

### Profile Based  
- **profile_search**: Search for individuals by profile characteristics
  - Examples: Find all Iranian males aged 25-35 with high risk scores, Show diplomats who have crime cases
  - Required: search_criteria

[... more categories ...]

## Response Format:
{json format specification}
""")

    print("\nUSER PROMPT:")
    print("""Classify and analyze this query: "Find all Syrian diplomats who visited Dubai Mall last week"

Session context:
- Session ID: 123e4567-e89b-12d3-a456-426614174000
- User preferences: {}
- This is the first query in this session""")

    print("\nEXPECTED LLM RESPONSE:")
    print(json.dumps({
        "classification": {
            "category": "location_time",
            "confidence": 0.95,
            "reasoning": "Query asks for people (Syrian diplomats) who visited a specific location (Dubai Mall) at a specific time (last week)"
        },
        "entities_detected": {
            "locations": ["Dubai Mall"],
            "identifiers": [],
            "time_references": ["last week"],
            "nationalities": ["SYR"],
            "risk_indicators": []
        },
        "initial_parameters": {
            "location": "Dubai Mall",
            "time_range": "last_week",
            "nationality": "SYR",
            "is_diplomat": True
        },
        "ambiguities": [],
        "context_requirements": {
            "needs_location_context": True,
            "needs_time_context": True,
            "needs_profile_context": True
        }
    }, indent=2))

    print("\n\n=== EXAMPLE 2: Contextual Query ===")
    print("\nQUERY: 'What about Iranians?'\n")

    print("SYSTEM PROMPT: [Same categories + context inheritance rules]\n")

    print("USER PROMPT:")
    print("""Current query: "What about Iranians?"

Previous query: "Find all Syrian diplomats who visited Dubai Mall last week"

Session context:
- Active filters: {"nationality": "SYR", "is_diplomat": true, "location": "Dubai Mall", "time_range": "last_week"}
- Time range context: last_week
- Location context: ["Dubai Mall"]
- Profile context: {"nationality": "SYR", "is_diplomat": true}

Recent interaction summary:
1. Find all Syrian diplomats who visited Dubai Mall last week → location_time (47 results)

Analyze this query considering the conversation context.""")

    print("\nEXPECTED LLM RESPONSE:")
    print(json.dumps({
        "classification": {
            "category": "location_time",
            "confidence": 0.9,
            "reasoning": "This is a continuation of the previous location_time query, just changing the nationality filter",
            "is_continuation": True
        },
        "context_inheritance": {
            "inherit_from_previous": True,
            "inheritance_type": "partial",
            "inherited_parameters": {
                "location": {
                    "value": "Dubai Mall",
                    "source": "previous_query",
                    "override": False
                },
                "time_range": {
                    "value": "last_week",
                    "source": "previous_query",
                    "override": False
                },
                "is_diplomat": {
                    "value": True,
                    "source": "previous_query",
                    "override": False
                }
            },
            "reference_resolution": {}
        },
        "new_parameters": {},
        "modified_parameters": {
            "nationality": {
                "old_value": "SYR",
                "new_value": "IRN",
                "modification_type": "replace"
            }
        },
        "entities_detected": {
            "locations": [],
            "identifiers": [],
            "time_references": [],
            "nationalities": ["IRN"],
            "risk_indicators": []
        },
        "ambiguities": []
    }, indent=2))


if __name__ == "__main__":
    example_first_llm_interaction()
