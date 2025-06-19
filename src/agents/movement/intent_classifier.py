# src/agents/movement/intent_classifier.py
"""
Movement Intent Classifier - First step in movement analysis
"""
import json
from enum import Enum
from typing import Dict, Optional

from .utils import parse_json_response
from ...core.logger import get_logger

logger = get_logger(__name__)


class MovementIntent(Enum):
    """Movement analysis intent types"""
    SINGLE_LOCATION = "single_location"
    MULTI_LOCATION_AND = "multi_location_and"
    MULTI_LOCATION_OR = "multi_location_or"
    MULTI_LOCATION_THRESHOLD = "multi_location_threshold"
    MOVEMENT_PATTERN = "movement_pattern"
    CO_PRESENCE = "co_presence"
    HEATMAP = "heatmap"
    SEQUENCE_PATTERNS = "sequence_patterns"
    PATTERN_DETECTION = "pattern_detection"
    ANOMALY_DETECTION = "anomaly_detection"
    PREDICTIVE_ANALYSIS = "predictive_analysis"
    UNKNOWN = "unknown"


MOVEMENT_INTENT_CLASSIFIER_PROMPT = """# Movement Intent Classifier

You are a **Movement Analysis Intent Classifier** that determines the **type of movement-related question** being asked in a natural language query.

---

## 🎯 YOUR ROLE

1. Analyze the input query and determine which movement analysis pattern(s) it belongs to.
2. If ambiguity exists, list possible interpretations.
3. Return a structured classification result.

---

## 🧠 Supported Intents:

| Intent Type | Description | Keywords/Patterns |
|------------|-------------|-------------------|
| single_location | Find people at ONE specific location | "at", "in", "who were at", "present at" |
| multi_location_and | People at ALL specified locations | "both", "all", "and", "visited A and B" |
| multi_location_or | People at ANY of the locations | "either", "or", "any of", "at least one" |
| multi_location_threshold | People at minimum N of M locations | "at least N of", "minimum", "3 out of 5" |
| movement_pattern | Track movements of specific person/group | "track", "movements", "where was", "path" |
| co_presence | Detect when devices/people met | "meet", "together", "with", "co-located" |
| heatmap | Population density or frequency maps | "density", "heatmap", "crowded", "popular" |
| sequence_patterns | Ordered location visits | "then", "followed by", "sequence", "A to B to C" |
| pattern_detection | Recurring visits | "every", "recurring", "pattern", "regularly" |
| anomaly_detection | Unusual behavior | "unusual", "anomaly", "suspicious", "abnormal" |
| predictive_analysis | Forecasting | "predict", "forecast", "will be", "expected" |
| unknown | Cannot determine intent | Query doesn't match any pattern |

---

## 📌 CLASSIFICATION EXAMPLES

| Query | Intent(s) |
|-------|-----------|
| "Who was at Dubai Mall yesterday?" | single_location |
| "Find people who visited both Mall A and Mall B" | multi_location_and |
| "Show anyone who was at Mall A or Mall B" | multi_location_or |
| "People who visited at least 3 of these 5 malls" | multi_location_threshold |
| "Track movements of phone 0501234567" | movement_pattern |
| "Did phones X and Y meet last week?" | co_presence |
| "Show density heatmap for Downtown" | heatmap |
| "Find people who went from Airport to Hotel to Mall" | sequence_patterns |
| "Who visits the mosque every Friday?" | pattern_detection |
| "Detect unusual night movements" | anomaly_detection |
| "Predict tomorrow's crowd at the mall" | predictive_analysis |
| "Show heatmap of people who meet regularly" | heatmap, co_presence, pattern_detection |

---

## 🚨 CRITICAL RULES

1. **Multiple Intents**: If query combines patterns, return ALL applicable intents
2. **Priority Order**: If ambiguous, prefer more specific intents over general ones
3. **Ambiguity Handling**: Flag unclear parts that need clarification
4. **Avoid Unknown**: Only use if no pattern matches even loosely
5. **Context Clues**: Look for temporal words, location counts, identity mentions

---

## 📤 OUTPUT FORMAT

Return only the following JSON structure. Do not add any extra text or explanation.

```json
{
  "reasoning": "Explain how you interpreted the query",
  "primary_intent": "<main_intent_type>",
  "secondary_intents": ["<additional_intent_1>", "<additional_intent_2>"],
  "entities_detected": {
    "has_identities": true,
    "location_count": 2,
    "has_time_constraints": true,
    "has_specific_patterns": false
  },
  "ambiguities": [
    {
      "issue": "What part is unclear?",
      "suggested_clarification": "How to clarify?",
      "affects_intent": true
    }
  ],
  "confidence": 0.95
}
```
"""


class MovementIntentClassifier:
    """Classifies movement queries into intent categories"""
    
    def __init__(self):
        self.intent_keywords = {
            MovementIntent.SINGLE_LOCATION: [
                "at", "in", "who were at", "present at", "visited", "went to"
            ],
            MovementIntent.MULTI_LOCATION_AND: [
                "both", "all", "and", "each", "every location", "all places"
            ],
            MovementIntent.MULTI_LOCATION_OR: [
                "either", "or", "any", "any of", "at least one", "one of"
            ],
            MovementIntent.MULTI_LOCATION_THRESHOLD: [
                "at least", "minimum", "out of", "threshold", "more than"
            ],
            MovementIntent.MOVEMENT_PATTERN: [
                "track", "movements", "where was", "path", "trajectory", "route"
            ],
            MovementIntent.CO_PRESENCE: [
                "meet", "together", "with", "co-located", "same place", "encounter"
            ],
            MovementIntent.HEATMAP: [
                "density", "heatmap", "crowded", "popular", "busy", "concentration"
            ],
            MovementIntent.SEQUENCE_PATTERNS: [
                "then", "followed by", "sequence", "after", "before", "order"
            ],
            MovementIntent.PATTERN_DETECTION: [
                "every", "recurring", "pattern", "regularly", "routine", "habit"
            ],
            MovementIntent.ANOMALY_DETECTION: [
                "unusual", "anomaly", "suspicious", "abnormal", "strange", "outlier"
            ],
            MovementIntent.PREDICTIVE_ANALYSIS: [
                "predict", "forecast", "will be", "expected", "future", "anticipate"
            ]
        }
    
    async def classify(self, query: str, llm_client=None) -> Dict:
        """
        Classify query intent using LLM
        Falls back to keyword-based classification if LLM unavailable
        """
        if llm_client:
            return await self._llm_classify(query, llm_client)
        else:
            return self._keyword_classify(query)
    
    async def _llm_classify(self, query: str, llm_client) -> Dict:
        """Use LLM for intent classification"""
        try:
            response = await llm_client.complete(
                system_prompt=MOVEMENT_INTENT_CLASSIFIER_PROMPT,
                user_prompt=f"Classify this movement query: {query}"
            )
            
            return parse_json_response(response)
        except Exception as e:
            logger.warning(f"LLM classification failed, falling back to keywords: {e}")
            return self._keyword_classify(query)
    
    def _keyword_classify(self, query: str) -> Dict:
        """Fallback keyword-based classification"""
        query_lower = query.lower()
        detected_intents = []
        
        # Check each intent type
        for intent, keywords in self.intent_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_intents.append(intent.value)
        
        # Detect entities
        entities = {
            "has_identities": any(x in query_lower for x in ["imsi", "phone", "eid", "device"]),
            "location_count": query_lower.count("mall") + query_lower.count("location") + query_lower.count("place"),
            "has_time_constraints": any(x in query_lower for x in ["yesterday", "today", "last", "between", "during"]),
            "has_specific_patterns": any(x in query_lower for x in ["every", "pattern", "regularly"])
        }
        
        # Determine primary intent
        if not detected_intents:
            primary_intent = MovementIntent.UNKNOWN.value
        else:
            primary_intent = detected_intents[0]
            
        return {
            "reasoning": "Classified using keyword matching",
            "primary_intent": primary_intent,
            "secondary_intents": detected_intents[1:] if len(detected_intents) > 1 else [],
            "entities_detected": entities,
            "ambiguities": [],
            "confidence": 0.7 if detected_intents else 0.3
        }
    
    def get_prompt_for_intent(self, intent: str) -> Optional[str]:
        """Get the appropriate prompt template for the classified intent"""
        intent_prompt_mapping = {
            MovementIntent.SINGLE_LOCATION.value: "SINGLE_LOCATION_PROMPT",
            MovementIntent.MULTI_LOCATION_AND.value: "MULTI_LOCATION_PROMPT",
            MovementIntent.MULTI_LOCATION_OR.value: "MULTI_LOCATION_PROMPT",
            MovementIntent.MULTI_LOCATION_THRESHOLD.value: "MULTI_LOCATION_PROMPT",
            MovementIntent.MOVEMENT_PATTERN.value: "MOVEMENT_PATTERN_PROMPT",
            MovementIntent.CO_PRESENCE.value: "CO_PRESENCE_PROMPT",
            MovementIntent.HEATMAP.value: "ANALYTICS_PROMPT",
            MovementIntent.SEQUENCE_PATTERNS.value: "SEQUENCE_PATTERN_PROMPT",
            MovementIntent.PATTERN_DETECTION.value: "PATTERN_DETECTION_PROMPT",
            MovementIntent.ANOMALY_DETECTION.value: "ANOMALY_DETECTION_PROMPT",
            MovementIntent.PREDICTIVE_ANALYSIS.value: "PREDICTIVE_ANALYSIS_PROMPT"
        }
        
        return intent_prompt_mapping.get(intent)