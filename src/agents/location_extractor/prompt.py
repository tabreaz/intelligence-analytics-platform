# src/agents/location_extractor/prompt.py
"""
Location Extractor Agent Prompts
"""

LOCATION_EXTRACTION_PROMPT = """
Extract ALL distinct PHYSICAL LOCATIONS where people are, were, or will be present. 
Return a JSON object with two arrays: "locations" for clear locations and "ambiguities" for vague references.

IMPORTANT EXCLUSIONS - DO NOT extract locations when:
1. Countries/cities appear as NATIONALITY (e.g., "Indian nationals", "UAE citizens")
2. Countries appear as TRAVEL HISTORY (e.g., "traveled to Syria", "visited Iraq last year")
3. Countries appear as PREVIOUS NATIONALITY (e.g., "Syrian refugees", "former Iraqi citizens")
4. The context is about someone's origin/background, not their physical presence
5. Countries appear as COMMUNICATION (e.g., "communicated with Syria", "calls to Iran", "messages from Pakistan")
6. Countries appear as VISA/DOCUMENT context (e.g., "Pakistani visa holders", "Iraqi passport")

ONLY extract locations when they indicate:
- Current physical presence ("people IN Dubai", "visitors AT the mall", "currently in Sharjah")
- Work/residence location ("living in Sharjah", "working in Dubai", "residing in Abu Dhabi")
- Movement/visit patterns ("went to Dubai Mall", "visited the airport", "traveled through Fujairah")
- Future plans with location ("will be at Dubai Expo", "meeting at DIFC tomorrow")

FIELD SELECTION RULES:
- For residence: use "home_city" or "residency_emirate"
- For workplace: use "work_location"
- For current presence/visits: use "visited_city" (for CITY/EMIRATE)
- For facilities: always use "geohash"

AMBIGUOUS LOCATIONS TO FLAG:
1. Generic Building Types (without specific names):
   - "government buildings" → Too many possibilities
   - "hospitals", "schools", "police stations", "shopping malls" → Which ones?
   - "banks", "hotels", "mosques" → Need specific names

2. Vague Area References:
   - "downtown", "city center" → Which city?
   - "industrial area", "tourist areas", "residential areas" → Too vague
   - "border areas", "coastal regions", "desert areas" → Too broad

3. Relative Locations Without Reference:
   - "nearby restaurants", "local shops" → Near what?
   - "the airport" → Which airport? (unless context is clear)
   - "surrounding areas", "that area" → What area?

4. Partial Names:
   - "the mall", "the park", "the beach" → Which specific one?
   - "the ministry", "the hospital" → Which one?

OUTPUT FORMAT:
{
  "locations": [
    {
      "location": "place name",
      "context": "spatial context (near, at, in, living, working, visiting)",
      "type": "CITY|EMIRATE|FACILITY|ADDRESS",
      "value": "normalized value",
      "field": "home_city|work_location|visited_city|geohash",
      "radius_meters": 200-5000 (only for FACILITY/ADDRESS),
      "confidence": 0.0-1.0
    }
  ],
  "ambiguities": [
    {
      "reference": "the ambiguous reference",
      "ambiguity_type": "generic_building_type|vague_area|relative_location|partial_name",
      "context": "full context from query",
      "suggestions": ["clarification question 1", "clarification question 2"],
      "potential_count": "estimated number of possibilities",
      "clarification_needed": true,
      "severity": "high|medium|low"
    }
  ]
}

SEVERITY GUIDELINES:
- "high": 50+ possible locations, critical for query understanding
- "medium": 10-50 possible locations, query can proceed with warnings
- "low": <10 possible locations, minor impact on results

EXAMPLES:

Input: "UAE nationals who traveled to Syria or communicated with contacts in Iran"
Output: {"locations": [], "ambiguities": []}
Reasoning: UAE is nationality, Syria is travel history, Iran is communication context - no physical locations

Input: "People near government buildings in Dubai"
Output: {
  "locations": [
    {"location": "Dubai", "context": "in", "type": "CITY", "value": "DUBAI", "field": "visited_city", "confidence": 0.95}
  ],
  "ambiguities": [
    {
      "reference": "government buildings",
      "ambiguity_type": "generic_building_type",
      "context": "near government buildings in Dubai",
      "suggestions": [
        "Which government buildings specifically?",
        "Examples: Dubai Municipality, Dubai Courts, RTA Head Office, GDRFA"
      ],
      "potential_count": "30+",
      "clarification_needed": true,
      "severity": "high"
    }
  ]
}

Input: "Syrian refugees living in Dubai visiting the mall"
Output: {
  "locations": [
    {"location": "Dubai", "context": "living", "type": "CITY", "value": "DUBAI", "field": "home_city", "confidence": 0.95}
  ],
  "ambiguities": [
    {
      "reference": "the mall",
      "ambiguity_type": "partial_name",
      "context": "visiting the mall",
      "suggestions": [
        "Which mall in Dubai?",
        "Options: Dubai Mall, Mall of Emirates, Ibn Battuta Mall, City Centre Deira"
      ],
      "potential_count": "15+",
      "clarification_needed": true,
      "severity": "medium"
    }
  ]
}
Reasoning: Syria is origin/nationality (excluded), Dubai is residence, "the mall" is ambiguous

Input: "Indians working in Dubai who visited Abu Dhabi Mall last week"
Output: {
  "locations": [
    {"location": "Dubai", "context": "working", "type": "CITY", "value": "DUBAI", "field": "work_location", "confidence": 0.95},
    {"location": "Abu Dhabi Mall", "context": "visited", "type": "FACILITY", "value": "Abu Dhabi Mall", "field": "geohash", "radius_meters": 300, "confidence": 0.95}
  ],
  "ambiguities": []
}
Reasoning: India is nationality (excluded), but Dubai and Abu Dhabi Mall are specific locations

Input: "Pakistani visa holders near banks"
Output: {
  "locations": [],
  "ambiguities": [
    {
      "reference": "banks",
      "ambiguity_type": "generic_building_type", 
      "context": "near banks",
      "suggestions": [
        "Which banks specifically?",
        "Examples: Emirates NBD, ADCB, FAB, Dubai Islamic Bank"
      ],
      "potential_count": "100+",
      "clarification_needed": true,
      "severity": "high"
    }
  ]
}
Reasoning: Pakistan is visa context (excluded), "banks" is too generic

Input: "People who were at DIFC yesterday and live in coastal areas"
Output: {
  "locations": [
    {"location": "DIFC", "context": "at", "type": "FACILITY", "value": "Dubai International Financial Centre", "field": "geohash", "radius_meters": 500, "confidence": 0.95}
  ],
  "ambiguities": [
    {
      "reference": "coastal areas",
      "ambiguity_type": "vague_area",
      "context": "live in coastal areas",
      "suggestions": [
        "Which coastal areas?",
        "Examples: JBR, Dubai Marina, Jumeirah, Corniche Abu Dhabi"
      ],
      "potential_count": "20+",
      "clarification_needed": true,
      "severity": "medium"
    }
  ]
}"""