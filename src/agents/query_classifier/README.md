# Query Classifier Agent

The Query Classifier Agent is responsible for:

1. Resolving context in continuation queries
2. Classifying queries into predefined categories
3. Identifying unsupported domains

## Key Features

### Context Resolution

- Detects continuation queries ("how about X", "what about Y")
- Resolves references from previous queries
- Builds complete context-aware queries

### Query Classification

- Classifies into categories defined in `config/query_categories.yaml`
- Identifies unsupported domains (weather, news, etc.)
- Provides confidence scores

### History Management

The agent returns a `store_in_history` flag to indicate whether the query should be stored in conversation history:

```python
result = await query_classifier.process(request)

if result['store_in_history']:
    # Store in session history for future context
    await session_manager.add_query_to_session(session_id, query_context)
else:
    # Skip storing (e.g., unsupported_domain queries)
    pass
```

## Usage Example

```python
from src.agents.query_classifier.agent import QueryClassifierAgent

# Initialize
agent = QueryClassifierAgent('query_classifier', config, config_manager)

# Process query with context
context = {
    'previous_query': {
        'query_text': 'Show high risk Syrians in Dubai',
        'category': 'profile_search',
        'extracted_params': {...}
    }
}

request = AgentRequest(
    request_id='...',
    prompt='how about Iranians',
    context=context
)

response = await agent.process(request)

# Response contains:
# - original_query: The user's input
# - context_aware_query: Fully resolved query
# - is_continuation: Boolean
# - classification: {category, confidence, reasoning}
# - domain_check: {is_supported, message}
# - store_in_history: Boolean flag
```

## Configuration

The agent uses categories defined in `config/query_categories.yaml`:

- Location-based queries
- Movement-based queries
- Profile-based queries
- Relationship-based queries
- Risk/Crime-based queries
- Geographic analysis
- Special categories (unsupported_domain, general_inquiry)

## Integration Notes

When integrating with an orchestrator:

1. **Pass Previous Context**: Only pass the last supported query as context
2. **Check store_in_history**: Use this flag to decide whether to store in SessionManager
3. **Handle Unsupported Domains**: Show appropriate message to user
4. **Use context_aware_query**: Pass this to downstream agents for processing