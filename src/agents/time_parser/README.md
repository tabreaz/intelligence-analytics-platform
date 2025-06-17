# Time Parser Agent

The Time Parser Agent extracts and parses time expressions from natural language queries to provide structured time data
for SQL generation and other downstream processing.

## Features

### Core Features

- **Absolute Date Extraction**: Parses specific dates like "2024-01-15" or "January 15th"
- **Relative Date Parsing**: Handles expressions like "yesterday", "last week", "3 days ago"
- **Time of Day Constraints**: Understands "morning", "evening", "9 to 5", etc.
- **Day of Week Constraints**: Processes "weekend", "weekday", "Monday through Friday"
- **Context-Aware Parsing**: Can use previous time ranges from conversation context
- **Default Range Fallback**: Provides sensible defaults when no time is specified

### Advanced Features (NEW)

- **Exclusion Support**: Handle queries like "except weekends", "excluding 10th hour"
- **Specific Selection**: Support "only Fridays", "just morning hours"
- **Date Expansion**: Convert ranges to individual dates for array-based queries
- **Composite Constraints**: Complex queries like "only Fridays between 12-14 hrs for last 3 months"
- **Event Mapping**: Map events like "Ramadan 2024" to specific date ranges
- **SQL Generation Hints**: Provides hints for optimal SQL query generation

## Architecture

```
time_parser/
├── __init__.py         # Package exports
├── agent.py            # Main agent implementation
├── models.py           # Data models (DateRange, HourConstraint, etc.)
├── constants.py        # Time indicators, patterns, and mappings
├── exceptions.py       # Custom exception classes
├── response_parser.py  # LLM response parsing and validation
├── example_usage.py    # Usage examples and testing
└── README.md          # This file
```

## Usage

### Basic Usage

```python
from src.agents.time_parser import TimeParserAgent
from src.agents.base_agent import AgentRequest

# Create agent
agent = TimeParserAgent('time_parser', config, config_manager)

# Parse time from query
request = AgentRequest(
    prompt="Show me data from yesterday morning"
)
response = await agent.process(request)

# Access results
time_data = response.result
if time_data['has_time_expressions']:
    print(f"Date ranges: {time_data['date_ranges']}")
    print(f"Hour constraints: {time_data['hour_constraints']}")
```

### Response Structure

```json
{
    "has_time_expressions": true,
    "date_ranges": [
        {
            "type": "relative",
            "start": "2024-01-14T00:00:00",
            "end": "2024-01-14T23:59:59",
            "original_text": "yesterday",
            "confidence": 0.95,
            "constraint_type": "include",
            "expand_to_dates": false
        }
    ],
    "excluded_date_ranges": [],
    "expanded_dates": null,
    "hour_constraints": [
        {
            "start_hour": 6,
            "end_hour": 12,
            "original_text": "morning",
            "constraint_type": "include",
            "excluded_hours": null,
            "days_applicable": null
        }
    ],
    "day_constraints": [
        {
            "days": ["weekday"],
            "constraint_type": "include",
            "original_text": "weekday"
        }
    ],
    "composite_constraints": null,
    "event_mappings": null,
    "raw_expressions": ["yesterday", "morning"],
    "summary": "1 date range(s) | Time: morning | Days: weekday",
    "parsing_confidence": 0.95,
    "default_range": null,
    "sql_hints": {
        "use_date_array": false,
        "use_hour_filter": true,
        "use_day_of_week": true,
        "suggested_approach": "complex_conditions"
    }
}
```

### With Context

```python
# Use previous time ranges as context
context = {
    'previous_time_ranges': [
        {
            "start": "2024-01-10T00:00:00",
            "end": "2024-01-15T23:59:59",
            "original_text": "last week"
        }
    ]
}

request = AgentRequest(
    prompt="Show morning activity",
    context=context
)
```

## Configuration

Add to `config/agents.yaml`:

```yaml
time_parser:
  enabled: true
  priority: 1
  llm_model: gpt-4
  default_time_range_days: 2  # Default range when no time specified
  timezone: UTC              # Timezone for parsing
  enable_context: true       # Use previous time ranges
  enable_training_data: true # Log to PostgreSQL for training
  cache_ttl: 300
```

## Time Mappings

### Hour of Day

- `morning`: 6:00 - 12:00
- `afternoon`: 12:00 - 18:00
- `evening`: 18:00 - 22:00
- `night`: 22:00 - 6:00

### Relative Times

- `yesterday`: Previous calendar day
- `today`: Current calendar day
- `last week`: Previous 7 days
- `last month`: Previous 30 days (approximate)

### Day Constraints

- `weekday`: Monday through Friday
- `weekend`: Saturday and Sunday
- Individual days: `monday`, `tuesday`, etc.

## Integration with Query Executor

The Time Parser Agent is designed to work seamlessly with the Query Executor:

```python
# First, understand the query
query_result = await query_understanding_agent.process(query)

# Extract time constraints
time_result = await time_parser_agent.process(query)

# Generate SQL with time constraints
sql_params = {
    **query_result['parameters'],
    'time_constraints': time_result['date_ranges']
}
```

## Testing

Run the example usage:

```bash
python -m src.agents.time_parser.example_usage
```

Run unit tests:

```bash
pytest tests/test_time_parser.py -v
```

## Advanced Usage Examples

### Exclusion Queries

```python
# Exclude weekends
query = "Show data from last week except weekends"
# Results in: date_ranges with last week + excluded_day_constraints for weekend

# Exclude specific hours
query = "Get all data from yesterday excluding the 10th hour"
# Results in: hour_constraints with excluded_hours: [10]

# Complex exclusion
query = "Data from 9-5 except lunch hour (12-13)"
# Results in: hour_constraints for 9-17 with excluded_hours: [12, 13]
```

### Specific Selection Queries

```python
# Only specific days
query = "Only Fridays from the last 3 months"
# Results in: date_ranges for 3 months + day_constraints include only "friday"

# Complex day-hour combination
query = "Only Fridays between 12-14 hrs for last 3 months"
# Results in: composite_constraints with day_hour_combinations
```

### SQL Generation

```python
from src.agents.time_parser.sql_helper import TimeSQLHelper

# Get time parsing result
time_result = await time_parser.process(request)

# Generate SQL WHERE clause
where_clause = TimeSQLHelper.generate_where_clause(
    time_result.result,
    timestamp_column="event_time"
)

# Generate complete example query
sql_query = TimeSQLHelper.generate_example_query(
    time_result.result,
    table_name="user_events",
    timestamp_column="event_time"
)
```

## PostgreSQL Integration

The Time Parser Agent integrates with PostgreSQL for training data collection:

### With Session Manager

```python
from src.core.session_manager import PostgreSQLSessionManager

# Initialize session manager
session_manager = PostgreSQLSessionManager(db_config)
await session_manager.initialize()

# Create agent with session manager
agent = TimeParserAgent('time_parser', config, config_manager, session_manager)

# Process with session context
request = AgentRequest(
    prompt="Show only Fridays for last 3 months",
    context={
        'session_id': session.session_id,
        'query_id': unique_query_id
    }
)
```

### Training Data Storage

When enabled, the agent logs:

- **Interaction Events**: LLM calls, timing, tokens, costs
- **Training Examples**: Query text, extracted time data, confidence scores
- **Performance Metrics**: Success rates, parsing accuracy

This data is stored in:

- `public.interaction_events` - LLM interaction details
- `public.training_examples` - Successful parsing examples

### Benefits

1. **Model Improvement**: Collected data helps improve time parsing accuracy
2. **Performance Tracking**: Monitor parsing success rates and patterns
3. **Debugging**: Trace issues with specific time expressions
4. **Analytics**: Understand common time query patterns

## Error Handling

The agent handles various error cases:

- Invalid date formats
- Ambiguous time expressions
- Missing time indicators
- LLM parsing failures

All errors are logged and the agent returns a valid structure with empty results rather than failing completely.