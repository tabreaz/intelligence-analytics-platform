#!/usr/bin/env python3
"""
Example usage of Time Parser Agent

This script demonstrates how to use the TimeParserAgent to extract
time expressions from natural language queries.
"""
import asyncio
import json
import uuid
from typing import Dict, Any

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

from src.agents.time_parser.agent import TimeParserAgent
from src.agents.base_agent import AgentRequest
from src.core.config_manager import ConfigManager
from src.core.logger import get_logger

logger = get_logger(__name__)

# Example queries to test
EXAMPLE_QUERIES = [
    # Absolute dates
    "Show many all the IRNs who are commuting for work between DXB and AUH?",
    "Show many people socializing near 24.2345, 54.23422 coordinates on regular basis?",
    # "What happened on 2024-01-15?",
    # "Get records from last Monday to Friday",
    #
    # # Relative dates
    # "Show activity from the last 7 days",
    # "What happened in the past week?",
    # "Show me data from 3 days ago",
    #
    # # Time of day constraints
    # "Show morning activity yesterday",
    # "What happened yesterday evening between 6 and 9 PM?",
    # "Get data from last night",
    #
    # # Combined constraints
    # "Show weekend activity from last week",
    # "Get morning data from the past 3 days",
    # "What happened on weekdays last month?",
    #
    # # Complex queries
    # "Show me all activity from 9 AM to 5 PM on weekdays last week",
    # "Get data from yesterday morning until today afternoon",
    # "What happened between January 10th and January 15th in the evening?",
    #
    # # Exclusion queries (NEW)
    # "Show data from last week except weekends",
    # "Get all data from yesterday excluding the 10th hour",
    # "Show activity from last month but not on Mondays",
    # "Data from 9-5 except lunch hour (12-13)",
    #
    # # Specific selection queries (NEW)
    # "Only Fridays from the last 3 months",
    # "Just morning hours on weekdays",
    # "Only Fridays between 12-14 hrs for last 3 months",
    # "Show data for June 1st through 5th",
    #
    # # Event-based queries (NEW)
    # "Show data from Ramadan 2024",
    # "Get records from UAE National Day week",
    # "Activity during Eid holidays",
    #
    # # No time expressions
    # "Show me all users",
    # "Get total count of records",
    # "List all available categories"
]


def print_result(query: str, result: Dict[str, Any], execution_time: float):
    """Pretty print parsing results"""
    print("\n" + "=" * 80)
    print(f"Query: {query}")
    print(f"Execution time: {execution_time:.3f}s")
    print("-" * 80)

    if result.get('error'):
        print(f"ERROR: {result['error']}")
        return

    # Check if time expressions were found
    if not result.get('has_time_expressions'):
        print("No time expressions found in query")
        if result.get('default_range'):
            print(f"Default range: {result['default_range']}")
        return

    # Print summary
    print(f"Summary: {result.get('summary', 'N/A')}")
    print(f"Confidence: {result.get('parsing_confidence', 0):.2%}")

    # Print date ranges
    if result.get('date_ranges'):
        print("\nDate Ranges (Include):")
        for idx, dr in enumerate(result['date_ranges'], 1):
            print(f"  {idx}. {dr['original_text']}:")
            print(f"     Start: {dr['start']}")
            print(f"     End:   {dr['end']}")
            print(f"     Type:  {dr['type']}")
            if dr.get('expand_to_dates'):
                print(f"     Expand: Yes")

    # Print excluded date ranges
    if result.get('excluded_date_ranges'):
        print("\nDate Ranges (Exclude):")
        for idx, dr in enumerate(result['excluded_date_ranges'], 1):
            print(f"  {idx}. {dr['original_text']}:")
            print(f"     Start: {dr['start']}")
            print(f"     End:   {dr['end']}")

    # Print expanded dates if available
    if result.get('expanded_dates'):
        print("\nExpanded Dates:")
        for ed in result['expanded_dates']:
            print(f"  Source: {ed['source_range']['original_text']}")
            print(f"  Dates: {len(ed['dates'])} individual dates")
            if len(ed['dates']) <= 10:
                for d in ed['dates'][:5]:
                    print(f"    - {d}")
                if len(ed['dates']) > 5:
                    print(f"    ... and {len(ed['dates']) - 5} more")

    # Print hour constraints
    if result.get('hour_constraints'):
        print("\nHour Constraints:")
        for idx, hc in enumerate(result['hour_constraints'], 1):
            constraint_type = hc.get('constraint_type', 'include')
            print(f"  {idx}. {hc['original_text']} ({constraint_type}):")
            print(f"     Hours: {hc['start_hour']:02d}:00 - {hc['end_hour']:02d}:00")
            if hc.get('days_applicable'):
                print(f"     Days: {', '.join(hc['days_applicable'])}")
            if hc.get('excluded_hours'):
                print(f"     Excluded hours: {hc['excluded_hours']}")

    # Print day constraints
    if result.get('day_constraints'):
        print("\nDay Constraints:")
        for dc in result['day_constraints']:
            constraint_type = dc.get('constraint_type', 'include')
            days = dc.get('days', [])
            print(f"  {dc.get('original_text', '')} ({constraint_type}): {', '.join(days)}")

    # Print composite constraints
    if result.get('composite_constraints'):
        print("\nComposite Constraints:")
        print(f"  {json.dumps(result['composite_constraints'], indent=4)}")

    # Print event mappings
    if result.get('event_mappings'):
        print("\nEvent Mappings:")
        for em in result['event_mappings']:
            print(f"  {em['event_name']}: {len(em['dates'])} dates")

    # Print SQL hints
    if result.get('sql_hints'):
        print("\nSQL Generation Hints:")
        for key, value in result['sql_hints'].items():
            print(f"  {key}: {value}")

    # Print raw expressions
    if result.get('raw_expressions'):
        print(f"\nRaw Expressions Found: {', '.join(result['raw_expressions'])}")


async def test_time_parser(query: str, agent: TimeParserAgent):
    """Test time parser with a single query"""
    try:
        # Create request
        request = AgentRequest(
            request_id=str(uuid.uuid4()),
            prompt=query,
            context={}
        )

        # Validate first
        is_valid = await agent.validate_request(request)
        if not is_valid:
            print(f"\nQuery '{query}' - No time expressions detected (skipped)")
            return

        # Process request
        response = await agent.process(request)

        # Print results
        print_result(query, response.result, response.execution_time)

        return response.result

    except Exception as e:
        logger.error(f"Failed to process query '{query}': {e}")
        print(f"\nERROR processing '{query}': {e}")
        return None


async def test_with_context():
    """Test time parser with context from previous queries"""
    print("\n" + "=" * 80)
    print("TESTING WITH CONTEXT")
    print("=" * 80)

    # Initialize
    config_manager = ConfigManager()
    agent_config = config_manager.get_agent_config('time_parser')
    agent = TimeParserAgent('time_parser', agent_config, config_manager)

    # First query establishes a time range
    query1 = "Show me data from last week"
    request1 = AgentRequest(
        request_id=str(uuid.uuid4()),
        prompt=query1,
        context={}
    )
    response1 = await agent.process(request1)
    print_result(query1, response1.result, response1.execution_time)

    # Second query uses context from first
    query2 = "Now show me just the morning activity"
    context = {
        'previous_time_ranges': response1.result.get('date_ranges', [])
    }
    request2 = AgentRequest(
        request_id=str(uuid.uuid4()),
        prompt=query2,
        context=context
    )
    response2 = await agent.process(request2)
    print_result(query2, response2.result, response2.execution_time)


async def test_with_session_manager():
    """Test time parser with session manager for training data"""
    print("\n" + "=" * 80)
    print("TESTING WITH SESSION MANAGER")
    print("=" * 80)

    try:
        # Initialize with session manager
        from src.core.session_manager import PostgreSQLSessionManager

        config_manager = ConfigManager()

        # Get database config
        db_config = {
            'host': 'localhost',
            'port': 5432,
            'user': 'tabreaz',
            'password': 'admin',
            'database': 'sigint'
        }

        # Initialize session manager
        session_manager = PostgreSQLSessionManager(db_config)
        await session_manager.initialize()

        # Create session
        session = await session_manager.create_session()
        print(f"Created session: {session.session_id}")

        # Initialize agent with session manager
        agent_config = config_manager.get_agent_config('time_parser')
        agent = TimeParserAgent('time_parser', agent_config, config_manager, session_manager)

        # Test query with session context
        query = "Show only Fridays between 12-14 hrs for last 3 months"
        request = AgentRequest(
            request_id=str(uuid.uuid4()),
            prompt=query,
            context={
                'session_id': session.session_id,
                'query_id': str(uuid.uuid4())
            }
        )

        response = await agent.process(request)
        print_result(query, response.result, response.execution_time)

        print("\nTraining data has been logged to PostgreSQL")

        # Close session manager
        await session_manager.close()

    except Exception as e:
        print(f"Session manager test failed: {e}")
        logger.error("Session manager test error", exc_info=True)


async def main():
    """Run example tests"""
    print("Time Parser Agent Example Usage")
    print("==============================")

    # Initialize config manager
    config_manager = ConfigManager()

    # Get agent configuration
    agent_config = config_manager.get_agent_config('time_parser')
    if not agent_config:
        print("Error: time_parser agent not found in configuration")
        return

    # Create agent instance
    agent = TimeParserAgent('time_parser', agent_config, config_manager)

    # Test individual queries
    print("\nTesting individual queries...")
    for query in EXAMPLE_QUERIES:
        await test_time_parser(query, agent)
        await asyncio.sleep(0.1)  # Small delay between requests

    # Test with context
    await test_with_context()

    # Test with session manager
    await test_with_session_manager()

    # Summary statistics
    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
