# src/agents/risk_filter/example_usage.py
"""
Example usage of Risk Filter Agent with SQL generation
"""
import asyncio
import uuid
from datetime import datetime

from dotenv import load_dotenv

from src.agents.base_agent import AgentRequest
from src.agents.risk_filter.agent import RiskFilterAgent
from src.agents.risk_filter.sql_helper import RiskFilterSQLHelper
from src.core.config_manager import ConfigManager

# Load environment variables
load_dotenv()


async def demonstrate_risk_filter_agent():
    """Demonstrate Risk Filter Agent with various queries"""

    # Initialize config manager
    config_manager = ConfigManager()

    # Get agent config
    agent_config = config_manager.get_agent_config('risk_filter')

    # Initialize agent and SQL helper (without session_manager for testing)
    agent = RiskFilterAgent(
        name='risk_filter',
        config=agent_config,
        config_manager=config_manager,
        session_manager=None
    )
    sql_helper = RiskFilterSQLHelper()

    # Example queries
    queries = [
        "Find high risk drug dealers who are not diplomats",
        "Show criminals with violent crimes but excluding financial crimes",
        "List people with risk scores between 0.4 and 0.8 who are under investigation",
        "Find dangerous individuals with murder score > 0.5 AND has crime cases"
    ]

    print("=" * 80)
    print("RISK FILTER AGENT - EXAMPLE USAGE")
    print("=" * 80)

    for i, query in enumerate(queries, 1):
        print(f"\n{'=' * 60}")
        print(f"Example {i}: {query}")
        print('=' * 60)

        # Create request
        request = AgentRequest(
            request_id=str(uuid.uuid4()),
            prompt=query,
            context={
                "session_id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat()
            },
            timestamp=datetime.utcnow().timestamp()
        )

        try:
            # Process query
            response = await agent.process(request)

            if response.status.value == "completed" and response.result:
                print("\n1. EXTRACTED FILTERS:")
                print("-" * 30)

                # Pretty print the extracted data
                data = response.result

                print(data)

                # Print reasoning if available
                if data.get('reasoning'):
                    print(f"Reasoning: {data['reasoning']}\n")

                # Inclusions
                if data.get('risk_scores') or data.get('flags') or data.get('crime_categories'):
                    print("Inclusions:")
                    if data.get('risk_scores'):
                        for field, filter_data in data['risk_scores'].items():
                            print(f"  • {field} {filter_data['operator']} {filter_data['value']}", end="")
                            if 'value2' in filter_data:
                                print(f" AND {filter_data['value2']}")
                            else:
                                print()

                    if data.get('flags'):
                        for field, value in data['flags'].items():
                            print(f"  • {field} = {value}")

                    if data.get('crime_categories'):
                        cats = data['crime_categories']
                        if cats.get('include'):
                            print(f"  • Crime categories: {', '.join(cats['include'])}")
                        if cats.get('severity_filter'):
                            print(f"  • Severity: {cats['severity_filter']}")

                # Exclusions
                exclusions = data.get('exclusions', {})
                if exclusions.get('risk_scores') or exclusions.get('flags'):
                    print("\nExclusions:")
                    if exclusions.get('risk_scores'):
                        for field, filter_data in exclusions['risk_scores'].items():
                            print(f"  • NOT {field} {filter_data['operator']} {filter_data['value']}")
                    if exclusions.get('flags'):
                        for field, value in exclusions['flags'].items():
                            print(f"  • NOT {field} = {value}")

                # Check for excluded crime categories
                if data.get('crime_categories') and data['crime_categories'].get('exclude'):
                    print(f"  • Excluded categories: {', '.join(data['crime_categories']['exclude'])}")

                # Generate SQL conditions
                print("\n2. SQL WHERE CONDITIONS:")
                print("-" * 30)

                # Reconstruct RiskFilterResult for SQL generation
                from src.agents.risk_filter.models import RiskFilterResult, ScoreFilter, OperatorType, CategoryFilter

                result = RiskFilterResult()

                # Reconstruct risk scores
                for field, filter_data in data.get('risk_scores', {}).items():
                    result.risk_scores[field] = ScoreFilter(
                        field=field,
                        operator=OperatorType(filter_data['operator']),
                        value=filter_data['value'],
                        value2=filter_data.get('value2')
                    )

                # Set flags
                result.flags = data.get('flags', {})

                # Reconstruct crime categories
                if data.get('crime_categories'):
                    cats = data['crime_categories']
                    result.crime_categories = CategoryFilter(
                        include=cats.get('include', []),
                        exclude=cats.get('exclude', []),
                        severity_filter=cats.get('severity_filter')
                    )

                # Reconstruct exclusions
                for field, filter_data in exclusions.get('risk_scores', {}).items():
                    result.exclude_scores[field] = ScoreFilter(
                        field=field,
                        operator=OperatorType(filter_data['operator']),
                        value=filter_data['value'],
                        value2=filter_data.get('value2')
                    )
                result.exclude_flags = exclusions.get('flags', {})

                # Generate SQL
                sql_conditions = sql_helper.build_sql_conditions(result)
                if sql_conditions:
                    for condition in sql_conditions:
                        print(f"  • {condition}")

                # Complete WHERE clause
                where_clause = sql_helper.build_complete_where_clause(result)
                print(f"\nComplete WHERE clause:")
                print(f"WHERE {where_clause}")

                # Required columns
                required_columns = sql_helper.get_required_columns(result)
                if required_columns:
                    print(f"\nRequired columns: {', '.join(required_columns)}")

                # Metadata
                print(f"\n3. METADATA:")
                print("-" * 30)
                if response.metadata and 'confidence' in response.metadata:
                    print(f"Confidence: {response.metadata['confidence']:.2f}")
                print(f"Processing time: {response.execution_time:.3f}s")
                if data.get('validation_warnings'):
                    print(f"Warnings: {', '.join(data['validation_warnings'])}")

            else:
                print(f"Error: {response.error}")

        except Exception as e:
            print(f"Exception: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(demonstrate_risk_filter_agent())
