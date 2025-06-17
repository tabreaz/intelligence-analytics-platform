# # src/agents/query_understanding/example_usage.py
# """
# Example showing how to use Query Understanding Controller + SQL Generation
# """
# import asyncio
# import json
# import os
# import sys
# from pathlib import Path
#
# from dotenv import load_dotenv
#
# # Add project root to path
# sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
#
# from src.agents.query_understanding.controller import QueryUnderstandingController
# from src.core.database.schema_manager import ClickHouseSchemaManager
# from src.core.session_manager import EnhancedSessionManager
# from src.agents.query_understanding.prompts.classification_prompt_builder import ClassificationPromptBuilder
# from src.agents.query_understanding.prompts.contextual_prompt_builder import ContextualPromptBuilder
# from src.agents.query_understanding.services.location_service import LocationExtractionService, GeohashStorageService
# from src.agents.location_extractor.agent import LocationExtractorAgent
# from src.core.llm.base_llm import LLMClientFactory
# from src.core.config_manager import ConfigManager
# from src.core.database.clickhouse_client import ClickHouseClient
# from src.core.logger import get_logger
#
# logger = get_logger(__name__)
#
#
# async def test_sql_generation(query_result: dict, sql_generator):
#     """Test SQL generation (without execution) for a query understanding result"""
#
#     print("\n" + "-" * 70)
#     print("SQL GENERATION TEST (No Execution)")
#     print("-" * 70)
#
#     # Extract the result part
#     controller_output = query_result.get('result', {})
#
#     # Add session_id and query_id if not present
#     if 'session_id' not in controller_output:
#         controller_output['session_id'] = query_result.get('session_id')
#     if 'query_id' not in controller_output:
#         controller_output['query_id'] = query_result.get('query_id')
#
#     try:
#         # Generate SQL directly using SQLGenerator
#         generation_result = await sql_generator.generate_sql(
#             controller_output,
#             max_correction_attempts=3
#         )
#
#         print(f"Generation Method: {generation_result.generation_method}")
#         print(f"Correction Attempts: {generation_result.correction_attempts}")
#
#         # Print SQL
#         print(f"\nGenerated SQL:")
#         print("-" * 70)
#         print(generation_result.sql)
#         print("-" * 70)
#
#         # Print result columns
#         print(f"\nExpected Result Columns:")
#         for col in generation_result.result_columns:
#             col_info = f"  - {col.name} ({col.type})"
#             if col.alias:
#                 col_info += f" [alias for: {col.alias}]"
#             print(col_info)
#
#         # Print token usage
#         print(f"\nToken Usage:")
#         print(f"  Prompt tokens: {generation_result.prompt_tokens}")
#         print(f"  Completion tokens: {generation_result.completion_tokens}")
#         print(f"  Total tokens: {generation_result.prompt_tokens + generation_result.completion_tokens}")
#
#     except Exception as e:
#         logger.error(f"SQL generation failed: {e}")
#         print(f"SQL generation failed: {str(e)}")
#
#
# async def main():
#     # Load environment variables
#     load_dotenv()
#
#     # Get project root directory
#     project_root = Path(__file__).parent.parent.parent.parent
#     config_dir = project_root / "config"
#
#     # Initialize config manager with correct config directory
#     config_manager = ConfigManager(config_dir=str(config_dir))
#
#     # 1. Get PostgreSQL configuration from config file
#     db_config = config_manager.get_database_config('postgresql')
#
#     # Convert to dict for session manager (it expects dict, not dataclass)
#     db_config_dict = {
#         'host': db_config.host,
#         'port': db_config.port,
#         'database': db_config.database,
#         'user': db_config.user,
#         'password': db_config.password
#     }
#
#     # 2. Get LLM configuration
#     # You can set default provider via environment variable or config
#     provider = os.getenv('LLM_PROVIDER', 'openai')
#     llm_config = config_manager.get_llm_config(provider)
#
#     # 3. Initialize components
#     session_manager = EnhancedSessionManager(db_config_dict)
#     await session_manager.initialize()
#
#     classification_builder = ClassificationPromptBuilder()
#     contextual_builder = ContextualPromptBuilder()
#     llm_client = LLMClientFactory.create_client(llm_config)
#
#     # 3a. Initialize location services
#     # Create location extractor agent
#     location_extractor = LocationExtractorAgent(
#         name="location_extractor",
#         config=config_manager.get_agent_config("location_extractor"),
#         config_manager=config_manager
#     )
#
#     # Create location service with session manager for tracking
#     location_service = LocationExtractionService(
#         location_extractor=location_extractor,
#         session_manager=session_manager
#     )
#
#     # Create ClickHouse client for geohash storage
#     ch_config = config_manager.get_database_config('clickhouse')
#     ch_client = ClickHouseClient(ch_config)
#     geohash_storage = GeohashStorageService(ch_client)
#
#     # 4. Create controller with location services
#     controller = QueryUnderstandingController(
#         session_manager=session_manager,
#         classification_builder=classification_builder,
#         contextual_builder=contextual_builder,
#         llm_client=llm_client,
#         location_service=location_service,
#         geohash_storage=geohash_storage
#     )
#
#     # 4a. Create SQL Generator for SQL generation only (no execution)
#     schema_manager = ClickHouseSchemaManager()
#     sql_validator = SQLValidator(schema_manager)
#     sql_generator = SQLGenerator(
#         llm_client=llm_client,
#         schema_manager=schema_manager,
#         sql_validator=sql_validator
#     )
#
#     # 5. Process first query (creates new session)
#     result1 = await controller.process_query(
#         query="Show me all Syrian nationals with high risk scores who visited Dubai Mall yesterday and spoken to QAT"
#     )
#
#     # 6. Print first query results
#     print("\n" + "=" * 50)
#     print("FIRST QUERY RESULTS")
#     print("=" * 50)
#     print(json.dumps(result1, indent=4))
#
#     # Extract session_id and query_id from the first query
#     session_id = result1.get('session_id')
#     query_id = result1.get('query_id')
#     print(f"\nUsing session_id: {session_id} for context continuity")
#     print(f"Query ID: {query_id}")
#     print(f"Has locations: {result1.get('has_locations')}")
#
#     # Test SQL generation for first query
#     if result1.get('result'):
#         await test_sql_generation(result1, sql_generator)
#
#     # 7. Process second query with same session (should inherit context)
#     result2 = await controller.process_query(
#         query="How about Iranians",
#         session_id=session_id  # Use same session to maintain context
#     )
#
#     # 8. Print second query results
#     print("\n" + "=" * 50)
#     print("SECOND QUERY RESULTS (WITH CONTEXT)")
#     print("=" * 50)
#     print(json.dumps(result2, indent=4))
#
#     # Test SQL generation for second query
#     if result2.get('result'):
#         await test_sql_generation(result2, sql_generator)
#
#     # Process third query - multi-location with time ranges
#     result3 = await controller.process_query(
#         query="Find people Al Bustan Towers within 200mts morning between 9-12 and find near 24.36712, 53.68987 between 15-18 hrs yesterday",
#         session_id=session_id  # Use same session to maintain context
#     )
#     print("\n" + "=" * 50)
#     print("THIRD QUERY RESULTS (Multi-location with time ranges)")
#     print("=" * 50)
#     print(json.dumps(result3, indent=4))
#
#     # Print location extraction details
#     if result3.get('has_locations'):
#         print(f"\nLocation Extraction Details:")
#         print(f"Query ID: {result3.get('query_id')}")
#         locations = result3.get('result', {}).get('entities_detected', {}).get('locations', {})
#         if locations:
#             print(f"Total locations: {locations.get('total_locations')}")
#             print(f"Geohash count: {locations.get('geohash_count')}")
#             print(f"Extracted by: {locations.get('extracted_by')}")
#
#     # Test SQL generation for third query
#     if result3.get('result'):
#         await test_sql_generation(result3, sql_generator)
#
#     result4 = await controller.process_query(
#         query="Restrict to US nationals",
#         session_id=session_id  # Use same session to maintain context
#     )
#     print("\n" + "=" * 50)
#     print("FOURTH QUERY RESULTS (Adding nationality filter)")
#     print("=" * 50)
#     print(json.dumps(result4, indent=4))
#
#     # Print location extraction details
#     if result4.get('has_locations'):
#         print(f"\nLocation Extraction Details:")
#         print(f"Query ID: {result4.get('query_id')}")
#         locations = result4.get('result', {}).get('entities_detected', {}).get('locations', {})
#         if locations:
#             print(f"Total locations: {locations.get('total_locations')}")
#             print(f"Geohash count: {locations.get('geohash_count')}")
#             print(f"Extracted by: {locations.get('extracted_by')}")
#
#     # Test SQL generation for fourth query
#     if result4.get('result'):
#         await test_sql_generation(result4, sql_generator)
#
#     print("\n" + "=" * 70)
#     print("EXAMPLE COMPLETED SUCCESSFULLY!")
#     print("=" * 70)
#     print("\nThis example demonstrated:")
#     print("1. Query understanding with location extraction")
#     print("2. Context inheritance across queries")
#     print("3. SQL generation from natural language")
#     print("4. Multi-location and time range handling")
#     print("5. Training data recording for future improvements")
#
#
# if __name__ == "__main__":
#     print("Starting Query Understanding Example...")
#     print("Make sure you have:")
#     print("1. PostgreSQL running with the session management schema")
#     print("2. A .env file with database and LLM credentials")
#     print("3. Required packages installed (asyncpg, etc.)\n")
#
#     asyncio.run(main())
