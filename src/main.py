# src/main.py
import asyncio
import os

from dotenv import load_dotenv

from src.agents.agent_manager import AgentManager
from src.api.app import create_app
from src.core.config_manager import ConfigManager
from src.core.logger import setup_logging, get_logger

# Load environment variables from .env file
load_dotenv()


# Setup logging
def setup_application():
    """Setup the application"""

    # Setup configuration
    config_manager = ConfigManager()

    # Setup logging
    setup_logging(config_manager.get('logging'))

    logger = get_logger(__name__)
    logger.info("Starting Intelligence Analytics Platform")

    # Initialize agent manager
    agent_manager = AgentManager(config_manager)

    logger.info(f"Initialized {len(agent_manager.list_agents())} agents")

    return config_manager, agent_manager


async def main():
    """Main application entry point"""

    config_manager, agent_manager = setup_application()
    logger = get_logger(__name__)

    try:
        # Example usage
        prompt = "people visiting Starbucks near Dubai Mall at 10-12am and then Dubai Airport at 4pm"

        logger.info(f"Processing prompt: {prompt}")

        # Process through agent pipeline
        results = await agent_manager.process_request(
            prompt=prompt,
            context={},
            classification="UNCLASS"
        )

        # Display results
        for agent_name, response in results.items():
            logger.info(f"Agent {agent_name}: {response.status.value}")
            if response.status.value == "completed":
                logger.info(f"Result summary: {response.result.get('summary', 'No summary')}")

        # Start web API (optional)
        if os.getenv('START_API', 'false').lower() == 'true':
            app = create_app(config_manager, agent_manager)
            import uvicorn
            # Use Server.serve directly to avoid nested asyncio.run
            config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
            server = uvicorn.Server(config)
            await server.serve()

    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
