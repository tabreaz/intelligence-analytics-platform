# src/api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

from src.core.config_manager import ConfigManager
from src.agents.agent_manager import AgentManager
from src.api.routes import agents, profile_analytics, profile_statistics, websocket
from src.api.dependencies import set_agent_manager, set_config_manager, set_llm_client
from src.core.logger import setup_logging, get_logger

# Initialize configuration
config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config')
config_manager = ConfigManager(config_dir=config_dir)

# Setup logging from config
logging_config = config_manager._configs.get('logging', {})
# Temporarily set database logger to DEBUG
if 'loggers' not in logging_config:
    logging_config['loggers'] = {}
logging_config['loggers']['src.core.database'] = {
    'level': 'DEBUG',
    'handlers': ['console'],
    'propagate': False
}
setup_logging(logging_config)
logger = get_logger(__name__)

# Initialize agent manager and config manager
logger.info("Initializing agent manager...")
agent_manager = AgentManager(config_manager)
set_agent_manager(agent_manager)
set_config_manager(config_manager)

# Initialize LLM client
from src.core.llm.base_llm import LLMClientFactory
try:
    # Get default LLM provider config
    llm_config = config_manager.get_llm_config()
    llm_client = LLMClientFactory.create_client(llm_config)
    set_llm_client(llm_client)
    logger.info(f"Initialized LLM client with provider: {llm_config.model}")
except Exception as e:
    logger.warning(f"Failed to initialize LLM client: {e}")
    # Continue without LLM client - will use fallback methods

logger.info(f"Initialized agents: {agent_manager.list_agents()}")

# Create FastAPI app
app = FastAPI(
    title="Intelligence Analytics Platform",
    description="AI-powered intelligence analytics with location extraction and profile analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(agents.router)
app.include_router(profile_analytics.router)
app.include_router(profile_statistics.router)
app.include_router(websocket.router)

# Connect WebSocket manager to ActivityLogger
from src.api.routes.websocket import manager as ws_manager
from src.core.activity_logger import set_websocket_manager
set_websocket_manager(ws_manager)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Intelligence Analytics Platform API",
        "version": "1.0.0",
        "endpoints": [
            "/docs",
            "/health",
            "/agents",
            "/profile-analytics/query",
            "/profile-analytics/demographics",
            "/profile-analytics/risk-statistics",
            "/profile-statistics/analyze",
            "/profile-statistics/dashboard-stats",
            "/profile-statistics/available-statistics"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agents": agent_manager.list_agents(),
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)