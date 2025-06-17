# Intelligence Analytics Platform - Architectural Insights

## Overview
This is an AI-powered intelligence analytics platform that combines natural language processing with large-scale data analytics. The system uses a modular agent-based architecture to process various intelligence tasks including location extraction, profile analysis, and data correlation.

## Core Architecture Patterns

### 1. Agent-Based Design
- **Base Agent Pattern**: All agents inherit from `BaseAgent` providing standardized interfaces
- **Agent Manager**: Orchestrates multiple agents with priority-based execution
- **Parallel & Sequential Processing**: Agents can run in parallel (same priority) or sequentially (different priorities)
- **Context Passing**: Agents share results through context for workflow chaining

### 2. Configuration Management
- **Centralized Config**: `ConfigManager` loads all YAML configs with environment variable substitution
- **Type Safety**: Uses dataclasses for config validation (DatabaseConfig, LLMConfig)
- **Environment Variables**: Supports `${VAR:default}` pattern in YAML files
- **Hot Reload Ready**: Configuration can be refreshed without restart

### 3. Database Architecture
- **ClickHouse**: Primary analytics database for large-scale telecom/profile data
  - Table: `telecom_db.phone_imsi_uid_latest` - User profiles with risk scores
  - Optimized for analytical queries with column-oriented storage
- **Redis**: High-performance cache and geohash storage
  - Stores location geohashes for fast lookup
  - Used by location extractor for validation

### 4. LLM Integration
- **Factory Pattern**: `LLMClientFactory` creates appropriate client based on config
- **Multi-Provider Support**: OpenAI, Anthropic, Local models
- **Abstraction Layer**: `BaseLLM` provides unified interface
- **Context-Aware**: Enhanced prompts with database schema and metadata

## Key Agents

### 1. Profile Analyzer Agent
**Purpose**: Natural language to SQL for profile analysis

**Key Features**:
- Converts user queries to ClickHouse SQL using LLM
- Pre-loads all metadata at initialization for context
- Dual summarization strategy:
  - Pattern-based (fast, no LLM): `ResultSummarizer`
  - Intelligent (LLM-based): `IntelligentResultSummarizer`
- Parallel execution of SQL and analysis for performance
- SQL injection prevention and validation

**Metadata Loading**:
- Score ranges (risk, drug dealing, murder scores)
- Enum values (gender, age groups, residency status)
- Cities, applications, crime categories
- Nationality codes with names

### 2. Location Extractor Agent
**Purpose**: Extract locations from text and convert to geohashes

**Key Features**:
- Fast validation using heuristics (no LLM for initial check)
- Single LLM call for location extraction
- Redis integration for geohash validation
- Google Places API support (optional)
- Returns geohashes for downstream processing

**Processing Flow**:
1. Validate request with heuristics
2. Extract locations via LLM
3. Convert to geohashes using Redis lookup
4. Return consolidated geohash list

### 3. Data Correlator Agent
**Purpose**: Query ClickHouse using geohashes from location extraction

**Key Features**:
- Depends on location extractor output
- Executes multiple correlation queries:
  - Recent activity analysis
  - User movement patterns
  - Temporal patterns
  - Activity type distribution
- Analyzes results for insights
- Currently uses placeholder tables (needs real schema)

## API Layer

### FastAPI Structure
- **Main App**: `/src/api/main.py` - Initializes all components
- **Dependency Injection**: Provides agent manager to routes
- **CORS Enabled**: For cross-origin requests
- **Auto Documentation**: FastAPI generates OpenAPI docs

### Key Endpoints

#### Profile Analysis
- `POST /profile/query`: Natural language profile queries
- `GET /profile/examples`: Example queries by category
- Direct agent execution without full workflow

#### Location Extraction
- `POST /extract-locations`: Extract locations from text
- Returns locations and geohashes
- Single agent execution

#### Agent Management
- `GET /agents`: List all agents with status
- `POST /agents/{name}/load-geohashes`: Load geohashes (location extractor only)

## Performance Optimizations

### 1. Metadata Pre-loading
- All database metadata loaded at initialization
- Reduces query time by providing full context to LLM
- Fallback metadata if database unavailable

### 2. Parallel Processing
- Profile Analyzer: SQL execution and LLM analysis run in parallel
- Agent Manager: Same priority agents execute concurrently
- Async/await throughout for non-blocking operations

### 3. Caching Strategy
- Redis for geohash lookups (location extractor)
- Configurable TTL per agent
- Result caching for repeated queries

### 4. Smart Summarization
- Pattern-based summarizer for common query types (no LLM needed)
- Falls back to LLM summarization for complex results
- Identifies query patterns: aggregate, detail, comparison

## Security Considerations

### 1. SQL Injection Prevention
- Keyword blacklist (DROP, TRUNCATE, DELETE, etc.)
- Only SELECT queries allowed
- Single statement enforcement
- Query validation before execution

### 2. Input Validation
- Pydantic models for API request/response
- Type checking at boundaries
- Context validation for agent chaining

### 3. Classification Support
- Request classification levels (UNCLASS, CONFIDENTIAL, etc.)
- Agent-aware classification handling
- Metadata tracking for audit

## Deployment Architecture

### 1. Container Ready
- Dockerfile and docker-compose.yml present
- Environment-based configuration
- Health check endpoints

### 2. Scalability Features
- Connection pooling (ClickHouse, Redis)
- Async processing throughout
- Modular agent system for horizontal scaling
- Independent agent deployment possible

### 3. Monitoring & Logging
- Structured logging with context
- Per-agent loggers
- Configurable log levels via YAML
- Execution time tracking

## Data Schema Insights

### Primary Table: `phone_imsi_uid_latest`
**Key Columns**:
- Identity: imsi, phone_no, uid, eid
- Demographics: gender, age, nationality, marital status
- Location: home_city, home_location (geohash), work_location
- Risk Scores: risk_score, drug_dealing_score, murder_score
- Criminal: has_crime_case, crime_categories, is_in_prison
- Arrays: applications_used, traveled_countries, driving_licenses

**Score Ranges** (from actual data):
- risk_score: 0.00 to 0.27
- drug_dealing_score: 0.00 to 0.90
- drug_addict_score: 0.00 to 0.80
- murder_score: 0.00 to 0.00

## Testing Strategy

### 1. Agent Testing
- Unit tests for individual agents
- Mock LLM responses for deterministic testing
- Integration tests with real databases

### 2. API Testing
- FastAPI TestClient for endpoint testing
- Request/response validation
- Error handling verification

### 3. Load Testing
- Scripts for geohash loading from ClickHouse
- Bulk data processing tests
- Concurrent request handling

## Future Improvements

### 1. Agent Enhancements
- Complete implementation of intelligence_analyzer and reporting agents
- Add more sophisticated correlation algorithms
- Implement agent result caching

### 2. Performance
- Query result streaming for large datasets
- Distributed agent execution
- GPU acceleration for LLM inference

### 3. Features
- Real-time data streaming support
- Advanced visualization integration
- Multi-language support for queries

## Common Commands

### Run API Server
```bash
python run_api.py
# or
uvicorn src.api.main:app --reload
```

### Load Geohashes
```bash
python scripts/load_geohashes_from_clickhouse.py
```

### Run Tests
```bash
pytest tests/test_profile_analyzer.py
pytest tests/test_profile_api.py
```

### Health Check
```bash
curl http://localhost:8000/health
```

## Environment Variables

### Required
- `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`: LLM provider key
- `CLICKHOUSE_HOST`, `CLICKHOUSE_USER`, `CLICKHOUSE_PASSWORD`: Database connection
- `REDIS_HOST`: Redis connection

### Optional
- `GOOGLE_PLACES_API_KEY`: For location enrichment
- `DEFAULT_LLM_PROVIDER`: Choose LLM provider (default: openai)
- `LOCAL_LLM_URL`: For local model deployment

## Architecture Decisions

### 1. Why Agent-Based?
- Modularity: Each agent has single responsibility
- Scalability: Agents can be deployed independently
- Flexibility: Easy to add new analysis capabilities
- Testability: Isolated agent testing

### 2. Why ClickHouse?
- Optimized for analytical queries on large datasets
- Column-oriented storage perfect for profile analysis
- Fast aggregations and filtering
- Supports complex data types (arrays, nested)

### 3. Why Dual Summarization?
- Performance: Pattern-based is fast for common queries
- Quality: LLM-based provides nuanced insights
- Cost: Reduces LLM calls for simple summaries
- Reliability: Fallback mechanism ensures results

### 4. Why Redis for Geohashes?
- Sub-millisecond lookups
- Memory-efficient storage
- Set operations for deduplication
- Persistence options available

This architecture provides a robust foundation for intelligence analytics with clear separation of concerns, performance optimization, and extensibility for future enhancements.

## Logging Convention
- Import: `import logging` (standard Python logging)
- Usage: `logger = logging.getLogger(__name__)`
- The logging is configured centrally by `src/core/logger.py` via `setup_logging()`
- All modules should use `logging.getLogger(__name__)` pattern
- Do NOT import logger from core.logger, just use standard logging module

## Session Management Implementation

### PostgreSQL Schema
- Database: sigint
- User: tabreaz  
- Password: admin
- Schema location: `/sql/create_session_management.sql`

### Query Understanding System
- Uses unified LLM approach (single call for classification + extraction)
- Maintains context between queries in same session
- Supports include/exclude parameter structure for complex filtering
- Dynamic QueryCategory enum loaded from YAML config

## Next Implementation: SQL Generation & Validation

### Phase 1: SQL Generator (Current Focus)
1. **LLM-Based SQL Generation**
   - Use `context_aware_query` from controller output
   - Category-aware prompt construction
   - Schema context optimization (only relevant tables)
   - Handle location joins with query_location_geohashes
   - Support for include/exclude parameter mapping

2. **SQL Validator Enhancement**
   - Validate against ClickHouse version-specific syntax
   - Check query_location_geohashes usage with query_id
   - Verify time range handling with toDateTime()
   - Validate array operations for travelled_country_codes
   - Ensure proper index usage (event_date, imsi, geohash7)

3. **Multi-Step Validation & Correction**
   - Initial SQL generation from controller output
   - Validation with detailed error messages
   - Auto-correction loop with LLM feedback
   - Fallback strategies for complex queries

### Phase 2: Query Execution (Future)
- Connection pooling for ClickHouse
- Query timeout management
- Result streaming for large datasets
- Execution metrics tracking
- Error recovery mechanisms

### Phase 3: Result Processing (Future)
- Result formatting based on query category
- Aggregation handling
- Visualization data preparation
- Export capabilities

### ClickHouse Version Check
```bash
# Get ClickHouse version
clickhouse-client --query="SELECT version()"

# Or via Python
import clickhouse_connect
client = clickhouse_connect.get_client(host='localhost')
version = client.query("SELECT version()").result_rows[0][0]
```

### SQL Generator Implementation (Completed)

**Architecture:**
- `sql_generator.py` - Main SQL generation logic with LLM integration
- `sql_validator.py` - Comprehensive SQL validation with ClickHouse support
- `response_parser.py` - Extracts SQL from LLM responses
- `constants.py` - Configuration for large-scale data handling
- `validators.py` - Input validation
- `exceptions.py` - Custom exception types

**Key Features for Scale:**
1. **CTE-First Approach**: Breaks complex queries into CTEs for billion-scale data
2. **Memory Management**: 3GB limit with appropriate SETTINGS clause
3. **Early Filtering**: PREWHERE on partition keys, aggressive WHERE clauses
4. **Smart Table Selection**: Never joins geo_live with phone_imsi_uid_latest
5. **Location Optimization**: Uses geohash7 for efficient spatial joins

**Data Volume Awareness:**
- movements: 4B+ records/day
- geo_live: 1.5-2B records/day (15-min buckets)
- phone_imsi_uid_latest: 300M profiles
- Active IMSI: 30M/day

**Usage Example:**
```python
from src.agents.query_executor import SQLGenerator

sql_generator = SQLGenerator(llm_client)
result = await sql_generator.generate_sql(controller_output)
print(f"SQL: {result.sql}")
print(f"Method: {result.generation_method}")  # direct/corrected/failed_validation
```

**Critical Design Decisions:**
1. **CTEs Over Subqueries**: Better memory management and query planning
2. **Geohash7 Only**: Simpler joins, better index usage than mixing geohash6/7
3. **Time Bucketing**: Use toStartOfFifteenMinutes() for co-location queries
4. **Sampling Support**: Suggested for demographic analysis on full dataset
5. **Temp Tables**: Prefix with `temp_query_` for very complex multi-step queries

**Next Steps:**
- Query Executor for running generated SQL
- Result formatter based on query category
- Performance monitoring and optimization suggestions

## Frontend Implementation

### Technology Stack
- **React 18** with TypeScript for type safety
- **Vite** for fast development and optimized builds
- **Material-UI v5** for consistent design system
- **Chart.js** with react-chartjs-2 for visualizations
- **Socket.io-client** for real-time updates
- **Axios** for API communication
- **React Router** for navigation

### Architecture Principles
1. **Modular Configuration**: All configs split into focused modules under `/frontend/src/config/`
   - `api.config.ts` - API endpoints and settings
   - `query/` - Query types, patterns, categories
   - `response-types/` - Response type configurations (profile, location, etc.)
   - `visualizations/` - Chart types, KPIs, dashboard configs
   - `ui/` - UI behaviors, animations, layouts

2. **Component Structure**:
   - `/components` - Reusable UI components
   - `/pages` - Route-based page components
   - `/services` - API service layer
   - `/hooks` - Custom React hooks
   - `/contexts` - React contexts (theme, auth)
   - `/types` - TypeScript type definitions
   - `/utils` - Utility functions

3. **Key Design Patterns**:
   - **Service Layer**: Centralized API client with interceptors
   - **Theme System**: CSS-in-JS with Material-UI theming
   - **Type Safety**: Full TypeScript coverage
   - **Real-time Updates**: WebSocket integration for live progress
   - **Responsive Design**: Mobile-first approach

### Response Handling
The frontend supports multiple response types with appropriate visualizations:

1. **Profile Responses**:
   - KPI cards (total count, risk distribution, demographics)
   - Data tables with sorting/filtering
   - Various charts based on data type
   - Export functionality

2. **Location-based Responses**:
   - Map visualizations (future)
   - Location statistics
   - Movement patterns

3. **Analytical Responses**:
   - Complex dashboards
   - Multi-chart layouts
   - Statistical summaries

### Visualization System
Supports 11+ chart types based on backend statistics:
- Basic: bar, line, pie, donut, column
- Advanced: scatter, radar, treemap, bubble
- Special: gauge, word cloud
- Auto-suggestion based on data characteristics

### Chat Interface Features
- Natural language query input
- Real-time typing indicators
- Progress tracking through agent stages
- Response format options (summary/detailed/dashboard)
- Query history with session support
- Export capabilities

### Configuration-Driven UI
- All UI behaviors configurable
- Support for future features (auth, sessions)
- Flexible KPI selection and randomization
- Theme-aware color schemes
- Responsive breakpoints

### Security & Performance
- JWT token management ready
- Request debouncing and throttling
- Lazy loading for large datasets
- Optimistic UI updates
- Error boundaries for graceful failures

### Future Frontend Enhancements
1. **Authentication Integration**: 
   - PostgreSQL session management
   - User preferences storage
   - Role-based access control

2. **Advanced Visualizations**:
   - Geospatial maps for location data
   - Network graphs for relationships
   - Real-time streaming dashboards

3. **Enhanced UX**:
   - Keyboard shortcuts
   - Voice input support
   - Collaborative features
   - Advanced filtering UI
EOF < /dev/null