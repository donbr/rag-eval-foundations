# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A production-ready 3-stage RAG evaluation pipeline implementing 6 retrieval strategies with RAGAS-based golden testset generation and Phoenix observability. The project includes Phase 4 golden testset management with PostgreSQL-backed versioning, cost tracking, and quality validation.

**Key Features**:
- **6 Retrieval Strategies**: Naive vector, semantic chunking, BM25, contextual compression, multi-query, ensemble
- **Phase 4 Golden Testset System**: Database-backed versioning with semantic versioning (major.minor.patch)
- **Cost Tracking**: Phoenix-integrated token usage and API cost monitoring
- **Quality Validation**: RAGAS metrics with configurable thresholds
- **Full Observability**: Phoenix tracing for all LLM operations

**Validated Performance** (October 2025): Complete pipeline tested with 269 PDF documents, full Phoenix integration, database-backed testset management, and comprehensive validation tooling.

## Key Commands

### Environment Setup

**IMPORTANT**: This project requires Python 3.13+ and uses `uv` for package management.

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment with Python 3.13
uv venv --python 3.13

# Activate virtual environment
source .venv/bin/activate  # On Unix/macOS
.venv\Scripts\activate     # On Windows

# Install dependencies from pyproject.toml
uv sync

# Verify critical dependencies
python -c "import langchain, openai, cohere, ragas, pypdf, matplotlib, seaborn, psycopg2, asyncpg; print('All dependencies verified')"
```

**Key Dependencies** (automatically installed via `uv sync`):
- LangChain ecosystem (core, openai, cohere, postgres, experimental)
- Phoenix observability with OpenTelemetry instrumentation
- PDF processing (pypdf) and visualization (matplotlib, seaborn)
- Vector database support (asyncpg, psycopg2-binary)
- Machine learning utilities (rank_bm25, rapidfuzz)

### Running the Application

The application provides a complete 3-stage evaluation pipeline from infrastructure through automated metrics.

#### Complete Pipeline (Recommended)
```bash
# Run the complete pipeline with orchestration script
python claude_code_scripts/run_rag_evaluation_pipeline.py

# With verbose logging
python claude_code_scripts/run_rag_evaluation_pipeline.py --verbose

# Skip Docker service management (if already running)
python claude_code_scripts/run_rag_evaluation_pipeline.py --skip-services

# Customize golden test set size (default: 10)
python claude_code_scripts/run_rag_evaluation_pipeline.py --testset-size 5
```

#### Individual Scripts (Manual Execution)
```bash
# Main evaluation script
python src/langchain_eval_foundations_e2e.py

# Generate golden test set with RAGAS
python src/langchain_eval_golden_testset.py

# Run experiments
python src/langchain_eval_experiments.py

# Interactive strategy comparison
python validation/retrieval_strategy_comparison.py
```

#### Validation & Testing Commands
```bash
# Essential validation sequence (run after main pipeline)
python validation/postgres_data_analysis.py        # Database analysis
python validation/retrieval_strategy_comparison.py # Strategy benchmarking  
python validation/validate_telemetry.py           # Phoenix tracing validation

# Dependency verification
python -c "import langchain, openai, cohere, ragas, pypdf, matplotlib, seaborn, psycopg2, asyncpg; print('All dependencies verified')"
```

#### Database Operations
```bash
# Connect to PostgreSQL
psql -h localhost -p 6024 -U langchain -d langchain

# Or use PGPASSWORD for scripting
PGPASSWORD=langchain psql -h localhost -p 6024 -U langchain -d langchain

# Query golden testsets
PGPASSWORD=langchain psql -h localhost -p 6024 -U langchain -d langchain -c "
  SELECT name, version_major, version_minor, version_patch, created_at
  FROM golden_testsets
  ORDER BY created_at DESC
  LIMIT 10;"

# Get latest version of a testset
PGPASSWORD=langchain psql -h localhost -p 6024 -U langchain -d langchain -c "
  SELECT get_latest_testset_version('financial_aid_baseline');"

# Check testset overview (uses view)
PGPASSWORD=langchain psql -h localhost -p 6024 -U langchain -d langchain -c "
  SELECT * FROM testset_overview;"

# Count documents in vector stores
PGPASSWORD=langchain psql -h localhost -p 6024 -U langchain -d langchain -c "
  SELECT COUNT(*) FROM mixed_baseline_documents;"
```

#### Log Management
```bash
# View recent pipeline logs
ls -la logs/

# View latest pipeline execution
tail -f logs/rag_evaluation_$(date +%Y%m%d)*.log

# Clean old logs (keep last 10)
ls -t logs/*.log | tail -n +11 | xargs rm -f
```

**Note:** 
- Run the main pipeline first to populate data before using validation scripts
- Validation scripts are updated to work with the current PDF-based data (financial aid documents)
- Scripts generate visualizations in `outputs/charts/` with data distributions, embedding analysis, and performance comparisons

### Required Services

#### Service Management
```bash
# Check service status and port conflicts BEFORE starting
./claude_code_scripts/check-services.sh

# Validate service ports are available
netstat -tulpn | grep -E ":(6024|6006|4317)\s" || echo "Ports available"

# Start all services with docker-compose (recommended)
docker-compose up -d

# Verify services are running
docker ps --filter 'label=project=rag-eval-foundations'

# Test database connection
psql -h localhost -p 6024 -U langchain -d langchain -c "\dt"

# Check Phoenix is responding
curl -s http://localhost:6006/health || echo "Phoenix not ready"

# Stop services
docker-compose down

# Remove services and data
docker-compose down -v
```

#### Port Configuration
The project uses specific ports to avoid conflicts:
- PostgreSQL: 6024 (default PostgreSQL uses 5432)
- Phoenix UI: 6006 (same as TensorBoard)
- Phoenix OTLP: 4317 (standard OpenTelemetry gRPC)

If you have port conflicts, set environment variables:
```bash
export POSTGRES_PORT=6025
export PHOENIX_UI_PORT=6007
export PHOENIX_OTLP_PORT=4318
docker-compose up -d
```

Or create a `.env` file (see `.env.example`).

#### Container Naming
All containers use the `rag-eval-` prefix:
- `rag-eval-pgvector` (PostgreSQL)
- `rag-eval-phoenix` (Phoenix observability)

This prevents conflicts with other projects' containers.

#### Manual Docker Management (Alternative)
```bash
# PostgreSQL with pgvector
docker run -it --rm --name pgvector-container \
  -e POSTGRES_USER=langchain \
  -e POSTGRES_PASSWORD=langchain \
  -e POSTGRES_DB=langchain \
  -p 6024:5432 \
  pgvector/pgvector:pg16

# Phoenix with timestamped project names
export TS=$(date +"%Y%m%d_%H%M%S")
docker run -it --rm --name phoenix-container \
  -e PHOENIX_PROJECT_NAME="retrieval-comparison-${TS}" \
  -p 6006:6006 \
  -p 4317:4317 \
  arizephoenix/phoenix:latest
```

## Configuration System

**Shared Configuration (October 2025):**
All scripts now use a centralized configuration system via `src/config.py`:

```python
from config import (
    PHOENIX_ENDPOINT,          # http://localhost:6006
    GOLDEN_TESTSET_NAME,       # mixed_golden_testset_phoenix
    LLM_MODEL,                 # gpt-4.1-mini (enforced)
    EMBEDDING_MODEL,           # text-embedding-3-small (enforced)
    COHERE_RERANK_MODEL,       # rerank-english-v3.0
    get_postgres_async_url,    # Helper for DB connection
    BASELINE_TABLE,            # mixed_baseline_documents
    SEMANTIC_TABLE,            # mixed_semantic_documents
)
```

**Benefits:**
- ✅ Single source of truth for all configuration
- ✅ Consistent dataset naming across upload/experiments/golden_testset scripts
- ✅ Model enforcement per CLAUDE.md requirements
- ✅ Environment variable support with sensible defaults
- ✅ Type-safe dataclasses with validation

See `src/config.py` for complete configuration options.

---

## Architecture

### Core Components

1. **Data Pipeline**:
   - **Primary**: PDF document ingestion from research sources (LLM interaction literature, AI usage studies)
   - **Secondary**: CSV ingestion from various datasets (optional, disabled by default)
   - Async document processing with metadata enrichment
   - PostgreSQL/pgvector storage with 1536-dimension embeddings
   
2. **Retrieval Architecture**:
   - **Factory Pattern**: `create_retrievers()` provides unified interface
   - **Async-First Design**: All operations use `asyncio` and `PGEngine` connection pooling
   - **Six Strategy Implementation**:
     - Naive Vector Search (OpenAI embeddings)
     - Semantic Chunking (SemanticChunker with percentile breakpoints)
     - BM25 (keyword-based term frequency)
     - Contextual Compression (Cohere reranking)
     - Multi-Query (LLM-generated query variations)
     - Ensemble (Reciprocal Rank Fusion with equal weights)

3. **Complete 3-Stage Evaluation Pipeline**: 
   - **Stage 1**: Infrastructure setup and manual strategy comparison
   - **Stage 2**: RAGAS golden test set generation with LLM wrappers
   - **Stage 3**: Automated evaluation with Phoenix experiment framework
   - **Observability-First**: Auto-instrumentation with structured tracing

### Retrieval Strategy Details

#### 1. Naive Vector Search
- Uses OpenAI's text-embedding-3-small model
- Stores embeddings in pgvector with dimension 1536
- Retrieves top-k documents by cosine similarity
- Good for semantic similarity but misses keyword matches

#### 2. Semantic Chunking Vector Search
- Splits documents using SemanticChunker with percentile breakpoints
- Creates more meaningful chunks based on semantic boundaries
- Better context preservation than fixed-size chunking

#### 3. BM25 Retriever
- Traditional information retrieval algorithm
- Ranks documents based on term frequency and inverse document frequency
- Excellent for exact keyword matches
- Complements semantic search in ensemble approaches

#### 4. Contextual Compression
- Uses Cohere's rerank-english-v3.0 model
- Filters and reranks retrieved documents based on query relevance
- Reduces noise and improves precision
- Helps manage token limits for LLM context

#### 5. Multi-Query Retriever
- Uses LLM to generate 3-5 query variations
- Retrieves documents for each query variant
- Returns unique union of all results
- Overcomes query formulation limitations

#### 6. Ensemble Retriever
- Combines multiple retrieval strategies
- Uses Reciprocal Rank Fusion for reranking
- Currently uses equal weights (25% each)
- Balances strengths of different approaches

### Key Dependencies & Architecture Patterns

#### Core Framework
- **LangChain**: Retriever abstraction and chain composition
- **OpenAI API**: Embeddings (text-embedding-3-small) and LLM (gpt-4.1-mini)
- **Cohere API**: Reranking with rerank-english-v3.0 model
- **PostgreSQL + pgvector**: Vector similarity search with SQL capabilities
- **Phoenix (Arize)**: OpenTelemetry-based LLM observability
- **RAGAS**: RAG evaluation framework with golden test set generation
- **PyPDF**: PDF document loading and processing
- **Matplotlib/Seaborn**: Data visualization for analysis scripts

#### Design Patterns
- **Configuration Dataclass**: Centralized `Config` class with environment overrides
- **Async Connection Management**: `PGEngine` handles connection pooling
- **Factory Pattern**: Consistent retriever creation and management
- **Chain Composition**: Standardized RAG chains for fair comparison
- **Error Handling**: Graceful degradation with structured logging

### Configuration Management

**Centralized Configuration** (`src/config.py`):
All configuration is centralized with environment variable overrides:

```python
from src.config import (
    # Phoenix settings
    PHOENIX_ENDPOINT,           # http://localhost:6006
    PHOENIX_OTLP_ENDPOINT,      # http://localhost:4317

    # Database settings
    get_postgres_async_url(),   # PostgreSQL async connection
    get_postgres_sync_url(),    # PostgreSQL sync connection
    BASELINE_TABLE,             # mixed_baseline_documents
    SEMANTIC_TABLE,             # mixed_semantic_documents

    # Model settings (ENFORCED)
    LLM_MODEL,                  # gpt-4.1-mini (only permitted model)
    EMBEDDING_MODEL,            # text-embedding-3-small (only permitted model)

    # Dataset settings
    GOLDEN_TESTSET_NAME,        # mixed_golden_testset_phoenix
    GOLDEN_TESTSET_SIZE,        # Default: 10

    # Dataclasses for structured config
    PhoenixSettings,
    DatabaseSettings,
    ModelSettings,
)
```

**Environment Variables** (`.env` file):
Required:
- `OPENAI_API_KEY`: OpenAI API key for embeddings and LLM
- `COHERE_API_KEY`: Cohere API key for reranking

Optional (with defaults):
- `PHOENIX_ENDPOINT`: Phoenix UI endpoint (default: http://localhost:6006)
- `PHOENIX_OTLP_ENDPOINT`: OTLP collector (default: http://localhost:4317)
- `POSTGRES_HOST`: Database host (default: localhost)
- `POSTGRES_PORT`: Database port (default: 6024)
- `GOLDEN_TESTSET_SIZE`: Testset size (default: 10)

See `.env.example` for complete list and configuration options.

### Development Tools & Code Quality (2025 Best Practices)

The project uses modern Python development practices with **Ruff** for linting and formatting, and includes development dependencies via UV's dependency groups.

#### ✅ **Validated Development Dependencies**
```bash
# Add development tools (tested and working)
uv add --dev ruff mypy pytest pre-commit

# Dependencies are stored in pyproject.toml under [dependency-groups]
# Current dev dependencies: ruff>=0.12.1, mypy>=1.16.1, pytest>=8.4.1
```

#### ✅ **Tested Code Quality Commands**
```bash
# Lint and check code (tested working)
ruff check src/ validation/ claude_code_scripts/

# Format code (tested working)
ruff format src/ validation/ claude_code_scripts/

# Fix auto-fixable issues (tested working)
ruff check --fix src/ validation/ claude_code_scripts/

# Type checking (tested working)
mypy --version  # Confirms mypy 1.16.1 available

# Check for issues without fixing
ruff check src/ --no-fix
```

#### 🔧 **Optional pyproject.toml Configuration**
Add tool configurations as needed:
```toml
[tool.ruff]
line-length = 88
target-version = "py313"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]

[tool.mypy]
python_version = "3.13"
warn_return_any = true
warn_unused_configs = true
```

**Note**: All commands above have been tested and verified working in this environment. Ruff successfully replaces black, flake8, and isort as a unified tool.

### Quick Validation Commands

```bash
# Verify the complete system works (tested July 2025)
python claude_code_scripts/run_rag_evaluation_pipeline.py --skip-services --testset-size 3

# Run individual validation scripts
python validation/postgres_data_analysis.py          # Database and embeddings analysis
python validation/retrieval_strategy_comparison.py  # Strategy benchmarking
python validation/validate_telemetry.py             # Phoenix tracing validation

# Check data integrity
PGPASSWORD=langchain psql -h localhost -p 6024 -U langchain -d langchain -c "SELECT COUNT(*) FROM mixed_baseline_documents;"
```

These commands provide quick verification that all components are working correctly.

## Development Notes

### Database Schema Architecture

**Vector Store Tables** (PGVector):
- `mixed_baseline_documents`: Standard chunking with OpenAI embeddings (1536-dim)
- `mixed_semantic_documents`: SemanticChunker-based chunking with embeddings
- Both use pgvector's HNSW indexing for similarity search

**Phase 4 Golden Testset Tables** (src/golden_testset/):
- `golden_testsets`: Testset metadata with semantic versioning (major.minor.patch)
- `golden_examples`: Individual examples linked to testsets
- Uses UUIDs, timestamps, and versioning for full traceability

**Database Utilities** (scripts/db/):
- `assert_schema.py`: Validate required tables, views, indexes exist
- `assert_indexes.py`: Verify performance-critical indexes
- `dry_run_sql.py`: Test SQL migrations safely before execution

**Data Sources**:
- **Primary**: PDF documents in `data/` (research papers, literature reviews)
- **Secondary**: CSV datasets (optional, disabled by default)
- Configuration in `src/config.py` controls which sources are loaded

### Phoenix Observability

Phoenix provides comprehensive tracing for LLM applications:
- **Traces**: Complete request lifecycle from query to response
- **Spans**: Individual operations (retrieval, LLM calls, reranking)
- **Metrics**: Latency, token usage, retrieval performance
- **Debugging**: Identify bottlenecks and failures

View traces at http://localhost:6006 after starting the Phoenix container.

### RAGAS Evaluation Concepts

RAGAS (Retrieval-Augmented Generation Assessment) provides metrics for:
- **Context Precision**: Signal-to-noise ratio of retrieved documents
- **Context Recall**: Coverage of relevant information
- **Faithfulness**: Factual accuracy of generated answers
- **Answer Relevancy**: How well the answer addresses the question

The framework supports both reference-free evaluation (using LLMs) and golden dataset evaluation.

### Common Issues and Solutions

#### Vector Search Returns No Results
```bash
# Test database connection
psql -h localhost -p 6024 -U langchain -d langchain -c "\dt"

# Check table contents
psql -h localhost -p 6024 -U langchain -d langchain -c "SELECT COUNT(*) FROM mixed_baseline_documents;"

# Verify embeddings are populated
psql -h localhost -p 6024 -U langchain -d langchain -c "SELECT COUNT(*) FROM mixed_baseline_documents WHERE embedding IS NOT NULL;"
```

#### API Rate Limits
- Monitor token usage in Phoenix traces
- Implement exponential backoff in API calls
- Consider caching embeddings for repeated runs
- Reduce batch sizes for embedding generation

#### Phoenix Connection Issues
```bash
# Verify Phoenix is running
docker ps --filter 'name=phoenix'

# Test Phoenix health endpoint
curl -s http://localhost:6006/health

# Check Phoenix traces
curl -s http://localhost:6006/v1/traces | jq '.data | length'

# Validate OTLP endpoint
telnet localhost 4317
```

#### Service Port Conflicts
```bash
# Check what's using our ports
lsof -i :6024  # PostgreSQL
lsof -i :6006  # Phoenix UI
lsof -i :4317  # Phoenix OTLP

# Use alternative ports via environment
export POSTGRES_PORT=6025
export PHOENIX_UI_PORT=6007
export PHOENIX_OTLP_PORT=4318
```


#### Async Event Loop Errors
- Common in Jupyter notebooks - use `asyncio.run()` for standalone scripts
- Ensure proper async context management with `PGEngine`
- Check for hanging connections with `docker logs rag-eval-pgvector`

### Code Organization

**Source Structure** (`src/`):
```
src/
├── config.py                           # Centralized configuration
├── data_loader.py                      # Data ingestion utilities
├── langchain_eval_foundations_e2e.py   # Stage 1: Main evaluation pipeline
├── langchain_eval_golden_testset.py    # Stage 2: RAGAS testset generation
├── langchain_eval_experiments.py       # Stage 3: Automated evaluation
├── upload_golden_testset_to_phoenix.py # Phoenix dataset upload
└── golden_testset/                     # Phase 4: Testset management
    ├── manager.py                      # CRUD operations
    ├── versioning.py                   # Semantic versioning
    ├── quality_validator.py            # RAGAS quality gates
    ├── cost_tracker.py                 # Cost tracking
    ├── phoenix_integration.py          # Phoenix upload integration
    ├── change_detector.py              # Change detection
    ├── transactions.py                 # Database transactions
    ├── validation_pipeline.py          # Validation orchestration
    └── tracing.py                      # OpenTelemetry tracing
```

**Scripts** (`scripts/`):
```
scripts/
├── db/                                 # Database utilities
│   ├── assert_schema.py               # Schema validation
│   ├── assert_indexes.py              # Index verification
│   ├── dry_run_sql.py                 # Safe SQL testing
│   ├── db_connection.py               # Connection pooling
│   └── db_migrate.py                  # Migration runner
└── validation/                         # Pre-push validation
    ├── check_implementation_status.py  # Task completion checks
    ├── validate_branch_phase.py        # Branch-phase alignment
    └── detect_scope_creep.py           # Scope validation
```

**Flows** (`flows/`):
```
flows/
└── golden_testset_flow.py              # Main Prefect 3.x hybrid flow

**Archived Flows** (see `docs/archived_flows/`):
- golden_testset_flow_alternate.py (clean reference)
- golden_testset_flow_prefect3.py (Prefect 3.x reference)
```

**Validation** (`validation/`):
```
validation/
├── postgres_data_analysis.py           # Database analysis & visualization
├── retrieval_strategy_comparison.py    # Interactive strategy benchmarks
└── validate_telemetry.py               # Phoenix tracing validation
```

### Phase 4: Golden Testset Management System

**Architecture**: Database-backed versioning with Phoenix cost integration (implemented October 2025)

#### **Core Components** (src/golden_testset/):

**Manager** (`manager.py`):
- `GoldenTestsetManager`: CRUD operations with async context manager
- Semantic versioning (major.minor.patch) with automatic version bumping
- Transaction support via `transactions.py` for atomic operations
- Example:
  ```python
  async with GoldenTestsetManager() as mgr:
      testset = await mgr.create_testset(name="baseline", examples=[...])
      v2 = await mgr.update_testset(testset.id, examples=[...], change_type="minor")
  ```

**Versioning** (`versioning.py`):
- `SemanticVersion`: Version comparison, bumping (major/minor/patch)
- `VersionManager`: Query latest versions, version history, diffs
- Database views: `latest_testsets`, `testset_overview`

**Cost Tracking** (`cost_tracker.py`, `phoenix_integration.py`):
- `PhoenixCostTracker`: Extract token usage and costs from Phoenix traces
- `PhoenixIntegration`: Upload testsets to Phoenix with versioned dataset names
- Integration with OpenAI and Cohere API pricing

**Quality Validation** (`quality_validator.py`):
- `QualityValidator`: Check RAGAS metrics against thresholds
- Configurable gates for context_precision, faithfulness, answer_relevancy
- Validation pipeline in `validation_pipeline.py`

**Change Detection** (`change_detector.py`):
- Detect semantic changes in testset updates
- Recommend appropriate version bumps (major/minor/patch)
- Support intelligent versioning decisions

#### **Prefect 3.x Flows** (flows/):

**Main Flow** (`golden_testset_flow.py`):
- Hybrid architecture: Clean core + optional enterprise features
- Maps to `.claude/tasks.yaml` for phase-based execution
- Modes: `--only-phase` (dev), `--production` (enterprise)
- Optional features: `--enable-quality-gates`, `--enable-cost-tracking`, `--enable-git`

**Usage**:
```bash
# Development: Run single phase
python flows/golden_testset_flow.py --only-phase phase1

# Production: All features
python flows/golden_testset_flow.py --production --enable-quality-gates --enable-cost-tracking

# Custom: Select features
python flows/golden_testset_flow.py --enable-monitoring --enable-git
```

#### **Database Schema Validation**:
```bash
# Validate all schema requirements
python scripts/db/assert_schema.py --check all

# Check specific tables
python scripts/db/assert_schema.py --require tables=golden_testsets,golden_examples

# Verify indexes exist
python scripts/db/assert_indexes.py

# Test SQL before running
python scripts/db/dry_run_sql.py --sql "SELECT get_latest_testset_version('baseline');"
```

#### **Key Design Decisions**:
- **PostgreSQL over JSON files**: ACID transactions, concurrent access, query capabilities
- **Semantic versioning**: Clear version progression with major/minor/patch
- **Phoenix integration**: Unified observability for costs and quality
- **Async-first**: All DB operations use asyncpg for performance
- **Prefect 3.x flows**: Modern orchestration with explicit future resolution

### Extending the Framework

#### Adding New Retrieval Strategies
1. **Implement in `create_retrievers()`**:
   ```python
   def create_retrievers(baseline_vectorstore, semantic_vectorstore, all_docs, llm):
       # Add your new retriever
       custom_retriever = YourCustomRetriever(...)
       
       return {
           "existing_strategies": ...,
           "your_strategy": custom_retriever
       }
   ```

2. **Update ensemble weights** if including in ensemble strategy
3. **Add Phoenix tracing tags** for observability
4. **Document strategy behavior** and use cases
5. **Test with validation scripts** in `validation/` directory

#### Adding Custom Evaluators
1. Implement evaluator following Phoenix experiment framework
2. Add to `langchain_eval_experiments.py`
3. Update golden test set if needed for your metrics

### Performance Optimization

#### Database Operations
- **Async Batch Ingestion**: Use `aadd_documents()` for large document sets
- **Connection Pooling**: `PGEngine` manages async connection lifecycle
- **Index Optimization**: pgvector uses HNSW indexing for similarity search
- **Chunk Size Tuning**: Balance context preservation vs. embedding quality

#### API Optimization
- **Embedding Caching**: Store embeddings to avoid regeneration
- **Concurrent Operations**: Leverage async patterns for parallel processing
- **Rate Limit Management**: Monitor Phoenix traces for API usage patterns
- **Batch Size Tuning**: Optimize OpenAI embedding batch sizes

#### Monitoring & Debugging
```bash
# Monitor database performance
psql -h localhost -p 6024 -U langchain -d langchain -c "SELECT * FROM pg_stat_activity;"

# Check Phoenix trace volume
curl -s http://localhost:6006/v1/traces | jq '.data | length'

# Monitor container resources
docker stats rag-eval-pgvector rag-eval-phoenix
```

### Model Requirements

**CRITICAL**: Only the following models are permitted for this project:

- **LLM Model**: `gpt-4.1-mini` (OpenAI)
- **Embedding Model**: `text-embedding-3-small` (OpenAI)

**Use of any other models is strictly prohibited.** All scripts must use these exact model names.

### Testing and Validation Requirements

**Test Structure** (tests/):
```
tests/
├── unit/                                   # Unit tests with pytest
│   ├── test_golden_testset_manager.py     # Manager CRUD operations
│   ├── test_quality_validator.py          # Quality validation logic
│   └── ...
├── integration/                            # Integration tests (future)
└── fixtures/                               # Test data and mocks
```

**Running Tests**:
```bash
# Run all unit tests
python -m pytest tests/unit/ -v

# Run specific test file
python -m pytest tests/unit/test_golden_testset_manager.py -v

# Run with coverage
python -m pytest tests/unit/ --cov=src/golden_testset --cov-report=html

# Run validation scripts (requires running services)
python validation/postgres_data_analysis.py
python validation/retrieval_strategy_comparison.py
python validation/validate_telemetry.py
```

**Validation Scripts** (scripts/validation/):
```bash
# Check implementation status against .claude/tasks.yaml
python scripts/validation/check_implementation_status.py

# Validate branch-phase alignment
python scripts/validation/validate_branch_phase.py

# Detect scope creep
python scripts/validation/detect_scope_creep.py
```

**Database Validation**:
```bash
# Validate schema
python scripts/db/assert_schema.py --check all

# Verify indexes
python scripts/db/assert_indexes.py

# Test SQL safely
python scripts/db/dry_run_sql.py --sql "SELECT * FROM testset_overview LIMIT 5;"
```

**Testing Philosophy**:
1. **Script Creation**: Always run newly created scripts to verify they work
2. **Integration Testing**: Test complete pipelines end-to-end
3. **Log Verification**: Confirm logging works by executing code
4. **Error Handling**: Test error conditions where possible
5. **Documentation Accuracy**: Ensure documented features work as described

**Validation through execution is required** - not optional.

### Current Data Configuration

**IMPORTANT**: The system is currently configured to load PDF documents by default:

```python
# Configuration in langchain_eval_foundations_e2e.py
load_pdfs: bool = True   # Research PDFs (enabled)
load_csvs: bool = False  # John Wick CSVs (disabled)
golden_testset_size: int = 10  # Number of examples in RAGAS golden test set
```

**To switch data sources**:
- For PDF-only processing (current): `load_pdfs=True, load_csvs=False`
- For CSV-only processing: `load_pdfs=False, load_csvs=True`  
- For mixed processing: `load_pdfs=True, load_csvs=True`

**To configure golden test set size**:
- Via command line: `--testset-size 5` (orchestration script)
- Via environment: `GOLDEN_TESTSET_SIZE=5` (in .env file)
- Via config: `golden_testset_size: int = 5` (modify Config class)

**Test Queries by Data Type**:
- **LLM Research PDFs**: "What factors influence user trust in AI systems?", "How do people adapt their interaction patterns with LLMs?", "What are the key challenges in human-AI collaboration?"
- **General CSV Datasets**: Flexible queries based on loaded dataset schema and content

### Cost Considerations

**Budget Planning**: Be aware of API costs when running scripts:
- **OpenAI**: ~$0.50-$2.00 per full pipeline run (depends on data size)
- **Cohere**: ~$0.10-$0.50 for reranking operations
- **Total**: Budget approximately $5 for experimentation and testing

Monitor token usage through Phoenix traces to optimize costs during development.

### Pipeline Orchestration

The `run_rag_evaluation_pipeline.py` script provides:
- Complete pipeline orchestration with error handling
- Comprehensive logging to `logs/` directory with timestamps
- Environment validation and Docker service management
- Progress tracking and execution summaries

### Feature Branch Push Strategy

**Push Timing**: Feature branches should be pushed to GitHub **after completing each phase** implementation and validation.

#### Pre-Push Validation Checklist

Before pushing feature branches, complete this validation sequence:

```bash
# 1. Validate phase completion against .claude/tasks.yaml
python scripts/validation/check_implementation_status.py

# 2. Check branch-phase alignment
python scripts/validation/validate_branch_phase.py

# 3. Run comprehensive tests
python -m pytest tests/unit/ -v
python validation/postgres_data_analysis.py
python validation/retrieval_strategy_comparison.py

# 4. Verify no scope creep
python scripts/validation/detect_scope_creep.py

# 5. Ensure all dependencies are satisfied
uv sync && python -c "import all_required_modules"
```

#### Push Commands

```bash
# Standard feature branch push (after validation)
git add .
git commit -m "Complete Phase X implementation with [key features]"
git push origin feature/phaseX-[description]

# Create pull request
gh pr create --title "Phase X: [Implementation Summary]" --body "## Implementation Summary
- [Key feature 1]
- [Key feature 2]
- [Validation results]

## Testing Evidence
- All unit tests passing
- Phase validation complete
- No scope creep detected"
```

#### Pull Request Guidelines

**Required Information:**
1. **Implementation Summary**: Key features and changes
2. **Testing Evidence**: Validation script outputs, test results
3. **Phase Alignment**: Confirmation of proper branch-phase mapping
4. **Breaking Changes**: Any API or interface changes
5. **Dependencies**: New packages or configuration requirements

**Review Criteria:**
- [ ] Phase requirements from `.claude/tasks.yaml` are met
- [ ] All tests passing and validation scripts successful
- [ ] No scope creep into other phases
- [ ] Proper error handling and logging
- [ ] Documentation updated for new features

### External Documentation

For additional context and deep-dive analysis:
- **[DeepWiki Documentation](https://deepwiki.com/donbr/rag-eval-foundations)**: Interactive Q&A, architecture diagrams, and performance analysis
- **[Technical Blog Post](docs/blog/langchain_eval_foundations_e2e_blog.md)**: Complete implementation walkthrough with code examples
- **[Learning Journey](docs/technical/langchain_eval_learning_journey.md)**: Detailed 3-stage progression guide
- **[Validation Scripts](validation/README.md)**: Interactive tools for data exploration and strategy comparison

### Next Steps

With this complete pipeline, you can:
1. **Customize Evaluation**: Add domain-specific test questions and metrics
2. **Extend Retrievers**: Implement custom retrieval strategies for your use case
3. **Scale Up**: Adapt the pipeline for larger datasets and production use
4. **CI/CD Integration**: Add evaluation gates to your deployment pipeline
5. **Fine-tune Performance**: Optimize retrieval weights and parameters based on metrics