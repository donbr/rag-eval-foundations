# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a complete 3-stage RAG (Retrieval-Augmented Generation) evaluation pipeline that compares different retrieval strategies using **research PDF documents** as the primary data source. The project implements a full toolkit including infrastructure setup, RAGAS golden test set generation, and automated evaluation with metrics.

**Current Focus**: The system is configured to work with LLM interaction research documents (`llm-interaction-literature-review.pdf`, `howpeopleuseai.pdf`) and supports flexible document loading (`load_pdfs: true`, `load_csvs: false`). The project provides a general-purpose RAG evaluation platform while maintaining backwards compatibility with various document types.

**Validated Performance** (July 2025): Complete pipeline tested and operational with 269 PDF documents, 6 retrieval strategies, full Phoenix observability, and comprehensive validation scripts. All core functionality verified working.

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

### Environment Variables
Required in `.env` file:
- `OPENAI_API_KEY`
- `COHERE_API_KEY`
- `POSTGRES_CONNECTION_STRING` (optional, uses default if not set)
- `PHOENIX_CLIENT_HEADERS` (for tracing authentication)
- `PHOENIX_COLLECTOR_ENDPOINT` (defaults to http://localhost:6006)

Optional configuration:
- `GOLDEN_TESTSET_SIZE` (number of examples to generate in golden test set, defaults to 10)

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

### Data Structure
**Primary Data (PDF Documents)**:
- Research PDFs stored in `data/` directory:
  - `llm-interaction-literature-review.pdf` - Comprehensive review of human-LLM interaction research
  - `howpeopleuseai.pdf` - Study on AI usage patterns and behaviors
  - `llm-interaction-literature-review.md` - Markdown version for enhanced processing
- Document metadata: `document_name`, `source_type`, `last_accessed_at`
- Tables: `mixed_baseline_documents` and `mixed_semantic_documents`

**Secondary Data (CSV, Optional)**:
- Movie reviews stored in `data/john_wick_[1-4].csv`
- Each review contains: Review_Title, Review_Text, Rating, Movie_Title
- Metadata includes: Review_Date, Author, Review_Url, last_accessed_at
- Golden test set in `data/mixed_golden_testset_phoenix.json`

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

Key files and their purposes:
- `langchain_eval_foundations_e2e.py`: Main evaluation pipeline
- `langchain_eval_golden_testset.py`: RAGAS test set generation
- `langchain_eval_experiments.py`: Experimental evaluation approaches
- `retrieval_strategy_comparison.py`: Interactive comparison tool
- `data_loader.py`: Data ingestion utilities
- `notebooks/`: Analysis and validation tools

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

**IMPORTANT**: When creating or modifying scripts, Claude Code MUST test and validate the functionality:

1. **Script Creation**: Always run newly created scripts to verify they work correctly
2. **Integration Testing**: Test complete pipelines end-to-end when orchestration scripts are created
3. **Log Verification**: Confirm that logging features work as designed by executing the code
4. **Error Handling**: Test error conditions when possible to validate error handling works
5. **Documentation Accuracy**: Ensure that documented features actually work as described

**Refusing to run tests is unacceptable** - validation through execution is a core requirement for reliable code delivery.

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