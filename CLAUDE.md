# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a complete 3-stage RAG (Retrieval-Augmented Generation) evaluation pipeline that compares different retrieval strategies using John Wick movie reviews. The project implements a full toolkit including infrastructure setup, RAGAS golden test set generation, and automated evaluation with metrics.

## Key Commands

### Environment Setup
```bash
# Create virtual environment with Python 3.13
uv venv --python 3.13

# Activate virtual environment
source .venv/bin/activate  # On Unix/macOS
.venv\Scripts\activate     # On Windows

# Install dependencies
uv sync

# Verify critical dependencies
python -c "import langchain, openai, cohere, ragas; print('Dependencies verified')"
```

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

#### Validation & Analysis Scripts
```bash
# PostgreSQL data analysis with visualizations
python validation/postgres_data_analysis.py

# Phoenix telemetry and tracing validation  
python validation/validate_telemetry.py

# Interactive retrieval strategy comparison
python validation/retrieval_strategy_comparison.py
```

**Note:** Run the main pipeline first to populate data before using validation scripts.

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
   - CSV ingestion from John Wick movie reviews
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

## Development Notes

### Data Structure
- Reviews stored in `data/john_wick_[1-4].csv`
- Each review contains: Review_Title, Review_Text, Rating, Movie_Title
- Metadata includes: Review_Date, Author, Review_Url, last_accessed_at
- Golden test set in `data/johnwick_golden_testset_phoenix.json`

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
psql -h localhost -p 6024 -U langchain -d langchain -c "SELECT COUNT(*) FROM johnwick_baseline_documents;"

# Verify embeddings are populated
psql -h localhost -p 6024 -U langchain -d langchain -c "SELECT COUNT(*) FROM johnwick_baseline_documents WHERE embedding IS NOT NULL;"
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

### Pipeline Orchestration

The `run_rag_evaluation_pipeline.py` script provides:
- Complete pipeline orchestration with error handling
- Comprehensive logging to `logs/` directory with timestamps
- Environment validation and Docker service management
- Progress tracking and execution summaries

### Next Steps

With this complete pipeline, you can:
1. **Customize Evaluation**: Add domain-specific test questions and metrics
2. **Extend Retrievers**: Implement custom retrieval strategies for your use case
3. **Scale Up**: Adapt the pipeline for larger datasets and production use
4. **CI/CD Integration**: Add evaluation gates to your deployment pipeline
5. **Fine-tune Performance**: Optimize retrieval weights and parameters based on metrics