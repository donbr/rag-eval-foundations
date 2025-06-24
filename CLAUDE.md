# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RAG (Retrieval-Augmented Generation) evaluation framework that compares different retrieval strategies using John Wick movie reviews. The project serves as a foundation for learning LangChain's retriever ecosystem before implementing automated evaluation with RAGAS golden test sets.

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
```

### Running the Application

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

# Start all services with docker-compose (recommended)
docker-compose up -d

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

## Architecture

### Core Components

1. **Data Loading** (`data_loader.py`): Loads John Wick movie reviews from CSV files into PostgreSQL/pgvector
2. **Retrieval Strategies**: Six different approaches implemented:
   - **Naive Vector Search**: Basic similarity search using OpenAI embeddings
   - **Semantic Chunking**: Uses SemanticChunker to create meaningful document segments
   - **BM25**: Keyword-based retrieval using term frequency
   - **Contextual Compression**: Filters chunks using Cohere reranking
   - **Multi-Query**: Generates query variations for broader coverage
   - **Ensemble**: Combines all strategies with equal weights (25% each)

3. **Evaluation Pipeline**: 
   - Uses RAGAS to generate golden test questions
   - Compares retrieval strategies against these questions
   - Traces all operations with Phoenix for debugging

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

### Key Dependencies
- **LangChain**: Core framework with community extensions
- **OpenAI API**: For embeddings (text-embedding-3-small) and chat (gpt-4.1-mini)
- **Cohere API**: For reranking in contextual compression
- **PostgreSQL + pgvector**: Vector database for similarity search
- **Phoenix (Arize)**: LLM observability platform using OpenTelemetry

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
- Check PostgreSQL connection: `psql -h localhost -p 6024 -U langchain -d langchain`
- Verify tables exist: `\dt` in psql
- Ensure data was loaded successfully (check logs)

#### API Rate Limits
- Implement exponential backoff in API calls
- Reduce batch sizes for embedding generation
- Consider caching embeddings for repeated runs

#### Phoenix Connection Issues
- Ensure container is running: `docker ps`
- Check endpoint configuration matches container ports
- Verify no firewall blocking localhost connections

#### Async Event Loop Errors
- Common in Jupyter notebooks
- Use `asyncio.run()` for standalone scripts
- Consider using synchronous alternatives for debugging

### Code Organization

Key files and their purposes:
- `langchain_eval_foundations_e2e.py`: Main evaluation pipeline
- `langchain_eval_golden_testset.py`: RAGAS test set generation
- `langchain_eval_experiments.py`: Experimental evaluation approaches
- `retrieval_strategy_comparison.py`: Interactive comparison tool
- `data_loader.py`: Data ingestion utilities
- `notebooks/`: Analysis and validation tools

### Extending the Framework

To add a new retrieval strategy:
1. Implement retriever in `create_retrievers()` function
2. Add to the retrievers dictionary with descriptive name
3. Update ensemble weights if including in ensemble
4. Document strategy in this file

### Performance Optimization

- **Batch Operations**: Use `aadd_documents()` for async batch ingestion
- **Connection Pooling**: PGEngine handles connection management
- **Embedding Cache**: Consider caching embeddings for large datasets
- **Chunk Size**: Balance between context and embedding quality
- **Concurrent Retrieval**: Async operations enable parallel processing

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

This foundation prepares for:
1. **Golden Test Sets**: Generate domain-specific evaluation questions
2. **Automated Metrics**: Implement RAGAS scoring pipeline
3. **A/B Testing**: Compare retrieval strategies systematically
4. **Production Pipeline**: CI/CD integration with evaluation gates
5. **Custom Retrievers**: Domain-specific retrieval strategies