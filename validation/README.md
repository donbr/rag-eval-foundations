# Validation Scripts

This directory contains interactive scripts for exploring, testing, and validating the RAG evaluation system components.

## Prerequisites

1. **Services Running**: Ensure Docker services are started
   ```bash
   docker-compose up -d
   ```

2. **Data Populated**: Run the main pipeline first to create vector stores and data
   ```bash
   python claude_code_scripts/run_rag_evaluation_pipeline.py
   ```

## Available Scripts

### 1. `postgres_data_analysis.py`
**Purpose:** Comprehensive PostgreSQL vector database analysis

**What it does:**
- Analyzes document distribution across John Wick movies (25 reviews each)
- Compares baseline chunking vs semantic chunking strategies
- Generates PCA visualization of embeddings in 2D space
- Shows content length statistics and rating distributions

**Outputs:**
- `outputs/charts/postgres_analysis/rating_distribution.png`
- `outputs/charts/postgres_analysis/chunking_comparison.png` 
- `outputs/charts/postgres_analysis/embedding_visualization.png`

**Run time:** ~30-60 seconds

### 2. `validate_telemetry.py`
**Purpose:** Phoenix OpenTelemetry tracing validation and demonstration

**What it does:**
- Tests simple and complex LLM chain patterns
- Demonstrates embedding generation with tracing
- Shows streaming responses with real-time trace updates
- Validates error handling and metadata capture
- Tests mock RAG pipeline tracing

**Outputs:**
- Real-time traces viewable at http://localhost:6006
- Console output showing various chain executions

**Run time:** ~2-3 minutes

### 3. `retrieval_strategy_comparison.py`
**Purpose:** Interactive comparison of all 6 retrieval strategies

**What it does:**
- Compares naive, semantic, BM25, compression, multiquery, and ensemble retrievers
- Tests with different query types (sentiment, specific aspects, complex questions)
- Runs performance benchmarks measuring speed and document retrieval
- Generates side-by-side strategy comparisons

**Outputs:**
- `outputs/charts/retrieval_analysis/retrieval_performance.png`
- Detailed console output showing retrieved documents per strategy
- Performance statistics and timing comparisons

**Run time:** ~3-5 minutes

## Usage Examples

### Quick Validation Run
```bash
# Run all validation scripts in sequence
python validation/postgres_data_analysis.py
python validation/validate_telemetry.py  
python validation/retrieval_strategy_comparison.py
```

### Individual Script Usage
```bash
# Just analyze the database
python validation/postgres_data_analysis.py

# Just test tracing
python validation/validate_telemetry.py

# Just compare retrieval strategies
python validation/retrieval_strategy_comparison.py
```

## What Each Script Validates

### Database Integrity (`postgres_data_analysis.py`)
- ✅ Vector store contains expected 100 documents
- ✅ Documents are evenly distributed across movies
- ✅ Embeddings are properly generated (1536 dimensions)
- ✅ Semantic chunking creates appropriate chunk sizes
- ✅ Metadata is correctly preserved

### Tracing & Observability (`validate_telemetry.py`)
- ✅ Phoenix tracing captures LLM operations
- ✅ Token usage and latency are tracked
- ✅ Complex chain operations are properly traced
- ✅ Error conditions are captured in traces
- ✅ Real-time streaming works with tracing

### Retrieval Performance (`retrieval_strategy_comparison.py`)
- ✅ All 6 retrieval strategies function correctly
- ✅ Different strategies excel at different query types
- ✅ Performance characteristics are measurable
- ✅ BM25 is fastest, ensemble is most comprehensive
- ✅ Contextual compression improves precision

## Troubleshooting

### Common Issues

**"Connection refused" errors:**
- Ensure Docker services are running: `docker-compose ps`
- Check PostgreSQL is healthy: `docker logs rag-eval-pgvector`

**"No such table" errors:**
- Run the main pipeline first to create vector stores
- Check that E2E pipeline completed successfully

**"No API key" errors:**
- Verify `.env` file has `OPENAI_API_KEY` and `COHERE_API_KEY`
- Check environment variables are loaded

**Phoenix traces not visible:**
- Ensure Phoenix container is running on port 6006
- Visit http://localhost:6006 to view traces
- Check container logs: `docker logs rag-eval-phoenix`

### Script-Specific Issues

**postgres_data_analysis.py fails:**
- Requires both baseline and semantic vector stores to exist
- Run the main E2E pipeline to create these tables

**validate_telemetry.py shows no traces:**
- Phoenix project name changes each run
- Look for latest project in Phoenix UI dropdown

**retrieval_strategy_comparison.py hangs:**
- Cohere API key may be missing or invalid
- BM25 requires documents to be loaded first

## Output Files

All validation scripts generate organized output files:

```
outputs/
└── charts/
    ├── postgres_analysis/
    │   ├── rating_distribution.png
    │   ├── chunking_comparison.png
    │   └── embedding_visualization.png
    └── retrieval_analysis/
        └── retrieval_performance.png
```

These files are excluded from version control via `.gitignore`.

## Integration with Main Pipeline

The validation scripts are designed to be run **after** the main pipeline:

1. **First:** `python claude_code_scripts/run_rag_evaluation_pipeline.py`
2. **Then:** Run any validation scripts to explore the results

This ensures all data structures, vector stores, and Phoenix experiments exist before validation begins.