# LangChain RAG Evaluation Learning Journey

This document explains the purpose of each `langchain_eval_` script and how they build upon each other to create a comprehensive RAG evaluation framework. These scripts form a progressive learning path from basic retrieval comparison to automated evaluation pipelines.

## Overview: The Three-Stage Journey

The three scripts represent an incremental learning journey:

1. **Foundation** → Build infrastructure and compare retrieval strategies manually
2. **Test Generation** → Create golden test sets for automated evaluation
3. **Experimentation** → Run automated experiments with metrics

## Script 1: `langchain_eval_foundations_e2e.py` - The Foundation

**Purpose**: Establishes the complete infrastructure for RAG evaluation and provides hands-on comparison of six different retrieval strategies.

### Key Learning Objectives
- Setting up PostgreSQL with pgvector for hybrid vector storage
- Implementing multiple retrieval strategies (naive vector, BM25, semantic chunking, etc.)
- Using Phoenix for LLM observability and tracing
- Creating standardized RAG chains for fair comparison

### What It Does
1. **Data Infrastructure**: Downloads John Wick movie reviews and loads them into PostgreSQL
2. **Vector Storage**: Creates two tables - baseline documents and semantic chunks
3. **Retrieval Strategies**: Implements and compares:
   - Naive vector similarity search
   - BM25 lexical matching
   - Contextual compression with Cohere reranking
   - Multi-query retrieval
   - Ensemble retrieval (combining multiple strategies)
   - Semantic chunking
4. **Observability**: Traces all operations with Phoenix for debugging and performance analysis
5. **Manual Evaluation**: Runs a sample question through all strategies for visual comparison

### Key Takeaway
This script teaches you how to build a robust foundation for RAG evaluation with multiple retrieval strategies that can be compared side-by-side. It emphasizes the importance of infrastructure and observability.

## Script 2: `langchain_eval_golden_testset.py` - Test Generation

**Purpose**: Generates a golden test set using RAGAS (Retrieval-Augmented Generation Assessment) to enable automated evaluation.

### Key Learning Objectives
- Understanding the concept of golden test sets in RAG evaluation
- Using RAGAS to automatically generate diverse test questions
- Uploading test sets to Phoenix for experiment tracking
- Preparing data for automated evaluation workflows

### What It Does
1. **Loads Documents**: Retrieves all John Wick reviews from PostgreSQL
2. **Generates Test Set**: Uses RAGAS TestsetGenerator with:
   - LLM wrapper (GPT-4) for question generation
   - Embedding wrapper for semantic understanding
   - Configurable test set size
3. **Creates Evaluation Data**: Produces questions with:
   - User input (the question)
   - Reference answer (expected response)
   - Reference contexts (relevant documents)
   - Synthesizer metadata (how the question was generated)
4. **Phoenix Integration**: Uploads the golden test set as a dataset for experiments

### Key Takeaway
This script bridges the gap between manual testing and automated evaluation by creating a reusable test set that represents diverse query patterns and expected outcomes.

## Script 3: `langchain_eval_experiments.py` - Automated Experiments

**Purpose**: Runs automated experiments using Phoenix's experiment framework to systematically evaluate each retrieval strategy against the golden test set.

### Key Learning Objectives
- Implementing custom evaluators for RAG systems
- Running experiments with Phoenix's experiment framework
- Measuring QA correctness and RAG relevance
- Comparing strategies quantitatively

### What It Does
1. **Loads Golden Test Set**: Retrieves the dataset created in Script 2
2. **Implements Evaluators**:
   - **QA Correctness**: Measures if the answer matches the reference
   - **RAG Relevance**: Evaluates if retrieved context is relevant to the query
3. **Runs Experiments**: For each retrieval strategy:
   - Executes all test questions
   - Captures responses and metadata
   - Applies evaluators to score performance
4. **Tracks Results**: Records experiment IDs and statuses in Phoenix

### Key Components
- **Task Functions**: Convert dataset examples into executable tasks
- **Enhanced Metadata**: Captures retrieved context for relevance evaluation
- **Batch Execution**: Runs all strategies systematically
- **Error Handling**: Gracefully handles failures in individual experiments

### Key Takeaway
This script completes the evaluation pipeline by automating the testing process and providing quantitative metrics for each retrieval strategy.

## The Learning Progression

### Stage 1: Manual Foundation (langchain_eval_foundations_e2e.py)
- **Focus**: Infrastructure and implementation
- **Output**: Visual comparison of retrieval strategies
- **Skills**: Vector databases, retrieval methods, observability

### Stage 2: Test Set Creation (langchain_eval_golden_testset.py)
- **Focus**: Evaluation data preparation
- **Output**: Reusable golden test set
- **Skills**: RAGAS, test generation, dataset management

### Stage 3: Automated Evaluation (langchain_eval_experiments.py)
- **Focus**: Systematic experimentation
- **Output**: Quantitative metrics and rankings
- **Skills**: Custom evaluators, experiment tracking, metrics analysis

## How the Scripts Connect

1. **Shared Infrastructure**: All scripts use the same PostgreSQL database and tables created in the foundation script
2. **Progressive Complexity**: Each script builds on the previous, adding layers of automation
3. **Unified Observability**: Phoenix provides consistent tracing and experiment tracking across all scripts
4. **Reusable Components**: Vector stores, retrievers, and chains from Script 1 are reused in Script 3

## Next Steps in Your Learning Journey

After mastering these three scripts, consider:

1. **Custom Evaluators**: Create domain-specific evaluation metrics
2. **Production Pipeline**: Integrate evaluation into CI/CD workflows
3. **Advanced Retrievers**: Implement hybrid search or learned sparse retrievers
4. **Cost Optimization**: Balance performance with API costs
5. **A/B Testing**: Use evaluation results to guide production deployments

## Running the Scripts in Order

```bash
# 1. Set up infrastructure and compare strategies
python src/langchain_eval_foundations_e2e.py

# 2. Generate golden test set
python src/langchain_eval_golden_testset.py

# 3. Run automated experiments
python src/langchain_eval_experiments.py
```

## Key Concepts to Master

- **Retrieval Strategies**: Understanding when to use semantic vs. lexical search
- **Evaluation Metrics**: QA correctness, relevance, faithfulness, answer quality
- **Observability**: Using traces to debug and optimize RAG pipelines
- **Experiment Design**: Creating representative test sets and fair comparisons
- **Production Readiness**: Moving from experiments to reliable systems

This learning journey prepares you to build, evaluate, and optimize RAG systems with confidence, using industry-standard tools and best practices.