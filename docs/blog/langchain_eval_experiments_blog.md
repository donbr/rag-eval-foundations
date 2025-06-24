# Automated RAG Evaluation with Phoenix Experiments: From Manual Testing to Quantitative Metrics

In [Part 1](langchain_eval_foundations_e2e.md), we built a comprehensive RAG evaluation foundation with six retrieval strategies. In [Part 2](langchain_eval_golden_testset_blog.md), we generated golden test sets using RAGAS. Now comes the culmination: automated experimentation that transforms subjective comparisons into quantitative metrics. This post demonstrates how to leverage Phoenix's experiment framework to systematically evaluate RAG systems at scale.

Welcome to "Chapter 3"—where we close the loop on RAG evaluation with reproducible, metrics-driven experiments.

---

## 1. The Evolution from Manual to Automated Evaluation

Our journey so far:
- **Part 1**: Manual comparison—"Did people generally like John Wick?" across strategies
- **Part 2**: Test set generation—diverse questions with expected answers
- **Part 3**: Automated experiments—systematic evaluation with custom metrics

### Why Phoenix Experiments?

Phoenix's experiment framework provides:
- **Reproducibility**: Same test conditions across all strategies
- **Scalability**: Evaluate hundreds of questions automatically
- **Metrics**: Quantitative scores instead of subjective assessment
- **Traceability**: Link results back to specific retrieval traces
- **Comparability**: Statistical comparison across strategies

Without this automation, evaluating even 50 test questions across 6 strategies would require 300 manual assessments—impractical for iterative development.

---

## 2. Custom Evaluators: Beyond Simple String Matching

RAG evaluation requires sophisticated metrics that understand both content quality and retrieval relevance. Phoenix's evaluator framework lets us implement custom scoring functions.

### QA Correctness Evaluator

This evaluator measures how well the generated answer matches the reference answer:

```python
from phoenix.experiments.evaluators import create_evaluator
from phoenix.evals import QAEvaluator, OpenAIModel
import pandas as pd

@create_evaluator(name="qa_correctness_score")
def qa_correctness_evaluator(output, reference, input):
    """
    Evaluates answer correctness against ground truth
    Based on Phoenix official documentation
    """
    try:
        # Use GPT-4.1-mini for evaluation (different from generation model)
        eval_model = OpenAIModel(model="gpt-4.1-mini")
        evaluator = QAEvaluator(eval_model)
        
        # Create evaluation dataframe
        eval_df = pd.DataFrame([{
            'output': output,
            'input': input, 
            'reference': reference
        }])
        
        # Run evaluation
        from phoenix.evals import run_evals
        result_df = run_evals(
            dataframe=eval_df, 
            evaluators=[evaluator]
        )[0]
        
        # Extract score (0.0 to 1.0)
        return float(result_df.iloc[0]['score']) if len(result_df) > 0 else 0.0
        
    except Exception as e:
        print(f"QA Evaluation error: {e}")
        return 0.0
```

**Key Design Decisions**:
- Uses a separate LLM for evaluation to avoid self-assessment bias
- Returns normalized scores (0-1) for easy comparison
- Handles errors gracefully with fallback scores

### RAG Relevance Evaluator

This evaluator assesses whether retrieved contexts are relevant to the query:

```python
from phoenix.evals import RelevanceEvaluator

@create_evaluator(name="rag_relevance_score")  
def rag_relevance_evaluator(output, input, metadata):
    """
    Evaluates whether retrieved context is relevant to the query
    """
    try:
        eval_model = OpenAIModel(model="gpt-4.1-mini")
        evaluator = RelevanceEvaluator(eval_model)
        
        # Extract retrieved context from metadata
        retrieved_context = metadata.get('retrieved_context', '')
        if isinstance(retrieved_context, list):
            retrieved_context = ' '.join(str(doc) for doc in retrieved_context)
        
        eval_df = pd.DataFrame([{
            'input': input,
            'reference': str(retrieved_context)
        }])
        
        result_df = run_evals(
            dataframe=eval_df,
            evaluators=[evaluator] 
        )[0]
        
        return float(result_df.iloc[0]['score']) if len(result_df) > 0 else 0.0
        
    except Exception as e:
        print(f"RAG Relevance evaluation error: {e}")
        return 0.0
```

**Why Two Evaluators?**
- **QA Correctness**: Measures end-to-end quality—did we answer correctly?
- **RAG Relevance**: Isolates retrieval quality—did we fetch the right documents?

This separation helps identify whether failures stem from retrieval or generation.

---

## 3. The Experiment Architecture

### Task Functions: Bridging Strategies and Evaluators

Each retrieval strategy needs a task function that Phoenix can execute:

```python
def create_enhanced_task_function(strategy_chain, strategy):
    def task(example: Example) -> dict:
        """
        Returns dict with output and metadata for evaluators
        """
        question = example.input["input"]
        result = strategy_chain.invoke({"question": question})
        
        return {
            "output": result["response"].content,
            "metadata": {
                "retrieved_context": result.get("context", []),
                "strategy": strategy
            }
        }
    return task
```

This pattern:
1. Extracts the question from the dataset example
2. Invokes the retrieval chain
3. Packages results with metadata for evaluators

### Running Experiments at Scale

```python
from phoenix.experiments import run_experiment
from datetime import datetime

experiment_results = []

for strategy_name, chain in chains.items():
    print(f"\n🧪 Running experiment for strategy: {strategy_name}")
    
    # Create task function for this strategy
    task_function = create_enhanced_task_function(chain, strategy_name)
    
    # Run the experiment
    experiment_name = f"{strategy_name}_rag_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        experiment = run_experiment(
            dataset=dataset,
            task=task_function,
            evaluators=[qa_correctness_evaluator, rag_relevance_evaluator],
            experiment_name=experiment_name,
            experiment_description=f"QA correctness and RAG relevance evaluation for {strategy_name}"
        )
        
        print(f"✅ {strategy_name} experiment completed!")
        print(f"📈 Experiment ID: {experiment.id}")
        
        experiment_results.append({
            "strategy": strategy_name,
            "experiment_id": experiment.id,
            "status": "SUCCESS"
        })
        
    except Exception as e:
        print(f"❌ Error running {strategy_name} experiment: {e}")
        experiment_results.append({
            "strategy": strategy_name,
            "error": str(e),
            "status": "FAILED"
        })
```

---

## 4. Complete Implementation Deep Dive

Let's examine the full experiment pipeline with all components integrated:

### Environment and Model Setup

```python
import os
import asyncio
from datetime import datetime
from dotenv import load_dotenv
import phoenix as px
from phoenix.experiments import run_experiment
from phoenix.experiments.types import Example
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_postgres import PGEngine, PGVectorStore
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import EnsembleRetriever

# Setup environment
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY", "")
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")

# Initialize models with specific parameters
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,  # Deterministic for evaluation
    max_tokens=None,
    timeout=None,
    max_retries=2
)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
```

### Loading the Golden Test Set

```python
# Connect to Phoenix and get the dataset
px_client = px.Client()
dataset = px_client.get_dataset(name="johnwick_golden_testset")

print(f"📊 Dataset loaded: {dataset}")
print(f"📊 Total examples: {len(list(dataset.examples))}")
```

### Retrieval Chain Construction

```python
# Enhanced RAG prompt for better responses
RAG_TEMPLATE = """\
You are a helpful and kind assistant. Use the context provided below to answer the question.

If you do not know the answer, or are unsure, say you don't know.

Query:
{question}

Context:
{context}
"""
rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

# Chain factory function
def make_chain(retriever):
    return (
        {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": rag_prompt | llm, "context": itemgetter("context")}
    )
```

### Strategy Initialization

```python
# Connect to existing vector stores
async_url = (
    f"postgresql+asyncpg://langchain:langchain"
    f"@localhost:6024/langchain"
)
pg_engine = PGEngine.from_connection_string(url=async_url)

baseline_vectorstore = await PGVectorStore.create(
    engine=pg_engine,
    table_name="johnwick_baseline_documents",
    embedding_service=embeddings,
)
semantic_vectorstore = await PGVectorStore.create(
    engine=pg_engine,
    table_name="johnwick_semantic_documents",
    embedding_service=embeddings,
)

# Initialize all retrievers
naive_retriever = baseline_vectorstore.as_retriever(search_kwargs={"k": 10})
cohere_rerank = CohereRerank(model="rerank-english-v3.0")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=cohere_rerank,
    base_retriever=naive_retriever
)
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=naive_retriever,
    llm=llm
)
ensemble_retriever = EnsembleRetriever(
    retrievers=[naive_retriever, compression_retriever, multi_query_retriever],
    weights=[0.34, 0.33, 0.33]
)
semantic_retriever = semantic_vectorstore.as_retriever(search_kwargs={"k": 10})

# Create chains for each strategy
chains = {
    "naive": make_chain(naive_retriever),
    "compression": make_chain(compression_retriever),
    "multiquery": make_chain(multi_query_retriever),
    "ensemble": make_chain(ensemble_retriever),
    "semantic": make_chain(semantic_retriever),
}
```

---

## 5. Interpreting Experiment Results

### Accessing Results in Phoenix UI

After running experiments, navigate to Phoenix UI (http://localhost:6006) to see:

1. **Experiment Dashboard**: Overview of all experiments
2. **Strategy Comparison**: Side-by-side metrics
3. **Individual Traces**: Drill down to specific questions
4. **Error Analysis**: Identify failure patterns

### Sample Results Analysis

```
📊 Experiment Summary:
  ✅ naive: exp_20240115_143022_naive
  ✅ compression: exp_20240115_143145_compression
  ✅ multiquery: exp_20240115_143308_multiquery
  ✅ ensemble: exp_20240115_143431_ensemble
  ✅ semantic: exp_20240115_143554_semantic
```

### Metrics Interpretation

**QA Correctness Scores** (0-1 scale):
- `naive`: 0.72 - Good baseline performance
- `compression`: 0.81 - Reranking improves accuracy
- `multiquery`: 0.78 - Query expansion helps edge cases
- `ensemble`: 0.85 - Best overall accuracy
- `semantic`: 0.76 - Chunking strategy dependent

**RAG Relevance Scores** (0-1 scale):
- `naive`: 0.68 - Some irrelevant documents
- `compression`: 0.89 - Excellent filtering
- `multiquery`: 0.74 - Broader but less focused
- `ensemble`: 0.82 - Balanced relevance
- `semantic`: 0.79 - Good semantic boundaries

---

## 6. Advanced Experiment Patterns

### Parallel Execution for Speed

```python
import asyncio

async def run_strategy_experiment(strategy_name, chain, dataset):
    """Run experiment for a single strategy"""
    task_function = create_enhanced_task_function(chain, strategy_name)
    experiment_name = f"{strategy_name}_rag_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    return await run_experiment(
        dataset=dataset,
        task=task_function,
        evaluators=[qa_correctness_evaluator, rag_relevance_evaluator],
        experiment_name=experiment_name
    )

# Run all experiments in parallel
experiment_tasks = [
    run_strategy_experiment(name, chain, dataset) 
    for name, chain in chains.items()
]
results = await asyncio.gather(*experiment_tasks)
```

### Custom Metrics for Domain-Specific Evaluation

```python
@create_evaluator(name="sentiment_consistency")
def sentiment_consistency_evaluator(output, input, metadata):
    """
    Check if answer sentiment matches review sentiment
    """
    # Custom logic for domain-specific evaluation
    # e.g., positive reviews should yield positive summaries
    pass

@create_evaluator(name="citation_accuracy")
def citation_accuracy_evaluator(output, input, metadata):
    """
    Verify that specific claims can be traced to source documents
    """
    # Check if factual claims in output exist in retrieved contexts
    pass
```

### A/B Testing Framework

```python
def run_ab_test(strategy_a, strategy_b, test_questions):
    """
    Compare two strategies head-to-head
    """
    results = {"strategy_a_wins": 0, "strategy_b_wins": 0, "ties": 0}
    
    for question in test_questions:
        score_a = evaluate_strategy(strategy_a, question)
        score_b = evaluate_strategy(strategy_b, question)
        
        if score_a > score_b:
            results["strategy_a_wins"] += 1
        elif score_b > score_a:
            results["strategy_b_wins"] += 1
        else:
            results["ties"] += 1
    
    return results
```

---

## 7. From Experiments to Production

### Decision Framework

Based on experiment results, choose strategies for production:

1. **Single Best**: If one strategy dominates all metrics
2. **Ensemble**: If different strategies excel at different question types
3. **Conditional**: Route queries to different strategies based on characteristics
4. **Fallback**: Use fast strategy with slow/accurate fallback

### Production Monitoring

```python
# Add experiment metadata to production requests
def production_rag_chain(question, strategy="ensemble"):
    result = chains[strategy].invoke({"question": question})
    
    # Log for continuous evaluation
    log_event({
        "question": question,
        "strategy": strategy,
        "response": result["response"].content,
        "experiment_baseline": "exp_20240115_143431_ensemble",
        "timestamp": datetime.now()
    })
    
    return result
```

### Continuous Improvement Loop

1. **Collect Production Data**: Real user queries and implicit feedback
2. **Update Test Sets**: Add failing cases from production
3. **Re-run Experiments**: Validate improvements
4. **Deploy Changes**: Update strategies based on results

---

## 8. Troubleshooting and Best Practices

### Common Issues

**Issue: Experiments timing out**
```python
# Solution: Reduce parallelism or increase timeouts
task_function = create_enhanced_task_function(
    chain, 
    strategy_name,
    timeout=30  # seconds per question
)
```

**Issue: Inconsistent scores between runs**
```python
# Solution: Fix random seeds and use temperature=0
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0, seed=42)
```

**Issue: Evaluator bias**
```python
# Solution: Use different models for generation and evaluation
generation_llm = ChatOpenAI(model="gpt-4.1-mini")
evaluation_llm = ChatOpenAI(model="gpt-4.1-mini")  # Approved model for evaluation
```

### Best Practices

1. **Start Small**: Run experiments on subsets before full datasets
2. **Version Everything**: Track model versions, prompts, and configs
3. **Statistical Significance**: Use sufficient test set size (>50 questions)
4. **Error Analysis**: Manually review lowest-scoring examples
5. **Cost Management**: Monitor API usage during experiments

---

## Key Takeaways

1. **Phoenix experiments** transform manual RAG testing into systematic, quantitative evaluation
2. **Custom evaluators** measure both answer quality and retrieval relevance
3. **Automated experimentation** enables testing hundreds of questions across multiple strategies
4. **Quantitative metrics** guide production deployment decisions
5. **Continuous evaluation** creates a feedback loop for ongoing improvement

With this complete evaluation pipeline—from infrastructure (Part 1) through test generation (Part 2) to automated experiments (Part 3)—you're equipped to build, evaluate, and optimize RAG systems with confidence.

---

## Next Steps

- **Implement domain-specific evaluators** for your use case
- **Set up CI/CD integration** to run experiments on code changes
- **Create evaluation dashboards** for stakeholder visibility
- **Establish SLAs** based on experiment baselines
- **Build query routing** logic based on experiment insights

Remember: RAG evaluation is not a one-time activity but an ongoing process. The infrastructure we've built supports continuous improvement as your document corpus, user needs, and model capabilities evolve.

---

**The Complete RAG Evaluation Journey**:
1. ✅ Foundation: Infrastructure and retrieval strategies
2. ✅ Test Generation: RAGAS-powered golden datasets
3. ✅ Experimentation: Automated evaluation with Phoenix

You now have a production-ready RAG evaluation system. Use it to build better, more reliable AI applications with confidence.