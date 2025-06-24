# Creating Golden Test Sets with RAGAS: Automated Evaluation Data Generation for RAG Systems

In [Part 1](langchain_eval_foundations_e2e.md), we built a robust foundation for RAG evaluation with multiple retrieval strategies, PostgreSQL + pgvector storage, and Phoenix observability. However, manually testing with ad-hoc queries like "Did people generally like John Wick?" doesn't scale. To systematically evaluate RAG systems, we need comprehensive test sets that cover diverse query types and expected outcomes. Enter **RAGAS** (Retrieval-Augmented Generation Assessment) and automated golden test set generation.

This post demonstrates how to automatically generate high-quality evaluation datasets using RAGAS, preparing us for quantitative comparison of retrieval strategies. Consider this your "Chapter 2" in the RAG evaluation journey.

---

## 1. The Challenge of RAG Test Set Creation

Creating evaluation data for RAG systems is uniquely challenging:

- **Domain Specificity**: Generic benchmarks rarely reflect your actual use case
- **Query Diversity**: Need questions spanning factual, analytical, and comparative types
- **Ground Truth**: Must establish reference answers and relevant contexts
- **Scale**: Manual creation of hundreds of test cases is impractical
- **Maintenance**: Test sets must evolve with your document corpus

Traditional approaches fail because:
- **Manual curation** is time-consuming and biased toward obvious queries
- **Synthetic generation** without LLMs produces unrealistic questions
- **Academic benchmarks** don't match production data distributions

RAGAS solves this by using LLMs to generate diverse, contextually relevant questions from your actual document corpus, complete with reference answers and source contexts.

---

## 2. RAGAS TestsetGenerator: Intelligent Question Synthesis

RAGAS employs multiple **synthesizers** to create different question types:

### Question Types Generated

1. **Factual Questions**: Direct information retrieval
   - Example: "What rating did John Smith give John Wick 3?"
   - Tests: Basic retrieval accuracy

2. **Multi-hop Questions**: Require connecting multiple pieces of information
   - Example: "Which John Wick movie received the most positive reviews from critics who also reviewed the first film?"
   - Tests: Complex reasoning and context aggregation

3. **Comparative Questions**: Analyze differences across documents
   - Example: "How did audience reception change between John Wick 2 and John Wick 3?"
   - Tests: Cross-document synthesis

4. **Analytical Questions**: Require interpretation and summary
   - Example: "What are the main criticisms of the John Wick franchise?"
   - Tests: Comprehension and summarization

### The Generation Process

```python
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Wrap LangChain models for RAGAS compatibility
generator_llm = LangchainLLMWrapper(llm)
generator_embeddings = LangchainEmbeddingsWrapper(embeddings)

# Initialize generator
generator = TestsetGenerator(
    llm=generator_llm,
    embedding_model=generator_embeddings
)

# Generate from your documents
golden_testset = generator.generate_with_langchain_docs(
    documents=docs,
    testset_size=10  # Number of questions to generate
)
```

RAGAS analyzes your documents to:
1. Identify key information and themes
2. Generate questions that test different retrieval capabilities
3. Create reference answers based on document content
4. Track which contexts support each answer

---

## 3. Integration with Phoenix for Dataset Management

Once generated, test sets need proper storage and versioning. Phoenix provides dataset management capabilities that integrate seamlessly with its tracing infrastructure.

### Transforming RAGAS Output for Phoenix

```python
def upload_to_phoenix(golden_testset, dataset_name: str = "johnwick_golden_testset") -> dict:
    testset_df = golden_testset.to_pandas()
    
    # Transform to Phoenix schema
    phoenix_df = pd.DataFrame({
        "input": testset_df["user_input"],          # The generated question
        "output": testset_df["reference"],          # Expected answer
        "contexts": testset_df["reference_contexts"].apply(
            lambda x: str(x) if isinstance(x, list) else str(x)
        ),
        "synthesizer": testset_df["synthesizer_name"],     # Question type
        "question_type": testset_df["synthesizer_name"],   # Duplicate for filtering
        "dataset_source": "ragas_golden_testset",
    })
    
    # Upload to Phoenix
    client = px.Client()
    dataset = client.upload_dataset(
        dataset_name=dataset_name,
        dataframe=phoenix_df,
        input_keys=["input"],
        output_keys=["output"],
        metadata_keys=["contexts", "synthesizer", "question_type", "dataset_source"]
    )
    
    return {
        "dataset_name": dataset_name,
        "num_samples": len(phoenix_df),
        "status": "success",
        "dataset": dataset,
    }
```

### Benefits of Phoenix Dataset Storage

1. **Version Control**: Track dataset evolution over time
2. **Experiment Association**: Link test sets to specific experiments
3. **Filtering**: Query subsets by question type or metadata
4. **Sharing**: Teams can access consistent evaluation data
5. **Visualization**: See question distribution and characteristics

---

## 4. Complete Implementation Walkthrough

Let's examine the full golden test set generation pipeline:

### Step 1: Environment Setup

```python
import os
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
from data_loader import load_docs_from_postgres
import phoenix as px
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY", "")
```

### Step 2: Initialize Models

```python
# Use GPT-4.1-mini for high-quality question generation
llm = ChatOpenAI(model="gpt-4.1-mini")  # Note: using approved model for evaluation
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Wrap for RAGAS compatibility
generator_llm = LangchainLLMWrapper(llm)
generator_embeddings = LangchainEmbeddingsWrapper(embeddings)
```

**Model Selection Considerations**:
- **LLM**: GPT-4 variants produce more nuanced questions than GPT-3.5
- **Embeddings**: Match your production embeddings model for consistency
- **Cost**: Balance quality needs with API costs

### Step 3: Load Documents

```python
# Reuse documents from your vector store
all_review_docs = load_docs_from_postgres("johnwick_baseline_documents")
```

This leverages the documents already ingested in Part 1, ensuring consistency between your retrieval corpus and test generation.

### Step 4: Generate Test Set

```python
def generate_testset(docs: list, llm, embeddings, testset_size: int = 10):
    generator = TestsetGenerator(llm=llm, embedding_model=embeddings)
    
    golden_testset = generator.generate_with_langchain_docs(
        documents=docs, 
        testset_size=testset_size
    )
    
    return golden_testset

# Generate a small test set (increase size for production)
golden_testset = generate_testset(
    all_review_docs, 
    generator_llm, 
    generator_embeddings, 
    testset_size=3  # Start small for testing
)
```

### Step 5: Upload to Phoenix

```python
dataset_result = upload_to_phoenix(
    golden_testset, 
    dataset_name="johnwick_golden_testset"
)

print(f"🚀 Workflow completed. Phoenix upload status: {dataset_result['status']}")
```

---

## 5. Understanding the Generated Data

Let's examine what RAGAS produces:

### Sample Generated Questions

```json
{
  "user_input": "What specific aspects of the action choreography do reviewers praise in John Wick 3?",
  "reference": "Reviewers consistently praise the innovative fight sequences, particularly the motorcycle chase scenes and the glass room battle. The hand-to-hand combat is described as 'balletic' and 'meticulously choreographed'.",
  "reference_contexts": [
    "Review from user445: The motorcycle sword fight is pure cinema gold...",
    "CriticReview99: The glass room sequence elevates action choreography to art..."
  ],
  "synthesizer_name": "analytical_synthesizer"
}
```

### Data Schema

- **user_input**: The generated question
- **reference**: Ground truth answer synthesized from documents
- **reference_contexts**: Source documents supporting the answer
- **synthesizer_name**: Type of question (factual, analytical, comparative, etc.)

---

## 6. Best Practices and Optimization

### Generation Parameters

```python
# For production test sets
production_config = {
    "testset_size": 100,      # Larger test sets for statistical significance
    "temperature": 0.7,       # Balance creativity and consistency
    "chunk_size": 1000,       # Process documents in batches
    "deduplication": True,    # Remove similar questions
}
```

### Quality Control

1. **Review Generated Questions**: Manually inspect a sample for:
   - Relevance to your use case
   - Answer accuracy
   - Question diversity

2. **Iterate on Document Selection**: 
   - Include diverse document types
   - Ensure temporal coverage
   - Balance positive/negative examples

3. **Version Your Test Sets**:
   ```python
   dataset_name = f"johnwick_golden_testset_{datetime.now().strftime('%Y%m%d')}"
   ```

### Cost Optimization

- **Start Small**: Generate 10-20 questions initially, then scale up
- **Reuse Test Sets**: Don't regenerate unless documents significantly change
- **Cache Embeddings**: Store document embeddings for faster regeneration

---

## 7. Integration with Evaluation Pipeline

The generated test set integrates seamlessly with the experiment framework (coming in Part 3):

```python
# In your experiment script
client = px.Client()
dataset = client.get_dataset(name="johnwick_golden_testset")

# Run experiments against the golden test set
for example in dataset.examples:
    question = example.input["input"]
    expected_answer = example.output["output"]
    reference_contexts = example.metadata["contexts"]
    
    # Evaluate each retrieval strategy
    # Compare against expected_answer
    # Measure context relevance
```

---

## 8. Troubleshooting Common Issues

### Issue: Generated Questions Too Similar

**Solution**: Increase document diversity or adjust synthesizer parameters
```python
generator = TestsetGenerator(
    llm=llm,
    embedding_model=embeddings,
    synthesizer_kwargs={"temperature": 0.8}  # Increase diversity
)
```

### Issue: Reference Answers Too Long

**Solution**: Post-process answers for conciseness
```python
def truncate_reference(reference, max_length=200):
    if len(reference) > max_length:
        return reference[:max_length] + "..."
    return reference
```

### Issue: Missing Context Documents

**Solution**: Ensure proper document metadata
```python
# Verify documents have sufficient metadata
for doc in docs:
    assert doc.page_content, "Empty document content"
    assert doc.metadata.get("source"), "Missing source metadata"
```

---

## Key Takeaways

1. **RAGAS automates test set creation** using LLMs to generate diverse, realistic questions from your document corpus
2. **Multiple question types** (factual, analytical, comparative) test different retrieval capabilities
3. **Phoenix integration** provides dataset versioning and experiment tracking
4. **Generated test sets include** questions, reference answers, and supporting contexts
5. **Quality control** through manual review and iterative refinement ensures test set relevance

With golden test sets in place, we're ready for **Part 3: Running Automated Experiments**—where we'll systematically evaluate each retrieval strategy using custom metrics and Phoenix's experiment framework.

---

**Next Steps**:
- Generate larger test sets (50-100 questions) for statistical significance
- Create domain-specific synthesizers for specialized question types
- Implement custom quality metrics for generated questions
- Set up automated test set refresh when documents update

Remember: The quality of your evaluation is only as good as your test data. Invest time in creating comprehensive, representative test sets that truly challenge your RAG system.