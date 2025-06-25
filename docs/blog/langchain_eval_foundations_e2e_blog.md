# LangChain RAG Evaluation Foundations: Building a Robust Testing Infrastructure

Evaluating Retrieval-Augmented Generation (RAG) systems presents unique challenges beyond standard ML evaluation. As large language models (LLMs) become more powerful, ensuring reliability, minimizing hallucinations, and maintaining consistency across every component (retrieval, ranking, generation) is crucial. This post lays the groundwork for a production-ready RAG evaluation pipeline by covering:

1. **LangChain Document Loaders**  
2. **Multiple Retriever Methods**  
3. **PostgreSQL + pgvector**  
4. **Arize Phoenix Observability**  

This post focuses on Stage 1 of our complete 3-stage evaluation pipeline. The full implementation includes RAGAS golden test set generation (Stage 2) and automated evaluation experiments (Stage 3), all available in the [project repository](https://github.com/donbr/rag-eval-foundations). For a comprehensive overview of the entire journey, see the [technical documentation](../technical/langchain_eval_learning_journey.md).

---

## 1. The Complexity of RAG Evaluation

RAG pipelines differ from standard NLP tasks in several ways:

- **Non‐Determinism**: The same input can result in different LLM outputs.
- **Lack of Ground Truth**: Hard to define a single “correct” answer, especially for open questions.
- **Subjective Criteria**: What’s “good enough” can vary by application (e.g., customer support vs. research summarization).
- **Multiple Failure Modes**: Breakdowns can occur at retrieval, ranking, chunking, or generation stages.

Therefore, a robust RAG evaluation framework must be:

- **Transparent** (trace every step end‐to‐end)
- **Measurable** (collect metrics on recall@k, latency, hallucination rate)
- **Comparable** (run multiple strategies under identical conditions)

---

## 2. LangChain Document Loaders: Ingesting Data Consistently

LangChain standardizes the concept of a **Document** (`page_content`, `metadata`, `id`) and provides hundreds of ready‐made loaders (PDFs, CSVs, HTML, SQL, etc.) :contentReference[oaicite:7]{index=7}.

For this foundational demo, we use `CSVLoader` to fetch **John Wick** reviews from GitHub‐hosted CSV files. Each row maps to a `Document` with:

- `page_content` (review text)
- `metadata` fields:  
  - `Review_Date`, `Review_Title`, `Review_Url`, `Author`, `Rating`  
  - Custom `Movie_Title` (“John Wick 1”, “John Wick 2”, …)  
  - Synthetic `last_accessed_at` timestamp (to simulate temporal evolution)

**Why CSVLoader + rich metadata?**  
- Allows filtering on any metadata field (e.g., “show me reviews rated ≥ 8 after Jan 1, 2024”).  
- Builds a controlled evaluation dataset: same domain (John Wick films 1–4) but variable content length and style.

> **Loading Example (Python)**  
> ```python
> from langchain_community.document_loaders.csv_loader import CSVLoader
> from datetime import datetime, timedelta
> 
> loader = CSVLoader(
>     file_path="data/john_wick_1.csv",
>     metadata_columns=["Review_Date", "Review_Title", "Review_Url", "Author", "Rating"]
> )
> docs = loader.load()
> for doc in docs:
>     doc.metadata["Movie_Title"] = "John Wick 1"
>     doc.metadata["Rating"] = int(doc.metadata.get("Rating", 0) or 0)
>     doc.metadata["last_accessed_at"] = (datetime.now() - timedelta(days=3)).isoformat()
> ```

---

## 3. Six Retrieval Strategies

LangChain’s **Retriever** interface abstracts diverse search mechanisms behind a common pattern :contentReference[oaicite:8]{index=8}. We compare six strategies on the **same “John Wick” dataset**:

1. **Naive Vector Similarity Search**  
   - **Implementation**: `PGVectorStore.as_retriever(search_kwargs={"k":10})`  
   - **Mechanics**: Compute query embedding, fetch top‐k by cosine similarity.  
   - **Pros**: Simple, fast.  
   - **Cons**: Purely semantic—misses lexical matches, can retrieve semantically similar but contextually irrelevant results.

2. **BM25 Lexical Matching**  
   - **Implementation**: `BM25Retriever.from_documents(all_docs)`  
   - **Mechanics**: TF–IDF scoring, good for exact keywords.  
   - **Pros**: Precise for queries with specific keywords.  
   - **Cons**: Lacks semantic understanding; struggles with paraphrases.

3. **Contextual Compression + Reranking**  
   - **Implementation**: `ContextualCompressionRetriever(base_compressor=CohereRerank(…​​), base_retriever=naive_retriever)`  
   - **Mechanics**:  
     1. Retrieve with naive vector.  
     2. Rerank & compress via Cohere’s `rerank-english-v3.0`.  
   - **Pros**: Filters out irrelevant docs, reduces prompt length.  
   - **Cons**: Extra LLM calls (Cohere), higher cost.

4. **Multi‐Query Retrieval**  
   - **Implementation**: `MultiQueryRetriever.from_llm(retriever=naive_retriever, llm=llm)`  
   - **Mechanics**: Generate multiple query expansions (e.g., synonyms, related concepts), then retrieve.  
   - **Pros**: Captures diverse facets of intent.  
   - **Cons**: More LLM calls, added complexity.

5. **Ensemble Retrieval**  
   - **Implementation**:  
     ```python
     retrievers = [bm25_retriever, naive_retriever, comp_retriever, multi_query_retriever]
     ensemble = EnsembleRetriever(retrievers=retrievers, weights=[0.25]*4)
     ```  
   - **Mechanics**: Weighted voting across multiple retrievers; documents scored by sum of weighted ranks.  
   - **Pros**: Leverages complementary strengths (semantic + lexical + compression + multi‐query).  
   - **Cons**: Harder to interpret, tunable hyperparameters (weights).

6. **Semantic Chunking**  
   - **Implementation**:  
     ```python
     from langchain_experimental.text_splitter import SemanticChunker
     chunker = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
     docs_chunks = chunker.split_documents(all_docs)
     semantic_store.aadd_documents(docs_chunks)
     semantic_retriever = semantic_store.as_retriever(search_kwargs={"k":10})
     ```  
   - **Mechanics**: Split documents into semantically coherent segments instead of fixed‐size chunks.  
   - **Pros**: Queries hit highly relevant segments, not arbitrary character intervals.  
   - **Cons**: Preprocessing overhead, more table rows.

By maintaining one standard “retrieval chain” wrapper:

```python
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

retrieval_chain = (
    {"context": itemgetter("question") | some_retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": rag_prompt | llm, "context": itemgetter("context")}
)
````

we ensure **apples‐to‐apples** comparison: each chain uses the same RAG prompt, LLM, and post‐processing.

---

## 4. PostgreSQL + pgvector: A Hybrid Vector Store

Using Postgres + pgvector is often overlooked, but it offers:

* **SQL familiarity**: Leverage existing RDBMS expertise.
* **Hybrid storage**: Store dense vectors alongside relational metadata (e.g., JSON columns).
* **ACID compliance**: Point‐in‐time recovery, transactions.
* **Cost efficiency**: No separate vector‐DB license or provisioning.

### Setup Steps:

1. **Run pgvector-enabled Postgres**

   ```bash
   docker run -it --rm --name pgvector-container \
     -e POSTGRES_USER=langchain \
     -e POSTGRES_PASSWORD=langchain \
     -e POSTGRES_DB=langchain \
     -p 6024:5432 \
     pgvector/pgvector:pg16
   ```

   ([github.com][1])

2. **Install Python Dependencies**

   ```bash
   pip install -qU langchain-postgres langchain-core langchain-openai sqlalchemy
   ```

3. **Initialize Vector Tables**

   ```python
   from langchain_postgres import PGEngine, PGVectorStore

   async_url = "postgresql+asyncpg://langchain:langchain@localhost:6024/langchain"
   pg_engine = PGEngine.from_connection_string(url=async_url)

   await pg_engine.ainit_vectorstore_table(
       table_name="johnwick_baseline_documents",
       vector_size=1536,
   )
   await pg_engine.ainit_vectorstore_table(
       table_name="johnwick_semantic_documents",
       vector_size=1536,
   )

   baseline_store = await PGVectorStore.create(
       engine=pg_engine,
       table_name="johnwick_baseline_documents",
       embedding_service=embeddings,
   )
   semantic_store = await PGVectorStore.create(
       engine=pg_engine,
       table_name="johnwick_semantic_documents",
       embedding_service=embeddings,
   )
   ```

4. **Ingest Documents & Chunks**

   ```python
   await baseline_store.aadd_documents(all_docs)

   chunker = SemanticChunker(embeddings=embeddings, breakpoint_threshold_type="percentile")
   semantic_chunks = chunker.split_documents(all_docs)
   await semantic_store.aadd_documents(semantic_chunks)
   ```

**Inspecting Tables** (optional):

```python
from sqlalchemy import create_engine
import pandas as pd

sync_str = async_url.replace("+asyncpg", "")
sync_engine = create_engine(sync_str)

df_baseline = pd.read_sql_table("johnwick_baseline_documents", sync_engine)
print(df_baseline.head())
```

This hybrid approach allows you to filter or join on JSON metadata (e.g., `WHERE metadata->>'Rating' >= '8'`) while also running fast vector operations (`ORDER BY embedding <-> query_embedding LIMIT 10`).

---

## 5. Arize Phoenix: Observability & Evaluation

**Arize Phoenix** is an open‐source observability platform for AI/LLM pipelines. It provides:

* **Trace Visualization**: See every span from retrieval calls to LLM invocations.
* **Metrics Dashboard**: Track latency, token usage, “hot” spans.
* **Evaluation Templates**: Run RAG‐specific metrics (e.g., recall\@k, answer accuracy) within the UI.
* **Framework‐Agnostic**: Supports LangChain, LlamaIndex, smolagents, etc. ([arize.com][2], [github.com][3]).

### Integrating Phoenix

1. **Launch Phoenix** locally via Docker.

  - Phoenix UI (HTTP collector on port 6006)
  - OTLP gRPC collector on port 4317


   ```bash
    export TS=$(date +"%Y%m%d_%H%M%S")
    docker run -it --rm --name phoenix-container \
    -e PHOENIX_PROJECT_NAME="retrieval-method-comparison-${TS}" \
    -p 6006:6006 \
    -p 4317:4317 \
    arizephoenix/phoenix:latest
   ```

2. **Set environment variable**:

   ```bash
   export PHOENIX_COLLECTOR_ENDPOINT="http://localhost:6006"
   ```

3. **Wrap retrieval chains** using Phoenix’s OpenTelemetry-based decorator:

   ```python
   from phoenix.otel import register

   tracer_provider = register(project_name=project_name, auto_instrument=True)
   tracer = tracer_provider.get_tracer(__name__)

   @tracer.chain
   def trace_naive_retrieval(question: str):
       result = naive_chain.invoke({"question": question})
       return {
           "response": result["response"].content,
           "context_docs": len(result["context"])
       }
   ```

4. **Invoke** each `trace_*` function. Phoenix automatically surfaces each span in its dashboard, highlighting:

   * How many documents each retriever fetched
   * Latency per span (e.g., embedding lookup, reranking)
   * Token usage on LLM calls

Once traces appear in Phoenix UI (`http://localhost:6006`), you can quickly identify bottlenecks:

* Which retrieval strategy takes the longest?
* Does BM25 spike CPU usage relative to vector search?
* Are there recalls\@k differences visible at the retrieval level?

By the time you run **RAGAS** or other automated evaluators, you’ll already have a high‐fidelity picture of where errors or slowdowns occur.

---

## 6. Minimal Execution Flow

Below is a condensed outline of our complete end‐to‐end setup. Refer to the full optimized code for details.

```python
import os, asyncio
from datetime import datetime, timedelta
from pathlib import Path
import requests
from dotenv import load_dotenv
from sqlalchemy import create_engine
import pandas as pd

from phoenix.otel import register
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_postgres import PGEngine, PGVectorStore
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import EnsembleRetriever

async def main():
    # 1. Load environment & configure Phoenix
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
    os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY", "")
    os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://localhost:6006"
    project_name = f"retrieval-method-comparison-{datetime.now():%Y%m%d_%H%M%S}"
    tracer_provider = register(project_name=project_name, auto_instrument=True)
    tracer = tracer_provider.get_tracer(__name__)

    # 2. Init LLM + embeddings
    llm = ChatOpenAI(model="gpt-4.1-mini")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    rag_prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. If you do not know, say 'I don't know'.\n\nQuery:\n{question}\n\nContext:\n{context}"
    )

    # 3. Postgres + pgvector setup (Docker container already up)
    async_url = "postgresql+asyncpg://langchain:langchain@localhost:6024/langchain"
    pg_engine = PGEngine.from_connection_string(url=async_url)
    await pg_engine.ainit_vectorstore_table("johnwick_baseline_documents", vector_size=1536,overwrite_existing=True)
    await pg_engine.ainit_vectorstore_table("johnwick_semantic_documents", vector_size=1536,overwrite_existing=True)
    baseline_store = await PGVectorStore.create(engine=pg_engine, table_name="johnwick_baseline_documents", embedding_service=embeddings)
    semantic_store = await PGVectorStore.create(engine=pg_engine, table_name="johnwick_semantic_documents", embedding_service=embeddings)

    # 4. Download + load CSVs as Document objects
    DATA_DIR = Path.cwd() / "data"
    DATA_DIR.mkdir(exist_ok=True)
    urls = [
        ("https://raw.githubusercontent.com/AI-Maker-Space/DataRepository/main/jw1.csv", "john_wick_1.csv"),
        ("https://raw.githubusercontent.com/AI-Maker-Space/DataRepository/main/jw2.csv", "john_wick_2.csv"),
        ("https://raw.githubusercontent.com/AI-Maker-Space/DataRepository/main/jw3.csv", "john_wick_3.csv"),
        ("https://raw.githubusercontent.com/AI-Maker-Space/DataRepository/main/jw4.csv", "john_wick_4.csv"),
    ]
    all_docs = []
    for idx, (url, fname) in enumerate(urls, start=1):
        target = DATA_DIR / fname
        if not target.exists():
            r = requests.get(url); r.raise_for_status(); target.write_bytes(r.content)
        loader = CSVLoader(file_path=target, metadata_columns=["Review_Date", "Review_Title", "Review_Url", "Author", "Rating"])
        docs = loader.load()
        for doc in docs:
            doc.metadata["Movie_Title"] = f"John Wick {idx}"
            doc.metadata["Rating"] = int(doc.metadata.get("Rating", 0) or 0)
            doc.metadata["last_accessed_at"] = (datetime.now() - timedelta(days=4 - idx)).isoformat()
        all_docs.extend(docs)

    # 5. Ingest docs into baseline & semantic vector stores
    await baseline_store.aadd_documents(all_docs)
    chunker = SemanticChunker(embeddings=embeddings, breakpoint_threshold_type="percentile")
    chunks = chunker.split_documents(all_docs)
    await semantic_store.aadd_documents(chunks)

    # 6. Build retrievers + retrieval chains
    naive_ret   = baseline_store.as_retriever(search_kwargs={"k":10})
    bm25_ret    = BM25Retriever.from_documents(all_docs)
    cohere_rev  = CohereRerank(model="rerank-english-v3.0")
    comp_ret    = ContextualCompressionRetriever(base_compressor=cohere_rev, base_retriever=naive_ret)
    multi_ret   = MultiQueryRetriever.from_llm(retriever=naive_ret, llm=llm)
    ensemble_ret = EnsembleRetriever(retrievers=[bm25_ret, naive_ret, comp_ret, multi_ret], weights=[0.25]*4)
    semantic_ret = semantic_store.as_retriever(search_kwargs={"k":10})

    def make_chain(retriever):
        return (
            {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
            | RunnablePassthrough.assign(context=itemgetter("context"))
            | {"response": rag_prompt | llm, "context": itemgetter("context")}
        )

    chains = {
        "naive": make_chain(naive_ret),
        "bm25": make_chain(bm25_ret),
        "compression": make_chain(comp_ret),
        "multiquery": make_chain(multi_ret),
        "ensemble": make_chain(ensemble_ret),
        "semantic": make_chain(semantic_ret),
    }

    # 7. Create Phoenix-traced versions of each chain
    tracers = {}
    for method_name, chain_callable in chains.items():
        @tracer.chain(name=f"chain.{method_name}")
        def _traced(question: str, fn=chain_callable, strategy=method_name):
            try:
                out = fn.invoke({"question": question})
                return {"response": out["response"].content,"context_docs": len(out["context"]),"retriever": strategy}
            except Exception as e:
                return {"error": str(e), "retriever": strategy}
        _traced.__name__ = f"traced_{method_name}"
        tracers[method_name] = _traced

    print("✅ Phoenix-traced retrieval functions are ready.")


    # 8. Run a sample query through all strategies
    query = "Did people generally like John Wick?"
    results = {label: tracers[label](query)["response"] for label in tracers}

    df_results = pd.DataFrame.from_dict(results, orient="index", columns=["Response"])
    print("\n📊 Retrieval Strategy Outputs:\n", df_results)

    # 9. (Optional) Inspect baseline pgvector table
    sync_str = async_url.replace("+asyncpg", "")
    sync_engine = create_engine(sync_str)
    df_baseline = pd.read_sql_table("johnwick_baseline_documents", con=sync_engine)
    # print(df_baseline.head())

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Further Reading & Citations

- [LangChain Retrievers](https://python.langchain.com/docs/concepts/retrieval/): uniform interface for diverse search backends
* [langchain-postgres](https://github.com/langchain-ai/langchain-postgres): official package implementations of core LangChain abstractions using Postgres
- [pgvector GitHub Repository](https://github.com/pgvector/pgvector): open‐source vector similarity extension for PostgreSQL
- [Arize Phoenix Documentation](https://docs.arize.com/phoenix):  AI observability & evaluation platform (open source)
- [RAGAS Evaluation Framework](https://docs.ragas.io/):  focus of future posts

> Next: In Part 2, we’ll leverage RAGAS to generate **golden test sets**...

```markdown
---

**About the Author**  
This blog is maintained by Don Branson (AI Engineer & Solutions Architect), who has a keen interest in graph solutions and use of evaluations to ensure business value. Don is currently mentoring an AI engineering bootcamp and emphasizes hands‐on, scalable workflows.
```

**Key Takeaways**

1. LangChain provides standardized document loaders for consistent ingestion across formats.
2. A variety of retrieval methods (naive, BM25, compression, multi‐query, ensemble, semantic chunking) can be compared using a single chain interface.
3. PostgreSQL + pgvector is a cost‐effective, hybrid vector store solution that co‐locates dense embeddings with JSON metadata.
4. Arize Phoenix makes RAG pipelines transparent by tracing every retrieval & generation span, facilitating debugging and evaluation.

With this foundation in place, you’re ready to proceed to **Part 2: Golden Test Set Creation using RAGAS**—where we’ll define ground‐truth examples.
