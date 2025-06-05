import os
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path

import asyncpg
import requests
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

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

# ─────────────────────────────────────────────────────────────────────────────────
# 0. Logging & Environment‐level Constants
# ─────────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Load and validate required API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment")
if not COHERE_API_KEY:
    raise RuntimeError("Missing COHERE_API_KEY in environment")

# Phoenix endpoint (for tracing)
PHOENIX_COLLECTOR_ENDPOINT = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:4317")

# PostgreSQL connection parameters (override via .env if desired)
POSTGRES_USER     = "langchain"
POSTGRES_PASSWORD = "langchain"
POSTGRES_HOST     = "localhost"
POSTGRES_PORT     = "6024"
POSTGRES_DB       = "langchain"

# Vector‐store & retrieval configuration
VECTOR_SIZE       = 1536
TABLE_BASELINE    = "johnwick_baseline_documents"
TABLE_SEMANTIC    = "johnwick_semantic_documents"
TOP_K             = int(os.getenv("TOP_K", "10"))

# Directory for raw CSV data
DATA_DIR = Path.cwd() / "data"

# RAG prompt template (could be moved to a file for easy editing)
RAG_TEMPLATE = """\
You are a helpful and kind assistant. Use the context provided below to answer the question.

If you do not know the answer, or are unsure, say you don't know.

Query:
{question}

Context:
{context}
"""


# ─────────────────────────────────────────────────────────────────────────────────
# Helper: Initialize Phoenix tracer
# ─────────────────────────────────────────────────────────────────────────────────
def init_tracer() -> "opentelemetry.trace.Tracer":
    project_name = f"retrieval-method-comparison-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tracer_provider = register(project_name=project_name, auto_instrument=True)
    tracer = tracer_provider.get_tracer(__name__)
    return tracer


# ─────────────────────────────────────────────────────────────────────────────────
# Helper: Setup PostgreSQL & pgvector tables
# ─────────────────────────────────────────────────────────────────────────────────
async def setup_postgres(tracer) -> PGEngine:
    """
    1. Constructs async_postgres URL from environment constants.
    2. Creates PGEngine.
    3. Drops & recreates baseline and semantic tables.
    4. Returns the PGEngine for downstream use.
    """
    async_url = (
        f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )
    pg_engine = PGEngine.from_connection_string(url=async_url)

    with tracer.start_as_current_span("init_baseline_table") as span:
        span.add_event(f"Dropping & creating table {TABLE_BASELINE}")
        await pg_engine.ainit_vectorstore_table(
            table_name=TABLE_BASELINE,
            vector_size=VECTOR_SIZE,
            overwrite_existing=True,
        )

    with tracer.start_as_current_span("init_semantic_table") as span:
        span.add_event(f"Dropping & creating table {TABLE_SEMANTIC}")
        await pg_engine.ainit_vectorstore_table(
            table_name=TABLE_SEMANTIC,
            vector_size=VECTOR_SIZE,
            overwrite_existing=True,
        )

    return pg_engine


# ─────────────────────────────────────────────────────────────────────────────────
# Helper: Download & load all review CSVs
# ─────────────────────────────────────────────────────────────────────────────────
async def load_all_reviews(data_dir: Path, urls: list[tuple[str, str]], tracer) -> list:
    """
    1. Downloads each CSV if not present.
    2. Loads via CSVLoader, annotates metadata.
    3. Returns a list of LangChain Document objects.
    """
    data_dir.mkdir(exist_ok=True)

    all_docs = []
    total = len(urls)

    for idx, (url, fname) in enumerate(urls, start=1):
        file_path = data_dir / fname
        if not file_path.exists():
            with tracer.start_as_current_span("download_csv") as span:
                span.add_event(f"Downloading {fname}")
                resp = requests.get(url)
                resp.raise_for_status()
                file_path.write_bytes(resp.content)
        else:
            with tracer.start_as_current_span("skip_download") as span:
                span.add_event(f"{fname} already downloaded")

        with tracer.start_as_current_span("parse_csv") as span:
            span.add_event(f"Parsing {fname} into documents")
            loader = CSVLoader(
                file_path=file_path,
                metadata_columns=["Review_Date", "Review_Title", "Review_Url", "Author", "Rating"],
            )
            docs = loader.load()

        for doc in docs:
            doc.metadata["Movie_Title"] = f"John Wick {idx}"
            doc.metadata["Rating"] = int(doc.metadata.get("Rating", 0) or 0)
            # Newer entries get a more recent last_accessed_at
            doc.metadata["last_accessed_at"] = (
                datetime.now() - timedelta(days=(total - idx))
            ).isoformat()

        all_docs.extend(docs)

    if not all_docs:
        raise RuntimeError("No documents were loaded—cannot proceed!")

    return all_docs


# ─────────────────────────────────────────────────────────────────────────────────
# Helper: Ingest documents into PGVectorStores, in batches
# ─────────────────────────────────────────────────────────────────────────────────
async def ingest_documents(tracer, pg_engine: PGEngine, all_review_docs: list) -> tuple[PGVectorStore, PGVectorStore]:
    """
    1. Creates baseline and semantic PGVectorStore instances.
    2. Ingests documents in chunks into baseline.
    3. Splits documents via SemanticChunker and ingests into semantic store.
    4. Returns (baseline_vectorstore, semantic_vectorstore).
    """
    # Create vector stores
    baseline_vs = await PGVectorStore.create(
        engine=pg_engine,
        table_name=TABLE_BASELINE,
        embedding_service=OpenAIEmbeddings(model=embeddings),
    )
    semantic_vs = await PGVectorStore.create(
        engine=pg_engine,
        table_name=TABLE_SEMANTIC,
        embedding_service=OpenAIEmbeddings(model=embeddings),
    )

    # Batch size for ingestion
    CHUNK_SIZE = 500

    # Ingest into baseline in chunks
    for i in range(0, len(all_review_docs), CHUNK_SIZE):
        batch = all_review_docs[i : i + CHUNK_SIZE]
        with tracer.start_as_current_span("ingest_baseline_chunk") as span:
            span.add_event(f"Ingesting baseline chunk {i // CHUNK_SIZE + 1}")
            await baseline_vs.aadd_documents(batch)

    # Create semantic chunks
    semantic_chunker = SemanticChunker(
        embeddings=OpenAIEmbeddings(model=embeddings),
        breakpoint_threshold_type="percentile",
    )
    semantic_docs = semantic_chunker.split_documents(all_review_docs)

    if not semantic_docs:
        raise RuntimeError("SemanticChunker returned zero chunks—something is wrong!")

    # Ingest semantic chunks in batches
    for i in range(0, len(semantic_docs), CHUNK_SIZE):
        chunk = semantic_docs[i : i + CHUNK_SIZE]
        with tracer.start_as_current_span("ingest_semantic_chunk") as span:
            span.add_event(f"Ingesting semantic chunk {i // CHUNK_SIZE + 1}")
            await semantic_vs.aadd_documents(chunk)

    return baseline_vs, semantic_vs


# ─────────────────────────────────────────────────────────────────────────────────
# Helper: Build retriever chains dictionary
# ─────────────────────────────────────────────────────────────────────────────────
def build_retrievers(
    all_docs: list,
    baseline_vs: PGVectorStore,
    semantic_vs: PGVectorStore,
    llm: ChatOpenAI,
    rag_prompt: ChatPromptTemplate,
) -> dict[str, RunnablePassthrough]:
    """
    Constructs and returns a dict mapping strategy names to LangChain RunnablePassthrough chains:
      - naive      (vector search)
      - bm25       (BM25Retriever)
      - compression (ContextualCompressionRetriever)
      - multiquery (MultiQueryRetriever)
      - ensemble   (EnsembleRetriever)
      - semantic   (vector search on semantic chunks)
    """
    # Base retriever: simple vector search on baseline
    naive_retriever = baseline_vs.as_retriever(search_kwargs={"k": TOP_K})

    # BM25 Retriever on raw documents
    bm25_retriever = BM25Retriever.from_documents(all_docs)

    # Cohere reranker (requires valid COHERE_API_KEY)
    cohere_rerank = CohereRerank(model="rerank-english-v3.0")

    # ContextualCompressionRetriever: first retrieve via naive, then rerank
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=cohere_rerank, base_retriever=naive_retriever
    )

    # MultiQueryRetriever: splits question into subqueries, uses naive
    multi_query_retriever = MultiQueryRetriever.from_llm(retriever=naive_retriever, llm=llm)

    # EnsembleRetriever: combine BM25, naive, compression, multiquery
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, naive_retriever, compression_retriever, multi_query_retriever],
        weights=[0.25, 0.25, 0.25, 0.25],
    )

    # Semantic retriever: vector search on semantic chunks
    semantic_retriever = semantic_vs.as_retriever(search_kwargs={"k": TOP_K})

    # Helper to wrap in a RunnablePassthrough chain
    def make_chain(retriever) -> RunnablePassthrough:
        return (
            {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
            | RunnablePassthrough.assign(context=itemgetter("context"))
            | {"response": rag_prompt | llm, "context": itemgetter("context")}
        )

    chains = {
        "naive":      make_chain(naive_retriever),
        "bm25":       make_chain(bm25_retriever),
        "compression":make_chain(compression_retriever),
        "multiquery": make_chain(multi_query_retriever),
        "ensemble":   make_chain(ensemble_retriever),
        "semantic":   make_chain(semantic_retriever),
    }

    return chains


# ─────────────────────────────────────────────────────────────────────────────────
# Helper: Create Phoenix‐traced wrapper for each chain
# ─────────────────────────────────────────────────────────────────────────────────
def make_traced(chain_callable: RunnablePassthrough, strategy: str, tracer) -> callable:
    """
    Returns a callable that wraps the given chain_callable in a Phoenix span named "chain.{strategy}".
    Captures "response" and "context_docs" from the chain's output.
    """

    def _inner(question: str) -> dict:
        with tracer.start_as_current_span(f"chain.{strategy}") as span:
            try:
                out = chain_callable.invoke({"question": question})
                span.set_attribute("retriever", strategy)
                span.set_attribute("context_docs", len(out["context"]))
                return {
                    "response": out["response"].content,
                    "context_docs": len(out["context"]),
                    "retriever": strategy,
                }
            except Exception as e:
                span.set_attribute("error", str(e))
                return {"error": str(e), "retriever": strategy}

    _inner.__name__ = f"traced_{strategy}"
    return _inner


# ─────────────────────────────────────────────────────────────────────────────────
# Helper: Run all strategies and produce a DataFrame of responses
# ─────────────────────────────────────────────────────────────────────────────────
def compare_strategies(tracers: dict[str, callable], question: str) -> pd.DataFrame:
    """
    Executes each traced retriever on the given question (synchronously).
    Returns a DataFrame with column ["Response"] and index=row name=strategy.
    """
    results = {}
    for label, fn in tracers.items():
        out = fn(question)
        results[label] = out.get("response", f"<error: {out.get('error')}>")

    df_results = pd.DataFrame.from_dict(results, orient="index", columns=["Response"])
    return df_results


# ─────────────────────────────────────────────────────────────────────────────────
# Helper: Inspect baseline PGVector table if requested via ENV flag
# ─────────────────────────────────────────────────────────────────────────────────
def inspect_baseline_table(async_url: str):
    """
    If INSPECT_BASELINE=true, prints the first few rows of the baseline PGVector table.
    """
    if os.getenv("INSPECT_BASELINE", "false").lower() not in ("1", "true"):
        return

    sync_conn_str = async_url.replace("+asyncpg", "")
    sync_engine = create_engine(sync_conn_str)

    try:
        df_baseline = pd.read_sql_table(TABLE_BASELINE, con=sync_engine)
        logger.info("Baseline table head:\n%s", df_baseline.head())
    except Exception as e:
        logger.error("Error reading baseline table: %s", e)
    finally:
        sync_engine.dispose()


# ─────────────────────────────────────────────────────────────────────────────────
# Main entrypoint
# ─────────────────────────────────────────────────────────────────────────────────
async def main():
    # ──────────────────────────────────────────────────────────────────────────────
    # 1. Initialize Tracer
    # ──────────────────────────────────────────────────────────────────────────────
    tracer = init_tracer()
    with tracer.start_as_current_span("script_start") as span:
        span.add_event("Starting retrieval‐method‐comparison workflow")

    # ──────────────────────────────────────────────────────────────────────────────
    # 2. Initialize LLM, Embeddings, and Prompt
    # ──────────────────────────────────────────────────────────────────────────────
    llm = ChatOpenAI(model="gpt-4.1-mini")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

    # ──────────────────────────────────────────────────────────────────────────────
    # 3. Setup PostgreSQL and PGVector Tables
    # ──────────────────────────────────────────────────────────────────────────────
    pg_engine = await setup_postgres(tracer)

    # ──────────────────────────────────────────────────────────────────────────────
    # 4. Download & Load Review Documents
    # ──────────────────────────────────────────────────────────────────────────────
    urls = [
        ("https://raw.githubusercontent.com/AI-Maker-Space/DataRepository/main/jw1.csv", "john_wick_1.csv"),
        ("https://raw.githubusercontent.com/AI-Maker-Space/DataRepository/main/jw2.csv", "john_wick_2.csv"),
        ("https://raw.githubusercontent.com/AI-Maker-Space/DataRepository/main/jw3.csv", "john_wick_3.csv"),
        ("https://raw.githubusercontent.com/AI-Maker-Space/DataRepository/main/jw4.csv", "john_wick_4.csv"),
    ]
    all_review_docs = await load_all_reviews(DATA_DIR, urls, tracer)

    # ──────────────────────────────────────────────────────────────────────────────
    # 5. Ingest Documents into Vector Stores
    # ──────────────────────────────────────────────────────────────────────────────
    baseline_vs, semantic_vs = await ingest_documents(tracer, pg_engine, all_review_docs)

    # ──────────────────────────────────────────────────────────────────────────────
    # 6. Build Retriever Chains
    # ──────────────────────────────────────────────────────────────────────────────
    chains = build_retrievers(all_review_docs, baseline_vs, semantic_vs, llm, rag_prompt)

    # ──────────────────────────────────────────────────────────────────────────────
    # 7. Wrap Chains in Phoenix‐traced Functions
    # ──────────────────────────────────────────────────────────────────────────────
    tracers = {name: make_traced(chain_callable, name, tracer) for name, chain_callable in chains.items()}
    logger.info("✅ Phoenix‐traced retrieval functions are ready.")

    # ──────────────────────────────────────────────────────────────────────────────
    # 8. Execute & Compare All Strategies
    # ──────────────────────────────────────────────────────────────────────────────
    question = "Did people generally like John Wick?"
    df_results = compare_strategies(tracers, question)
    logger.info("📊 Retrieval Strategy Outputs:\n%s", df_results)

    # ──────────────────────────────────────────────────────────────────────────────
    # 9. (Optional) Inspect baseline PGVector table
    # ──────────────────────────────────────────────────────────────────────────────
    async_url = (
        f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )
    inspect_baseline_table(async_url)

    # ──────────────────────────────────────────────────────────────────────────────
    # 10. Cleanup
    # ──────────────────────────────────────────────────────────────────────────────
    pg_engine.close()
    with tracer.start_as_current_span("script_end") as span:
        span.add_event("Finished retrieval‐method‐comparison workflow")


if __name__ == "__main__":
    asyncio.run(main())
