import os
import asyncio
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

async def main():
    # ──────────────────────────────────────────────────────────────────────────────
    # 1. Environment & Tracer Setup
    # ──────────────────────────────────────────────────────────────────────────────
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
    os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY", "")
    os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://localhost:4317"

    project_name = f"retrieval-method-comparison-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tracer_provider = register(project_name=project_name, auto_instrument=True)
    tracer = tracer_provider.get_tracer(__name__)

    # ──────────────────────────────────────────────────────────────────────────────
    # 2. Initialize LLM + Embeddings
    # ──────────────────────────────────────────────────────────────────────────────
    llm = ChatOpenAI(model="gpt-4.1-mini")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    RAG_TEMPLATE = """\
You are a helpful and kind assistant. Use the context provided below to answer the question.

If you do not know the answer, or are unsure, say you don't know.

Query:
{question}

Context:
{context}
"""
    rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

    # ──────────────────────────────────────────────────────────────────────────────
    # 3. Postgres + pgvector Setup (with DuplicateTableError handling)
    # ──────────────────────────────────────────────────────────────────────────────

    POSTGRES_USER     = "langchain"
    POSTGRES_PASSWORD = "langchain"
    POSTGRES_HOST     = "localhost"
    POSTGRES_PORT     = "6024"
    POSTGRES_DB       = "langchain"
    VECTOR_SIZE       = 1536
    TABLE_BASELINE    = "johnwick_baseline_documents"
    TABLE_SEMANTIC    = "johnwick_semantic_documents"

    async_url = (
        f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )
    pg_engine = PGEngine.from_connection_string(url=async_url)

    # ──────────────────────────────────────────────────────────────────────────────
    # Drop & recreate baseline table
    # ──────────────────────────────────────────────────────────────────────────────
    await pg_engine.ainit_vectorstore_table(
        table_name=TABLE_BASELINE,
        vector_size=VECTOR_SIZE,
        overwrite_existing=True,  # drop if exists, then create
    )

    # ──────────────────────────────────────────────────────────────────────────────
    # Drop & recreate semantic table
    # ──────────────────────────────────────────────────────────────────────────────
    await pg_engine.ainit_vectorstore_table(
        table_name=TABLE_SEMANTIC,
        vector_size=VECTOR_SIZE,
        overwrite_existing=True,  # drop if exists, then create
    )

    baseline_vectorstore = await PGVectorStore.create(
        engine=pg_engine,
        table_name=TABLE_BASELINE,
        embedding_service=embeddings,
    )
    semantic_vectorstore = await PGVectorStore.create(
        engine=pg_engine,
        table_name=TABLE_SEMANTIC,
        embedding_service=embeddings,
    )

    # ──────────────────────────────────────────────────────────────────────────────
    # 4. Data Ingestion
    # ──────────────────────────────────────────────────────────────────────────────
    DATA_DIR = Path.cwd() / "data"
    DATA_DIR.mkdir(exist_ok=True)
    urls = [
        ("https://raw.githubusercontent.com/AI-Maker-Space/DataRepository/main/jw1.csv", "john_wick_1.csv"),
        ("https://raw.githubusercontent.com/AI-Maker-Space/DataRepository/main/jw2.csv", "john_wick_2.csv"),
        ("https://raw.githubusercontent.com/AI-Maker-Space/DataRepository/main/jw3.csv", "john_wick_3.csv"),
        ("https://raw.githubusercontent.com/AI-Maker-Space/DataRepository/main/jw4.csv", "john_wick_4.csv"),
    ]

    all_review_docs = []
    for idx, (url, fname) in enumerate(urls, start=1):
        file_path = DATA_DIR / fname
        if not file_path.exists():
            resp = requests.get(url)
            resp.raise_for_status()
            file_path.write_bytes(resp.content)

        loader = CSVLoader(
            file_path=file_path,
            metadata_columns=["Review_Date", "Review_Title", "Review_Url", "Author", "Rating"]
        )
        docs = loader.load()
        for doc in docs:
            doc.metadata["Movie_Title"]     = f"John Wick {idx}"
            doc.metadata["Rating"]          = int(doc.metadata.get("Rating", 0) or 0)
            doc.metadata["last_accessed_at"] = (datetime.now() - timedelta(days=4 - idx)).isoformat()
        all_review_docs.extend(docs)

    # ──────────────────────────────────────────────────────────────────────────────
    # 5. Ingest into Vector Stores
    # ──────────────────────────────────────────────────────────────────────────────
    await baseline_vectorstore.aadd_documents(all_review_docs)

    semantic_chunker = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile"
    )
    semantic_docs = semantic_chunker.split_documents(all_review_docs)
    await semantic_vectorstore.aadd_documents(semantic_docs)

    # ──────────────────────────────────────────────────────────────────────────────
    # 6. Build Retriever Chains
    # ──────────────────────────────────────────────────────────────────────────────
    naive_retriever   = baseline_vectorstore.as_retriever(search_kwargs={"k": 10})
    bm25_retriever    = BM25Retriever.from_documents(all_review_docs)
    cohere_rerank     = CohereRerank(model="rerank-english-v3.0")
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=cohere_rerank,
        base_retriever=naive_retriever
    )
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=naive_retriever,
        llm=llm
    )
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, naive_retriever, compression_retriever, multi_query_retriever],
        weights=[0.25, 0.25, 0.25, 0.25]
    )
    semantic_retriever = semantic_vectorstore.as_retriever(search_kwargs={"k": 10})

    def make_chain(retriever):
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

    # ──────────────────────────────────────────────────────────────────────────────
    # 7. TRACE: Unique Span Names + Retriever Tag
    # ──────────────────────────────────────────────────────────────────────────────
    tracers = {}
    for method_name, chain_callable in chains.items():
        @tracer.chain(name=f"chain.{method_name}")
        def _traced(question: str, fn=chain_callable, strategy=method_name):
            try:
                out = fn.invoke({"question": question})
                return {
                    "response": out["response"].content,
                    "context_docs": len(out["context"]),
                    "retriever": strategy,  # added as span attribute
                }
            except Exception as e:
                return {"error": str(e), "retriever": strategy}

        _traced.__name__ = f"traced_{method_name}"
        tracers[method_name] = _traced

    print("✅ Phoenix-traced retrieval functions are ready.")

    # ──────────────────────────────────────────────────────────────────────────────
    # 8. Execute & Compare All Strategies
    # ──────────────────────────────────────────────────────────────────────────────
    question = "Did people generally like John Wick?"
    results = {label: tracers[label](question)["response"] for label in tracers}

    df_results = pd.DataFrame.from_dict(results, orient="index", columns=["Response"])
    print("\n📊 Retrieval Strategy Outputs:\n", df_results)

    # ──────────────────────────────────────────────────────────────────────────────
    # 9. (Optional) Inspect pgvector table via pandas
    # ──────────────────────────────────────────────────────────────────────────────
    sync_conn_str = async_url.replace("+asyncpg", "")
    sync_engine = create_engine(sync_conn_str)
    df_baseline = pd.read_sql_table("johnwick_baseline_documents", con=sync_engine)
    print(df_baseline.head())

if __name__ == "__main__":
    asyncio.run(main())
