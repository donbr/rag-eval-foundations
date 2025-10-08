import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from operator import itemgetter
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_cohere import CohereRerank
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGEngine, PGVectorStore

# Phoenix setup - using latest 2025 best practices with arize-phoenix-otel
from phoenix.otel import register

# Import shared configuration
from config import (
    BASELINE_TABLE,
    EMBEDDING_MODEL,
    LLM_MODEL,
    PHOENIX_ENDPOINT,
    SEMANTIC_TABLE,
    VECTOR_SIZE,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress verbose HTTP request logs since we have Phoenix tracing
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)

# Centralized prompt template
RAG_PROMPT = ChatPromptTemplate.from_template("""You are a helpful assistant. Use the context below to answer the question.
If you don't know the answer, say you don't know.

Question: {question}
Context: {context}""")


@dataclass
class Config:
    """E2E Pipeline-specific configuration (data loading settings)

    Note: Database, Phoenix, and model settings now come from shared config.py
    This class only contains settings specific to the E2E pipeline execution.
    """

    # API Keys
    openai_api_key: str
    cohere_api_key: str

    # Phoenix project naming (timestamped for E2E runs)
    project_name: str = (
        f"retrieval-evaluation-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    # Data settings (E2E-specific)
    data_urls: list[tuple] = None
    load_pdfs: bool = True  # Flag to enable/disable PDF loading
    load_csvs: bool = (
        False  # Flag to enable/disable CSV loading (disabled for PDF-only processing)
    )
    overwrite_existing_tables: bool = True

    def __post_init__(self):
        if self.data_urls is None:
            self.data_urls = [
                (
                    "https://raw.githubusercontent.com/AI-Maker-Space/DataRepository/main/jw1.csv",
                    "john_wick_1.csv",
                ),
                (
                    "https://raw.githubusercontent.com/AI-Maker-Space/DataRepository/main/jw2.csv",
                    "john_wick_2.csv",
                ),
                (
                    "https://raw.githubusercontent.com/AI-Maker-Space/DataRepository/main/jw3.csv",
                    "john_wick_3.csv",
                ),
                (
                    "https://raw.githubusercontent.com/AI-Maker-Space/DataRepository/main/jw4.csv",
                    "john_wick_4.csv",
                ),
            ]

    @property
    def async_url(self) -> str:
        return f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"


def setup_environment() -> Config:
    """Setup environment and return configuration"""
    load_dotenv()

    config = Config(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        cohere_api_key=os.getenv("COHERE_API_KEY", ""),
    )

    # Set environment variables
    os.environ["OPENAI_API_KEY"] = config.openai_api_key
    os.environ["COHERE_API_KEY"] = config.cohere_api_key
    os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = PHOENIX_ENDPOINT

    return config


def setup_phoenix_tracing(config: Config):
    """Setup Phoenix tracing with auto-instrumentation (best practice)"""
    return register(project_name=config.project_name, auto_instrument=True, batch=True)


async def setup_vector_store(
    config: Config, table_name: str, embeddings
) -> PGVectorStore:
    """Reusable function to setup vector stores"""
    pg_engine = PGEngine.from_connection_string(url=config.async_url)

    await pg_engine.ainit_vectorstore_table(
        table_name=table_name,
        vector_size=VECTOR_SIZE,
        overwrite_existing=config.overwrite_existing_tables,
    )

    return await PGVectorStore.create(
        engine=pg_engine,
        table_name=table_name,
        embedding_service=embeddings,
    )


def create_retrievers(
    baseline_vectorstore, semantic_vectorstore, all_docs, llm
) -> dict[str, Any]:
    """Create all retrieval strategies"""
    retrievers = {}

    # Basic retrievers
    retrievers["naive"] = baseline_vectorstore.as_retriever(search_kwargs={"k": 10})
    retrievers["semantic"] = semantic_vectorstore.as_retriever(search_kwargs={"k": 10})
    retrievers["bm25"] = BM25Retriever.from_documents(all_docs)

    # Advanced retrievers
    cohere_rerank = CohereRerank(model="rerank-english-v3.0")
    retrievers["compression"] = ContextualCompressionRetriever(
        base_compressor=cohere_rerank, base_retriever=retrievers["naive"]
    )

    retrievers["multiquery"] = MultiQueryRetriever.from_llm(
        retriever=retrievers["naive"], llm=llm
    )

    retrievers["ensemble"] = EnsembleRetriever(
        retrievers=[
            retrievers["bm25"],
            retrievers["naive"],
            retrievers["compression"],
            retrievers["multiquery"],
        ],
        weights=[0.25, 0.25, 0.25, 0.25],
    )

    return retrievers


async def load_pdf_documents(data_dir: Path) -> list:
    """Load PDF documents from the data directory"""
    pdf_docs = []
    pdf_files = list(data_dir.glob("*.pdf"))

    if not pdf_files:
        logger.info("No PDF files found in data directory")
        return pdf_docs

    logger.info(f"Found {len(pdf_files)} PDF files to load")

    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()

            # Add metadata for PDFs
            for doc in docs:
                doc.metadata.update(
                    {
                        "source_type": "pdf",
                        "document_name": pdf_file.stem,
                        "last_accessed_at": datetime.now().isoformat(),
                    }
                )

            pdf_docs.extend(docs)
            logger.info(f"Loaded {len(docs)} pages from {pdf_file.name}")

        except Exception as e:
            logger.error(f"Error loading PDF {pdf_file.name}: {e}")
            continue

    return pdf_docs


async def load_and_process_data(config: "Config") -> list:
    """Load and process both CSV and PDF data"""
    data_dir = Path.cwd() / "data"
    data_dir.mkdir(exist_ok=True)

    all_docs = []
    csv_docs = []
    pdf_docs = []

    # Load CSV data (John Wick reviews) - only if enabled
    if config.load_csvs:
        logger.info("📥 Loading CSV data...")
        for idx, (url, filename) in enumerate(config.data_urls, start=1):
            file_path = data_dir / filename

            # Download if not exists
            if not file_path.exists():
                try:
                    response = requests.get(url)
                    response.raise_for_status()
                    file_path.write_bytes(response.content)
                except requests.RequestException as e:
                    logger.error(f"Error downloading {filename}: {e}")
                    continue

            # Load documents
            try:
                loader = CSVLoader(
                    file_path=file_path,
                    metadata_columns=[
                        "Review_Date",
                        "Review_Title",
                        "Review_Url",
                        "Author",
                        "Rating",
                    ],
                )
                docs = loader.load()

                # Add metadata
                for doc in docs:
                    doc.metadata.update(
                        {
                            "source_type": "csv",
                            "Movie_Title": f"John Wick {idx}",
                            "Rating": int(doc.metadata.get("Rating", 0) or 0),
                            "last_accessed_at": (
                                datetime.now() - timedelta(days=4 - idx)
                            ).isoformat(),
                        }
                    )

                csv_docs.extend(docs)

            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")
                continue

        logger.info(f"Loaded {len(csv_docs)} CSV documents")
        all_docs.extend(csv_docs)

    # Load PDF data if enabled
    if config.load_pdfs:
        logger.info("📄 Loading PDF data...")
        pdf_docs = await load_pdf_documents(data_dir)
        logger.info(f"Loaded {len(pdf_docs)} PDF documents")
        all_docs.extend(pdf_docs)

    logger.info(
        f"📊 Total documents loaded: {len(all_docs)} (CSV: {len(csv_docs)}, PDF: {len(pdf_docs)})"
    )

    return all_docs


def create_rag_chain(retriever, llm, method_name: str):
    """Create a simple RAG chain with method identification - Phoenix auto-traces this"""
    chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
        }
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": RAG_PROMPT | llm, "context": itemgetter("context")}
    )

    # Use uniform span name with retriever tag for easier Phoenix filtering
    return chain.with_config(
        {
            "run_name": f"rag_chain_{method_name}",
            "span_attributes": {"retriever": method_name},
        }
    )


async def run_evaluation(question: str, chains: dict[str, Any]) -> dict[str, str]:
    """Run evaluation across all retrieval strategies"""
    results = {}

    for method_name, chain in chains.items():
        try:
            result = await chain.ainvoke({"question": question})
            response_content = result["response"].content
            results[method_name] = response_content
        except Exception as e:
            logger.error(f"Error with {method_name}: {e}")
            results[method_name] = f"Error: {str(e)}"

    return results


async def main():
    """Main execution function - loads both CSV and PDF documents for RAG evaluation"""
    try:
        # Setup
        config = setup_environment()
        setup_phoenix_tracing(config)

        logger.info(f"✅ Phoenix tracing configured for project: {config.project_name}")
        logger.info(
            f"📁 Table names: baseline='{BASELINE_TABLE}', semantic='{SEMANTIC_TABLE}'"
        )

        # Initialize models
        llm = ChatOpenAI(model=LLM_MODEL)
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

        # Load data
        logger.info("📥 Loading and processing documents from CSV and PDF sources...")
        all_docs = await load_and_process_data(config)

        if not all_docs:
            raise ValueError("No documents loaded successfully")

        # Setup vector stores
        logger.info("🔧 Setting up vector stores...")
        baseline_vectorstore = await setup_vector_store(
            config, BASELINE_TABLE, embeddings
        )
        semantic_vectorstore = await setup_vector_store(
            config, SEMANTIC_TABLE, embeddings
        )

        # Ingest data
        logger.info("📊 Ingesting documents...")
        await baseline_vectorstore.aadd_documents(all_docs)

        # Semantic chunking
        semantic_chunker = SemanticChunker(
            embeddings=embeddings, breakpoint_threshold_type="percentile"
        )
        semantic_docs = semantic_chunker.split_documents(all_docs)
        await semantic_vectorstore.aadd_documents(semantic_docs)

        # Create retrievers and chains
        logger.info("⚙️ Creating retrieval strategies...")
        retrievers = create_retrievers(
            baseline_vectorstore, semantic_vectorstore, all_docs, llm
        )

        # Create RAG chains
        chains = {
            name: create_rag_chain(retriever, llm, name)
            for name, retriever in retrievers.items()
        }

        # Run evaluation
        logger.info("🔍 Running evaluation...")
        # Use questions appropriate for financial aid documents
        test_questions = [
            "What are the eligibility requirements for Federal Pell Grants?",
            "How does the Direct Loan Program work?",
            "What is the process for verifying financial aid applications?",
        ]
        question = test_questions[0]  # Start with the first question

        results = await run_evaluation(question, chains)

        # Log results
        logger.info("\n📊 Retrieval Strategy Results:")
        logger.info("=" * 50)
        for method, response in results.items():
            logger.info(f"\n{method:15} {response}")

        logger.info(f"\n✅ Evaluation complete! View traces at: {PHOENIX_ENDPOINT}")

    except Exception as e:
        logger.error(f"❌ Error during execution: {e}")
        raise
    finally:
        logger.info("🔄 Cleanup completed")


if __name__ == "__main__":
    asyncio.run(main())
