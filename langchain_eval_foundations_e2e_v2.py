import os
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any

import requests
from dotenv import load_dotenv

# Phoenix setup - using latest 2025 best practices with arize-phoenix-otel
from phoenix.otel import register
from opentelemetry import trace
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
    """Centralized configuration management"""
    # API Keys
    openai_api_key: str
    cohere_api_key: str
    
    # Phoenix settings
    phoenix_endpoint: str = "http://localhost:6006"
    project_name: str = f"retrieval-evaluation-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Database settings
    postgres_user: str = "langchain"
    postgres_password: str = "langchain"
    postgres_host: str = "localhost"
    postgres_port: str = "6024"
    postgres_db: str = "langchain"
    vector_size: int = 1536
    table_baseline: str = "johnwick_baseline_documents"
    table_semantic: str = "johnwick_semantic_documents"
    overwrite_existing_tables: bool = True
    
    # Model settings
    model_name: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"
    
    # Data settings
    data_urls: List[tuple] = None
    
    def __post_init__(self):
        if self.data_urls is None:
            self.data_urls = [
                ("https://raw.githubusercontent.com/AI-Maker-Space/DataRepository/main/jw1.csv", "john_wick_1.csv"),
                ("https://raw.githubusercontent.com/AI-Maker-Space/DataRepository/main/jw2.csv", "john_wick_2.csv"),
                ("https://raw.githubusercontent.com/AI-Maker-Space/DataRepository/main/jw3.csv", "john_wick_3.csv"),
                ("https://raw.githubusercontent.com/AI-Maker-Space/DataRepository/main/jw4.csv", "john_wick_4.csv"),
            ]
    
    @property
    def async_url(self) -> str:
        return f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"


def setup_environment() -> Config:
    """Setup environment and return configuration"""
    load_dotenv()
    
    config = Config(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        cohere_api_key=os.getenv("COHERE_API_KEY", "")
    )
    
    # Set environment variables
    os.environ["OPENAI_API_KEY"] = config.openai_api_key
    os.environ["COHERE_API_KEY"] = config.cohere_api_key
    os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = config.phoenix_endpoint
    
    return config


def setup_phoenix_tracing(config: Config):
    """Setup Phoenix tracing with auto-instrumentation (best practice)"""
    return register(
        project_name=config.project_name,
        auto_instrument=True,
        batch=True
    )


async def setup_vector_store(config: Config, table_name: str, embeddings) -> PGVectorStore:
    """Reusable function to setup vector stores"""
    pg_engine = PGEngine.from_connection_string(url=config.async_url)
    
    await pg_engine.ainit_vectorstore_table(
        table_name=table_name,
        vector_size=config.vector_size,
        overwrite_existing=config.overwrite_existing_tables,
    )
    
    return await PGVectorStore.create(
        engine=pg_engine,
        table_name=table_name,
        embedding_service=embeddings,
    )

def create_retrievers(baseline_vectorstore, semantic_vectorstore, all_docs, llm) -> Dict[str, Any]:
    """Create all retrieval strategies"""
    retrievers = {}
    
    # Basic retrievers
    retrievers["naive"] = baseline_vectorstore.as_retriever(search_kwargs={"k": 10})
    retrievers["semantic"] = semantic_vectorstore.as_retriever(search_kwargs={"k": 10})
    retrievers["bm25"] = BM25Retriever.from_documents(all_docs)
    
    # Advanced retrievers
    cohere_rerank = CohereRerank(model="rerank-english-v3.0")
    retrievers["compression"] = ContextualCompressionRetriever(
        base_compressor=cohere_rerank,
        base_retriever=retrievers["naive"]
    )
    
    retrievers["multiquery"] = MultiQueryRetriever.from_llm(
        retriever=retrievers["naive"],
        llm=llm
    )
    
    retrievers["ensemble"] = EnsembleRetriever(
        retrievers=[
            retrievers["bm25"], 
            retrievers["naive"], 
            retrievers["compression"], 
            retrievers["multiquery"]
        ],
        weights=[0.25, 0.25, 0.25, 0.25]
    )
    
    return retrievers

async def load_and_process_data(config: "Config") -> List:
    """Load and process John Wick movie review data"""
    data_dir = Path.cwd() / "data"
    data_dir.mkdir(exist_ok=True)
    
    all_docs = []
    
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
                metadata_columns=["Review_Date", "Review_Title", "Review_Url", "Author", "Rating"]
            )
            docs = loader.load()
            
            # Add metadata
            for doc in docs:
                doc.metadata.update({
                    "Movie_Title": f"John Wick {idx}",
                    "Rating": int(doc.metadata.get("Rating", 0) or 0),
                    "last_accessed_at": (datetime.now() - timedelta(days=4 - idx)).isoformat()
                })
            
            all_docs.extend(docs)
            
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            continue
    
    return all_docs

def create_rag_chain(retriever, llm, method_name: str):
    """Create a simple RAG chain with method identification - Phoenix auto-traces this"""
    chain = (
        {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": RAG_PROMPT | llm, "context": itemgetter("context")}
    )
    
    # Use uniform span name with retriever tag for easier Phoenix filtering
    return chain.with_config({
        "run_name": f"rag_chain_{method_name}",
        "span_attributes": {"retriever": method_name}
    })

async def run_evaluation(question: str, traced_chains: Dict[str, Any]) -> Dict[str, str]:
    """Run evaluation across all traced retrieval strategies with proper span nesting"""
    from opentelemetry import trace
    
    otel_tracer = trace.get_tracer(__name__)
    
    async def run_single_method(method_name: str, chain):
        """Run a single method evaluation with its own child span"""
        # Child spans will be nested under the parent span
        with otel_tracer.start_as_current_span(f"retrieval_method_{method_name}") as span:
            span.set_attribute("method_name", method_name)
            span.set_attribute("question", question)
            span.set_attribute("retrieval_strategy", method_name)
            
            try:
                result = await chain(question)
                response_content = result["response"].content
                span.set_attribute("response_length", len(response_content))
                span.set_attribute("context_count", len(result.get("context", [])))
                span.set_attribute("status", "success")
                return response_content
            except Exception as e:
                span.set_attribute("status", "error")
                span.set_attribute("error_message", str(e))
                span.record_exception(e)
                logger.error(f"Error with {method_name}: {e}")
                return f"Error: {str(e)}"
    
    # Run evaluation methods sequentially to ensure proper span nesting
    # This will create child spans that are properly nested under the parent
    results = {}
    for name, chain in traced_chains.items():
        results[name] = await run_single_method(name, chain)
    
    return results


async def main():
    """Main execution function"""
    try:
        # Setup
        config = setup_environment()
        tracer_provider = setup_phoenix_tracing(config)

        logger.info(f"✅ Phoenix tracing configured for project: {config.project_name}")
        
        # Initialize models
        llm = ChatOpenAI(model=config.model_name)
        embeddings = OpenAIEmbeddings(model=config.embedding_model)
        
        # Load data
        logger.info("📥 Loading and processing data...")
        all_docs = await load_and_process_data(config)
        
        if not all_docs:
            raise ValueError("No documents loaded successfully")
        
        # Setup vector stores
        logger.info("🔧 Setting up vector stores...")
        baseline_vectorstore = await setup_vector_store(config, config.table_baseline, embeddings)
        semantic_vectorstore = await setup_vector_store(config, config.table_semantic, embeddings)
        
        # Ingest data
        logger.info("📊 Ingesting documents...")
        await baseline_vectorstore.aadd_documents(all_docs)
        
        # Semantic chunking
        semantic_chunker = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile"
        )
        semantic_docs = semantic_chunker.split_documents(all_docs)
        await semantic_vectorstore.aadd_documents(semantic_docs)
        
        # Create retrievers and chains
        logger.info("⚙️ Creating retrieval strategies...")
        retrievers = create_retrievers(baseline_vectorstore, semantic_vectorstore, all_docs, llm)
        
        # Create RAG chains
        chains = {
            name: create_rag_chain(retriever, llm, name)
            for name, retriever in retrievers.items()
        }

        # Create async traced wrappers using Phoenix decorators
        tracer = tracer_provider.get_tracer(__name__)
        traced_chains = {}
        
        for method_name, chain_callable in chains.items():
            @tracer.chain(name=f"evaluation.{method_name}")
            async def traced_chain(question: str, chain=chain_callable):
                return await chain.ainvoke({"question": question})
            
            traced_chains[method_name] = traced_chain
        
        # Run evaluation
        logger.info("🔍 Running evaluation...")
        question = "Did people generally like John Wick?"
        
        # Get OpenTelemetry tracer for manual span creation
        otel_tracer = trace.get_tracer(__name__)
        
        with otel_tracer.start_as_current_span("evaluation_all_methods") as span:
            span.set_attribute("question", question)
            span.set_attribute("num_methods", len(traced_chains))
            span.set_attribute("retrieval_methods", list(traced_chains.keys()))
            span.set_attribute("evaluation_type", "parallel_retrieval_comparison")
            
            results = await run_evaluation(question, traced_chains)
            
            # Add result summary to parent span
            span.set_attribute("evaluation_completed", True)
            span.set_attribute("successful_methods", len([r for r in results.values() if not r.startswith("Error:")]))
            span.set_attribute("failed_methods", len([r for r in results.values() if r.startswith("Error:")]))
        
        # Log results
        logger.info("\n📊 Retrieval Strategy Results:")
        logger.info("=" * 50)
        for method, response in results.items():
            logger.info(f"\n{method:15} {response}")
        
        logger.info(f"\n✅ Evaluation complete! View traces at: {config.phoenix_endpoint}")
        
    except Exception as e:
        logger.error(f"❌ Error during execution: {e}")
        raise
    finally:
        logger.info("🔄 Cleanup completed")


if __name__ == "__main__":
    asyncio.run(main())