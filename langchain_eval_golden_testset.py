# optimized_golden_testset.py

import os
from pathlib import Path
from datetime import datetime, timedelta

import requests
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from langchain_core.documents import Document

from phoenix.otel import register
import phoenix as px

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import default_transforms, apply_transforms
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import (
    SingleHopSpecificQuerySynthesizer,
    MultiHopAbstractQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer,
)


def load_docs_from_postgres(
    table_name: str = "johnwick_baseline_documents",
) -> list[Document]:
    """
    Loads documents from a PostgreSQL table into a list of LangChain Documents.

    This function connects to the PostgreSQL database using credentials from environment
    variables, reads the specified table into a pandas DataFrame, and then converts
    each row into a LangChain Document object.

    Args:
        table_name: The name of the table to load the documents from.

    Returns:
        A list of LangChain Document objects.
    """
    # Load environment variables from .env file for local development
    load_dotenv()

    # Retrieve database connection details from environment variables
    # with sensible defaults for the project.
    POSTGRES_USER = os.getenv("POSTGRES_USER", "langchain")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "langchain")
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT", "6024")
    POSTGRES_DB = os.getenv("POSTGRES_DB", "langchain")

    # Construct the synchronous database connection string for SQLAlchemy
    sync_conn_str = (
        f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@"
        f"{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )

    # Create a SQLAlchemy engine to connect to the database
    engine = create_engine(sync_conn_str)

    # Use a direct SQL query to select specific columns.
    # This avoids issues with pandas/SQLAlchemy not recognizing the 'vector' type,
    # which can cause table reflection to fail and result in missing columns.
    try:
        with engine.connect() as connection:
            # Quoting the table name makes it case-sensitive and handles special chars
            query = text(f'SELECT content, langchain_metadata FROM "{table_name}"')
            df = pd.read_sql_query(query, connection)
    except Exception as e:
        print(f"Error executing query: {e}")
        print("Please ensure the table name is correct and that the 'langchain-eval-foundations-e2e.py' script has run successfully.")
        return []

    # Convert the DataFrame to a list of LangChain Document objects.
    # The PGVectorStore integration stores page content in 'content'
    # and metadata in the 'langchain_metadata' JSONB column.
    documents = [
        Document(
            page_content=row["content"],
            metadata=row["langchain_metadata"],
        )
        for _, row in df.iterrows()
    ]

    print(f"Successfully loaded {len(documents)} documents from the '{table_name}' table.")
    return documents


def download_csvs(data_dir: Path, urls: list[tuple[str, str]], tracer):
    """
    Download CSV files if not already present.
    """
    for url, fname in urls:
        file_path = data_dir / fname
        if not file_path.exists():
            with tracer.start_as_current_span("csv_download") as span:
                span.add_event(f"Downloading {fname}")
            resp = requests.get(url)
            resp.raise_for_status()
            file_path.write_bytes(resp.content)
        else:
            with tracer.start_as_current_span("csv_skip") as span:
                span.add_event(f"{fname} already exists.")


def load_and_annotate_docs(data_dir: Path, tracer) -> list:
    """
    Load CSVs via CSVLoader and annotate metadata (Movie_Title, Rating, last_accessed_at).
    """
    from langchain_community.document_loaders.csv_loader import CSVLoader

    all_docs = []
    csv_files = sorted(data_dir.glob("john_wick_*.csv"))
    total = len(csv_files)

    for i, csv_path in enumerate(csv_files, start=1):
        with tracer.start_as_current_span("csv_load") as span:
            span.add_event(f"Loading {csv_path.name}")
        loader = CSVLoader(
            file_path=csv_path,
            metadata_columns=["Review_Date", "Review_Title", "Review_Url", "Author", "Rating"],
        )
        docs = loader.load()
        for doc in docs:
            doc.metadata["Movie_Title"] = f"John Wick {i}"
            doc.metadata["Rating"] = int(doc.metadata.get("Rating") or 0)
            # Newer movies → more recent last_accessed_at
            doc.metadata["last_accessed_at"] = (
                datetime.now() - timedelta(days=(total - i))
            ).isoformat()
        all_docs.extend(docs)

    return all_docs


def build_knowledge_graph(docs: list, llm, embeddings, tracer) -> KnowledgeGraph:
    """
    Build a RAGAS KnowledgeGraph from documents, apply default transforms.
    """
    kg = KnowledgeGraph()

    # Add a document node for each chunk
    for doc in docs:
        with tracer.start_as_current_span("kg_node_add") as span:
            node_id = hash(doc.page_content) % 1_000_000
            span.add_event(f"Adding document node id={node_id}")

        kg.nodes.append(
            Node(
                type=NodeType.DOCUMENT,
                properties={
                    "page_content": doc.page_content,
                    "document_metadata": doc.metadata,
                },
            )
        )

    # Compute transforms
    with tracer.start_as_current_span("kg_transforms") as span:
        span.add_event("Computing default transforms")
        transforms = default_transforms(
            documents=docs, llm=llm, embedding_model=embeddings
        )

    # Apply transforms
    with tracer.start_as_current_span("kg_apply_transforms") as span:
        span.add_event("Applying transforms")
        apply_transforms(kg, transforms)

    return kg


def generate_and_push_testset(
    kg: KnowledgeGraph, llm, embeddings, tracer, hf_repo: str, testset_size: int = 10
):
    """
    Generate a RAGAS gold testset of size `testset_size`, push to HF Hub, and return the testset object.
    """
    with tracer.start_as_current_span("testset_init") as span:
        span.add_event("Initializing TestsetGenerator")
        generator = TestsetGenerator(llm=llm, embedding_model=embeddings, knowledge_graph=kg)

    # Define distribution: 50% single-hop, 25% multi-hop abstract, 25% multi-hop specific
    query_distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=llm), 0.50),
        (MultiHopAbstractQuerySynthesizer(llm=llm), 0.25),
        (MultiHopSpecificQuerySynthesizer(llm=llm), 0.25),
    ]

    with tracer.start_as_current_span("testset_generate") as span:
        span.add_event(f"Generating testset (size={testset_size})")
        golden_testset = generator.generate(
            testset_size=testset_size, query_distribution=query_distribution
        )

    with tracer.start_as_current_span("hf_conversion") as span:
        span.add_event("Converting testset to HF Dataset")
        hf_dataset = golden_testset.to_hf_dataset()

    with tracer.start_as_current_span("hf_push") as span:
        span.add_event(f"Pushing dataset to HF repo {hf_repo}")
        hf_dataset.push_to_hub(hf_repo)

    return golden_testset


def upload_to_phoenix(golden_testset, tracer, dataset_name: str = "johnwick_golden_testset") -> dict:
    """
    Upload the entire testset as a Phoenix dataset. On error, fallback to local JSON.
    """
    with tracer.start_as_current_span("phoenix_upload") as span:
        span.add_event("Converting RAGAS testset to DataFrame")
        testset_df = golden_testset.to_pandas()

        phoenix_df = pd.DataFrame(
            {
                "input": testset_df["user_input"],
                "output": testset_df["reference"],
                "contexts": testset_df["reference_contexts"].apply(
                    lambda x: str(x) if isinstance(x, list) else str(x)
                ),
                "synthesizer": testset_df["synthesizer_name"],
                "question_type": testset_df["synthesizer_name"],
                "dataset_source": "ragas_golden_testset",
            }
        )

        span.add_event(f"Uploading {len(phoenix_df)} samples to Phoenix")
        try:
            client = px.Client()
            dataset = client.upload_dataset(
                dataset_name=dataset_name,
                dataframe=phoenix_df,
                input_keys=["input"],
                output_keys=["output"],
                metadata_keys=["contexts", "synthesizer", "question_type", "dataset_source"],
            )
            span.add_event(f"Uploaded dataset {dataset}")
            return {
                "dataset_name": dataset_name,
                "num_samples": len(phoenix_df),
                "status": "success",
                "dataset": dataset,
            }

        except Exception as e:
            span.add_event(f"Error uploading to Phoenix: {e}")
            fallback_path = f"data/{dataset_name}_phoenix.json"
            os.makedirs(os.path.dirname(fallback_path), exist_ok=True)
            phoenix_df.to_json(fallback_path, orient="records", indent=2)
            span.add_event(f"Saved fallback JSON to {fallback_path}")
            return {
                "dataset_name": dataset_name,
                "num_samples": len(phoenix_df),
                "status": "fallback_saved",
                "file_path": fallback_path,
            }


def main():
    # ──────────────────────────────────────────────────────────────────────────────
    # 1. Environment & Tracer Setup
    # ──────────────────────────────────────────────────────────────────────────────
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
    os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY", "")
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")
    # These two are optional unless you plan to swap in Qdrant later
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
    QDRANT_API_URL = os.getenv("QDRANT_API_URL", "")

    project_name = f"golden-testset-generation-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tracer_provider = register(project_name=project_name, auto_instrument=True)
    tracer = tracer_provider.get_tracer(__name__)

    with tracer.start_as_current_span("script_start") as span:
        span.add_event("Starting golden_testset workflow")

    # ──────────────────────────────────────────────────────────────────────────────
    # 2. Initialize LLM + Embeddings
    # ──────────────────────────────────────────────────────────────────────────────
    with tracer.start_as_current_span("llm_setup") as span:
        span.add_event("Initializing LLM and Embeddings")
        llm = ChatOpenAI(model="gpt-4.1-mini")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        generator_llm = LangchainLLMWrapper(llm)
        generator_embeddings = LangchainEmbeddingsWrapper(embeddings)

    # ──────────────────────────────────────────────────────────────────────────────
    # 3. Data Ingestion
    # ──────────────────────────────────────────────────────────────────────────────
    with tracer.start_as_current_span("data_setup") as span:
        span.add_event("Creating data directory")
        DATA_DIR = Path.cwd() / "data"
        DATA_DIR.mkdir(exist_ok=True)

    urls = [
        ("https://raw.githubusercontent.com/AI-Maker-Space/DataRepository/main/jw1.csv", "john_wick_1.csv"),
        ("https://raw.githubusercontent.com/AI-Maker-Space/DataRepository/main/jw2.csv", "john_wick_2.csv"),
        ("https://raw.githubusercontent.com/AI-Maker-Space/DataRepository/main/jw3.csv", "john_wick_3.csv"),
        ("https://raw.githubusercontent.com/AI-Maker-Space/DataRepository/main/jw4.csv", "john_wick_4.csv"),
    ]
    download_csvs(DATA_DIR, urls, tracer)

    with tracer.start_as_current_span("doc_loading") as span:
        # span.add_event("Loading and annotating documents")
        # all_review_docs = load_and_annotate_docs(DATA_DIR, tracer)
        span.add_event("Loading and annotating documents from PostgreSQL")
        all_review_docs = load_docs_from_postgres()

    # ──────────────────────────────────────────────────────────────────────────────
    # 4. Knowledge‐Graph Construction & Transformation
    # ──────────────────────────────────────────────────────────────────────────────
    with tracer.start_as_current_span("kg_construction") as span:
        span.add_event("Building and transforming knowledge graph")
        ragas_kg = build_knowledge_graph(all_review_docs, generator_llm, generator_embeddings, tracer)

    # ──────────────────────────────────────────────────────────────────────────────
    # 5. Generate & Push Golden Testset
    # ──────────────────────────────────────────────────────────────────────────────
    HF_REPO = "dwb2023/johnwick_testset"
    with tracer.start_as_current_span("testset_section") as span:
        span.add_event(f"Generating and pushing testset to HF ({HF_REPO})")
        golden_testset = generate_and_push_testset(
            ragas_kg, generator_llm, generator_embeddings, tracer, HF_REPO
        )

    # ──────────────────────────────────────────────────────────────────────────────
    # 6. Upload to Phoenix (if desired)
    # ──────────────────────────────────────────────────────────────────────────────
    with tracer.start_as_current_span("phoenix_section") as span:
        span.add_event("Uploading testset to Phoenix")
        dataset_result = upload_to_phoenix(golden_testset, tracer, dataset_name="johnwick_golden_testset")

    with tracer.start_as_current_span("script_end") as span:
        span.add_event("Finished golden_testset workflow")
        span.add_event(f"Phoenix upload status: {dataset_result['status']}")

    print(f"🚀 Workflow completed. Phoenix upload status: {dataset_result['status']}")


if __name__ == "__main__":
    main()
