#!/usr/bin/env python3
import os
import asyncio
import time
from datetime import datetime, timedelta
from pathlib import Path
import logging
import sys
import argparse

import pandas as pd
import requests
from dotenv import load_dotenv
from sqlalchemy import create_engine

import phoenix as px
from phoenix.otel import register
# from phoenix.trace import SpanExporter # Ensure this or similar is available if needed for manual flushing or specific configs
from phoenix.experiments import run_experiment # Added for Phoenix Experiments

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
    # +++ Configuration for limiting records +++
    MAX_RECORDS_FROM_TESTSET = 5  # Set to 0 or a negative number to process all records
    # +++ End Configuration +++

    parser = argparse.ArgumentParser(description="Run RAG evaluation experiments with Phoenix.")
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="RGF0YXNldDox",
        help="The ID of the Phoenix dataset to use for evaluation."
    )
    parser.add_argument(
        "--dataset-version-id",
        type=str,
        default="RGF0YXNldFZlcnNpb246MQ==",
        help="The version ID of the Phoenix dataset."
    )
    args = parser.parse_args()

    # ──────────────────────────────────────────────────────────────────────────────
    # 1. Environment & Tracer Setup
    # ──────────────────────────────────────────────────────────────────────────────
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
    os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY", "")
    
    # Store original collector endpoint for OTel, then clear for px.Client initialization
    original_collector_endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT")
    px_collector_for_otel = "http://localhost:4317" # Define the OTel collector endpoint
    
    if "PHOENIX_COLLECTOR_ENDPOINT" in os.environ:
        del os.environ["PHOENIX_COLLECTOR_ENDPOINT"]

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Arize Phoenix Dataset Fetching
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    golden_testset_df = pd.DataFrame() # Initialize an empty DataFrame
    phoenix_eval_dataset = None # For the uploaded Phoenix Dataset object

    try:
        print("Attempting to connect to Phoenix server for dataset retrieval...")
        # Initialize client - hoping it defaults to http://localhost:6006 without PHOENIX_COLLECTOR_ENDPOINT set
        px_client = px.Client()
        
        # Restore/Set PHOENIX_COLLECTOR_ENDPOINT for OTel tracing before it's used by register()
        # This needs to happen before register() is called.
        # The placement of register() itself might need to be after this block if it was before.
        os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = px_collector_for_otel

        golden_dataset_id = args.dataset_id
        golden_dataset_version_id = args.dataset_version_id
        
        print(f"Fetching Phoenix dataset with id='{golden_dataset_id}' and version_id='{golden_dataset_version_id}'...")
        phoenix_dataset_object = px_client.get_dataset(id=golden_dataset_id, version_id=golden_dataset_version_id)
        print("Successfully fetched Phoenix dataset object.")

        # Convert to Pandas DataFrame
        if hasattr(phoenix_dataset_object, 'to_pandas'):
            golden_testset_df = phoenix_dataset_object.to_pandas()
            print("Converted dataset to Pandas DataFrame using .to_pandas().")
        elif hasattr(phoenix_dataset_object, 'as_dataframe'):
            golden_testset_df = phoenix_dataset_object.as_dataframe()
            print("Converted dataset to Pandas DataFrame using .as_dataframe().")
        elif hasattr(phoenix_dataset_object, 'df') and phoenix_dataset_object.df is not None:
            golden_testset_df = phoenix_dataset_object.df
            print("Accessed dataset via .df attribute.")
        else:
            # Fallback for other possible structures if needed, e.g. list of records
            try:
                records = list(phoenix_dataset_object)
                if records and isinstance(records[0], dict):
                    golden_testset_df = pd.DataFrame(records)
                    print("Converted iterable Phoenix dataset to DataFrame.")
                else:
                    raise ValueError("Dataset is iterable but not in a recognized format.")
            except TypeError:
                 raise NotImplementedError(
                    "Could not automatically extract data to a DataFrame from the fetched Phoenix dataset object. "
                    "Please inspect `phoenix_dataset_object` (e.g. type, dir()) to determine how to access its records."
                )

        if golden_testset_df.empty:
            raise ValueError("Phoenix dataset was fetched but resulted in an empty DataFrame.")

        # --- Verifying and Standardizing the Query Column ---
        query_column_name = None
        # Common possible names for the query column in datasets
        possible_query_columns = ['question', 'query', 'query_text', 'input_text', 'prompt', 'text', 'input']
        for col_name in possible_query_columns:
            if col_name in golden_testset_df.columns:
                query_column_name = col_name
                break
        
        if not query_column_name:
            raise ValueError(
                f"Could not automatically find a suitable query column in the fetched dataset. "
                f"Please check the column names. Expected one of {possible_query_columns}. "
                f"Available columns: {list(golden_testset_df.columns)}"
            )
        
        # Standardize the query column to 'question' for consistency with the rest of the script
        if query_column_name != 'question':
            golden_testset_df.rename(columns={query_column_name: 'question'}, inplace=True)
            print(f"Renamed dataset query column from '{query_column_name}' to 'question'.")
        # --- End Query Column Handling ---
        
        # +++ Limit number of records from testset if configured +++
        if MAX_RECORDS_FROM_TESTSET > 0 and not golden_testset_df.empty:
            original_count = len(golden_testset_df)
            print(f"Limiting golden testset from {original_count} to a maximum of {MAX_RECORDS_FROM_TESTSET} records.")
            golden_testset_df = golden_testset_df.head(MAX_RECORDS_FROM_TESTSET)
        # +++ End limit records +++
        
        print(f"Successfully loaded and processed golden testset. Number of records: {len(golden_testset_df)}.")
        print("Dataset preview (first 3 rows):")
        print(golden_testset_df.head(3))

        # Upload to Phoenix to create a formal Phoenix Dataset for experiments
        if not golden_testset_df.empty:
            print("Uploading processed dataset to Phoenix for experiment use...")
            # Ensure only relevant columns for the experiment are uploaded, typically input and reference (if any)
            # For now, we only have 'question' as input.
            # If you have expected outputs, include that column and specify in output_keys.
            columns_to_upload = ['question'] 
            if 'output' in golden_testset_df.columns: # Example if you had an expected output column
                 columns_to_upload.append('output')

            phoenix_eval_dataset = px_client.upload_dataset(
                dataframe=golden_testset_df[columns_to_upload], # Select only necessary columns
                input_keys=["question"],
                output_keys=["output"] if 'output' in columns_to_upload else [],
                dataset_name=f"JohnWick_GoldenQueries_Eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                # id=dataset_id, # Optional: re-use or specify ID if needed for versioning
                # version_id=dataset_version_id # Optional
            )
            print(f"Successfully uploaded dataset to Phoenix. Phoenix Dataset ID: {phoenix_eval_dataset.id}")

    except Exception as e:
        print(f"ERROR: Could not fetch or process the Phoenix dataset: {e}")
        print("The script will proceed with a default query if this was a critical failure for dataset loading.")
        # golden_testset_df remains empty, so the fallback logic later will be used.
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Ensure PHOENIX_COLLECTOR_ENDPOINT is set correctly before OTel registration
    # This was potentially problematic if original_collector_endpoint was None from the start.
    # Simplified: if it was something else, it's stored. If not, we use px_collector_for_otel.
    if original_collector_endpoint is not None:
        os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = original_collector_endpoint
    else:
        os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = px_collector_for_otel
        
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
    # 4. Data Ingestion (unchanged)
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
    # 5. Ingest into Vector Stores (unchanged)
    # ──────────────────────────────────────────────────────────────────────────────
    await baseline_vectorstore.aadd_documents(all_review_docs)

    semantic_chunker = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile"
    )
    semantic_docs = semantic_chunker.split_documents(all_review_docs)
    await semantic_vectorstore.aadd_documents(semantic_docs)

    # ──────────────────────────────────────────────────────────────────────────────
    # 6. Build Retriever Chains (unchanged)
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
    # 8. Execute & Compare All Strategies using Phoenix Experiments
    # ──────────────────────────────────────────────────────────────────────────────
    experiment_summary_records = []

    if phoenix_eval_dataset:
        print(f"\nStarting Phoenix Experiments using uploaded dataset ID: {phoenix_eval_dataset.id}")
        
        # Temporarily unset PHOENIX_COLLECTOR_ENDPOINT for the run_experiment loop
        # to ensure Phoenix API calls within run_experiment target the API server (e.g., 6006)
        # and not the OTel collector (e.g., 4317).
        env_for_exp_api_calls_original_otel_endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT")
        if "PHOENIX_COLLECTOR_ENDPOINT" in os.environ:
            del os.environ["PHOENIX_COLLECTOR_ENDPOINT"]
            print("Temporarily unset PHOENIX_COLLECTOR_ENDPOINT for run_experiment API calls.")

        for strategy_name, traced_callable in tracers.items():
            
            # Define the task function for this specific strategy
            def experiment_task_for_strategy(example_row: dict):
                query_data = example_row['question'] # This is potentially {'input': 'query string'} from the dataset
                actual_query_text: str
                if isinstance(query_data, dict) and 'input' in query_data:
                    actual_query_text = query_data['input']
                elif isinstance(query_data, str):
                    actual_query_text = query_data
                else:
                    # Fallback if the format is unexpected
                    print(f"Warning: Query data in experiment_task for strategy '{strategy_name}' is in an unexpected format: {query_data}. Using its string representation.")
                    actual_query_text = str(query_data)
                
                # The traced_callable already handles exceptions and returns a dict
                return traced_callable(question=actual_query_text) # Pass the extracted string

            experiment_display_name = f"RAG_EVAL_{strategy_name.upper()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            print(f"  Running experiment for strategy: '{strategy_name}' (Experiment Name: {experiment_display_name})...")
            
            try:
                # run_experiment will iterate through dataset examples and call experiment_task_for_strategy
                # The traces generated by @tracer.chain within traced_callable will be associated with this experiment
                experiment_object = run_experiment(
                    dataset=phoenix_eval_dataset,
                    task=experiment_task_for_strategy,
                    experiment_name=experiment_display_name,
                    # evaluators=[], # Add evaluators here if you define them
                )
                print(f"    ✅ Experiment '{experiment_display_name}' completed. Phoenix Experiment ID: {experiment_object.id}")
                experiment_summary_records.append({
                    "strategy": strategy_name,
                    "phoenix_experiment_id": experiment_object.id,
                    "phoenix_experiment_url": experiment_object.url,
                    "status": "SUCCESS"
                })
            except Exception as e:
                print(f"    ❌ ERROR running experiment for strategy '{strategy_name}': {e}")
                experiment_summary_records.append({
                    "strategy": strategy_name,
                    "phoenix_experiment_id": None,
                    "phoenix_experiment_url": None,
                    "status": "FAILURE",
                    "error_message": str(e)
                })
        
        # Restore PHOENIX_COLLECTOR_ENDPOINT after run_experiment loop for subsequent OTel operations
        if env_for_exp_api_calls_original_otel_endpoint is not None:
            os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = env_for_exp_api_calls_original_otel_endpoint
            print(f"Restored PHOENIX_COLLECTOR_ENDPOINT to: {env_for_exp_api_calls_original_otel_endpoint}")
        elif px_collector_for_otel: # Fallback if it was cleared and not set before
             os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = px_collector_for_otel
             print(f"Fallback: Set PHOENIX_COLLECTOR_ENDPOINT to: {px_collector_for_otel} for OTel flush")

    elif not golden_testset_df.empty:
        print("\nWARNING: Phoenix dataset upload failed or was skipped, but local golden testset dataframe is available.")
        print("Consider running a simplified local loop if Phoenix experiments are not critical or re-check upload issues.")
        # You could add a fallback to the old loop here if desired, processing golden_testset_df directly.
    else:
        print("\nWARNING: No golden testset available (neither Phoenix dataset nor local DataFrame). Skipping experiment execution.")
        # Fallback to a single default query to ensure script can trace something
        if not tracers: 
             print("No tracers defined, cannot run default query.")
        else:
            print("Falling back to a single default query for tracing check...")
            default_query = "Did people generally like John Wick?"
            default_strategy_name = list(tracers.keys())[0]
            default_tracer = tracers[default_strategy_name]
            try:
                print(f"  Running default query with '{default_strategy_name}' strategy...")
                result = default_tracer(question=default_query)
                print(f"    Default query result: {result.get('response', 'No response')}")
                if result.get("error"):
                    print(f"    Default query error: {result.get('error')}")
            except Exception as e:
                print(f"    ERROR running default query: {e}")

    # Print summary of Phoenix experiments
    if experiment_summary_records:
        df_experiment_summary = pd.DataFrame(experiment_summary_records)
        print("\n\n📊 Phoenix Experiment Run Summary:\n")
        print(df_experiment_summary.to_string(index=False))
        
        summary_filename = f"phoenix_experiment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_experiment_summary.to_csv(summary_filename, index=False)
        print(f"\nSaved Phoenix experiment summary to {summary_filename}")
    else:
        print("\nNo Phoenix experiment results were generated or recorded.")

    # ──────────────────────────────────────────────────────────────────────────────
    # 9. (Optional) Inspect pgvector table via pandas
    # ──────────────────────────────────────────────────────────────────────────────
    # sync_conn_str = async_url.replace("+asyncpg", "")
    # sync_engine = create_engine(sync_conn_str)
    # df_baseline = pd.read_sql_table("johnwick_baseline_documents", con=sync_engine)
    # print(df_baseline.head())

    # If original_collector_endpoint was None and we set it, 
    # it might be good practice to remove it if it wasn't there to begin with after OTel setup.
    # However, otel.register() will use the value at the time of its call.
    # For simplicity, we ensure it's set for OTel, and px.Client() initializes without it.
    if original_collector_endpoint is None:
        del os.environ["PHOENIX_COLLECTOR_ENDPOINT"]

    print("\nFlushing OTel traces...")
    if tracer_provider: # Ensure tracer_provider was initialized
        tracer_provider.force_flush()
        print("OTel traces flushed. Waiting a few seconds for export...")
        time.sleep(5) # Wait 5 seconds
        print("Done waiting.")
    else:
        print("Tracer provider not found, skipping flush.")

if __name__ == "__main__":
    asyncio.run(main())
