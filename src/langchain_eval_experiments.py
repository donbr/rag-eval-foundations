import asyncio
import os
from datetime import datetime
from operator import itemgetter

import phoenix as px
from dotenv import load_dotenv
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_cohere import CohereRerank
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGEngine, PGVectorStore
from phoenix.evals import OpenAIModel, QAEvaluator, RelevanceEvaluator
from phoenix.experiments import run_experiment
from phoenix.experiments.evaluators import create_evaluator
from phoenix.experiments.types import Example

# Import shared configuration
from config import (
    BASELINE_TABLE,
    COHERE_RERANK_MODEL,
    EMBEDDING_MODEL,
    GOLDEN_TESTSET_NAME,
    LLM_MODEL,
    PHOENIX_ENDPOINT,
    SEMANTIC_TABLE,
    get_postgres_async_url,
)


# QA Correctness Evaluator (from Phoenix official docs)
@create_evaluator(name="qa_correctness_score")
def qa_correctness_evaluator(output, reference, input):
    """
    Evaluates answer correctness against ground truth
    Based on Phoenix official documentation:
    https://arize.com/docs/phoenix/evaluation/evals
    """
    try:
        # Using approved model for Phoenix evaluation
        eval_model = OpenAIModel(model="gpt-4.1-mini")

        # Create QA evaluator with the model (official Phoenix pattern)
        evaluator = QAEvaluator(eval_model)

        # The dataframe columns expected by Phoenix QAEvaluator are:
        # 'output', 'input', 'reference' (from official docs)
        import pandas as pd

        eval_df = pd.DataFrame(
            [{"output": output, "input": input, "reference": reference}]
        )

        # Use run_evals as shown in official docs
        from phoenix.evals import run_evals

        result_df = run_evals(dataframe=eval_df, evaluators=[evaluator])[
            0
        ]  # QA evaluator is first in list

        # Extract score from result
        return float(result_df.iloc[0]["score"]) if len(result_df) > 0 else 0.0

    except Exception as e:
        print(f"QA Evaluation error: {e}")
        return 0.0


@create_evaluator(name="rag_relevance_score")
def rag_relevance_evaluator(output, input, metadata):
    """
    Evaluates whether retrieved context is relevant to the query
    Based on Phoenix RAG Relevance documentation
    """
    try:
        eval_model = OpenAIModel(model="gpt-4.1-mini")
        evaluator = RelevanceEvaluator(eval_model)

        # Get retrieved context from metadata
        retrieved_context = metadata.get("retrieved_context", "")
        if isinstance(retrieved_context, list):
            retrieved_context = " ".join(str(doc) for doc in retrieved_context)

        import pandas as pd

        eval_df = pd.DataFrame([{"input": input, "reference": str(retrieved_context)}])

        from phoenix.evals import run_evals

        result_df = run_evals(dataframe=eval_df, evaluators=[evaluator])[0]

        return float(result_df.iloc[0]["score"]) if len(result_df) > 0 else 0.0

    except Exception as e:
        print(f"RAG Relevance evaluation error: {e}")
        return 0.0


# Updated experiment execution for your main() function:
# This captures retrieval context needed for RAG relevance evaluation
def create_enhanced_task_function(strategy_chain, strategy):
    def task(example: Example) -> dict:
        """
        Modified to return dict with metadata for Phoenix evaluators
        """
        question = example.input["input"]
        result = strategy_chain.invoke({"question": question})

        return {
            "output": result["response"].content,
            "metadata": {
                "retrieved_context": result.get("context", []),
                "strategy": strategy,
            },
        }

    return task


async def main():
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
    os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY", "")
    os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")

    # Connect to Phoenix and get the dataset
    print(f"🔗 Connecting to Phoenix at: {PHOENIX_ENDPOINT}")
    px_client = px.Client(endpoint=PHOENIX_ENDPOINT)

    # Query for the golden testset dataset (handle versioned naming)
    print(f"🔍 Looking for dataset: {GOLDEN_TESTSET_NAME}")

    # Use HTTP API to list all datasets (Phoenix SDK doesn't have list_datasets)
    import requests

    datasets_response = requests.get(f"{PHOENIX_ENDPOINT}/v1/datasets")
    datasets_response.raise_for_status()
    datasets_data = datasets_response.json()["data"]

    print(f"📋 Found {len(datasets_data)} total datasets")

    # Search for matching dataset
    matching_datasets = [
        d for d in datasets_data if "golden_testset" in d["name"].lower()
    ]

    if not matching_datasets:
        available_names = [d["name"] for d in datasets_data]
        raise ValueError(
            f"No golden testset dataset found. Available datasets: {available_names}"
        )

    # Use the most recent one
    dataset_info = matching_datasets[-1]
    dataset_name = dataset_info["name"]
    print(f"✅ Using dataset: {dataset_name}")

    # Get the dataset using Phoenix SDK
    dataset = px_client.get_dataset(name=dataset_name)

    print(f"📊 Dataset loaded: {dataset_name}")
    print(f"📊 Dataset ID: {dataset.id}")
    print(f"📊 Total examples: {len(list(dataset.examples))}")

    llm = ChatOpenAI(
        model=LLM_MODEL, temperature=0, max_tokens=None, timeout=None, max_retries=2
    )
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    RAG_TEMPLATE = """\
You are a helpful and kind assistant. Use the context provided below to answer the question.

If you do not know the answer, or are unsure, say you don't know.

Query:
{question}

Context:
{context}
"""
    rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

    # Use shared configuration
    async_url = get_postgres_async_url()
    pg_engine = PGEngine.from_connection_string(url=async_url)

    baseline_vectorstore = await PGVectorStore.create(
        engine=pg_engine,
        table_name=BASELINE_TABLE,
        embedding_service=embeddings,
    )
    semantic_vectorstore = await PGVectorStore.create(
        engine=pg_engine,
        table_name=SEMANTIC_TABLE,
        embedding_service=embeddings,
    )

    naive_retriever = baseline_vectorstore.as_retriever(search_kwargs={"k": 10})
    cohere_rerank = CohereRerank(model=COHERE_RERANK_MODEL)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=cohere_rerank, base_retriever=naive_retriever
    )
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=naive_retriever, llm=llm
    )
    ensemble_retriever = EnsembleRetriever(
        retrievers=[naive_retriever, compression_retriever, multi_query_retriever],
        weights=[0.34, 0.33, 0.33],
    )
    semantic_retriever = semantic_vectorstore.as_retriever(search_kwargs={"k": 10})

    def make_chain(retriever):
        return (
            {
                "context": itemgetter("question") | retriever,
                "question": itemgetter("question"),
            }
            | RunnablePassthrough.assign(context=itemgetter("context"))
            | {"response": rag_prompt | llm, "context": itemgetter("context")}
        )

    chains = {
        "naive": make_chain(naive_retriever),
        "compression": make_chain(compression_retriever),
        "multiquery": make_chain(multi_query_retriever),
        "ensemble": make_chain(ensemble_retriever),
        "semantic": make_chain(semantic_retriever),
    }

    experiment_results = []

    for strategy_name, chain in chains.items():
        print(f"\n🧪 Running experiment for strategy: {strategy_name}")

        # Run the experiment
        experiment_name = (
            f"{strategy_name}_rag_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        try:
            experiment = run_experiment(
                dataset=dataset,
                task=create_enhanced_task_function(chain, strategy_name),
                evaluators=[qa_correctness_evaluator, rag_relevance_evaluator],
                experiment_name=experiment_name,
                experiment_description=f"QA correctness and RAG relevance evaluation for {strategy_name}",
            )

            print(f"✅ {strategy_name} experiment completed!")
            print(f"📈 Experiment ID: {experiment.id}")

            experiment_results.append(
                {
                    "strategy": strategy_name,
                    "experiment_id": experiment.id,
                    "status": "SUCCESS",
                }
            )

        except Exception as e:
            print(f"❌ Error running {strategy_name} experiment: {e}")
            experiment_results.append(
                {"strategy": strategy_name, "error": str(e), "status": "FAILED"}
            )

    print("\n📊 Experiment Summary:")
    for result in experiment_results:
        if result["status"] == "SUCCESS":
            print(f"  ✅ {result['strategy']}: {result['experiment_id']}")
        else:
            print(f"  ❌ {result['strategy']}: {result['error']}")

    return experiment_results


if __name__ == "__main__":
    results = asyncio.run(main())
