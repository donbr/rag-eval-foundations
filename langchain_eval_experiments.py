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

async def main():
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
    os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY", "")
    os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")

    # Connect to Phoenix and get the dataset (ONCE)
    px_client = px.Client()
    dataset = px_client.get_dataset(name="johnwick_golden_testset")
    
    print(f"📊 Dataset loaded: {dataset}")
    print(f"📊 Total examples: {len(list(dataset.examples))}")

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )
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

    POSTGRES_USER     = "langchain"
    POSTGRES_PASSWORD = "langchain"
    POSTGRES_HOST     = "localhost"
    POSTGRES_PORT     = "6024"
    POSTGRES_DB       = "langchain"
    TABLE_BASELINE    = "johnwick_baseline_documents"
    TABLE_SEMANTIC    = "johnwick_semantic_documents"

    async_url = (
        f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )
    pg_engine = PGEngine.from_connection_string(url=async_url)

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

    def make_chain(retriever):
        return (
            {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
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
        
        def create_task_function(strategy_chain, strategy):
            """Factory function to create task function for each strategy"""
            def task(example: Example) -> str:
                """
                CORRECT task function signature - takes Example object, returns string
                """
                try:
                    # map question to input key from dataset
                    question = example.input["input"]
                    
                    # Invoke the chain for this strategy
                    result = strategy_chain.invoke({"question": question})
                    
                    # Return the response content
                    return result["response"].content
                    
                except Exception as e:
                    print(f"❌ Error in {strategy} task: {e}")
                    return f"Error in {strategy}: {str(e)}"
            
            return task

        # Create task function for this strategy
        task_function = create_task_function(chain, strategy_name)
        
        # Run the experiment
        experiment_name = f"{strategy_name}_rag_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            experiment = run_experiment(
                dataset=dataset,
                task=task_function,
                experiment_name=experiment_name
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

    print(f"\n📊 Experiment Summary:")
    for result in experiment_results:
        if result["status"] == "SUCCESS":
            print(f"  ✅ {result['strategy']}: {result['experiment_id']}")
        else:
            print(f"  ❌ {result['strategy']}: {result['error']}")

    return experiment_results

if __name__ == "__main__":
    results = asyncio.run(main()) 