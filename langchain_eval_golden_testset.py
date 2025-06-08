# langchain_eval_golden_testset.py

import os
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from langchain_core.documents import Document

import phoenix as px

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator

def load_docs_from_postgres(
    table_name: str = "johnwick_baseline_documents",
) -> list[Document]:
    load_dotenv()

    POSTGRES_USER = os.getenv("POSTGRES_USER", "langchain")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "langchain")
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT", "6024")
    POSTGRES_DB = os.getenv("POSTGRES_DB", "langchain")

    sync_conn_str = (
        f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@"
        f"{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )

    engine = create_engine(sync_conn_str)

    try:
        with engine.connect() as connection:
            query = text(f'SELECT content, langchain_metadata FROM "{table_name}"')
            df = pd.read_sql_query(query, connection)
    except Exception as e:
        print(f"Error executing query: {e}")
        print("Please ensure the table name is correct and that the 'langchain-eval-foundations-e2e.py' script has run successfully.")
        return []

    documents = [
        Document(
            page_content=row["content"],
            metadata=row["langchain_metadata"],
        )
        for _, row in df.iterrows()
    ]

    print(f"Successfully loaded {len(documents)} documents from the '{table_name}' table.")
    return documents

def generate_testset(
    docs: list, llm, embeddings, testset_size: int = 10
):

    generator = TestsetGenerator(llm=llm, embedding_model=embeddings)

    golden_testset = generator.generate_with_langchain_docs(
        documents=docs, testset_size=testset_size
    )

    return golden_testset


def upload_to_phoenix(golden_testset, dataset_name: str = "johnwick_golden_testset") -> dict:
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

    px_dataset_name = dataset_name
    # px_dataset_name = f"{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    client = px.Client()
    dataset = client.upload_dataset(
        dataset_name=px_dataset_name,
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

def main():

    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
    os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY", "")

    llm = ChatOpenAI(model="gpt-4.1-mini")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    generator_llm = LangchainLLMWrapper(llm)
    generator_embeddings = LangchainEmbeddingsWrapper(embeddings)

    all_review_docs = load_docs_from_postgres("johnwick_baseline_documents")

    golden_testset = generate_testset(
        all_review_docs, generator_llm, generator_embeddings, 3
    )

    dataset_result = upload_to_phoenix(golden_testset, dataset_name="johnwick_golden_testset")

    print(f"🚀 Workflow completed. Phoenix upload status: {dataset_result['status']}")

if __name__ == "__main__":
    main()
