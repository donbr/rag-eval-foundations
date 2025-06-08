# langchain_eval_golden_testset.py

import os
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from data_loader import load_docs_from_postgres

import phoenix as px

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator

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
