# langchain_eval_golden_testset.py

import os
import json
from datetime import datetime

import pandas as pd
from data_loader import load_docs_from_postgres
from langchain_eval_foundations_e2e import setup_environment

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

    golden_testset_df = golden_testset.to_pandas()

    return golden_testset_df


def upload_to_phoenix(golden_testset, dataset_name: str = "mixed_golden_testset") -> dict:
    testset_df = golden_testset.to_pandas()

    phoenix_df = pd.DataFrame(
        {
            "input": testset_df["user_input"],
            "output": testset_df["reference"],
            "contexts": testset_df["reference_contexts"].apply(
                lambda x: json.dumps(x) if isinstance(x, (list, dict)) else str(x)
            ),
            "synthesizer": testset_df["synthesizer_name"],
            "question_type": testset_df["synthesizer_name"],
            "dataset_source": "ragas_golden_testset",
        }
    )

    # Use timestamped dataset name for immutable snapshots
    px_dataset_name = f"{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

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

    # Setup configuration using the centralized config system
    config = setup_environment()

    llm = ChatOpenAI(model=config.model_name)
    embeddings = OpenAIEmbeddings(model=config.embedding_model)
    generator_llm = LangchainLLMWrapper(llm)
    generator_embeddings = LangchainEmbeddingsWrapper(embeddings)

    all_review_docs = load_docs_from_postgres(config.table_baseline)
    print(f"📊 Loaded {len(all_review_docs)} documents from database")

    # Use configurable testset size with environment variable override
    testset_size = int(os.getenv("GOLDEN_TESTSET_SIZE", config.golden_testset_size))
    print(f"🧪 Generating golden test set with {testset_size} examples")
    
    golden_testset_dataframe = generate_testset(
        all_review_docs, generator_llm, generator_embeddings, testset_size
    )

    golden_testset_json = golden_testset_dataframe.to_json(orient='records', lines=True)
    with open("golden_testset.json", "w") as f:
        f.write(golden_testset_json)

    # dataset_result = upload_to_phoenix(golden_testset, dataset_name="mixed_golden_testset")

    # print(f"🚀 Workflow completed. Phoenix upload status: {dataset_result['status']}")

if __name__ == "__main__":
    main()
