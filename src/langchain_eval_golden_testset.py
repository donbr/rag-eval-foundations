# langchain_eval_golden_testset.py

import os
import json
import asyncio
from datetime import datetime

import pandas as pd
from data_loader import load_docs_from_postgres
from langchain_eval_foundations_e2e import setup_environment

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator

# Import our Phase 4 Phoenix integration
from golden_testset.phoenix_integration import PhoenixIntegration, PhoenixConfig
from golden_testset.manager import GoldenTestsetManager

# Import shared configuration
from config import GOLDEN_TESTSET_NAME, GOLDEN_TESTSET_SIZE, LLM_MODEL, EMBEDDING_MODEL

def generate_testset(
    docs: list, llm, embeddings, testset_size: int = 10
):

    generator = TestsetGenerator(llm=llm, embedding_model=embeddings)

    golden_testset = generator.generate_with_langchain_docs(
        documents=docs, testset_size=testset_size
    )

    golden_testset_df = golden_testset.to_pandas()

    return golden_testset_df


async def upload_to_phoenix_integrated(golden_testset_df: pd.DataFrame, phoenix_integration: PhoenixIntegration, dataset_name: str = GOLDEN_TESTSET_NAME) -> dict:
    """Upload RAGAS golden testset to Phoenix using our Phase 4 integration."""

    # Transform RAGAS DataFrame to the format expected by PhoenixIntegration
    testset_data = {
        "examples": [],
        "metadata": {
            "source": "ragas_golden_testset",
            "generation_method": "automated",
            "created_at": datetime.now().isoformat(),
            "num_samples": len(golden_testset_df),
            "data_types": {
                "questions": "user_input",
                "ground_truth": "reference",
                "contexts": "reference_contexts",
                "synthesizer": "synthesizer_name"
            },
            "source_documents": len(golden_testset_df),
            "generation_model": LLM_MODEL
        }
    }

    # Convert DataFrame rows to examples format
    for _, row in golden_testset_df.iterrows():
        example = {
            "question": row["user_input"],
            "ground_truth": row["reference"],
            "contexts": row["reference_contexts"] if isinstance(row["reference_contexts"], list) else [str(row["reference_contexts"])],
            "metadata": {
                "synthesizer_name": row.get("synthesizer_name", "unknown"),
                "evolution_type": row.get("evolution_type", "simple"),
                "source": "ragas_testset_generator",
                "generated_at": datetime.now().isoformat()
            }
        }
        testset_data["examples"].append(example)

    try:
        # Upload using our PhoenixIntegration.upload_external_testset method
        result = await phoenix_integration.upload_external_testset(testset_data, dataset_name)

        return {
            "dataset_name": result.get("dataset_name", dataset_name),
            "version": result.get("version", "unknown"),
            "num_samples": len(testset_data["examples"]),
            "status": "success",
            "phoenix_dataset_id": result.get("dataset_id", "unknown"),
            "upload_timestamp": result.get("created_at", datetime.now().isoformat())
        }

    except Exception as e:
        return {
            "dataset_name": dataset_name,
            "num_samples": len(testset_data["examples"]),
            "status": "failed",
            "error": str(e),
            "upload_timestamp": datetime.now().isoformat()
        }

async def main():
    """Main function with integrated Phoenix upload using Phase 4 architecture."""

    # Setup configuration using the centralized config system
    config = setup_environment()

    llm = ChatOpenAI(model=config.model_name)
    embeddings = OpenAIEmbeddings(model=config.embedding_model)
    generator_llm = LangchainLLMWrapper(llm)
    generator_embeddings = LangchainEmbeddingsWrapper(embeddings)

    all_review_docs = load_docs_from_postgres(config.table_baseline)
    print(f"📊 Loaded {len(all_review_docs)} documents from database")

    # Use configurable testset size from shared config
    testset_size = GOLDEN_TESTSET_SIZE
    print(f"🧪 Generating golden test set with {testset_size} examples")

    golden_testset_dataframe = generate_testset(
        all_review_docs, generator_llm, generator_embeddings, testset_size
    )

    # Save to JSON for backup/compatibility
    golden_testset_json = golden_testset_dataframe.to_json(orient='records', lines=True)
    with open("golden_testset.json", "w") as f:
        f.write(golden_testset_json)
    print(f"💾 Saved golden testset to golden_testset.json")

    # Initialize Phoenix integration with Phase 4 architecture
    print(f"🔥 Initializing Phoenix integration...")
    manager = GoldenTestsetManager()
    phoenix_config = PhoenixConfig()
    phoenix_integration = PhoenixIntegration(manager, phoenix_config)

    # Upload to Phoenix using our integrated approach
    print(f"🚀 Uploading golden testset to Phoenix...")
    dataset_result = await upload_to_phoenix_integrated(
        golden_testset_dataframe,
        phoenix_integration,
        dataset_name=GOLDEN_TESTSET_NAME
    )

    # Display results
    if dataset_result["status"] == "success":
        print(f"✅ Phoenix upload successful!")
        print(f"   📦 Dataset: {dataset_result['dataset_name']}")
        print(f"   🏷️  Version: {dataset_result['version']}")
        print(f"   📊 Samples: {dataset_result['num_samples']}")
        print(f"   🆔 Phoenix ID: {dataset_result['phoenix_dataset_id']}")
        print(f"   🕐 Timestamp: {dataset_result['upload_timestamp']}")
    else:
        print(f"❌ Phoenix upload failed: {dataset_result['error']}")
        print(f"   📊 Attempted samples: {dataset_result['num_samples']}")

    print(f"🎉 Workflow completed with integrated Phoenix upload!")

if __name__ == "__main__":
    asyncio.run(main())
