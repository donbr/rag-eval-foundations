# langchain_eval_golden_testset.py

import asyncio
import logging
from datetime import datetime

import pandas as pd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator

# Import shared configuration
from config import (
    BASELINE_TABLE,
    EMBEDDING_MODEL,
    GOLDEN_TESTSET_NAME,
    GOLDEN_TESTSET_SIZE,
    LLM_MODEL,
)
from data_loader import load_docs_from_postgres
from golden_testset.manager import GoldenTestsetManager

# Import our Phase 4 Phoenix integration
from golden_testset.phoenix_integration import PhoenixConfig, PhoenixIntegration
from langchain_eval_foundations_e2e import setup_environment

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_testset(docs: list, llm, embeddings, testset_size: int = 10):
    generator = TestsetGenerator(llm=llm, embedding_model=embeddings)

    golden_testset = generator.generate_with_langchain_docs(
        documents=docs, testset_size=testset_size
    )

    golden_testset_df = golden_testset.to_pandas()

    return golden_testset_df


async def upload_to_phoenix_integrated(
    golden_testset_df: pd.DataFrame,
    phoenix_integration: PhoenixIntegration,
    dataset_name: str = GOLDEN_TESTSET_NAME,
) -> dict:
    """Upload RAGAS golden testset to Phoenix using our Phase 4 integration."""

    logger.info(f"📤 Preparing {len(golden_testset_df)} examples for Phoenix upload")
    logger.info(f"📋 Available DataFrame columns: {list(golden_testset_df.columns)}")

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
                "synthesizer": "synthesizer_name",
            },
            "source_documents": len(golden_testset_df),
            "generation_model": LLM_MODEL,
        },
    }

    # Convert DataFrame rows to examples format
    valid_examples = 0
    for idx, row in golden_testset_df.iterrows():
        # Handle different possible column names from RAGAS
        user_input = (
            row.get("user_input")
            or row.get("question")
            or row.get("input", "")
        )
        reference = (
            row.get("reference")
            or row.get("ground_truth")
            or row.get("expected_output", "")
        )
        contexts = row.get("reference_contexts") or row.get("contexts", [])

        # Skip empty examples
        if not user_input or not reference:
            logger.warning(f"⚠️ Skipping example {idx}: missing input or reference")
            continue

        example = {
            "question": str(user_input),
            "ground_truth": str(reference),
            "contexts": contexts if isinstance(contexts, list) else [str(contexts)],
            "metadata": {
                "synthesizer_name": row.get("synthesizer_name", "unknown"),
                "evolution_type": row.get("evolution_type", "simple"),
                "source": "ragas_testset_generator",
                "generated_at": datetime.now().isoformat(),
            },
        }
        testset_data["examples"].append(example)
        valid_examples += 1

    logger.info(f"✅ Prepared {valid_examples}/{len(golden_testset_df)} valid examples")

    try:
        # Upload using our PhoenixIntegration.upload_external_testset method
        result = await phoenix_integration.upload_external_testset(
            testset_data, dataset_name
        )

        return {
            "dataset_name": result.get("dataset_name", dataset_name),
            "version": result.get("version", "unknown"),
            "num_samples": len(testset_data["examples"]),
            "status": "success",
            "phoenix_dataset_id": result.get("dataset_id", "unknown"),
            "upload_timestamp": result.get("created_at", datetime.now().isoformat()),
        }

    except Exception as e:
        return {
            "dataset_name": dataset_name,
            "num_samples": len(testset_data["examples"]),
            "status": "failed",
            "error": str(e),
            "upload_timestamp": datetime.now().isoformat(),
        }


async def main():
    """Main function with integrated Phoenix upload using Phase 4 architecture."""

    # Setup environment (loads API keys and sets env variables)
    setup_environment()

    # Use shared configuration constants
    llm = ChatOpenAI(model=LLM_MODEL)
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    generator_llm = LangchainLLMWrapper(llm)

    # RAGAS requires LangchainEmbeddingsWrapper (despite deprecation warning)
    generator_embeddings = LangchainEmbeddingsWrapper(embeddings)

    all_review_docs = load_docs_from_postgres(BASELINE_TABLE)
    logger.info(f"📊 Loaded {len(all_review_docs)} documents from database")

    # Use configurable testset size from shared config
    testset_size = GOLDEN_TESTSET_SIZE
    logger.info(f"🧪 Generating golden test set with {testset_size} examples")

    golden_testset_dataframe = generate_testset(
        all_review_docs, generator_llm, generator_embeddings, testset_size
    )

    # Validate DataFrame is not empty
    if golden_testset_dataframe.empty:
        logger.error("❌ Generated testset is empty! Cannot proceed.")
        return

    logger.info(f"✅ Generated testset with {len(golden_testset_dataframe)} examples")
    logger.info(f"📋 DataFrame columns: {list(golden_testset_dataframe.columns)}")
    logger.info(f"📊 DataFrame shape: {golden_testset_dataframe.shape}")

    # Show first example for debugging
    if len(golden_testset_dataframe) > 0:
        first_example = golden_testset_dataframe.iloc[0].to_dict()
        logger.info(f"📝 Sample example keys: {list(first_example.keys())}")

    # Save to JSON for backup/compatibility
    golden_testset_json = golden_testset_dataframe.to_json(orient="records", lines=True)

    if not golden_testset_json or golden_testset_json.strip() == "":
        logger.error("❌ JSON serialization produced empty output!")
        return

    with open("golden_testset.json", "w") as f:
        f.write(golden_testset_json)

    # Verify file was written
    import os
    file_size = os.path.getsize("golden_testset.json")
    logger.info(f"💾 Saved golden testset to golden_testset.json ({file_size} bytes)")

    # Initialize Phoenix integration with Phase 4 architecture
    print("🔥 Initializing Phoenix integration...")
    manager = GoldenTestsetManager()
    phoenix_config = PhoenixConfig()
    phoenix_integration = PhoenixIntegration(manager, phoenix_config)

    # Upload to Phoenix using our integrated approach
    print("🚀 Uploading golden testset to Phoenix...")
    dataset_result = await upload_to_phoenix_integrated(
        golden_testset_dataframe, phoenix_integration, dataset_name=GOLDEN_TESTSET_NAME
    )

    # Display results
    if dataset_result["status"] == "success":
        print("✅ Phoenix upload successful!")
        print(f"   📦 Dataset: {dataset_result['dataset_name']}")
        print(f"   🏷️  Version: {dataset_result['version']}")
        print(f"   📊 Samples: {dataset_result['num_samples']}")
        print(f"   🆔 Phoenix ID: {dataset_result['phoenix_dataset_id']}")
        print(f"   🕐 Timestamp: {dataset_result['upload_timestamp']}")
    else:
        print(f"❌ Phoenix upload failed: {dataset_result['error']}")
        print(f"   📊 Attempted samples: {dataset_result['num_samples']}")

    print("🎉 Workflow completed with integrated Phoenix upload!")


if __name__ == "__main__":
    asyncio.run(main())
