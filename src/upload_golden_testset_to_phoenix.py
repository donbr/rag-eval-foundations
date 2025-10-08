#!/usr/bin/env python3
"""
Upload existing golden_testset.json to Phoenix using Phase 4 integration.

This script loads the golden_testset.json file that was already generated
and uploads it to Phoenix using our PhoenixIntegration class.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

# Import shared configuration
from config import GOLDEN_TESTSET_NAME, LLM_MODEL
from golden_testset.manager import GoldenTestsetManager
from golden_testset.phoenix_integration import PhoenixConfig, PhoenixIntegration


async def upload_existing_golden_testset(
    json_file_path: str = "./golden_testset.json",
    dataset_name: str = GOLDEN_TESTSET_NAME,
) -> dict:
    """Upload existing golden testset JSON to Phoenix."""

    print(f"📁 Loading golden testset from {json_file_path}...")

    # Check if file exists
    json_path = Path(json_file_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Golden testset file not found: {json_file_path}")

    # Get file size for progress info
    file_size = json_path.stat().st_size
    print(f"   File size: {file_size:,} bytes")

    # Load the JSON file
    with open(json_path) as f:
        json_content = f.read()

    # Parse JSONL format (each line is a JSON object)
    golden_testset_records = []
    print("📋 Parsing JSONL format...")
    for line_num, line in enumerate(json_content.strip().split("\n"), 1):
        if line.strip():
            try:
                record = json.loads(line)
                golden_testset_records.append(record)
            except json.JSONDecodeError as e:
                print(f"   ⚠️  Warning: Skipping malformed line {line_num}: {e}")
                continue

    print(
        f"✅ Successfully parsed {len(golden_testset_records)} examples from JSON file"
    )

    # Validate data quality
    print("\n🔍 Validating data quality...")
    valid_count = 0
    for i, record in enumerate(golden_testset_records):
        has_input = bool(record.get("user_input", "").strip())
        has_reference = bool(record.get("reference", "").strip())

        if has_input and has_reference:
            valid_count += 1
        else:
            missing_field = "input" if not has_input else "reference"
            print(f"   ⚠️  Example {i + 1}: Missing {missing_field}")

    total = len(golden_testset_records)
    print(
        f"✅ Validation complete: {valid_count}/{total} examples "
        "have required fields"
    )

    # Transform to the format expected by PhoenixIntegration
    testset_data = {
        "examples": [],
        "metadata": {
            "source": "ragas_golden_testset",
            "generation_method": "automated",
            "created_at": datetime.now().isoformat(),
            "num_samples": len(golden_testset_records),
            "data_types": {
                "questions": "user_input",
                "ground_truth": "reference",
                "contexts": "reference_contexts",
                "synthesizer": "synthesizer_name",
            },
            "source_documents": len(golden_testset_records),
            "generation_model": LLM_MODEL,
        },
    }

    # Convert each record to Phoenix format
    for record in golden_testset_records:
        example = {
            "question": record["user_input"],
            "ground_truth": record["reference"],
            "contexts": record["reference_contexts"]
            if isinstance(record["reference_contexts"], list)
            else [str(record["reference_contexts"])],
            "metadata": {
                "synthesizer_name": record.get("synthesizer_name", "unknown"),
                "evolution_type": record.get("evolution_type", "simple"),
                "source": "ragas_testset_generator",
                "generated_at": datetime.now().isoformat(),
            },
        }
        testset_data["examples"].append(example)

    print(f"🔄 Transformed {len(testset_data['examples'])} examples to Phoenix format")

    # Initialize Phoenix integration
    print("\n🔥 Initializing Phoenix integration...")
    manager = GoldenTestsetManager()
    phoenix_config = PhoenixConfig()
    print(f"   Phoenix endpoint: {phoenix_config.endpoint}")
    print(f"   OTLP endpoint: {phoenix_config.otlp_endpoint}")
    phoenix_integration = PhoenixIntegration(manager, phoenix_config)

    # Upload to Phoenix using the new external upload method
    print("\n🚀 Uploading golden testset to Phoenix...")
    print(f"   Dataset name: {dataset_name}")
    print(f"   Examples to upload: {len(testset_data['examples'])}")
    try:
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
    """Main function to upload existing golden testset to Phoenix."""

    print("🎯 Phoenix Upload Tool - Using Existing Golden Testset")
    print("=" * 60)

    # Upload the existing golden_testset.json file
    result = await upload_existing_golden_testset()

    # Display results
    print("\n" + "=" * 60)
    if result["status"] == "success":
        print("✅ Phoenix upload successful!")
        print("\n📊 Upload Summary:")
        print(f"   📦 Dataset Name: {result['dataset_name']}")
        print(f"   🏷️  Version: {result['version']}")
        print(f"   📊 Samples Uploaded: {result['num_samples']}")
        print(f"   🆔 Phoenix Dataset ID: {result['phoenix_dataset_id']}")
        print(f"   🕐 Timestamp: {result['upload_timestamp']}")
        print("\n🔗 View Dataset:")
        print("   Phoenix UI: http://localhost:6006")
        print(
            f"   Direct Link: http://localhost:6006/datasets/{result['phoenix_dataset_id']}"
        )
        print("\n✨ Next Steps:")
        print("   - Run experiments using this dataset")
        print("   - View evaluation metrics in Phoenix UI")
        print("   - Compare retrieval strategies")
    else:
        print("❌ Phoenix upload failed!")
        print("\n❗ Error Details:")
        print(f"   {result['error']}")
        print("\n📊 Attempt Summary:")
        print(f"   Dataset: {result['dataset_name']}")
        print(f"   Attempted samples: {result['num_samples']}")
        print(f"   Timestamp: {result['upload_timestamp']}")
        print("\n🔧 Troubleshooting:")
        print("   - Check Phoenix is running: docker ps | grep phoenix")
        print("   - Verify endpoint: curl http://localhost:6006/health")
        print("   - Check logs: docker logs rag-eval-phoenix")

    print("\n" + "=" * 60)
    print("🎉 Upload process completed!")


if __name__ == "__main__":
    asyncio.run(main())
