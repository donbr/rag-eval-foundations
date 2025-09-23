"""Phoenix integration for versioned golden testset upload and management.

This module provides functionality to:
1. Upload versioned golden testsets to Phoenix
2. Track dataset versions in Phoenix
3. Support incremental updates and rollbacks
4. Provide observability for RAG evaluations
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from decimal import Decimal

import aiohttp
from opentelemetry import trace as otel_trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode

from .manager import GoldenTestsetManager

logger = logging.getLogger(__name__)


@dataclass
class PhoenixConfig:
    """Configuration for Phoenix integration."""

    endpoint: str = os.getenv("PHOENIX_ENDPOINT", "http://localhost:6006")
    api_key: str | None = os.getenv("PHOENIX_API_KEY")
    client_headers: str | None = os.getenv("PHOENIX_CLIENT_HEADERS")
    otlp_endpoint: str = os.getenv("PHOENIX_OTLP_ENDPOINT", "http://localhost:4317")
    project_name: str = "golden-testset-management"
    upload_timeout: int = 30  # seconds
    batch_size: int = 100
    enable_tracing: bool = True

    def get_headers(self) -> dict[str, str]:
        """Get headers for Phoenix API requests."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.client_headers:
            try:
                additional = json.loads(self.client_headers)
                headers.update(additional)
            except json.JSONDecodeError:
                logger.warning("Invalid PHOENIX_CLIENT_HEADERS format")
        return headers


class PhoenixIntegration:
    """Manages Phoenix integration for golden testsets."""

    def __init__(
        self, manager: GoldenTestsetManager, config: PhoenixConfig | None = None
    ):
        """Initialize Phoenix integration.

        Args:
            manager: Golden testset manager instance
            config: Phoenix configuration
        """
        self.manager = manager
        self.config = config or PhoenixConfig()
        self.tracer = None

        if self.config.enable_tracing:
            self._setup_tracing()

    def _setup_tracing(self):
        """Setup OpenTelemetry tracing for Phoenix."""
        resource = Resource.create(
            {
                "service.name": self.config.project_name,
                "service.version": "1.0.0",
            }
        )

        provider = TracerProvider(resource=resource)

        # Configure OTLP exporter
        otlp_exporter = OTLPSpanExporter(
            endpoint=self.config.otlp_endpoint,
            insecure=True,
        )

        # Add batch processor
        span_processor = BatchSpanProcessor(otlp_exporter)
        provider.add_span_processor(span_processor)

        # Set global tracer provider
        otel_trace.set_tracer_provider(provider)
        self.tracer = otel_trace.get_tracer(__name__)

        logger.info(f"OpenTelemetry tracing configured: {self.config.otlp_endpoint}")

    async def upload_testset(
        self, version: str | None = None, dry_run: bool = False
    ) -> dict[str, Any]:
        """Upload golden testset to Phoenix.

        Args:
            version: Version to upload (defaults to active version)
            dry_run: If True, simulate upload without sending data

        Returns:
            Upload result with dataset ID and metrics
        """
        span = None
        if self.tracer:
            span = self.tracer.start_span("upload_golden_testset")

        try:
            # Get testset data
            if version:
                testset = await self.manager.get_version(version)
            else:
                testset = await self.manager.get_active_version()

            if not testset:
                raise ValueError("No testset found to upload")

            version_str = testset["version"]

            # Prepare dataset
            dataset = self._prepare_phoenix_dataset(testset)

            if dry_run:
                logger.info(
                    f"[DRY RUN] Would upload {len(dataset)} examples "
                    f"for version {version_str}"
                )
                return {
                    "dry_run": True,
                    "version": version_str,
                    "example_count": len(dataset),
                    "dataset_name": f"golden_testset_v{version_str}",
                }

            # Upload to Phoenix
            result = await self._upload_to_phoenix(dataset, version_str)

            # Update metadata
            await self._update_upload_metadata(version_str, result)

            if span:
                span.set_status(Status(StatusCode.OK))
                span.set_attribute("testset.version", version_str)
                span.set_attribute("testset.size", len(dataset))
                span.set_attribute("upload.success", True)

            logger.info(f"Successfully uploaded testset v{version_str} to Phoenix")
            return result

        except Exception as e:
            logger.error(f"Failed to upload testset: {e}")
            if span:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
            raise

        finally:
            if span:
                span.end()

    def _prepare_phoenix_dataset(self, testset: dict[str, Any]) -> list[dict[str, Any]]:
        """Prepare testset data for Phoenix upload.

        Args:
            testset: Raw testset data from database

        Returns:
            List of examples formatted for Phoenix
        """
        examples = []
        testset_data = testset.get("testset_data", {})

        # Extract examples from RAGAS format
        if "examples" in testset_data:
            for example in testset_data["examples"]:
                phoenix_example = {
                    "id": example.get("id", ""),
                    "question": example.get("question", ""),
                    "ground_truth": example.get("ground_truth", ""),
                    "contexts": example.get("contexts", []),
                    "metadata": {
                        "version": testset["version"],
                        "created_at": testset["created_at"].isoformat()
                        if testset.get("created_at")
                        else None,
                        "source_documents": example.get("metadata", {}).get(
                            "source_documents", []
                        ),
                        "generation_model": testset.get("metadata", {}).get(
                            "generation_model", "gpt-4.1-mini"
                        ),
                    },
                }

                # Add quality metrics if available
                if "quality_metrics" in testset:
                    metrics = testset["quality_metrics"]
                    phoenix_example["metadata"]["quality_score"] = metrics.get(
                        "overall_score", 0.0
                    )
                    phoenix_example["metadata"]["diversity_score"] = metrics.get(
                        "diversity_score", 0.0
                    )

                examples.append(phoenix_example)

        return examples

    async def _upload_to_phoenix(
        self, dataset: list[dict[str, Any]], version: str
    ) -> dict[str, Any]:
        """Upload dataset to Phoenix.

        Args:
            dataset: Prepared dataset examples
            version: Version string

        Returns:
            Upload result with dataset ID and URL
        """
        dataset_name = f"golden_testset_v{version.replace('.', '_')}"
        upload_timestamp = datetime.utcnow().isoformat()

        # Create Phoenix dataset payload
        payload = {
            "name": dataset_name,
            "description": f"Golden testset version {version}",
            "examples": dataset,
            "metadata": {
                "version": version,
                "uploaded_at": upload_timestamp,
                "example_count": len(dataset),
                "project": self.config.project_name,
            },
        }

        # Upload via HTTP API
        async with aiohttp.ClientSession() as session:
            url = f"{self.config.endpoint}/v1/datasets"
            headers = self.config.get_headers()

            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.config.upload_timeout),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(
                        f"Phoenix upload failed: {response.status} - {error_text}"
                    )

                result = await response.json()

                return {
                    "dataset_id": result.get("id"),
                    "dataset_name": dataset_name,
                    "version": version,
                    "example_count": len(dataset),
                    "upload_timestamp": upload_timestamp,
                    "phoenix_url": f"{self.config.endpoint}/datasets/{result.get('id')}",
                }

    async def _update_upload_metadata(
        self, version: str, upload_result: dict[str, Any]
    ):
        """Update testset metadata with Phoenix upload information.

        Args:
            version: Testset version
            upload_result: Result from Phoenix upload
        """
        metadata_update = {
            "phoenix_dataset_id": upload_result["dataset_id"],
            "phoenix_dataset_name": upload_result["dataset_name"],
            "phoenix_upload_timestamp": upload_result["upload_timestamp"],
            "phoenix_url": upload_result["phoenix_url"],
        }

        # Update metadata in database
        await self.manager.update_metadata(version, metadata_update)

    async def list_phoenix_datasets(self) -> list[dict[str, Any]]:
        """List all golden testset datasets in Phoenix.

        Returns:
            List of dataset information
        """
        async with aiohttp.ClientSession() as session:
            url = f"{self.config.endpoint}/v1/datasets"
            headers = self.config.get_headers()

            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(
                        f"Failed to list datasets: {response.status} - {error_text}"
                    )

                datasets = await response.json()

                # Filter for golden testset datasets
                golden_datasets = [
                    d
                    for d in datasets
                    if d.get("name", "").startswith("golden_testset_")
                ]

                return golden_datasets

    async def delete_phoenix_dataset(self, version: str, force: bool = False) -> bool:
        """Delete a golden testset dataset from Phoenix.

        Args:
            version: Version to delete
            force: Force deletion without confirmation

        Returns:
            True if deleted successfully
        """
        dataset_name = f"golden_testset_v{version.replace('.', '_')}"

        # Find dataset ID
        datasets = await self.list_phoenix_datasets()
        dataset = next((d for d in datasets if d["name"] == dataset_name), None)

        if not dataset:
            logger.warning(f"Dataset {dataset_name} not found in Phoenix")
            return False

        if not force:
            logger.info(f"Would delete dataset {dataset_name} (ID: {dataset['id']})")
            return False

        # Delete via API
        async with aiohttp.ClientSession() as session:
            url = f"{self.config.endpoint}/v1/datasets/{dataset['id']}"
            headers = self.config.get_headers()

            async with session.delete(url, headers=headers) as response:
                if response.status != 204:
                    error_text = await response.text()
                    raise Exception(
                        f"Failed to delete dataset: {response.status} - {error_text}"
                    )

                logger.info(f"Deleted dataset {dataset_name} from Phoenix")
                return True

    async def sync_all_versions(self, dry_run: bool = False) -> dict[str, Any]:
        """Sync all testset versions to Phoenix.

        Args:
            dry_run: If True, simulate sync without uploading

        Returns:
            Sync results for all versions
        """
        versions = await self.manager.list_versions()
        results = {}

        for version_info in versions:
            version = version_info["version"]
            try:
                result = await self.upload_testset(version=version, dry_run=dry_run)
                results[version] = {"status": "success", "result": result}
            except Exception as e:
                logger.error(f"Failed to sync version {version}: {e}")
                results[version] = {"status": "error", "error": str(e)}

        return results

    async def compare_versions_in_phoenix(self, v1: str, v2: str) -> dict[str, Any]:
        """Compare two testset versions in Phoenix.

        Args:
            v1: First version
            v2: Second version

        Returns:
            Comparison metrics and visualization URL
        """
        # Ensure both versions are uploaded
        await self.upload_testset(version=v1)
        await self.upload_testset(version=v2)

        # Get datasets
        datasets = await self.list_phoenix_datasets()
        ds1_name = f"golden_testset_v{v1.replace('.', '_')}"
        ds2_name = f"golden_testset_v{v2.replace('.', '_')}"

        ds1 = next((d for d in datasets if d["name"] == ds1_name), None)
        ds2 = next((d for d in datasets if d["name"] == ds2_name), None)

        if not ds1 or not ds2:
            raise ValueError("Could not find datasets for comparison")

        # Create comparison in Phoenix
        comparison_url = (
            f"{self.config.endpoint}/compare?dataset1={ds1['id']}&dataset2={ds2['id']}"
        )

        return {
            "v1": v1,
            "v2": v2,
            "dataset1_id": ds1["id"],
            "dataset2_id": ds2["id"],
            "comparison_url": comparison_url,
        }

    async def configure_model_pricing(
        self, model_name: str, name_pattern: str, provider: str,
        input_cost_per_million: float, output_cost_per_million: float
    ) -> Dict[str, Any]:
        """Configure model pricing in Phoenix.

        Args:
            model_name: Human-readable model name
            name_pattern: Regex pattern to match model name in traces
            provider: Model provider (e.g., 'openai', 'anthropic')
            input_cost_per_million: Cost per 1 million input tokens
            output_cost_per_million: Cost per 1 million output tokens

        Returns:
            Configuration result
        """
        payload = {
            "model_name": model_name,
            "name_pattern": name_pattern,
            "provider": provider,
            "input_cost_per_million": input_cost_per_million,
            "output_cost_per_million": output_cost_per_million,
            "start_date": datetime.utcnow().isoformat()
        }

        async with aiohttp.ClientSession() as session:
            url = f"{self.config.endpoint}/v1/model_pricing"
            headers = self.config.get_headers()

            async with session.post(url, json=payload, headers=headers) as response:
                if response.status not in [200, 201]:
                    error_text = await response.text()
                    raise Exception(f"Failed to configure model pricing: {response.status} - {error_text}")

                result = await response.json()
                logger.info(f"Configured pricing for {model_name}: ${input_cost_per_million}/1M input, ${output_cost_per_million}/1M output")
                return result

    async def get_trace_costs(self, trace_ids: List[str]) -> Dict[str, Dict[str, float]]:
        """Get cost summaries for specific traces using GraphQL.

        Args:
            trace_ids: List of trace IDs to get costs for

        Returns:
            Dictionary mapping trace_id to cost summary (prompt, completion, total)
        """
        query = """
        query GetTraceCosts($traceIds: [ID!]!) {
            traces(ids: $traceIds) {
                id
                costSummary {
                    prompt
                    completion
                    total
                }
            }
        }
        """

        payload = {
            "query": query,
            "variables": {"traceIds": trace_ids}
        }

        async with aiohttp.ClientSession() as session:
            url = f"{self.config.endpoint}/graphql"
            headers = self.config.get_headers()

            async with session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Failed to get trace costs: {response.status} - {error_text}")

                result = await response.json()

                costs = {}
                for trace in result.get("data", {}).get("traces", []):
                    trace_id = trace["id"]
                    cost_summary = trace.get("costSummary", {})
                    costs[trace_id] = {
                        "prompt": float(cost_summary.get("prompt", 0)),
                        "completion": float(cost_summary.get("completion", 0)),
                        "total": float(cost_summary.get("total", 0))
                    }

                return costs

    async def get_session_costs(self, session_id: str) -> Dict[str, Any]:
        """Get aggregated costs for a session.

        Args:
            session_id: Session identifier

        Returns:
            Session cost summary with breakdown by model
        """
        # Query traces for this session
        query = """
        query GetSessionTraces($sessionId: String!) {
            traces(filter: {sessionId: $sessionId}) {
                id
                spans {
                    attributes {
                        llm {
                            modelName
                            tokenCount {
                                prompt
                                completion
                                total
                            }
                        }
                    }
                }
                costSummary {
                    prompt
                    completion
                    total
                }
            }
        }
        """

        payload = {
            "query": query,
            "variables": {"sessionId": session_id}
        }

        async with aiohttp.ClientSession() as session:
            url = f"{self.config.endpoint}/graphql"
            headers = self.config.get_headers()

            async with session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Failed to get session costs: {response.status} - {error_text}")

                result = await response.json()

                traces = result.get("data", {}).get("traces", [])

                total_cost = 0.0
                total_prompt_cost = 0.0
                total_completion_cost = 0.0
                model_breakdown = {}

                for trace in traces:
                    cost_summary = trace.get("costSummary", {})
                    trace_cost = float(cost_summary.get("total", 0))
                    trace_prompt_cost = float(cost_summary.get("prompt", 0))
                    trace_completion_cost = float(cost_summary.get("completion", 0))

                    total_cost += trace_cost
                    total_prompt_cost += trace_prompt_cost
                    total_completion_cost += trace_completion_cost

                    # Extract model information from spans
                    for span in trace.get("spans", []):
                        llm_attrs = span.get("attributes", {}).get("llm", {})
                        model_name = llm_attrs.get("modelName")
                        if model_name:
                            if model_name not in model_breakdown:
                                model_breakdown[model_name] = {
                                    "cost": 0.0,
                                    "prompt_tokens": 0,
                                    "completion_tokens": 0,
                                    "total_tokens": 0
                                }

                            model_breakdown[model_name]["cost"] += trace_cost
                            token_count = llm_attrs.get("tokenCount", {})
                            model_breakdown[model_name]["prompt_tokens"] += token_count.get("prompt", 0)
                            model_breakdown[model_name]["completion_tokens"] += token_count.get("completion", 0)
                            model_breakdown[model_name]["total_tokens"] += token_count.get("total", 0)

                return {
                    "session_id": session_id,
                    "total_cost": total_cost,
                    "prompt_cost": total_prompt_cost,
                    "completion_cost": total_completion_cost,
                    "model_breakdown": model_breakdown,
                    "trace_count": len(traces)
                }

    async def setup_default_model_pricing(self):
        """Configure default pricing for common models."""
        default_models = [
            {
                "model_name": "GPT-4.1 Mini",
                "name_pattern": "gpt-4.1-mini",
                "provider": "openai",
                "input_cost": 0.15,  # $0.15 per 1M tokens
                "output_cost": 0.60   # $0.60 per 1M tokens
            },
            {
                "model_name": "Text Embedding 3 Small",
                "name_pattern": "text-embedding-3-small",
                "provider": "openai",
                "input_cost": 0.02,  # $0.02 per 1M tokens
                "output_cost": 0.0   # No output cost for embeddings
            },
            {
                "model_name": "Cohere Rerank English v3",
                "name_pattern": "rerank-english-v3.0",
                "provider": "cohere",
                "input_cost": 0.20,  # $0.20 per 1M tokens
                "output_cost": 0.0   # No output cost for reranking
            }
        ]

        results = []
        for model in default_models:
            try:
                result = await self.configure_model_pricing(
                    model_name=model["model_name"],
                    name_pattern=model["name_pattern"],
                    provider=model["provider"],
                    input_cost_per_million=model["input_cost"],
                    output_cost_per_million=model["output_cost"]
                )
                results.append({"model": model["model_name"], "status": "success", "result": result})
            except Exception as e:
                logger.error(f"Failed to configure pricing for {model['model_name']}: {e}")
                results.append({"model": model["model_name"], "status": "error", "error": str(e)})

        return results


async def main():
    """CLI entry point for Phoenix integration."""
    import argparse

    parser = argparse.ArgumentParser(description="Phoenix golden testset integration")
    parser.add_argument("--upload", action="store_true", help="Upload active testset")
    parser.add_argument("--version", type=str, help="Specific version to upload")
    parser.add_argument("--list", action="store_true", help="List Phoenix datasets")
    parser.add_argument("--sync-all", action="store_true", help="Sync all versions")
    parser.add_argument(
        "--compare", nargs=2, metavar=("V1", "V2"), help="Compare two versions"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Simulate actions without execution"
    )
    parser.add_argument("--setup-pricing", action="store_true", help="Configure default model pricing")
    parser.add_argument("--get-costs", type=str, help="Get costs for session ID")
    parser.add_argument("--trace-costs", nargs="+", help="Get costs for specific trace IDs")

    args = parser.parse_args()

    # Initialize
    from .transactions import init_database

    await init_database()
    manager = GoldenTestsetManager()
    phoenix = PhoenixIntegration(manager)

    try:
        if args.upload:
            result = await phoenix.upload_testset(
                version=args.version, dry_run=args.dry_run
            )
            print(json.dumps(result, indent=2))

        elif args.list:
            datasets = await phoenix.list_phoenix_datasets()
            for ds in datasets:
                print(f"- {ds['name']}: {ds.get('id', 'unknown')}")

        elif args.sync_all:
            results = await phoenix.sync_all_versions(dry_run=args.dry_run)
            print(json.dumps(results, indent=2))

        elif args.compare:
            result = await phoenix.compare_versions_in_phoenix(
                args.compare[0], args.compare[1]
            )
            print(json.dumps(result, indent=2))
            print(f"\nComparison URL: {result['comparison_url']}")

        elif args.setup_pricing:
            results = await phoenix.setup_default_model_pricing()
            print("Model pricing configuration results:")
            for result in results:
                status = "✅" if result["status"] == "success" else "❌"
                print(f"{status} {result['model']}: {result['status']}")

        elif args.get_costs:
            costs = await phoenix.get_session_costs(args.get_costs)
            print(json.dumps(costs, indent=2))

        elif args.trace_costs:
            costs = await phoenix.get_trace_costs(args.trace_costs)
            print(json.dumps(costs, indent=2))

        else:
            parser.print_help()

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(asyncio.run(main()))
