"""Phoenix integration for versioned golden testset upload and management.

This module provides functionality to:
1. Upload versioned golden testsets to Phoenix
2. Track dataset versions in Phoenix
3. Support incremental updates and rollbacks
4. Provide observability for RAG evaluations
"""

import asyncio
import io
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import aiohttp
import pandas as pd
import phoenix as px
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
        self.phoenix_client = None

        if self.config.enable_tracing:
            self._setup_tracing()

        # Initialize Phoenix client for dataset uploads
        self._init_phoenix_client()

    def _init_phoenix_client(self):
        """Initialize Phoenix client for dataset operations."""
        try:
            # Initialize Phoenix client with endpoint
            self.phoenix_client = px.Client(endpoint=self.config.endpoint)
            logger.info(f"Phoenix client initialized: {self.config.endpoint}")
        except Exception as e:
            logger.warning(
                f"Failed to initialize Phoenix client: {e}. Will use HTTP fallback."
            )
            self.phoenix_client = None

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

    async def upload_external_testset(
        self, testset_data: dict[str, Any], dataset_name: str = "external_testset"
    ) -> dict[str, Any]:
        """Upload external testset data (e.g., from RAGAS) to Phoenix.

        Args:
            testset_data: External testset data with examples and metadata
            dataset_name: Name for the Phoenix dataset

        Returns:
            Upload result with dataset ID and metrics
        """
        span = None
        if self.tracer:
            span = self.tracer.start_span("upload_external_testset")

        try:
            # Generate version for external data
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            version_str = f"external_{timestamp}"

            # Prepare dataset for Phoenix
            dataset = self._prepare_external_phoenix_dataset(testset_data)

            if not dataset:
                raise ValueError("No valid examples found in testset data")

            # Upload to Phoenix
            result = await self._upload_to_phoenix(dataset, version_str)

            # Store minimal metadata for tracking
            await self._update_external_upload_metadata(
                dataset_name, version_str, result, testset_data
            )

            logger.info(
                f"Successfully uploaded external testset {dataset_name} "
                f"version {version_str} with {len(dataset)} examples"
            )

            return {
                "dataset_name": dataset_name,
                "version": version_str,
                "dataset_id": result["dataset_id"],
                "created_at": result["upload_timestamp"],
                "num_examples": len(dataset),
                "phoenix_url": result.get("dataset_url", ""),
                "status": "success",
            }

        except Exception as e:
            logger.error(f"Failed to upload external testset: {e}")
            raise

        finally:
            if span:
                span.end()

    def _validate_dataset(self, dataset: list[dict[str, Any]]) -> tuple[bool, str]:
        """Validate dataset before upload.

        Args:
            dataset: List of examples to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not dataset:
            return False, "Dataset is empty"

        if not isinstance(dataset, list):
            return False, "Dataset must be a list of examples"

        for i, example in enumerate(dataset):
            if not isinstance(example, dict):
                return False, f"Example {i} is not a dictionary"

            # Check for either old format (question/ground_truth) or new format
            # (input/reference)
            has_old_format = "question" in example or "ground_truth" in example
            has_new_format = "input" in example or "reference" in example

            if not (has_old_format or has_new_format):
                return (
                    False,
                    f"Example {i} missing required fields "
                    f"(input/reference or question/ground_truth)",
                )

            # Validate non-empty values
            input_val = example.get("input") or example.get("question")
            ref_val = example.get("reference") or example.get("ground_truth")

            if not input_val or not str(input_val).strip():
                return False, f"Example {i} has empty input/question"

            if not ref_val or not str(ref_val).strip():
                return False, f"Example {i} has empty reference/ground_truth"

        logger.info(f"✅ Dataset validation passed: {len(dataset)} examples")
        return True, ""

    def _prepare_external_phoenix_dataset(
        self, testset_data: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Prepare external testset data for Phoenix upload.

        Args:
            testset_data: External testset data with examples and metadata

        Returns:
            List of Phoenix-compatible examples
        """
        examples = []

        # Handle different input formats
        raw_examples = testset_data.get("examples", [])
        if not raw_examples:
            # Fallback for direct list format
            raw_examples = testset_data if isinstance(testset_data, list) else []

        for example in raw_examples:
            # Ensure required fields exist
            if not isinstance(example, dict):
                continue

            phoenix_example = {
                "input": example.get("question", example.get("input", "")),
                "reference": example.get("ground_truth", example.get("reference", "")),
                "contexts": example.get("contexts", []),
                "metadata": {
                    "source": "external_upload",
                    "synthesizer_name": example.get("metadata", {}).get(
                        "synthesizer_name", "unknown"
                    ),
                    "evolution_type": example.get("metadata", {}).get(
                        "evolution_type", "simple"
                    ),
                    "generated_at": example.get("metadata", {}).get(
                        "generated_at", datetime.utcnow().isoformat()
                    ),
                    **testset_data.get("metadata", {}),
                },
            }

            # Validate required fields
            if phoenix_example["input"] and phoenix_example["reference"]:
                examples.append(phoenix_example)

        return examples

    async def _update_external_upload_metadata(
        self, dataset_name: str, version: str, upload_result: dict, testset_data: dict
    ) -> None:
        """Store metadata for external upload tracking.

        Args:
            dataset_name: Name of the dataset
            version: Version string
            upload_result: Result from Phoenix upload
            testset_data: Original testset data
        """
        # This could store minimal tracking info in a separate table
        # For now, just log the upload
        metadata = {
            "dataset_name": dataset_name,
            "version": version,
            "upload_timestamp": upload_result.get("upload_timestamp"),
            "dataset_id": upload_result.get("dataset_id"),
            "num_examples": len(testset_data.get("examples", [])),
            "source": testset_data.get("metadata", {}).get("source", "unknown"),
        }

        logger.info(f"External upload metadata: {metadata}")

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

    async def _upload_to_phoenix_sdk(
        self, dataset: list[dict[str, Any]], dataset_name: str, version: str
    ) -> dict[str, Any]:
        """Upload dataset to Phoenix using SDK.

        Args:
            dataset: Prepared dataset examples
            dataset_name: Name for the dataset
            version: Version string

        Returns:
            Upload result with dataset ID and URL
        """
        if not self.phoenix_client:
            raise ValueError(
                "Phoenix client not initialized. Check endpoint configuration."
            )

        upload_timestamp = datetime.utcnow().isoformat()

        logger.info("📤 Preparing dataset for Phoenix SDK upload...")

        # Convert to pandas DataFrame (Phoenix SDK expects this format)
        df_data = []
        for example in dataset:
            df_data.append(
                {
                    "input": example.get("input", ""),
                    "reference": example.get("reference", ""),
                    "metadata": json.dumps(example.get("metadata", {})),
                }
            )

        df = pd.DataFrame(df_data)
        logger.info(f"📊 Created DataFrame with {len(df)} rows")

        # Upload using Phoenix SDK
        try:
            logger.info("🚀 Uploading to Phoenix via SDK...")
            result = self.phoenix_client.upload_dataset(
                dataset_name=dataset_name,
                dataframe=df,
                input_keys=["input"],
                output_keys=["reference"],
            )

            dataset_id = result.id if hasattr(result, "id") else str(result)
            logger.info(f"✅ Upload successful! Dataset ID: {dataset_id}")

            return {
                "dataset_id": dataset_id,
                "dataset_name": dataset_name,
                "version": version,
                "example_count": len(dataset),
                "upload_timestamp": upload_timestamp,
                "phoenix_url": f"{self.config.endpoint}/datasets/{dataset_id}",
            }

        except Exception as e:
            logger.error(f"❌ Phoenix SDK upload failed: {e}")
            raise

    async def _upload_to_phoenix(
        self, dataset: list[dict[str, Any]], version: str
    ) -> dict[str, Any]:
        """Upload dataset to Phoenix using SDK or HTTP fallback.

        Args:
            dataset: Prepared dataset examples
            version: Version string

        Returns:
            Upload result with dataset ID and URL
        """
        dataset_name = f"golden_testset_v{version.replace('.', '_')}"

        # Validate dataset before upload
        is_valid, error_msg = self._validate_dataset(dataset)
        if not is_valid:
            raise ValueError(f"Dataset validation failed: {error_msg}")

        # Try SDK upload first (preferred method)
        if self.phoenix_client:
            try:
                logger.info("Using Phoenix SDK for upload (recommended method)")
                return await self._upload_to_phoenix_sdk(dataset, dataset_name, version)
            except Exception as e:
                logger.warning(
                    f"Phoenix SDK upload failed: {e}. Trying HTTP fallback..."
                )

        # Fallback to HTTP upload
        return await self._upload_to_phoenix_http(dataset, dataset_name, version)

    async def _upload_to_phoenix_http(
        self, dataset: list[dict[str, Any]], dataset_name: str, version: str
    ) -> dict[str, Any]:
        """Upload dataset to Phoenix via HTTP API (fallback method).

        Args:
            dataset: Prepared dataset examples
            dataset_name: Name for the dataset
            version: Version string

        Returns:
            Upload result with dataset ID and URL
        """
        upload_timestamp = datetime.utcnow().isoformat()

        logger.info("📤 Preparing multipart upload to /v1/datasets/upload...")

        # Convert dataset to JSONL format for upload
        jsonl_data = []
        for example in dataset:
            jsonl_data.append(
                json.dumps(
                    {
                        "input": example.get("input", ""),
                        "output": example.get("reference", ""),
                        "metadata": example.get("metadata", {}),
                    }
                )
            )

        jsonl_content = "\n".join(jsonl_data)

        # Upload via HTTP API using multipart/form-data
        async with aiohttp.ClientSession() as session:
            url = f"{self.config.endpoint}/v1/datasets/upload"
            headers = self.config.get_headers()
            # Remove Content-Type to let aiohttp set it with boundary
            headers.pop("Content-Type", None)

            # Create form data
            form = aiohttp.FormData()
            form.add_field(
                "file",
                io.BytesIO(jsonl_content.encode("utf-8")),
                filename=f"{dataset_name}.jsonl",
                content_type="application/jsonl",
            )
            form.add_field("dataset_name", dataset_name)
            form.add_field("input_keys", '["input"]')
            form.add_field("output_keys", '["output"]')

            logger.info(f"🚀 Uploading {len(dataset)} examples to {url}...")

            async with session.post(
                url,
                data=form,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.config.upload_timeout),
            ) as response:
                if response.status not in [200, 201]:
                    error_text = await response.text()
                    raise Exception(
                        f"Phoenix HTTP upload failed: {response.status} - {error_text}"
                    )

                result = await response.json()
                dataset_id = result.get("id") or result.get("dataset_id", "unknown")

                logger.info(f"✅ HTTP upload successful! Dataset ID: {dataset_id}")

                return {
                    "dataset_id": dataset_id,
                    "dataset_name": dataset_name,
                    "version": version,
                    "example_count": len(dataset),
                    "upload_timestamp": upload_timestamp,
                    "phoenix_url": f"{self.config.endpoint}/datasets/{dataset_id}",
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
        self,
        model_name: str,
        name_pattern: str,
        provider: str,
        input_cost_per_million: float,
        output_cost_per_million: float,
    ) -> dict[str, Any]:
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
            "start_date": datetime.utcnow().isoformat(),
        }

        async with aiohttp.ClientSession() as session:
            url = f"{self.config.endpoint}/v1/model_pricing"
            headers = self.config.get_headers()

            async with session.post(url, json=payload, headers=headers) as response:
                if response.status not in [200, 201]:
                    error_text = await response.text()
                    raise Exception(
                        f"Failed to configure model pricing: "
                        f"{response.status} - {error_text}"
                    )

                result = await response.json()
                logger.info(
                    f"Configured pricing for {model_name}: "
                    f"${input_cost_per_million}/1M input, "
                    f"${output_cost_per_million}/1M output"
                )
                return result

    async def get_trace_costs(
        self, trace_ids: list[str]
    ) -> dict[str, dict[str, float]]:
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

        payload = {"query": query, "variables": {"traceIds": trace_ids}}

        async with aiohttp.ClientSession() as session:
            url = f"{self.config.endpoint}/graphql"
            headers = self.config.get_headers()

            async with session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(
                        f"Failed to get trace costs: {response.status} - {error_text}"
                    )

                result = await response.json()

                costs = {}
                for trace in result.get("data", {}).get("traces", []):
                    trace_id = trace["id"]
                    cost_summary = trace.get("costSummary", {})
                    costs[trace_id] = {
                        "prompt": float(cost_summary.get("prompt", 0)),
                        "completion": float(cost_summary.get("completion", 0)),
                        "total": float(cost_summary.get("total", 0)),
                    }

                return costs

    async def get_session_costs(self, session_id: str) -> dict[str, Any]:
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

        payload = {"query": query, "variables": {"sessionId": session_id}}

        async with aiohttp.ClientSession() as session:
            url = f"{self.config.endpoint}/graphql"
            headers = self.config.get_headers()

            async with session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(
                        f"Failed to get session costs: {response.status} - {error_text}"
                    )

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
                                    "total_tokens": 0,
                                }

                            model_breakdown[model_name]["cost"] += trace_cost
                            token_count = llm_attrs.get("tokenCount", {})
                            model_breakdown[model_name]["prompt_tokens"] += (
                                token_count.get("prompt", 0)
                            )
                            model_breakdown[model_name]["completion_tokens"] += (
                                token_count.get("completion", 0)
                            )
                            model_breakdown[model_name]["total_tokens"] += (
                                token_count.get("total", 0)
                            )

                return {
                    "session_id": session_id,
                    "total_cost": total_cost,
                    "prompt_cost": total_prompt_cost,
                    "completion_cost": total_completion_cost,
                    "model_breakdown": model_breakdown,
                    "trace_count": len(traces),
                }

    async def setup_default_model_pricing(self):
        """Configure default pricing for common models."""
        default_models = [
            {
                "model_name": "GPT-4.1 Mini",
                "name_pattern": "gpt-4.1-mini",
                "provider": "openai",
                "input_cost": 0.15,  # $0.15 per 1M tokens
                "output_cost": 0.60,  # $0.60 per 1M tokens
            },
            {
                "model_name": "Text Embedding 3 Small",
                "name_pattern": "text-embedding-3-small",
                "provider": "openai",
                "input_cost": 0.02,  # $0.02 per 1M tokens
                "output_cost": 0.0,  # No output cost for embeddings
            },
            {
                "model_name": "Cohere Rerank English v3",
                "name_pattern": "rerank-english-v3.0",
                "provider": "cohere",
                "input_cost": 0.20,  # $0.20 per 1M tokens
                "output_cost": 0.0,  # No output cost for reranking
            },
        ]

        results = []
        for model in default_models:
            try:
                result = await self.configure_model_pricing(
                    model_name=model["model_name"],
                    name_pattern=model["name_pattern"],
                    provider=model["provider"],
                    input_cost_per_million=model["input_cost"],
                    output_cost_per_million=model["output_cost"],
                )
                results.append(
                    {
                        "model": model["model_name"],
                        "status": "success",
                        "result": result,
                    }
                )
            except Exception as e:
                logger.error(
                    f"Failed to configure pricing for {model['model_name']}: {e}"
                )
                results.append(
                    {"model": model["model_name"], "status": "error", "error": str(e)}
                )

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
    parser.add_argument(
        "--setup-pricing", action="store_true", help="Configure default model pricing"
    )
    parser.add_argument("--get-costs", type=str, help="Get costs for session ID")
    parser.add_argument(
        "--trace-costs", nargs="+", help="Get costs for specific trace IDs"
    )

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
