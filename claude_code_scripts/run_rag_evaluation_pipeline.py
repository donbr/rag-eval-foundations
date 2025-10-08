#!/usr/bin/env python3
"""
RAG Evaluation Pipeline Orchestrator

This script provides a comprehensive, repeatable process for executing the complete
RAG evaluation pipeline. It orchestrates the three main components in the correct order:

1. Main E2E Pipeline: Data loading, vector store setup, and retrieval strategy testing
2. Golden Test Set Generation: RAGAS-based evaluation question generation
3. Automated Experiments: Systematic evaluation using Phoenix framework

The script includes proper dependency checking, service management, error handling,
and detailed logging to ensure a smooth and repeatable evaluation process.

Usage:
    python claude_code_scripts/run_rag_evaluation_pipeline.py [--skip-services] [--verbose]

Prerequisites:
    - .env file with OPENAI_API_KEY and COHERE_API_KEY
    - Docker installed and running
    - Python virtual environment with dependencies installed

Author: Claude Code
Date: 2025-06-23
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def setup_logging(verbose: bool = False) -> logging.Logger:
    """
    Configure logging with timestamps and appropriate verbosity.

    Args:
        verbose: If True, set DEBUG level; otherwise INFO level

    Returns:
        Configured logger instance
    """
    log_level = logging.DEBUG if verbose else logging.INFO

    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Configure logging format
    log_format = "%(asctime)s - %(levelname)s - %(message)s"

    # Setup file and console handlers
    log_filename = (
        log_dir / f"rag_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler(sys.stdout)],
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_filename}")
    return logger


def check_environment() -> tuple[bool, list[str]]:
    """
    Validate the environment setup including .env file and required API keys.

    Returns:
        Tuple of (success: bool, errors: List[str])
    """
    errors = []

    # Check for .env file
    env_file = Path(".env")
    if not env_file.exists():
        errors.append(".env file not found. Copy .env.example and add your API keys.")
        return False, errors

    # Load and check required environment variables
    try:
        from dotenv import load_dotenv

        load_dotenv()

        required_vars = ["OPENAI_API_KEY", "COHERE_API_KEY"]
        for var in required_vars:
            value = os.getenv(var)
            if not value or value.strip() == "":
                errors.append(f"Missing or empty environment variable: {var}")
            elif var == "OPENAI_API_KEY" and not value.startswith("sk-"):
                errors.append(f"Invalid OpenAI API key format: {var}")

    except ImportError:
        errors.append("python-dotenv not installed. Run: uv sync")
    except Exception as e:
        errors.append(f"Error loading .env file: {str(e)}")

    # Check for required directories
    required_dirs = ["src", "data"]
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            errors.append(f"Required directory not found: {dir_name}")

    # Check for required source files
    required_files = [
        "src/langchain_eval_foundations_e2e.py",
        "src/langchain_eval_golden_testset.py",
        "src/langchain_eval_experiments.py",
    ]
    for file_path in required_files:
        if not Path(file_path).exists():
            errors.append(f"Required source file not found: {file_path}")

    return len(errors) == 0, errors


def check_docker_availability() -> tuple[bool, str]:
    """
    Check if Docker is installed and running.

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        result = subprocess.run(
            ["docker", "--version"], capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return False, "Docker is not installed or not in PATH"

        # Test Docker daemon connectivity
        result = subprocess.run(
            ["docker", "ps"], capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return False, "Docker daemon is not running. Start Docker and try again."

        return True, "Docker is available and running"

    except subprocess.TimeoutExpired:
        return False, "Docker command timed out"
    except FileNotFoundError:
        return False, "Docker is not installed"
    except Exception as e:
        return False, f"Error checking Docker: {str(e)}"


def run_service_check() -> tuple[bool, str]:
    """
    Run the service check script to verify container status.

    Returns:
        Tuple of (success: bool, output: str)
    """
    try:
        result = subprocess.run(
            ["./claude_code_scripts/check-services.sh"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return True, result.stdout
    except subprocess.TimeoutExpired:
        return False, "Service check timed out"
    except FileNotFoundError:
        return False, "claude_code_scripts/check-services.sh script not found"
    except Exception as e:
        return False, f"Error running service check: {str(e)}"


def start_docker_services(logger: logging.Logger) -> bool:
    """
    Start the required Docker services (PostgreSQL and Phoenix).

    Args:
        logger: Logger instance for output

    Returns:
        True if services started successfully, False otherwise
    """
    logger.info("🐳 Starting Docker services...")

    try:
        # Start services
        result = subprocess.run(
            ["docker-compose", "up", "-d"], capture_output=True, text=True, timeout=120
        )

        if result.returncode != 0:
            logger.error(f"Failed to start Docker services: {result.stderr}")
            return False

        logger.info("Docker services started. Waiting for health checks...")

        # Wait for services to be healthy
        max_retries = 30
        retry_count = 0

        while retry_count < max_retries:
            success, output = run_service_check()
            if success and "All services are running!" in output:
                logger.info("✅ All services are healthy and running")
                return True

            retry_count += 1
            time.sleep(2)
            logger.debug(f"Waiting for services... ({retry_count}/{max_retries})")

        logger.error("❌ Services failed to become healthy within timeout")
        return False

    except subprocess.TimeoutExpired:
        logger.error("Docker service startup timed out")
        return False
    except Exception as e:
        logger.error(f"Error starting Docker services: {str(e)}")
        return False


def execute_pipeline_step(
    step_name: str, script_path: str, logger: logging.Logger, timeout: int = 600
) -> tuple[bool, str]:
    """
    Execute a single pipeline step with proper error handling and logging.

    Args:
        step_name: Human-readable name of the step
        script_path: Path to the Python script to execute
        logger: Logger instance for output
        timeout: Maximum execution time in seconds

    Returns:
        Tuple of (success: bool, output: str)
    """
    logger.info(f"🚀 Starting {step_name}...")
    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, "-m", script_path.replace("/", ".").replace(".py", "")],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path.cwd(),
        )

        execution_time = time.time() - start_time

        if result.returncode == 0:
            logger.info(
                f"✅ {step_name} completed successfully in {execution_time:.1f}s"
            )
            logger.debug(f"Output: {result.stdout}")
            return True, result.stdout
        else:
            logger.error(f"❌ {step_name} failed with return code {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            return False, result.stderr

    except subprocess.TimeoutExpired:
        execution_time = time.time() - start_time
        logger.error(f"❌ {step_name} timed out after {execution_time:.1f}s")
        return False, f"Timeout after {timeout}s"
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(
            f"❌ {step_name} failed with exception after {execution_time:.1f}s: {str(e)}"
        )
        return False, str(e)


def print_summary(
    results: dict[str, tuple[bool, str]], total_time: float, logger: logging.Logger
):
    """
    Print a comprehensive summary of the pipeline execution.

    Args:
        results: Dictionary mapping step names to (success, output) tuples
        total_time: Total execution time in seconds
        logger: Logger instance for output
    """
    logger.info("\n" + "=" * 80)
    logger.info("🎯 RAG EVALUATION PIPELINE SUMMARY")
    logger.info("=" * 80)

    success_count = sum(1 for success, _ in results.values() if success)
    total_steps = len(results)

    logger.info("📊 Execution Summary:")
    logger.info(f"   Total Steps: {total_steps}")
    logger.info(f"   Successful: {success_count}")
    logger.info(f"   Failed: {total_steps - success_count}")
    logger.info(f"   Total Time: {total_time:.1f}s ({total_time / 60:.1f} minutes)")

    logger.info("\n📋 Step Details:")
    for step_name, (success, output) in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        logger.info(f"   {step_name:<40} {status}")

    if success_count == total_steps:
        logger.info("\n🎉 ALL STEPS COMPLETED SUCCESSFULLY!")
        logger.info("📺 View results at: http://localhost:6006")
        logger.info("📊 Dataset experiments: http://localhost:6006/datasets")
    else:
        logger.info("\n⚠️  PIPELINE COMPLETED WITH ERRORS")
        logger.info("   Check the logs above for error details")
        logger.info("   Fix issues and re-run the failed steps")


def main():
    """
    Main orchestration function that executes the complete RAG evaluation pipeline.

    This function coordinates the execution of all three pipeline components:
    1. Environment validation and service setup
    2. Sequential execution of the three main scripts
    3. Comprehensive error handling and reporting
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Execute the complete RAG evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python claude_code_scripts/run_rag_evaluation_pipeline.py                    # Standard execution
    python claude_code_scripts/run_rag_evaluation_pipeline.py --verbose          # Debug logging
    python claude_code_scripts/run_rag_evaluation_pipeline.py --skip-services    # Skip Docker service check
    python claude_code_scripts/run_rag_evaluation_pipeline.py --testset-size 5   # Generate 5 test examples
        """,
    )
    parser.add_argument(
        "--skip-services",
        action="store_true",
        help="Skip Docker service startup (assume services are already running)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose debug logging"
    )
    parser.add_argument(
        "--testset-size",
        type=int,
        default=None,
        help="Number of examples to generate in golden test set (overrides config default)",
    )
    args = parser.parse_args()

    # Initialize logging
    logger = setup_logging(args.verbose)
    start_time = time.time()

    logger.info("🎯 Starting RAG Evaluation Pipeline")
    logger.info("=" * 80)

    # Step 1: Environment validation
    logger.info("🔍 Step 1: Environment Validation")
    env_success, env_errors = check_environment()
    if not env_success:
        logger.error("❌ Environment validation failed:")
        for error in env_errors:
            logger.error(f"   - {error}")
        sys.exit(1)
    logger.info("✅ Environment validation passed")

    # Step 2: Docker service management
    if not args.skip_services:
        logger.info("\n🐳 Step 2: Docker Service Management")

        # Check Docker availability
        docker_ok, docker_msg = check_docker_availability()
        if not docker_ok:
            logger.error(f"❌ Docker check failed: {docker_msg}")
            sys.exit(1)
        logger.info(f"✅ {docker_msg}")

        # Check current service status
        service_ok, service_output = run_service_check()
        if service_ok and "All services are running!" in service_output:
            logger.info("✅ Services are already running")
        else:
            # Start services
            if not start_docker_services(logger):
                logger.error("❌ Failed to start Docker services")
                sys.exit(1)
    else:
        logger.info("\n⏭️  Step 2: Skipping Docker service management (--skip-services)")

    # Step 3: Pipeline execution
    logger.info("\n🚀 Step 3: Pipeline Execution")

    # Set testset size environment variable if specified
    if args.testset_size is not None:
        os.environ["GOLDEN_TESTSET_SIZE"] = str(args.testset_size)
        logger.info(f"🧪 Setting golden test set size to {args.testset_size} examples")

    # Define the pipeline steps
    pipeline_steps = [
        ("Main E2E Pipeline", "src/langchain_eval_foundations_e2e.py", 600),
        ("Golden Test Set Generation", "src/langchain_eval_golden_testset.py", 300),
        ("Automated Experiments", "src/langchain_eval_experiments.py", 900),
    ]

    # Execute each step
    results = {}
    for step_name, script_path, timeout in pipeline_steps:
        success, output = execute_pipeline_step(step_name, script_path, logger, timeout)
        results[step_name] = (success, output)

        if not success:
            logger.error(f"❌ {step_name} failed. Stopping pipeline execution.")
            break

        # Small delay between steps for system stability
        time.sleep(2)

    # Step 4: Summary and cleanup
    total_time = time.time() - start_time
    print_summary(results, total_time, logger)

    # Exit with appropriate code
    all_successful = all(success for success, _ in results.values())
    sys.exit(0 if all_successful else 1)


if __name__ == "__main__":
    main()
