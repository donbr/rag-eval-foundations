"""
Golden Testset Management Package

A comprehensive package for managing versioned golden testsets with PostgreSQL storage,
semantic versioning, quality tracking, and Phoenix integration.

Key Components:
- GoldenTestsetManager: Main management class
- GoldenTestset: Testset data model
- GoldenExample: Individual example data model
- Enums: ChangeType, TestsetStatus, ValidationStatus

Example usage:
    from golden_testset import GoldenTestsetManager, GoldenExample

    async with GoldenTestsetManager() as manager:
        examples = [GoldenExample(question="...", ground_truth="...")]
        testset = await manager.create_testset("my_testset", "Description", examples)
"""

from .manager import (
    GoldenTestsetManager,
    GoldenTestset,
    GoldenExample,
    ChangeType,
    TestsetStatus,
    ValidationStatus,
    create_testset_from_dict,
    export_testset_to_dict
)

# Phase 3: Quality Validation (preserved - has test coverage)
from .quality_validator import (
    QualityValidator,
    QualityMetrics,
    ValidationResult,
    validate_testset_quality,
    check_quality_gates
)

# Note: validation_pipeline archived (no usage in main pipeline)
# Note: cost_tracker, tracing, transactions, hybrid_cost_manager archived
# See docs/archived_modules/README.md for details

__version__ = "1.0.0"
__all__ = [
    # Core Phase 1 & 2
    "GoldenTestsetManager",
    "GoldenTestset",
    "GoldenExample",
    "ChangeType",
    "TestsetStatus",
    "ValidationStatus",
    "create_testset_from_dict",
    "export_testset_to_dict",

    # Phase 3: Quality Validation (has test coverage)
    "QualityValidator",
    "QualityMetrics",
    "ValidationResult",
    "validate_testset_quality",
    "check_quality_gates",
]