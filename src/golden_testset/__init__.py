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

# Phase 3: Quality Validation Pipeline
from .quality_validator import (
    QualityValidator,
    QualityMetrics,
    ValidationResult,
    validate_testset_quality,
    check_quality_gates
)

from .validation_pipeline import (
    ValidationPipeline,
    ValidationStage,
    StrictnessLevel,
    PipelineResult,
    validate_with_pipeline,
    check_activation_readiness
)

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

    # Phase 3: Quality Validation
    "QualityValidator",
    "QualityMetrics",
    "ValidationResult",
    "validate_testset_quality",
    "check_quality_gates",
    "ValidationPipeline",
    "ValidationStage",
    "StrictnessLevel",
    "PipelineResult",
    "validate_with_pipeline",
    "check_activation_readiness"
]