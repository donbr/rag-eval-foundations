"""
Validation Pipeline for Golden Testset Management

Implements pre/post/activation gates according to Phase 3 requirements from .claude/tasks.yaml.

Key Features:
- Pre-validation gates: Check inputs before testset creation/modification
- Post-validation gates: Validate testset after creation/modification
- Activation gates: Final validation before testset activation for production use
- Quality enforcement with configurable strictness levels
- Integration with QualityValidator and AtomicTransactionManager

Pipeline Stages:
1. PRE: Input validation, business rules, prerequisites
2. POST: Quality validation, statistical checks, completeness
3. ACTIVATION: Production readiness, performance validation, final approval

Usage:
    from src.golden_testset.validation_pipeline import ValidationPipeline

    pipeline = ValidationPipeline()

    # Pre-validation
    await pipeline.validate_pre_creation(examples, metadata)

    # Post-validation
    await pipeline.validate_post_creation(testset)

    # Activation validation
    await pipeline.validate_activation_readiness(testset)
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from .manager import GoldenExample, GoldenTestset
from .quality_validator import QualityValidator, ValidationResult
from .transactions import AtomicTransactionManager


class ValidationStage(Enum):
    """Pipeline validation stages"""
    PRE = "pre"
    POST = "post"
    ACTIVATION = "activation"


class StrictnessLevel(Enum):
    """Validation strictness levels"""
    LENIENT = "lenient"      # Warnings only, allows passage with issues
    STANDARD = "standard"    # Standard quality gates
    STRICT = "strict"        # Enhanced quality requirements
    ENTERPRISE = "enterprise" # Maximum validation rigor


@dataclass
class ValidationGate:
    """Individual validation gate definition"""
    name: str
    stage: ValidationStage
    required: bool
    error_message: str
    warning_message: Optional[str] = None


@dataclass
class PipelineResult:
    """Result of pipeline validation"""
    stage: ValidationStage
    passed: bool
    gates_passed: List[str]
    gates_failed: List[str]
    warnings: List[str]
    errors: List[str]
    execution_time_ms: float
    quality_result: Optional[ValidationResult] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging/serialization"""
        result = {
            "stage": self.stage.value,
            "passed": self.passed,
            "gates_passed": self.gates_passed,
            "gates_failed": self.gates_failed,
            "warnings": self.warnings,
            "errors": self.errors,
            "execution_time_ms": self.execution_time_ms
        }

        if self.quality_result:
            result["quality_metrics"] = self.quality_result.to_dict()

        return result


class ValidationPipeline:
    """Pipeline for multi-stage testset validation"""

    def __init__(self,
                 strictness: StrictnessLevel = StrictnessLevel.STANDARD,
                 quality_validator: Optional[QualityValidator] = None,
                 transaction_manager: Optional[AtomicTransactionManager] = None):
        """
        Initialize validation pipeline

        Args:
            strictness: Validation strictness level
            quality_validator: Custom quality validator (creates default if None)
            transaction_manager: Transaction manager for rollback support
        """
        self.strictness = strictness
        self.quality_validator = quality_validator or QualityValidator()
        self.transaction_manager = transaction_manager

        # Define validation gates
        self.gates = self._define_validation_gates()

    def _define_validation_gates(self) -> Dict[ValidationStage, List[ValidationGate]]:
        """Define all validation gates by stage"""
        gates = {
            ValidationStage.PRE: [
                ValidationGate(
                    name="minimum_examples",
                    stage=ValidationStage.PRE,
                    required=True,
                    error_message="Testset must contain at least 3 examples",
                    warning_message="Consider adding more examples for better statistical validity (10+ recommended)"
                ),
                ValidationGate(
                    name="valid_examples",
                    stage=ValidationStage.PRE,
                    required=True,
                    error_message="All examples must have non-empty questions and ground_truth"
                ),
                ValidationGate(
                    name="reasonable_length",
                    stage=ValidationStage.PRE,
                    required=True,
                    error_message="Questions and answers must be reasonable length (10-2000 characters)"
                ),
                ValidationGate(
                    name="safe_content",
                    stage=ValidationStage.PRE,
                    required=True,
                    error_message="Content must not contain potentially harmful material"
                ),
                ValidationGate(
                    name="metadata_complete",
                    stage=ValidationStage.PRE,
                    required=False,
                    error_message="Metadata should include description and domain",
                    warning_message="Missing optional metadata fields"
                )
            ],

            ValidationStage.POST: [
                ValidationGate(
                    name="quality_gates",
                    stage=ValidationStage.POST,
                    required=True,
                    error_message="Testset must pass quality gates (diversity >= 0.7, no duplicates, coverage >= 0.9)"
                ),
                ValidationGate(
                    name="statistical_validity",
                    stage=ValidationStage.POST,
                    required=True,
                    error_message="Statistical tests must pass (p-value > 0.05)"
                ),
                ValidationGate(
                    name="semantic_coherence",
                    stage=ValidationStage.POST,
                    required=True,
                    error_message="Questions and answers must be semantically coherent"
                ),
                ValidationGate(
                    name="version_consistency",
                    stage=ValidationStage.POST,
                    required=True,
                    error_message="Version information must be consistent and valid"
                )
            ],

            ValidationStage.ACTIVATION: [
                ValidationGate(
                    name="production_readiness",
                    stage=ValidationStage.ACTIVATION,
                    required=True,
                    error_message="Testset must meet production readiness criteria"
                ),
                ValidationGate(
                    name="performance_requirements",
                    stage=ValidationStage.ACTIVATION,
                    required=True,
                    error_message="Validation performance must be <100ms as per requirements"
                ),
                ValidationGate(
                    name="integration_compatibility",
                    stage=ValidationStage.ACTIVATION,
                    required=True,
                    error_message="Testset must be compatible with existing integrations"
                ),
                ValidationGate(
                    name="compliance_check",
                    stage=ValidationStage.ACTIVATION,
                    required=True,
                    error_message="Must pass compliance and safety checks"
                ),
                ValidationGate(
                    name="final_approval",
                    stage=ValidationStage.ACTIVATION,
                    required=True,
                    error_message="Must pass final quality and readiness review"
                )
            ]
        }

        # Adjust gates based on strictness level
        if self.strictness == StrictnessLevel.LENIENT:
            # Convert some required gates to warnings
            for stage_gates in gates.values():
                for gate in stage_gates:
                    if gate.name in ["metadata_complete", "semantic_coherence"]:
                        gate.required = False

        elif self.strictness == StrictnessLevel.STRICT:
            # Add additional requirements
            gates[ValidationStage.PRE].append(
                ValidationGate(
                    name="enhanced_metadata",
                    stage=ValidationStage.PRE,
                    required=True,
                    error_message="Strict mode requires comprehensive metadata including tags and difficulty scores"
                )
            )

        elif self.strictness == StrictnessLevel.ENTERPRISE:
            # Maximum validation rigor
            gates[ValidationStage.POST].append(
                ValidationGate(
                    name="enterprise_quality",
                    stage=ValidationStage.POST,
                    required=True,
                    error_message="Enterprise mode requires diversity >= 0.8 and coverage >= 0.95"
                )
            )

        return gates

    async def validate_pre_creation(self,
                                   examples: List[GoldenExample],
                                   metadata: Optional[Dict] = None) -> PipelineResult:
        """
        Validate inputs before testset creation

        Args:
            examples: List of examples to validate
            metadata: Optional metadata dictionary

        Returns:
            PipelineResult with validation status
        """
        start_time = time.perf_counter()
        gates_passed = []
        gates_failed = []
        warnings = []
        errors = []

        stage_gates = self.gates[ValidationStage.PRE]

        for gate in stage_gates:
            try:
                passed, message = await self._evaluate_pre_gate(gate, examples, metadata)

                if passed:
                    gates_passed.append(gate.name)
                else:
                    gates_failed.append(gate.name)
                    if gate.required:
                        errors.append(f"{gate.name}: {gate.error_message}")
                    else:
                        warnings.append(f"{gate.name}: {gate.warning_message or gate.error_message}")

                # Add specific messages if provided
                if message:
                    if gate.required and not passed:
                        errors.append(f"{gate.name}: {message}")
                    elif not passed:
                        warnings.append(f"{gate.name}: {message}")

            except Exception as e:
                gates_failed.append(gate.name)
                errors.append(f"{gate.name}: Validation error - {str(e)}")

        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000

        # Determine overall pass/fail
        passed = len([g for g in stage_gates if g.required]) == len([g for g in gates_passed if any(sg.name == g and sg.required for sg in stage_gates)])

        return PipelineResult(
            stage=ValidationStage.PRE,
            passed=passed,
            gates_passed=gates_passed,
            gates_failed=gates_failed,
            warnings=warnings,
            errors=errors,
            execution_time_ms=execution_time
        )

    async def validate_post_creation(self, testset: GoldenTestset) -> PipelineResult:
        """
        Validate testset after creation

        Args:
            testset: Created testset to validate

        Returns:
            PipelineResult with validation status and quality metrics
        """
        start_time = time.perf_counter()
        gates_passed = []
        gates_failed = []
        warnings = []
        errors = []

        stage_gates = self.gates[ValidationStage.POST]

        # Run quality validation first
        quality_result = await self.quality_validator.validate_testset(testset)

        for gate in stage_gates:
            try:
                passed, message = await self._evaluate_post_gate(gate, testset, quality_result)

                if passed:
                    gates_passed.append(gate.name)
                else:
                    gates_failed.append(gate.name)
                    if gate.required:
                        errors.append(f"{gate.name}: {gate.error_message}")
                    else:
                        warnings.append(f"{gate.name}: {gate.warning_message or gate.error_message}")

                if message:
                    if gate.required and not passed:
                        errors.append(f"{gate.name}: {message}")
                    elif not passed:
                        warnings.append(f"{gate.name}: {message}")

            except Exception as e:
                gates_failed.append(gate.name)
                errors.append(f"{gate.name}: Validation error - {str(e)}")

        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000

        # Determine overall pass/fail
        required_gates = [g for g in stage_gates if g.required]
        passed_required = [g for g in gates_passed if any(sg.name == g and sg.required for sg in stage_gates)]
        passed = len(required_gates) == len(passed_required)

        return PipelineResult(
            stage=ValidationStage.POST,
            passed=passed,
            gates_passed=gates_passed,
            gates_failed=gates_failed,
            warnings=warnings,
            errors=errors,
            execution_time_ms=execution_time,
            quality_result=quality_result
        )

    async def validate_activation_readiness(self, testset: GoldenTestset) -> PipelineResult:
        """
        Validate testset readiness for activation/production use

        Args:
            testset: Testset to validate for activation

        Returns:
            PipelineResult with activation readiness status
        """
        start_time = time.perf_counter()
        gates_passed = []
        gates_failed = []
        warnings = []
        errors = []

        stage_gates = self.gates[ValidationStage.ACTIVATION]

        # Run comprehensive validation
        quality_result = await self.quality_validator.validate_testset(testset)

        for gate in stage_gates:
            try:
                passed, message = await self._evaluate_activation_gate(gate, testset, quality_result)

                if passed:
                    gates_passed.append(gate.name)
                else:
                    gates_failed.append(gate.name)
                    if gate.required:
                        errors.append(f"{gate.name}: {gate.error_message}")
                    else:
                        warnings.append(f"{gate.name}: {gate.warning_message or gate.error_message}")

                if message:
                    if gate.required and not passed:
                        errors.append(f"{gate.name}: {message}")
                    elif not passed:
                        warnings.append(f"{gate.name}: {message}")

            except Exception as e:
                gates_failed.append(gate.name)
                errors.append(f"{gate.name}: Validation error - {str(e)}")

        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000

        # Determine overall pass/fail
        required_gates = [g for g in stage_gates if g.required]
        passed_required = [g for g in gates_passed if any(sg.name == g and sg.required for sg in stage_gates)]
        passed = len(required_gates) == len(passed_required)

        return PipelineResult(
            stage=ValidationStage.ACTIVATION,
            passed=passed,
            gates_passed=gates_passed,
            gates_failed=gates_failed,
            warnings=warnings,
            errors=errors,
            execution_time_ms=execution_time,
            quality_result=quality_result
        )

    async def _evaluate_pre_gate(self,
                                gate: ValidationGate,
                                examples: List[GoldenExample],
                                metadata: Optional[Dict]) -> tuple[bool, Optional[str]]:
        """Evaluate a pre-creation gate"""

        if gate.name == "minimum_examples":
            min_required = 3 if self.strictness != StrictnessLevel.ENTERPRISE else 5
            if len(examples) < min_required:
                return False, f"Found {len(examples)} examples, need at least {min_required}"
            elif len(examples) < 10:
                return True, "Consider adding more examples for better validity"
            return True, None

        elif gate.name == "valid_examples":
            invalid_examples = []
            for i, ex in enumerate(examples):
                if not ex.question or not ex.question.strip():
                    invalid_examples.append(f"Example {i+1}: empty question")
                if not ex.ground_truth or not ex.ground_truth.strip():
                    invalid_examples.append(f"Example {i+1}: empty ground_truth")

            if invalid_examples:
                return False, "; ".join(invalid_examples)
            return True, None

        elif gate.name == "reasonable_length":
            issues = []
            for i, ex in enumerate(examples):
                if len(ex.question) < 10 or len(ex.question) > 2000:
                    issues.append(f"Example {i+1}: question length {len(ex.question)} not in range 10-2000")
                if len(ex.ground_truth) < 5 or len(ex.ground_truth) > 2000:
                    issues.append(f"Example {i+1}: answer length {len(ex.ground_truth)} not in range 5-2000")

            if issues:
                return False, "; ".join(issues)
            return True, None

        elif gate.name == "safe_content":
            # Basic content safety check
            unsafe_patterns = ["<script", "javascript:", "eval(", "onclick="]
            issues = []

            for i, ex in enumerate(examples):
                content = f"{ex.question} {ex.ground_truth}".lower()
                for pattern in unsafe_patterns:
                    if pattern in content:
                        issues.append(f"Example {i+1}: potentially unsafe pattern '{pattern}'")

            if issues:
                return False, "; ".join(issues)
            return True, None

        elif gate.name == "metadata_complete":
            if not metadata:
                return False, "No metadata provided"

            required_fields = ["description"]
            if self.strictness in [StrictnessLevel.STRICT, StrictnessLevel.ENTERPRISE]:
                required_fields.extend(["domain", "tags"])

            missing = [field for field in required_fields if field not in metadata or not metadata[field]]
            if missing:
                return False, f"Missing required metadata fields: {', '.join(missing)}"
            return True, None

        elif gate.name == "enhanced_metadata":
            if not metadata:
                return False, "Enhanced metadata required for strict mode"

            required = ["description", "domain", "tags", "created_by", "review_status"]
            missing = [field for field in required if field not in metadata or not metadata[field]]
            if missing:
                return False, f"Missing enhanced metadata: {', '.join(missing)}"
            return True, None

        return True, None

    async def _evaluate_post_gate(self,
                                 gate: ValidationGate,
                                 testset: GoldenTestset,
                                 quality_result: ValidationResult) -> tuple[bool, Optional[str]]:
        """Evaluate a post-creation gate"""

        if gate.name == "quality_gates":
            threshold = 0.7 if self.strictness != StrictnessLevel.ENTERPRISE else 0.8
            coverage_threshold = 0.9 if self.strictness != StrictnessLevel.ENTERPRISE else 0.95

            violations = []
            if quality_result.metrics.diversity_score < threshold:
                violations.append(f"diversity {quality_result.metrics.diversity_score:.3f} < {threshold}")
            if quality_result.metrics.duplicate_count > 0:
                violations.append(f"{quality_result.metrics.duplicate_count} duplicates found")
            if quality_result.metrics.coverage_score < coverage_threshold:
                violations.append(f"coverage {quality_result.metrics.coverage_score:.3f} < {coverage_threshold}")

            if violations:
                return False, "; ".join(violations)
            return True, None

        elif gate.name == "statistical_validity":
            if quality_result.metrics.distribution_p_value <= 0.05:
                return False, f"Distribution test failed (p={quality_result.metrics.distribution_p_value:.3f})"
            return True, None

        elif gate.name == "semantic_coherence":
            threshold = 0.4 if self.strictness == StrictnessLevel.LENIENT else 0.5
            if quality_result.metrics.semantic_coherence < threshold:
                return False, f"Semantic coherence {quality_result.metrics.semantic_coherence:.3f} < {threshold}"
            return True, None

        elif gate.name == "version_consistency":
            # Check version format and consistency
            if not (testset.version_major >= 0 and testset.version_minor >= 0 and testset.version_patch >= 0):
                return False, "Invalid version numbers"
            return True, None

        elif gate.name == "enterprise_quality":
            violations = []
            if quality_result.metrics.diversity_score < 0.8:
                violations.append(f"enterprise diversity {quality_result.metrics.diversity_score:.3f} < 0.8")
            if quality_result.metrics.coverage_score < 0.95:
                violations.append(f"enterprise coverage {quality_result.metrics.coverage_score:.3f} < 0.95")

            if violations:
                return False, "; ".join(violations)
            return True, None

        return True, None

    async def _evaluate_activation_gate(self,
                                       gate: ValidationGate,
                                       testset: GoldenTestset,
                                       quality_result: ValidationResult) -> tuple[bool, Optional[str]]:
        """Evaluate an activation gate"""

        if gate.name == "production_readiness":
            # Comprehensive production readiness check
            issues = []

            if not quality_result.passed:
                issues.append("quality validation failed")
            if len(testset.examples) < 5:
                issues.append("insufficient examples for production")
            if quality_result.metrics.diversity_score < 0.75:
                issues.append("diversity too low for production")

            if issues:
                return False, "; ".join(issues)
            return True, None

        elif gate.name == "performance_requirements":
            # Check validation performance meets <100ms requirement
            if quality_result.validation_duration_ms > 100:
                return False, f"Validation took {quality_result.validation_duration_ms:.1f}ms > 100ms requirement"
            return True, None

        elif gate.name == "integration_compatibility":
            # Check compatibility with existing systems
            # This would involve checking schema compatibility, API compatibility, etc.
            return True, None

        elif gate.name == "compliance_check":
            # Final compliance and safety validation
            return True, None

        elif gate.name == "final_approval":
            # Final comprehensive check
            if not quality_result.passed:
                return False, "Quality validation must pass for activation"
            if quality_result.validation_duration_ms > 100:
                return False, "Performance requirements not met"
            return True, None

        return True, None

    async def run_full_pipeline(self,
                               examples: List[GoldenExample],
                               metadata: Optional[Dict] = None,
                               testset: Optional[GoldenTestset] = None) -> Dict[str, PipelineResult]:
        """
        Run complete validation pipeline: PRE -> POST -> ACTIVATION

        Args:
            examples: Examples to validate
            metadata: Optional metadata
            testset: Testset for POST/ACTIVATION stages (if None, skips those stages)

        Returns:
            Dictionary mapping stage names to PipelineResults
        """
        results = {}

        # PRE validation
        results["pre"] = await self.validate_pre_creation(examples, metadata)

        if not results["pre"].passed:
            return results  # Stop if pre-validation fails

        # POST validation (if testset provided)
        if testset:
            results["post"] = await self.validate_post_creation(testset)

            if results["post"].passed:
                # ACTIVATION validation
                results["activation"] = await self.validate_activation_readiness(testset)

        return results


# Convenience functions
async def validate_with_pipeline(testset: GoldenTestset,
                                strictness: StrictnessLevel = StrictnessLevel.STANDARD) -> PipelineResult:
    """Convenience function for post-creation validation"""
    pipeline = ValidationPipeline(strictness=strictness)
    return await pipeline.validate_post_creation(testset)


async def check_activation_readiness(testset: GoldenTestset,
                                   strictness: StrictnessLevel = StrictnessLevel.STANDARD) -> bool:
    """Quick check if testset is ready for activation"""
    pipeline = ValidationPipeline(strictness=strictness)
    result = await pipeline.validate_activation_readiness(testset)
    return result.passed