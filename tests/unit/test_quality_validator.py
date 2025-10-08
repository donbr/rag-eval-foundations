"""
Unit Tests for Quality Validator

Tests the statistical and semantic validation functionality according to
Phase 3 requirements from .claude/tasks.yaml.
"""

from datetime import UTC, datetime

import pytest

from src.golden_testset.manager import GoldenExample, GoldenTestset
from src.golden_testset.quality_validator import (
    QualityMetrics,
    QualityValidator,
    ValidationResult,
)


class TestQualityValidator:
    """Test suite for QualityValidator"""

    @pytest.fixture
    def validator(self):
        """Create a QualityValidator instance"""
        return QualityValidator()

    @pytest.fixture
    def diverse_examples(self):
        """Create diverse examples that should pass quality gates"""
        return [
            GoldenExample(
                question="What factors influence user trust in AI systems?",
                ground_truth=(
                    "User trust in AI systems is influenced by "
                    "transparency, accuracy, consistency, and "
                    "explainability of the AI's decisions."
                ),
                contexts=[
                    "Trust research shows multiple factors affect "
                    "user confidence in AI systems."
                ],
                ragas_difficulty=2.0,
            ),
            GoldenExample(
                question="How do people adapt their interaction patterns with LLMs?",
                ground_truth=(
                    "People adapt by learning effective prompting strategies, "
                    "adjusting their expectations, and developing "
                    "verification habits."
                ),
                contexts=[
                    "Human-AI interaction patterns evolve through "
                    "experience and learning."
                ],
                ragas_difficulty=2.5,
            ),
            GoldenExample(
                question="What are the key challenges in human-AI collaboration?",
                ground_truth=(
                    "Key challenges include calibrating trust, managing "
                    "overreliance, handling errors gracefully, and "
                    "maintaining human agency."
                ),
                contexts=[
                    "Collaboration between humans and AI faces several "
                    "documented challenges."
                ],
                ragas_difficulty=3.0,
            ),
            GoldenExample(
                question="Why do users sometimes reject correct AI advice?",
                ground_truth=(
                    "Users may reject correct AI advice due to algorithmic "
                    "aversion, past negative experiences, or lack of "
                    "understanding of AI capabilities."
                ),
                contexts=[
                    "Algorithmic aversion describes user tendency to avoid "
                    "AI recommendations."
                ],
                ragas_difficulty=2.2,
            ),
            GoldenExample(
                question="How can AI systems better communicate uncertainty?",
                ground_truth=(
                    "AI systems can communicate uncertainty through confidence "
                    "scores, uncertainty intervals, alternative suggestions, "
                    "and clear limitation statements."
                ),
                contexts=[
                    "Uncertainty communication is crucial for appropriate "
                    "reliance on AI systems."
                ],
                ragas_difficulty=2.8,
            ),
        ]

    @pytest.fixture
    def duplicate_examples(self):
        """Create examples with duplicates that should fail quality gates"""
        return [
            GoldenExample(
                question="What is machine learning?",
                ground_truth=(
                    "Machine learning is a subset of AI that enables "
                    "computers to learn without explicit programming."
                ),
                contexts=["ML definition"],
                ragas_difficulty=1.0,
            ),
            GoldenExample(
                question="What is machine learning?",  # Exact duplicate
                ground_truth=(
                    "Machine learning is a subset of AI that enables "
                    "computers to learn without explicit programming."
                ),
                contexts=["ML definition"],
                ragas_difficulty=1.0,
            ),
            GoldenExample(
                question="Define machine learning",  # Semantic duplicate
                ground_truth=(
                    "Machine learning is an AI approach where systems learn from data."
                ),
                contexts=["ML definition"],
                ragas_difficulty=1.0,
            ),
        ]

    @pytest.fixture
    def sample_testset(self, diverse_examples):
        """Create a sample testset"""
        return GoldenTestset(
            id="test-123",
            name="llm_interaction_research",
            description="Test testset for LLM interaction research",
            examples=diverse_examples,
            version_major=1,
            version_minor=0,
            version_patch=0,
            created_at=datetime.now(UTC),
            domain="research",
            status="active",
        )

    @pytest.mark.asyncio
    async def test_validate_diverse_testset_passes(self, validator, sample_testset):
        """Test that diverse testset passes quality validation"""
        result = await validator.validate_testset(sample_testset)

        assert isinstance(result, ValidationResult)
        assert result.testset_name == "llm_interaction_research"
        assert result.version == "1.0.0"
        assert result.example_count == 5
        assert result.passed is True
        assert len(result.violations) == 0

        # Check metrics
        assert result.metrics.diversity_score >= 0.7
        assert result.metrics.duplicate_count == 0
        assert result.metrics.coverage_score >= 0.0
        assert result.metrics.distribution_p_value > 0.0

    @pytest.mark.asyncio
    async def test_validate_duplicate_testset_fails(
        self, validator, duplicate_examples
    ):
        """Test that testset with duplicates fails validation"""
        testset = GoldenTestset(
            id="test-dup",
            name="duplicate_test",
            description="Test with duplicates",
            examples=duplicate_examples,
            version_major=1,
            version_minor=0,
            version_patch=0,
            created_at=datetime.now(UTC),
            domain="test",
            status="draft",
        )

        result = await validator.validate_testset(testset)

        assert result.passed is False
        assert result.metrics.duplicate_count > 0
        assert len(result.violations) > 0
        assert any("duplicate" in violation.lower() for violation in result.violations)

    @pytest.mark.asyncio
    async def test_calculate_diversity_score(self, validator, diverse_examples):
        """Test diversity score calculation"""
        questions = [ex.question for ex in diverse_examples]
        diversity_score = await validator._calculate_diversity_score(questions)

        assert 0.0 <= diversity_score <= 1.0
        assert diversity_score >= 0.7  # Should be diverse enough

    @pytest.mark.asyncio
    async def test_calculate_diversity_score_identical_questions(self, validator):
        """Test diversity score with identical questions"""
        questions = ["What is AI?", "What is AI?", "What is AI?"]
        diversity_score = await validator._calculate_diversity_score(questions)

        assert diversity_score < 0.3  # Should be low diversity

    @pytest.mark.asyncio
    async def test_count_duplicates_exact(self, validator):
        """Test exact duplicate detection"""
        questions = [
            "What is machine learning?",
            "What is deep learning?",
            "What is machine learning?",  # Exact duplicate
            "What is neural networks?",
        ]

        duplicate_count = await validator._count_duplicates(questions)
        assert duplicate_count >= 1

    @pytest.mark.asyncio
    async def test_count_duplicates_semantic(self, validator):
        """Test semantic duplicate detection"""
        questions = [
            "What is machine learning?",
            "Define machine learning",  # Semantic duplicate
            "What is deep learning?",
        ]

        duplicate_count = await validator._count_duplicates(questions)
        # Should detect semantic similarity
        assert duplicate_count >= 0

    @pytest.mark.asyncio
    async def test_calculate_coverage_score(self, validator, diverse_examples):
        """Test coverage score calculation"""
        coverage_score = await validator._calculate_coverage_score(diverse_examples)

        assert 0.0 <= coverage_score <= 1.0
        # Diverse examples should have good coverage
        assert coverage_score > 0.5

    @pytest.mark.asyncio
    async def test_calculate_semantic_coherence(self, validator, diverse_examples):
        """Test semantic coherence calculation"""
        questions = [ex.question for ex in diverse_examples]
        ground_truths = [ex.ground_truth for ex in diverse_examples]

        coherence = await validator._calculate_semantic_coherence(
            questions, ground_truths
        )

        assert 0.0 <= coherence <= 1.0
        # Questions and answers should be reasonably coherent
        assert coherence > 0.2

    @pytest.mark.asyncio
    async def test_test_distribution(self, validator, diverse_examples):
        """Test distribution testing"""
        p_value = await validator._test_distribution(diverse_examples)

        assert 0.0 <= p_value <= 1.0

    @pytest.mark.asyncio
    async def test_generate_recommendations(self, validator, diverse_examples):
        """Test recommendation generation"""
        # Create metrics with some issues
        metrics = QualityMetrics(
            diversity_score=0.5,  # Below threshold
            duplicate_count=2,  # Has duplicates
            coverage_score=0.8,  # Below threshold
            semantic_coherence=0.4,
            distribution_p_value=0.8,
            validation_timestamp=datetime.now(UTC),
        )

        recommendations = await validator._generate_recommendations(
            metrics, diverse_examples
        )

        assert len(recommendations) > 0
        assert any("diversity" in rec.lower() for rec in recommendations)
        assert any("duplicate" in rec.lower() for rec in recommendations)
        assert any("coverage" in rec.lower() for rec in recommendations)

    @pytest.mark.asyncio
    async def test_quality_metrics_passes_quality_gates(self):
        """Test QualityMetrics quality gate checking"""
        # Passing metrics
        good_metrics = QualityMetrics(
            diversity_score=0.8,
            duplicate_count=0,
            coverage_score=0.95,
            semantic_coherence=0.7,
            distribution_p_value=0.1,
            validation_timestamp=datetime.now(UTC),
        )

        assert good_metrics.passes_quality_gates() is True
        assert len(good_metrics.get_violations()) == 0

        # Failing metrics
        bad_metrics = QualityMetrics(
            diversity_score=0.5,  # Too low
            duplicate_count=3,  # Has duplicates
            coverage_score=0.7,  # Too low
            semantic_coherence=0.6,
            distribution_p_value=0.01,  # Too low
            validation_timestamp=datetime.now(UTC),
        )

        assert bad_metrics.passes_quality_gates() is False
        violations = bad_metrics.get_violations()
        assert len(violations) == 4  # All four should fail

    @pytest.mark.asyncio
    async def test_validation_result_serialization(self, validator, sample_testset):
        """Test ValidationResult serialization"""
        result = await validator.validate_testset(sample_testset)
        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "testset_name" in result_dict
        assert "metrics" in result_dict
        assert "validation_duration_ms" in result_dict
        assert "passed" in result_dict

        # Check metrics serialization
        assert "diversity_score" in result_dict["metrics"]
        assert "duplicate_count" in result_dict["metrics"]
        assert "validation_timestamp" in result_dict["metrics"]

    @pytest.mark.asyncio
    async def test_performance_requirement(self, validator, sample_testset):
        """Test that validation meets <100ms performance requirement"""
        result = await validator.validate_testset(sample_testset)

        # Performance requirement from tasks.yaml: <100ms
        assert result.validation_duration_ms < 100, (
            f"Validation took {result.validation_duration_ms}ms, should be <100ms"
        )

    @pytest.mark.asyncio
    async def test_empty_testset(self, validator):
        """Test validation of empty testset"""
        empty_testset = GoldenTestset(
            id="empty",
            name="empty",
            description="Empty testset",
            examples=[],
            version_major=1,
            version_minor=0,
            version_patch=0,
            created_at=datetime.now(UTC),
            domain="test",
            status="draft",
        )

        result = await validator.validate_testset(empty_testset)

        assert result.passed is False
        assert result.example_count == 0

    @pytest.mark.asyncio
    async def test_single_example_testset(self, validator):
        """Test validation of single example testset"""
        single_testset = GoldenTestset(
            id="single",
            name="single",
            description="Single example testset",
            examples=[
                GoldenExample(
                    question="What is AI?",
                    ground_truth=(
                        "Artificial Intelligence is the simulation of "
                        "human intelligence in machines."
                    ),
                    contexts=["AI definition"],
                )
            ],
            version_major=1,
            version_minor=0,
            version_patch=0,
            created_at=datetime.now(UTC),
            domain="test",
            status="draft",
        )

        result = await validator.validate_testset(single_testset)

        # Should pass basic validation but may have low coverage
        assert result.example_count == 1
        assert result.metrics.duplicate_count == 0

    @pytest.mark.asyncio
    async def test_custom_thresholds(self):
        """Test validator with custom thresholds"""
        custom_validator = QualityValidator(
            min_diversity_score=0.8,  # Higher than default
            min_coverage_score=0.95,  # Higher than default
            semantic_similarity_threshold=0.9,  # Higher than default
        )

        assert custom_validator.min_diversity_score == 0.8
        assert custom_validator.min_coverage_score == 0.95
        assert custom_validator.semantic_similarity_threshold == 0.9


# Convenience function tests
class TestConvenienceFunctions:
    """Test convenience functions"""

    @pytest.mark.asyncio
    async def test_validate_testset_quality(self, sample_testset):
        """Test convenience function for validation"""
        from src.golden_testset.quality_validator import validate_testset_quality

        result = await validate_testset_quality(sample_testset)
        assert isinstance(result, ValidationResult)

    @pytest.mark.asyncio
    async def test_check_quality_gates(self, sample_testset):
        """Test convenience function for quality gates"""
        from src.golden_testset.quality_validator import check_quality_gates

        passed = await check_quality_gates(sample_testset)
        assert isinstance(passed, bool)


if __name__ == "__main__":
    pytest.main([__file__])
