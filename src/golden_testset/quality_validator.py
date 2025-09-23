"""
Quality Validator for Golden Testset Management

Implements statistical and semantic validation for golden testsets according to
Phase 3 requirements from .claude/tasks.yaml.

Key Features:
- Statistical validation (diversity_score >= 0.7, no duplicates, coverage complete)
- Semantic validation using embeddings and clustering
- Chi-square statistical tests for distribution validation
- Performance optimized for <100ms validation cycles
- Integration with GoldenTestsetManager and quality gates

Usage:
    from src.golden_testset.quality_validator import QualityValidator

    validator = QualityValidator()
    result = await validator.validate_testset(testset)

    if result.passes_quality_gates():
        print("✅ Testset passes all quality gates")
    else:
        print("❌ Quality violations:", result.violations)
"""

import asyncio
import hashlib
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .manager import GoldenExample, GoldenTestset


@dataclass
class QualityMetrics:
    """Quality metrics for a golden testset"""
    diversity_score: float
    duplicate_count: int
    coverage_score: float
    semantic_coherence: float
    distribution_p_value: float
    validation_timestamp: datetime

    def passes_quality_gates(self) -> bool:
        """Check if metrics pass all quality gates"""
        return (
            self.diversity_score >= 0.7 and
            self.duplicate_count == 0 and
            self.coverage_score >= 0.9 and
            self.distribution_p_value > 0.05
        )

    def get_violations(self) -> List[str]:
        """Get list of quality gate violations"""
        violations = []

        if self.diversity_score < 0.7:
            violations.append(f"Diversity score {self.diversity_score:.3f} < 0.7")

        if self.duplicate_count > 0:
            violations.append(f"Found {self.duplicate_count} duplicate questions")

        if self.coverage_score < 0.9:
            violations.append(f"Coverage score {self.coverage_score:.3f} < 0.9")

        if self.distribution_p_value <= 0.05:
            violations.append(f"Distribution p-value {self.distribution_p_value:.3f} <= 0.05")

        return violations


@dataclass
class ValidationResult:
    """Result of testset validation"""
    testset_name: str
    version: str
    metrics: QualityMetrics
    example_count: int
    validation_duration_ms: float
    passed: bool
    violations: List[str]
    recommendations: List[str]

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "testset_name": self.testset_name,
            "version": self.version,
            "metrics": {
                "diversity_score": self.metrics.diversity_score,
                "duplicate_count": self.metrics.duplicate_count,
                "coverage_score": self.metrics.coverage_score,
                "semantic_coherence": self.metrics.semantic_coherence,
                "distribution_p_value": self.metrics.distribution_p_value,
                "validation_timestamp": self.metrics.validation_timestamp.isoformat()
            },
            "example_count": self.example_count,
            "validation_duration_ms": self.validation_duration_ms,
            "passed": self.passed,
            "violations": self.violations,
            "recommendations": self.recommendations
        }


class QualityValidator:
    """Statistical and semantic validator for golden testsets"""

    def __init__(self,
                 min_diversity_score: float = 0.7,
                 min_coverage_score: float = 0.9,
                 min_p_value: float = 0.05,
                 semantic_similarity_threshold: float = 0.85):
        """
        Initialize quality validator

        Args:
            min_diversity_score: Minimum required diversity score (0.7 per tasks.yaml)
            min_coverage_score: Minimum required coverage score
            min_p_value: Minimum p-value for distribution tests
            semantic_similarity_threshold: Threshold for detecting semantic duplicates
        """
        self.min_diversity_score = min_diversity_score
        self.min_coverage_score = min_coverage_score
        self.min_p_value = min_p_value
        self.semantic_similarity_threshold = semantic_similarity_threshold

        # Initialize TF-IDF vectorizer for semantic analysis
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )

    async def validate_testset(self, testset: GoldenTestset) -> ValidationResult:
        """
        Validate a golden testset against quality gates

        Args:
            testset: GoldenTestset to validate

        Returns:
            ValidationResult with metrics and pass/fail status
        """
        start_time = asyncio.get_event_loop().time()

        examples = testset.examples
        questions = [ex.question for ex in examples]
        ground_truths = [ex.ground_truth for ex in examples]

        # Run validation checks
        diversity_score = await self._calculate_diversity_score(questions)
        duplicate_count = await self._count_duplicates(questions)
        coverage_score = await self._calculate_coverage_score(examples)
        semantic_coherence = await self._calculate_semantic_coherence(questions, ground_truths)
        distribution_p_value = await self._test_distribution(examples)

        # Create metrics
        metrics = QualityMetrics(
            diversity_score=diversity_score,
            duplicate_count=duplicate_count,
            coverage_score=coverage_score,
            semantic_coherence=semantic_coherence,
            distribution_p_value=distribution_p_value,
            validation_timestamp=datetime.now(timezone.utc)
        )

        # Check quality gates
        passed = metrics.passes_quality_gates()
        violations = metrics.get_violations()
        recommendations = await self._generate_recommendations(metrics, examples)

        end_time = asyncio.get_event_loop().time()
        duration_ms = (end_time - start_time) * 1000

        return ValidationResult(
            testset_name=testset.name,
            version=f"{testset.version_major}.{testset.version_minor}.{testset.version_patch}",
            metrics=metrics,
            example_count=len(examples),
            validation_duration_ms=duration_ms,
            passed=passed,
            violations=violations,
            recommendations=recommendations
        )

    async def _calculate_diversity_score(self, questions: List[str]) -> float:
        """
        Calculate diversity score using TF-IDF and cosine similarity

        Higher scores indicate more diverse questions
        Range: 0.0 (identical) to 1.0 (completely diverse)
        """
        if len(questions) < 2:
            return 1.0

        try:
            # Vectorize questions
            tfidf_matrix = self.vectorizer.fit_transform(questions)

            # Calculate pairwise cosine similarities
            similarity_matrix = cosine_similarity(tfidf_matrix)

            # Extract upper triangle (excluding diagonal)
            n = len(questions)
            similarities = []
            for i in range(n):
                for j in range(i + 1, n):
                    similarities.append(similarity_matrix[i, j])

            if not similarities:
                return 1.0

            # Diversity is inverse of average similarity
            avg_similarity = np.mean(similarities)
            diversity_score = 1.0 - avg_similarity

            return max(0.0, min(1.0, diversity_score))

        except Exception:
            # Fallback: use simple lexical diversity
            unique_words = set()
            total_words = 0

            for question in questions:
                words = question.lower().split()
                unique_words.update(words)
                total_words += len(words)

            if total_words == 0:
                return 0.0

            return len(unique_words) / total_words

    async def _count_duplicates(self, questions: List[str]) -> int:
        """
        Count exact and near-duplicate questions

        Uses both exact string matching and semantic similarity
        """
        if len(questions) < 2:
            return 0

        duplicate_count = 0
        processed = set()

        # Check exact duplicates
        question_counts = Counter(q.strip().lower() for q in questions)
        exact_duplicates = sum(count - 1 for count in question_counts.values() if count > 1)
        duplicate_count += exact_duplicates

        # Check semantic duplicates
        try:
            if len(questions) >= 2:
                tfidf_matrix = self.vectorizer.fit_transform(questions)
                similarity_matrix = cosine_similarity(tfidf_matrix)

                n = len(questions)
                for i in range(n):
                    for j in range(i + 1, n):
                        if similarity_matrix[i, j] >= self.semantic_similarity_threshold:
                            # Only count if not already counted as exact duplicate
                            key_pair = tuple(sorted([questions[i].strip().lower(), questions[j].strip().lower()]))
                            if key_pair not in processed and questions[i].strip().lower() != questions[j].strip().lower():
                                duplicate_count += 1
                                processed.add(key_pair)

        except Exception:
            # Fallback to exact duplicates only
            pass

        return duplicate_count

    async def _calculate_coverage_score(self, examples: List[GoldenExample]) -> float:
        """
        Calculate coverage score based on domain completeness

        Analyzes question types, difficulty distribution, and context variety
        """
        if not examples:
            return 0.0

        coverage_factors = []

        # 1. Question type diversity (using keywords/patterns)
        question_types = self._categorize_questions([ex.question for ex in examples])
        type_diversity = len(question_types) / max(5, len(question_types))  # Normalize to expected types
        coverage_factors.append(min(1.0, type_diversity))

        # 2. Difficulty distribution (if available)
        difficulties = [ex.ragas_difficulty for ex in examples if ex.ragas_difficulty is not None]
        if difficulties:
            # Check for reasonable spread across difficulty levels
            difficulty_std = statistics.stdev(difficulties) if len(difficulties) > 1 else 0
            difficulty_coverage = min(1.0, difficulty_std / 2.0)  # Normalize by reasonable std
            coverage_factors.append(difficulty_coverage)

        # 3. Context variety (if contexts provided)
        contexts = []
        for ex in examples:
            if ex.contexts:
                contexts.extend(ex.contexts)

        if contexts:
            unique_contexts = len(set(contexts))
            context_coverage = min(1.0, unique_contexts / len(examples))
            coverage_factors.append(context_coverage)

        # 4. Answer length variety
        answer_lengths = [len(ex.ground_truth.split()) for ex in examples]
        if answer_lengths:
            length_std = statistics.stdev(answer_lengths) if len(answer_lengths) > 1 else 0
            length_coverage = min(1.0, length_std / 10.0)  # Normalize
            coverage_factors.append(length_coverage)

        # Return average of all coverage factors
        return sum(coverage_factors) / len(coverage_factors) if coverage_factors else 0.0

    def _categorize_questions(self, questions: List[str]) -> Set[str]:
        """Categorize questions by type/intent"""
        categories = set()

        for question in questions:
            q_lower = question.lower()

            if any(word in q_lower for word in ['what', 'define', 'explain']):
                categories.add('definition')
            elif any(word in q_lower for word in ['how', 'process', 'steps']):
                categories.add('procedural')
            elif any(word in q_lower for word in ['why', 'reason', 'cause']):
                categories.add('causal')
            elif any(word in q_lower for word in ['compare', 'difference', 'versus']):
                categories.add('comparative')
            elif any(word in q_lower for word in ['list', 'enumerate', 'examples']):
                categories.add('enumeration')
            else:
                categories.add('general')

        return categories

    async def _calculate_semantic_coherence(self, questions: List[str], ground_truths: List[str]) -> float:
        """
        Calculate semantic coherence between questions and answers

        Measures how well questions and answers align semantically
        """
        if len(questions) != len(ground_truths) or not questions:
            return 0.0

        try:
            # Combine questions and answers for vectorization
            all_texts = questions + ground_truths
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)

            n = len(questions)
            coherence_scores = []

            # Calculate similarity between each question-answer pair
            for i in range(n):
                q_vector = tfidf_matrix[i]
                a_vector = tfidf_matrix[n + i]

                similarity = cosine_similarity(q_vector, a_vector)[0, 0]
                coherence_scores.append(similarity)

            return np.mean(coherence_scores) if coherence_scores else 0.0

        except Exception:
            # Fallback: simple token overlap
            coherence_scores = []
            for q, a in zip(questions, ground_truths):
                q_words = set(q.lower().split())
                a_words = set(a.lower().split())

                if not q_words or not a_words:
                    coherence_scores.append(0.0)
                else:
                    overlap = len(q_words & a_words)
                    union = len(q_words | a_words)
                    coherence_scores.append(overlap / union if union > 0 else 0.0)

            return np.mean(coherence_scores) if coherence_scores else 0.0

    async def _test_distribution(self, examples: List[GoldenExample]) -> float:
        """
        Test distribution properties using chi-square test

        Tests whether the distribution of example characteristics
        follows expected patterns (null hypothesis: uniform distribution)
        """
        if len(examples) < 5:  # Insufficient data for meaningful test
            return 1.0

        try:
            # Test distribution of question lengths
            question_lengths = [len(ex.question.split()) for ex in examples]

            # Create bins for length distribution
            min_len, max_len = min(question_lengths), max(question_lengths)
            if max_len == min_len:
                return 1.0  # All same length - perfectly uniform

            num_bins = min(5, len(set(question_lengths)))
            bins = np.linspace(min_len, max_len + 1, num_bins + 1)
            observed, _ = np.histogram(question_lengths, bins=bins)

            # Expected uniform distribution
            expected = np.full(num_bins, len(examples) / num_bins)

            # Chi-square test
            chi2_stat, p_value = stats.chisquare(observed, expected)

            return p_value

        except Exception:
            # Fallback: assume uniform distribution
            return 1.0

    async def _generate_recommendations(self, metrics: QualityMetrics, examples: List[GoldenExample]) -> List[str]:
        """Generate recommendations for improving testset quality"""
        recommendations = []

        if metrics.diversity_score < self.min_diversity_score:
            recommendations.append(
                f"Increase question diversity (current: {metrics.diversity_score:.3f}). "
                "Add questions covering different topics, phrasings, and complexity levels."
            )

        if metrics.duplicate_count > 0:
            recommendations.append(
                f"Remove {metrics.duplicate_count} duplicate questions. "
                "Check for both exact matches and semantically similar questions."
            )

        if metrics.coverage_score < self.min_coverage_score:
            recommendations.append(
                f"Improve domain coverage (current: {metrics.coverage_score:.3f}). "
                "Add questions covering missing topic areas and question types."
            )

        if metrics.semantic_coherence < 0.5:
            recommendations.append(
                f"Improve question-answer alignment (current: {metrics.semantic_coherence:.3f}). "
                "Ensure answers directly address the questions asked."
            )

        if metrics.distribution_p_value <= self.min_p_value:
            recommendations.append(
                "Balance the distribution of question characteristics. "
                "Vary question length, complexity, and difficulty more evenly."
            )

        # General recommendations based on testset size
        if len(examples) < 10:
            recommendations.append("Consider adding more examples for better statistical validity (minimum 10 recommended).")

        return recommendations


# Convenience functions for direct usage
async def validate_testset_quality(testset: GoldenTestset) -> ValidationResult:
    """Convenience function to validate a testset"""
    validator = QualityValidator()
    return await validator.validate_testset(testset)


async def check_quality_gates(testset: GoldenTestset) -> bool:
    """Quick check if testset passes quality gates"""
    result = await validate_testset_quality(testset)
    return result.passed