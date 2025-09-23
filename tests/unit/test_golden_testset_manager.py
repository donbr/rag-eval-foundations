#!/usr/bin/env python3
"""
Unit Tests for GoldenTestsetManager

Comprehensive test suite covering:
- CRUD operations for testsets and examples
- Version management and semantic versioning
- Quality metrics tracking and validation
- Change detection integration
- Error handling and edge cases
- Async database operations
- Performance requirements

Test Requirements:
- Uses pytest-asyncio for async testing
- Mock database operations to avoid external dependencies
- Test data fixtures for consistent test scenarios
- Performance benchmarking for critical operations
"""

import pytest
import asyncio
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
from typing import Dict, List, Any
import json

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from golden_testset.manager import (
    GoldenTestsetManager, GoldenTestset, GoldenExample,
    ChangeType, TestsetStatus, ValidationStatus,
    create_testset_from_dict, export_testset_to_dict
)
from golden_testset.versioning import SemanticVersion, VersionBump
from golden_testset.change_detector import ChangeDetectionResult, ChangeRecord

# Test fixtures and sample data

@pytest.fixture
def sample_examples():
    """Sample golden examples for testing"""
    return [
        GoldenExample(
            question="What is the Federal Pell Grant eligibility requirement?",
            ground_truth="Students must demonstrate exceptional financial need and be enrolled in an eligible undergraduate program",
            contexts=["Federal Pell Grants are need-based financial aid."],
            ragas_question_type="simple",
            ragas_difficulty=2.0
        ),
        GoldenExample(
            question="How long is the Direct Loan repayment period?",
            ground_truth="Standard repayment is 10 years, but extended and income-driven plans are available",
            contexts=["Direct Loans have various repayment options."],
            ragas_question_type="simple",
            ragas_difficulty=1.5
        ),
        GoldenExample(
            question="What documents are required for FAFSA verification?",
            ground_truth="Tax returns, W-2 forms, bank statements, and investment records may be required",
            contexts=["FAFSA verification requires documentation."],
            ragas_question_type="complex",
            ragas_difficulty=3.0
        )
    ]

@pytest.fixture
def sample_testset(sample_examples):
    """Sample golden testset for testing"""
    return GoldenTestset(
        id="test-uuid-123",
        name="financial_aid_baseline",
        description="Baseline testset for financial aid questions",
        version_major=1,
        version_minor=0,
        version_patch=0,
        version_label=None,
        status=TestsetStatus.APPROVED,
        domain="financial_aid",
        examples=sample_examples,
        metadata={"created_by": "test_user", "source": "manual_curation"},
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )

@pytest.fixture
def mock_db_manager():
    """Mock database manager for testing"""
    mock = AsyncMock()

    # Mock connection context manager
    mock_conn = AsyncMock()
    mock.get_connection.return_value.__aenter__.return_value = mock_conn
    mock.get_connection.return_value.__aexit__.return_value = None

    return mock, mock_conn

class TestGoldenTestsetManagerCreation:
    """Test GoldenTestsetManager initialization and setup"""

    @pytest.mark.asyncio
    async def test_manager_initialization_default(self):
        """Test manager initialization with default parameters"""
        with patch('golden_testset.manager.DatabaseConnectionManager') as mock_db:
            manager = GoldenTestsetManager()

            assert manager.connection_string == "postgresql://langchain:langchain@localhost:6024/langchain"
            assert manager.db_manager is None

    @pytest.mark.asyncio
    async def test_manager_initialization_custom(self):
        """Test manager initialization with custom parameters"""
        custom_url = "postgresql://user:pass@localhost:5432/testdb"

        with patch('golden_testset.manager.DatabaseConnectionManager') as mock_db:
            manager = GoldenTestsetManager(connection_string=custom_url)

            assert manager.connection_string == custom_url
            assert manager.db_manager is None

    @pytest.mark.asyncio
    async def test_async_context_manager(self, mock_db_manager):
        """Test async context manager functionality"""
        mock_db, mock_conn = mock_db_manager

        with patch('golden_testset.manager.DatabaseConnectionManager', return_value=mock_db):
            async with GoldenTestsetManager() as manager:
                assert manager is not None
                assert manager.db_manager == mock_db

            # Verify cleanup was called
            mock_db.close.assert_called_once()

class TestTestsetCRUDOperations:
    """Test Create, Read, Update, Delete operations for testsets"""

    @pytest.mark.asyncio
    async def test_create_testset_success(self, mock_db_manager, sample_examples):
        """Test successful testset creation"""
        mock_db, mock_conn = mock_db_manager

        # Mock database responses
        mock_conn.fetchval.return_value = "test-uuid-123"  # For INSERT
        mock_conn.fetchrow.return_value = {
            'id': 'test-uuid-123',
            'name': 'test_testset',
            'description': 'Test description',
            'version_major': 1,
            'version_minor': 0,
            'version_patch': 0,
            'version_label': None,
            'status': 'approved',
            'domain': 'test_domain',
            'metadata': '{"created_by": "test"}',
            'created_at': datetime.now(timezone.utc),
            'updated_at': datetime.now(timezone.utc)
        }

        with patch('golden_testset.manager.DatabaseConnectionManager', return_value=mock_db):
            async with GoldenTestsetManager() as manager:
                testset = await manager.create_testset(
                    name="test_testset",
                    description="Test description",
                    examples=sample_examples,
                    domain="test_domain"
                )

                assert testset.name == "test_testset"
                assert testset.version_major == 1
                assert testset.version_minor == 0
                assert testset.version_patch == 0
                assert len(testset.examples) == 3

                # Verify database calls
                mock_conn.execute.assert_called()

    @pytest.mark.asyncio
    async def test_create_testset_duplicate_name(self, mock_db_manager, sample_examples):
        """Test testset creation with duplicate name"""
        mock_db, mock_conn = mock_db_manager

        # Mock duplicate key error
        from asyncpg.exceptions import UniqueViolationError
        mock_conn.fetchval.side_effect = UniqueViolationError("duplicate key")

        with patch('golden_testset.manager.DatabaseConnectionManager', return_value=mock_db):
            async with GoldenTestsetManager() as manager:
                with pytest.raises(ValueError, match="Testset with name"):
                    await manager.create_testset(
                        name="duplicate_testset",
                        description="Test description",
                        examples=sample_examples
                    )

    @pytest.mark.asyncio
    async def test_get_testset_by_name_success(self, mock_db_manager, sample_testset):
        """Test successful testset retrieval by name"""
        mock_db, mock_conn = mock_db_manager

        # Mock database response
        testset_row = {
            'id': sample_testset.id,
            'name': sample_testset.name,
            'description': sample_testset.description,
            'version_major': sample_testset.version_major,
            'version_minor': sample_testset.version_minor,
            'version_patch': sample_testset.version_patch,
            'version_label': sample_testset.version_label,
            'status': sample_testset.status.value,
            'domain': sample_testset.domain,
            'metadata': json.dumps(sample_testset.metadata),
            'created_at': sample_testset.created_at,
            'updated_at': sample_testset.updated_at
        }

        example_rows = [
            {
                'id': f'ex-{i}',
                'testset_id': sample_testset.id,
                'question': ex.question,
                'ground_truth': ex.ground_truth,
                'metadata': json.dumps(ex.metadata or {}),
                'created_at': datetime.now(timezone.utc)
            }
            for i, ex in enumerate(sample_testset.examples)
        ]

        mock_conn.fetchrow.return_value = testset_row
        mock_conn.fetch.return_value = example_rows

        with patch('golden_testset.manager.DatabaseConnectionManager', return_value=mock_db):
            async with GoldenTestsetManager() as manager:
                result = await manager.get_testset_by_name("financial_aid_baseline")

                assert result is not None
                assert result.name == sample_testset.name
                assert result.version_major == 1
                assert len(result.examples) == 3

    @pytest.mark.asyncio
    async def test_get_testset_not_found(self, mock_db_manager):
        """Test testset retrieval when testset doesn't exist"""
        mock_db, mock_conn = mock_db_manager

        mock_conn.fetchrow.return_value = None

        with patch('golden_testset.manager.DatabaseConnectionManager', return_value=mock_db):
            async with GoldenTestsetManager() as manager:
                result = await manager.get_testset_by_name("nonexistent_testset")

                assert result is None

    @pytest.mark.asyncio
    async def test_update_testset_success(self, mock_db_manager, sample_testset):
        """Test successful testset update"""
        mock_db, mock_conn = mock_db_manager

        # Mock successful update
        mock_conn.execute.return_value = "UPDATE 1"

        with patch('golden_testset.manager.DatabaseConnectionManager', return_value=mock_db):
            async with GoldenTestsetManager() as manager:
                updated_testset = sample_testset
                updated_testset.description = "Updated description"

                await manager.update_testset(updated_testset)

                # Verify update was called
                mock_conn.execute.assert_called()

    @pytest.mark.asyncio
    async def test_delete_testset_success(self, mock_db_manager):
        """Test successful testset deletion"""
        mock_db, mock_conn = mock_db_manager

        mock_conn.execute.return_value = "DELETE 1"

        with patch('golden_testset.manager.DatabaseConnectionManager', return_value=mock_db):
            async with GoldenTestsetManager() as manager:
                success = await manager.delete_testset("test_testset")

                assert success is True
                mock_conn.execute.assert_called()

    @pytest.mark.asyncio
    async def test_list_testsets(self, mock_db_manager):
        """Test listing all testsets"""
        mock_db, mock_conn = mock_db_manager

        mock_testsets = [
            {
                'id': 'test-1',
                'name': 'testset_1',
                'description': 'First testset',
                'version_major': 1,
                'version_minor': 0,
                'version_patch': 0,
                'version_label': None,
                'status': 'approved',
                'domain': 'test',
                'metadata': '{}',
                'created_at': datetime.now(timezone.utc),
                'updated_at': datetime.now(timezone.utc)
            },
            {
                'id': 'test-2',
                'name': 'testset_2',
                'description': 'Second testset',
                'version_major': 1,
                'version_minor': 1,
                'version_patch': 0,
                'version_label': None,
                'status': 'draft',
                'domain': 'test',
                'metadata': '{}',
                'created_at': datetime.now(timezone.utc),
                'updated_at': datetime.now(timezone.utc)
            }
        ]

        mock_conn.fetch.return_value = mock_testsets

        with patch('golden_testset.manager.DatabaseConnectionManager', return_value=mock_db):
            async with GoldenTestsetManager() as manager:
                testsets = await manager.list_testsets()

                assert len(testsets) == 2
                assert testsets[0].name == 'testset_1'
                assert testsets[1].version_minor == 1

class TestVersionManagement:
    """Test version management functionality"""

    @pytest.mark.asyncio
    async def test_create_new_version_major(self, mock_db_manager, sample_testset):
        """Test creating a new major version"""
        mock_db, mock_conn = mock_db_manager

        # Mock current version lookup
        mock_conn.fetchrow.side_effect = [
            {  # Current version
                'version_major': 1,
                'version_minor': 2,
                'version_patch': 3
            },
            {  # New version created
                'id': 'new-uuid',
                'name': sample_testset.name,
                'description': sample_testset.description,
                'version_major': 2,
                'version_minor': 0,
                'version_patch': 0,
                'version_label': None,
                'status': 'draft',
                'domain': sample_testset.domain,
                'metadata': json.dumps(sample_testset.metadata),
                'created_at': datetime.now(timezone.utc),
                'updated_at': datetime.now(timezone.utc)
            }
        ]
        mock_conn.fetchval.return_value = 'new-uuid'
        mock_conn.fetch.return_value = []  # No examples for simplicity

        with patch('golden_testset.manager.DatabaseConnectionManager', return_value=mock_db):
            async with GoldenTestsetManager() as manager:
                new_version = await manager.create_new_version(
                    sample_testset.name,
                    VersionBump.MAJOR,
                    "Breaking changes in major update"
                )

                assert new_version.version_major == 2
                assert new_version.version_minor == 0
                assert new_version.version_patch == 0

    @pytest.mark.asyncio
    async def test_get_latest_version(self, mock_db_manager):
        """Test getting latest version of a testset"""
        mock_db, mock_conn = mock_db_manager

        mock_conn.fetchrow.return_value = {
            'version_major': 1,
            'version_minor': 2,
            'version_patch': 3
        }

        with patch('golden_testset.manager.DatabaseConnectionManager', return_value=mock_db):
            async with GoldenTestsetManager() as manager:
                version = await manager.get_latest_version("test_testset")

                assert version.major == 1
                assert version.minor == 2
                assert version.patch == 3

    @pytest.mark.asyncio
    async def test_get_version_history(self, mock_db_manager):
        """Test getting version history for a testset"""
        mock_db, mock_conn = mock_db_manager

        version_history = [
            {'version_major': 1, 'version_minor': 0, 'version_patch': 0, 'created_at': datetime.now(timezone.utc)},
            {'version_major': 1, 'version_minor': 1, 'version_patch': 0, 'created_at': datetime.now(timezone.utc)},
            {'version_major': 2, 'version_minor': 0, 'version_patch': 0, 'created_at': datetime.now(timezone.utc)}
        ]

        mock_conn.fetch.return_value = version_history

        with patch('golden_testset.manager.DatabaseConnectionManager', return_value=mock_db):
            async with GoldenTestsetManager() as manager:
                history = await manager.get_version_history("test_testset")

                assert len(history) == 3
                assert history[0]['version_major'] == 1
                assert history[2]['version_major'] == 2

class TestQualityMetrics:
    """Test quality metrics and validation functionality"""

    @pytest.mark.asyncio
    async def test_update_quality_metrics(self, mock_db_manager):
        """Test updating quality metrics for a testset"""
        mock_db, mock_conn = mock_db_manager

        mock_conn.execute.return_value = "INSERT 1"

        metrics = {
            'accuracy': 0.85,
            'precision': 0.78,
            'recall': 0.82,
            'f1_score': 0.80,
            'coverage': 0.90,
            'diversity': 0.75
        }

        with patch('golden_testset.manager.DatabaseConnectionManager', return_value=mock_db):
            async with GoldenTestsetManager() as manager:
                await manager.update_quality_metrics(
                    testset_id="test-uuid",
                    metrics=metrics,
                    validation_status=ValidationStatus.PASSED
                )

                mock_conn.execute.assert_called()

    @pytest.mark.asyncio
    async def test_get_quality_metrics(self, mock_db_manager):
        """Test retrieving quality metrics for a testset"""
        mock_db, mock_conn = mock_db_manager

        mock_metrics = {
            'testset_id': 'test-uuid',
            'accuracy': 0.85,
            'precision': 0.78,
            'recall': 0.82,
            'f1_score': 0.80,
            'coverage': 0.90,
            'diversity': 0.75,
            'validation_status': 'passed',
            'updated_at': datetime.now(timezone.utc)
        }

        mock_conn.fetchrow.return_value = mock_metrics

        with patch('golden_testset.manager.DatabaseConnectionManager', return_value=mock_db):
            async with GoldenTestsetManager() as manager:
                metrics = await manager.get_quality_metrics("test-uuid")

                assert metrics is not None
                assert metrics['accuracy'] == 0.85
                assert metrics['validation_status'] == 'passed'

class TestChangeDetectionIntegration:
    """Test integration with change detection module"""

    @pytest.mark.asyncio
    async def test_detect_changes_in_testset(self, mock_db_manager, sample_examples):
        """Test change detection for testset updates"""
        mock_db, mock_conn = mock_db_manager

        with patch('golden_testset.manager.DatabaseConnectionManager', return_value=mock_db):
            with patch('golden_testset.manager.detect_testset_changes') as mock_detect:
                # Mock change detection result
                mock_changes = [
                    ChangeRecord(
                        document_id="example_1",
                        change_type=ChangeType.MODIFIED,
                        details={'reason': 'ground_truth updated'}
                    )
                ]

                mock_result = ChangeDetectionResult(
                    total_documents=3,
                    changes_detected=1,
                    change_records=mock_changes,
                    processing_time_ms=50.0,
                    cache_hit_rate=0.0
                )

                mock_detect.return_value = mock_result

                async with GoldenTestsetManager() as manager:
                    # This would be a real method in the manager
                    result = await manager.detect_testset_changes(
                        "test_testset",
                        sample_examples
                    )

                    assert result.changes_detected == 1
                    assert result.processing_time_ms < 100.0

class TestErrorHandling:
    """Test error handling and edge cases"""

    @pytest.mark.asyncio
    async def test_database_connection_error(self):
        """Test handling of database connection errors"""
        with patch('golden_testset.manager.DatabaseManager') as mock_db_class:
            mock_db = AsyncMock()
            mock_db.initialize.side_effect = ConnectionError("Database unreachable")
            mock_db_class.return_value = mock_db

            with pytest.raises(ConnectionError):
                async with GoldenTestsetManager() as manager:
                    pass

    @pytest.mark.asyncio
    async def test_invalid_version_bump(self, mock_db_manager):
        """Test invalid version bump operations"""
        mock_db, mock_conn = mock_db_manager

        mock_conn.fetchrow.return_value = None  # No current version

        with patch('golden_testset.manager.DatabaseConnectionManager', return_value=mock_db):
            async with GoldenTestsetManager() as manager:
                with pytest.raises(ValueError, match="Testset not found"):
                    await manager.create_new_version(
                        "nonexistent_testset",
                        VersionBump.MINOR
                    )

    @pytest.mark.asyncio
    async def test_invalid_testset_data(self, mock_db_manager):
        """Test creation with invalid testset data"""
        mock_db, mock_conn = mock_db_manager

        with patch('golden_testset.manager.DatabaseConnectionManager', return_value=mock_db):
            async with GoldenTestsetManager() as manager:
                with pytest.raises(ValueError, match="Examples list cannot be empty"):
                    await manager.create_testset(
                        name="invalid_testset",
                        description="Test",
                        examples=[]  # Empty examples
                    )

class TestUtilityFunctions:
    """Test utility functions for testset management"""

    def test_create_testset_from_dict(self, sample_examples):
        """Test creating testset from dictionary"""
        testset_dict = {
            'id': 'test-123',
            'name': 'test_testset',
            'description': 'Test description',
            'version': {'major': 1, 'minor': 0, 'patch': 0},
            'status': 'approved',
            'domain': 'test',
            'examples': [
                {
                    'question': 'Test question?',
                    'ground_truth': 'Test answer',
                    'metadata': {'category': 'test'}
                }
            ],
            'metadata': {'created_by': 'test'},
            'created_at': datetime.now(timezone.utc).isoformat(),
            'updated_at': datetime.now(timezone.utc).isoformat()
        }

        testset = create_testset_from_dict(testset_dict)

        assert testset.name == 'test_testset'
        assert testset.version_major == 1
        assert len(testset.examples) == 1
        assert testset.examples[0].question == 'Test question?'

    def test_export_testset_to_dict(self, sample_testset):
        """Test exporting testset to dictionary"""
        testset_dict = export_testset_to_dict(sample_testset)

        assert testset_dict['name'] == sample_testset.name
        assert testset_dict['version']['major'] == sample_testset.version_major
        assert len(testset_dict['examples']) == len(sample_testset.examples)
        assert 'created_at' in testset_dict
        assert 'updated_at' in testset_dict

class TestPerformance:
    """Test performance requirements and benchmarks"""

    @pytest.mark.asyncio
    async def test_bulk_operations_performance(self, mock_db_manager):
        """Test performance of bulk operations"""
        mock_db, mock_conn = mock_db_manager

        # Mock fast database responses
        mock_conn.executemany.return_value = None

        with patch('golden_testset.manager.DatabaseConnectionManager', return_value=mock_db):
            async with GoldenTestsetManager() as manager:
                # Test bulk example insertion
                examples = [
                    GoldenExample(
                        question=f"Question {i}",
                        ground_truth=f"Answer {i}",
                        metadata={'index': i}
                    )
                    for i in range(100)
                ]

                start_time = asyncio.get_event_loop().time()

                # This would be a real bulk method
                await manager.bulk_insert_examples("test_testset", examples)

                elapsed = (asyncio.get_event_loop().time() - start_time) * 1000

                # Should complete bulk operations in under 500ms
                assert elapsed < 500.0

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])