#!/usr/bin/env python3
"""
Atomic Transaction Management for Golden Testset Operations

Provides ACID transaction support for complex operations that involve multiple
database tables and need rollback capabilities on failure.

Key Features:
- Atomic testset creation with examples
- Transactional version updates
- Rollback support for failed operations
- Savepoint management for complex transactions
- Integration with change detection and validation
- Performance monitoring for transaction overhead

Transaction Types:
- CREATE_TESTSET: Create testset with examples atomically
- UPDATE_VERSION: Update testset version with change tracking
- BULK_IMPORT: Import multiple testsets with validation
- QUALITY_UPDATE: Update quality metrics with validation
- APPROVAL_WORKFLOW: Multi-step approval with rollback
"""

import asyncio
import asyncpg
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, AsyncGenerator, Callable, Union
from enum import Enum
import json
import time
import uuid

# Logging setup
logger = logging.getLogger(__name__)

class TransactionType(Enum):
    """Types of transactions for monitoring and optimization"""
    CREATE_TESTSET = "create_testset"
    UPDATE_VERSION = "update_version"
    BULK_IMPORT = "bulk_import"
    QUALITY_UPDATE = "quality_update"
    APPROVAL_WORKFLOW = "approval_workflow"
    CHANGE_DETECTION = "change_detection"
    VALIDATION_PIPELINE = "validation_pipeline"

class TransactionStatus(Enum):
    """Transaction execution status"""
    PENDING = "pending"
    ACTIVE = "active"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"

@dataclass
class TransactionMetrics:
    """Transaction performance and execution metrics"""
    transaction_id: str
    transaction_type: TransactionType
    status: TransactionStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    operations_count: int = 0
    affected_rows: int = 0
    savepoints_used: int = 0
    rollback_reason: Optional[str] = None
    performance_target_ms: float = 5000.0  # 5 second default target
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SavepointContext:
    """Context for managing savepoints within transactions"""
    name: str
    created_at: datetime
    operation_count: int
    description: str = ""

class AtomicTransactionManager:
    """
    Manages atomic transactions with rollback support and performance monitoring
    """

    def __init__(self, db_manager, performance_target_ms: float = 5000.0):
        self.db_manager = db_manager
        self.performance_target_ms = performance_target_ms
        self.active_transactions: Dict[str, TransactionMetrics] = {}
        self._transaction_counter = 0

    @asynccontextmanager
    async def transaction(
        self,
        transaction_type: TransactionType = TransactionType.CREATE_TESTSET,
        isolation_level: str = "read_committed",
        readonly: bool = False,
        description: str = ""
    ) -> AsyncGenerator[asyncpg.Connection, None]:
        """
        Async context manager for atomic transactions with monitoring

        Args:
            transaction_type: Type of transaction for monitoring
            isolation_level: PostgreSQL isolation level
            readonly: Whether transaction is read-only
            description: Human-readable description

        Yields:
            Database connection within transaction context
        """
        transaction_id = f"txn_{int(time.time() * 1000)}_{self._transaction_counter}"
        self._transaction_counter += 1

        metrics = TransactionMetrics(
            transaction_id=transaction_id,
            transaction_type=transaction_type,
            status=TransactionStatus.PENDING,
            start_time=datetime.now(timezone.utc),
            performance_target_ms=self.performance_target_ms,
            metadata={"description": description, "isolation_level": isolation_level, "readonly": readonly}
        )

        self.active_transactions[transaction_id] = metrics

        try:
            async with self.db_manager.get_connection() as conn:
                # Start transaction with specified isolation level
                if isolation_level != "read_committed":
                    await conn.execute(f"SET TRANSACTION ISOLATION LEVEL {isolation_level.upper()}")

                if readonly:
                    await conn.execute("SET TRANSACTION READ ONLY")

                metrics.status = TransactionStatus.ACTIVE
                logger.info(f"Started transaction {transaction_id} ({transaction_type.value})")

                async with conn.transaction():
                    yield conn

                # Transaction committed successfully
                metrics.status = TransactionStatus.COMMITTED
                metrics.end_time = datetime.now(timezone.utc)
                metrics.duration_ms = (metrics.end_time - metrics.start_time).total_seconds() * 1000

                logger.info(f"Committed transaction {transaction_id} in {metrics.duration_ms:.2f}ms")

                # Check performance target
                if metrics.duration_ms > metrics.performance_target_ms:
                    logger.warning(
                        f"Transaction {transaction_id} exceeded target: "
                        f"{metrics.duration_ms:.2f}ms > {metrics.performance_target_ms}ms"
                    )

        except Exception as e:
            # Transaction failed - log rollback
            metrics.status = TransactionStatus.ROLLED_BACK
            metrics.end_time = datetime.now(timezone.utc)
            metrics.duration_ms = (metrics.end_time - metrics.start_time).total_seconds() * 1000
            metrics.rollback_reason = str(e)

            logger.error(f"Transaction {transaction_id} rolled back after {metrics.duration_ms:.2f}ms: {e}")
            raise

        finally:
            # Cleanup
            if transaction_id in self.active_transactions:
                del self.active_transactions[transaction_id]

    @asynccontextmanager
    async def savepoint(
        self,
        conn: asyncpg.Connection,
        name: str,
        description: str = ""
    ) -> AsyncGenerator[SavepointContext, None]:
        """
        Create a savepoint within an existing transaction

        Args:
            conn: Active database connection in transaction
            name: Savepoint name (must be unique within transaction)
            description: Human-readable description

        Yields:
            SavepointContext for managing the savepoint
        """
        savepoint_name = f"sp_{name}_{int(time.time() * 1000)}"

        context = SavepointContext(
            name=savepoint_name,
            created_at=datetime.now(timezone.utc),
            operation_count=0,
            description=description
        )

        try:
            await conn.execute(f"SAVEPOINT {savepoint_name}")
            logger.debug(f"Created savepoint {savepoint_name}: {description}")

            yield context

        except Exception as e:
            # Rollback to savepoint
            await conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
            logger.warning(f"Rolled back to savepoint {savepoint_name}: {e}")
            raise

        finally:
            # Release savepoint if transaction is still active
            try:
                await conn.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                logger.debug(f"Released savepoint {savepoint_name}")
            except:
                # Savepoint may already be released or transaction rolled back
                pass

    async def create_testset_atomic(
        self,
        testset_data: Dict[str, Any],
        examples: List[Dict[str, Any]],
        validate_quality: bool = True
    ) -> str:
        """
        Atomically create testset with all examples and validation

        Args:
            testset_data: Testset metadata and configuration
            examples: List of example dictionaries
            validate_quality: Whether to run quality validation

        Returns:
            Created testset ID

        Raises:
            ValueError: If validation fails
            Exception: If creation fails
        """
        async with self.transaction(
            TransactionType.CREATE_TESTSET,
            description=f"Create testset '{testset_data.get('name', 'unnamed')}' with {len(examples)} examples"
        ) as conn:

            # Insert testset record
            testset_id = await conn.fetchval("""
                INSERT INTO golden_testsets (
                    id, name, description, version_major, version_minor, version_patch,
                    status, domain, created_by, created_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                RETURNING id
            """,
                str(uuid.uuid4()),
                testset_data['name'],
                testset_data.get('description', ''),
                testset_data.get('version_major', 1),
                testset_data.get('version_minor', 0),
                testset_data.get('version_patch', 0),
                testset_data.get('status', 'draft'),
                testset_data.get('domain'),
                testset_data.get('created_by', 'system'),
                datetime.now(timezone.utc),
                datetime.now(timezone.utc)
            )

            # Bulk insert examples with savepoint
            async with self.savepoint(conn, "examples", "Insert all examples") as sp:
                example_records = [
                    (str(uuid.uuid4()), testset_id, ex['question'], ex['ground_truth'],
                     json.dumps(ex.get('metadata', {})), datetime.now(timezone.utc))
                    for ex in examples
                ]

                await conn.executemany("""
                    INSERT INTO golden_examples (
                        id, testset_id, question, ground_truth, metadata, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                """, example_records)

                sp.operation_count = len(example_records)

            # Quality validation with savepoint
            if validate_quality:
                async with self.savepoint(conn, "quality", "Validate quality metrics") as sp:
                    # Calculate basic quality metrics
                    stats = await conn.fetchrow("""
                        SELECT
                            COUNT(*) as example_count,
                            AVG(LENGTH(question)) as avg_question_length,
                            AVG(LENGTH(ground_truth)) as avg_answer_length
                        FROM golden_examples
                        WHERE testset_id = $1
                    """, testset_id)

                    # Quality validation rules
                    if stats['example_count'] < 3:
                        raise ValueError(f"Insufficient examples: {stats['example_count']} < 3 minimum")

                    if stats['avg_question_length'] < 10:
                        raise ValueError(f"Questions too short: avg {stats['avg_question_length']:.1f} chars")

                    if stats['avg_answer_length'] < 5:
                        raise ValueError(f"Answers too short: avg {stats['avg_answer_length']:.1f} chars")

                    # Insert quality record
                    await conn.execute("""
                        INSERT INTO testset_quality_metrics (
                            testset_id, example_count, avg_question_length, avg_answer_length,
                            validation_status, updated_at
                        ) VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                        testset_id,
                        stats['example_count'],
                        stats['avg_question_length'],
                        stats['avg_answer_length'],
                        'passed',
                        datetime.now(timezone.utc)
                    )

                    sp.operation_count = 1

            logger.info(f"Created testset {testset_id} with {len(examples)} examples")
            return testset_id

    async def update_version_atomic(
        self,
        testset_name: str,
        version_bump: str,
        changes: List[Dict[str, Any]],
        change_summary: str = ""
    ) -> str:
        """
        Atomically create new version with change tracking

        Args:
            testset_name: Name of testset to version
            version_bump: Type of version bump (major/minor/patch)
            changes: List of changes to apply
            change_summary: Summary of changes made

        Returns:
            New version testset ID
        """
        async with self.transaction(
            TransactionType.UPDATE_VERSION,
            description=f"Update {testset_name} version with {version_bump} bump"
        ) as conn:

            # Get current version
            current = await conn.fetchrow("""
                SELECT id, version_major, version_minor, version_patch
                FROM golden_testsets
                WHERE name = $1
                ORDER BY version_major DESC, version_minor DESC, version_patch DESC
                LIMIT 1
            """, testset_name)

            if not current:
                raise ValueError(f"Testset '{testset_name}' not found")

            # Calculate new version
            major, minor, patch = current['version_major'], current['version_minor'], current['version_patch']

            if version_bump == "major":
                major += 1
                minor = 0
                patch = 0
            elif version_bump == "minor":
                minor += 1
                patch = 0
            elif version_bump == "patch":
                patch += 1
            else:
                raise ValueError(f"Invalid version bump: {version_bump}")

            # Copy testset to new version
            async with self.savepoint(conn, "version_copy", "Copy testset to new version") as sp:
                new_testset_id = await conn.fetchval("""
                    INSERT INTO golden_testsets (
                        id, name, description, version_major, version_minor, version_patch,
                        status, domain, created_by, created_at, updated_at
                    )
                    SELECT
                        $1, name, description, $2, $3, $4,
                        'draft', domain, created_by, $5, $6
                    FROM golden_testsets
                    WHERE id = $7
                    RETURNING id
                """,
                    str(uuid.uuid4()),
                    major, minor, patch,
                    datetime.now(timezone.utc),
                    datetime.now(timezone.utc),
                    current['id']
                )

                # Copy examples
                await conn.execute("""
                    INSERT INTO golden_examples (
                        id, testset_id, question, ground_truth, metadata, created_at
                    )
                    SELECT
                        gen_random_uuid(), $1, question, ground_truth, metadata, $2
                    FROM golden_examples
                    WHERE testset_id = $3
                """, new_testset_id, datetime.now(timezone.utc), current['id'])

                sp.operation_count = 2

            # Record version changes
            async with self.savepoint(conn, "changes", "Record version changes") as sp:
                for change in changes:
                    await conn.execute("""
                        INSERT INTO testset_version_changes (
                            id, testset_id, change_type, change_description,
                            old_value, new_value, created_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                        str(uuid.uuid4()),
                        new_testset_id,
                        change['type'],
                        change['description'],
                        json.dumps(change.get('old_value')),
                        json.dumps(change.get('new_value')),
                        datetime.now(timezone.utc)
                    )

                sp.operation_count = len(changes)

            logger.info(f"Created version {major}.{minor}.{patch} for {testset_name}")
            return new_testset_id

    async def bulk_import_atomic(
        self,
        testsets_data: List[Dict[str, Any]],
        validate_each: bool = True,
        fail_fast: bool = False
    ) -> Dict[str, Any]:
        """
        Atomically import multiple testsets with validation

        Args:
            testsets_data: List of testset data to import
            validate_each: Whether to validate each testset
            fail_fast: Whether to stop on first failure

        Returns:
            Import results with success/failure counts
        """
        results = {
            'imported_count': 0,
            'failed_count': 0,
            'testset_ids': [],
            'errors': []
        }

        async with self.transaction(
            TransactionType.BULK_IMPORT,
            description=f"Bulk import {len(testsets_data)} testsets"
        ) as conn:

            for i, testset_data in enumerate(testsets_data):
                try:
                    async with self.savepoint(conn, f"testset_{i}", f"Import testset {i+1}") as sp:
                        examples = testset_data.pop('examples', [])

                        # Create testset
                        testset_id = await self._create_single_testset(
                            conn, testset_data, examples, validate_each
                        )

                        results['imported_count'] += 1
                        results['testset_ids'].append(testset_id)
                        sp.operation_count = 1

                except Exception as e:
                    results['failed_count'] += 1
                    error_msg = f"Failed to import testset {i+1}: {e}"
                    results['errors'].append(error_msg)

                    logger.warning(error_msg)

                    if fail_fast:
                        raise ValueError(f"Bulk import failed at testset {i+1}: {e}")

            if results['failed_count'] > 0 and not fail_fast:
                logger.warning(f"Bulk import completed with {results['failed_count']} failures")

            return results

    async def _create_single_testset(
        self,
        conn: asyncpg.Connection,
        testset_data: Dict[str, Any],
        examples: List[Dict[str, Any]],
        validate: bool
    ) -> str:
        """Helper method to create a single testset within existing transaction"""

        testset_id = str(uuid.uuid4())

        # Insert testset
        await conn.execute("""
            INSERT INTO golden_testsets (
                id, name, description, version_major, version_minor, version_patch,
                status, domain, created_by, created_at, updated_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        """,
            testset_id,
            testset_data['name'],
            testset_data.get('description', ''),
            testset_data.get('version_major', 1),
            testset_data.get('version_minor', 0),
            testset_data.get('version_patch', 0),
            testset_data.get('status', 'draft'),
            testset_data.get('domain'),
            testset_data.get('created_by', 'system'),
            datetime.now(timezone.utc),
            datetime.now(timezone.utc)
        )

        # Insert examples
        if examples:
            example_records = [
                (str(uuid.uuid4()), testset_id, ex['question'], ex['ground_truth'],
                 json.dumps(ex.get('metadata', {})), datetime.now(timezone.utc))
                for ex in examples
            ]

            await conn.executemany("""
                INSERT INTO golden_examples (
                    id, testset_id, question, ground_truth, metadata, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6)
            """, example_records)

        # Validation
        if validate and examples:
            if len(examples) < 1:
                raise ValueError("At least 1 example required")

        return testset_id

    def get_transaction_stats(self) -> Dict[str, Any]:
        """Get statistics about transaction performance"""
        return {
            'active_transactions': len(self.active_transactions),
            'active_transaction_ids': list(self.active_transactions.keys()),
            'performance_target_ms': self.performance_target_ms
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on transaction system"""
        try:
            # Test basic transaction
            async with self.transaction(
                TransactionType.CREATE_TESTSET,
                readonly=True,
                description="Health check transaction"
            ) as conn:
                result = await conn.fetchval("SELECT 1")
                assert result == 1

            return {
                'status': 'healthy',
                'transaction_system': 'operational',
                'active_transactions': len(self.active_transactions)
            }

        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'active_transactions': len(self.active_transactions)
            }


# Convenience functions for common transaction patterns

async def with_atomic_testset_creation(
    db_manager,
    testset_data: Dict[str, Any],
    examples: List[Dict[str, Any]],
    validate: bool = True
) -> str:
    """Convenience function for atomic testset creation"""
    tx_manager = AtomicTransactionManager(db_manager)
    return await tx_manager.create_testset_atomic(testset_data, examples, validate)

async def with_atomic_version_update(
    db_manager,
    testset_name: str,
    version_bump: str,
    changes: List[Dict[str, Any]]
) -> str:
    """Convenience function for atomic version updates"""
    tx_manager = AtomicTransactionManager(db_manager)
    return await tx_manager.update_version_atomic(testset_name, version_bump, changes)

async def with_atomic_bulk_import(
    db_manager,
    testsets_data: List[Dict[str, Any]],
    fail_fast: bool = False
) -> Dict[str, Any]:
    """Convenience function for atomic bulk imports"""
    tx_manager = AtomicTransactionManager(db_manager)
    return await tx_manager.bulk_import_atomic(testsets_data, fail_fast=fail_fast)