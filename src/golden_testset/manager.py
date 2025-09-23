"""
Golden Testset Manager

Core manager class for managing versioned golden testsets with PostgreSQL storage.
Provides CRUD operations, version management, and atomic transactions.

Example usage:
    async with GoldenTestsetManager() as manager:
        testset = await manager.create_testset(
            name="financial_aid_baseline",
            description="Financial aid golden testset",
            examples=examples_list
        )

        # Update to new version
        new_version = await manager.update_testset(
            testset_id=testset.id,
            examples=updated_examples,
            change_type="minor"
        )
"""

import asyncio
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import asyncpg

# Database connection utility
sys_path = Path(__file__).parent.parent.parent / "scripts"
import sys

sys.path.insert(0, str(sys_path))
from db_connection import ConnectionConfig, DatabaseConnectionManager

# Import version types
from .versioning import SemanticVersion


class ChangeType(Enum):
    """Types of changes that can trigger version bumps"""
    MAJOR = "major"      # Breaking changes, incompatible API changes
    MINOR = "minor"      # New features, backward compatible
    PATCH = "patch"      # Bug fixes, small improvements


class TestsetStatus(Enum):
    """Testset lifecycle status"""
    DRAFT = "draft"
    REVIEW = "review"
    APPROVED = "approved"
    DEPRECATED = "deprecated"


class ValidationStatus(Enum):
    """Validation pipeline status"""
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"


@dataclass
class GoldenExample:
    """Individual example in a golden testset"""
    id: str | None = None
    question: str = ""
    ground_truth: str = ""
    contexts: list[str] = field(default_factory=list)

    # RAGAS metadata
    ragas_question_type: str | None = None
    ragas_evolution_type: str | None = None
    ragas_difficulty: float | None = None

    # Retrieval metadata
    retrieval_strategy: str | None = None
    retrieval_score: float | None = None

    # Quality metrics
    context_precision: float | None = None
    context_recall: float | None = None
    faithfulness: float | None = None
    answer_relevancy: float | None = None

    # Embeddings
    question_embedding: list[float] | None = None
    ground_truth_embedding: list[float] | None = None

    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'question': self.question,
            'ground_truth': self.ground_truth,
            'contexts': self.contexts,
            'ragas_question_type': self.ragas_question_type,
            'ragas_evolution_type': self.ragas_evolution_type,
            'ragas_difficulty': self.ragas_difficulty,
            'retrieval_strategy': self.retrieval_strategy,
            'retrieval_score': self.retrieval_score,
            'context_precision': self.context_precision,
            'context_recall': self.context_recall,
            'faithfulness': self.faithfulness,
            'answer_relevancy': self.answer_relevancy,
            'question_embedding': self.question_embedding,
            'ground_truth_embedding': self.ground_truth_embedding
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'GoldenExample':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class GoldenTestset:
    """A versioned golden testset"""
    id: str | None = None
    name: str = ""
    description: str = ""
    version_major: int = 1
    version_minor: int = 0
    version_patch: int = 0
    version_label: str | None = None

    domain: str | None = None
    source_type: str = "manual"
    status: TestsetStatus = TestsetStatus.DRAFT
    validation_status: ValidationStatus = ValidationStatus.PENDING

    # Metadata
    created_at: datetime | None = None
    created_by: str = "system"
    tags: list[str] = field(default_factory=list)

    # Phoenix integration
    phoenix_project_id: str | None = None
    phoenix_experiment_id: str | None = None

    # Quality tracking
    quality_score: float | None = None

    # Examples
    examples: list[GoldenExample] = field(default_factory=list)

    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.created_at is None:
            self.created_at = datetime.now(UTC)

    @property
    def version_string(self) -> str:
        """Get semantic version string"""
        base = f"{self.version_major}.{self.version_minor}.{self.version_patch}"
        if self.version_label:
            return f"{base}-{self.version_label}"
        return base

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'version_major': self.version_major,
            'version_minor': self.version_minor,
            'version_patch': self.version_patch,
            'version_label': self.version_label,
            'domain': self.domain,
            'source_type': self.source_type,
            'status': self.status.value if isinstance(self.status, TestsetStatus) else self.status,
            'validation_status': self.validation_status.value if isinstance(self.validation_status, ValidationStatus) else self.validation_status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'created_by': self.created_by,
            'tags': self.tags,
            'phoenix_project_id': self.phoenix_project_id,
            'phoenix_experiment_id': self.phoenix_experiment_id,
            'quality_score': self.quality_score,
            'examples': [ex.to_dict() for ex in self.examples]
        }


class GoldenTestsetManager:
    """
    Manager for golden testsets with version control and atomic operations

    Features:
    - CRUD operations for testsets and examples
    - Semantic versioning with automatic bumping
    - Atomic transactions
    - Quality tracking and validation
    - Phoenix experiment integration
    """

    def __init__(self, connection_string: str | None = None):
        self.connection_string = connection_string or os.environ.get(
            "DATABASE_URL",
            "postgresql://langchain:langchain@localhost:6024/langchain"
        )
        self.db_manager = None
        self._connection = None

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()

    async def connect(self) -> None:
        """Establish database connection"""
        config = ConnectionConfig(
            host="localhost",
            port=6024,
            database="langchain",
            user="langchain",
            password="langchain",
            min_size=5,
            max_size=20
        )

        self.db_manager = DatabaseConnectionManager(config)
        await self.db_manager.initialize()

    async def disconnect(self) -> None:
        """Close database connection"""
        if self.db_manager:
            await self.db_manager.close()

    def get_connection(self):
        """Get database connection from pool (async context manager)"""
        if not self.db_manager:
            raise RuntimeError("Database manager not initialized. Use 'async with manager' or call connect() first.")
        return self.db_manager.get_connection()

    def transaction(self):
        """Get database transaction (async context manager)"""
        if not self.db_manager:
            raise RuntimeError("Database manager not initialized. Use 'async with manager' or call connect() first.")
        return self.db_manager.transaction()

    # =========================================================================
    # CRUD Operations
    # =========================================================================

    async def create_testset(
        self,
        name: str,
        description: str,
        examples: list[GoldenExample],
        domain: str | None = None,
        source_type: str = "manual",
        created_by: str = "system",
        phoenix_project_id: str | None = None,
        phoenix_experiment_id: str | None = None,
        tags: list[str] | None = None,
        version_label: str | None = None
    ) -> GoldenTestset:
        """
        Create a new golden testset

        Args:
            name: Unique name for the testset
            description: Human-readable description
            examples: List of golden examples
            domain: Domain/category (e.g., 'financial_aid')
            source_type: How testset was created ('manual', 'ragas', 'imported')
            created_by: Creator identifier
            phoenix_project_id: Phoenix project for tracking
            phoenix_experiment_id: Phoenix experiment for tracking
            tags: Optional tags for organization
            version_label: Optional label for version (e.g., 'beta', 'rc1')

        Returns:
            Created GoldenTestset instance

        Raises:
            ValueError: If name already exists
            ValidationError: If examples are invalid
        """
        async with self.transaction() as conn:
            # Check if name already exists
            existing = await conn.fetchval("""
                SELECT id FROM golden_testsets WHERE name = $1
            """, name)

            if existing:
                raise ValueError(f"Testset with name '{name}' already exists")

            # Create testset
            testset = GoldenTestset(
                name=name,
                description=description,
                domain=domain,
                source_type=source_type,
                created_by=created_by,
                phoenix_project_id=phoenix_project_id,
                phoenix_experiment_id=phoenix_experiment_id,
                tags=tags or [],
                version_label=version_label,
                examples=examples
            )

            # Insert testset record
            await conn.execute("""
                INSERT INTO golden_testsets (
                    id, name, description, version_major, version_minor, version_patch, version_label,
                    domain, source_type, status, validation_status, created_at, created_by,
                    phoenix_project_id, phoenix_experiment_id, quality_score
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
            """,
                testset.id, testset.name, testset.description,
                testset.version_major, testset.version_minor, testset.version_patch, testset.version_label,
                testset.domain, testset.source_type, testset.status.value, testset.validation_status.value,
                testset.created_at, testset.created_by,
                testset.phoenix_project_id, testset.phoenix_experiment_id, testset.quality_score
            )

            # Insert examples
            for example in examples:
                await self._insert_example(conn, testset.id, example)

            # Create version record
            await self._create_version_record(
                conn, testset.id, testset.version_string,
                ChangeType.MAJOR, "Initial testset creation",
                len(examples), 0, 0, created_by
            )

            # Calculate and update quality score
            quality_score = await self._calculate_quality_score(conn, testset.id)
            await conn.execute("""
                UPDATE golden_testsets SET quality_score = $1 WHERE id = $2
            """, quality_score, testset.id)
            testset.quality_score = quality_score

            return testset

    async def get_testset(
        self,
        testset_id: str | None = None,
        name: str | None = None,
        version: str | None = None,
        include_examples: bool = True
    ) -> GoldenTestset | None:
        """
        Retrieve a golden testset by ID, name, or name+version

        Args:
            testset_id: UUID of testset
            name: Name of testset (gets latest version if version not specified)
            version: Specific version string (requires name)
            include_examples: Whether to load examples

        Returns:
            GoldenTestset if found, None otherwise
        """
        if not testset_id and not name:
            raise ValueError("Must provide either testset_id or name")

        async with self.get_connection() as conn:
            if testset_id:
                # Get by ID
                query = """
                    SELECT * FROM golden_testsets WHERE id = $1
                """
                row = await conn.fetchrow(query, testset_id)
            elif version:
                # Get by name and version
                parts = version.split('.')
                if len(parts) < 3:
                    raise ValueError("Version must be in format 'major.minor.patch'")

                major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
                label = parts[3] if len(parts) > 3 else None

                query = """
                    SELECT * FROM golden_testsets
                    WHERE name = $1 AND version_major = $2 AND version_minor = $3 AND version_patch = $4
                    AND ($5::text IS NULL OR version_label = $5)
                """
                row = await conn.fetchrow(query, name, major, minor, patch, label)
            else:
                # Get latest version by name
                query = """
                    SELECT * FROM golden_testsets
                    WHERE name = $1
                    ORDER BY version_major DESC, version_minor DESC, version_patch DESC
                    LIMIT 1
                """
                row = await conn.fetchrow(query, name)

            if not row:
                return None

            # Convert row to testset
            testset = self._row_to_testset(row)

            # Load examples if requested
            if include_examples:
                examples = await self._load_examples(conn, testset.id)
                testset.examples = examples

            return testset

    async def list_testsets(
        self,
        domain: str | None = None,
        status: TestsetStatus | None = None,
        created_by: str | None = None,
        include_examples: bool = False,
        limit: int = 100,
        offset: int = 0
    ) -> list[GoldenTestset]:
        """
        List golden testsets with optional filtering

        Args:
            domain: Filter by domain
            status: Filter by status
            created_by: Filter by creator
            include_examples: Whether to load examples for each testset
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of matching testsets
        """
        conn = await self.get_connection()
        try:
            # Build query with filters
            conditions = []
            params = []
            param_idx = 1

            if domain:
                conditions.append(f"domain = ${param_idx}")
                params.append(domain)
                param_idx += 1

            if status:
                conditions.append(f"status = ${param_idx}")
                params.append(status.value if isinstance(status, TestsetStatus) else status)
                param_idx += 1

            if created_by:
                conditions.append(f"created_by = ${param_idx}")
                params.append(created_by)
                param_idx += 1

            where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

            query = f"""
                SELECT * FROM golden_testsets
                {where_clause}
                ORDER BY created_at DESC
                LIMIT ${param_idx} OFFSET ${param_idx + 1}
            """
            params.extend([limit, offset])

            rows = await conn.fetch(query, *params)
            testsets = []

            for row in rows:
                testset = self._row_to_testset(row)

                if include_examples:
                    examples = await self._load_examples(conn, testset.id)
                    testset.examples = examples

                testsets.append(testset)

            return testsets

        finally:
            await self.release_connection(conn)

    async def update_testset(
        self,
        testset_id: str,
        examples: list[GoldenExample] | None = None,
        description: str | None = None,
        change_type: ChangeType = ChangeType.PATCH,
        change_summary: str | None = None,
        updated_by: str = "system",
        phoenix_experiment_id: str | None = None,
        tags: list[str] | None = None
    ) -> GoldenTestset:
        """
        Update a testset and create a new version

        Args:
            testset_id: ID of testset to update
            examples: New examples (if provided)
            description: Updated description
            change_type: Type of change for version bumping
            change_summary: Description of changes
            updated_by: User making the update
            phoenix_experiment_id: Updated Phoenix experiment ID
            tags: Updated tags

        Returns:
            Updated testset with new version

        Raises:
            ValueError: If testset not found
        """
        conn = await self.get_connection()
        try:
            async with conn.transaction():
                # Get current testset
                current = await self.get_testset(testset_id=testset_id, include_examples=True)
                if not current:
                    raise ValueError(f"Testset not found: {testset_id}")

                # Calculate new version
                new_major, new_minor, new_patch = self._bump_version(
                    current.version_major, current.version_minor, current.version_patch, change_type
                )

                # Create new testset record
                new_testset = GoldenTestset(
                    id=str(uuid.uuid4()),  # New ID for new version
                    name=current.name,
                    description=description or current.description,
                    version_major=new_major,
                    version_minor=new_minor,
                    version_patch=new_patch,
                    version_label=current.version_label,
                    domain=current.domain,
                    source_type=current.source_type,
                    status=TestsetStatus.DRAFT,  # New versions start as draft
                    validation_status=ValidationStatus.PENDING,
                    created_by=updated_by,
                    phoenix_project_id=current.phoenix_project_id,
                    phoenix_experiment_id=phoenix_experiment_id or current.phoenix_experiment_id,
                    examples=examples or current.examples
                )

                # Insert new testset
                await conn.execute("""
                    INSERT INTO golden_testsets (
                        id, name, description, version_major, version_minor, version_patch, version_label,
                        domain, source_type, status, validation_status, created_at, created_by,
                        phoenix_project_id, phoenix_experiment_id, quality_score
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                """,
                    new_testset.id, new_testset.name, new_testset.description,
                    new_testset.version_major, new_testset.version_minor, new_testset.version_patch, new_testset.version_label,
                    new_testset.domain, new_testset.source_type, new_testset.status.value, new_testset.validation_status.value,
                    new_testset.created_at, new_testset.created_by,
                    new_testset.phoenix_project_id, new_testset.phoenix_experiment_id, new_testset.quality_score
                )

                # Insert new examples
                for example in new_testset.examples:
                    await self._insert_example(conn, new_testset.id, example)

                # Calculate change statistics
                examples_added, examples_modified, examples_removed = self._calculate_changes(
                    current.examples, new_testset.examples
                )

                # Create version record
                await self._create_version_record(
                    conn, new_testset.id, new_testset.version_string,
                    change_type, change_summary or f"Updated via {change_type.value}",
                    examples_added, examples_modified, examples_removed, updated_by
                )

                # Calculate and update quality score
                quality_score = await self._calculate_quality_score(conn, new_testset.id)
                await conn.execute("""
                    UPDATE golden_testsets SET quality_score = $1 WHERE id = $2
                """, quality_score, new_testset.id)
                new_testset.quality_score = quality_score

                return new_testset

        finally:
            await self.release_connection(conn)

    async def delete_testset(
        self,
        testset_id: str,
        force: bool = False
    ) -> bool:
        """
        Delete a testset and all its examples

        Args:
            testset_id: ID of testset to delete
            force: If True, delete even if testset is approved

        Returns:
            True if deleted, False if not found

        Raises:
            ValueError: If trying to delete approved testset without force
        """
        conn = await self.get_connection()
        try:
            async with conn.transaction():
                # Get testset info
                testset = await conn.fetchrow("""
                    SELECT status FROM golden_testsets WHERE id = $1
                """, testset_id)

                if not testset:
                    return False

                # Check if approved
                if testset['status'] == 'approved' and not force:
                    raise ValueError("Cannot delete approved testset without force=True")

                # Delete examples first (foreign key constraint)
                await conn.execute("""
                    DELETE FROM golden_examples WHERE testset_id = $1
                """, testset_id)

                # Delete version records
                await conn.execute("""
                    DELETE FROM testset_versions WHERE testset_id = $1
                """, testset_id)

                # Delete quality metrics
                await conn.execute("""
                    DELETE FROM testset_quality_metrics WHERE testset_id = $1
                """, testset_id)

                # Delete approval logs
                await conn.execute("""
                    DELETE FROM testset_approval_log WHERE testset_id = $1
                """, testset_id)

                # Delete testset
                await conn.execute("""
                    DELETE FROM golden_testsets WHERE id = $1
                """, testset_id)

                return True

        finally:
            await self.release_connection(conn)

    # =========================================================================
    # Version Management
    # =========================================================================

    async def get_versions(self, name: str) -> list[dict[str, Any]]:
        """Get all versions of a testset by name"""
        conn = await self.get_connection()
        try:
            rows = await conn.fetch("""
                SELECT
                    id, version_major, version_minor, version_patch, version_label,
                    status, validation_status, created_at, created_by, quality_score
                FROM golden_testsets
                WHERE name = $1
                ORDER BY version_major DESC, version_minor DESC, version_patch DESC
            """, name)

            return [dict(row) for row in rows]

        finally:
            await self.release_connection(conn)

    async def approve_testset(
        self,
        testset_id: str,
        reviewer: str,
        comments: str | None = None,
        checklist: dict[str, bool] | None = None
    ) -> bool:
        """
        Approve a testset for production use

        Args:
            testset_id: ID of testset to approve
            reviewer: Who is approving
            comments: Review comments
            checklist: Review checklist items

        Returns:
            True if approved successfully
        """
        conn = await self.get_connection()
        try:
            async with conn.transaction():
                # Update testset status
                result = await conn.execute("""
                    UPDATE golden_testsets
                    SET status = 'approved', validation_status = 'passed'
                    WHERE id = $1 AND status != 'deprecated'
                """, testset_id)

                if result == "UPDATE 0":
                    return False

                # Log approval
                await conn.execute("""
                    INSERT INTO testset_approval_log (
                        testset_id, reviewer, review_status, review_comments, review_checklist
                    ) VALUES ($1, $2, $3, $4, $5)
                """, testset_id, reviewer, 'approved', comments, json.dumps(checklist or {}))

                return True

        finally:
            await self.release_connection(conn)

    # =========================================================================
    # Helper Methods
    # =========================================================================

    async def _insert_example(self, conn: asyncpg.Connection, testset_id: str, example: GoldenExample) -> None:
        """Insert a single example into the database"""
        await conn.execute("""
            INSERT INTO golden_examples (
                id, testset_id, question, ground_truth, contexts,
                ragas_question_type, ragas_evolution_type, ragas_difficulty,
                retrieval_strategy, retrieval_score,
                context_precision, context_recall, faithfulness, answer_relevancy,
                question_embedding, ground_truth_embedding
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
        """,
            example.id, testset_id, example.question, example.ground_truth, example.contexts,
            example.ragas_question_type, example.ragas_evolution_type, example.ragas_difficulty,
            example.retrieval_strategy, example.retrieval_score,
            example.context_precision, example.context_recall, example.faithfulness, example.answer_relevancy,
            example.question_embedding, example.ground_truth_embedding
        )

    async def _load_examples(self, conn: asyncpg.Connection, testset_id: str) -> list[GoldenExample]:
        """Load all examples for a testset"""
        rows = await conn.fetch("""
            SELECT * FROM golden_examples WHERE testset_id = $1 ORDER BY created_at
        """, testset_id)

        examples = []
        for row in rows:
            example = GoldenExample(
                id=row['id'],
                question=row['question'],
                ground_truth=row['ground_truth'],
                contexts=row['contexts'] or [],
                ragas_question_type=row['ragas_question_type'],
                ragas_evolution_type=row['ragas_evolution_type'],
                ragas_difficulty=row['ragas_difficulty'],
                retrieval_strategy=row['retrieval_strategy'],
                retrieval_score=row['retrieval_score'],
                context_precision=row['context_precision'],
                context_recall=row['context_recall'],
                faithfulness=row['faithfulness'],
                answer_relevancy=row['answer_relevancy'],
                question_embedding=row['question_embedding'],
                ground_truth_embedding=row['ground_truth_embedding']
            )
            examples.append(example)

        return examples

    def _row_to_testset(self, row: asyncpg.Record) -> GoldenTestset:
        """Convert database row to GoldenTestset object"""
        return GoldenTestset(
            id=row['id'],
            name=row['name'],
            description=row['description'],
            version_major=row['version_major'],
            version_minor=row['version_minor'],
            version_patch=row['version_patch'],
            version_label=row['version_label'],
            domain=row['domain'],
            source_type=row['source_type'],
            status=TestsetStatus(row['status']),
            validation_status=ValidationStatus(row['validation_status']),
            created_at=row['created_at'],
            created_by=row['created_by'],
            phoenix_project_id=row['phoenix_project_id'],
            phoenix_experiment_id=row['phoenix_experiment_id'],
            quality_score=row['quality_score']
        )

    def _bump_version(
        self,
        current_major: int,
        current_minor: int,
        current_patch: int,
        change_type: ChangeType
    ) -> tuple[int, int, int]:
        """Calculate new version numbers based on change type"""
        if change_type == ChangeType.MAJOR:
            return current_major + 1, 0, 0
        elif change_type == ChangeType.MINOR:
            return current_major, current_minor + 1, 0
        else:  # PATCH
            return current_major, current_minor, current_patch + 1

    def _calculate_changes(
        self,
        old_examples: list[GoldenExample],
        new_examples: list[GoldenExample]
    ) -> tuple[int, int, int]:
        """Calculate added, modified, and removed examples"""
        old_ids = {ex.id for ex in old_examples}
        new_ids = {ex.id for ex in new_examples}

        added = len(new_ids - old_ids)
        removed = len(old_ids - new_ids)

        # Check for modifications (same ID but different content)
        modified = 0
        old_by_id = {ex.id: ex for ex in old_examples}
        new_by_id = {ex.id: ex for ex in new_examples}

        for ex_id in old_ids & new_ids:
            old_ex = old_by_id[ex_id]
            new_ex = new_by_id[ex_id]

            # Compare content (simplified)
            if (old_ex.question != new_ex.question or
                old_ex.ground_truth != new_ex.ground_truth or
                old_ex.contexts != new_ex.contexts):
                modified += 1

        return added, modified, removed

    async def _create_version_record(
        self,
        conn: asyncpg.Connection,
        testset_id: str,
        version_string: str,
        change_type: ChangeType,
        change_summary: str,
        examples_added: int,
        examples_modified: int,
        examples_removed: int,
        created_by: str
    ) -> None:
        """Create a version history record"""
        await conn.execute("""
            INSERT INTO testset_versions (
                id, testset_id, version_string, change_type, change_summary,
                examples_added, examples_modified, examples_removed, created_by
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """,
            str(uuid.uuid4()), testset_id, version_string, change_type.value, change_summary,
            examples_added, examples_modified, examples_removed, created_by
        )

    async def _calculate_quality_score(self, conn: asyncpg.Connection, testset_id: str) -> float:
        """Calculate overall quality score for a testset"""
        # Get average quality metrics from examples
        result = await conn.fetchrow("""
            SELECT
                AVG(context_precision) as avg_precision,
                AVG(context_recall) as avg_recall,
                AVG(faithfulness) as avg_faithfulness,
                AVG(answer_relevancy) as avg_relevancy,
                COUNT(*) as example_count
            FROM golden_examples
            WHERE testset_id = $1
            AND context_precision IS NOT NULL
        """, testset_id)

        if not result or result['example_count'] == 0:
            return 0.0

        # Weighted average of quality metrics
        precision = result['avg_precision'] or 0.0
        recall = result['avg_recall'] or 0.0
        faithfulness = result['avg_faithfulness'] or 0.0
        relevancy = result['avg_relevancy'] or 0.0

        # Equal weighting for now
        quality_score = (precision + recall + faithfulness + relevancy) / 4.0

        return round(quality_score, 3)

    async def detect_testset_changes(self, testset_name: str, current_examples: list[GoldenExample]) -> Any:
        """Detect changes in a testset using change detection module"""
        from .change_detector import create_baseline_hashes, detect_testset_changes

        # Get current testset for baseline
        testset = await self.get_testset(testset_name)
        if not testset:
            raise ValueError(f"Testset '{testset_name}' not found")

        # Convert to dict format for change detection
        baseline_examples = [ex.to_dict() for ex in testset.examples]
        current_examples_dicts = [ex.to_dict() for ex in current_examples]

        # Create baseline hashes
        baseline_hashes = create_baseline_hashes(testset_name, baseline_examples)

        # Detect changes
        return await detect_testset_changes(testset_name, current_examples_dicts, baseline_hashes)

    async def bulk_insert_examples(self, testset_name: str, examples: list[GoldenExample]) -> None:
        """Bulk insert examples for performance testing"""
        async with self.get_connection() as conn:
            testset = await self.get_testset(testset_name)
            if not testset:
                raise ValueError(f"Testset '{testset_name}' not found")

            # Use executemany for bulk operations
            example_data = [
                (str(uuid.uuid4()), testset.id, ex.question, ex.ground_truth,
                 json.dumps(ex.metadata or {}), datetime.now(UTC))
                for ex in examples
            ]

            await conn.executemany("""
                INSERT INTO golden_examples (id, testset_id, question, ground_truth, metadata, created_at)
                VALUES ($1, $2, $3, $4, $5, $6)
            """, example_data)

    async def create_new_version(self, testset_name: str, bump_type: str, description: str = "") -> GoldenTestset:
        """Create a new version of an existing testset"""
        from .versioning import VersionManager

        async with self.get_connection() as conn:
            # Get current version
            current_version_row = await conn.fetchrow("""
                SELECT version_major, version_minor, version_patch
                FROM golden_testsets
                WHERE name = $1
                ORDER BY version_major DESC, version_minor DESC, version_patch DESC
                LIMIT 1
            """, testset_name)

            if not current_version_row:
                raise ValueError(f"Testset '{testset_name}' not found")

            current_version = SemanticVersion(
                major=current_version_row['version_major'],
                minor=current_version_row['version_minor'],
                patch=current_version_row['version_patch']
            )

            # Bump version
            from .versioning import VersionBump
            version_manager = VersionManager()
            bump_enum = VersionBump(bump_type) if isinstance(bump_type, str) else bump_type
            new_version = version_manager.bump_version(current_version, bump_enum)

            # Get existing testset data
            existing_testset = await self.get_testset(testset_name)
            if not existing_testset:
                raise ValueError(f"Testset '{testset_name}' not found")

            # Create new version with updated version numbers
            new_testset = GoldenTestset(
                id=str(uuid.uuid4()),
                name=existing_testset.name,
                description=description or existing_testset.description,
                version_major=new_version.major,
                version_minor=new_version.minor,
                version_patch=new_version.patch,
                version_label=new_version.label,
                status=TestsetStatus.DRAFT,
                domain=existing_testset.domain,
                examples=existing_testset.examples.copy(),
                metadata=existing_testset.metadata.copy(),
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC)
            )

            # Insert new version
            await conn.fetchval("""
                INSERT INTO golden_testsets (
                    id, name, description, version_major, version_minor, version_patch,
                    version_label, status, domain, metadata, created_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                RETURNING id
            """,
                new_testset.id, new_testset.name, new_testset.description,
                new_testset.version_major, new_testset.version_minor, new_testset.version_patch,
                new_testset.version_label, new_testset.status.value, new_testset.domain,
                json.dumps(new_testset.metadata), new_testset.created_at, new_testset.updated_at
            )

            # Copy examples
            for example in new_testset.examples:
                await self._insert_example(conn, new_testset.id, example)

            return new_testset

    async def get_latest_version(self, testset_name: str) -> SemanticVersion | None:
        """Get the latest version of a testset"""
        async with self.get_connection() as conn:
            version_row = await conn.fetchrow("""
                SELECT version_major, version_minor, version_patch
                FROM golden_testsets
                WHERE name = $1
                ORDER BY version_major DESC, version_minor DESC, version_patch DESC
                LIMIT 1
            """, testset_name)

            if not version_row:
                return None

            return SemanticVersion(
                major=version_row['version_major'],
                minor=version_row['version_minor'],
                patch=version_row['version_patch']
            )

    async def get_version_history(self, testset_name: str) -> list[dict[str, Any]]:
        """Get version history for a testset"""
        async with self.get_connection() as conn:
            rows = await conn.fetch("""
                SELECT version_major, version_minor, version_patch, created_at
                FROM golden_testsets
                WHERE name = $1
                ORDER BY version_major DESC, version_minor DESC, version_patch DESC
            """, testset_name)

            return [dict(row) for row in rows]

    async def update_quality_metrics(self, testset_id: str, metrics: dict[str, float], validation_status: ValidationStatus) -> None:
        """Update quality metrics for a testset"""
        async with self.get_connection() as conn:
            await conn.execute("""
                INSERT INTO testset_quality_metrics (
                    testset_id, accuracy, precision, recall, f1_score, coverage, diversity,
                    validation_status, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (testset_id) DO UPDATE SET
                    accuracy = EXCLUDED.accuracy,
                    precision = EXCLUDED.precision,
                    recall = EXCLUDED.recall,
                    f1_score = EXCLUDED.f1_score,
                    coverage = EXCLUDED.coverage,
                    diversity = EXCLUDED.diversity,
                    validation_status = EXCLUDED.validation_status,
                    updated_at = EXCLUDED.updated_at
            """,
                testset_id,
                metrics.get('accuracy', 0.0),
                metrics.get('precision', 0.0),
                metrics.get('recall', 0.0),
                metrics.get('f1_score', 0.0),
                metrics.get('coverage', 0.0),
                metrics.get('diversity', 0.0),
                validation_status.value,
                datetime.now(UTC)
            )

    async def get_quality_metrics(self, testset_id: str) -> dict[str, Any] | None:
        """Get quality metrics for a testset"""
        async with self.get_connection() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM testset_quality_metrics WHERE testset_id = $1
            """, testset_id)

            return dict(row) if row else None


# =========================================================================
# Convenience Functions
# =========================================================================

def create_testset_from_dict(data: dict[str, Any]) -> GoldenTestset:
    """Create testset from dictionary (e.g., loaded from JSON)"""
    examples_data = data.pop('examples', [])
    examples = [GoldenExample.from_dict(ex) for ex in examples_data]

    testset = GoldenTestset(**data)
    testset.examples = examples

    return testset


def export_testset_to_dict(testset: GoldenTestset) -> dict[str, Any]:
    """Export testset to dictionary for serialization"""
    return testset.to_dict()


# Example usage and testing
if __name__ == "__main__":
    async def test_manager():
        """Test the manager functionality"""
        async with GoldenTestsetManager() as manager:
            # Create sample examples
            examples = [
                GoldenExample(
                    question="What are the eligibility requirements for Federal Pell Grants?",
                    ground_truth="Federal Pell Grants are available to undergraduate students who demonstrate exceptional financial need and have not earned a bachelor's degree.",
                    contexts=["Federal Pell Grants are need-based grants for undergraduate students."],
                    ragas_question_type="simple",
                    ragas_evolution_type="reasoning",
                    ragas_difficulty=2.5,
                    retrieval_strategy="semantic_chunking",
                    retrieval_score=0.8842,
                    context_precision=0.89,
                    context_recall=0.76,
                    faithfulness=0.92,
                    answer_relevancy=0.88
                )
            ]

            # Create testset
            testset = await manager.create_testset(
                name="test_financial_aid",
                description="Test financial aid testset",
                examples=examples,
                domain="financial_aid",
                source_type="manual"
            )

            print(f"Created testset: {testset.name} v{testset.version_string}")
            print(f"Quality score: {testset.quality_score}")

    # Run test
    asyncio.run(test_manager())
