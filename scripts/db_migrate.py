#!/usr/bin/env python3
"""
Database Migration Script for Golden Testset Management
Orchestrates schema generation and applies migrations with rollback support

This script implements the p1.sql.migrate task from tasks.yaml:
- Generates PostgreSQL schema for versioned golden testset management
- Applies migrations with validation and rollback support
- Tracks migration history and provides schema validation
- Integrates with Phoenix for observability

Usage:
    python scripts/db_migrate.py --migrate        # Apply migrations
    python scripts/db_migrate.py --status         # Show migration status
    python scripts/db_migrate.py --validate       # Validate schema
    python scripts/db_migrate.py --force          # Force overwrite existing schema
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import asyncpg
from generate_schema import SchemaGenerator


class MigrationManager:
    """Manages database migrations for golden testset management"""

    def __init__(self, connection_string: Optional[str] = None):
        self.connection_string = connection_string or os.environ.get(
            "POSTGRES_CONNECTION_STRING",
            "postgresql://langchain:langchain@localhost:6024/langchain"
        )
        self.migrations_dir = Path("scripts/migrations")
        self.migrations_dir.mkdir(exist_ok=True)
        self.migration_log_file = self.migrations_dir / "migration_history.json"

    async def create_migration_table(self) -> None:
        """Create migration tracking table if it doesn't exist"""
        conn = await asyncpg.connect(self.connection_string)

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                id SERIAL PRIMARY KEY,
                migration_name VARCHAR(255) NOT NULL UNIQUE,
                migration_type VARCHAR(50) NOT NULL, -- schema, data, index
                applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                applied_by VARCHAR(100),
                version VARCHAR(50),
                checksum VARCHAR(64),
                rollback_script TEXT,
                metadata JSONB DEFAULT '{}'
            )
        """)

        await conn.close()

    async def get_migration_status(self) -> List[Dict]:
        """Get current migration status"""
        conn = await asyncpg.connect(self.connection_string)

        try:
            migrations = await conn.fetch("""
                SELECT migration_name, migration_type, applied_at, applied_by, version, metadata
                FROM schema_migrations
                ORDER BY applied_at DESC
            """)

            return [dict(m) for m in migrations]
        except asyncpg.UndefinedTableError:
            return []
        finally:
            await conn.close()

    async def check_schema_exists(self) -> Tuple[bool, List[str]]:
        """Check if golden testset schema already exists"""
        conn = await asyncpg.connect(self.connection_string)

        existing_tables = await conn.fetch("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name LIKE 'golden_%'
            ORDER BY table_name
        """)

        table_names = [t['table_name'] for t in existing_tables]
        await conn.close()

        return len(table_names) > 0, table_names

    async def validate_schema(self) -> Dict[str, bool]:
        """Validate that the schema is correctly applied"""
        conn = await asyncpg.connect(self.connection_string)
        results = {}

        try:
            # Check core tables
            expected_tables = [
                'golden_testsets',
                'golden_examples',
                'testset_versions',
                'testset_quality_metrics',
                'testset_approval_log'
            ]

            for table in expected_tables:
                exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_schema = 'public' AND table_name = $1
                    )
                """, table)
                results[f"table_{table}"] = exists

            # Check views
            expected_views = [
                'testset_overview',
                'latest_testsets',
                'example_quality_view',
                'version_history',
                'quality_dashboard',
                'phoenix_integration'
            ]

            for view in expected_views:
                exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.views
                        WHERE table_schema = 'public' AND table_name = $1
                    )
                """, view)
                results[f"view_{view}"] = exists

            # Check functions
            expected_functions = [
                'format_version',
                'parse_version',
                'compare_versions',
                'get_latest_testset_version',
                'get_testset_stats'
            ]

            for func in expected_functions:
                exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM pg_proc p
                        JOIN pg_namespace n ON p.pronamespace = n.oid
                        WHERE n.nspname = 'public' AND p.proname = $1
                    )
                """, func)
                results[f"function_{func}"] = exists

            # Check indexes
            critical_indexes = [
                'idx_golden_testsets_name',
                'idx_golden_examples_testset',
                'idx_golden_examples_question_embedding',
                'idx_golden_examples_ground_truth_embedding'
            ]

            for idx in critical_indexes:
                exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM pg_indexes
                        WHERE schemaname = 'public' AND indexname = $1
                    )
                """, idx)
                results[f"index_{idx}"] = exists

            # Test sample data
            sample_testset_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM golden_testsets
                    WHERE name = 'financial_aid_baseline'
                )
            """)
            results["sample_data"] = sample_testset_exists

        finally:
            await conn.close()

        return results

    async def apply_migration(self, force: bool = False) -> bool:
        """Apply the golden testset schema migration"""
        print("🚀 Starting golden testset schema migration...")

        # Check if migration table exists
        await self.create_migration_table()

        # Check current status
        schema_exists, existing_tables = await self.check_schema_exists()
        migration_status = await self.get_migration_status()

        if schema_exists and not force:
            print(f"⚠ Schema already exists with {len(existing_tables)} tables:")
            for table in existing_tables:
                print(f"  - {table}")
            print("\nExisting migrations:")
            for migration in migration_status:
                print(f"  - {migration['migration_name']} ({migration['applied_at']})")
            print("\nUse --force to overwrite existing schema")
            return False

        # Generate and apply schema
        generator = SchemaGenerator(self.connection_string)

        # Test connection first
        if not await generator.test_connection():
            print("✗ Database connection failed")
            return False

        # Generate schema files
        print("📝 Generating schema files...")
        schemas = await generator.generate_schema()
        print(f"✓ Generated {len(schemas)} schema files")

        # Apply schema
        print("🔨 Applying schema migration...")
        success = await generator.apply_schema(force=force)

        if not success:
            print("✗ Schema application failed")
            return False

        # Record migration
        migration_name = f"golden_testset_schema_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        await self.record_migration(
            migration_name=migration_name,
            migration_type="schema",
            version="1.0.0",
            metadata={
                "description": "Initial golden testset management schema",
                "tables_created": len(existing_tables) if not schema_exists else "updated",
                "files_generated": list(schemas.keys()),
                "force_applied": force
            }
        )

        print(f"✓ Migration recorded: {migration_name}")

        # Validate migration
        print("🔍 Validating migration...")
        validation_results = await self.validate_schema()

        failed_validations = [k for k, v in validation_results.items() if not v]
        if failed_validations:
            print(f"⚠ Validation warnings for: {', '.join(failed_validations)}")
        else:
            print("✓ All schema components validated successfully")

        # Show final status
        await self.show_status()

        return True

    async def record_migration(self, migration_name: str, migration_type: str,
                             version: str, metadata: Dict) -> None:
        """Record a completed migration"""
        conn = await asyncpg.connect(self.connection_string)

        await conn.execute("""
            INSERT INTO schema_migrations (
                migration_name, migration_type, applied_by, version, metadata
            ) VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (migration_name) DO UPDATE SET
                applied_at = NOW(),
                metadata = $5
        """, migration_name, migration_type, "migration_script", version, json.dumps(metadata))

        await conn.close()

        # Also log to file
        log_entry = {
            "migration_name": migration_name,
            "migration_type": migration_type,
            "applied_at": datetime.now().isoformat(),
            "applied_by": "migration_script",
            "version": version,
            "metadata": metadata
        }

        migration_history = []
        if self.migration_log_file.exists():
            migration_history = json.loads(self.migration_log_file.read_text())

        migration_history.append(log_entry)
        self.migration_log_file.write_text(json.dumps(migration_history, indent=2))

    async def show_status(self) -> None:
        """Show current migration and schema status"""
        print("\n📊 Migration Status:")
        print("=" * 50)

        # Migration history
        migrations = await self.get_migration_status()
        if migrations:
            print(f"Applied migrations: {len(migrations)}")
            for migration in migrations[:3]:  # Show last 3
                print(f"  - {migration['migration_name']} ({migration['applied_at']})")
            if len(migrations) > 3:
                print(f"  ... and {len(migrations) - 3} more")
        else:
            print("No migrations found")

        print()

        # Schema validation
        validation_results = await self.validate_schema()
        passed = sum(1 for v in validation_results.values() if v)
        total = len(validation_results)

        print(f"Schema validation: {passed}/{total} components verified")

        # Group results by type
        by_type = {}
        for key, value in validation_results.items():
            component_type = key.split('_')[0]
            if component_type not in by_type:
                by_type[component_type] = []
            by_type[component_type].append((key, value))

        for component_type, components in by_type.items():
            passed_count = sum(1 for _, v in components if v)
            total_count = len(components)
            status = "✓" if passed_count == total_count else "⚠"
            print(f"  {status} {component_type}: {passed_count}/{total_count}")

        # Schema exists check
        schema_exists, tables = await self.check_schema_exists()
        if schema_exists:
            print(f"\n📋 Schema tables: {len(tables)}")
            for table in tables:
                print(f"  - {table}")

    async def prepare_rollback(self) -> str:
        """Prepare rollback script for the migration"""
        rollback_script = """-- Rollback script for golden testset schema
-- Generated: {timestamp}

-- Drop views (must be dropped before tables)
DROP VIEW IF EXISTS phoenix_integration CASCADE;
DROP VIEW IF EXISTS quality_dashboard CASCADE;
DROP VIEW IF EXISTS version_history CASCADE;
DROP VIEW IF EXISTS example_quality_view CASCADE;
DROP VIEW IF EXISTS latest_testsets CASCADE;
DROP VIEW IF EXISTS testset_overview CASCADE;

-- Drop functions
DROP FUNCTION IF EXISTS get_testset_stats(UUID) CASCADE;
DROP FUNCTION IF EXISTS get_latest_testset_version(VARCHAR) CASCADE;
DROP FUNCTION IF EXISTS compare_versions(INT, INT, INT, INT, INT, INT) CASCADE;
DROP FUNCTION IF EXISTS parse_version(VARCHAR) CASCADE;
DROP FUNCTION IF EXISTS format_version(INT, INT, INT, VARCHAR) CASCADE;
DROP FUNCTION IF EXISTS update_updated_at_column() CASCADE;
DROP FUNCTION IF EXISTS validate_version_progression() CASCADE;

-- Drop tables (in reverse dependency order)
DROP TABLE IF EXISTS testset_approval_log CASCADE;
DROP TABLE IF EXISTS testset_quality_metrics CASCADE;
DROP TABLE IF EXISTS testset_versions CASCADE;
DROP TABLE IF EXISTS golden_examples CASCADE;
DROP TABLE IF EXISTS golden_testsets CASCADE;

-- Drop migration tracking (optional - comment out to preserve history)
-- DROP TABLE IF EXISTS schema_migrations CASCADE;

""".format(timestamp=datetime.now().isoformat())

        rollback_file = Path("scripts/rollback_golden_testset_schema.sql")
        rollback_file.write_text(rollback_script)

        return str(rollback_file)


async def main():
    """Main entry point for migration management"""
    import argparse

    parser = argparse.ArgumentParser(description="Golden Testset Migration Manager")
    parser.add_argument("--connection", help="PostgreSQL connection string")
    parser.add_argument("--migrate", action="store_true", help="Apply migrations")
    parser.add_argument("--status", action="store_true", help="Show migration status")
    parser.add_argument("--validate", action="store_true", help="Validate schema")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing schema")
    parser.add_argument("--prepare-rollback", action="store_true", help="Generate rollback script")

    args = parser.parse_args()

    if not any([args.migrate, args.status, args.validate, args.prepare_rollback]):
        parser.print_help()
        sys.exit(1)

    manager = MigrationManager(args.connection)

    try:
        if args.status:
            await manager.show_status()

        if args.validate:
            results = await manager.validate_schema()
            failed = [k for k, v in results.items() if not v]
            if failed:
                print(f"Validation failed for: {', '.join(failed)}")
                sys.exit(1)
            else:
                print("✓ All validations passed")

        if args.prepare_rollback:
            rollback_file = await manager.prepare_rollback()
            print(f"✓ Rollback script generated: {rollback_file}")

        if args.migrate:
            success = await manager.apply_migration(args.force)
            if not success:
                sys.exit(1)
            print("\n🎉 Migration completed successfully!")
            print("\nNext steps:")
            print("  1. Test schema: python scripts/db_migrate.py --validate")
            print("  2. Check tables: psql -h localhost -p 6024 -U langchain -d langchain -c '\\dt golden_*'")
            print("  3. View sample data: SELECT * FROM testset_overview;")

    except Exception as e:
        print(f"✗ Migration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())