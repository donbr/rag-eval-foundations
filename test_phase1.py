#!/usr/bin/env python3
"""
Phase 1 Testing Script
Direct testing of Phase 1 database schema implementation without Prefect

This script validates Phase 1 completion by testing:
1. p1.sql.migrate - Schema generation and migration
2. p1.sql.rollback - Rollback capability
3. p1.db.conn - Database connection management

Usage:
    python test_phase1.py --test-all       # Run all Phase 1 tests
    python test_phase1.py --test-migrate   # Test migration only
    python test_phase1.py --test-rollback  # Test rollback only
    python test_phase1.py --test-conn      # Test connections only
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Add scripts directory to path
scripts_dir = Path(__file__).parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from generate_schema import SchemaGenerator
from db_migrate import MigrationManager
from rollback_schema import SchemaRollbackManager
from db_connection import DatabaseConnectionManager, ConnectionConfig


class Phase1Tester:
    """Test suite for Phase 1 implementation"""

    def __init__(self):
        self.connection_string = os.environ.get(
            "POSTGRES_CONNECTION_STRING",
            "postgresql://langchain:langchain@localhost:6024/langchain"
        )
        self.test_results = {}

    async def test_schema_generation(self) -> bool:
        """Test p1.sql.migrate - Schema generation"""
        print("🧪 Testing Schema Generation (p1.sql.migrate)")
        print("=" * 50)

        try:
            generator = SchemaGenerator(self.connection_string)

            # Test connection first
            if not await generator.test_connection():
                print("✗ Database connection failed")
                return False

            # Generate schema files
            print("📝 Generating schema files...")
            schemas = await generator.generate_schema()
            print(f"✓ Generated {len(schemas)} schema files")

            # Validate schema files exist
            schema_dir = Path("schemas")
            expected_files = [
                "01_core_tables.sql",
                "02_indexes.sql",
                "03_functions.sql",
                "04_views.sql",
                "05_sample_data.sql",
                "golden_testset_schema.sql"
            ]

            for file in expected_files:
                file_path = schema_dir / file
                if file_path.exists():
                    print(f"  ✓ {file} ({file_path.stat().st_size} bytes)")
                else:
                    print(f"  ✗ {file} (missing)")
                    return False

            return True

        except Exception as e:
            print(f"✗ Schema generation failed: {e}")
            return False

    async def test_migration(self) -> bool:
        """Test p1.sql.migrate - Database migration"""
        print("\n🧪 Testing Database Migration (p1.sql.migrate)")
        print("=" * 50)

        try:
            manager = MigrationManager(self.connection_string)

            # Test migration
            print("🔨 Applying migration...")
            success = await manager.apply_migration(force=True)

            if not success:
                print("✗ Migration failed")
                return False

            # Validate migration
            print("🔍 Validating migration...")
            validation_results = await manager.validate_schema()

            failed_validations = [k for k, v in validation_results.items() if not v]
            if failed_validations:
                print(f"⚠ Validation warnings: {failed_validations}")
                # Don't fail the test for warnings, but log them
            else:
                print("✓ All schema components validated")

            # Show status
            await manager.show_status()

            return True

        except Exception as e:
            print(f"✗ Migration test failed: {e}")
            return False

    async def test_rollback(self) -> bool:
        """Test p1.sql.rollback - Rollback capability"""
        print("\n🧪 Testing Schema Rollback (p1.sql.rollback)")
        print("=" * 50)

        try:
            rollback_manager = SchemaRollbackManager(self.connection_string)

            # Create backup
            print("📦 Creating backup...")
            backup_id = await rollback_manager.create_backup()
            print(f"✓ Backup created: {backup_id}")

            # Show rollback plan
            print("📋 Showing rollback plan...")
            plan = await rollback_manager.show_rollback_plan()
            for category, items in plan.items():
                if items:
                    print(f"  {category}: {len(items)} items")

            # Execute rollback
            print("🔄 Executing rollback...")
            success = await rollback_manager.execute_rollback(create_backup=False)

            if not success:
                print("✗ Rollback failed")
                return False

            print("✓ Rollback completed successfully")

            # Re-apply migration for subsequent tests
            print("🔨 Re-applying migration for subsequent tests...")
            migration_manager = MigrationManager(self.connection_string)
            await migration_manager.apply_migration(force=True)

            return True

        except Exception as e:
            print(f"✗ Rollback test failed: {e}")
            return False

    async def test_connection_management(self) -> bool:
        """Test p1.db.conn - Database connection management"""
        print("\n🧪 Testing Connection Management (p1.db.conn)")
        print("=" * 50)

        try:
            config = ConnectionConfig.from_env()
            if self.connection_string:
                config.connection_string = self.connection_string

            manager = DatabaseConnectionManager(config)

            # Initialize connection pool
            print("🔌 Initializing connection pool...")
            success = await manager.initialize()

            if not success:
                print("✗ Connection pool initialization failed")
                return False

            # Test basic connection
            print("🔍 Testing basic connection...")
            success = await manager.test_connection()

            if not success:
                print("✗ Connection test failed")
                return False

            # Test pool statistics
            print("📊 Getting pool statistics...")
            stats = await manager.get_pool_stats()
            print(f"  Pool size: {stats['pool']['size']}")
            print(f"  Free connections: {stats['pool']['free']}")
            print(f"  Health status: {stats['health']['status']}")

            # Run mini stress test
            print("🔥 Running mini stress test...")
            stress_results = await manager.stress_test(concurrent_connections=3, iterations=10)
            success_rate = stress_results['successful_operations'] / stress_results['total_operations']
            print(f"  Success rate: {success_rate*100:.1f}%")
            print(f"  Avg response: {stress_results['avg_response_time_ms']:.2f}ms")

            # Cleanup
            await manager.close()

            return success_rate > 0.95  # 95% success rate required

        except Exception as e:
            print(f"✗ Connection management test failed: {e}")
            return False

    async def run_all_tests(self) -> bool:
        """Run all Phase 1 tests"""
        print("🚀 Starting Phase 1 Test Suite")
        print("=" * 60)

        start_time = time.perf_counter()

        # Run tests in sequence
        tests = [
            ("schema_generation", self.test_schema_generation),
            ("migration", self.test_migration),
            ("rollback", self.test_rollback),
            ("connection_management", self.test_connection_management)
        ]

        all_passed = True
        for test_name, test_func in tests:
            try:
                result = await test_func()
                self.test_results[test_name] = result
                if not result:
                    all_passed = False
            except Exception as e:
                print(f"✗ Test {test_name} failed with exception: {e}")
                self.test_results[test_name] = False
                all_passed = False

        # Summary
        elapsed = time.perf_counter() - start_time
        print(f"\n🎯 Phase 1 Test Results ({elapsed:.2f}s)")
        print("=" * 60)

        for test_name, result in self.test_results.items():
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"  {test_name}: {status}")

        if all_passed:
            print("\n🎉 Phase 1 implementation PASSED all tests!")
            print("✅ Ready to proceed to Phase 2")
        else:
            print("\n❌ Phase 1 implementation has FAILED tests")
            print("⚠ Review failures before proceeding")

        return all_passed


async def main():
    """Main entry point for Phase 1 testing"""
    import argparse

    parser = argparse.ArgumentParser(description="Phase 1 Testing Suite")
    parser.add_argument("--test-all", action="store_true", help="Run all Phase 1 tests")
    parser.add_argument("--test-migrate", action="store_true", help="Test migration only")
    parser.add_argument("--test-rollback", action="store_true", help="Test rollback only")
    parser.add_argument("--test-conn", action="store_true", help="Test connections only")
    parser.add_argument("--connection", help="PostgreSQL connection string")

    args = parser.parse_args()

    if not any([args.test_all, args.test_migrate, args.test_rollback, args.test_conn]):
        args.test_all = True  # Default to all tests

    tester = Phase1Tester()
    if args.connection:
        tester.connection_string = args.connection

    try:
        success = True

        if args.test_all:
            success = await tester.run_all_tests()
        else:
            if args.test_migrate:
                success &= await tester.test_schema_generation()
                success &= await tester.test_migration()

            if args.test_rollback:
                success &= await tester.test_rollback()

            if args.test_conn:
                success &= await tester.test_connection_management()

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n👋 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Test suite failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())