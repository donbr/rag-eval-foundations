#!/usr/bin/env python3
"""
Schema Assertion Script for Golden Testset Management
Validates that all required database objects exist and are properly configured

Usage:
    python assert_schema.py --require tables=golden_testsets,golden_examples
    python assert_schema.py --require views=testset_overview,latest_testsets
    python assert_schema.py --check all
"""

import asyncio
import asyncpg
import argparse
import json
import sys
from typing import List, Dict, Any, Optional
from pathlib import Path
import os

class SchemaAsserter:
    """Validates database schema requirements"""

    def __init__(self, connection_string: Optional[str] = None):
        self.connection_string = connection_string or os.environ.get(
            "DATABASE_URL",
            "postgresql://langchain:langchain@localhost:6024/langchain"
        )
        self.errors = []
        self.warnings = []

    async def connect(self) -> asyncpg.Connection:
        """Establish database connection"""
        try:
            return await asyncpg.connect(self.connection_string)
        except Exception as e:
            print(f"❌ Failed to connect to database: {e}")
            sys.exit(1)

    async def check_tables(self, required_tables: List[str]) -> bool:
        """Verify required tables exist"""
        conn = await self.connect()
        try:
            # Get all tables in public schema
            existing = await conn.fetch("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_type = 'BASE TABLE'
            """)

            existing_tables = {row['table_name'] for row in existing}

            # Check each required table
            all_exist = True
            for table in required_tables:
                if table in existing_tables:
                    print(f"  ✓ Table '{table}' exists")

                    # Check table structure
                    columns = await conn.fetch("""
                        SELECT column_name, data_type, is_nullable
                        FROM information_schema.columns
                        WHERE table_schema = 'public' AND table_name = $1
                        ORDER BY ordinal_position
                    """, table)

                    if columns:
                        print(f"    → {len(columns)} columns defined")
                else:
                    print(f"  ✗ Table '{table}' NOT FOUND")
                    self.errors.append(f"Missing table: {table}")
                    all_exist = False

            return all_exist

        finally:
            await conn.close()

    async def check_views(self, required_views: List[str]) -> bool:
        """Verify required views exist"""
        conn = await self.connect()
        try:
            # Get all views in public schema
            existing = await conn.fetch("""
                SELECT table_name
                FROM information_schema.views
                WHERE table_schema = 'public'
            """)

            existing_views = {row['table_name'] for row in existing}

            # Check each required view
            all_exist = True
            for view in required_views:
                if view in existing_views:
                    print(f"  ✓ View '{view}' exists")

                    # Test view is queryable
                    try:
                        count = await conn.fetchval(f"SELECT COUNT(*) FROM {view}")
                        print(f"    → {count} rows accessible")
                    except Exception as e:
                        print(f"    ⚠ View exists but query failed: {e}")
                        self.warnings.append(f"View '{view}' query failed")
                else:
                    print(f"  ✗ View '{view}' NOT FOUND")
                    self.errors.append(f"Missing view: {view}")
                    all_exist = False

            return all_exist

        finally:
            await conn.close()

    async def check_functions(self, required_functions: List[str]) -> bool:
        """Verify required functions exist"""
        conn = await self.connect()
        try:
            # Get all functions in public schema
            existing = await conn.fetch("""
                SELECT p.proname as function_name
                FROM pg_proc p
                JOIN pg_namespace n ON p.pronamespace = n.oid
                WHERE n.nspname = 'public'
            """)

            existing_functions = {row['function_name'] for row in existing}

            # Check each required function
            all_exist = True
            for func in required_functions:
                if func in existing_functions:
                    print(f"  ✓ Function '{func}()' exists")
                else:
                    print(f"  ✗ Function '{func}()' NOT FOUND")
                    self.errors.append(f"Missing function: {func}()")
                    all_exist = False

            return all_exist

        finally:
            await conn.close()

    async def check_indexes(self, table_name: Optional[str] = None) -> bool:
        """Verify indexes are properly configured"""
        conn = await self.connect()
        try:
            if table_name:
                # Check indexes for specific table
                indexes = await conn.fetch("""
                    SELECT
                        indexname,
                        indexdef,
                        tablename
                    FROM pg_indexes
                    WHERE schemaname = 'public'
                    AND tablename = $1
                """, table_name)
            else:
                # Check all indexes
                indexes = await conn.fetch("""
                    SELECT
                        indexname,
                        indexdef,
                        tablename
                    FROM pg_indexes
                    WHERE schemaname = 'public'
                    AND tablename LIKE 'golden_%'
                """)

            if not indexes:
                print(f"  ⚠ No indexes found for {'table ' + table_name if table_name else 'golden_* tables'}")
                self.warnings.append("No indexes found")
                return False

            print(f"  ✓ Found {len(indexes)} indexes")

            # Check for critical indexes
            critical_indexes = [
                'golden_testsets_pkey',
                'golden_examples_pkey',
                'idx_golden_testsets_name',
                'idx_golden_testsets_version',
                'idx_golden_examples_testset'
            ]

            index_names = {idx['indexname'] for idx in indexes}

            for critical in critical_indexes:
                if critical in index_names:
                    print(f"    ✓ Critical index '{critical}' exists")
                else:
                    print(f"    ⚠ Critical index '{critical}' missing")
                    self.warnings.append(f"Missing critical index: {critical}")

            return True

        finally:
            await conn.close()

    async def check_constraints(self) -> bool:
        """Verify constraints are properly configured"""
        conn = await self.connect()
        try:
            # Check constraints
            constraints = await conn.fetch("""
                SELECT
                    conname as constraint_name,
                    contype as constraint_type,
                    conrelid::regclass as table_name
                FROM pg_constraint
                WHERE connamespace = 'public'::regnamespace
                AND conrelid::regclass::text LIKE 'golden_%'
            """)

            if not constraints:
                print(f"  ⚠ No constraints found")
                self.warnings.append("No constraints found")
                return False

            print(f"  ✓ Found {len(constraints)} constraints")

            # Count constraint types
            types = {}
            for c in constraints:
                ctype = c['constraint_type']
                types[ctype] = types.get(ctype, 0) + 1

            print(f"    → Primary keys: {types.get('p', 0)}")
            print(f"    → Foreign keys: {types.get('f', 0)}")
            print(f"    → Unique constraints: {types.get('u', 0)}")
            print(f"    → Check constraints: {types.get('c', 0)}")

            return True

        finally:
            await conn.close()

    async def check_extensions(self) -> bool:
        """Verify required extensions are installed"""
        conn = await self.connect()
        try:
            extensions = await conn.fetch("""
                SELECT extname, extversion
                FROM pg_extension
                WHERE extname IN ('uuid-ossp', 'vector', 'pgcrypto')
            """)

            required = {'uuid-ossp', 'vector'}
            installed = {ext['extname'] for ext in extensions}

            all_installed = True
            for req in required:
                if req in installed:
                    version = next(e['extversion'] for e in extensions if e['extname'] == req)
                    print(f"  ✓ Extension '{req}' v{version} installed")
                else:
                    print(f"  ✗ Extension '{req}' NOT INSTALLED")
                    self.errors.append(f"Missing extension: {req}")
                    all_installed = False

            return all_installed

        finally:
            await conn.close()

    async def check_all(self) -> bool:
        """Run all schema checks"""
        print("🔍 Running comprehensive schema validation...")
        print("=" * 60)

        all_passed = True

        # Check extensions
        print("\n📦 Extensions:")
        if not await self.check_extensions():
            all_passed = False

        # Check tables
        print("\n📊 Tables:")
        required_tables = [
            'golden_testsets',
            'golden_examples',
            'testset_versions',
            'testset_quality_metrics',
            'testset_approval_log',
            'schema_migrations'
        ]
        if not await self.check_tables(required_tables):
            all_passed = False

        # Check views
        print("\n👁 Views:")
        required_views = [
            'testset_overview',
            'latest_testsets',
            'example_quality_view',
            'version_history',
            'quality_dashboard'
        ]
        if not await self.check_views(required_views):
            all_passed = False

        # Check functions
        print("\n⚙️ Functions:")
        required_functions = [
            'format_version',
            'parse_version',
            'compare_versions',
            'get_latest_testset_version',
            'get_testset_stats',
            'validate_version_progression'
        ]
        if not await self.check_functions(required_functions):
            all_passed = False

        # Check indexes
        print("\n🔍 Indexes:")
        if not await self.check_indexes():
            all_passed = False

        # Check constraints
        print("\n🔒 Constraints:")
        if not await self.check_constraints():
            all_passed = False

        return all_passed

    async def run(self, requirements: Dict[str, str]) -> int:
        """Run schema assertions based on requirements"""

        if 'check' in requirements and requirements['check'] == 'all':
            success = await self.check_all()
        else:
            success = True

            if 'tables' in requirements:
                print("📊 Checking required tables...")
                tables = requirements['tables'].split(',')
                if not await self.check_tables(tables):
                    success = False

            if 'views' in requirements:
                print("\n👁 Checking required views...")
                views = requirements['views'].split(',')
                if not await self.check_views(views):
                    success = False

            if 'functions' in requirements:
                print("\n⚙️ Checking required functions...")
                functions = requirements['functions'].split(',')
                if not await self.check_functions(functions):
                    success = False

            if 'indexes' in requirements:
                print("\n🔍 Checking indexes...")
                table = requirements['indexes'] if requirements['indexes'] != 'all' else None
                if not await self.check_indexes(table):
                    success = False

        # Print summary
        print("\n" + "=" * 60)
        if self.errors:
            print("❌ SCHEMA VALIDATION FAILED")
            print(f"   {len(self.errors)} errors found:")
            for error in self.errors:
                print(f"   • {error}")
        elif self.warnings:
            print("⚠️ SCHEMA VALIDATION PASSED WITH WARNINGS")
            print(f"   {len(self.warnings)} warnings:")
            for warning in self.warnings:
                print(f"   • {warning}")
        else:
            print("✅ SCHEMA VALIDATION PASSED")
            print("   All requirements met!")

        return 0 if success else 1


def parse_requirements(args_list: List[str]) -> Dict[str, str]:
    """Parse --require arguments into a dictionary"""
    requirements = {}

    for arg in args_list:
        if '=' in arg:
            key, value = arg.split('=', 1)
            requirements[key] = value
        else:
            requirements[arg] = 'true'

    return requirements


async def main():
    parser = argparse.ArgumentParser(
        description='Assert database schema requirements for golden testset management'
    )
    parser.add_argument(
        '--require',
        action='append',
        help='Requirements to check (e.g., tables=golden_testsets,golden_examples)',
        default=[]
    )
    parser.add_argument(
        '--check',
        choices=['all', 'performance'],
        help='Run comprehensive checks'
    )
    parser.add_argument(
        '--connection',
        help='Database connection string',
        default=None
    )

    args = parser.parse_args()

    # Parse requirements
    requirements = parse_requirements(args.require)

    if args.check:
        requirements['check'] = args.check

    if not requirements:
        # Default to checking all if no specific requirements
        requirements['check'] = 'all'

    # Run assertions
    asserter = SchemaAsserter(args.connection)
    exit_code = await asserter.run(requirements)

    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())