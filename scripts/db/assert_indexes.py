#!/usr/bin/env python3
"""
Index Performance Assertion Script for Golden Testset Management
Validates that indexes exist and are performing optimally

Usage:
    python assert_indexes.py --check performance
    python assert_indexes.py --table golden_testsets
    python assert_indexes.py --analyze all
"""

import asyncio
import asyncpg
import argparse
import sys
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import os

class IndexAsserter:
    """Validates database indexes and their performance"""

    def __init__(self, connection_string: Optional[str] = None):
        self.connection_string = connection_string or os.environ.get(
            "DATABASE_URL",
            "postgresql://langchain:langchain@localhost:6024/langchain"
        )
        self.errors = []
        self.warnings = []
        self.performance_issues = []

    async def connect(self) -> asyncpg.Connection:
        """Establish database connection"""
        try:
            return await asyncpg.connect(self.connection_string)
        except Exception as e:
            print(f"❌ Failed to connect to database: {e}")
            sys.exit(1)

    async def check_index_exists(self, conn: asyncpg.Connection, index_name: str) -> bool:
        """Check if a specific index exists"""
        result = await conn.fetchval("""
            SELECT EXISTS (
                SELECT 1
                FROM pg_indexes
                WHERE schemaname = 'public'
                AND indexname = $1
            )
        """, index_name)
        return result

    async def get_index_stats(self, conn: asyncpg.Connection, index_name: str) -> Dict[str, Any]:
        """Get statistics for a specific index"""
        stats = await conn.fetchrow("""
            SELECT
                schemaname,
                relname as tablename,
                indexrelname as indexname,
                pg_size_pretty(pg_relation_size(indexrelname::regclass)) as index_size,
                idx_scan as scan_count,
                idx_tup_read as tuples_read,
                idx_tup_fetch as tuples_fetched
            FROM pg_stat_user_indexes
            WHERE indexrelname = $1
        """, index_name)

        if stats:
            return dict(stats)
        return {}

    async def check_critical_indexes(self) -> bool:
        """Verify all critical indexes exist"""
        conn = await self.connect()
        try:
            critical_indexes = [
                # Primary keys
                ('golden_testsets_pkey', 'golden_testsets', 'PRIMARY KEY'),
                ('golden_examples_pkey', 'golden_examples', 'PRIMARY KEY'),

                # Foreign key indexes
                ('idx_golden_examples_testset', 'golden_examples', 'FOREIGN KEY'),

                # Search indexes
                ('idx_golden_testsets_name', 'golden_testsets', 'SEARCH'),
                ('idx_golden_testsets_version', 'golden_testsets', 'SEARCH'),
                ('idx_golden_testsets_status', 'golden_testsets', 'FILTER'),
                ('idx_golden_testsets_domain', 'golden_testsets', 'FILTER'),

                # Vector indexes (if using pgvector)
                ('idx_golden_examples_question_embedding', 'golden_examples', 'VECTOR'),
                ('idx_golden_examples_ground_truth_embedding', 'golden_examples', 'VECTOR'),
            ]

            all_exist = True
            print("🔍 Checking critical indexes...")
            print("-" * 60)

            for index_name, table_name, index_type in critical_indexes:
                exists = await self.check_index_exists(conn, index_name)

                if exists:
                    stats = await self.get_index_stats(conn, index_name)
                    size = stats.get('index_size', 'N/A')
                    scans = stats.get('scan_count', 0)

                    print(f"  ✓ {index_name} [{index_type}]")
                    print(f"    Table: {table_name}, Size: {size}, Scans: {scans}")

                    # Check if index is being used
                    if scans == 0 and index_type != 'VECTOR':
                        self.warnings.append(f"Index '{index_name}' has never been used")
                else:
                    print(f"  ✗ {index_name} [{index_type}] - MISSING")
                    self.errors.append(f"Missing critical index: {index_name}")
                    all_exist = False

            return all_exist

        finally:
            await conn.close()

    async def check_index_performance(self) -> bool:
        """Check index performance metrics"""
        conn = await self.connect()
        try:
            print("\n📊 Index Performance Analysis...")
            print("-" * 60)

            # Get all indexes for golden_* tables
            indexes = await conn.fetch("""
                SELECT
                    indexname,
                    tablename,
                    pg_size_pretty(pg_relation_size(indexname::regclass)) as index_size,
                    pg_relation_size(indexname::regclass) as size_bytes
                FROM pg_indexes
                WHERE schemaname = 'public'
                AND tablename LIKE 'golden_%'
                ORDER BY pg_relation_size(indexname::regclass) DESC
            """)

            if not indexes:
                self.warnings.append("No indexes found for golden_* tables")
                return False

            print(f"Found {len(indexes)} indexes\n")

            # Check for bloated indexes (> 100MB is concerning for our use case)
            bloat_threshold = 100 * 1024 * 1024  # 100MB
            bloated = []

            for idx in indexes:
                size_bytes = idx['size_bytes']
                if size_bytes > bloat_threshold:
                    bloated.append(idx['indexname'])
                    print(f"  ⚠ {idx['indexname']}: {idx['index_size']} (LARGE)")
                    self.warnings.append(f"Index '{idx['indexname']}' is large ({idx['index_size']})")

            # Check for unused indexes
            unused = await conn.fetch("""
                SELECT
                    indexrelname as indexname,
                    idx_scan as scan_count
                FROM pg_stat_user_indexes
                WHERE schemaname = 'public'
                AND idx_scan = 0
                AND indexrelname LIKE 'idx_golden_%'
            """)

            if unused:
                print(f"\n  ⚠ Found {len(unused)} unused indexes:")
                for idx in unused:
                    print(f"    • {idx['indexname']}")
                    self.warnings.append(f"Unused index: {idx['indexname']}")

            # Check for duplicate/redundant indexes
            duplicates = await self.check_duplicate_indexes(conn)
            if duplicates:
                print(f"\n  ⚠ Found potential duplicate indexes:")
                for dup in duplicates:
                    print(f"    • {dup}")
                    self.warnings.append(f"Potential duplicate index: {dup}")

            # Performance test: Check query performance with indexes
            perf_passed = await self.test_query_performance(conn)

            return perf_passed and len(bloated) == 0

        finally:
            await conn.close()

    async def check_duplicate_indexes(self, conn: asyncpg.Connection) -> List[str]:
        """Check for duplicate or redundant indexes"""
        # Get index definitions
        indexes = await conn.fetch("""
            SELECT
                indexname,
                tablename,
                indexdef
            FROM pg_indexes
            WHERE schemaname = 'public'
            AND tablename LIKE 'golden_%'
            AND indexname NOT LIKE '%_pkey'
        """)

        duplicates = []
        index_defs = {}

        for idx in indexes:
            # Extract column list from index definition
            indexdef = idx['indexdef']
            table = idx['tablename']

            # Simple duplicate detection based on table and columns
            # This is a simplified version - real duplicate detection would be more complex
            key = f"{table}:{indexdef.split('(')[1].split(')')[0] if '(' in indexdef else ''}"

            if key in index_defs:
                duplicates.append(f"{idx['indexname']} duplicates {index_defs[key]}")
            else:
                index_defs[key] = idx['indexname']

        return duplicates

    async def test_query_performance(self, conn: asyncpg.Connection) -> bool:
        """Test actual query performance with indexes"""
        print("\n⚡ Testing query performance...")
        print("-" * 60)

        test_queries = [
            (
                "Version lookup by name",
                """
                SELECT * FROM golden_testsets
                WHERE name = 'financial_aid_baseline'
                ORDER BY version_major DESC, version_minor DESC, version_patch DESC
                LIMIT 1
                """
            ),
            (
                "Examples by testset",
                """
                SELECT COUNT(*) FROM golden_examples
                WHERE testset_id = (
                    SELECT id FROM golden_testsets LIMIT 1
                )
                """
            ),
            (
                "Active testsets",
                """
                SELECT * FROM golden_testsets
                WHERE status = 'approved'
                """
            )
        ]

        all_fast = True
        performance_threshold_ms = 10  # Queries should complete in < 10ms

        for query_name, query in test_queries:
            try:
                # Warm up
                await conn.fetch(query)

                # Time the query
                start = time.time()
                await conn.fetch(query)
                elapsed_ms = (time.time() - start) * 1000

                if elapsed_ms < performance_threshold_ms:
                    print(f"  ✓ {query_name}: {elapsed_ms:.2f}ms")
                else:
                    print(f"  ⚠ {query_name}: {elapsed_ms:.2f}ms (SLOW)")
                    self.performance_issues.append(
                        f"{query_name} took {elapsed_ms:.2f}ms (threshold: {performance_threshold_ms}ms)"
                    )
                    all_fast = False

                # Get query plan
                plan = await conn.fetch(f"EXPLAIN (FORMAT JSON, ANALYZE) {query}")
                plan_json = plan[0]['QUERY PLAN'][0]

                # Check if indexes were used
                plan_str = str(plan_json)
                if 'Index Scan' in plan_str or 'Index Only Scan' in plan_str:
                    print(f"    → Using index scan")
                elif 'Seq Scan' in plan_str:
                    print(f"    ⚠ Using sequential scan (no index)")
                    self.warnings.append(f"{query_name} not using indexes effectively")

            except Exception as e:
                print(f"  ✗ {query_name}: ERROR - {e}")
                self.errors.append(f"Query test failed: {query_name}")
                all_fast = False

        return all_fast

    async def analyze_table(self, table_name: str) -> None:
        """Analyze a specific table's indexes"""
        conn = await self.connect()
        try:
            print(f"\n📋 Analyzing indexes for table: {table_name}")
            print("-" * 60)

            # Get all indexes for the table
            indexes = await conn.fetch("""
                SELECT
                    indexname,
                    indexdef,
                    pg_size_pretty(pg_relation_size(indexname::regclass)) as size
                FROM pg_indexes
                WHERE schemaname = 'public'
                AND tablename = $1
            """, table_name)

            if not indexes:
                print(f"  No indexes found for table '{table_name}'")
                return

            print(f"  Found {len(indexes)} indexes:\n")

            for idx in indexes:
                print(f"  • {idx['indexname']}")
                print(f"    Size: {idx['size']}")
                print(f"    Definition: {idx['indexdef'][:100]}...")

                # Get usage stats
                stats = await conn.fetchrow("""
                    SELECT
                        idx_scan,
                        idx_tup_read,
                        idx_tup_fetch
                    FROM pg_stat_user_indexes
                    WHERE indexrelname = $1
                """, idx['indexname'])

                if stats:
                    print(f"    Usage: {stats['idx_scan']} scans, {stats['idx_tup_read']} reads")
                print()

            # Check for missing indexes (simplified heuristic)
            print("  📊 Index recommendations:")

            # Check if foreign key columns have indexes
            fk_columns = await conn.fetch("""
                SELECT
                    kcu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                    ON tc.constraint_name = kcu.constraint_name
                WHERE tc.table_name = $1
                AND tc.constraint_type = 'FOREIGN KEY'
            """, table_name)

            for fk in fk_columns:
                col_name = fk['column_name']
                # Check if index exists for this column
                idx_exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT 1 FROM pg_indexes
                        WHERE tablename = $1
                        AND indexdef LIKE '%' || $2 || '%'
                    )
                """, table_name, col_name)

                if not idx_exists:
                    print(f"    ⚠ Consider adding index on foreign key column: {col_name}")
                    self.warnings.append(f"Missing index on FK column {table_name}.{col_name}")

        finally:
            await conn.close()

    async def run_checks(self, check_type: str, table: Optional[str] = None) -> int:
        """Run index checks based on type"""
        success = True

        if check_type == 'performance':
            success = await self.check_critical_indexes()
            if success:
                success = await self.check_index_performance()

        elif check_type == 'all':
            success = await self.check_critical_indexes()
            if success:
                success = await self.check_index_performance()

            # Analyze all golden_* tables
            tables = ['golden_testsets', 'golden_examples', 'testset_versions',
                     'testset_quality_metrics', 'testset_approval_log']
            for t in tables:
                await self.analyze_table(t)

        elif table:
            await self.analyze_table(table)

        # Print summary
        print("\n" + "=" * 60)
        print("📊 INDEX VALIDATION SUMMARY")
        print("=" * 60)

        if self.errors:
            print(f"\n❌ FAILED: {len(self.errors)} errors")
            for error in self.errors:
                print(f"   • {error}")

        if self.performance_issues:
            print(f"\n⚠️ PERFORMANCE ISSUES: {len(self.performance_issues)}")
            for issue in self.performance_issues:
                print(f"   • {issue}")

        if self.warnings:
            print(f"\n⚠️ WARNINGS: {len(self.warnings)}")
            for warning in self.warnings[:5]:  # Show first 5
                print(f"   • {warning}")
            if len(self.warnings) > 5:
                print(f"   ... and {len(self.warnings) - 5} more")

        if not self.errors and not self.performance_issues:
            print("\n✅ All index checks passed!")
            print("   • All critical indexes exist")
            print("   • Query performance is optimal")
            print("   • No major issues detected")

        return 0 if success and not self.errors else 1


async def main():
    parser = argparse.ArgumentParser(
        description='Assert index performance for golden testset management'
    )
    parser.add_argument(
        '--check',
        choices=['performance', 'all'],
        help='Type of check to perform',
        default='performance'
    )
    parser.add_argument(
        '--table',
        help='Analyze indexes for specific table'
    )
    parser.add_argument(
        '--analyze',
        choices=['all'],
        help='Run comprehensive analysis'
    )
    parser.add_argument(
        '--connection',
        help='Database connection string',
        default=None
    )

    args = parser.parse_args()

    asserter = IndexAsserter(args.connection)

    # Determine what to check
    if args.analyze == 'all':
        exit_code = await asserter.run_checks('all')
    elif args.table:
        exit_code = await asserter.run_checks('table', args.table)
    else:
        exit_code = await asserter.run_checks(args.check or 'performance')

    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())