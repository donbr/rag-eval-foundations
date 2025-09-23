#!/usr/bin/env python3
"""
SQL Dry Run and Validation Script for Golden Testset Management
Validates SQL scripts without executing them, particularly useful for rollback scripts

Usage:
    python dry_run_sql.py schemas/rollback_golden_testset.sql --check syntax
    python dry_run_sql.py schemas/golden_testset_schema.sql --validate
    python dry_run_sql.py --test-rollback schemas/rollback_golden_testset.sql
"""

import asyncio
import asyncpg
import argparse
import sys
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import os
import tempfile

class SQLValidator:
    """Validates SQL scripts without executing them"""

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

    def parse_sql_file(self, file_path: Path) -> List[str]:
        """Parse SQL file into individual statements"""
        if not file_path.exists():
            raise FileNotFoundError(f"SQL file not found: {file_path}")

        content = file_path.read_text()

        # Remove comments
        content = re.sub(r'--.*?\n', '\n', content)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)

        # Split by semicolons (simplified - doesn't handle all edge cases)
        statements = []
        current_statement = ""

        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue

            current_statement += line + '\n'

            if line.endswith(';'):
                if current_statement.strip():
                    statements.append(current_statement.strip())
                current_statement = ""

        # Add final statement if it doesn't end with semicolon
        if current_statement.strip():
            statements.append(current_statement.strip())

        return statements

    async def check_syntax(self, statements: List[str]) -> bool:
        """Check SQL syntax without executing"""
        conn = await self.connect()
        try:
            all_valid = True

            print(f"🔍 Checking syntax for {len(statements)} statements...")
            print("-" * 60)

            for i, statement in enumerate(statements, 1):
                try:
                    # Use EXPLAIN to validate syntax without execution
                    if statement.upper().strip().startswith(('SELECT', 'WITH')):
                        await conn.fetchval(f"EXPLAIN {statement}")
                    elif statement.upper().strip().startswith(('INSERT', 'UPDATE', 'DELETE')):
                        # For DML, we can't easily dry run, so just prepare
                        await conn.prepare(statement)
                    elif statement.upper().strip().startswith(('CREATE', 'DROP', 'ALTER')):
                        # DDL statements - can't easily validate without executing
                        # So we do basic parsing checks
                        if self.validate_ddl_syntax(statement):
                            print(f"  ✓ Statement {i}: DDL syntax looks valid")
                        else:
                            print(f"  ✗ Statement {i}: DDL syntax issues")
                            all_valid = False
                    else:
                        # Other statements - try to prepare
                        await conn.prepare(statement)

                    if i <= 10:  # Show first 10
                        stmt_preview = statement[:50].replace('\n', ' ')
                        print(f"  ✓ Statement {i}: {stmt_preview}...")

                except Exception as e:
                    stmt_preview = statement[:50].replace('\n', ' ')
                    print(f"  ✗ Statement {i}: {stmt_preview}...")
                    print(f"    Error: {str(e)[:100]}...")
                    self.errors.append(f"Syntax error in statement {i}: {e}")
                    all_valid = False

            return all_valid

        finally:
            await conn.close()

    def validate_ddl_syntax(self, statement: str) -> bool:
        """Basic DDL syntax validation"""
        statement_upper = statement.upper().strip()

        # Basic patterns for common DDL statements
        ddl_patterns = [
            r'^CREATE\s+(TABLE|VIEW|INDEX|FUNCTION|TRIGGER)',
            r'^DROP\s+(TABLE|VIEW|INDEX|FUNCTION|TRIGGER)',
            r'^ALTER\s+(TABLE|VIEW)',
            r'^COMMENT\s+ON',
        ]

        for pattern in ddl_patterns:
            if re.match(pattern, statement_upper):
                return True

        # Check for obvious syntax issues
        if statement_upper.count('(') != statement_upper.count(')'):
            return False

        return True

    async def validate_rollback_script(self, file_path: Path) -> bool:
        """Validate a rollback script specifically"""
        print(f"🔄 Validating rollback script: {file_path.name}")
        print("-" * 60)

        statements = self.parse_sql_file(file_path)

        if not statements:
            print("  ✗ No statements found in rollback script")
            self.errors.append("Empty rollback script")
            return False

        # Check that it's actually a rollback script
        rollback_indicators = ['DROP TABLE', 'DROP VIEW', 'DROP FUNCTION', 'DROP TRIGGER', 'DROP INDEX']
        has_drops = False

        for statement in statements:
            for indicator in rollback_indicators:
                if indicator in statement.upper():
                    has_drops = True
                    break

        if not has_drops:
            print("  ⚠ Script doesn't contain DROP statements - may not be a rollback script")
            self.warnings.append("Rollback script doesn't contain DROP statements")

        # Check for dangerous patterns
        dangerous_patterns = [
            'DELETE FROM',
            'TRUNCATE',
            'DROP DATABASE',
            'DROP SCHEMA'
        ]

        for statement in statements:
            for pattern in dangerous_patterns:
                if pattern in statement.upper():
                    print(f"  ⚠ Found potentially dangerous operation: {pattern}")
                    self.warnings.append(f"Dangerous operation in rollback: {pattern}")

        # Check order - should generally DROP in reverse order of creation
        table_drops = []
        view_drops = []

        for statement in statements:
            if 'DROP TABLE' in statement.upper():
                # Extract table name
                match = re.search(r'DROP TABLE\s+(?:IF EXISTS\s+)?(\w+)', statement.upper())
                if match:
                    table_drops.append(match.group(1))
            elif 'DROP VIEW' in statement.upper():
                match = re.search(r'DROP VIEW\s+(?:IF EXISTS\s+)?(\w+)', statement.upper())
                if match:
                    view_drops.append(match.group(1))

        # Views should be dropped before tables
        if table_drops and view_drops:
            view_positions = [i for i, stmt in enumerate(statements) if 'DROP VIEW' in stmt.upper()]
            table_positions = [i for i, stmt in enumerate(statements) if 'DROP TABLE' in stmt.upper()]

            if view_positions and table_positions:
                if max(view_positions) > min(table_positions):
                    print("  ⚠ Views are being dropped after tables - may cause dependency issues")
                    self.warnings.append("Views dropped after tables")

        print(f"\n  📊 Rollback script analysis:")
        print(f"    • Total statements: {len(statements)}")
        print(f"    • Table drops: {len(table_drops)}")
        print(f"    • View drops: {len(view_drops)}")

        # Check syntax
        syntax_valid = await self.check_syntax(statements)

        return syntax_valid

    async def test_rollback_sandbox(self, file_path: Path) -> bool:
        """Test rollback in a temporary schema (sandbox)"""
        print(f"🧪 Testing rollback in sandbox environment...")
        print("-" * 60)

        conn = await self.connect()
        try:
            # Create temporary schema for testing
            test_schema = f"rollback_test_{int(asyncio.get_event_loop().time())}"

            print(f"  📦 Creating test schema: {test_schema}")
            await conn.execute(f"CREATE SCHEMA {test_schema}")

            try:
                # Set search path to test schema
                await conn.execute(f"SET search_path TO {test_schema}, public")

                # First, create some test objects to rollback
                print("  🏗️ Creating test objects...")
                test_objects = [
                    f"CREATE TABLE {test_schema}.test_golden_testsets (id UUID PRIMARY KEY)",
                    f"CREATE TABLE {test_schema}.test_golden_examples (id UUID PRIMARY KEY, testset_id UUID)",
                    f"CREATE VIEW {test_schema}.test_view AS SELECT * FROM {test_schema}.test_golden_testsets",
                    f"CREATE INDEX test_idx ON {test_schema}.test_golden_testsets(id)"
                ]

                for obj_sql in test_objects:
                    await conn.execute(obj_sql)

                print("  ✓ Test objects created")

                # Read and modify rollback script for test schema
                statements = self.parse_sql_file(file_path)
                modified_statements = []

                for statement in statements:
                    # Replace object names to use test schema
                    modified = statement.replace('golden_', f'{test_schema}.test_golden_')
                    modified = modified.replace('DROP TABLE', f'DROP TABLE IF EXISTS')
                    modified = modified.replace('DROP VIEW', f'DROP VIEW IF EXISTS')
                    modified = modified.replace('DROP INDEX', f'DROP INDEX IF EXISTS')
                    modified = modified.replace('DROP FUNCTION', f'DROP FUNCTION IF EXISTS')
                    modified_statements.append(modified)

                # Execute rollback statements
                print("  🔄 Executing rollback statements...")
                for i, statement in enumerate(modified_statements):
                    try:
                        await conn.execute(statement)
                        if i < 5:  # Show first few
                            stmt_preview = statement[:50].replace('\n', ' ')
                            print(f"    ✓ {stmt_preview}...")
                    except Exception as e:
                        print(f"    ✗ Failed: {statement[:50]}...")
                        print(f"      Error: {e}")
                        self.errors.append(f"Rollback test failed: {e}")
                        return False

                print("  ✅ Rollback test completed successfully")
                return True

            finally:
                # Clean up test schema
                print(f"  🧹 Cleaning up test schema: {test_schema}")
                await conn.execute(f"DROP SCHEMA {test_schema} CASCADE")

        except Exception as e:
            print(f"  ✗ Sandbox test failed: {e}")
            self.errors.append(f"Sandbox test error: {e}")
            return False

        finally:
            await conn.close()

    async def analyze_dependencies(self, file_path: Path) -> None:
        """Analyze object dependencies in SQL script"""
        print(f"🔗 Analyzing object dependencies...")
        print("-" * 60)

        statements = self.parse_sql_file(file_path)

        # Track objects and their dependencies
        objects = {'tables': [], 'views': [], 'functions': [], 'indexes': []}
        dependencies = []

        for statement in statements:
            upper_stmt = statement.upper()

            # Extract object creations
            if 'CREATE TABLE' in upper_stmt:
                match = re.search(r'CREATE TABLE\s+(\w+)', upper_stmt)
                if match:
                    objects['tables'].append(match.group(1))

            elif 'CREATE VIEW' in upper_stmt:
                match = re.search(r'CREATE VIEW\s+(\w+)', upper_stmt)
                if match:
                    objects['views'].append(match.group(1))

            elif 'CREATE FUNCTION' in upper_stmt:
                match = re.search(r'CREATE FUNCTION\s+(\w+)', upper_stmt)
                if match:
                    objects['functions'].append(match.group(1))

            elif 'CREATE INDEX' in upper_stmt:
                match = re.search(r'CREATE INDEX\s+(\w+)', upper_stmt)
                if match:
                    objects['indexes'].append(match.group(1))

            # Find dependencies (simplified)
            if 'REFERENCES' in upper_stmt:
                ref_match = re.search(r'REFERENCES\s+(\w+)', upper_stmt)
                if ref_match:
                    dependencies.append(f"Foreign key reference to {ref_match.group(1)}")

        print(f"  📊 Object Summary:")
        for obj_type, obj_list in objects.items():
            if obj_list:
                print(f"    • {obj_type.title()}: {len(obj_list)}")
                for obj in obj_list[:3]:  # Show first 3
                    print(f"      - {obj}")
                if len(obj_list) > 3:
                    print(f"      ... and {len(obj_list) - 3} more")

        if dependencies:
            print(f"\n  🔗 Dependencies found:")
            for dep in dependencies:
                print(f"    • {dep}")

    async def run_validation(self, file_path: Path, check_types: List[str]) -> int:
        """Run specified validation checks"""
        print(f"📄 Validating SQL file: {file_path.name}")
        print("=" * 60)

        success = True

        if 'syntax' in check_types:
            statements = self.parse_sql_file(file_path)
            if not await self.check_syntax(statements):
                success = False

        if 'rollback' in check_types:
            if not await self.validate_rollback_script(file_path):
                success = False

        if 'sandbox' in check_types:
            if not await self.test_rollback_sandbox(file_path):
                success = False

        if 'dependencies' in check_types:
            await self.analyze_dependencies(file_path)

        # Print summary
        print("\n" + "=" * 60)
        print("📊 VALIDATION SUMMARY")
        print("=" * 60)

        if self.errors:
            print(f"\n❌ ERRORS: {len(self.errors)}")
            for error in self.errors:
                print(f"   • {error}")

        if self.warnings:
            print(f"\n⚠️ WARNINGS: {len(self.warnings)}")
            for warning in self.warnings:
                print(f"   • {warning}")

        if not self.errors:
            print("\n✅ SQL validation passed!")
            if not self.warnings:
                print("   • No issues found")
            else:
                print(f"   • {len(self.warnings)} warnings to review")

        return 0 if success and not self.errors else 1


async def main():
    parser = argparse.ArgumentParser(
        description='Validate SQL scripts for golden testset management'
    )
    parser.add_argument(
        'sql_file',
        help='Path to SQL file to validate'
    )
    parser.add_argument(
        '--check',
        choices=['syntax', 'rollback', 'dependencies'],
        action='append',
        help='Type of check to perform (can be used multiple times)',
        default=[]
    )
    parser.add_argument(
        '--test-rollback',
        action='store_true',
        help='Test rollback script in sandbox environment'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Run all validation checks'
    )
    parser.add_argument(
        '--connection',
        help='Database connection string',
        default=None
    )

    args = parser.parse_args()

    sql_file = Path(args.sql_file)

    # Determine what checks to run
    check_types = args.check or []

    if args.validate:
        check_types = ['syntax', 'dependencies']
        # Add rollback check if it looks like a rollback script
        if 'rollback' in sql_file.name.lower():
            check_types.append('rollback')

    if args.test_rollback:
        check_types.append('sandbox')

    if not check_types:
        check_types = ['syntax']  # Default to syntax check

    validator = SQLValidator(args.connection)
    exit_code = await validator.run_validation(sql_file, check_types)

    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())