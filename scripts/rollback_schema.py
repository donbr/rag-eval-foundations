#!/usr/bin/env python3
"""
Schema Rollback Manager for Golden Testset Management
Provides safe rollback capabilities for database schema changes

This script implements the p1.sql.rollback task from tasks.yaml:
- Safe rollback of golden testset schema changes
- Backup and restore capabilities
- Dependency-aware cleanup (views before tables)
- Migration history preservation
- Data export before rollback

Usage:
    python scripts/rollback_schema.py --rollback           # Execute rollback
    python scripts/rollback_schema.py --backup            # Create backup only
    python scripts/rollback_schema.py --list-backups      # Show available backups
    python scripts/rollback_schema.py --restore BACKUP_ID # Restore from backup
    python scripts/rollback_schema.py --dry-run           # Show what would be removed
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import asyncpg


class SchemaRollbackManager:
    """Manages safe rollback of golden testset schema"""

    def __init__(self, connection_string: Optional[str] = None):
        self.connection_string = connection_string or os.environ.get(
            "POSTGRES_CONNECTION_STRING",
            "postgresql://langchain:langchain@localhost:6024/langchain"
        )
        self.backups_dir = Path("backups")
        self.backups_dir.mkdir(exist_ok=True)

    async def create_backup(self) -> str:
        """Create a backup of golden testset data before rollback"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_id = f"golden_testset_backup_{timestamp}"
        backup_dir = self.backups_dir / backup_id
        backup_dir.mkdir(exist_ok=True)

        print(f"📦 Creating backup: {backup_id}")

        conn = await asyncpg.connect(self.connection_string)

        try:
            # Export all golden testset data
            tables_to_backup = [
                'golden_testsets',
                'golden_examples',
                'testset_versions',
                'testset_quality_metrics',
                'testset_approval_log',
                'schema_migrations'
            ]

            backup_metadata = {
                "backup_id": backup_id,
                "created_at": datetime.now().isoformat(),
                "connection_info": {
                    "host": self.connection_string.split('@')[1].split('/')[0] if '@' in self.connection_string else "localhost",
                    "database": self.connection_string.split('/')[-1] if '/' in self.connection_string else "langchain"
                },
                "tables": [],
                "schema_info": {}
            }

            for table in tables_to_backup:
                # Check if table exists
                table_exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_schema = 'public' AND table_name = $1
                    )
                """, table)

                if not table_exists:
                    print(f"  ⚠ Table {table} does not exist, skipping")
                    continue

                # Export table data
                rows = await conn.fetch(f"SELECT * FROM {table}")
                data = [dict(row) for row in rows]

                # Handle datetime and UUID serialization
                for row in data:
                    for key, value in row.items():
                        if hasattr(value, 'isoformat'):
                            row[key] = value.isoformat()
                        elif hasattr(value, '__str__') and str(type(value)) == "<class 'uuid.UUID'>":
                            row[key] = str(value)

                # Save table data
                table_file = backup_dir / f"{table}.json"
                table_file.write_text(json.dumps(data, indent=2, default=str))

                print(f"  ✓ Exported {len(data)} rows from {table}")
                backup_metadata["tables"].append({
                    "name": table,
                    "row_count": len(data),
                    "file": f"{table}.json"
                })

                # Get table schema
                schema_info = await conn.fetch("""
                    SELECT column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = $1
                    ORDER BY ordinal_position
                """, table)

                backup_metadata["schema_info"][table] = [dict(col) for col in schema_info]

            # Export views
            views = await conn.fetch("""
                SELECT table_name, view_definition
                FROM information_schema.views
                WHERE table_schema = 'public' AND table_name LIKE '%testset%'
            """)

            view_definitions = {}
            for view in views:
                view_definitions[view['table_name']] = view['view_definition']

            if view_definitions:
                views_file = backup_dir / "views.json"
                views_file.write_text(json.dumps(view_definitions, indent=2))
                backup_metadata["views"] = list(view_definitions.keys())
                print(f"  ✓ Exported {len(view_definitions)} views")

            # Export functions
            functions = await conn.fetch("""
                SELECT p.proname as function_name, pg_get_functiondef(p.oid) as definition
                FROM pg_proc p
                JOIN pg_namespace n ON p.pronamespace = n.oid
                WHERE n.nspname = 'public' AND (
                    p.proname LIKE '%version%' OR
                    p.proname LIKE '%testset%' OR
                    p.proname = 'update_updated_at_column'
                )
            """)

            function_definitions = {}
            for func in functions:
                function_definitions[func['function_name']] = func['definition']

            if function_definitions:
                functions_file = backup_dir / "functions.json"
                functions_file.write_text(json.dumps(function_definitions, indent=2))
                backup_metadata["functions"] = list(function_definitions.keys())
                print(f"  ✓ Exported {len(function_definitions)} functions")

            # Save backup metadata
            metadata_file = backup_dir / "backup_metadata.json"
            metadata_file.write_text(json.dumps(backup_metadata, indent=2))

            print(f"✓ Backup created: {backup_dir}")
            return backup_id

        except Exception as e:
            print(f"✗ Backup failed: {e}")
            raise
        finally:
            await conn.close()

    async def list_backups(self) -> List[Dict]:
        """List available backups"""
        backups = []

        for backup_dir in self.backups_dir.glob("golden_testset_backup_*"):
            if backup_dir.is_dir():
                metadata_file = backup_dir / "backup_metadata.json"
                if metadata_file.exists():
                    try:
                        metadata = json.loads(metadata_file.read_text())
                        backups.append(metadata)
                    except Exception as e:
                        print(f"⚠ Could not read backup metadata for {backup_dir.name}: {e}")

        return sorted(backups, key=lambda x: x.get('created_at', ''), reverse=True)

    async def show_rollback_plan(self) -> Dict[str, List[str]]:
        """Show what would be removed during rollback"""
        conn = await asyncpg.connect(self.connection_string)

        try:
            plan = {
                "views": [],
                "functions": [],
                "tables": [],
                "indexes": [],
                "triggers": []
            }

            # Views to be dropped
            views = await conn.fetch("""
                SELECT table_name
                FROM information_schema.views
                WHERE table_schema = 'public' AND (
                    table_name LIKE '%testset%' OR
                    table_name = 'phoenix_integration'
                )
                ORDER BY table_name
            """)
            plan["views"] = [v['table_name'] for v in views]

            # Functions to be dropped
            functions = await conn.fetch("""
                SELECT p.proname
                FROM pg_proc p
                JOIN pg_namespace n ON p.pronamespace = n.oid
                WHERE n.nspname = 'public' AND (
                    p.proname LIKE '%version%' OR
                    p.proname LIKE '%testset%' OR
                    p.proname = 'update_updated_at_column'
                )
                ORDER BY p.proname
            """)
            plan["functions"] = [f['proname'] for f in functions]

            # Tables to be dropped
            tables = await conn.fetch("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name LIKE 'golden_%'
                ORDER BY table_name
            """)
            plan["tables"] = [t['table_name'] for t in tables]

            # Indexes to be dropped
            indexes = await conn.fetch("""
                SELECT indexname
                FROM pg_indexes
                WHERE schemaname = 'public' AND (
                    indexname LIKE 'idx_golden_%' OR
                    indexname LIKE 'idx_%testset%'
                )
                ORDER BY indexname
            """)
            plan["indexes"] = [i['indexname'] for i in indexes]

            # Triggers to be dropped
            triggers = await conn.fetch("""
                SELECT trigger_name, event_object_table
                FROM information_schema.triggers
                WHERE trigger_schema = 'public' AND (
                    trigger_name LIKE '%testset%' OR
                    trigger_name LIKE '%golden_%'
                )
                ORDER BY trigger_name
            """)
            plan["triggers"] = [f"{t['trigger_name']} (on {t['event_object_table']})" for t in triggers]

            return plan

        finally:
            await conn.close()

    async def execute_rollback(self, create_backup: bool = True) -> bool:
        """Execute the rollback process"""
        print("🔄 Starting golden testset schema rollback...")

        # Create backup first
        backup_id = None
        if create_backup:
            backup_id = await self.create_backup()

        try:
            conn = await asyncpg.connect(self.connection_string)

            # Execute rollback in correct order (dependencies first)
            rollback_sql = """
            -- Rollback golden testset schema
            -- Generated: {timestamp}

            -- Drop views (must be dropped before tables)
            DROP VIEW IF EXISTS phoenix_integration CASCADE;
            DROP VIEW IF EXISTS quality_dashboard CASCADE;
            DROP VIEW IF EXISTS version_history CASCADE;
            DROP VIEW IF EXISTS example_quality_view CASCADE;
            DROP VIEW IF EXISTS latest_testsets CASCADE;
            DROP VIEW IF EXISTS testset_overview CASCADE;

            -- Drop triggers
            DROP TRIGGER IF EXISTS update_golden_examples_updated_at ON golden_examples;
            DROP TRIGGER IF EXISTS validate_testset_version ON golden_testsets;

            -- Drop functions
            DROP FUNCTION IF EXISTS update_updated_at_column() CASCADE;
            DROP FUNCTION IF EXISTS validate_version_progression() CASCADE;
            DROP FUNCTION IF EXISTS get_testset_stats(UUID) CASCADE;
            DROP FUNCTION IF EXISTS get_latest_testset_version(VARCHAR) CASCADE;
            DROP FUNCTION IF EXISTS compare_versions(INT, INT, INT, INT, INT, INT) CASCADE;
            DROP FUNCTION IF EXISTS parse_version(VARCHAR) CASCADE;
            DROP FUNCTION IF EXISTS format_version(INT, INT, INT, VARCHAR) CASCADE;

            -- Drop tables (in reverse dependency order)
            DROP TABLE IF EXISTS testset_approval_log CASCADE;
            DROP TABLE IF EXISTS testset_quality_metrics CASCADE;
            DROP TABLE IF EXISTS testset_versions CASCADE;
            DROP TABLE IF EXISTS golden_examples CASCADE;
            DROP TABLE IF EXISTS golden_testsets CASCADE;
            """.format(timestamp=datetime.now().isoformat())

            print("🗑 Executing rollback SQL...")

            # Execute each statement separately for better error handling
            statements = [stmt.strip() for stmt in rollback_sql.split(';') if stmt.strip()]

            for stmt in statements:
                if stmt.startswith('--') or not stmt:
                    continue
                try:
                    await conn.execute(stmt)
                    # Extract object name for logging
                    if 'DROP' in stmt:
                        words = stmt.split()
                        if len(words) >= 3:
                            obj_type = words[1]
                            obj_name = words[4] if len(words) > 4 else words[3]
                            print(f"  ✓ Dropped {obj_type.lower()}: {obj_name}")
                except Exception as e:
                    # Continue with other statements even if one fails
                    print(f"  ⚠ Warning: {e}")

            await conn.close()

            # Record rollback in migration history (if table still exists)
            try:
                conn = await asyncpg.connect(self.connection_string)
                await conn.execute("""
                    INSERT INTO schema_migrations (
                        migration_name, migration_type, applied_by, version, metadata
                    ) VALUES ($1, $2, $3, $4, $5)
                """, f"rollback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "rollback", "rollback_script", "rollback",
                    json.dumps({"backup_id": backup_id, "rollback_at": datetime.now().isoformat()}))
                await conn.close()
            except:
                # Migration table might have been dropped
                pass

            print("✓ Rollback completed successfully")

            if backup_id:
                print(f"💾 Data backed up to: {backup_id}")
                print(f"   To restore: python scripts/rollback_schema.py --restore {backup_id}")

            return True

        except Exception as e:
            print(f"✗ Rollback failed: {e}")
            if backup_id:
                print(f"💾 Data is safely backed up in: {backup_id}")
            raise

    async def restore_from_backup(self, backup_id: str) -> bool:
        """Restore golden testset schema and data from backup"""
        backup_dir = self.backups_dir / backup_id

        if not backup_dir.exists():
            print(f"✗ Backup not found: {backup_id}")
            return False

        metadata_file = backup_dir / "backup_metadata.json"
        if not metadata_file.exists():
            print(f"✗ Backup metadata not found: {backup_id}")
            return False

        print(f"🔄 Restoring from backup: {backup_id}")

        metadata = json.loads(metadata_file.read_text())
        conn = await asyncpg.connect(self.connection_string)

        try:
            # First, apply the schema (functions, tables, views)
            # This would require re-running the schema generation
            print("  ℹ Note: Restore requires schema to be recreated first")
            print("    Run: python scripts/db_migrate.py --migrate --force")
            print("    Then: python scripts/rollback_schema.py --restore", backup_id)

            # For now, just restore data if tables exist
            for table_info in metadata.get("tables", []):
                table_name = table_info["name"]
                table_file = backup_dir / table_info["file"]

                if not table_file.exists():
                    print(f"  ⚠ Data file not found: {table_file}")
                    continue

                # Check if table exists
                table_exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_schema = 'public' AND table_name = $1
                    )
                """, table_name)

                if not table_exists:
                    print(f"  ⚠ Table {table_name} does not exist, skipping data restore")
                    continue

                # Clear existing data
                await conn.execute(f"TRUNCATE TABLE {table_name} CASCADE")

                # Load data
                data = json.loads(table_file.read_text())

                if data:
                    # Prepare insert statement
                    columns = list(data[0].keys())
                    placeholders = ', '.join(f'${i+1}' for i in range(len(columns)))
                    insert_sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"

                    # Insert data
                    for row in data:
                        values = [row[col] for col in columns]
                        await conn.execute(insert_sql, *values)

                    print(f"  ✓ Restored {len(data)} rows to {table_name}")
                else:
                    print(f"  ✓ Table {table_name} restored (empty)")

            print(f"✓ Restore completed from backup: {backup_id}")
            return True

        except Exception as e:
            print(f"✗ Restore failed: {e}")
            return False
        finally:
            await conn.close()


async def main():
    """Main entry point for rollback management"""
    import argparse

    parser = argparse.ArgumentParser(description="Golden Testset Schema Rollback Manager")
    parser.add_argument("--connection", help="PostgreSQL connection string")
    parser.add_argument("--rollback", action="store_true", help="Execute rollback")
    parser.add_argument("--backup", action="store_true", help="Create backup only")
    parser.add_argument("--list-backups", action="store_true", help="List available backups")
    parser.add_argument("--restore", help="Restore from backup (provide backup ID)")
    parser.add_argument("--dry-run", action="store_true", help="Show rollback plan without executing")
    parser.add_argument("--no-backup", action="store_true", help="Skip backup during rollback")

    args = parser.parse_args()

    if not any([args.rollback, args.backup, args.list_backups, args.restore, args.dry_run]):
        parser.print_help()
        sys.exit(1)

    manager = SchemaRollbackManager(args.connection)

    try:
        if args.dry_run:
            plan = await manager.show_rollback_plan()
            print("📋 Rollback Plan:")
            print("=" * 40)
            for category, items in plan.items():
                if items:
                    print(f"\n{category.title()} to be dropped ({len(items)}):")
                    for item in items:
                        print(f"  - {item}")

        if args.list_backups:
            backups = await manager.list_backups()
            if backups:
                print("📦 Available Backups:")
                print("=" * 40)
                for backup in backups:
                    print(f"ID: {backup['backup_id']}")
                    print(f"Created: {backup['created_at']}")
                    print(f"Tables: {len(backup.get('tables', []))}")
                    print(f"Views: {len(backup.get('views', []))}")
                    print(f"Functions: {len(backup.get('functions', []))}")
                    print("-" * 40)
            else:
                print("No backups found")

        if args.backup:
            backup_id = await manager.create_backup()
            print(f"✓ Backup created: {backup_id}")

        if args.restore:
            success = await manager.restore_from_backup(args.restore)
            if not success:
                sys.exit(1)

        if args.rollback:
            success = await manager.execute_rollback(not args.no_backup)
            if not success:
                sys.exit(1)
            print("\n🎉 Rollback completed successfully!")
            print("\nTo recreate schema:")
            print("  python scripts/db_migrate.py --migrate")

    except Exception as e:
        print(f"✗ Operation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())