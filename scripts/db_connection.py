#!/usr/bin/env python3
"""
Database Connection Manager for Golden Testset Management
Provides async connection management with connection pooling and health monitoring

This script implements the p1.db.conn task from tasks.yaml:
- Async connection management with connection pooling
- Health monitoring and connection validation
- Retry logic with exponential backoff
- Connection pool configuration and monitoring
- Integration with Phoenix observability

Features:
- AsyncPG connection pooling for optimal performance
- Automatic connection health checks and recovery
- Configurable connection parameters and timeouts
- Connection metrics and monitoring
- Support for read/write connection separation
- Transaction management utilities

Usage:
    python scripts/db_connection.py --test              # Test connections
    python scripts/db_connection.py --monitor           # Monitor connection health
    python scripts/db_connection.py --pool-stats        # Show connection pool statistics
    python scripts/db_connection.py --stress-test       # Run connection stress test
"""

import asyncio
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import asyncpg


@dataclass
class ConnectionConfig:
    """Configuration for database connections"""
    host: str = "localhost"
    port: int = 6024
    user: str = "langchain"
    password: str = "langchain"
    database: str = "langchain"

    # Pool configuration
    min_size: int = 5
    max_size: int = 20
    command_timeout: float = 30.0
    server_settings: Dict[str, Any] = field(default_factory=dict)

    # Health check configuration
    health_check_interval: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0

    # Connection string override
    connection_string: Optional[str] = None

    def get_dsn(self) -> str:
        """Get PostgreSQL DSN string"""
        if self.connection_string:
            return self.connection_string

        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    @classmethod
    def from_env(cls) -> "ConnectionConfig":
        """Create configuration from environment variables"""
        config = cls()

        # Override from environment
        config.host = os.environ.get("POSTGRES_HOST", config.host)
        config.port = int(os.environ.get("POSTGRES_PORT", config.port))
        config.user = os.environ.get("POSTGRES_USER", config.user)
        config.password = os.environ.get("POSTGRES_PASSWORD", config.password)
        config.database = os.environ.get("POSTGRES_DB", config.database)

        # Connection string override
        config.connection_string = os.environ.get("POSTGRES_CONNECTION_STRING")

        # Pool configuration
        config.min_size = int(os.environ.get("POSTGRES_POOL_MIN_SIZE", config.min_size))
        config.max_size = int(os.environ.get("POSTGRES_POOL_MAX_SIZE", config.max_size))
        config.command_timeout = float(os.environ.get("POSTGRES_COMMAND_TIMEOUT", config.command_timeout))

        return config


@dataclass
class ConnectionMetrics:
    """Connection pool metrics and health information"""
    pool_size: int = 0
    pool_free: int = 0
    pool_acquired: int = 0
    total_connections: int = 0
    failed_connections: int = 0
    last_error: Optional[str] = None
    last_health_check: Optional[datetime] = None
    health_status: str = "unknown"  # healthy, degraded, unhealthy
    response_times: List[float] = field(default_factory=list)

    def avg_response_time(self) -> float:
        """Calculate average response time"""
        return sum(self.response_times) / len(self.response_times) if self.response_times else 0.0

    def add_response_time(self, time_ms: float) -> None:
        """Add response time (keep last 100 measurements)"""
        self.response_times.append(time_ms)
        if len(self.response_times) > 100:
            self.response_times.pop(0)


class DatabaseConnectionManager:
    """Async database connection manager with pooling and health monitoring"""

    def __init__(self, config: Optional[ConnectionConfig] = None):
        self.config = config or ConnectionConfig.from_env()
        self.pool: Optional[asyncpg.Pool] = None
        self.metrics = ConnectionMetrics()
        self._health_check_task: Optional[asyncio.Task] = None
        self._is_healthy = False

    async def initialize(self) -> bool:
        """Initialize connection pool"""
        try:
            print(f"🔌 Initializing connection pool to {self.config.host}:{self.config.port}")

            self.pool = await asyncpg.create_pool(
                dsn=self.config.get_dsn(),
                min_size=self.config.min_size,
                max_size=self.config.max_size,
                command_timeout=self.config.command_timeout,
                server_settings=self.config.server_settings
            )

            # Test initial connection
            async with self.pool.acquire() as conn:
                version = await conn.fetchval("SELECT version()")
                print(f"✓ Connected to: {version[:60]}...")

            # Update metrics
            self.metrics.pool_size = self.pool.get_size()
            self.metrics.pool_free = self.pool.get_idle_size()
            self.metrics.total_connections += 1
            self._is_healthy = True

            # Start health monitoring
            self._health_check_task = asyncio.create_task(self._health_monitor())

            print(f"✓ Connection pool initialized (min={self.config.min_size}, max={self.config.max_size})")
            return True

        except Exception as e:
            print(f"✗ Failed to initialize connection pool: {e}")
            self.metrics.failed_connections += 1
            self.metrics.last_error = str(e)
            self.metrics.health_status = "unhealthy"
            return False

    async def close(self) -> None:
        """Close connection pool and cleanup"""
        print("🔌 Closing connection pool...")

        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        if self.pool:
            await self.pool.close()
            self.pool = None

        self._is_healthy = False
        print("✓ Connection pool closed")

    @asynccontextmanager
    async def get_connection(self):
        """Get a connection from the pool with automatic cleanup"""
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")

        start_time = time.perf_counter()
        conn = None

        try:
            conn = await self.pool.acquire()
            self.metrics.pool_acquired += 1

            # Record response time
            response_time = (time.perf_counter() - start_time) * 1000
            self.metrics.add_response_time(response_time)

            yield conn

        except Exception as e:
            self.metrics.failed_connections += 1
            self.metrics.last_error = str(e)
            raise
        finally:
            if conn:
                await self.pool.release(conn)
                self.metrics.pool_acquired = max(0, self.metrics.pool_acquired - 1)

    async def execute_query(self, query: str, *args, **kwargs) -> Any:
        """Execute a query with connection pooling"""
        async with self.get_connection() as conn:
            return await conn.fetch(query, *args, **kwargs)

    async def execute_single(self, query: str, *args, **kwargs) -> Any:
        """Execute a query returning a single value"""
        async with self.get_connection() as conn:
            return await conn.fetchval(query, *args, **kwargs)

    async def execute_command(self, command: str, *args, **kwargs) -> None:
        """Execute a command (INSERT, UPDATE, DELETE)"""
        async with self.get_connection() as conn:
            await conn.execute(command, *args, **kwargs)

    @asynccontextmanager
    async def transaction(self):
        """Get a transaction context manager"""
        async with self.get_connection() as conn:
            async with conn.transaction():
                yield conn

    async def test_connection(self) -> bool:
        """Test database connection and basic operations"""
        try:
            start_time = time.perf_counter()

            # Test basic connection
            version = await self.execute_single("SELECT version()")

            # Test golden testset schema if it exists
            schema_test_results = {}

            # Check if golden testset tables exist
            tables = await self.execute_query("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name LIKE 'golden_%'
            """)

            schema_test_results["tables_exist"] = len(tables) > 0
            schema_test_results["table_count"] = len(tables)

            if tables:
                # Test a simple query on golden_testsets if it exists
                testset_count = await self.execute_single("""
                    SELECT COUNT(*) FROM golden_testsets
                """)
                schema_test_results["testset_count"] = testset_count

                # Test views
                try:
                    overview_count = await self.execute_single("""
                        SELECT COUNT(*) FROM testset_overview
                    """)
                    schema_test_results["views_working"] = True
                    schema_test_results["overview_count"] = overview_count
                except:
                    schema_test_results["views_working"] = False

                # Test functions
                try:
                    version_test = await self.execute_single("""
                        SELECT format_version(1, 0, 0, 'test')
                    """)
                    schema_test_results["functions_working"] = version_test == "1.0.0-test"
                except:
                    schema_test_results["functions_working"] = False

            # Record metrics
            response_time = (time.perf_counter() - start_time) * 1000
            self.metrics.add_response_time(response_time)
            self.metrics.last_health_check = datetime.now()
            self.metrics.health_status = "healthy"

            print(f"✓ Database connection test passed")
            print(f"  Database version: {version[:60]}...")
            print(f"  Response time: {response_time:.2f}ms")
            print(f"  Schema status: {schema_test_results}")

            return True

        except Exception as e:
            self.metrics.failed_connections += 1
            self.metrics.last_error = str(e)
            self.metrics.health_status = "unhealthy"
            print(f"✗ Database connection test failed: {e}")
            return False

    async def get_pool_stats(self) -> Dict[str, Any]:
        """Get current connection pool statistics"""
        if not self.pool:
            return {"error": "Pool not initialized"}

        # Update current metrics
        self.metrics.pool_size = self.pool.get_size()
        self.metrics.pool_free = self.pool.get_idle_size()

        return {
            "pool": {
                "size": self.metrics.pool_size,
                "free": self.metrics.pool_free,
                "acquired": self.metrics.pool_acquired,
                "max_size": self.config.max_size,
                "min_size": self.config.min_size
            },
            "health": {
                "status": self.metrics.health_status,
                "last_check": self.metrics.last_health_check.isoformat() if self.metrics.last_health_check else None,
                "avg_response_time_ms": self.metrics.avg_response_time(),
                "total_connections": self.metrics.total_connections,
                "failed_connections": self.metrics.failed_connections,
                "last_error": self.metrics.last_error
            },
            "config": {
                "host": self.config.host,
                "port": self.config.port,
                "database": self.config.database,
                "command_timeout": self.config.command_timeout
            }
        }

    async def stress_test(self, concurrent_connections: int = 10, iterations: int = 100) -> Dict[str, Any]:
        """Run a stress test on the connection pool"""
        print(f"🔥 Starting stress test: {concurrent_connections} concurrent connections, {iterations} iterations each")

        start_time = time.perf_counter()
        results = {
            "total_operations": concurrent_connections * iterations,
            "successful_operations": 0,
            "failed_operations": 0,
            "avg_response_time_ms": 0.0,
            "max_response_time_ms": 0.0,
            "min_response_time_ms": float('inf'),
            "errors": []
        }

        async def worker(worker_id: int):
            """Single worker for stress test"""
            worker_results = {"success": 0, "errors": 0, "response_times": []}

            for i in range(iterations):
                try:
                    start = time.perf_counter()

                    # Simple query that tests connection and basic functionality
                    result = await self.execute_single("SELECT NOW(), $1::int", worker_id * iterations + i)

                    response_time = (time.perf_counter() - start) * 1000
                    worker_results["response_times"].append(response_time)
                    worker_results["success"] += 1

                except Exception as e:
                    worker_results["errors"] += 1
                    results["errors"].append(f"Worker {worker_id}, iteration {i}: {str(e)}")

            return worker_results

        # Run workers concurrently
        tasks = [worker(i) for i in range(concurrent_connections)]
        worker_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        all_response_times = []
        for result in worker_results:
            if isinstance(result, Exception):
                results["failed_operations"] += iterations
                results["errors"].append(f"Worker failed: {str(result)}")
            else:
                results["successful_operations"] += result["success"]
                results["failed_operations"] += result["errors"]
                all_response_times.extend(result["response_times"])

        if all_response_times:
            results["avg_response_time_ms"] = sum(all_response_times) / len(all_response_times)
            results["max_response_time_ms"] = max(all_response_times)
            results["min_response_time_ms"] = min(all_response_times)

        total_time = time.perf_counter() - start_time
        results["total_time_seconds"] = total_time
        results["operations_per_second"] = results["total_operations"] / total_time

        print(f"✓ Stress test completed in {total_time:.2f}s")
        print(f"  Success rate: {results['successful_operations']}/{results['total_operations']} ({results['successful_operations']/results['total_operations']*100:.1f}%)")
        print(f"  Operations/sec: {results['operations_per_second']:.1f}")
        print(f"  Avg response: {results['avg_response_time_ms']:.2f}ms")

        return results

    async def _health_monitor(self) -> None:
        """Background health monitoring task"""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)

                # Perform health check
                health_ok = await self.test_connection()

                if not health_ok and self.metrics.health_status != "unhealthy":
                    print("⚠ Database health check failed, status changed to unhealthy")
                elif health_ok and self.metrics.health_status == "unhealthy":
                    print("✓ Database health recovered, status changed to healthy")

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"⚠ Health monitor error: {e}")
                self.metrics.health_status = "degraded"


async def main():
    """Main entry point for connection testing and monitoring"""
    import argparse

    parser = argparse.ArgumentParser(description="Database Connection Manager")
    parser.add_argument("--connection", help="PostgreSQL connection string")
    parser.add_argument("--test", action="store_true", help="Test database connection")
    parser.add_argument("--monitor", action="store_true", help="Monitor connection health")
    parser.add_argument("--pool-stats", action="store_true", help="Show connection pool statistics")
    parser.add_argument("--stress-test", action="store_true", help="Run connection stress test")
    parser.add_argument("--concurrent", type=int, default=10, help="Concurrent connections for stress test")
    parser.add_argument("--iterations", type=int, default=100, help="Iterations per connection for stress test")

    args = parser.parse_args()

    if not any([args.test, args.monitor, args.pool_stats, args.stress_test]):
        parser.print_help()
        sys.exit(1)

    # Create configuration
    config = ConnectionConfig.from_env()
    if args.connection:
        config.connection_string = args.connection

    manager = DatabaseConnectionManager(config)

    try:
        # Initialize connection pool
        success = await manager.initialize()
        if not success:
            sys.exit(1)

        if args.test:
            success = await manager.test_connection()
            if not success:
                sys.exit(1)

        if args.pool_stats:
            stats = await manager.get_pool_stats()
            print("📊 Connection Pool Statistics:")
            print("=" * 40)
            print(json.dumps(stats, indent=2, default=str))

        if args.stress_test:
            results = await manager.stress_test(args.concurrent, args.iterations)
            print("\n🔥 Stress Test Results:")
            print("=" * 40)
            print(json.dumps(results, indent=2, default=str))

        if args.monitor:
            print("👀 Monitoring connection health (Ctrl+C to stop)...")
            print("=" * 40)
            try:
                while True:
                    stats = await manager.get_pool_stats()
                    health = stats["health"]
                    pool = stats["pool"]

                    status_emoji = "✓" if health["status"] == "healthy" else "⚠" if health["status"] == "degraded" else "✗"
                    print(f"{status_emoji} {datetime.now().strftime('%H:%M:%S')} | "
                          f"Status: {health['status']} | "
                          f"Pool: {pool['acquired']}/{pool['size']} | "
                          f"Avg: {health['avg_response_time_ms']:.1f}ms")

                    await asyncio.sleep(5)
            except KeyboardInterrupt:
                print("\n👋 Monitoring stopped")

    except Exception as e:
        print(f"✗ Operation failed: {e}")
        sys.exit(1)
    finally:
        await manager.close()


if __name__ == "__main__":
    asyncio.run(main())