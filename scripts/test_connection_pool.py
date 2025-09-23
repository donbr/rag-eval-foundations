#!/usr/bin/env python3
"""
Connection Pool Testing Script for Golden Testset Management
Tests database connection pool performance and configuration

Usage:
    python test_connection_pool.py --min 5 --max 20
    python test_connection_pool.py --stress-test --connections 10 --iterations 100
    python test_connection_pool.py --benchmark --duration 30
"""

import asyncio
import asyncpg
import argparse
import sys
import time
import statistics
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import os
import random
from concurrent.futures import ThreadPoolExecutor
import gc

class ConnectionPoolTester:
    """Tests database connection pool functionality and performance"""

    def __init__(self, connection_string: Optional[str] = None):
        self.connection_string = connection_string or os.environ.get(
            "DATABASE_URL",
            "postgresql://langchain:langchain@localhost:6024/langchain"
        )
        self.pool = None
        self.test_results = {}

    async def create_pool(self, min_size: int = 5, max_size: int = 20) -> asyncpg.Pool:
        """Create connection pool with specified parameters"""
        try:
            pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=min_size,
                max_size=max_size,
                command_timeout=30,
                server_settings={
                    'application_name': 'golden_testset_pool_test',
                    'jit': 'off'  # Disable JIT for consistent timing
                }
            )
            print(f"✓ Connection pool created (min={min_size}, max={max_size})")
            return pool
        except Exception as e:
            print(f"❌ Failed to create connection pool: {e}")
            sys.exit(1)

    async def test_basic_connectivity(self, pool: asyncpg.Pool) -> bool:
        """Test basic pool connectivity"""
        print("\n🔌 Testing basic connectivity...")
        print("-" * 50)

        try:
            # Test connection acquisition
            start_time = time.time()
            async with pool.acquire() as conn:
                # Test basic query
                result = await conn.fetchval("SELECT version()")
                acquisition_time = (time.time() - start_time) * 1000

                print(f"  ✓ Connection acquired in {acquisition_time:.2f}ms")
                print(f"  ✓ Database version: {result[:50]}...")

                # Test transaction
                async with conn.transaction():
                    await conn.fetchval("SELECT 1")
                    print("  ✓ Transaction test passed")

                return True

        except Exception as e:
            print(f"  ✗ Basic connectivity test failed: {e}")
            return False

    async def test_pool_sizing(self, min_size: int, max_size: int) -> Dict[str, Any]:
        """Test pool sizing and growth"""
        print(f"\n📊 Testing pool sizing (min={min_size}, max={max_size})...")
        print("-" * 50)

        pool = await self.create_pool(min_size, max_size)
        try:
            results = {
                'min_size': min_size,
                'max_size': max_size,
                'initial_size': None,
                'peak_size': None,
                'growth_time': None
            }

            # Check initial pool size
            initial_size = pool.get_size()
            results['initial_size'] = initial_size
            print(f"  📈 Initial pool size: {initial_size}")

            # Acquire multiple connections to test growth
            connections = []
            start_time = time.time()

            try:
                # Gradually acquire connections up to max_size
                for i in range(max_size):
                    conn = await pool.acquire()
                    connections.append(conn)
                    current_size = pool.get_size()

                    if i < 5:  # Show first few acquisitions
                        print(f"    Connection {i+1}: pool size = {current_size}")

                    results['peak_size'] = max(results.get('peak_size') or 0, current_size)

                growth_time = (time.time() - start_time) * 1000
                results['growth_time'] = growth_time

                print(f"  📊 Peak pool size: {results['peak_size']}")
                print(f"  ⏱️ Growth time: {growth_time:.2f}ms")

                # Test that we can't exceed max_size
                try:
                    extra_conn = await asyncio.wait_for(pool.acquire(), timeout=1.0)
                    # If we get here, the pool allowed more than max_size
                    await pool.release(extra_conn)
                    print("  ⚠️ Pool allowed more connections than max_size")
                except asyncio.TimeoutError:
                    print("  ✓ Pool correctly enforced max_size limit")

            finally:
                # Release all connections
                for conn in connections:
                    await pool.release(conn)

            # Check pool shrinks back to min_size (eventually)
            await asyncio.sleep(0.1)  # Give pool time to shrink
            final_size = pool.get_size()
            print(f"  📉 Final pool size: {final_size}")

            return results

        finally:
            await pool.close()

    async def stress_test_pool(self, connections: int, iterations: int) -> Dict[str, Any]:
        """Stress test the connection pool"""
        print(f"\n🔥 Stress testing pool ({connections} concurrent, {iterations} iterations each)...")
        print("-" * 70)

        # Create pool with appropriate sizing
        pool_size = min(connections * 2, 50)  # Don't go crazy with pool size
        pool = await self.create_pool(min_size=5, max_size=pool_size)

        try:
            results = {
                'total_operations': connections * iterations,
                'successful_operations': 0,
                'failed_operations': 0,
                'total_time': 0,
                'avg_response_time': 0,
                'min_response_time': float('inf'),
                'max_response_time': 0,
                'operations_per_second': 0,
                'response_times': []
            }

            async def worker(worker_id: int) -> List[float]:
                """Worker function for stress testing"""
                worker_times = []

                for i in range(iterations):
                    try:
                        start = time.time()

                        async with pool.acquire() as conn:
                            # Simulate realistic database operations
                            await conn.fetchval("SELECT COUNT(*) FROM golden_testsets")

                            # Random delay to simulate processing
                            await asyncio.sleep(random.uniform(0.001, 0.005))

                            # Another query
                            await conn.fetchval("SELECT version()")

                        elapsed = (time.time() - start) * 1000
                        worker_times.append(elapsed)

                        if i == 0:  # First iteration for this worker
                            print(f"    Worker {worker_id}: First operation {elapsed:.2f}ms")

                    except Exception as e:
                        print(f"    Worker {worker_id} iteration {i} failed: {e}")
                        results['failed_operations'] += 1

                return worker_times

            # Run stress test
            start_time = time.time()

            # Create tasks for concurrent workers
            tasks = [worker(i) for i in range(connections)]
            worker_results = await asyncio.gather(*tasks, return_exceptions=True)

            total_time = time.time() - start_time

            # Collect results
            all_times = []
            for worker_times in worker_results:
                if isinstance(worker_times, list):
                    all_times.extend(worker_times)
                    results['successful_operations'] += len(worker_times)

            if all_times:
                results['response_times'] = all_times
                results['total_time'] = total_time
                results['avg_response_time'] = statistics.mean(all_times)
                results['min_response_time'] = min(all_times)
                results['max_response_time'] = max(all_times)
                results['operations_per_second'] = len(all_times) / total_time

                print(f"\n  📊 Stress test results:")
                print(f"    • Total operations: {results['total_operations']}")
                print(f"    • Successful: {results['successful_operations']}")
                print(f"    • Failed: {results['failed_operations']}")
                print(f"    • Success rate: {(results['successful_operations']/results['total_operations']*100):.1f}%")
                print(f"    • Total time: {total_time:.2f}s")
                print(f"    • Avg response: {results['avg_response_time']:.2f}ms")
                print(f"    • Min response: {results['min_response_time']:.2f}ms")
                print(f"    • Max response: {results['max_response_time']:.2f}ms")
                print(f"    • Operations/sec: {results['operations_per_second']:.1f}")

                # Check for performance issues
                if results['avg_response_time'] > 100:
                    print(f"    ⚠️ High average response time: {results['avg_response_time']:.2f}ms")

                if results['failed_operations'] > 0:
                    print(f"    ⚠️ Some operations failed: {results['failed_operations']}")

            return results

        finally:
            await pool.close()

    async def benchmark_pool(self, duration: int) -> Dict[str, Any]:
        """Benchmark pool performance over time"""
        print(f"\n⚡ Benchmarking pool for {duration} seconds...")
        print("-" * 50)

        pool = await self.create_pool(min_size=10, max_size=30)

        try:
            results = {
                'duration': duration,
                'total_operations': 0,
                'operations_per_second': 0,
                'avg_response_time': 0,
                'response_times': []
            }

            async def benchmark_worker():
                """Continuous benchmark worker"""
                while True:
                    try:
                        start = time.time()
                        async with pool.acquire() as conn:
                            await conn.fetchval("SELECT COUNT(*) FROM golden_testsets")
                        elapsed = (time.time() - start) * 1000
                        results['response_times'].append(elapsed)
                        results['total_operations'] += 1
                    except Exception:
                        pass
                    await asyncio.sleep(0.01)  # Small delay

            # Run benchmark for specified duration
            benchmark_task = asyncio.create_task(benchmark_worker())

            await asyncio.sleep(duration)
            benchmark_task.cancel()

            try:
                await benchmark_task
            except asyncio.CancelledError:
                pass

            # Calculate results
            if results['response_times']:
                results['operations_per_second'] = results['total_operations'] / duration
                results['avg_response_time'] = statistics.mean(results['response_times'])

                print(f"  📊 Benchmark results:")
                print(f"    • Duration: {duration}s")
                print(f"    • Total operations: {results['total_operations']}")
                print(f"    • Operations/sec: {results['operations_per_second']:.1f}")
                print(f"    • Avg response: {results['avg_response_time']:.2f}ms")

                # Performance thresholds
                if results['operations_per_second'] < 50:
                    print(f"    ⚠️ Low throughput: {results['operations_per_second']:.1f} ops/sec")

                if results['avg_response_time'] > 50:
                    print(f"    ⚠️ High latency: {results['avg_response_time']:.2f}ms")

            return results

        finally:
            await pool.close()

    async def test_connection_limits(self) -> Dict[str, Any]:
        """Test connection limits and error handling"""
        print(f"\n🚫 Testing connection limits...")
        print("-" * 50)

        # Create a small pool to test limits
        pool = await self.create_pool(min_size=2, max_size=5)

        try:
            results = {
                'max_connections_reached': False,
                'timeout_handled': False,
                'error_recovery': False
            }

            # Try to acquire more connections than available
            connections = []
            try:
                print("  📈 Acquiring connections until limit...")
                for i in range(10):  # Try to get more than max_size
                    try:
                        conn = await asyncio.wait_for(pool.acquire(), timeout=1.0)
                        connections.append(conn)
                        print(f"    Connection {i+1}: acquired")
                    except asyncio.TimeoutError:
                        print(f"    Connection {i+1}: timeout (expected)")
                        results['timeout_handled'] = True
                        break

                if len(connections) >= 5:
                    results['max_connections_reached'] = True

            finally:
                # Release connections
                for conn in connections:
                    await pool.release(conn)

            # Test error recovery
            try:
                async with pool.acquire() as conn:
                    await conn.fetchval("SELECT 1")
                results['error_recovery'] = True
                print("  ✓ Pool recovered after limit test")
            except Exception as e:
                print(f"  ✗ Pool recovery failed: {e}")

            return results

        finally:
            await pool.close()

    async def run_tests(self, test_config: Dict[str, Any]) -> int:
        """Run all specified tests"""
        print("🧪 Connection Pool Test Suite")
        print("=" * 60)

        success = True
        self.test_results = {}

        # Basic connectivity test
        if test_config.get('basic', True):
            pool = await self.create_pool(test_config.get('min_size', 5),
                                        test_config.get('max_size', 20))
            try:
                if not await self.test_basic_connectivity(pool):
                    success = False
            finally:
                await pool.close()

        # Pool sizing test
        if test_config.get('sizing', True):
            sizing_results = await self.test_pool_sizing(
                test_config.get('min_size', 5),
                test_config.get('max_size', 20)
            )
            self.test_results['sizing'] = sizing_results

        # Stress test
        if test_config.get('stress', False):
            stress_results = await self.stress_test_pool(
                test_config.get('connections', 10),
                test_config.get('iterations', 50)
            )
            self.test_results['stress'] = stress_results

            # Check if stress test passed
            if stress_results['failed_operations'] > 0:
                success = False

        # Benchmark
        if test_config.get('benchmark', False):
            benchmark_results = await self.benchmark_pool(
                test_config.get('duration', 10)
            )
            self.test_results['benchmark'] = benchmark_results

        # Connection limits test
        if test_config.get('limits', True):
            limits_results = await self.test_connection_limits()
            self.test_results['limits'] = limits_results

        # Print summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)

        if success:
            print("✅ All connection pool tests passed!")
            print("\n🔧 Configuration recommendations:")
            if 'sizing' in self.test_results:
                sizing = self.test_results['sizing']
                print(f"  • Pool sizing: min={sizing['min_size']}, max={sizing['max_size']}")
            if 'stress' in self.test_results:
                stress = self.test_results['stress']
                print(f"  • Expected throughput: ~{stress['operations_per_second']:.0f} ops/sec")
            if 'benchmark' in self.test_results:
                benchmark = self.test_results['benchmark']
                print(f"  • Expected latency: ~{benchmark['avg_response_time']:.1f}ms avg")
        else:
            print("❌ Some connection pool tests failed!")
            print("   Review the test output above for details")

        return 0 if success else 1


async def main():
    parser = argparse.ArgumentParser(
        description='Test database connection pool for golden testset management'
    )
    parser.add_argument(
        '--min',
        type=int,
        default=5,
        help='Minimum pool size (default: 5)'
    )
    parser.add_argument(
        '--max',
        type=int,
        default=20,
        help='Maximum pool size (default: 20)'
    )
    parser.add_argument(
        '--stress-test',
        action='store_true',
        help='Run stress test'
    )
    parser.add_argument(
        '--connections',
        type=int,
        default=10,
        help='Number of concurrent connections for stress test (default: 10)'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=50,
        help='Number of iterations per connection for stress test (default: 50)'
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run benchmark test'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=10,
        help='Benchmark duration in seconds (default: 10)'
    )
    parser.add_argument(
        '--basic-only',
        action='store_true',
        help='Run only basic connectivity tests'
    )
    parser.add_argument(
        '--connection',
        help='Database connection string',
        default=None
    )

    args = parser.parse_args()

    # Configure test suite
    test_config = {
        'min_size': args.min,
        'max_size': args.max,
        'basic': True,
        'sizing': not args.basic_only,
        'stress': args.stress_test,
        'connections': args.connections,
        'iterations': args.iterations,
        'benchmark': args.benchmark,
        'duration': args.duration,
        'limits': not args.basic_only
    }

    tester = ConnectionPoolTester(args.connection)
    exit_code = await tester.run_tests(test_config)

    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())