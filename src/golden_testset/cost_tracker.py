"""Cost tracking for golden testset generation with token usage monitoring.

This module provides functionality to:
1. Track token usage for each generation
2. Calculate costs based on model pricing
3. Monitor budget thresholds and alerts
4. Store historical cost data for analysis
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import ROUND_UP, Decimal
from typing import Any

# Import database connection manager
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from scripts.db_connection import DatabaseConnectionManager, ConnectionConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelPricing:
    """Pricing information for AI models."""

    model_name: str
    input_price_per_1k: Decimal  # Price per 1000 input tokens
    output_price_per_1k: Decimal  # Price per 1000 output tokens
    effective_date: datetime
    currency: str = "USD"

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> Decimal:
        """Calculate total cost for given token usage.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Total cost in currency units
        """
        input_cost = (Decimal(input_tokens) / Decimal(1000)) * self.input_price_per_1k
        output_cost = (
            Decimal(output_tokens) / Decimal(1000)
        ) * self.output_price_per_1k
        total_cost = input_cost + output_cost

        # Round up to 4 decimal places (cents precision)
        return total_cost.quantize(Decimal("0.0001"), rounding=ROUND_UP)


@dataclass
class TokenUsage:
    """Token usage for a generation session."""

    session_id: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate total tokens if not provided."""
        if self.total_tokens == 0:
            self.total_tokens = self.input_tokens + self.output_tokens


@dataclass
class CostBudget:
    """Budget configuration for cost tracking."""

    daily_limit: Decimal = Decimal("10.00")
    monthly_limit: Decimal = Decimal("300.00")
    per_generation_limit: Decimal = Decimal("5.00")
    alert_threshold: float = 0.8  # Alert at 80% of budget
    hard_stop_threshold: float = 0.95  # Stop at 95% of budget


class CostTracker:
    """Tracks costs and token usage for golden testset generation."""

    # Model pricing as of 2025
    DEFAULT_PRICING = {
        "gpt-4.1-mini": ModelPricing(
            model_name="gpt-4.1-mini",
            input_price_per_1k=Decimal("0.00015"),
            output_price_per_1k=Decimal("0.0006"),
            effective_date=datetime(2025, 1, 1),
        ),
        "text-embedding-3-small": ModelPricing(
            model_name="text-embedding-3-small",
            input_price_per_1k=Decimal("0.00002"),
            output_price_per_1k=Decimal("0"),  # Embeddings have no output cost
            effective_date=datetime(2025, 1, 1),
        ),
        "rerank-english-v3.0": ModelPricing(
            model_name="rerank-english-v3.0",
            input_price_per_1k=Decimal("0.0002"),
            output_price_per_1k=Decimal("0"),  # Reranking has no output cost
            effective_date=datetime(2025, 1, 1),
        ),
    }

    def __init__(self, budget: CostBudget | None = None, db_manager = None):
        """Initialize cost tracker.

        Args:
            budget: Budget configuration
            db_manager: Database connection manager
        """
        self.budget = budget or CostBudget()
        self.pricing = self.DEFAULT_PRICING.copy()
        self.current_session: str | None = None
        self.session_usage: dict[str, list[TokenUsage]] = {}
        self.db_manager = db_manager or DatabaseConnectionManager(ConnectionConfig())

    async def start_session(
        self, session_id: str, metadata: dict[str, Any] = None
    ) -> str:
        """Start a new cost tracking session.

        Args:
            session_id: Unique session identifier
            metadata: Optional session metadata

        Returns:
            Session ID
        """
        self.current_session = session_id
        self.session_usage[session_id] = []

        # Store session start in database
        async with self.db_manager.get_connection() as conn:
            await conn.execute(
                """
                INSERT INTO golden_testset_cost_tracking
                (session_id, started_at, metadata, status)
                VALUES ($1, $2, $3, 'active')
            """,
                session_id,
                datetime.utcnow(),
                json.dumps(metadata or {}),
            )

        logger.info(f"Started cost tracking session: {session_id}")
        return session_id

    async def track_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        metadata: dict[str, Any] = None,
    ) -> dict[str, Any]:
        """Track token usage for current session.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            metadata: Optional usage metadata

        Returns:
            Usage summary with cost calculation
        """
        if not self.current_session:
            raise ValueError("No active session. Call start_session first.")

        usage = TokenUsage(
            session_id=self.current_session,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            metadata=metadata or {},
        )

        self.session_usage[self.current_session].append(usage)

        # Calculate cost
        cost = Decimal(0)
        if model in self.pricing:
            cost = self.pricing[model].calculate_cost(input_tokens, output_tokens)

        # Store in database
        async with self.db_manager.get_connection() as conn:
            await conn.execute(
                """
                INSERT INTO golden_testset_token_usage
                (session_id, model, input_tokens, output_tokens, total_tokens,
                 cost, timestamp, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
                self.current_session,
                model,
                input_tokens,
                output_tokens,
                usage.total_tokens,
                float(cost),
                usage.timestamp,
                json.dumps(usage.metadata),
            )

        # Check budget
        budget_status = await self._check_budget(cost)

        return {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": usage.total_tokens,
            "cost": float(cost),
            "budget_status": budget_status,
        }

    async def _check_budget(self, new_cost: Decimal) -> dict[str, Any]:
        """Check budget status with new cost.

        Args:
            new_cost: Cost to add

        Returns:
            Budget status information
        """
        # Get current spending
        daily_spent = await self.get_daily_spending()
        monthly_spent = await self.get_monthly_spending()
        session_spent = await self.get_session_spending()

        # Calculate new totals
        new_daily = daily_spent + new_cost
        new_monthly = monthly_spent + new_cost
        new_session = session_spent + new_cost

        status = {
            "daily": {
                "spent": float(new_daily),
                "limit": float(self.budget.daily_limit),
                "percentage": float(new_daily / self.budget.daily_limit * 100),
            },
            "monthly": {
                "spent": float(new_monthly),
                "limit": float(self.budget.monthly_limit),
                "percentage": float(new_monthly / self.budget.monthly_limit * 100),
            },
            "session": {
                "spent": float(new_session),
                "limit": float(self.budget.per_generation_limit),
                "percentage": float(
                    new_session / self.budget.per_generation_limit * 100
                ),
            },
            "alerts": [],
        }

        # Check for alerts
        for period in ["daily", "monthly", "session"]:
            pct = status[period]["percentage"] / 100
            if pct >= self.budget.hard_stop_threshold:
                status["alerts"].append(
                    {
                        "level": "critical",
                        "message": f"{period.capitalize()} budget hard stop reached ({pct:.0%})",
                    }
                )
            elif pct >= self.budget.alert_threshold:
                status["alerts"].append(
                    {
                        "level": "warning",
                        "message": f"{period.capitalize()} budget alert threshold reached ({pct:.0%})",
                    }
                )

        return status

    async def get_daily_spending(self, date: datetime | None = None) -> Decimal:
        """Get total spending for a specific day.

        Args:
            date: Date to check (defaults to today)

        Returns:
            Total spending for the day
        """
        if date is None:
            date = datetime.utcnow()

        start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)

        async with self.db_manager.get_connection() as conn:
            result = await conn.fetchval(
                """
                SELECT COALESCE(SUM(cost), 0)
                FROM golden_testset_token_usage
                WHERE timestamp >= $1 AND timestamp < $2
            """,
                start_of_day,
                end_of_day,
            )

        return Decimal(str(result))

    async def get_monthly_spending(self, date: datetime | None = None) -> Decimal:
        """Get total spending for a specific month.

        Args:
            date: Date in the month to check (defaults to current month)

        Returns:
            Total spending for the month
        """
        if date is None:
            date = datetime.utcnow()

        start_of_month = date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        # Calculate end of month
        if date.month == 12:
            end_of_month = start_of_month.replace(year=date.year + 1, month=1)
        else:
            end_of_month = start_of_month.replace(month=date.month + 1)

        async with self.db_manager.get_connection() as conn:
            result = await conn.fetchval(
                """
                SELECT COALESCE(SUM(cost), 0)
                FROM golden_testset_token_usage
                WHERE timestamp >= $1 AND timestamp < $2
            """,
                start_of_month,
                end_of_month,
            )

        return Decimal(str(result))

    async def get_session_spending(self, session_id: str | None = None) -> Decimal:
        """Get total spending for a session.

        Args:
            session_id: Session to check (defaults to current session)

        Returns:
            Total spending for the session
        """
        session = session_id or self.current_session
        if not session:
            return Decimal(0)

        async with self.db_manager.get_connection() as conn:
            result = await conn.fetchval(
                """
                SELECT COALESCE(SUM(cost), 0)
                FROM golden_testset_token_usage
                WHERE session_id = $1
            """,
                session,
            )

        return Decimal(str(result))

    async def end_session(self) -> dict[str, Any]:
        """End the current cost tracking session.

        Returns:
            Session summary with total costs and usage
        """
        if not self.current_session:
            raise ValueError("No active session to end")

        session_id = self.current_session

        # Calculate session totals
        total_cost = await self.get_session_spending(session_id)
        usage_summary = await self.get_session_summary(session_id)

        # Update session status
        async with self.db_manager.get_connection() as conn:
            await conn.execute(
                """
                UPDATE golden_testset_cost_tracking
                SET ended_at = $1, status = 'completed', total_cost = $2
                WHERE session_id = $3
            """,
                datetime.utcnow(),
                float(total_cost),
                session_id,
            )

        # Clear current session
        self.current_session = None

        logger.info(f"Ended session {session_id}: Total cost ${total_cost}")

        return {
            "session_id": session_id,
            "total_cost": float(total_cost),
            "usage_summary": usage_summary,
        }

    async def get_session_summary(self, session_id: str) -> dict[str, Any]:
        """Get detailed summary for a session.

        Args:
            session_id: Session to summarize

        Returns:
            Detailed usage and cost breakdown
        """
        async with self.db_manager.get_connection() as conn:
            # Get usage by model
            model_usage = await conn.fetch(
                """
                SELECT model,
                       SUM(input_tokens) as total_input,
                       SUM(output_tokens) as total_output,
                       SUM(total_tokens) as total_tokens,
                       SUM(cost) as total_cost,
                       COUNT(*) as api_calls
                FROM golden_testset_token_usage
                WHERE session_id = $1
                GROUP BY model
            """,
                session_id,
            )

            # Get session info
            session_info = await conn.fetchrow(
                """
                SELECT started_at, ended_at, status, total_cost, metadata
                FROM golden_testset_cost_tracking
                WHERE session_id = $1
            """,
                session_id,
            )

        summary = {
            "session_id": session_id,
            "status": session_info["status"] if session_info else "unknown",
            "started_at": session_info["started_at"].isoformat()
            if session_info
            else None,
            "ended_at": session_info["ended_at"].isoformat()
            if session_info and session_info["ended_at"]
            else None,
            "total_cost": float(session_info["total_cost"]) if session_info else 0.0,
            "model_breakdown": [],
        }

        for row in model_usage:
            summary["model_breakdown"].append(
                {
                    "model": row["model"],
                    "input_tokens": row["total_input"],
                    "output_tokens": row["total_output"],
                    "total_tokens": row["total_tokens"],
                    "cost": float(row["total_cost"]),
                    "api_calls": row["api_calls"],
                }
            )

        return summary

    async def get_cost_report(
        self, start_date: datetime, end_date: datetime
    ) -> dict[str, Any]:
        """Generate cost report for a date range.

        Args:
            start_date: Report start date
            end_date: Report end date

        Returns:
            Comprehensive cost report
        """
        async with self.db_manager.get_connection() as conn:
            # Daily breakdown
            daily_costs = await conn.fetch(
                """
                SELECT DATE(timestamp) as date,
                       SUM(cost) as total_cost,
                       SUM(total_tokens) as total_tokens,
                       COUNT(DISTINCT session_id) as sessions
                FROM golden_testset_token_usage
                WHERE timestamp >= $1 AND timestamp < $2
                GROUP BY DATE(timestamp)
                ORDER BY date
            """,
                start_date,
                end_date,
            )

            # Model breakdown
            model_costs = await conn.fetch(
                """
                SELECT model,
                       SUM(cost) as total_cost,
                       SUM(input_tokens) as total_input,
                       SUM(output_tokens) as total_output,
                       COUNT(*) as api_calls
                FROM golden_testset_token_usage
                WHERE timestamp >= $1 AND timestamp < $2
                GROUP BY model
                ORDER BY total_cost DESC
            """,
                start_date,
                end_date,
            )

            # Total summary
            total_summary = await conn.fetchrow(
                """
                SELECT SUM(cost) as total_cost,
                       SUM(total_tokens) as total_tokens,
                       COUNT(DISTINCT session_id) as total_sessions,
                       COUNT(*) as total_api_calls
                FROM golden_testset_token_usage
                WHERE timestamp >= $1 AND timestamp < $2
            """,
                start_date,
                end_date,
            )

        report = {
            "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "summary": {
                "total_cost": float(total_summary["total_cost"] or 0),
                "total_tokens": total_summary["total_tokens"] or 0,
                "total_sessions": total_summary["total_sessions"] or 0,
                "total_api_calls": total_summary["total_api_calls"] or 0,
            },
            "daily_breakdown": [
                {
                    "date": row["date"].isoformat(),
                    "cost": float(row["total_cost"]),
                    "tokens": row["total_tokens"],
                    "sessions": row["sessions"],
                }
                for row in daily_costs
            ],
            "model_breakdown": [
                {
                    "model": row["model"],
                    "cost": float(row["total_cost"]),
                    "input_tokens": row["total_input"],
                    "output_tokens": row["total_output"],
                    "api_calls": row["api_calls"],
                }
                for row in model_costs
            ],
        }

        return report

    def update_pricing(self, model: str, pricing: ModelPricing):
        """Update pricing for a model.

        Args:
            model: Model name
            pricing: New pricing information
        """
        self.pricing[model] = pricing
        logger.info(
            f"Updated pricing for {model}: "
            f"${pricing.input_price_per_1k}/1k input, "
            f"${pricing.output_price_per_1k}/1k output"
        )


async def create_cost_tables():
    """Create database tables for cost tracking."""
    db_manager = DatabaseConnectionManager(ConnectionConfig())
    async with db_manager.get_connection() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS golden_testset_cost_tracking (
                session_id VARCHAR(255) PRIMARY KEY,
                started_at TIMESTAMP NOT NULL,
                ended_at TIMESTAMP,
                status VARCHAR(50) NOT NULL,
                total_cost DECIMAL(10, 4),
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS golden_testset_token_usage (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(255) NOT NULL,
                model VARCHAR(100) NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                total_tokens INTEGER NOT NULL,
                cost DECIMAL(10, 4) NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                metadata JSONB,
                FOREIGN KEY (session_id) REFERENCES golden_testset_cost_tracking(session_id)
            )
        """)

        # Create indexes
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_token_usage_session
            ON golden_testset_token_usage(session_id)
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_token_usage_timestamp
            ON golden_testset_token_usage(timestamp)
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_token_usage_model
            ON golden_testset_token_usage(model)
        """)

    logger.info("Cost tracking tables created successfully")


async def main():
    """CLI entry point for cost tracking."""
    import argparse

    parser = argparse.ArgumentParser(description="Golden testset cost tracking")
    parser.add_argument("--report", action="store_true", help="Generate cost report")
    parser.add_argument("--start", type=str, help="Report start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="Report end date (YYYY-MM-DD)")
    parser.add_argument("--session", type=str, help="Get summary for specific session")
    parser.add_argument(
        "--create-tables", action="store_true", help="Create database tables"
    )

    args = parser.parse_args()

    try:
        if args.create_tables:
            await create_cost_tables()
            print("Cost tracking tables created")

        elif args.report:
            tracker = CostTracker()

            if args.start and args.end:
                start = datetime.fromisoformat(args.start)
                end = datetime.fromisoformat(args.end)
            else:
                # Default to current month
                end = datetime.utcnow()
                start = end.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

            report = await tracker.get_cost_report(start, end)
            print(json.dumps(report, indent=2))

        elif args.session:
            tracker = CostTracker()
            summary = await tracker.get_session_summary(args.session)
            print(json.dumps(summary, indent=2))

        else:
            parser.print_help()

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(asyncio.run(main()))
