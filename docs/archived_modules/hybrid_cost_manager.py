"""Hybrid cost management that leverages Phoenix for cost calculation with custom budget controls.

This module provides the optimal architecture that:
1. Uses Phoenix's built-in cost tracking and calculation
2. Maintains custom budget management and alerting
3. Provides unified cost reporting and monitoring
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from decimal import Decimal

from .phoenix_integration import PhoenixIntegration, PhoenixConfig
from .manager import GoldenTestsetManager

logger = logging.getLogger(__name__)


@dataclass
class CostBudget:
    """Budget configuration for cost tracking."""

    daily_limit: Decimal = Decimal("10.00")
    monthly_limit: Decimal = Decimal("300.00")
    per_generation_limit: Decimal = Decimal("5.00")
    alert_threshold: float = 0.8  # Alert at 80% of budget
    hard_stop_threshold: float = 0.95  # Stop at 95% of budget
    currency: str = "USD"


@dataclass
class CostAlert:
    """Cost alert configuration."""

    alert_type: str  # "warning" or "critical"
    threshold_type: str  # "daily", "monthly", "session"
    threshold_percentage: float
    current_amount: Decimal
    limit_amount: Decimal
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def message(self) -> str:
        """Generate alert message."""
        pct = self.current_amount / self.limit_amount * 100
        return (
            f"{self.threshold_type.capitalize()} {self.alert_type}: "
            f"${self.current_amount:.4f} / ${self.limit_amount:.2f} ({pct:.1f}%)"
        )


class HybridCostManager:
    """Hybrid cost management combining Phoenix costs with custom budget controls."""

    def __init__(
        self,
        phoenix_integration: PhoenixIntegration,
        budget: Optional[CostBudget] = None,
    ):
        """Initialize hybrid cost manager.

        Args:
            phoenix_integration: Phoenix integration instance
            budget: Budget configuration
        """
        self.phoenix = phoenix_integration
        self.budget = budget or CostBudget()
        self.current_session: Optional[str] = None
        self.active_alerts: List[CostAlert] = []

    async def start_session(
        self, session_id: str, metadata: Dict[str, Any] = None
    ) -> str:
        """Start a new cost tracking session.

        Args:
            session_id: Unique session identifier
            metadata: Optional session metadata

        Returns:
            Session ID
        """
        self.current_session = session_id

        # Set up session context for Phoenix tracing
        # This would typically be done via OpenTelemetry context
        logger.info(f"Started hybrid cost tracking session: {session_id}")

        # Clear previous alerts for new session
        self.active_alerts = [
            alert for alert in self.active_alerts if alert.threshold_type != "session"
        ]

        return session_id

    async def track_operation_cost(
        self, operation_type: str, trace_id: str
    ) -> Dict[str, Any]:
        """Track cost for a specific operation using Phoenix.

        Args:
            operation_type: Type of operation (e.g., "generation", "embedding", "rerank")
            trace_id: Phoenix trace ID

        Returns:
            Cost summary with budget status
        """
        if not self.current_session:
            raise ValueError("No active session. Call start_session first.")

        # Get cost from Phoenix
        trace_costs = await self.phoenix.get_trace_costs([trace_id])
        trace_cost = trace_costs.get(trace_id, {})

        operation_cost = Decimal(str(trace_cost.get("total", 0)))

        # Check budget status
        budget_status = await self._check_budget_with_new_cost(operation_cost)

        # Generate alerts if necessary
        await self._process_budget_alerts(budget_status)

        return {
            "operation_type": operation_type,
            "trace_id": trace_id,
            "cost": float(operation_cost),
            "prompt_cost": trace_cost.get("prompt", 0),
            "completion_cost": trace_cost.get("completion", 0),
            "budget_status": budget_status,
            "alerts": [alert.message for alert in self.active_alerts],
        }

    async def get_session_summary(
        self, session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get cost summary for a session using Phoenix data.

        Args:
            session_id: Session to summarize (defaults to current session)

        Returns:
            Comprehensive session cost summary
        """
        session = session_id or self.current_session
        if not session:
            raise ValueError("No session specified")

        # Get Phoenix session costs
        phoenix_costs = await self.phoenix.get_session_costs(session)

        # Add budget analysis
        total_cost = Decimal(str(phoenix_costs["total_cost"]))
        budget_status = await self._check_budget_with_new_cost(
            Decimal(0)
        )  # Current status

        # Calculate efficiency metrics
        model_breakdown = phoenix_costs.get("model_breakdown", {})
        efficiency_metrics = self._calculate_efficiency_metrics(model_breakdown)

        return {
            **phoenix_costs,
            "budget_analysis": budget_status,
            "efficiency_metrics": efficiency_metrics,
            "cost_per_trace": float(total_cost / max(phoenix_costs["trace_count"], 1)),
            "projected_daily_cost": float(total_cost * 24) if total_cost > 0 else 0.0,
            "budget_remaining": {
                "daily": float(
                    self.budget.daily_limit - await self._get_daily_spending()
                ),
                "monthly": float(
                    self.budget.monthly_limit - await self._get_monthly_spending()
                ),
                "session": float(self.budget.per_generation_limit - total_cost),
            },
        }

    async def _check_budget_with_new_cost(self, new_cost: Decimal) -> Dict[str, Any]:
        """Check budget status with potential new cost.

        Args:
            new_cost: Additional cost to consider

        Returns:
            Budget status analysis
        """
        # Get current spending from Phoenix via session data
        daily_spent = await self._get_daily_spending()
        monthly_spent = await self._get_monthly_spending()
        session_spent = await self._get_session_spending()

        # Calculate new totals
        new_daily = daily_spent + new_cost
        new_monthly = monthly_spent + new_cost
        new_session = session_spent + new_cost

        status = {
            "daily": {
                "spent": float(new_daily),
                "limit": float(self.budget.daily_limit),
                "percentage": float(new_daily / self.budget.daily_limit * 100),
                "remaining": float(self.budget.daily_limit - new_daily),
            },
            "monthly": {
                "spent": float(new_monthly),
                "limit": float(self.budget.monthly_limit),
                "percentage": float(new_monthly / self.budget.monthly_limit * 100),
                "remaining": float(self.budget.monthly_limit - new_monthly),
            },
            "session": {
                "spent": float(new_session),
                "limit": float(self.budget.per_generation_limit),
                "percentage": float(
                    new_session / self.budget.per_generation_limit * 100
                ),
                "remaining": float(self.budget.per_generation_limit - new_session),
            },
            "status": "ok",
        }

        # Determine overall status
        max_percentage = (
            max(
                status["daily"]["percentage"],
                status["monthly"]["percentage"],
                status["session"]["percentage"],
            )
            / 100
        )

        if max_percentage >= self.budget.hard_stop_threshold:
            status["status"] = "critical"
        elif max_percentage >= self.budget.alert_threshold:
            status["status"] = "warning"

        return status

    async def _process_budget_alerts(self, budget_status: Dict[str, Any]):
        """Process budget status and generate alerts.

        Args:
            budget_status: Current budget status
        """
        current_alerts = []

        for period in ["daily", "monthly", "session"]:
            period_data = budget_status[period]
            percentage = period_data["percentage"] / 100

            if percentage >= self.budget.hard_stop_threshold:
                alert = CostAlert(
                    alert_type="critical",
                    threshold_type=period,
                    threshold_percentage=percentage,
                    current_amount=Decimal(str(period_data["spent"])),
                    limit_amount=Decimal(str(period_data["limit"])),
                )
                current_alerts.append(alert)

            elif percentage >= self.budget.alert_threshold:
                alert = CostAlert(
                    alert_type="warning",
                    threshold_type=period,
                    threshold_percentage=percentage,
                    current_amount=Decimal(str(period_data["spent"])),
                    limit_amount=Decimal(str(period_data["limit"])),
                )
                current_alerts.append(alert)

        # Update active alerts (replace alerts of same type)
        self.active_alerts = [
            alert
            for alert in self.active_alerts
            if alert.threshold_type not in [a.threshold_type for a in current_alerts]
        ]
        self.active_alerts.extend(current_alerts)

        # Log new alerts
        for alert in current_alerts:
            if alert.alert_type == "critical":
                logger.critical(alert.message)
            else:
                logger.warning(alert.message)

    async def _get_daily_spending(self) -> Decimal:
        """Get total daily spending from Phoenix."""
        # This would ideally query Phoenix for all traces from today
        # For now, we'll implement a simplified version
        if self.current_session:
            session_costs = await self.phoenix.get_session_costs(self.current_session)
            return Decimal(str(session_costs["total_cost"]))
        return Decimal(0)

    async def _get_monthly_spending(self) -> Decimal:
        """Get total monthly spending from Phoenix."""
        # This would ideally query Phoenix for all traces from current month
        # For now, we'll implement a simplified version
        if self.current_session:
            session_costs = await self.phoenix.get_session_costs(self.current_session)
            return Decimal(str(session_costs["total_cost"]))
        return Decimal(0)

    async def _get_session_spending(self) -> Decimal:
        """Get total session spending from Phoenix."""
        if not self.current_session:
            return Decimal(0)

        session_costs = await self.phoenix.get_session_costs(self.current_session)
        return Decimal(str(session_costs["total_cost"]))

    def _calculate_efficiency_metrics(
        self, model_breakdown: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate cost efficiency metrics.

        Args:
            model_breakdown: Model-wise cost breakdown from Phoenix

        Returns:
            Efficiency metrics
        """
        if not model_breakdown:
            return {}

        total_cost = sum(model["cost"] for model in model_breakdown.values())
        total_tokens = sum(model["total_tokens"] for model in model_breakdown.values())

        metrics = {
            "cost_per_token": float(total_cost / max(total_tokens, 1)),
            "most_expensive_model": max(
                model_breakdown.items(), key=lambda x: x[1]["cost"]
            )[0]
            if model_breakdown
            else None,
            "model_distribution": {
                model: {
                    "cost_percentage": float(data["cost"] / max(total_cost, 1) * 100),
                    "token_percentage": float(
                        data["total_tokens"] / max(total_tokens, 1) * 100
                    ),
                }
                for model, data in model_breakdown.items()
            },
        }

        return metrics

    async def end_session(self) -> Dict[str, Any]:
        """End the current cost tracking session.

        Returns:
            Final session summary
        """
        if not self.current_session:
            raise ValueError("No active session to end")

        session_id = self.current_session

        # Get final summary
        summary = await self.get_session_summary(session_id)

        # Clear session
        self.current_session = None

        # Clear session alerts
        self.active_alerts = [
            alert for alert in self.active_alerts if alert.threshold_type != "session"
        ]

        logger.info(
            f"Ended hybrid cost session {session_id}: Total cost ${summary['total_cost']:.4f}"
        )

        return summary

    async def get_cost_report(
        self, start_date: datetime, end_date: datetime
    ) -> Dict[str, Any]:
        """Generate comprehensive cost report.

        Args:
            start_date: Report start date
            end_date: Report end date

        Returns:
            Comprehensive cost report combining Phoenix data with budget analysis
        """
        # This would query Phoenix for traces in the date range
        # For now, return a structure showing what the report would contain

        report = {
            "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "summary": {
                "total_cost": 0.0,
                "total_traces": 0,
                "average_cost_per_trace": 0.0,
            },
            "budget_analysis": {
                "budget_utilization": 0.0,
                "projected_monthly_cost": 0.0,
                "cost_trends": [],
            },
            "efficiency_analysis": {
                "cost_per_token_trend": [],
                "model_efficiency_ranking": [],
            },
            "recommendations": [],
        }

        return report


async def main():
    """CLI entry point for hybrid cost management."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Hybrid cost management with Phoenix integration"
    )
    parser.add_argument(
        "--setup", action="store_true", help="Setup Phoenix model pricing"
    )
    parser.add_argument(
        "--session-summary", type=str, help="Get summary for session ID"
    )
    parser.add_argument(
        "--budget-status", action="store_true", help="Check current budget status"
    )

    args = parser.parse_args()

    # Initialize components
    phoenix_config = PhoenixConfig()
    manager = GoldenTestsetManager()
    phoenix = PhoenixIntegration(manager, phoenix_config)
    cost_manager = HybridCostManager(phoenix)

    try:
        if args.setup:
            results = await phoenix.setup_default_model_pricing()
            print("Phoenix model pricing setup:")
            for result in results:
                status = "✅" if result["status"] == "success" else "❌"
                print(f"{status} {result['model']}")

        elif args.session_summary:
            summary = await cost_manager.get_session_summary(args.session_summary)
            print(json.dumps(summary, indent=2))

        elif args.budget_status:
            if cost_manager.current_session:
                status = await cost_manager._check_budget_with_new_cost(Decimal(0))
                print("Current budget status:")
                print(json.dumps(status, indent=2))
            else:
                print("No active session")

        else:
            parser.print_help()

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(asyncio.run(main()))
