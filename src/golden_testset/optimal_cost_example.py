"""Example usage of the optimal cost tracking architecture.

This demonstrates how to use the hybrid approach that leverages Phoenix's
built-in cost tracking with custom budget management.
"""

import asyncio
import logging
from datetime import datetime

from .hybrid_cost_manager import HybridCostManager, CostBudget
from .phoenix_integration import PhoenixIntegration, PhoenixConfig
from .manager import GoldenTestsetManager
from .tracing import (
    cost_tracked_operation,
    set_cost_tracking_attributes,
    get_current_trace_id,
)

logger = logging.getLogger(__name__)


async def demonstrate_optimal_architecture():
    """Demonstrate the optimal cost tracking architecture."""

    print("🚀 Optimal Cost Tracking Architecture Demo")
    print("=" * 50)

    # 1. Setup Components
    print("\n1️⃣ Setting up components...")

    phoenix_config = PhoenixConfig()
    manager = GoldenTestsetManager()
    phoenix = PhoenixIntegration(manager, phoenix_config)

    # Custom budget configuration
    budget = CostBudget(
        daily_limit=20.00,  # $20/day
        monthly_limit=500.00,  # $500/month
        per_generation_limit=5.00,  # $5/generation
        alert_threshold=0.8,  # Alert at 80%
        hard_stop_threshold=0.95,  # Stop at 95%
    )

    cost_manager = HybridCostManager(phoenix, budget)
    print("✅ HybridCostManager initialized with Phoenix integration")

    # 2. Configure Model Pricing (one-time setup)
    print("\n2️⃣ Configuring model pricing in Phoenix...")
    try:
        pricing_results = await phoenix.setup_default_model_pricing()
        for result in pricing_results:
            status = "✅" if result["status"] == "success" else "❌"
            print(f"   {status} {result['model']}")
    except Exception as e:
        print(f"   ⚠️  Pricing config failed: {e} (expected without Phoenix running)")

    # 3. Start Cost Tracking Session
    print("\n3️⃣ Starting cost tracking session...")
    session_id = "demo_session_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    await cost_manager.start_session(session_id, {"demo": True, "user": "claude-code"})
    print(f"✅ Session started: {session_id}")

    # 4. Simulate Operations with Cost Tracking
    print("\n4️⃣ Simulating LLM operations with Phoenix cost tracking...")

    operations = [
        {
            "name": "golden_testset_generation",
            "model": "gpt-4.1-mini",
            "input": 1500,
            "output": 500,
        },
        {
            "name": "document_embedding",
            "model": "text-embedding-3-small",
            "input": 2000,
            "output": 0,
        },
        {
            "name": "result_reranking",
            "model": "rerank-english-v3.0",
            "input": 800,
            "output": 0,
        },
    ]

    trace_ids = []

    for i, op in enumerate(operations):
        print(f"\n   Operation {i + 1}: {op['name']}")

        # Use Phoenix-integrated cost tracking
        with cost_tracked_operation(
            operation_name=op["name"],
            model=op["model"],
            input_tokens=op["input"],
            output_tokens=op["output"],
            session_id=session_id,
        ):
            # Simulate the actual operation
            await asyncio.sleep(0.1)

            # Get trace ID for cost retrieval
            trace_id = get_current_trace_id()
            if trace_id:
                trace_ids.append(trace_id)
                print(f"   📊 Trace ID: {trace_id}")

                # In real usage, Phoenix would calculate costs automatically
                # Here we simulate the cost tracking
                try:
                    cost_summary = await cost_manager.track_operation_cost(
                        op["name"], trace_id
                    )
                    print(f"   💰 Cost: ${cost_summary.get('cost', 0):.4f}")

                    # Check for budget alerts
                    alerts = cost_summary.get("alerts", [])
                    for alert in alerts:
                        print(f"   🚨 {alert}")

                except Exception as e:
                    print(f"   ⚠️  Cost tracking failed: {e} (expected without Phoenix)")

    # 5. Get Session Summary
    print("\n5️⃣ Getting session cost summary...")
    try:
        summary = await cost_manager.get_session_summary(session_id)
        print(f"   📈 Total cost: ${summary.get('total_cost', 0):.4f}")
        print(f"   📊 Trace count: {summary.get('trace_count', 0)}")
        print(
            f"   🏦 Budget status: {summary.get('budget_analysis', {}).get('status', 'unknown')}"
        )

        # Model breakdown
        model_breakdown = summary.get("model_breakdown", {})
        if model_breakdown:
            print("   📋 Model breakdown:")
            for model, data in model_breakdown.items():
                print(
                    f"      - {model}: ${data.get('cost', 0):.4f} ({data.get('total_tokens', 0)} tokens)"
                )

    except Exception as e:
        print(f"   ⚠️  Session summary failed: {e} (expected without Phoenix)")

    # 6. Demonstrate Budget Management
    print("\n6️⃣ Budget management features...")
    print(f"   💳 Daily limit: ${budget.daily_limit}")
    print(f"   📅 Monthly limit: ${budget.monthly_limit}")
    print(f"   🎯 Per-generation limit: ${budget.per_generation_limit}")
    print(f"   ⚡ Alert threshold: {budget.alert_threshold * 100}%")
    print(f"   🛑 Hard stop threshold: {budget.hard_stop_threshold * 100}%")

    # 7. End Session
    print("\n7️⃣ Ending session...")
    try:
        final_summary = await cost_manager.end_session()
        print(f"✅ Session ended: Total cost ${final_summary.get('total_cost', 0):.4f}")
    except Exception as e:
        print(f"   ⚠️  Session end failed: {e} (expected without Phoenix)")

    print("\n🎉 Optimal Architecture Demo Complete!")
    print("\nKey Benefits:")
    print("✅ Phoenix handles automatic cost calculation")
    print("✅ Custom budget management with alerts")
    print("✅ OpenTelemetry integration for observability")
    print("✅ Session-based cost aggregation")
    print("✅ Model-specific cost tracking")
    print("✅ Real-time budget monitoring")


async def show_migration_benefits():
    """Show the benefits of migrating from custom to hybrid approach."""

    print("\n📊 Migration Benefits Analysis")
    print("=" * 40)

    comparison = [
        ("Cost Calculation", "Manual (3 models)", "Phoenix (unlimited models)"),
        ("Token Tracking", "Manual instrumentation", "Automatic via OpenInference"),
        ("Model Pricing", "Hardcoded in code", "UI-configurable"),
        ("Cost Reports", "Custom SQL queries", "GraphQL API + UI"),
        ("Budget Management", "✅ Custom alerts", "✅ Retained"),
        ("Observability", "Basic logging", "Full Phoenix tracing"),
        ("Maintenance", "High (manual updates)", "Low (Phoenix managed)"),
        ("Accuracy", "Prone to drift", "Always current"),
    ]

    print(f"{'Feature':<20} {'Custom (Old)':<25} {'Hybrid (New)'}")
    print("-" * 70)
    for feature, old, new in comparison:
        print(f"{feature:<20} {old:<25} {new}")

    print("\n💡 Recommendation: Use HybridCostManager for new implementations")
    print("📦 Legacy cost_tracker.py kept for backward compatibility")


def show_usage_examples():
    """Show code examples for the optimal architecture."""

    print("\n💻 Usage Examples")
    print("=" * 30)

    examples = {
        "Basic Setup": """
from src.golden_testset.hybrid_cost_manager import HybridCostManager, CostBudget
from src.golden_testset.phoenix_integration import PhoenixIntegration, PhoenixConfig

# Initialize hybrid cost manager
phoenix = PhoenixIntegration(manager, PhoenixConfig())
cost_manager = HybridCostManager(phoenix, CostBudget(daily_limit=50.00))
""",
        "Cost Tracking": """
from src.golden_testset.tracing import cost_tracked_operation

# Track LLM operation with Phoenix
with cost_tracked_operation("generation", "gpt-4.1-mini", 1000, 300, session_id):
    result = await llm.generate(prompt)
    # Phoenix automatically calculates cost
""",
        "Budget Monitoring": """
# Check budget status
budget_status = await cost_manager._check_budget_with_new_cost(Decimal(2.50))
if budget_status["status"] == "critical":
    raise Exception("Budget limit exceeded!")
""",
        "Session Management": """
# Start session
session_id = await cost_manager.start_session("testset_gen_123")

# ... perform operations ...

# Get summary with Phoenix costs + budget analysis
summary = await cost_manager.get_session_summary()
print(f"Total: ${summary['total_cost']:.4f}")
""",
    }

    for title, code in examples.items():
        print(f"\n📝 {title}:")
        print(code)


async def main():
    """Main demonstration function."""

    logging.basicConfig(level=logging.INFO)

    await demonstrate_optimal_architecture()
    await show_migration_benefits()
    show_usage_examples()

    print(f"\n📚 Next Steps:")
    print("1. Start Phoenix: docker-compose up -d")
    print("2. Run migration: python -m src.golden_testset.cost_tracker --migrate")
    print("3. Use HybridCostManager in your applications")
    print("4. Monitor costs via Phoenix UI: http://localhost:6006")


if __name__ == "__main__":
    asyncio.run(main())
