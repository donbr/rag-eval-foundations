# Archived Flow Implementations

## Prefect Flow Variants

These are reference implementations of the golden testset management flow, archived October 8, 2025.

| File | Size | Description | Status |
|------|------|-------------|--------|
| golden_testset_flow_alternate.py | 9.8KB | Clean Prefect 3.x implementation | Reference |
| golden_testset_flow_prefect3.py | 9.3KB | Pure Prefect 3.x version | Reference |

**Current Implementation:** `flows/golden_testset_flow.py` (16KB hybrid approach)

## Why Archived

The project consolidated on a single hybrid flow (`flows/golden_testset_flow.py`) that combines:
- Clean Prefect 3.x core (1:1 YAML mapping, explicit future resolution)
- Optional enterprise features (Git workflow, quality gates, monitoring, cost tracking)
- Dual CLI modes: Simple development (`--only-phase`) and advanced production (`--production`)
- Flexible composition: Optional enterprise layers

These alternates are kept as reference implementations showing different architectural approaches:

### golden_testset_flow_alternate.py
- **Purpose:** Clean implementation without enterprise features
- **Approach:** Minimalist Prefect 3.x, focuses on core task execution
- **Use Case:** Reference for pure orchestration patterns

### golden_testset_flow_prefect3.py
- **Purpose:** Prefect 3.x migration reference
- **Approach:** Maps tasks.yaml phases 1:1 to Prefect flows
- **Use Case:** Reference for Prefect 3.x async patterns

## Architectural Decision

**Decision:** Use hybrid approach in `flows/golden_testset_flow.py`

**Rationale:**
1. Provides flexibility - simple when needed, powerful when required
2. Clean core execution with optional enterprise features
3. Single codebase serves both development and production use cases
4. Avoids maintaining multiple flow implementations

**Trade-offs:**
- More complex than pure alternatives
- Requires understanding of optional feature flags
- Benefits: Scalability from dev to production

## Restoration

These files can be restored if you want to switch flow implementations:

```bash
# Restore clean alternative
git mv docs/archived_flows/golden_testset_flow_alternate.py flows/

# Or restore Prefect 3.x version
git mv docs/archived_flows/golden_testset_flow_prefect3.py flows/

# Update references in scripts/CI if needed
```

## Related Documentation

- Current hybrid flow: `flows/golden_testset_flow.py`
- Tasks definition: `.claude/tasks.yaml`
- Prefect 3.x docs: https://docs.prefect.io/
- Project decision log: See CLAUDE.md "Golden Testset Flow Architecture Decision"
