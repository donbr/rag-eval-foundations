# Archived Golden Testset Modules

These modules were archived on October 8, 2025 as part of codebase consolidation (Phase 4).

## Archived Modules

| Module | Lines | Reason | Future Use |
|--------|-------|--------|------------|
| cost_tracker.py | 725 | No usage in main pipeline or tests | May restore if cost tracking needed |
| tracing.py | 674 | No usage in main pipeline or tests | May restore for debugging |
| validation_pipeline.py | 660 | No usage in main pipeline or tests | May restore for validation features |
| transactions.py | 603 | No usage in main pipeline or tests | May restore for transaction support |
| hybrid_cost_manager.py | 475 | No usage in main pipeline or tests | Superseded by Phoenix native cost tracking |
| optimal_cost_example.py | 263 | Example/demo code | Reference only |

**Total Archived:** 3,400 lines

## Active Modules (Kept)

The following modules remain active because they have test coverage or are used by core scripts:

- ✅ **manager.py** - Core testset management (used by all scripts)
- ✅ **phoenix_integration.py** - Phoenix upload/integration (used by all scripts)
- ✅ **versioning.py** - Semantic versioning (has unit tests in `tests/unit/test_golden_testset_manager.py`)
- ✅ **change_detector.py** - Change detection (has unit tests in `tests/unit/test_golden_testset_manager.py`)
- ✅ **quality_validator.py** - Quality validation (has unit tests in `tests/unit/test_quality_validator.py`)

## Archival Criteria

Modules were archived if they met **ALL** criteria:
1. ❌ Not imported by any core script (experiments, e2e, golden_testset, upload)
2. ❌ Not imported by any test file
3. ❌ Not referenced in flows or validation scripts
4. ❌ No active development or recent commits

## Restoration Process

If you need to restore any archived module:

```bash
# Restore specific module
git mv docs/archived_modules/cost_tracker.py src/golden_testset/

# Update __init__.py to re-export
# Add imports and __all__ entry

# Run tests
pytest tests/unit/
```

## Analysis Details

**Dependency audit performed:** October 7-8, 2025

**Tools used:**
- `grep -r "from golden_testset.<module>" --include="*.py"`
- Static analysis of all Python files in src/, flows/, tests/
- Review of git commit history

**Results:**
- 0 imports found for archived modules (excluding self-references)
- 100% of active modules have either test coverage OR direct usage in main pipeline
- Conservative approach: preserved all tested modules even if not currently used

## References

- Cleanup commit: `git log --oneline | grep "archive unused"`
- Full audit report: See git commit message for dependency analysis
- Related PR: Phase 4 Phoenix Integration + Config Consolidation + Cleanup
