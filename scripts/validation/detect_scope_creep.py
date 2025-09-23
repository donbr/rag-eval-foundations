#!/usr/bin/env python3
"""
Scope Creep Detection Script

Monitors implementation scope during development and detects when work
exceeds the intended phase boundaries defined in .claude/tasks.yaml.

Usage:
    python scripts/validation/detect_scope_creep.py --phase current --warn-threshold 80%
    python scripts/validation/detect_scope_creep.py --phase phase2 --detailed-analysis
    python scripts/validation/detect_scope_creep.py --phase current --strict
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml


class ScopeCreepDetector:
    """Detects scope creep by analyzing file changes against phase boundaries"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.tasks_yaml_path = self.project_root / ".claude" / "tasks.yaml"

        if not self.tasks_yaml_path.exists():
            raise FileNotFoundError(f"Tasks file not found: {self.tasks_yaml_path}")

        with open(self.tasks_yaml_path) as f:
            self.tasks_config = yaml.safe_load(f)

        self.phase_scope_map = self._build_phase_scope_map()

    def _build_phase_scope_map(self) -> Dict[str, Dict]:
        """Build mapping of phases to their allowed file scopes"""
        scope_map = {
            "phase1": {
                "name": "Database Schema & Infrastructure",
                "allowed_paths": [
                    "scripts/db/",
                    "schemas/",
                    "src/golden_testset/__init__.py",  # Basic structure only
                ],
                "allowed_files": [
                    "scripts/db/assert_schema.py",
                    "scripts/db/assert_indexes.py",
                    "scripts/db/dry_run_sql.py",
                    "scripts/test_connection_pool.py",
                    "schemas/golden_testset_schema.sql",
                    "schemas/golden_testset_indexes.sql",
                    "schemas/rollback_golden_testset.sql",
                ],
                "forbidden_patterns": [
                    "*/manager.py",
                    "*/versioning.py",
                    "*/quality_validator.py",
                    "*/phoenix_integration.py",
                    "*/cli.py",
                ]
            },
            "phase2": {
                "name": "Core Manager & Versioning",
                "allowed_paths": [
                    "scripts/db/",  # Inherited from Phase 1
                    "schemas/",     # Inherited from Phase 1
                    "src/golden_testset/",
                    "tests/unit/",
                ],
                "allowed_files": [
                    "src/golden_testset/manager.py",
                    "src/golden_testset/versioning.py",
                    "src/golden_testset/change_detector.py",
                    "src/golden_testset/transactions.py",
                    "src/golden_testset/__init__.py",
                    "tests/unit/test_golden_testset_manager.py",
                    "tests/unit/test_versioning.py",
                    "tests/unit/test_change_detector.py",
                ],
                "forbidden_patterns": [
                    "*/quality_validator.py",
                    "*/validation_pipeline.py",
                    "*/phoenix_integration.py",
                    "*/cost_tracker.py",
                    "*/cli.py",
                ]
            },
            "phase3": {
                "name": "Quality Validation Pipeline",
                "allowed_paths": [
                    "scripts/db/",     # Inherited
                    "schemas/",        # Inherited
                    "src/golden_testset/", # Inherited + new files
                    "tests/unit/",     # Inherited
                    "tests/integration/",
                ],
                "allowed_files": [
                    "src/golden_testset/quality_validator.py",
                    "src/golden_testset/validation_pipeline.py",
                    "tests/unit/test_quality_validator.py",
                    "tests/unit/test_validation_pipeline.py",
                ],
                "forbidden_patterns": [
                    "*/phoenix_integration.py",
                    "*/cost_tracker.py",
                    "*/tracing.py",
                    "*/cli.py",
                ]
            },
            "phase4": {
                "name": "Phoenix & Cost Integration",
                "allowed_files": [
                    "src/golden_testset/phoenix_integration.py",
                    "src/golden_testset/cost_tracker.py",
                    "src/golden_testset/tracing.py",
                    "tests/integration/test_phoenix.py",
                    "tests/test_cost_tracker.py",
                    "tests/test_tracing.py",
                ],
                "forbidden_patterns": [
                    "*/cli.py",
                    "workflows/*",
                ]
            },
            "phase5": {
                "name": "CLI Tools & Automation",
                "allowed_files": [
                    "src/golden_testset/cli.py",
                    "scripts/testset_manager.sh",
                    "workflows/",
                    "tests/test_cli.py",
                    "tests/integration/test_workflows.py",
                ],
                "forbidden_patterns": []
            }
        }
        return scope_map

    def get_current_branch(self) -> str:
        """Get current Git branch name"""
        try:
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True,
                text=True,
                check=True,
                cwd=self.project_root
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to get current branch: {e}")

    def parse_branch_phase(self, branch_name: str) -> Optional[str]:
        """Extract phase from branch name"""
        if not branch_name.startswith("feature/phase"):
            return None

        parts = branch_name.replace("feature/phase", "").split("-", 1)
        if not parts:
            return None

        try:
            phase_num = int(parts[0])
            return f"phase{phase_num}"
        except ValueError:
            return None

    def get_modified_files(self) -> List[str]:
        """Get list of modified files in current branch"""
        try:
            # Get all files tracked by git in the project
            result = subprocess.run(
                ["git", "ls-files"],
                capture_output=True,
                text=True,
                check=True,
                cwd=self.project_root
            )
            tracked_files = result.stdout.strip().split('\n')

            # Also check for untracked files that might be new
            result = subprocess.run(
                ["git", "ls-files", "--others", "--exclude-standard"],
                capture_output=True,
                text=True,
                check=True,
                cwd=self.project_root
            )
            untracked_files = result.stdout.strip().split('\n')

            all_files = [f for f in tracked_files + untracked_files if f.strip()]
            return all_files

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to get modified files: {e}")

    def check_file_scope_compliance(self, phase: str, files: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """Check if files comply with phase scope"""
        if phase not in self.phase_scope_map:
            return [], [], [f"Unknown phase: {phase}"]

        scope = self.phase_scope_map[phase]
        compliant = []
        violations = []
        warnings = []

        for file_path in files:
            if not file_path.strip():
                continue

            # Check if file is explicitly forbidden
            is_forbidden = False
            for pattern in scope.get("forbidden_patterns", []):
                if self._matches_pattern(file_path, pattern):
                    violations.append(f"{file_path} (forbidden by pattern: {pattern})")
                    is_forbidden = True
                    break

            if is_forbidden:
                continue

            # Check if file is explicitly allowed
            is_allowed = False

            # Check allowed files
            if file_path in scope.get("allowed_files", []):
                compliant.append(file_path)
                is_allowed = True
                continue

            # Check allowed paths
            for allowed_path in scope.get("allowed_paths", []):
                if file_path.startswith(allowed_path):
                    compliant.append(file_path)
                    is_allowed = True
                    break

            if not is_allowed:
                # Check if this might belong to a later phase
                future_phase = self._suggest_correct_phase(file_path)
                if future_phase and future_phase != phase:
                    violations.append(f"{file_path} (belongs in {future_phase})")
                else:
                    warnings.append(f"{file_path} (not explicitly defined for {phase})")

        return compliant, violations, warnings

    def _matches_pattern(self, file_path: str, pattern: str) -> bool:
        """Check if file path matches a glob-like pattern"""
        if "*" in pattern:
            # Simple glob matching
            parts = pattern.split("*")
            if len(parts) == 2:
                prefix, suffix = parts
                return file_path.startswith(prefix) and file_path.endswith(suffix)
        return file_path == pattern

    def _suggest_correct_phase(self, file_path: str) -> Optional[str]:
        """Suggest which phase a file belongs to"""
        file_name = Path(file_path).name

        # Pattern-based suggestions
        if "manager.py" in file_name or "versioning.py" in file_name:
            return "phase2"
        elif "quality" in file_name or "validation" in file_name:
            return "phase3"
        elif "phoenix" in file_name or "cost" in file_name or "tracing" in file_name:
            return "phase4"
        elif "cli.py" in file_name or file_path.startswith("workflows/"):
            return "phase5"

        return None

    def calculate_scope_creep_percentage(self, phase: str, files: List[str]) -> float:
        """Calculate percentage of scope creep (0-100%)"""
        compliant, violations, warnings = self.check_file_scope_compliance(phase, files)

        total_files = len([f for f in files if f.strip() and not f.startswith(".")])
        if total_files == 0:
            return 0.0

        violation_count = len(violations)
        return (violation_count / total_files) * 100

    def analyze_scope_creep(self, phase: str, warn_threshold: float = 80.0) -> Dict:
        """Analyze scope creep for a given phase"""
        files = self.get_modified_files()
        compliant, violations, warnings = self.check_file_scope_compliance(phase, files)
        creep_percentage = self.calculate_scope_creep_percentage(phase, files)

        return {
            "phase": phase,
            "phase_name": self.phase_scope_map.get(phase, {}).get("name", phase),
            "total_files": len([f for f in files if f.strip()]),
            "compliant_files": compliant,
            "violations": violations,
            "warnings": warnings,
            "scope_creep_percentage": creep_percentage,
            "exceeds_threshold": creep_percentage > warn_threshold,
            "status": "❌ VIOLATION" if violations else ("⚠️  WARNING" if warnings else "✅ COMPLIANT")
        }


def main():
    parser = argparse.ArgumentParser(description="Detect scope creep in implementation")
    parser.add_argument("--phase", required=True,
                       help="Phase to check (current|phase1|phase2|...)")
    parser.add_argument("--warn-threshold", type=float, default=80.0,
                       help="Warning threshold percentage (default: 80)")
    parser.add_argument("--detailed-analysis", action="store_true",
                       help="Show detailed analysis")
    parser.add_argument("--strict", action="store_true",
                       help="Treat warnings as violations")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")

    args = parser.parse_args()

    try:
        detector = ScopeCreepDetector()

        # Resolve "current" phase
        if args.phase == "current":
            branch_name = detector.get_current_branch()
            args.phase = detector.parse_branch_phase(branch_name)
            if not args.phase:
                print(f"❌ Cannot determine phase from branch: {branch_name}")
                sys.exit(1)

        # Analyze scope creep
        analysis = detector.analyze_scope_creep(args.phase, args.warn_threshold)

        # Report results
        print(f"🔍 Scope Creep Analysis: {analysis['phase']} - {analysis['phase_name']}")
        print(f"Status: {analysis['status']}")
        print(f"Scope creep: {analysis['scope_creep_percentage']:.1f}%")

        if args.detailed_analysis or args.verbose:
            print(f"\n📊 File Analysis ({analysis['total_files']} total files):")

            if analysis['compliant_files']:
                print(f"\n✅ Compliant files ({len(analysis['compliant_files'])}):")
                for file_path in analysis['compliant_files']:
                    print(f"   • {file_path}")

            if analysis['violations']:
                print(f"\n❌ Violations ({len(analysis['violations'])}):")
                for violation in analysis['violations']:
                    print(f"   • {violation}")

            if analysis['warnings']:
                print(f"\n⚠️  Warnings ({len(analysis['warnings'])}):")
                for warning in analysis['warnings']:
                    print(f"   • {warning}")

        # Exit with appropriate code
        has_violations = len(analysis['violations']) > 0
        has_warnings = len(analysis['warnings']) > 0
        exceeds_threshold = analysis['exceeds_threshold']

        if has_violations or (args.strict and has_warnings) or exceeds_threshold:
            if has_violations:
                print(f"\n❌ Scope violations detected")
            if args.strict and has_warnings:
                print(f"\n❌ Warnings treated as violations (--strict mode)")
            if exceeds_threshold:
                print(f"\n❌ Scope creep exceeds threshold ({args.warn_threshold}%)")
            sys.exit(1)

        print("\n✅ No scope creep detected")

    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()