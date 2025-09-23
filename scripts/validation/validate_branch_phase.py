#!/usr/bin/env python3
"""
Branch-Phase Alignment Validation Script

Validates that the current Git branch name aligns with the actual implementation
status according to .claude/tasks.yaml phase definitions.

Usage:
    python scripts/validation/validate_branch_phase.py --current-branch --check-scope
    python scripts/validation/validate_branch_phase.py --new-branch --check-alignment
    python scripts/validation/validate_branch_phase.py --pre-commit
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


class BranchPhaseValidator:
    """Validates branch-phase alignment according to .claude/tasks.yaml"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.tasks_yaml_path = self.project_root / ".claude" / "tasks.yaml"

        if not self.tasks_yaml_path.exists():
            raise FileNotFoundError(f"Tasks file not found: {self.tasks_yaml_path}")

        with open(self.tasks_yaml_path) as f:
            self.tasks_config = yaml.safe_load(f)

        self.phase_definitions = self._extract_phase_definitions()

    def _extract_phase_definitions(self) -> Dict[str, Dict]:
        """Extract phase definitions from tasks.yaml"""
        phases = {}
        for phase in self.tasks_config.get("phases", []):
            phase_id = phase["id"]
            phases[phase_id] = {
                "name": phase["name"],
                "tasks": [task["id"] for task in phase.get("tasks", [])],
                "description": phase["name"]
            }
        return phases

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

    def parse_branch_name(self, branch_name: str) -> Optional[Tuple[str, str]]:
        """Parse branch name to extract phase info"""
        if not branch_name.startswith("feature/phase"):
            return None

        # Parse format: feature/phase{N}-{name}
        parts = branch_name.replace("feature/phase", "").split("-", 1)
        if len(parts) < 2:
            return None

        try:
            phase_num = int(parts[0])
            phase_name = parts[1]
            return f"phase{phase_num}", phase_name
        except ValueError:
            return None

    def detect_implementation_status(self) -> List[str]:
        """Detect which phases have been implemented based on file existence"""
        implemented_phases = []

        # Phase 1: Database Schema & Infrastructure
        if (self.project_root / "scripts" / "db").exists():
            implemented_phases.append("phase1")

        # Phase 2: Core Manager & Versioning
        golden_testset_dir = self.project_root / "src" / "golden_testset"
        if (golden_testset_dir / "manager.py").exists() and \
           (golden_testset_dir / "versioning.py").exists():
            implemented_phases.append("phase2")

        # Phase 3: Quality Validation Pipeline
        if (golden_testset_dir / "quality_validator.py").exists():
            implemented_phases.append("phase3")

        # Phase 4: Phoenix & Cost Integration
        if (golden_testset_dir / "phoenix_integration.py").exists():
            implemented_phases.append("phase4")

        # Phase 5: CLI Tools & Automation
        if (golden_testset_dir / "cli.py").exists():
            implemented_phases.append("phase5")

        return implemented_phases

    def validate_branch_alignment(self, branch_name: str) -> Tuple[bool, str]:
        """Validate that branch name aligns with implementation status"""
        branch_info = self.parse_branch_name(branch_name)
        if not branch_info:
            return False, f"Branch name '{branch_name}' doesn't follow phase naming convention"

        branch_phase, branch_name_part = branch_info
        implemented_phases = self.detect_implementation_status()

        # Check if branch phase exists in tasks.yaml
        if branch_phase not in self.phase_definitions:
            return False, f"Phase '{branch_phase}' not found in .claude/tasks.yaml"

        # Check alignment with implementation
        if not implemented_phases:
            # No implementation yet - Phase 1 branch is appropriate
            if branch_phase == "phase1":
                return True, "✅ Branch aligned: No implementation, Phase 1 branch appropriate"
            else:
                return False, f"❌ No implementation detected, but branch is {branch_phase}"

        latest_implemented = implemented_phases[-1]
        latest_phase_num = int(latest_implemented.replace("phase", ""))
        branch_phase_num = int(branch_phase.replace("phase", ""))

        # Branch should be current implementation + 1, or match current implementation
        if branch_phase_num == latest_phase_num + 1:
            return True, f"✅ Branch aligned: Ready for next phase ({branch_phase}) after {latest_implemented}"
        elif branch_phase_num == latest_phase_num:
            return True, f"✅ Branch aligned: Working on current phase ({branch_phase})"
        elif branch_phase_num < latest_phase_num:
            return False, f"❌ Branch behind implementation: {branch_phase} branch with {latest_implemented} implemented"
        else:
            return False, f"❌ Branch ahead of implementation: {branch_phase} branch but only {latest_implemented} implemented"

    def check_scope_boundaries(self, branch_name: str) -> Tuple[bool, List[str]]:
        """Check if current work stays within phase scope boundaries"""
        branch_info = self.parse_branch_name(branch_name)
        if not branch_info:
            return False, ["Branch name doesn't follow phase convention"]

        branch_phase, _ = branch_info
        warnings = []

        # Check for files that might belong to other phases
        implementation_status = self.detect_implementation_status()

        # Phase scope validation
        if branch_phase == "phase1":
            # Phase 1 should only have database and infrastructure
            if (self.project_root / "src" / "golden_testset" / "manager.py").exists():
                warnings.append("⚠️  manager.py exists - belongs in Phase 2")

        elif branch_phase == "phase2":
            # Phase 2 should not have quality validation or Phoenix integration
            if (self.project_root / "src" / "golden_testset" / "quality_validator.py").exists():
                warnings.append("⚠️  quality_validator.py exists - belongs in Phase 3")

        elif branch_phase == "phase3":
            # Phase 3 should not have Phoenix integration or CLI tools
            if (self.project_root / "src" / "golden_testset" / "phoenix_integration.py").exists():
                warnings.append("⚠️  phoenix_integration.py exists - belongs in Phase 4")

        return len(warnings) == 0, warnings


def main():
    parser = argparse.ArgumentParser(description="Validate branch-phase alignment")
    parser.add_argument("--current-branch", action="store_true",
                       help="Validate current branch alignment")
    parser.add_argument("--new-branch", action="store_true",
                       help="Validate new branch creation")
    parser.add_argument("--check-scope", action="store_true",
                       help="Check scope boundaries")
    parser.add_argument("--check-alignment", action="store_true",
                       help="Check phase alignment")
    parser.add_argument("--pre-commit", action="store_true",
                       help="Pre-commit validation")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")

    args = parser.parse_args()

    try:
        validator = BranchPhaseValidator()

        if args.current_branch or args.pre_commit:
            branch_name = validator.get_current_branch()
            print(f"Current branch: {branch_name}")

            # Validate alignment
            is_aligned, message = validator.validate_branch_alignment(branch_name)
            print(message)

            if not is_aligned:
                print("\n🔧 Suggested actions:")
                implementation_status = validator.detect_implementation_status()
                if implementation_status:
                    latest = implementation_status[-1]
                    next_phase_num = int(latest.replace("phase", "")) + 1
                    print(f"   git checkout -b feature/phase{next_phase_num}-{{name}} main")
                sys.exit(1)

            # Check scope if requested
            if args.check_scope:
                scope_ok, warnings = validator.check_scope_boundaries(branch_name)
                if warnings:
                    print("\n⚠️  Scope warnings:")
                    for warning in warnings:
                        print(f"   {warning}")
                    if not scope_ok:
                        sys.exit(1)

        if args.new_branch or args.check_alignment:
            branch_name = validator.get_current_branch()
            is_aligned, message = validator.validate_branch_alignment(branch_name)
            print(message)

            if not is_aligned:
                sys.exit(1)

        print("✅ Validation passed")

    except Exception as e:
        print(f"❌ Validation failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()