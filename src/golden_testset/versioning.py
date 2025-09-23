"""
Semantic Versioning System for Golden Testsets

Implements semantic versioning logic with automatic version bumping,
conflict detection, and validation rules.

This module provides:
- Semantic version parsing and comparison
- Automatic version bumping based on change types
- Version conflict detection and resolution
- Validation of version progression rules

Example usage:
    from golden_testset.versioning import VersionManager, VersionBump

    version_mgr = VersionManager()

    # Parse versions
    v1 = version_mgr.parse_version("1.2.3-beta")
    v2 = version_mgr.parse_version("1.2.4")

    # Compare versions
    if version_mgr.compare_versions(v1, v2) < 0:
        print(f"{v1} is older than {v2}")

    # Bump version
    new_version = version_mgr.bump_version(v1, VersionBump.MINOR)
    print(f"Bumped {v1} to {new_version}")
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Tuple, Dict, Any
from datetime import datetime


class VersionBump(Enum):
    """Types of version bumps following semantic versioning"""
    MAJOR = "major"      # Breaking changes: x.0.0
    MINOR = "minor"      # New features: x.y.0
    PATCH = "patch"      # Bug fixes: x.y.z


class VersionValidationError(Exception):
    """Raised when version validation fails"""
    pass


class VersionConflictError(Exception):
    """Raised when version conflicts are detected"""
    pass


@dataclass
class SemanticVersion:
    """Represents a semantic version with optional pre-release label"""
    major: int
    minor: int
    patch: int
    label: Optional[str] = None

    def __post_init__(self):
        """Validate version components"""
        if self.major < 0 or self.minor < 0 or self.patch < 0:
            raise VersionValidationError("Version components must be non-negative")

    def __str__(self) -> str:
        """String representation of version"""
        base = f"{self.major}.{self.minor}.{self.patch}"
        if self.label:
            return f"{base}-{self.label}"
        return base

    def __repr__(self) -> str:
        return f"SemanticVersion({self.major}, {self.minor}, {self.patch}, {self.label!r})"

    def __eq__(self, other) -> bool:
        """Equality comparison"""
        if not isinstance(other, SemanticVersion):
            return False
        return (self.major == other.major and
                self.minor == other.minor and
                self.patch == other.patch and
                self.label == other.label)

    def __lt__(self, other) -> bool:
        """Less than comparison for sorting"""
        if not isinstance(other, SemanticVersion):
            return NotImplemented
        return self._compare_to(other) < 0

    def __le__(self, other) -> bool:
        """Less than or equal comparison"""
        if not isinstance(other, SemanticVersion):
            return NotImplemented
        return self._compare_to(other) <= 0

    def __gt__(self, other) -> bool:
        """Greater than comparison"""
        if not isinstance(other, SemanticVersion):
            return NotImplemented
        return self._compare_to(other) > 0

    def __ge__(self, other) -> bool:
        """Greater than or equal comparison"""
        if not isinstance(other, SemanticVersion):
            return NotImplemented
        return self._compare_to(other) >= 0

    def _compare_to(self, other: 'SemanticVersion') -> int:
        """
        Compare this version to another

        Returns:
            -1 if this < other
             0 if this == other
             1 if this > other
        """
        # Compare major.minor.patch first
        if self.major != other.major:
            return -1 if self.major < other.major else 1
        if self.minor != other.minor:
            return -1 if self.minor < other.minor else 1
        if self.patch != other.patch:
            return -1 if self.patch < other.patch else 1

        # Handle labels (pre-release versions)
        # No label (release) > label (pre-release)
        if self.label is None and other.label is not None:
            return 1
        if self.label is not None and other.label is None:
            return -1
        if self.label is None and other.label is None:
            return 0

        # Both have labels - compare lexicographically
        if self.label < other.label:
            return -1
        elif self.label > other.label:
            return 1
        return 0

    @property
    def is_prerelease(self) -> bool:
        """Check if this is a pre-release version"""
        return self.label is not None

    @property
    def is_stable(self) -> bool:
        """Check if this is a stable release (major > 0, no label)"""
        return self.major > 0 and self.label is None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'major': self.major,
            'minor': self.minor,
            'patch': self.patch,
            'label': self.label
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SemanticVersion':
        """Create from dictionary"""
        return cls(**data)


class VersionManager:
    """
    Manages semantic versioning operations for golden testsets

    Features:
    - Parse version strings into SemanticVersion objects
    - Compare versions according to semantic versioning rules
    - Bump versions based on change types
    - Validate version progressions
    - Detect and resolve version conflicts
    """

    # Regex pattern for semantic version parsing
    VERSION_PATTERN = re.compile(
        r'^(?P<major>0|[1-9]\d*)'
        r'\.(?P<minor>0|[1-9]\d*)'
        r'\.(?P<patch>0|[1-9]\d*)'
        r'(?:-(?P<label>[0-9A-Za-z\-]+(?:\.[0-9A-Za-z\-]+)*))?$'
    )

    def __init__(self):
        self.validation_rules = {
            'allow_prereleases': True,
            'require_progression': True,  # New versions must be greater than previous
            'allow_downgrades': False,    # Don't allow version downgrades
            'max_major': 999,            # Maximum major version
            'reserved_labels': ['dev', 'test', 'internal']  # Reserved pre-release labels
        }

    def parse_version(self, version_string: str) -> SemanticVersion:
        """
        Parse a version string into a SemanticVersion object

        Args:
            version_string: Version string (e.g., "1.2.3", "2.0.0-beta")

        Returns:
            SemanticVersion object

        Raises:
            VersionValidationError: If version string is invalid

        Examples:
            >>> vm = VersionManager()
            >>> v = vm.parse_version("1.2.3-beta")
            >>> print(v.major, v.minor, v.patch, v.label)
            1 2 3 beta
        """
        if not isinstance(version_string, str):
            raise VersionValidationError(f"Version must be a string, got {type(version_string)}")

        match = self.VERSION_PATTERN.match(version_string.strip())
        if not match:
            raise VersionValidationError(
                f"Invalid version format: '{version_string}'. "
                f"Expected format: 'major.minor.patch' or 'major.minor.patch-label'"
            )

        try:
            major = int(match.group('major'))
            minor = int(match.group('minor'))
            patch = int(match.group('patch'))
            label = match.group('label')

            # Validate against rules
            if major > self.validation_rules['max_major']:
                raise VersionValidationError(f"Major version {major} exceeds maximum {self.validation_rules['max_major']}")

            if label and label in self.validation_rules['reserved_labels']:
                raise VersionValidationError(f"Label '{label}' is reserved")

            return SemanticVersion(major, minor, patch, label)

        except ValueError as e:
            raise VersionValidationError(f"Invalid version components: {e}")

    def compare_versions(self, v1: SemanticVersion, v2: SemanticVersion) -> int:
        """
        Compare two semantic versions

        Args:
            v1: First version
            v2: Second version

        Returns:
            -1 if v1 < v2, 0 if v1 == v2, 1 if v1 > v2

        Examples:
            >>> vm = VersionManager()
            >>> v1 = vm.parse_version("1.0.0")
            >>> v2 = vm.parse_version("1.0.1")
            >>> vm.compare_versions(v1, v2)  # Returns -1
        """
        return v1._compare_to(v2)

    def bump_version(
        self,
        current: SemanticVersion,
        bump_type: VersionBump,
        label: Optional[str] = None
    ) -> SemanticVersion:
        """
        Bump a version according to semantic versioning rules

        Args:
            current: Current version
            bump_type: Type of bump (MAJOR, MINOR, PATCH)
            label: Optional pre-release label for new version

        Returns:
            New version with appropriate components incremented

        Examples:
            >>> vm = VersionManager()
            >>> v = vm.parse_version("1.2.3")
            >>> vm.bump_version(v, VersionBump.MINOR)
            SemanticVersion(1, 3, 0, None)
        """
        if bump_type == VersionBump.MAJOR:
            return SemanticVersion(current.major + 1, 0, 0, label)
        elif bump_type == VersionBump.MINOR:
            return SemanticVersion(current.major, current.minor + 1, 0, label)
        elif bump_type == VersionBump.PATCH:
            return SemanticVersion(current.major, current.minor, current.patch + 1, label)
        else:
            raise ValueError(f"Invalid bump type: {bump_type}")

    def validate_progression(
        self,
        current: SemanticVersion,
        new: SemanticVersion,
        allow_equal: bool = False
    ) -> bool:
        """
        Validate that a new version follows progression rules

        Args:
            current: Current version
            new: Proposed new version
            allow_equal: Whether to allow same version (for idempotent operations)

        Returns:
            True if progression is valid

        Raises:
            VersionValidationError: If progression is invalid
        """
        comparison = self.compare_versions(current, new)

        if not allow_equal and comparison == 0:
            raise VersionValidationError(
                f"New version {new} is the same as current version {current}"
            )

        if self.validation_rules['require_progression'] and comparison > 0:
            if not self.validation_rules['allow_downgrades']:
                raise VersionValidationError(
                    f"New version {new} is older than current version {current}. "
                    f"Downgrades are not allowed."
                )

        # Additional rules for pre-releases
        if new.is_prerelease and not self.validation_rules['allow_prereleases']:
            raise VersionValidationError(
                f"Pre-release versions are not allowed: {new}"
            )

        return True

    def get_next_versions(
        self,
        current: SemanticVersion,
        include_prereleases: bool = True
    ) -> Dict[str, SemanticVersion]:
        """
        Get possible next versions for a given current version

        Args:
            current: Current version
            include_prereleases: Whether to include pre-release options

        Returns:
            Dictionary of bump type to next version

        Examples:
            >>> vm = VersionManager()
            >>> v = vm.parse_version("1.2.3")
            >>> next_versions = vm.get_next_versions(v)
            >>> next_versions['major']
            SemanticVersion(2, 0, 0, None)
        """
        next_versions = {}

        # Standard bumps
        next_versions['major'] = self.bump_version(current, VersionBump.MAJOR)
        next_versions['minor'] = self.bump_version(current, VersionBump.MINOR)
        next_versions['patch'] = self.bump_version(current, VersionBump.PATCH)

        # Pre-release versions if enabled
        if include_prereleases and self.validation_rules['allow_prereleases']:
            next_versions['major_alpha'] = self.bump_version(current, VersionBump.MAJOR, 'alpha')
            next_versions['major_beta'] = self.bump_version(current, VersionBump.MAJOR, 'beta')
            next_versions['major_rc'] = self.bump_version(current, VersionBump.MAJOR, 'rc')

            next_versions['minor_alpha'] = self.bump_version(current, VersionBump.MINOR, 'alpha')
            next_versions['minor_beta'] = self.bump_version(current, VersionBump.MINOR, 'beta')
            next_versions['minor_rc'] = self.bump_version(current, VersionBump.MINOR, 'rc')

        return next_versions

    def detect_conflicts(
        self,
        existing_versions: List[SemanticVersion],
        proposed_version: SemanticVersion
    ) -> List[str]:
        """
        Detect potential conflicts with existing versions

        Args:
            existing_versions: List of existing versions
            proposed_version: Version being proposed

        Returns:
            List of conflict descriptions (empty if no conflicts)
        """
        conflicts = []

        # Check for exact duplicates
        for existing in existing_versions:
            if existing == proposed_version:
                conflicts.append(f"Version {proposed_version} already exists")

        # Check for version ordering issues
        sorted_versions = sorted(existing_versions)
        if sorted_versions:
            latest = sorted_versions[-1]

            # Warning if proposed version is much older than latest
            if self.compare_versions(proposed_version, latest) < 0:
                major_diff = latest.major - proposed_version.major
                if major_diff > 1:
                    conflicts.append(
                        f"Proposed version {proposed_version} is {major_diff} major versions "
                        f"behind latest {latest}"
                    )

        # Check for suspicious jumps
        if sorted_versions:
            latest = sorted_versions[-1]
            comparison = self.compare_versions(proposed_version, latest)

            if comparison > 0:  # Proposed is newer
                major_jump = proposed_version.major - latest.major
                minor_jump = proposed_version.minor - latest.minor

                if major_jump > 1:
                    conflicts.append(
                        f"Large major version jump from {latest} to {proposed_version}"
                    )
                elif major_jump == 0 and minor_jump > 5:
                    conflicts.append(
                        f"Large minor version jump from {latest} to {proposed_version}"
                    )

        return conflicts

    def suggest_version(
        self,
        existing_versions: List[SemanticVersion],
        change_description: str = "",
        prefer_type: Optional[VersionBump] = None
    ) -> SemanticVersion:
        """
        Suggest an appropriate next version based on existing versions and changes

        Args:
            existing_versions: List of existing versions
            change_description: Description of changes being made
            prefer_type: Preferred bump type (if any)

        Returns:
            Suggested next version
        """
        if not existing_versions:
            # First version
            return SemanticVersion(1, 0, 0)

        # Get latest version
        latest = max(existing_versions)

        # Use preferred type if specified
        if prefer_type:
            return self.bump_version(latest, prefer_type)

        # Analyze change description for hints
        description_lower = change_description.lower()

        # Keywords that suggest major changes
        major_keywords = ['breaking', 'incompatible', 'remove', 'deprecated', 'major']
        # Keywords that suggest minor changes
        minor_keywords = ['feature', 'add', 'new', 'enhance', 'minor']
        # Keywords that suggest patch changes
        patch_keywords = ['fix', 'bug', 'patch', 'typo', 'correction']

        if any(keyword in description_lower for keyword in major_keywords):
            return self.bump_version(latest, VersionBump.MAJOR)
        elif any(keyword in description_lower for keyword in minor_keywords):
            return self.bump_version(latest, VersionBump.MINOR)
        elif any(keyword in description_lower for keyword in patch_keywords):
            return self.bump_version(latest, VersionBump.PATCH)
        else:
            # Default to patch for unclear changes
            return self.bump_version(latest, VersionBump.PATCH)

    def get_version_history(
        self,
        versions: List[Tuple[SemanticVersion, datetime, str]]
    ) -> List[Dict[str, Any]]:
        """
        Generate version history with metadata

        Args:
            versions: List of (version, timestamp, description) tuples

        Returns:
            List of version history entries with analysis
        """
        history = []
        sorted_versions = sorted(versions, key=lambda x: x[1])  # Sort by timestamp

        for i, (version, timestamp, description) in enumerate(sorted_versions):
            entry = {
                'version': str(version),
                'timestamp': timestamp.isoformat(),
                'description': description,
                'is_prerelease': version.is_prerelease,
                'is_stable': version.is_stable
            }

            # Add progression analysis
            if i > 0:
                prev_version, prev_timestamp, _ = sorted_versions[i - 1]

                # Calculate time between versions
                time_diff = timestamp - prev_timestamp
                entry['days_since_previous'] = time_diff.days

                # Determine bump type
                comparison = self.compare_versions(prev_version, version)
                if comparison < 0:
                    if version.major > prev_version.major:
                        entry['bump_type'] = 'major'
                    elif version.minor > prev_version.minor:
                        entry['bump_type'] = 'minor'
                    elif version.patch > prev_version.patch:
                        entry['bump_type'] = 'patch'
                    else:
                        entry['bump_type'] = 'unknown'
                else:
                    entry['bump_type'] = 'none' if comparison == 0 else 'downgrade'

            history.append(entry)

        return history


# =========================================================================
# Utility Functions
# =========================================================================

def parse_version_string(version_string: str) -> SemanticVersion:
    """Convenience function to parse a version string"""
    vm = VersionManager()
    return vm.parse_version(version_string)


def compare_version_strings(v1: str, v2: str) -> int:
    """Convenience function to compare version strings"""
    vm = VersionManager()
    version1 = vm.parse_version(v1)
    version2 = vm.parse_version(v2)
    return vm.compare_versions(version1, version2)


def bump_version_string(
    version_string: str,
    bump_type: VersionBump,
    label: Optional[str] = None
) -> str:
    """Convenience function to bump a version string"""
    vm = VersionManager()
    current = vm.parse_version(version_string)
    new_version = vm.bump_version(current, bump_type, label)
    return str(new_version)


# =========================================================================
# Testing and Examples
# =========================================================================

if __name__ == "__main__":
    def run_examples():
        """Run example usage of the versioning system"""
        print("🔢 Semantic Versioning System Examples")
        print("=" * 50)

        vm = VersionManager()

        # Example 1: Parse versions
        print("\n1. Parsing versions:")
        versions = ["1.0.0", "1.2.3-beta", "2.0.0-rc.1", "0.1.0"]
        parsed = []
        for v_str in versions:
            v = vm.parse_version(v_str)
            parsed.append(v)
            print(f"   {v_str} -> {v} (prerelease: {v.is_prerelease})")

        # Example 2: Compare versions
        print("\n2. Comparing versions:")
        for i in range(len(parsed) - 1):
            v1, v2 = parsed[i], parsed[i + 1]
            comparison = vm.compare_versions(v1, v2)
            op = "<" if comparison < 0 else ">" if comparison > 0 else "=="
            print(f"   {v1} {op} {v2}")

        # Example 3: Bump versions
        print("\n3. Bumping versions:")
        base = vm.parse_version("1.2.3")
        for bump_type in VersionBump:
            bumped = vm.bump_version(base, bump_type)
            print(f"   {base} + {bump_type.value} -> {bumped}")

        # Example 4: Get next versions
        print("\n4. Next version options:")
        next_versions = vm.get_next_versions(base)
        for bump_type, version in next_versions.items():
            print(f"   {bump_type}: {version}")

        # Example 5: Conflict detection
        print("\n5. Conflict detection:")
        existing = [vm.parse_version(v) for v in ["1.0.0", "1.1.0", "1.2.0"]]
        conflicts = vm.detect_conflicts(existing, vm.parse_version("1.1.0"))
        print(f"   Conflicts for 1.1.0: {conflicts}")

        # Example 6: Version suggestion
        print("\n6. Version suggestion:")
        suggested = vm.suggest_version(existing, "Added new feature for user management")
        print(f"   Suggested version: {suggested}")

        print("\n✅ Examples completed successfully!")

    run_examples()