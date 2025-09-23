#!/usr/bin/env python3
"""
Change Detection Module for Golden Testset Management

Provides efficient document change detection using content hashing and comparison algorithms.
Supports multiple hashing strategies with performance optimization under 100ms requirement.

Key Features:
- Content-based change detection using SHA-256 and MD5 hashing
- Incremental change detection for large document sets
- Batch processing for performance optimization
- Integration with golden testset versioning system
- Conflict detection and resolution strategies

Performance Requirements:
- Single document change detection: < 10ms
- Batch processing (100 documents): < 100ms
- Hash computation optimization with caching
- Async processing for concurrent operations
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ChangeType(Enum):
    """Types of changes that can be detected"""
    ADDED = "added"           # New document/example added
    MODIFIED = "modified"     # Existing document/example changed
    DELETED = "deleted"       # Document/example removed
    METADATA = "metadata"     # Metadata-only changes
    CONTENT = "content"       # Content changes
    STRUCTURE = "structure"   # Structure/format changes

class HashAlgorithm(Enum):
    """Supported hashing algorithms"""
    MD5 = "md5"
    SHA256 = "sha256"
    SHA512 = "sha512"
    BLAKE2B = "blake2b"

@dataclass
class DocumentHash:
    """Represents a document's hash information"""
    document_id: str
    content_hash: str
    metadata_hash: str
    algorithm: HashAlgorithm
    computed_at: datetime
    file_size: Optional[int] = None
    checksum_parts: Dict[str, str] = field(default_factory=dict)

@dataclass
class ChangeRecord:
    """Record of detected changes"""
    document_id: str
    change_type: ChangeType
    old_hash: Optional[DocumentHash] = None
    new_hash: Optional[DocumentHash] = None
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    details: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0  # Confidence in change detection (0.0 to 1.0)

@dataclass
class ChangeDetectionResult:
    """Result of change detection analysis"""
    total_documents: int
    changes_detected: int
    change_records: List[ChangeRecord]
    processing_time_ms: float
    cache_hit_rate: float = 0.0
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

class HashCache:
    """In-memory cache for document hashes with TTL"""

    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[DocumentHash, datetime]] = {}
        self._access_order: List[str] = []

    def get(self, document_id: str) -> Optional[DocumentHash]:
        """Get hash from cache if not expired"""
        if document_id not in self._cache:
            return None

        doc_hash, cached_at = self._cache[document_id]

        # Check TTL
        if (datetime.now(timezone.utc) - cached_at).total_seconds() > self.ttl_seconds:
            self._remove(document_id)
            return None

        # Update access order
        if document_id in self._access_order:
            self._access_order.remove(document_id)
        self._access_order.append(document_id)

        return doc_hash

    def put(self, document_id: str, doc_hash: DocumentHash) -> None:
        """Store hash in cache with LRU eviction"""
        # Remove if already exists
        if document_id in self._cache:
            self._access_order.remove(document_id)

        # Evict oldest if at capacity
        while len(self._cache) >= self.max_size:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]

        # Add new entry
        self._cache[document_id] = (doc_hash, datetime.now(timezone.utc))
        self._access_order.append(document_id)

    def _remove(self, document_id: str) -> None:
        """Remove entry from cache"""
        if document_id in self._cache:
            del self._cache[document_id]
            if document_id in self._access_order:
                self._access_order.remove(document_id)

    def clear(self) -> None:
        """Clear all cached entries"""
        self._cache.clear()
        self._access_order.clear()

    def size(self) -> int:
        """Get current cache size"""
        return len(self._cache)

class ContentHasher:
    """Efficient content hashing with multiple algorithms"""

    @staticmethod
    def hash_content(content: str, algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> str:
        """Compute hash of content using specified algorithm"""
        content_bytes = content.encode('utf-8')

        if algorithm == HashAlgorithm.MD5:
            return hashlib.md5(content_bytes).hexdigest()
        elif algorithm == HashAlgorithm.SHA256:
            return hashlib.sha256(content_bytes).hexdigest()
        elif algorithm == HashAlgorithm.SHA512:
            return hashlib.sha512(content_bytes).hexdigest()
        elif algorithm == HashAlgorithm.BLAKE2B:
            return hashlib.blake2b(content_bytes).hexdigest()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")

    @staticmethod
    def hash_metadata(metadata: Dict[str, Any], algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> str:
        """Compute hash of metadata dictionary"""
        # Normalize metadata for consistent hashing
        normalized = json.dumps(metadata, sort_keys=True, default=str)
        return ContentHasher.hash_content(normalized, algorithm)

    @staticmethod
    def hash_document_parts(
        content: str,
        metadata: Dict[str, Any],
        algorithm: HashAlgorithm = HashAlgorithm.SHA256
    ) -> Dict[str, str]:
        """Compute hashes for different parts of a document"""
        return {
            'full_content': ContentHasher.hash_content(content, algorithm),
            'metadata': ContentHasher.hash_metadata(metadata, algorithm),
            'content_only': ContentHasher.hash_content(content.strip(), algorithm),
            'structure': ContentHasher.hash_content(
                json.dumps({'type': type(content).__name__, 'length': len(content)}),
                algorithm
            )
        }

class ChangeDetector:
    """Main change detection engine with performance optimization"""

    def __init__(
        self,
        hash_algorithm: HashAlgorithm = HashAlgorithm.SHA256,
        enable_caching: bool = True,
        cache_size: int = 10000,
        cache_ttl: int = 3600,
        performance_target_ms: float = 100.0
    ):
        self.hash_algorithm = hash_algorithm
        self.enable_caching = enable_caching
        self.performance_target_ms = performance_target_ms

        # Initialize cache if enabled
        self.cache = HashCache(cache_size, cache_ttl) if enable_caching else None

        # Performance tracking
        self.performance_metrics = {
            'total_detections': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_processing_time_ms': 0.0,
            'avg_detection_time_ms': 0.0
        }

    async def compute_document_hash(
        self,
        document_id: str,
        content: str,
        metadata: Dict[str, Any]
    ) -> DocumentHash:
        """Compute comprehensive hash for a document"""
        start_time = time.perf_counter()

        # Check cache first
        if self.cache:
            cached = self.cache.get(document_id)
            if cached:
                self.performance_metrics['cache_hits'] += 1
                return cached
            self.performance_metrics['cache_misses'] += 1

        # Compute hashes
        content_hash = ContentHasher.hash_content(content, self.hash_algorithm)
        metadata_hash = ContentHasher.hash_metadata(metadata, self.hash_algorithm)
        checksum_parts = ContentHasher.hash_document_parts(content, metadata, self.hash_algorithm)

        doc_hash = DocumentHash(
            document_id=document_id,
            content_hash=content_hash,
            metadata_hash=metadata_hash,
            algorithm=self.hash_algorithm,
            computed_at=datetime.now(timezone.utc),
            file_size=len(content.encode('utf-8')),
            checksum_parts=checksum_parts
        )

        # Cache result
        if self.cache:
            self.cache.put(document_id, doc_hash)

        # Update metrics
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self.performance_metrics['total_processing_time_ms'] += elapsed_ms
        self.performance_metrics['total_detections'] += 1

        return doc_hash

    async def detect_changes(
        self,
        current_documents: Dict[str, Dict[str, Any]],
        baseline_hashes: Dict[str, DocumentHash]
    ) -> ChangeDetectionResult:
        """Detect changes between current documents and baseline hashes"""
        start_time = time.perf_counter()

        changes: List[ChangeRecord] = []
        current_doc_ids = set(current_documents.keys())
        baseline_doc_ids = set(baseline_hashes.keys())

        # Detect deletions
        deleted_ids = baseline_doc_ids - current_doc_ids
        for doc_id in deleted_ids:
            changes.append(ChangeRecord(
                document_id=doc_id,
                change_type=ChangeType.DELETED,
                old_hash=baseline_hashes[doc_id],
                details={'reason': 'Document not found in current set'}
            ))

        # Detect additions and modifications
        for doc_id, doc_data in current_documents.items():
            content = doc_data.get('content', '')
            metadata = doc_data.get('metadata', {})

            # Compute current hash
            current_hash = await self.compute_document_hash(doc_id, content, metadata)

            if doc_id not in baseline_hashes:
                # New document
                changes.append(ChangeRecord(
                    document_id=doc_id,
                    change_type=ChangeType.ADDED,
                    new_hash=current_hash,
                    details={'reason': 'New document added'}
                ))
            else:
                # Check for modifications
                baseline_hash = baseline_hashes[doc_id]
                change_record = self._analyze_document_changes(
                    doc_id, baseline_hash, current_hash
                )
                if change_record:
                    changes.append(change_record)

        # Calculate metrics
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        cache_hit_rate = 0.0
        if self.cache and (self.performance_metrics['cache_hits'] + self.performance_metrics['cache_misses']) > 0:
            total_cache_requests = self.performance_metrics['cache_hits'] + self.performance_metrics['cache_misses']
            cache_hit_rate = self.performance_metrics['cache_hits'] / total_cache_requests

        return ChangeDetectionResult(
            total_documents=len(current_documents),
            changes_detected=len(changes),
            change_records=changes,
            processing_time_ms=processing_time_ms,
            cache_hit_rate=cache_hit_rate,
            performance_metrics={
                'per_document_avg_ms': processing_time_ms / max(len(current_documents), 1),
                'meets_performance_target': processing_time_ms <= self.performance_target_ms,
                'cache_efficiency': cache_hit_rate
            }
        )

    def _analyze_document_changes(
        self,
        doc_id: str,
        baseline_hash: DocumentHash,
        current_hash: DocumentHash
    ) -> Optional[ChangeRecord]:
        """Analyze specific changes between two document hashes"""
        if baseline_hash.content_hash == current_hash.content_hash:
            # No content changes
            if baseline_hash.metadata_hash != current_hash.metadata_hash:
                # Metadata-only change
                return ChangeRecord(
                    document_id=doc_id,
                    change_type=ChangeType.METADATA,
                    old_hash=baseline_hash,
                    new_hash=current_hash,
                    details={
                        'reason': 'Metadata changed, content unchanged',
                        'content_unchanged': True
                    }
                )
            # No changes detected
            return None

        # Content has changed - analyze type of change
        change_details = self._classify_content_change(baseline_hash, current_hash)

        return ChangeRecord(
            document_id=doc_id,
            change_type=change_details['type'],
            old_hash=baseline_hash,
            new_hash=current_hash,
            details=change_details,
            confidence=change_details.get('confidence', 1.0)
        )

    def _classify_content_change(
        self,
        baseline_hash: DocumentHash,
        current_hash: DocumentHash
    ) -> Dict[str, Any]:
        """Classify the type and extent of content changes"""
        details = {
            'type': ChangeType.CONTENT,
            'confidence': 1.0,
            'size_change': None,
            'structural_change': False
        }

        # Check size change
        if baseline_hash.file_size and current_hash.file_size:
            size_diff = current_hash.file_size - baseline_hash.file_size
            size_change_pct = (size_diff / baseline_hash.file_size) * 100 if baseline_hash.file_size > 0 else 0

            details['size_change'] = {
                'bytes_diff': size_diff,
                'percent_change': size_change_pct,
                'significant': abs(size_change_pct) > 10  # > 10% change
            }

        # Check structural changes
        baseline_structure = baseline_hash.checksum_parts.get('structure', '')
        current_structure = current_hash.checksum_parts.get('structure', '')

        if baseline_structure != current_structure:
            details['structural_change'] = True
            details['type'] = ChangeType.STRUCTURE
            details['reason'] = 'Document structure or format changed'
        else:
            details['reason'] = 'Document content modified'

        return details

    async def batch_detect_changes(
        self,
        document_batches: List[Dict[str, Dict[str, Any]]],
        baseline_hashes: Dict[str, DocumentHash],
        batch_size: int = 100
    ) -> ChangeDetectionResult:
        """Process changes in batches for large document sets"""
        all_changes: List[ChangeRecord] = []
        total_docs = 0
        total_time = 0.0

        for batch in document_batches:
            batch_result = await self.detect_changes(batch, baseline_hashes)
            all_changes.extend(batch_result.change_records)
            total_docs += batch_result.total_documents
            total_time += batch_result.processing_time_ms

            # Check performance target per batch
            if batch_result.processing_time_ms > self.performance_target_ms:
                logger.warning(
                    f"Batch processing exceeded target: {batch_result.processing_time_ms:.2f}ms"
                )

        cache_hit_rate = 0.0
        if self.cache:
            total_requests = self.performance_metrics['cache_hits'] + self.performance_metrics['cache_misses']
            if total_requests > 0:
                cache_hit_rate = self.performance_metrics['cache_hits'] / total_requests

        return ChangeDetectionResult(
            total_documents=total_docs,
            changes_detected=len(all_changes),
            change_records=all_changes,
            processing_time_ms=total_time,
            cache_hit_rate=cache_hit_rate,
            performance_metrics={
                'batch_count': len(document_batches),
                'avg_batch_time_ms': total_time / len(document_batches) if document_batches else 0,
                'meets_performance_target': all(
                    batch_time <= self.performance_target_ms
                    for batch_time in [total_time / len(document_batches)]
                )
            }
        )

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics"""
        total_ops = self.performance_metrics['total_detections']

        return {
            'total_operations': total_ops,
            'total_processing_time_ms': self.performance_metrics['total_processing_time_ms'],
            'average_time_per_operation_ms': (
                self.performance_metrics['total_processing_time_ms'] / total_ops
                if total_ops > 0 else 0
            ),
            'cache_statistics': {
                'hits': self.performance_metrics['cache_hits'],
                'misses': self.performance_metrics['cache_misses'],
                'hit_rate': (
                    self.performance_metrics['cache_hits'] /
                    (self.performance_metrics['cache_hits'] + self.performance_metrics['cache_misses'])
                    if (self.performance_metrics['cache_hits'] + self.performance_metrics['cache_misses']) > 0
                    else 0
                ),
                'cache_size': self.cache.size() if self.cache else 0
            },
            'performance_target_ms': self.performance_target_ms,
            'meets_target': (
                self.performance_metrics['total_processing_time_ms'] / total_ops <= self.performance_target_ms
                if total_ops > 0 else True
            )
        }

    def reset_metrics(self) -> None:
        """Reset performance metrics"""
        self.performance_metrics = {
            'total_detections': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_processing_time_ms': 0.0,
            'avg_detection_time_ms': 0.0
        }

        if self.cache:
            self.cache.clear()

# Utility functions for integration with golden testset manager

async def detect_testset_changes(
    testset_name: str,
    current_examples: List[Dict[str, Any]],
    baseline_hashes: Dict[str, DocumentHash],
    detector: Optional[ChangeDetector] = None
) -> ChangeDetectionResult:
    """Detect changes in a golden testset"""
    if detector is None:
        detector = ChangeDetector()

    # Convert examples to document format
    documents = {}
    for i, example in enumerate(current_examples):
        doc_id = f"{testset_name}_example_{i}"
        documents[doc_id] = {
            'content': json.dumps(example, sort_keys=True),
            'metadata': {
                'testset_name': testset_name,
                'example_index': i,
                'type': 'golden_example'
            }
        }

    return await detector.detect_changes(documents, baseline_hashes)

def create_baseline_hashes(
    testset_name: str,
    examples: List[Dict[str, Any]],
    algorithm: HashAlgorithm = HashAlgorithm.SHA256
) -> Dict[str, DocumentHash]:
    """Create baseline hashes for a golden testset"""
    hashes = {}

    for i, example in enumerate(examples):
        doc_id = f"{testset_name}_example_{i}"
        content = json.dumps(example, sort_keys=True)
        metadata = {
            'testset_name': testset_name,
            'example_index': i,
            'type': 'golden_example'
        }

        content_hash = ContentHasher.hash_content(content, algorithm)
        metadata_hash = ContentHasher.hash_metadata(metadata, algorithm)
        checksum_parts = ContentHasher.hash_document_parts(content, metadata, algorithm)

        hashes[doc_id] = DocumentHash(
            document_id=doc_id,
            content_hash=content_hash,
            metadata_hash=metadata_hash,
            algorithm=algorithm,
            computed_at=datetime.now(timezone.utc),
            file_size=len(content.encode('utf-8')),
            checksum_parts=checksum_parts
        )

    return hashes