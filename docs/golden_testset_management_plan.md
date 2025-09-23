# Golden Testset Management Plan for RAG Evaluation

## 1. Executive Summary

This document outlines a comprehensive strategy for managing golden testsets in a PostgreSQL database with Phoenix observability integration. The approach emphasizes immutable versioning, quality validation, and seamless integration with the existing RAGAS-based evaluation pipeline.

### Key Objectives
- Implement versioned, immutable golden testset storage in PostgreSQL
- Establish quality gates and validation pipelines for testset generation
- Integrate with Phoenix for observability and experiment tracking
- Provide robust rollback and recovery mechanisms
- Enable cost tracking and budget management for API usage

### Design Principles
- **Immutability**: Testsets are never modified after creation
- **Traceability**: Complete lineage from source documents to testset versions
- **Atomicity**: All database operations use transactions
- **Observability**: Full instrumentation via Phoenix/OpenTelemetry
- **Backward Compatibility**: Existing workflows continue to function

## 2. Architecture Design

### 2.1 System Components

```mermaid
graph TB
    subgraph "Data Sources"
        DS[Document Store]
        PG[PostgreSQL/pgvector]
    end

    subgraph "Generation Layer"
        RG[RAGAS Generator]
        VL[Validation Pipeline]
        CT[Cost Tracker]
    end

    subgraph "Storage Layer"
        GT[Golden Testset Tables]
        PX[Phoenix Datasets]
    end

    subgraph "Management Layer"
        GM[GoldenTestsetManager]
        CLI[CLI Tools]
        MON[Monitoring]
    end

    DS --> RG
    PG --> RG
    RG --> VL
    VL --> GT
    VL --> PX
    RG --> CT
    GT <--> GM
    PX <--> GM
    GM --> CLI
    GM --> MON
```

### 2.2 Data Flow

1. **Generation Phase**
   - Load documents from PostgreSQL
   - Generate testset using RAGAS
   - Track token usage and costs
   - Validate quality metrics

2. **Storage Phase**
   - Store in PostgreSQL with version
   - Upload to Phoenix as dataset
   - Update metadata and audit logs
   - Activate version if validation passes

3. **Usage Phase**
   - Retrieve active testset version
   - Run experiments with Phoenix
   - Collect performance metrics
   - Monitor quality over time

### 2.3 Version Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Draft: Generate
    Draft --> Validated: Quality Check
    Validated --> Active: Promote
    Active --> Archived: New Version
    Validated --> Failed: Validation Failed
    Failed --> [*]
    Active --> Rollback: Issues Found
    Rollback --> Active: Previous Version
```

## 3. Database Schema

### 3.1 Core Tables

```sql
-- Main versioned testset storage
CREATE TABLE golden_testset_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version VARCHAR(20) NOT NULL,  -- Semantic versioning (e.g., "1.2.3")
    status VARCHAR(20) NOT NULL,   -- 'draft', 'validated', 'active', 'archived'
    created_at TIMESTAMP DEFAULT NOW(),
    activated_at TIMESTAMP,
    archived_at TIMESTAMP,
    testset_data JSONB NOT NULL,   -- Complete testset as JSON
    is_active BOOLEAN DEFAULT false,
    created_by VARCHAR(100),
    UNIQUE(version),
    CHECK (status IN ('draft', 'validated', 'active', 'archived'))
);

-- Generation metadata and parameters
CREATE TABLE golden_testset_metadata (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    testset_version_id UUID REFERENCES golden_testset_versions(id) ON DELETE CASCADE,
    source_docs_hash VARCHAR(64) NOT NULL,
    source_docs_count INTEGER NOT NULL,
    ragas_version VARCHAR(20),
    langchain_version VARCHAR(20),
    generation_params JSONB NOT NULL,  -- All RAGAS parameters
    personas JSONB,                    -- Persona configurations
    query_distribution JSONB,          -- Question type distribution
    token_usage JSONB,                 -- Input/output token counts
    generation_cost DECIMAL(10,4),
    generation_duration_seconds INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Quality validation results
CREATE TABLE golden_testset_quality_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    testset_version_id UUID REFERENCES golden_testset_versions(id) ON DELETE CASCADE,
    validation_timestamp TIMESTAMP DEFAULT NOW(),
    validation_status VARCHAR(20) NOT NULL,  -- 'passed', 'failed', 'warning'
    diversity_score DECIMAL(4,3),            -- 0.000 to 1.000
    coverage_score DECIMAL(4,3),
    distribution_chi_square DECIMAL(10,4),
    distribution_p_value DECIMAL(6,4),
    duplicate_count INTEGER DEFAULT 0,
    validation_details JSONB,
    CHECK (validation_status IN ('passed', 'failed', 'warning'))
);

-- Audit log for all operations
CREATE TABLE golden_testset_audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    testset_version_id UUID REFERENCES golden_testset_versions(id),
    operation VARCHAR(50) NOT NULL,
    operation_timestamp TIMESTAMP DEFAULT NOW(),
    performed_by VARCHAR(100),
    details JSONB,
    result VARCHAR(20),  -- 'success', 'failure'
    error_message TEXT
);

-- Indexes for performance
CREATE INDEX idx_testset_version_active ON golden_testset_versions(is_active) WHERE is_active = true;
CREATE INDEX idx_testset_version_status ON golden_testset_versions(status);
CREATE INDEX idx_testset_metadata_hash ON golden_testset_metadata(source_docs_hash);
CREATE INDEX idx_audit_log_timestamp ON golden_testset_audit_log(operation_timestamp DESC);
CREATE INDEX idx_quality_metrics_status ON golden_testset_quality_metrics(validation_status);

-- View for current active testset
CREATE VIEW current_golden_testset AS
SELECT
    v.*,
    m.source_docs_hash,
    m.generation_params,
    m.token_usage,
    m.generation_cost,
    q.diversity_score,
    q.coverage_score
FROM golden_testset_versions v
LEFT JOIN golden_testset_metadata m ON v.id = m.testset_version_id
LEFT JOIN golden_testset_quality_metrics q ON v.id = q.testset_version_id
WHERE v.is_active = true;
```

### 3.2 Migration Strategy

1. **Phase 1**: Create new tables alongside existing structure
2. **Phase 2**: Migrate existing JSON testsets to new schema
3. **Phase 3**: Update application to use new tables
4. **Phase 4**: Deprecate old storage method

## 4. Implementation Roadmap

### Phase 1: Infrastructure Setup (Week 1)

#### Task 1.1: Database Schema Creation
**Description**: Implement PostgreSQL schema with versioning tables
**Quality Gates**:
- All tables created successfully with constraints
- Indexes optimize query performance (<100ms for lookups)
- Foreign key relationships maintain referential integrity
- Rollback scripts tested and verified

**References**:
- [PostgreSQL Audit Trigger Pattern](https://wiki.postgresql.org/wiki/Audit_trigger)
- [Temporal Tables Design](https://www.postgresql.org/docs/current/ddl-partitioning.html)

#### Task 1.2: Connection Management Updates
**Description**: Extend PGEngine integration for new tables
**Quality Gates**:
- Connection pooling configured (min: 5, max: 20)
- Async operations maintain performance
- Transaction isolation levels appropriate
- Error handling with automatic retry

**Code Example**:
```python
from langchain_postgres import PGEngine
from contextlib import asynccontextmanager

class TestsetDB:
    def __init__(self, connection_string: str):
        self.engine = PGEngine.from_connection_string(connection_string)

    @asynccontextmanager
    async def transaction(self):
        async with self.engine.acquire() as conn:
            async with conn.transaction():
                yield conn
```

### Phase 2: Core Implementation (Week 2)

#### Task 2.1: GoldenTestsetManager Class
**Description**: Core management class for testset operations
**Quality Gates**:
- All CRUD operations atomic with transactions
- Version conflicts prevented by database constraints
- Comprehensive error handling with meaningful messages
- Unit test coverage >90%

**Implementation Structure**:
```python
class GoldenTestsetManager:
    """Manages golden testset lifecycle with versioning"""

    async def create_version(
        self,
        testset_df: pd.DataFrame,
        metadata: Dict[str, Any],
        auto_promote: bool = False
    ) -> TestsetVersion:
        """Create new testset version with validation"""

    async def get_active_version(self) -> Optional[TestsetVersion]:
        """Retrieve currently active testset"""

    async def promote_version(
        self,
        version: str,
        force: bool = False
    ) -> bool:
        """Promote validated version to active status"""

    async def rollback_version(
        self,
        to_version: str,
        reason: str
    ) -> bool:
        """Rollback to previous version with audit trail"""

    async def compare_versions(
        self,
        version_a: str,
        version_b: str
    ) -> VersionComparison:
        """Statistical comparison between versions"""
```

**References**:
- [Async Context Managers in Python](https://docs.python.org/3/library/contextlib.html#contextlib.asynccontextmanager)
- LangChain PGEngine patterns from existing codebase

#### Task 2.2: Version Control Implementation
**Description**: Semantic versioning with automated bumping
**Quality Gates**:
- Semantic version parsing and validation
- Automatic version bumping based on changes
- Version history maintained immutably
- Conflict prevention via unique constraints

**Version Rules**:
- **MAJOR** (X.0.0): Breaking changes to testset structure
- **MINOR** (1.X.0): New question types or significant content changes
- **PATCH** (1.2.X): Bug fixes or minor adjustments

**References**:
- [Semantic Versioning 2.0.0](https://semver.org/)
- [Python semver library](https://python-semver.readthedocs.io/)

### Phase 3: Quality & Validation (Week 3)

#### Task 3.1: Document Change Detection
**Description**: Detect changes in source documents to trigger regeneration
**Quality Gates**:
- SHA-256 hashing for content comparison
- Metadata changes tracked separately
- Configurable sensitivity thresholds
- Performance impact <100ms per check

**Implementation**:
```python
import hashlib
import json
from typing import List, Optional

def calculate_document_hash(docs: List[Document]) -> str:
    """Generate deterministic hash of document collection"""
    content = {
        'documents': [
            {
                'content': doc.page_content,
                'metadata': doc.metadata
            } for doc in sorted(docs, key=lambda d: d.page_content[:100])
        ]
    }
    serialized = json.dumps(content, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()

async def detect_changes(
    current_hash: str,
    threshold: float = 1.0
) -> Optional[ChangeReport]:
    """Detect significant changes requiring testset regeneration"""
    stored_hash = await get_latest_source_hash()
    if current_hash != stored_hash:
        return ChangeReport(
            changed=True,
            similarity=calculate_similarity(current_hash, stored_hash),
            requires_regeneration=similarity < threshold
        )
    return None
```

**References**:
- [RAGAS Document Processing](https://docs.ragas.io/en/latest/concepts/testset_generation.html)
- Content-based hashing best practices

#### Task 3.2: Quality Validation Pipeline
**Description**: Comprehensive quality checks before activation
**Quality Gates**:
- Statistical distribution validation (Chi-square test)
- Diversity score >0.7 (semantic similarity)
- No duplicate questions (exact match + fuzzy)
- Coverage of all document sources

**Validation Components**:
```python
class QualityValidator:
    """Validates testset quality before promotion"""

    async def validate(self, testset_df: pd.DataFrame) -> ValidationResult:
        checks = [
            self.check_distribution(),
            self.check_diversity(),
            self.check_duplicates(),
            self.check_coverage(),
            self.check_completeness()
        ]

        results = await asyncio.gather(*checks)
        return ValidationResult(
            passed=all(r.passed for r in results),
            score=sum(r.score for r in results) / len(results),
            details=results
        )

    def check_distribution(self) -> ValidationCheck:
        """Validate question type distribution matches target"""
        # Chi-square test against expected distribution

    def check_diversity(self) -> ValidationCheck:
        """Calculate semantic diversity of questions"""
        # Use embeddings to measure similarity

    def check_coverage(self) -> ValidationCheck:
        """Ensure all document sources represented"""
        # Map questions to source documents
```

**References**:
- [RAGAS Testset Quality Metrics](https://docs.ragas.io/en/latest/concepts/metrics.html)
- [Statistical Tests for Distribution Comparison](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html)

### Phase 4: Integration (Week 4)

#### Task 4.1: Phoenix Dataset Integration
**Description**: Versioned dataset upload to Phoenix
**Quality Gates**:
- Successful upload with full metadata
- Version tracking in dataset names
- Experiment linkage maintained
- Rollback capability preserved

**Implementation**:
```python
from phoenix.client import Client
from datetime import datetime

async def upload_to_phoenix_versioned(
    testset_df: pd.DataFrame,
    version: str,
    metadata: Dict[str, Any]
) -> PhoenixDataset:
    """Upload versioned testset to Phoenix"""

    px_client = Client()

    # Prepare Phoenix-compatible dataframe
    phoenix_df = pd.DataFrame({
        'input': testset_df['user_input'],
        'output': testset_df['reference'],
        'contexts': testset_df['reference_contexts'].apply(json.dumps),
        'metadata': testset_df.apply(
            lambda row: {
                'version': version,
                'synthesizer': row['synthesizer_name'],
                'question_type': row.get('evolution_type', 'unknown'),
                **metadata
            },
            axis=1
        )
    })

    # Create versioned dataset name
    dataset_name = f"golden_testset_v{version}_{datetime.now():%Y%m%d_%H%M%S}"

    dataset = px_client.datasets.create_dataset(
        name=dataset_name,
        dataframe=phoenix_df,
        input_keys=['input'],
        output_keys=['output'],
        metadata_keys=['contexts', 'metadata']
    )

    return dataset
```

**References**:
- [Phoenix Dataset API](https://docs.arize.com/phoenix/datasets-and-experiments/how-to-datasets)
- [Phoenix Experiment Integration](https://docs.arize.com/phoenix/datasets-and-experiments/how-to-experiments)

#### Task 4.2: Cost Tracking Implementation
**Description**: Track token usage and generation costs
**Quality Gates**:
- Token counts accurate within 1%
- Cost calculation includes all API calls
- Budget alerts at configurable thresholds
- Historical trend analysis available

**RAGAS Cost Tracking**:
```python
from ragas.testset import TestsetGenerator
from ragas.llms import get_token_usage_for_openai

async def generate_with_cost_tracking(
    documents: List[Document],
    size: int = 10
) -> Tuple[Dataset, CostReport]:
    """Generate testset with comprehensive cost tracking"""

    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings
    )

    # Generate with token tracking
    testset = generator.generate(
        testset_size=size,
        token_usage_parser=get_token_usage_for_openai
    )

    # Calculate costs based on current pricing
    cost_report = CostReport(
        input_tokens=testset.token_usage['input_tokens'],
        output_tokens=testset.token_usage['output_tokens'],
        total_cost=testset.total_cost(
            cost_per_input_token=5/1e6,  # $5 per million
            cost_per_output_token=15/1e6  # $15 per million
        ),
        model='gpt-4.1-mini',
        timestamp=datetime.now()
    )

    return testset, cost_report
```

**References**:
- [RAGAS Token Usage Tracking](https://docs.ragas.io/en/latest/howtos/applications/cost.html)
- [OpenAI Pricing](https://openai.com/pricing)

### Phase 5: Operations & Monitoring (Week 5)

#### Task 5.1: CLI Management Tools
**Description**: Command-line interface for testset operations
**Quality Gates**:
- Commands follow Unix philosophy
- Comprehensive help documentation
- Error messages actionable
- JSON output for automation

**CLI Structure**:
```python
# testset_cli.py
import click
from typing import Optional

@click.group()
def cli():
    """Golden Testset Management CLI"""
    pass

@cli.command()
@click.option('--size', default=10, help='Number of test examples')
@click.option('--auto-promote', is_flag=True, help='Auto-promote if valid')
def create(size: int, auto_promote: bool):
    """Create new testset version"""
    # Implementation

@cli.command()
@click.option('--format', type=click.Choice(['table', 'json']), default='table')
def list_versions(format: str):
    """List all testset versions"""
    # Implementation

@cli.command()
@click.argument('version')
@click.option('--force', is_flag=True, help='Skip validation')
def activate(version: str, force: bool):
    """Activate specific version"""
    # Implementation

@cli.command()
@click.argument('to_version')
@click.option('--reason', required=True, help='Rollback reason')
def rollback(to_version: str, reason: str):
    """Rollback to previous version"""
    # Implementation

@cli.command()
@click.argument('version_a')
@click.argument('version_b')
@click.option('--output', type=click.Path(), help='Save comparison report')
def compare(version_a: str, version_b: str, output: Optional[str]):
    """Compare two testset versions"""
    # Implementation
```

**References**:
- [Click Documentation](https://click.palletsprojects.com/)
- Unix philosophy for CLI design

#### Task 5.2: Monitoring & Observability
**Description**: Phoenix tracing and metrics dashboard
**Quality Gates**:
- All operations traced via OpenTelemetry
- Metrics dashboard operational
- Alert thresholds configurable
- Query response time <2 seconds

**Tracing Implementation**:
```python
from opentelemetry import trace
from phoenix.otel import register

tracer = trace.get_tracer(__name__)

class TracedTestsetManager(GoldenTestsetManager):
    """Testset manager with Phoenix observability"""

    @tracer.start_as_current_span("create_testset_version")
    async def create_version(self, **kwargs):
        span = trace.get_current_span()
        span.set_attribute("testset.size", kwargs.get('size'))
        span.set_attribute("testset.auto_promote", kwargs.get('auto_promote'))

        try:
            result = await super().create_version(**kwargs)
            span.set_attribute("testset.version", result.version)
            span.set_status(StatusCode.OK)
            return result
        except Exception as e:
            span.record_exception(e)
            span.set_status(StatusCode.ERROR, str(e))
            raise
```

**References**:
- [Phoenix Tracing Guide](https://docs.arize.com/phoenix/tracing/how-to-tracing)
- [OpenTelemetry Python](https://opentelemetry.io/docs/instrumentation/python/)

## 5. Quality Gates Summary

### Critical Quality Gates

| Component | Quality Gate | Target | Measurement |
|-----------|-------------|--------|-------------|
| Database Operations | Transaction Success Rate | 99.9% | Audit log analysis |
| Version Control | No Version Conflicts | 100% | Unique constraint violations |
| Document Detection | Change Detection Accuracy | >95% | Manual validation sample |
| Quality Validation | Diversity Score | >0.7 | Semantic similarity calculation |
| Phoenix Upload | Upload Success Rate | >99% | API response tracking |
| Cost Tracking | Token Count Accuracy | ±1% | Comparison with OpenAI usage |
| CLI Operations | Command Success Rate | >95% | Error log analysis |
| Performance | Query Response Time | <2s | P95 latency metric |
| Test Coverage | Unit Test Coverage | >90% | pytest-cov report |
| Integration Tests | End-to-End Success | 100% | CI/CD pipeline |

### Validation Pipeline Stages

1. **Pre-Generation Validation**
   - Source documents available
   - Configuration valid
   - Budget remaining

2. **Post-Generation Validation**
   - Statistical distribution check
   - Diversity measurement
   - Duplicate detection
   - Coverage verification

3. **Pre-Activation Validation**
   - Quality scores above threshold
   - No blocking issues
   - Approval workflow (if configured)

4. **Post-Activation Monitoring**
   - Performance metrics tracking
   - Error rate monitoring
   - User feedback collection

## 6. Technical References

### RAGAS Documentation
- [Testset Generation](https://docs.ragas.io/en/latest/concepts/testset_generation.html)
- [Knowledge Graph Persistence](https://docs.ragas.io/en/latest/howtos/customizations/testgenerator.html)
- [Token Usage Tracking](https://docs.ragas.io/en/latest/howtos/applications/cost.html)
- [Persona-based Generation](https://docs.ragas.io/en/latest/howtos/customizations/testgenerator/persona_generator.html)
- [Query Distribution](https://docs.ragas.io/en/latest/getstarted/rag_testset_generation.html#_snippet_9)

### Phoenix/Arize Documentation
- [Dataset Management](https://docs.arize.com/phoenix/datasets-and-experiments/how-to-datasets)
- [Experiment Framework](https://docs.arize.com/phoenix/datasets-and-experiments/how-to-experiments)
- [Tracing Setup](https://docs.arize.com/phoenix/tracing/how-to-tracing)
- [Evaluation Integration](https://docs.arize.com/phoenix/evaluation/how-to-evals)
- [Cost Tracking](https://docs.arize.com/phoenix/tracing/how-to-tracing/cost-tracking)

### PostgreSQL Patterns
- [Audit Trigger Pattern](https://wiki.postgresql.org/wiki/Audit_trigger)
- [Temporal Tables](https://www.postgresql.org/docs/current/ddl-partitioning.html)
- [JSONB Best Practices](https://www.postgresql.org/docs/current/datatype-json.html)
- [Transaction Isolation](https://www.postgresql.org/docs/current/transaction-iso.html)

### LangChain Resources
- [PGVectorStore](https://python.langchain.com/docs/integrations/vectorstores/pgvector)
- [Async Operations](https://python.langchain.com/docs/expression_language/interface)
- [Document Loaders](https://python.langchain.com/docs/modules/data_connection/document_loaders)

### Python Libraries
- [asyncio](https://docs.python.org/3/library/asyncio.html) - Async programming
- [asyncpg](https://magicstack.github.io/asyncpg/) - PostgreSQL async driver
- [semver](https://python-semver.readthedocs.io/) - Semantic versioning
- [click](https://click.palletsprojects.com/) - CLI framework
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/) - Async testing

## 7. Testing Strategy

### 7.1 Unit Testing

```python
# test_golden_testset_manager.py
import pytest
import pytest_asyncio
from unittest.mock import Mock, AsyncMock

@pytest.mark.asyncio
async def test_create_version():
    """Test testset version creation"""
    manager = GoldenTestsetManager(db_url="postgresql://...")

    testset_df = pd.DataFrame({
        'user_input': ['question1', 'question2'],
        'reference': ['answer1', 'answer2']
    })

    version = await manager.create_version(
        testset_df=testset_df,
        metadata={'source': 'test'},
        auto_promote=False
    )

    assert version.status == 'draft'
    assert version.version.startswith('1.')

@pytest.mark.asyncio
async def test_version_rollback():
    """Test version rollback functionality"""
    manager = GoldenTestsetManager(db_url="postgresql://...")

    # Setup: Create and activate version
    v1 = await manager.create_version(...)
    await manager.promote_version(v1.version)

    v2 = await manager.create_version(...)
    await manager.promote_version(v2.version)

    # Test rollback
    result = await manager.rollback_version(
        to_version=v1.version,
        reason="Performance regression"
    )

    assert result == True
    current = await manager.get_active_version()
    assert current.version == v1.version
```

### 7.2 Integration Testing

```python
# test_integration.py
@pytest.mark.integration
async def test_end_to_end_workflow():
    """Test complete testset lifecycle"""

    # 1. Generate testset
    testset = await generate_testset(documents, size=5)

    # 2. Validate quality
    validation = await validate_quality(testset)
    assert validation.passed

    # 3. Store in database
    version = await store_testset(testset)
    assert version.id is not None

    # 4. Upload to Phoenix
    dataset = await upload_to_phoenix(testset, version)
    assert dataset.name.startswith('golden_testset')

    # 5. Run experiment
    experiment = await run_experiment_with_testset(version)
    assert experiment.status == 'completed'
```

### 7.3 Performance Testing

```python
# test_performance.py
@pytest.mark.performance
async def test_query_performance():
    """Test query response times"""
    manager = GoldenTestsetManager(db_url="postgresql://...")

    start_time = time.time()
    version = await manager.get_active_version()
    duration = time.time() - start_time

    assert duration < 2.0  # Must respond within 2 seconds

@pytest.mark.performance
async def test_concurrent_operations():
    """Test concurrent testset operations"""
    manager = GoldenTestsetManager(db_url="postgresql://...")

    tasks = [
        manager.get_active_version(),
        manager.list_versions(),
        manager.get_version_metadata('1.0.0')
    ]

    start_time = time.time()
    results = await asyncio.gather(*tasks)
    duration = time.time() - start_time

    assert duration < 5.0  # Concurrent ops within 5 seconds
    assert all(r is not None for r in results)
```

## 8. Migration Plan

### 8.1 Current State Assessment

1. **Existing Testset Storage**:
   - JSON files: `golden_testset.json`
   - Phoenix datasets (if uploaded)
   - No versioning or quality tracking

2. **Migration Requirements**:
   - Preserve existing testsets
   - Maintain backward compatibility
   - Zero downtime migration

### 8.2 Migration Steps

```python
# migration_script.py
async def migrate_existing_testsets():
    """Migrate existing JSON testsets to new schema"""

    # Step 1: Load existing testset
    with open('golden_testset.json', 'r') as f:
        existing_data = json.load(f)

    testset_df = pd.DataFrame(existing_data)

    # Step 2: Calculate metadata
    metadata = {
        'source': 'legacy_migration',
        'original_format': 'json',
        'migration_date': datetime.now().isoformat()
    }

    # Step 3: Create version 0.1.0 (legacy)
    manager = GoldenTestsetManager()
    version = await manager.create_version(
        testset_df=testset_df,
        metadata=metadata,
        version='0.1.0',  # Legacy version number
        auto_promote=True
    )

    print(f"Migrated testset to version {version.version}")

    # Step 4: Update application configuration
    await update_config_for_new_schema()
```

### 8.3 Rollback Plan

1. **Database Rollback**:
   ```sql
   -- Rollback script
   BEGIN;
   DROP TABLE IF EXISTS golden_testset_audit_log CASCADE;
   DROP TABLE IF EXISTS golden_testset_quality_metrics CASCADE;
   DROP TABLE IF EXISTS golden_testset_metadata CASCADE;
   DROP TABLE IF EXISTS golden_testset_versions CASCADE;
   DROP VIEW IF EXISTS current_golden_testset;
   COMMIT;
   ```

2. **Application Rollback**:
   - Revert to previous code version
   - Restore JSON-based testset loading
   - Update configuration files

## 9. Operational Procedures

### 9.1 Regular Operations

#### Daily Tasks
- Monitor testset quality metrics
- Review generation costs
- Check for source document changes

#### Weekly Tasks
- Review testset performance in experiments
- Analyze quality trends
- Update documentation if needed

#### Monthly Tasks
- Full testset regeneration (if needed)
- Cost analysis and optimization
- Performance review and tuning

### 9.2 Troubleshooting Guide

| Issue | Diagnosis | Resolution |
|-------|-----------|------------|
| Generation fails | Check API keys and quotas | Verify OpenAI/Cohere keys valid |
| Validation fails | Review quality metrics | Adjust generation parameters |
| Version conflict | Check unique constraints | Use different version number |
| Rollback fails | Check version exists | Verify target version in database |
| Phoenix upload fails | Check Phoenix service | Verify Phoenix running and accessible |
| High costs | Review token usage | Optimize prompt templates |
| Slow queries | Check database indexes | Run ANALYZE on tables |

### 9.3 Emergency Procedures

**Critical Issue: Active Testset Corrupted**
1. Immediately rollback to previous version
2. Notify team via monitoring alerts
3. Investigate root cause
4. Regenerate if necessary

**Budget Exceeded**
1. Pause automatic generation
2. Review token usage patterns
3. Optimize generation parameters
4. Request budget increase if justified

## 10. Appendices

### Appendix A: Configuration Template

```yaml
# golden_testset_config.yaml
database:
  connection_string: "postgresql://langchain:langchain@localhost:6024/langchain"
  pool_size: 20
  timeout: 30

generation:
  default_size: 10
  max_size: 100
  auto_promote: false
  quality_threshold: 0.7

ragas:
  model: "gpt-4.1-mini"
  embedding_model: "text-embedding-3-small"
  personas_enabled: true
  query_distribution:
    simple: 0.5
    multi_context: 0.3
    reasoning: 0.2

phoenix:
  endpoint: "http://localhost:6006"
  project_name: "golden-testset-management"

monitoring:
  alert_email: "team@example.com"
  cost_limit_daily: 10.00
  quality_alert_threshold: 0.6

versioning:
  strategy: "semantic"  # or "timestamp", "incremental"
  auto_bump: true
```

### Appendix B: Example Workflows

#### Workflow 1: Scheduled Regeneration
```python
# scheduled_regeneration.py
import asyncio
from datetime import datetime, timedelta

async def scheduled_regeneration():
    """Weekly testset regeneration with validation"""

    manager = GoldenTestsetManager()

    # Check if regeneration needed
    current = await manager.get_active_version()
    if current.age_days < 7:
        print("Testset is recent, skipping regeneration")
        return

    # Generate new version
    new_testset = await generate_testset(size=20)

    # Validate quality
    validation = await validate_quality(new_testset)
    if not validation.passed:
        await send_alert("Quality validation failed", validation.details)
        return

    # Create and promote new version
    version = await manager.create_version(
        testset_df=new_testset,
        metadata={'schedule': 'weekly'},
        auto_promote=True
    )

    print(f"Successfully regenerated testset: {version.version}")
```

#### Workflow 2: A/B Testing Versions
```python
# ab_testing.py
async def ab_test_versions(version_a: str, version_b: str):
    """Run A/B test between two testset versions"""

    manager = GoldenTestsetManager()
    px_client = Client()

    # Load both versions
    testset_a = await manager.get_version(version_a)
    testset_b = await manager.get_version(version_b)

    # Run experiments
    exp_a = await run_experiment(
        dataset=testset_a,
        task=evaluation_task,
        evaluators=[qa_evaluator, relevance_evaluator]
    )

    exp_b = await run_experiment(
        dataset=testset_b,
        task=evaluation_task,
        evaluators=[qa_evaluator, relevance_evaluator]
    )

    # Compare results
    comparison = await manager.compare_versions(version_a, version_b)

    # Decision logic
    if exp_b.avg_score > exp_a.avg_score * 1.05:  # 5% improvement
        await manager.promote_version(version_b)
        print(f"Version {version_b} promoted based on A/B test")
    else:
        print(f"Version {version_a} retained, insufficient improvement")
```

### Appendix C: Monitoring Dashboard Queries

```sql
-- Active testset summary
SELECT
    v.version,
    v.activated_at,
    m.source_docs_count,
    m.generation_cost,
    q.diversity_score,
    q.coverage_score
FROM golden_testset_versions v
JOIN golden_testset_metadata m ON v.id = m.testset_version_id
JOIN golden_testset_quality_metrics q ON v.id = q.testset_version_id
WHERE v.is_active = true;

-- Generation cost trends (last 30 days)
SELECT
    DATE(m.created_at) as date,
    COUNT(*) as generations,
    SUM(m.generation_cost) as total_cost,
    AVG(m.generation_cost) as avg_cost
FROM golden_testset_metadata m
WHERE m.created_at > NOW() - INTERVAL '30 days'
GROUP BY DATE(m.created_at)
ORDER BY date DESC;

-- Quality metrics over time
SELECT
    v.version,
    v.created_at,
    q.diversity_score,
    q.coverage_score,
    q.validation_status
FROM golden_testset_versions v
JOIN golden_testset_quality_metrics q ON v.id = q.testset_version_id
ORDER BY v.created_at DESC
LIMIT 10;

-- Failed validations analysis
SELECT
    v.version,
    q.validation_status,
    q.validation_details->>'failure_reason' as reason,
    COUNT(*) as occurrences
FROM golden_testset_versions v
JOIN golden_testset_quality_metrics q ON v.id = q.testset_version_id
WHERE q.validation_status = 'failed'
GROUP BY v.version, q.validation_status, reason
ORDER BY occurrences DESC;
```

## Conclusion

This comprehensive plan provides a robust framework for managing golden testsets with enterprise-grade reliability, observability, and quality assurance. The implementation leverages best practices from RAGAS, Phoenix, and PostgreSQL while maintaining backward compatibility and enabling future scalability.

Key success factors:
- **Immutable versioning** ensures reproducibility
- **Quality gates** prevent bad testsets from production
- **Cost tracking** enables budget management
- **Phoenix integration** provides observability
- **Automated workflows** reduce operational overhead

The phased implementation approach allows for incremental delivery while maintaining system stability throughout the migration process.