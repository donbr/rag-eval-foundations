#!/usr/bin/env python3
"""
Golden Testset Schema Generator
Generates PostgreSQL schema for versioned golden testset management

This script creates the database schema required for:
- Versioned golden testset storage (immutable with semantic versioning)
- Phoenix experiment tracking integration
- Quality validation and approval workflows
- Metadata tracking and lineage

Schema Design:
- golden_testsets: Main testset records with semantic versioning
- golden_examples: Individual Q&A examples with retrieval context
- testset_versions: Version lineage and approval status
- testset_quality_metrics: Automated validation scores
- testset_approval_log: Manual review and approval history

Features:
- Immutable versioning (MAJOR.MINOR.PATCH)
- UUID primary keys for distributed consistency
- JSONB for flexible metadata storage
- Pgvector integration for semantic search
- Audit trails and lineage tracking
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import asyncpg


class SchemaGenerator:
    """PostgreSQL schema generator for golden testset management"""

    def __init__(self, connection_string: Optional[str] = None):
        self.connection_string = connection_string or os.environ.get(
            "POSTGRES_CONNECTION_STRING",
            "postgresql://langchain:langchain@localhost:6024/langchain"
        )
        self.schema_dir = Path("schemas")
        self.schema_dir.mkdir(exist_ok=True)

    async def generate_schema(self) -> Dict[str, str]:
        """Generate all schema SQL files"""
        schemas = {}

        # 1. Core tables schema
        schemas["01_core_tables.sql"] = self._generate_core_tables()

        # 2. Indexes and constraints
        schemas["02_indexes.sql"] = self._generate_indexes()

        # 3. Functions and triggers
        schemas["03_functions.sql"] = self._generate_functions()

        # 4. Views for common queries
        schemas["04_views.sql"] = self._generate_views()

        # 5. Sample data (for testing)
        schemas["05_sample_data.sql"] = self._generate_sample_data()

        # Write schema files
        for filename, content in schemas.items():
            schema_file = self.schema_dir / filename
            schema_file.write_text(content)
            print(f"Generated: {schema_file}")

        # Generate combined schema
        combined_schema = self._generate_combined_schema(schemas)
        combined_file = self.schema_dir / "golden_testset_schema.sql"
        combined_file.write_text(combined_schema)
        print(f"Generated combined schema: {combined_file}")

        return schemas

    def _generate_core_tables(self) -> str:
        """Generate core table definitions"""
        timestamp = datetime.now().isoformat()
        return f"""-- Golden Testset Management Schema - Core Tables
-- Generated: {timestamp}

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

-- Main golden testset records with semantic versioning
CREATE TABLE IF NOT EXISTS golden_testsets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    version_major INTEGER NOT NULL DEFAULT 1,
    version_minor INTEGER NOT NULL DEFAULT 0,
    version_patch INTEGER NOT NULL DEFAULT 0,
    version_label VARCHAR(50), -- alpha, beta, rc1, etc.

    -- Metadata
    domain VARCHAR(100), -- financial_aid, movie_reviews, etc.
    source_type VARCHAR(50), -- ragas, manual, imported
    ragas_config JSONB, -- RAGAS generation parameters

    -- Lifecycle
    status VARCHAR(20) DEFAULT 'draft', -- draft, review, approved, deprecated
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(100),
    approved_at TIMESTAMP WITH TIME ZONE,
    approved_by VARCHAR(100),
    deprecated_at TIMESTAMP WITH TIME ZONE,

    -- Phoenix integration
    phoenix_project_id VARCHAR(255),
    phoenix_experiment_id VARCHAR(255),

    -- Quality tracking
    quality_score DECIMAL(3,2), -- 0.00 to 1.00
    validation_status VARCHAR(20) DEFAULT 'pending', -- pending, passed, failed

    -- Constraints
    CONSTRAINT valid_version CHECK (version_major >= 1 AND version_minor >= 0 AND version_patch >= 0),
    CONSTRAINT valid_status CHECK (status IN ('draft', 'review', 'approved', 'deprecated')),
    CONSTRAINT valid_validation_status CHECK (validation_status IN ('pending', 'passed', 'failed')),
    CONSTRAINT unique_version UNIQUE (name, version_major, version_minor, version_patch, version_label)
);

-- Individual Q&A examples within testsets
CREATE TABLE IF NOT EXISTS golden_examples (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    testset_id UUID NOT NULL REFERENCES golden_testsets(id) ON DELETE CASCADE,

    -- Q&A content
    question TEXT NOT NULL,
    ground_truth TEXT NOT NULL,
    contexts TEXT[], -- Retrieved context documents

    -- RAGAS metadata
    ragas_question_type VARCHAR(50), -- simple, complex, multi_context, conditional
    ragas_evolution_type VARCHAR(50), -- reasoning, multi_context, conditional
    ragas_difficulty DECIMAL(2,1), -- 1.0 to 5.0

    -- Retrieval context
    retrieval_strategy VARCHAR(100), -- naive, semantic, bm25, ensemble, etc.
    retrieval_score DECIMAL(5,4), -- similarity/relevance score
    retrieval_metadata JSONB, -- strategy-specific metadata

    -- Quality metrics
    context_precision DECIMAL(3,2),
    context_recall DECIMAL(3,2),
    faithfulness DECIMAL(3,2),
    answer_relevancy DECIMAL(3,2),

    -- Audit
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Vector search support
    question_embedding vector(1536), -- OpenAI text-embedding-3-small
    ground_truth_embedding vector(1536)
);

-- Version lineage and relationships
CREATE TABLE IF NOT EXISTS testset_versions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    testset_id UUID NOT NULL REFERENCES golden_testsets(id) ON DELETE CASCADE,
    parent_version_id UUID REFERENCES testset_versions(id),

    -- Version info
    version_string VARCHAR(50) NOT NULL, -- "1.2.3-beta"
    change_type VARCHAR(20) NOT NULL, -- major, minor, patch
    change_summary TEXT,

    -- Migration info
    examples_added INTEGER DEFAULT 0,
    examples_modified INTEGER DEFAULT 0,
    examples_removed INTEGER DEFAULT 0,

    -- Lifecycle
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(100),

    CONSTRAINT valid_change_type CHECK (change_type IN ('major', 'minor', 'patch'))
);

-- Quality validation metrics and automated checks
CREATE TABLE IF NOT EXISTS testset_quality_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    testset_id UUID NOT NULL REFERENCES golden_testsets(id) ON DELETE CASCADE,

    -- Metric type and values
    metric_type VARCHAR(50) NOT NULL, -- coverage, diversity, difficulty, consistency
    metric_value DECIMAL(5,4) NOT NULL,
    metric_details JSONB,

    -- Computation info
    computed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    computed_by VARCHAR(100), -- system, human, script_name
    computation_method VARCHAR(100), -- ragas, custom, manual

    -- Thresholds and validation
    threshold_min DECIMAL(5,4),
    threshold_max DECIMAL(5,4),
    passes_threshold BOOLEAN
);

-- Manual review and approval history
CREATE TABLE IF NOT EXISTS testset_approval_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    testset_id UUID NOT NULL REFERENCES golden_testsets(id) ON DELETE CASCADE,

    -- Review details
    reviewer VARCHAR(100) NOT NULL,
    review_status VARCHAR(20) NOT NULL, -- approved, rejected, needs_revision
    review_comments TEXT,
    review_checklist JSONB, -- structured review criteria

    -- Timing
    reviewed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    CONSTRAINT valid_review_status CHECK (review_status IN ('approved', 'rejected', 'needs_revision'))
);

"""

    def _generate_indexes(self) -> str:
        """Generate index definitions for performance"""
        timestamp = datetime.now().isoformat()
        return f"""-- Golden Testset Schema - Indexes and Performance
-- Generated: {timestamp}

-- Core lookup indexes
CREATE INDEX IF NOT EXISTS idx_golden_testsets_name ON golden_testsets(name);
CREATE INDEX IF NOT EXISTS idx_golden_testsets_version ON golden_testsets(version_major, version_minor, version_patch);
CREATE INDEX IF NOT EXISTS idx_golden_testsets_status ON golden_testsets(status);
CREATE INDEX IF NOT EXISTS idx_golden_testsets_domain ON golden_testsets(domain);
CREATE INDEX IF NOT EXISTS idx_golden_testsets_created_at ON golden_testsets(created_at);

-- Phoenix integration indexes
CREATE INDEX IF NOT EXISTS idx_golden_testsets_phoenix_project ON golden_testsets(phoenix_project_id);
CREATE INDEX IF NOT EXISTS idx_golden_testsets_phoenix_experiment ON golden_testsets(phoenix_experiment_id);

-- Example lookup and search indexes
CREATE INDEX IF NOT EXISTS idx_golden_examples_testset ON golden_examples(testset_id);
CREATE INDEX IF NOT EXISTS idx_golden_examples_question_gin ON golden_examples USING gin(to_tsvector('english', question));
CREATE INDEX IF NOT EXISTS idx_golden_examples_ground_truth_gin ON golden_examples USING gin(to_tsvector('english', ground_truth));
CREATE INDEX IF NOT EXISTS idx_golden_examples_retrieval_strategy ON golden_examples(retrieval_strategy);

-- Vector similarity search indexes (HNSW for performance)
CREATE INDEX IF NOT EXISTS idx_golden_examples_question_embedding ON golden_examples
USING hnsw (question_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS idx_golden_examples_ground_truth_embedding ON golden_examples
USING hnsw (ground_truth_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- Quality metrics indexes
CREATE INDEX IF NOT EXISTS idx_golden_examples_quality ON golden_examples(context_precision, context_recall, faithfulness, answer_relevancy);
CREATE INDEX IF NOT EXISTS idx_quality_metrics_testset_type ON testset_quality_metrics(testset_id, metric_type);
CREATE INDEX IF NOT EXISTS idx_quality_metrics_computed_at ON testset_quality_metrics(computed_at);

-- Version tracking indexes
CREATE INDEX IF NOT EXISTS idx_testset_versions_testset ON testset_versions(testset_id);
CREATE INDEX IF NOT EXISTS idx_testset_versions_parent ON testset_versions(parent_version_id);
CREATE INDEX IF NOT EXISTS idx_testset_versions_created_at ON testset_versions(created_at);

-- Approval tracking indexes
CREATE INDEX IF NOT EXISTS idx_approval_log_testset ON testset_approval_log(testset_id);
CREATE INDEX IF NOT EXISTS idx_approval_log_reviewer ON testset_approval_log(reviewer);
CREATE INDEX IF NOT EXISTS idx_approval_log_status ON testset_approval_log(review_status);
CREATE INDEX IF NOT EXISTS idx_approval_log_reviewed_at ON testset_approval_log(reviewed_at);

-- Composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_testsets_name_version_status ON golden_testsets(name, version_major DESC, version_minor DESC, version_patch DESC, status);
CREATE INDEX IF NOT EXISTS idx_examples_testset_quality ON golden_examples(testset_id, context_precision DESC, context_recall DESC);

"""

    def _generate_functions(self) -> str:
        """Generate utility functions and triggers"""
        timestamp = datetime.now().isoformat()
        return f"""-- Golden Testset Schema - Functions and Triggers
-- Generated: {timestamp}

-- Function to generate semantic version string
CREATE OR REPLACE FUNCTION format_version(major INT, minor INT, patch INT, label VARCHAR DEFAULT NULL)
RETURNS VARCHAR AS $$
BEGIN
    IF label IS NULL OR label = '' THEN
        RETURN format('%s.%s.%s', major, minor, patch);
    ELSE
        RETURN format('%s.%s.%s-%s', major, minor, patch, label);
    END IF;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to parse version string into components
CREATE OR REPLACE FUNCTION parse_version(version_string VARCHAR)
RETURNS TABLE(major INT, minor INT, patch INT, label VARCHAR) AS $$
DECLARE
    parts TEXT[];
    version_part TEXT;
    label_part TEXT;
BEGIN
    -- Split on '-' to separate version from label
    parts := string_to_array(version_string, '-');
    version_part := parts[1];

    IF array_length(parts, 1) > 1 THEN
        label_part := parts[2];
    ELSE
        label_part := NULL;
    END IF;

    -- Parse version numbers
    parts := string_to_array(version_part, '.');

    IF array_length(parts, 1) >= 3 THEN
        major := parts[1]::INT;
        minor := parts[2]::INT;
        patch := parts[3]::INT;
        label := label_part;
        RETURN NEXT;
    ELSE
        RAISE EXCEPTION 'Invalid version format: %', version_string;
    END IF;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to compare semantic versions
CREATE OR REPLACE FUNCTION compare_versions(v1_major INT, v1_minor INT, v1_patch INT, v2_major INT, v2_minor INT, v2_patch INT)
RETURNS INT AS $$
BEGIN
    IF v1_major > v2_major THEN RETURN 1;
    ELSIF v1_major < v2_major THEN RETURN -1;
    ELSIF v1_minor > v2_minor THEN RETURN 1;
    ELSIF v1_minor < v2_minor THEN RETURN -1;
    ELSIF v1_patch > v2_patch THEN RETURN 1;
    ELSIF v1_patch < v2_patch THEN RETURN -1;
    ELSE RETURN 0;
    END IF;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to get latest version of a testset
CREATE OR REPLACE FUNCTION get_latest_testset_version(testset_name VARCHAR)
RETURNS TABLE(
    id UUID,
    name VARCHAR,
    version_string VARCHAR,
    status VARCHAR,
    created_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        t.id,
        t.name,
        format_version(t.version_major, t.version_minor, t.version_patch, t.version_label),
        t.status,
        t.created_at
    FROM golden_testsets t
    WHERE t.name = testset_name
    ORDER BY t.version_major DESC, t.version_minor DESC, t.version_patch DESC, t.created_at DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql STABLE;

-- Function to calculate testset summary statistics
CREATE OR REPLACE FUNCTION get_testset_stats(testset_uuid UUID)
RETURNS TABLE(
    total_examples INT,
    avg_context_precision DECIMAL(3,2),
    avg_context_recall DECIMAL(3,2),
    avg_faithfulness DECIMAL(3,2),
    avg_answer_relevancy DECIMAL(3,2),
    retrieval_strategies TEXT[]
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*)::INT as total_examples,
        ROUND(AVG(e.context_precision), 2) as avg_context_precision,
        ROUND(AVG(e.context_recall), 2) as avg_context_recall,
        ROUND(AVG(e.faithfulness), 2) as avg_faithfulness,
        ROUND(AVG(e.answer_relevancy), 2) as avg_answer_relevancy,
        array_agg(DISTINCT e.retrieval_strategy) as retrieval_strategies
    FROM golden_examples e
    WHERE e.testset_id = testset_uuid;
END;
$$ LANGUAGE plpgsql STABLE;

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply updated_at trigger to golden_examples
DROP TRIGGER IF EXISTS update_golden_examples_updated_at ON golden_examples;
CREATE TRIGGER update_golden_examples_updated_at
    BEFORE UPDATE ON golden_examples
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Trigger to validate version progression
CREATE OR REPLACE FUNCTION validate_version_progression()
RETURNS TRIGGER AS $$
DECLARE
    latest_version RECORD;
BEGIN
    -- Get the latest version for this testset name
    SELECT version_major, version_minor, version_patch
    INTO latest_version
    FROM golden_testsets
    WHERE name = NEW.name
    ORDER BY version_major DESC, version_minor DESC, version_patch DESC
    LIMIT 1;

    -- If this is the first version, allow it
    IF latest_version IS NULL THEN
        RETURN NEW;
    END IF;

    -- Validate that the new version is greater than or equal to the latest
    -- Allow same version for idempotent operations (will be handled by unique constraint)
    IF compare_versions(NEW.version_major, NEW.version_minor, NEW.version_patch,
                       latest_version.version_major, latest_version.version_minor, latest_version.version_patch) < 0 THEN
        RAISE EXCEPTION 'New version %.%.% must be greater than or equal to latest version %.%.%',
            NEW.version_major, NEW.version_minor, NEW.version_patch,
            latest_version.version_major, latest_version.version_minor, latest_version.version_patch;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply version validation trigger
DROP TRIGGER IF EXISTS validate_testset_version ON golden_testsets;
CREATE TRIGGER validate_testset_version
    BEFORE INSERT ON golden_testsets
    FOR EACH ROW EXECUTE FUNCTION validate_version_progression();

"""

    def _generate_views(self) -> str:
        """Generate convenient views for common queries"""
        timestamp = datetime.now().isoformat()
        return f"""-- Golden Testset Schema - Views
-- Generated: {timestamp}

-- View for testset overview with version strings and stats
CREATE OR REPLACE VIEW testset_overview AS
SELECT
    t.id,
    t.name,
    format_version(t.version_major, t.version_minor, t.version_patch, t.version_label) as version,
    t.description,
    t.domain,
    t.source_type,
    t.status,
    t.validation_status,
    t.quality_score,
    t.created_at,
    t.created_by,
    t.approved_at,
    t.approved_by,
    t.phoenix_project_id,
    t.phoenix_experiment_id,

    -- Example statistics (computed)
    COALESCE(stats.total_examples, 0) as total_examples,
    stats.avg_context_precision,
    stats.avg_context_recall,
    stats.avg_faithfulness,
    stats.avg_answer_relevancy,
    stats.retrieval_strategies
FROM golden_testsets t
LEFT JOIN LATERAL (
    SELECT
        COUNT(*)::INT as total_examples,
        ROUND(AVG(e.context_precision), 2) as avg_context_precision,
        ROUND(AVG(e.context_recall), 2) as avg_context_recall,
        ROUND(AVG(e.faithfulness), 2) as avg_faithfulness,
        ROUND(AVG(e.answer_relevancy), 2) as avg_answer_relevancy,
        array_agg(DISTINCT e.retrieval_strategy) as retrieval_strategies
    FROM golden_examples e
    WHERE e.testset_id = t.id
) stats ON true;

-- View for latest versions of each testset
CREATE OR REPLACE VIEW latest_testsets AS
SELECT DISTINCT ON (t.name)
    t.id,
    t.name,
    format_version(t.version_major, t.version_minor, t.version_patch, t.version_label) as version,
    t.description,
    t.domain,
    t.source_type,
    t.status,
    t.validation_status,
    t.quality_score,
    t.created_at,
    t.created_by,
    t.approved_at,
    t.approved_by
FROM golden_testsets t
ORDER BY t.name, t.version_major DESC, t.version_minor DESC, t.version_patch DESC, t.created_at DESC;

-- View for examples with quality scores and metadata
CREATE OR REPLACE VIEW example_quality_view AS
SELECT
    e.id,
    e.testset_id,
    t.name as testset_name,
    format_version(t.version_major, t.version_minor, t.version_patch, t.version_label) as testset_version,
    e.question,
    e.ground_truth,
    e.contexts,
    e.retrieval_strategy,
    e.retrieval_score,

    -- Quality metrics
    e.context_precision,
    e.context_recall,
    e.faithfulness,
    e.answer_relevancy,

    -- Overall quality score (weighted average)
    ROUND((e.context_precision * 0.25 + e.context_recall * 0.25 + e.faithfulness * 0.25 + e.answer_relevancy * 0.25), 2) as overall_quality,

    -- RAGAS metadata
    e.ragas_question_type,
    e.ragas_evolution_type,
    e.ragas_difficulty,

    e.created_at,
    e.updated_at
FROM golden_examples e
JOIN golden_testsets t ON e.testset_id = t.id;

-- View for version history and changes
CREATE OR REPLACE VIEW version_history AS
SELECT
    t.id as testset_id,
    t.name as testset_name,
    format_version(t.version_major, t.version_minor, t.version_patch, t.version_label) as version,
    v.change_type,
    v.change_summary,
    v.examples_added,
    v.examples_modified,
    v.examples_removed,
    v.created_at as version_created_at,
    v.created_by as version_created_by,

    -- Parent version info
    pt.name as parent_testset_name,
    pv.version_string as parent_version
FROM golden_testsets t
LEFT JOIN testset_versions v ON t.id = v.testset_id
LEFT JOIN testset_versions pv ON v.parent_version_id = pv.id
LEFT JOIN golden_testsets pt ON pv.testset_id = pt.id
ORDER BY t.name, t.version_major DESC, t.version_minor DESC, t.version_patch DESC;

-- View for quality metrics dashboard
CREATE OR REPLACE VIEW quality_dashboard AS
SELECT
    t.id as testset_id,
    t.name as testset_name,
    format_version(t.version_major, t.version_minor, t.version_patch, t.version_label) as version,
    t.domain,
    t.status,
    t.validation_status,

    -- Computed quality metrics
    qm.metric_type,
    qm.metric_value,
    qm.passes_threshold,
    qm.computed_at,
    qm.computation_method,

    -- Approval status
    al.review_status as latest_review_status,
    al.reviewer as latest_reviewer,
    al.reviewed_at as latest_review_date
FROM golden_testsets t
LEFT JOIN testset_quality_metrics qm ON t.id = qm.testset_id
LEFT JOIN LATERAL (
    SELECT review_status, reviewer, reviewed_at
    FROM testset_approval_log
    WHERE testset_id = t.id
    ORDER BY reviewed_at DESC
    LIMIT 1
) al ON true;

-- View for Phoenix integration tracking
CREATE OR REPLACE VIEW phoenix_integration AS
SELECT
    t.id as testset_id,
    t.name as testset_name,
    format_version(t.version_major, t.version_minor, t.version_patch, t.version_label) as version,
    t.phoenix_project_id,
    t.phoenix_experiment_id,
    t.created_at as testset_created_at,

    -- Example count and quality summary
    COUNT(e.id) as total_examples,
    ROUND(AVG(e.context_precision), 2) as avg_context_precision,
    ROUND(AVG(e.context_recall), 2) as avg_context_recall,
    ROUND(AVG(e.faithfulness), 2) as avg_faithfulness,
    ROUND(AVG(e.answer_relevancy), 2) as avg_answer_relevancy,

    -- Retrieval strategy distribution
    array_agg(DISTINCT e.retrieval_strategy) as retrieval_strategies,
    COUNT(DISTINCT e.retrieval_strategy) as strategy_count
FROM golden_testsets t
LEFT JOIN golden_examples e ON t.id = e.testset_id
WHERE t.phoenix_project_id IS NOT NULL
GROUP BY t.id, t.name, t.version_major, t.version_minor, t.version_patch, t.version_label,
         t.phoenix_project_id, t.phoenix_experiment_id, t.created_at;

"""

    def _generate_sample_data(self) -> str:
        """Generate sample data for testing"""
        timestamp = datetime.now().isoformat()
        return f"""-- Golden Testset Schema - Sample Data
-- Generated: {timestamp}
-- This is test data for development and validation

-- Sample testset (version 1.0.0)
INSERT INTO golden_testsets (
    id, name, description, version_major, version_minor, version_patch,
    domain, source_type, status, validation_status, quality_score,
    created_by, phoenix_project_id, phoenix_experiment_id
) VALUES (
    'a0e8f4a7-8b4c-4c8d-9e5f-1a2b3c4d5e6f',
    'financial_aid_baseline',
    'Baseline golden testset for financial aid documents generated from RAGAS',
    1, 0, 0,
    'financial_aid', 'ragas', 'approved', 'passed', 0.85,
    'system', 'retrieval-evaluation-20250922', 'baseline-testset-v1'
) ON CONFLICT (id) DO NOTHING;

-- Sample examples for the testset
INSERT INTO golden_examples (
    id, testset_id, question, ground_truth, contexts,
    ragas_question_type, ragas_evolution_type, ragas_difficulty,
    retrieval_strategy, retrieval_score,
    context_precision, context_recall, faithfulness, answer_relevancy
) VALUES
(
    'b1f9e5b8-9c5d-5d9e-ae6f-2b3c4d5e6f71',
    'a0e8f4a7-8b4c-4c8d-9e5f-1a2b3c4d5e6f',
    'What are the eligibility requirements for Federal Pell Grants?',
    'Federal Pell Grants are available to undergraduate students who demonstrate exceptional financial need and have not earned a bachelor''s degree. Students must be enrolled in an eligible program, maintain satisfactory academic progress, and complete the FAFSA.',
    ARRAY[
        'Federal Pell Grants are need-based grants for undergraduate students who have not earned a bachelor''s or professional degree.',
        'To be eligible for a Pell Grant, students must demonstrate financial need through the FAFSA and meet academic requirements.'
    ],
    'simple', 'reasoning', 2.5,
    'semantic_chunking', 0.8842,
    0.89, 0.76, 0.92, 0.88
),
(
    'c2eae6c9-ad6e-6eaf-bf71-3c4d5e6f7182',
    'a0e8f4a7-8b4c-4c8d-9e5f-1a2b3c4d5e6f',
    'How does the Direct Loan Program work for graduate students?',
    'Graduate students can borrow both subsidized and unsubsidized Direct Loans. The interest rate and borrowing limits are set annually. Graduate students are also eligible for PLUS loans for additional funding beyond Direct Loan limits.',
    ARRAY[
        'The Direct Loan Program offers federal student loans directly from the U.S. Department of Education.',
        'Graduate students have access to both subsidized and unsubsidized loans with specific borrowing limits.',
        'PLUS loans are available for graduate students who need additional funding beyond Direct Loan limits.'
    ],
    'complex', 'multi_context', 3.2,
    'ensemble', 0.9156,
    0.82, 0.85, 0.89, 0.91
),
(
    'd3ebf7da-be7f-7fbf-c182-4d5e6f718293',
    'a0e8f4a7-8b4c-4c8d-9e5f-1a2b3c4d5e6f',
    'What documentation is required for FAFSA verification?',
    'FAFSA verification typically requires tax transcripts or tax returns, W-2 forms, bank statements, investment records, and records of untaxed income. Some students may also need to provide Social Security benefits statements or child support records.',
    ARRAY[
        'Verification documentation includes federal tax transcripts, W-2 forms, and bank statements.',
        'Students selected for verification must provide proof of income, assets, and household composition.',
        'The specific documents required depend on the items selected for verification by the Department of Education.'
    ],
    'simple', 'reasoning', 2.8,
    'contextual_compression', 0.8734,
    0.91, 0.79, 0.86, 0.84
);

-- Sample version record
INSERT INTO testset_versions (
    id, testset_id, version_string, change_type, change_summary,
    examples_added, examples_modified, examples_removed, created_by
) VALUES (
    'e4ec18eb-cf81-81c1-d192-5e6f71829304',
    'a0e8f4a7-8b4c-4c8d-9e5f-1a2b3c4d5e6f',
    '1.0.0',
    'major',
    'Initial baseline testset created from RAGAS generation using financial aid PDFs',
    3, 0, 0,
    'system'
) ON CONFLICT DO NOTHING;

-- Sample quality metrics
INSERT INTO testset_quality_metrics (
    testset_id, metric_type, metric_value, metric_details,
    computed_by, computation_method, threshold_min, passes_threshold
) VALUES
(
    'a0e8f4a7-8b4c-4c8d-9e5f-1a2b3c4d5e6f',
    'coverage', 0.87,
    '{{"domain_coverage": {{"pell_grants": 0.9, "direct_loans": 0.85, "verification": 0.85}}}}'::jsonb,
    'validation_script', 'automated', 0.80, true
),
(
    'a0e8f4a7-8b4c-4c8d-9e5f-1a2b3c4d5e6f',
    'diversity', 0.82,
    '{{"question_types": {{"simple": 2, "complex": 1}}, "difficulty_range": [2.5, 3.2]}}'::jsonb,
    'validation_script', 'automated', 0.75, true
),
(
    'a0e8f4a7-8b4c-4c8d-9e5f-1a2b3c4d5e6f',
    'quality', 0.85,
    '{{"avg_context_precision": 0.87, "avg_context_recall": 0.80, "avg_faithfulness": 0.89, "avg_answer_relevancy": 0.88}}'::jsonb,
    'validation_script', 'ragas', 0.80, true
);

-- Sample approval record
INSERT INTO testset_approval_log (
    testset_id, reviewer, review_status, review_comments, review_checklist
) VALUES (
    'a0e8f4a7-8b4c-4c8d-9e5f-1a2b3c4d5e6f',
    'data_scientist',
    'approved',
    'Baseline testset meets quality thresholds and covers core financial aid topics. Ready for evaluation experiments.',
    '{{"quality_check": true, "coverage_check": true, "diversity_check": true, "accuracy_check": true}}'::jsonb
) ON CONFLICT DO NOTHING;

"""

    def _generate_combined_schema(self, schemas: Dict[str, str]) -> str:
        """Generate a combined schema file with all components"""
        header = f"""-- Golden Testset Management - Complete Schema
-- Generated: {datetime.now().isoformat()}
--
-- This schema provides:
-- - Versioned golden testset storage with semantic versioning
-- - Phoenix/Arize experiment tracking integration
-- - Quality validation and approval workflows
-- - Vector similarity search for questions and answers
-- - Audit trails and lineage tracking
--
-- Usage:
--   psql -h localhost -p 6024 -U langchain -d langchain -f schemas/golden_testset_schema.sql
--
-- Components:
--   1. Core tables (testsets, examples, versions, quality metrics, approvals)
--   2. Indexes for performance (HNSW vector indexes, GIN text search)
--   3. Functions and triggers (version validation, statistics)
--   4. Views for common queries (overview, latest versions, quality dashboard)
--   5. Sample data for testing

"""

        combined = header
        for filename in ["01_core_tables.sql", "02_indexes.sql", "03_functions.sql", "04_views.sql", "05_sample_data.sql"]:
            combined += f"\n-- =============================================================================\n"
            combined += f"-- {filename}\n"
            combined += f"-- =============================================================================\n\n"
            combined += schemas[filename]
            combined += "\n"

        return combined

    async def test_connection(self) -> bool:
        """Test database connection"""
        try:
            conn = await asyncpg.connect(self.connection_string)
            version = await conn.fetchval("SELECT version()")
            await conn.close()
            print(f"✓ Database connection successful: {version[:50]}...")
            return True
        except Exception as e:
            print(f"✗ Database connection failed: {e}")
            return False

    async def apply_schema(self, force: bool = False) -> bool:
        """Apply the generated schema to the database"""
        try:
            conn = await asyncpg.connect(self.connection_string)

            # Read combined schema
            schema_file = self.schema_dir / "golden_testset_schema.sql"
            if not schema_file.exists():
                print(f"✗ Schema file not found: {schema_file}")
                return False

            schema_sql = schema_file.read_text()

            # Check if tables already exist
            existing_tables = await conn.fetch("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name LIKE 'golden_%'
            """)

            if existing_tables and not force:
                print(f"⚠ Found {len(existing_tables)} existing golden testset tables:")
                for table in existing_tables:
                    print(f"  - {table['table_name']}")
                print("Use --force to overwrite existing schema")
                await conn.close()
                return False

            # Apply schema
            print("Applying golden testset schema...")
            await conn.execute(schema_sql)
            print("✓ Schema applied successfully")

            # Verify tables were created
            new_tables = await conn.fetch("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name LIKE 'golden_%'
                ORDER BY table_name
            """)

            print(f"✓ Created {len(new_tables)} tables:")
            for table in new_tables:
                print(f"  - {table['table_name']}")

            await conn.close()
            return True

        except Exception as e:
            print(f"✗ Schema application failed: {e}")
            return False


async def main():
    """Main entry point for schema generation"""
    import argparse

    parser = argparse.ArgumentParser(description="Golden Testset Schema Generator")
    parser.add_argument("--connection", help="PostgreSQL connection string")
    parser.add_argument("--apply", action="store_true", help="Apply schema to database")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing schema")
    parser.add_argument("--test-connection", action="store_true", help="Test database connection only")

    args = parser.parse_args()

    generator = SchemaGenerator(args.connection)

    if args.test_connection:
        success = await generator.test_connection()
        sys.exit(0 if success else 1)

    # Generate schema files
    print("Generating golden testset schema...")
    schemas = await generator.generate_schema()
    print(f"✓ Generated {len(schemas)} schema files in {generator.schema_dir}")

    if args.apply:
        success = await generator.apply_schema(args.force)
        if not success:
            sys.exit(1)

        print("\n✓ Golden testset schema ready for use!")
        print("\nNext steps:")
        print("  1. Verify tables: \\dt golden_*")
        print("  2. Check views: \\dv *testset*")
        print("  3. Test functions: SELECT get_latest_testset_version('financial_aid_baseline');")


if __name__ == "__main__":
    asyncio.run(main())