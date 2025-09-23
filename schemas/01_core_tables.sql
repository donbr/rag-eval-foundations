-- Golden Testset Management Schema - Core Tables
-- Generated: 2025-09-23T08:11:31.730991

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

