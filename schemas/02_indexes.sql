-- Golden Testset Schema - Indexes and Performance
-- Generated: 2025-09-23T08:11:31.731003

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

