-- Golden Testset Schema - Views
-- Generated: 2025-09-23T08:11:31.731006

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

