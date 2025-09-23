-- Golden Testset Schema - Sample Data
-- Generated: 2025-09-23T08:11:31.731008
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
    '{"domain_coverage": {"pell_grants": 0.9, "direct_loans": 0.85, "verification": 0.85}}'::jsonb,
    'validation_script', 'automated', 0.80, true
),
(
    'a0e8f4a7-8b4c-4c8d-9e5f-1a2b3c4d5e6f',
    'diversity', 0.82,
    '{"question_types": {"simple": 2, "complex": 1}, "difficulty_range": [2.5, 3.2]}'::jsonb,
    'validation_script', 'automated', 0.75, true
),
(
    'a0e8f4a7-8b4c-4c8d-9e5f-1a2b3c4d5e6f',
    'quality', 0.85,
    '{"avg_context_precision": 0.87, "avg_context_recall": 0.80, "avg_faithfulness": 0.89, "avg_answer_relevancy": 0.88}'::jsonb,
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
    '{"quality_check": true, "coverage_check": true, "diversity_check": true, "accuracy_check": true}'::jsonb
) ON CONFLICT DO NOTHING;

