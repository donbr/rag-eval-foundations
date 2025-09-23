-- Golden Testset Schema - Functions and Triggers
-- Generated: 2025-09-23T08:11:31.731004

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

