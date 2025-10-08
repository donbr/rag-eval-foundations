# RAGAS to Phoenix Integration - Code Dependencies and Flow

This document captures the code architecture, dependencies, and execution flow for the RAGAS golden testset to Phoenix integration.

## Architecture Overview

```mermaid
graph TB
    subgraph "Entry Point"
        Entry[langchain_eval_golden_testset.py]
    end

    subgraph "Configuration Layer"
        Config[setup_environment]
        EnvVars[Environment Variables]
    end

    subgraph "Data Layer"
        DataLoader[data_loader.py]
        PostgreSQL[(PostgreSQL Database)]
        JSON[golden_testset.json]
    end

    subgraph "RAGAS Pipeline"
        RAGAS[RAGAS TestsetGenerator]
        LLMWrapper[LangchainLLMWrapper]
        EmbWrapper[LangchainEmbeddingsWrapper]
        OpenAI[OpenAI APIs]
    end

    subgraph "Phase 4 Integration"
        Manager[GoldenTestsetManager]
        Phoenix[PhoenixIntegration]
        HybridCost[HybridCostManager]
        Tracing[tracing.py]
    end

    subgraph "External Services"
        PhoenixPlatform[Phoenix/Arize Platform]
        OpenAIService[OpenAI Service]
    end

    Entry --> Config
    Entry --> DataLoader
    Entry --> RAGAS
    Entry --> Manager

    Config --> EnvVars
    DataLoader --> PostgreSQL

    RAGAS --> LLMWrapper
    RAGAS --> EmbWrapper
    LLMWrapper --> OpenAI
    EmbWrapper --> OpenAI
    OpenAI --> OpenAIService

    Manager --> Phoenix
    Phoenix --> HybridCost
    Phoenix --> Tracing
    Phoenix --> PhoenixPlatform

    Entry --> JSON
    Phoenix --> PostgreSQL

    style Entry fill:#e1f5fe
    style RAGAS fill:#f3e5f5
    style Phoenix fill:#e8f5e8
    style PhoenixPlatform fill:#fff3e0
```

## Code Dependencies Flow

```mermaid
flowchart TD
    subgraph "Import Dependencies"
        I1[asyncio, json, datetime]
        I2[pandas, os]
        I3[langchain_openai]
        I4[ragas.llms, ragas.embeddings]
        I5[ragas.testset]
        I6[data_loader]
        I7[langchain_eval_foundations_e2e]
        I8[golden_testset.phoenix_integration]
        I9[golden_testset.manager]
    end

    subgraph "Function Definitions"
        F1[generate_testset]
        F2[upload_to_phoenix_integrated]
        F3[main - async]
    end

    subgraph "Execution Flow"
        E1[Setup Environment & Config]
        E2[Initialize LLM & Embeddings]
        E3[Load Documents from PostgreSQL]
        E4[Generate RAGAS Testset]
        E5[Save JSON Backup]
        E6[Initialize Phoenix Integration]
        E7[Transform Data Format]
        E8[Upload to Phoenix]
        E9[Display Results]
    end

    I1 --> F3
    I2 --> F2
    I3 --> E2
    I4 --> E2
    I5 --> F1
    I6 --> E3
    I7 --> E1
    I8 --> E6
    I9 --> E6

    F3 --> E1
    E1 --> E2
    E2 --> E3
    E3 --> F1
    F1 --> E4
    E4 --> E5
    E5 --> E6
    E6 --> F2
    F2 --> E7
    E7 --> E8
    E8 --> E9

    style F3 fill:#e3f2fd
    style E4 fill:#f1f8e9
    style E8 fill:#fff8e1
```

## Module Dependencies

### Core Application Files

```mermaid
classDiagram
    class langchain_eval_golden_testset {
        +generate_testset()
        +upload_to_phoenix_integrated()
        +main()
    }

    class data_loader {
        +load_docs_from_postgres()
    }

    class setup_environment {
        +Config
        +model_name
        +embedding_model
        +table_baseline
    }

    class GoldenTestsetManager {
        +async operations
        +database management
    }

    class PhoenixIntegration {
        +upload_testset()
        +_prepare_phoenix_dataset()
        +_upload_to_phoenix()
        +setup_default_model_pricing()
    }

    class PhoenixConfig {
        +endpoint
        +api_key
        +project_name
    }

    langchain_eval_golden_testset --> data_loader
    langchain_eval_golden_testset --> setup_environment
    langchain_eval_golden_testset --> GoldenTestsetManager
    langchain_eval_golden_testset --> PhoenixIntegration
    PhoenixIntegration --> PhoenixConfig
    PhoenixIntegration --> GoldenTestsetManager
```

## Detailed Function Flow

### 1. Main Function Execution

```mermaid
flowchart TD
    Start([python langchain_eval_golden_testset.py]) --> SetupEnv[setup_environment]
    SetupEnv --> InitLLM[Initialize ChatOpenAI]
    InitLLM --> InitEmbed[Initialize OpenAIEmbeddings]
    InitEmbed --> WrapLLM[LangchainLLMWrapper]
    WrapLLM --> WrapEmbed[LangchainEmbeddingsWrapper]
    WrapEmbed --> LoadDocs[load_docs_from_postgres]
    LoadDocs --> CheckSize{Check GOLDEN_TESTSET_SIZE}
    CheckSize --> GenTestset[generate_testset]
    GenTestset --> SaveJSON[Save to golden_testset.json]
    SaveJSON --> InitManager[GoldenTestsetManager]
    InitManager --> InitPhoenix[PhoenixIntegration]
    InitPhoenix --> Transform[Transform Data Format]
    Transform --> Upload[upload_to_phoenix_integrated]
    Upload --> CheckResult{Upload Success?}
    CheckResult -->|Yes| Success[Display Success Info]
    CheckResult -->|No| Error[Display Error Info]
    Success --> End([Workflow Complete])
    Error --> End
```

### 2. RAGAS Generation Pipeline

```mermaid
flowchart LR
    subgraph "RAGAS Internal Pipeline"
        A[HeadlinesExtractor] --> B[HeadlineSplitter]
        B --> C[SummaryExtractor]
        C --> D[CustomNodeFilter]
        D --> E[EmbeddingExtractor]
        E --> F[ThemesExtractor]
        F --> G[NERExtractor]
        G --> H[CosineSimilarityBuilder]
        H --> I[OverlapScoreBuilder]
        I --> J[Generate Personas]
        J --> K[Generate Scenarios]
        K --> L[Generate Q&A Pairs]
    end

    Input[Document List] --> A
    L --> Output[DataFrame with testset]

    style A fill:#ffebee
    style L fill:#e8f5e8
```

### 3. Phoenix Upload Process

```mermaid
flowchart TD
    Input[RAGAS DataFrame] --> Transform[Transform to Phoenix Format]
    Transform --> Metadata[Add Metadata]
    Metadata --> Call[phoenix_integration.upload_testset]
    Call --> Prepare[_prepare_phoenix_dataset]
    Prepare --> Version[Generate Version]
    Version --> Upload[_upload_to_phoenix]
    Upload --> HTTP[HTTP POST to Phoenix]
    HTTP --> UpdateMeta[_update_upload_metadata]
    UpdateMeta --> Result[Return Upload Result]

    style Input fill:#e3f2fd
    style HTTP fill:#fff3e0
    style Result fill:#e8f5e8
```

## Data Structure Transformations

### RAGAS DataFrame → Phoenix Format

```python
# Input: RAGAS DataFrame
{
    "user_input": str,
    "reference": str,
    "reference_contexts": List[str],
    "synthesizer_name": str,
    "evolution_type": str
}

# Transformation Logic
def transform_ragas_to_phoenix(df):
    testset_data = {
        "examples": [],
        "metadata": {
            "source": "ragas_golden_testset",
            "generation_method": "automated",
            "created_at": datetime.now().isoformat(),
            "num_samples": len(df)
        }
    }

    for _, row in df.iterrows():
        example = {
            "question": row["user_input"],
            "ground_truth": row["reference"],
            "contexts": row["reference_contexts"],
            "metadata": {
                "synthesizer_name": row.get("synthesizer_name"),
                "source": "ragas_testset_generator"
            }
        }
        testset_data["examples"].append(example)

    return testset_data

# Output: Phoenix Compatible Format
{
    "examples": [
        {
            "question": str,
            "ground_truth": str,
            "contexts": List[str],
            "metadata": Dict[str, Any]
        }
    ],
    "metadata": Dict[str, Any]
}
```

## Configuration Dependencies

### Environment Variables
- `GOLDEN_TESTSET_SIZE`: Controls testset size
- `OPENAI_API_KEY`: Required for RAGAS generation
- `DATABASE_URL`: PostgreSQL connection
- `PHOENIX_ENDPOINT`: Phoenix platform endpoint

### Model Configuration
- **LLM Model**: `gpt-4.1-mini` (hardcoded)
- **Embedding Model**: `text-embedding-3-small` (hardcoded)
- **Database Table**: `mixed_baseline_documents`

## Error Handling Flow

```mermaid
flowchart TD
    Start[Function Entry] --> TryBlock{Try Block}
    TryBlock --> Success[Successful Execution]
    TryBlock --> Exception[Exception Caught]

    Exception --> LogError[Log Error Details]
    LogError --> ReturnError[Return Error Dict]

    Success --> ReturnSuccess[Return Success Dict]

    ReturnError --> ErrorDisplay[Display Error Message]
    ReturnSuccess --> SuccessDisplay[Display Success Message]

    ErrorDisplay --> Continue[Continue Execution]
    SuccessDisplay --> Continue

    style Exception fill:#ffebee
    style Success fill:#e8f5e8
```

## File I/O Operations

### Input Files
- **Database**: PostgreSQL `mixed_baseline_documents` table
- **Config**: Environment variables and setup configuration

### Output Files
- **Local Backup**: `golden_testset.json` (RAGAS format)
- **Phoenix Dataset**: Versioned upload to Phoenix platform
- **Database Metadata**: Upload tracking in PostgreSQL

## Async/Await Pattern

```python
# Main execution pattern
async def main():
    # Sync operations
    config = setup_environment()
    llm = ChatOpenAI(model=config.model_name)

    # Sync RAGAS generation
    golden_testset_df = generate_testset(docs, llm, embeddings, size)

    # Async Phoenix operations
    phoenix_integration = PhoenixIntegration(manager, config)
    result = await upload_to_phoenix_integrated(df, phoenix_integration)

    return result

# Entry point
if __name__ == "__main__":
    asyncio.run(main())
```

## Integration Points with Phase 4 Architecture

### Phoenix Integration Class Usage
```python
# Initialization
manager = GoldenTestsetManager()
phoenix_config = PhoenixConfig()
phoenix_integration = PhoenixIntegration(manager, phoenix_config)

# Upload operation
result = await phoenix_integration.upload_testset(testset_data)
```

### Cost Tracking Integration
- Uses `HybridCostManager` for token usage tracking
- OpenTelemetry tracing via `tracing.py`
- Session-based cost aggregation

### Database Integration
- PostgreSQL for document storage and metadata
- Async operations via `GoldenTestsetManager`
- Version tracking and audit trails

This architecture provides a robust, scalable solution for generating and managing golden testsets with full Phoenix platform integration.