# Architecture Diagrams

This directory contains architectural diagrams for the RAG Evaluation Foundation project created with [Excalidraw](https://excalidraw.com/).

## Diagram Overview

### 📊 pipeline.excalidraw
**Status**: ⚠️ In Progress  
**Purpose**: Main RAG evaluation pipeline showing the flow from data ingestion through retrieval strategies to evaluation metrics.

### 🔍 query.excalidraw  
**Status**: ⚠️ Minimal Content  
**Purpose**: Query processing flow diagram showing how user queries are handled by different retrieval strategies.

### 💾 data-pipeline.excalidraw
**Status**: ❌ Empty  
**Purpose**: Intended to show the data ingestion pipeline from CSV files to PostgreSQL/pgvector storage.

## Viewing and Editing

### Option 1: VS Code Extension
1. Install the [Excalidraw extension](https://marketplace.visualstudio.com/items?itemName=pomdtr.excalidraw-editor)
2. Open `.excalidraw` files directly in VS Code

### Option 2: Web Editor
1. Visit [excalidraw.com](https://excalidraw.com/)
2. File → Open → Select the `.excalidraw` file

### Option 3: View Exports
Check the `exports/` directory for PNG/SVG versions (if available).

## Exporting Diagrams

When updating diagrams, please export them:

1. **PNG Format** (for documentation):
   - File → Export image → PNG
   - Save to `diagrams/exports/[name].png`
   - Use 2x scale for clarity

2. **SVG Format** (for web):
   - File → Export image → SVG
   - Save to `diagrams/exports/[name].svg`

## Diagram Standards

- **Colors**: Use consistent colors for components
  - 🟦 Blue: External services (PostgreSQL, Phoenix)
  - 🟩 Green: Retrieval strategies
  - 🟨 Yellow: User inputs/outputs
  - 🟥 Red: Error states
  
- **Labels**: Clear, descriptive text for all components
- **Flow**: Left-to-right or top-to-bottom
- **Grouping**: Related components should be visually grouped

## Planned Diagrams

1. **System Architecture Overview** - High-level view of all components
2. **Retrieval Strategy Comparison** - Visual comparison of 6 strategies
3. **Experiment Flow** - How experiments are executed with Phoenix
4. **Data Model** - PostgreSQL schema and relationships

## Contributing

When adding or updating diagrams:
1. Update this README with the diagram's purpose
2. Export to both PNG and SVG formats
3. Reference the diagram in relevant documentation
4. Commit both source (.excalidraw) and exports