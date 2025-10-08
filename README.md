# 🚀 RAG Evaluation Pipeline: Complete 3-Stage Toolkit for LangChain & RAGAS

*Welcome, brave soul, to the wild world of Retrieval-Augmented Generation evaluation! You're about to embark on a journey that's part software engineering, part data science, and part digital archaeology. Don't worry—we've got your back.*

For a deep-dive into the technical concepts and a complete walkthrough of the code, see the [full technical guide](docs/blog/langchain_eval_foundations_e2e_blog.md).

## 📚 What You'll Learn

- How to implement and compare 6 different retrieval strategies
- The difference between semantic search, keyword search (BM25), and hybrid approaches
- How to use Phoenix for LLM observability and debugging
- Setting up a vector database with PostgreSQL and pgvector
- Complete RAGAS implementation for golden test sets and automated evaluation

📖 **Want deeper technical insights?** Check out the [DeepWiki documentation](https://deepwiki.com/donbr/rag-eval-foundations) for architecture diagrams and detailed explanations.

## 📋 Prerequisites

- Python 3.13+ (required by the project)
- Basic understanding of LLMs and embeddings
- Familiarity with async Python (we use asyncio)
- ~$5 in API credits (OpenAI + Cohere)
- Docker installed and running

## 🎯 What You're Getting Into

**You're getting a complete 3-stage RAG evaluation pipeline!** This isn't just a foundation - it's a full toolkit that takes you from infrastructure setup through automated evaluation with RAGAS golden test sets.

You're building:
1. **Stage 1**: A comparison engine for 6 different retrieval strategies
2. **Stage 2**: Automated test generation with RAGAS golden datasets
3. **Stage 3**: Systematic evaluation with metrics and experiment tracking

Think of it as a complete RAG evaluation laboratory, with research PDFs (AI/HCI literature) as your test dataset. By the end, you'll have objective metrics telling you which retrieval strategy performs best.

## 🎯 The Complete 3-Stage Pipeline

This toolkit implements a production-ready evaluation pipeline that progresses through three stages:

### 🏭 Stage 1: Foundation & Infrastructure
**Script**: `langchain_eval_foundations_e2e.py`
- Sets up PostgreSQL with pgvector for hybrid search
- Implements 6 different retrieval strategies
- Provides side-by-side comparison with Phoenix tracing
- **You learn**: How different retrieval methods work and when to use each

### 🧪 Stage 2: Golden Test Set Generation  
**Script**: `langchain_eval_golden_testset.py`
- Uses RAGAS to automatically generate diverse test questions
- Creates ground-truth answers and reference contexts
- Uploads datasets to Phoenix for experiment tracking
- **You get**: A reusable test set for consistent evaluation

### 📊 Stage 3: Automated Evaluation
**Script**: `langchain_eval_experiments.py`
- Runs all strategies against the golden test set
- Calculates QA correctness and relevance metrics
- Provides quantitative rankings and performance data
- **You discover**: Which strategy objectively performs best

🔄 **All three stages work together** to give you a complete evaluation workflow from setup to metrics!

---

## 🚀 Quick Start (for the impatient)

### Option A: One-Command Pipeline (Recommended)

```bash
# 1. Clone and setup
git clone <repo-url>
cd rag-eval-foundations
cp .env.example .env  # Edit with your API keys

# 2. Install dependencies
uv venv --python 3.13 && source .venv/bin/activate
uv sync

# 3. Run the complete pipeline
python claude_code_scripts/run_rag_evaluation_pipeline.py
```

The orchestration script will:
- ✅ Validate your environment and API keys
- 🐳 Start Docker services (PostgreSQL + Phoenix)
- 🔄 Execute all 3 pipeline steps in correct order
- 📊 Generate comprehensive evaluation results

### Option B: Manual Step-by-Step

```bash
# 1. Clone and setup
git clone <repo-url>
cd rag-eval-foundations
cp .env.example .env  # Edit with your API keys

# 2. Start services
docker-compose up -d  # Or use individual docker run commands below

# 3. Install and run manually
uv venv --python 3.13 && source .venv/bin/activate
uv sync
python src/langchain_eval_foundations_e2e.py
python src/langchain_eval_golden_testset.py
python src/langchain_eval_experiments.py
```

## 🌉 Next Steps: Deepen Your Understanding

Now that you've got the pipeline running, here's where to go next:

### 📖 **For Technical Deep Dives**
- **[DeepWiki Documentation](https://deepwiki.com/donbr/rag-eval-foundations)**: Interactive architecture diagrams, detailed retrieval strategy analysis, and performance benchmarks
- **[Technical Blog Post](docs/blog/langchain_eval_foundations_e2e_blog.md)**: Complete walkthrough of the implementation with code examples

### 🧪 **For Hands-On Exploration**
- **[Validation Scripts](validation/README.md)**: Interactive tools to explore your data and compare strategies
  - `postgres_data_analysis.py`: Visualize embeddings and chunking strategies
  - `retrieval_strategy_comparison.py`: Benchmark and compare all 6 strategies
  - `validate_telemetry.py`: Understand Phoenix tracing in depth

### 🚀 **For Production Readiness**
- Learn about [RAGAS golden test sets](https://docs.ragas.io/) for automated evaluation
- Explore [Phoenix documentation](https://docs.arize.com/phoenix) for advanced observability
- Check out [LangChain's retriever docs](https://python.langchain.com/docs/modules/data_connection/retrievers/) for custom implementations

## 🎯 Pipeline Orchestration Script

The `claude_code_scripts/run_rag_evaluation_pipeline.py` script provides a comprehensive, repeatable process for executing all 3 pipeline steps with proper error handling and logging.

### Features
- **🔍 Environment Validation**: Checks .env file, API keys, and dependencies
- **🐳 Service Management**: Automatically starts Docker services if needed
- **📋 Step-by-Step Execution**: Runs all 3 scripts in correct dependency order
- **📊 Comprehensive Logging**: Detailed logs with timestamps and progress tracking
- **❌ Error Handling**: Graceful failure recovery and clear error messages

### Usage Examples

```bash
# Standard execution (recommended)
python claude_code_scripts/run_rag_evaluation_pipeline.py

# Skip Docker service management (if already running)
python claude_code_scripts/run_rag_evaluation_pipeline.py --skip-services

# Enable verbose debug logging
python claude_code_scripts/run_rag_evaluation_pipeline.py --verbose

# Get help
python claude_code_scripts/run_rag_evaluation_pipeline.py --help
```

### Pipeline Steps Executed

1. **Main E2E Pipeline** (`langchain_eval_foundations_e2e.py`)
   - Loads documents from configured sources (PDFs by default)
   - Creates PostgreSQL vector stores
   - Tests 6 retrieval strategies
   - Generates Phoenix traces

2. **Golden Test Set Generation** (`langchain_eval_golden_testset.py`)
   - Uses RAGAS to generate evaluation questions
   - Uploads test set to Phoenix for experiments

3. **Automated Experiments** (`langchain_eval_experiments.py`)
   - Runs systematic evaluation on all strategies
   - Calculates QA correctness and relevance scores
   - Creates detailed experiment reports in Phoenix

### 📁 Data Source Configuration

The pipeline supports multiple document formats and can be configured via the Config class in `langchain_eval_foundations_e2e.py`:

**Default Configuration (current):**
```python
load_pdfs: bool = True      # Research PDFs from data/ directory (enabled)
load_csvs: bool = False     # CSV datasets (disabled by default)
load_markdowns: bool = True # Markdown documents (enabled)
```

**Data Sources by Type:**
- **PDFs**: Research papers, technical documents (AI/HCI literature included)
- **CSVs**: Structured datasets with metadata columns
- **Markdown**: Documentation files split on header boundaries

**Example Data Included:**
- Current dataset: 269 PDF documents on human-LLM interaction and AI usage research
- Topics: Prompt engineering, trust calibration, cognitive collaboration, interface design

**To use different data:**
1. Place your PDFs in `data/` directory
2. Update config flags in `langchain_eval_foundations_e2e.py`
3. Run the pipeline - system automatically adapts to your documents

**Example queries for current research data:**
- "What factors influence user trust in AI systems?"
- "How do novice users struggle with prompt engineering?"
- "What design strategies mitigate automation bias?"

### Logs and Output

The script creates detailed logs in the `logs/` directory with timestamps. All output includes:
- ✅ Success indicators for each step
- ⏱️ Execution time tracking  
- 🔗 Direct links to Phoenix UI for viewing results
- 📊 Summary statistics and experiment IDs

## 🛠️ Pre-Flight Checklist

### Step 1: Gather Your Supplies

This project uses `uv`, an extremely fast Python package and project manager.

1.  **Install `uv`**

    If you don't have `uv` installed, open your terminal and run the official installer:

    ```bash
    # Install uv (macOS & Linux)
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
    For Windows and other installation methods, please refer to the [official uv installation guide](https://docs.astral.sh/uv/getting-started/installation).

2.  **Create Environment & Install Dependencies**

    With `uv` installed, you can create a virtual environment and install all the necessary packages from `pyproject.toml` in two commands:

    ```bash
    # Create a virtual environment with Python 3.13+
    uv venv --python 3.13

    # Activate the virtual environment
    # On macOS/Linux:
    source .venv/bin/activate
    # On Windows (CMD):
    # .venv\Scripts\activate.bat

    # Install dependencies into the virtual environment
    uv sync
    ```

*If you're new to `uv`, think of `uv venv` as a replacement for `python -m venv` and `uv sync` as a much faster version of `pip install -r requirements.txt`.*

### Step 2: Secret Agent Setup
Create a `.env` file (because hardcoding API keys is how we end up on r/ProgrammerHumor):

```bash
OPENAI_API_KEY=sk-your-actual-key-not-this-placeholder
COHERE_API_KEY=your-cohere-key-goes-here
PHOENIX_COLLECTOR_ENDPOINT="http://localhost:6006"
# Optional:
HUGGINGFACE_TOKEN=hf_your_token_here
PHOENIX_CLIENT_HEADERS='...'  # For cloud Phoenix instances
```

📝 **Note:** See `.env.example` for a complete template with all supported variables.

**Pro tip:** Yes, you need both keys. Yes, they cost money. Yes, it's worth it. Think of it as buying premium gas for your AI Ferrari.

---

## 🐳 Docker: Your New Best Friends

**Quick heads-up:** We're using interactive mode (`-it --rm`) for easy cleanup - when you kill these containers, all data vanishes. Perfect for demos, terrible if you want to keep anything. For persistent setups, use `docker-compose` instead.

### Friend #1: PostgreSQL + pgvector (The Data Vault)


```bash
docker run -it --rm --name pgvector-container \
  -e POSTGRES_USER=langchain \
  -e POSTGRES_PASSWORD=langchain \
  -e POSTGRES_DB=langchain \
  -p 6024:5432 \
  pgvector/pgvector:pg16
```

*This is your vector database. It's like a regular database, but it can do math with meanings. Fancy stuff.*

### Friend #2: Phoenix Observability (The All-Seeing Eye)
```bash
export TS=$(date +"%Y%m%d_%H%M%S")
docker run -it --rm --name phoenix-container \
  -e PHOENIX_PROJECT_NAME="retrieval-comparison-${TS}" \
  -p 6006:6006 \
  -p 4317:4317 \
  arizephoenix/phoenix:latest
```

*Phoenix watches everything your AI does and tells you where it went wrong. It's like having a really helpful, non-judgmental therapist for your code.*

⚠️ **Port Notes:** 
- Port 6006: Phoenix UI (view traces here)
- Port 4317: OpenTelemetry collector (receives trace data)

### Alternative: Use Docker Compose (Easier!)
```bash
docker-compose up -d
```
This starts both PostgreSQL and Phoenix with the correct settings.

---

## 🚦 The Moment of Truth

### Launch Sequence
```bash
python src/langchain_eval_foundations_e2e.py
```

**What should happen:**
1. 📥 Loads research PDF documents (AI/HCI literature from the data/ directory)
2. 🗄️ Creates fancy vector tables in PostgreSQL
3. 🔍 Tests 6 different ways to search for information
4. 📊 Shows you which method gives the best answers
5. 🕵️ Sends traces to Phoenix UI at `http://localhost:6006`

**Expected runtime:** 2-5 minutes (depending on number of PDFs loaded)

---

## 🚨 When Things Go Sideways (And They Will)

### The Greatest Hits of Failure

**"ModuleNotFoundError: No module named 'whatever'"**
- *Translation:* You forgot to install something
- *Solution:* Make sure you ran `uv sync` after activating your venv
- *Encouragement:* Even senior developers forget to activate their venv

**"Connection refused" or "Port already in use"**
- *Translation:* Docker containers aren't happy
- *Solution:* 
  ```bash
  docker ps  # Check what's running
  docker logs pgvector-container  # Check PostgreSQL logs
  docker logs phoenix-container   # Check Phoenix logs
  # If ports are taken:
  lsof -i :6024  # Check what's using PostgreSQL port
  lsof -i :6006  # Check what's using Phoenix port
  ```
- *Encouragement:* Docker is like a moody teenager—sometimes you just need to restart everything

**"Invalid API key" or "Rate limit exceeded"**
- *Translation:* OpenAI/Cohere is giving you the cold shoulder
- *Solution:* Check your `.env` file, verify your API keys have credits
- *Encouragement:* At least the error is clear! Better than "something went wrong" 🤷

**"Async this, await that, event loop already running"**
- *Translation:* Python's async system is having an existential crisis
- *Solution:* Restart your Python session, try again
- *Encouragement:* Async programming is hard. If it was easy, we'd all be doing it

---

## 🆘 Emergency Protocols

### When All Else Fails: The Claude/ChatGPT Lifeline

Copy your error message and ask:
> "I'm running this RAG evaluation foundations setup with LangChain, PostgreSQL, and Phoenix. I'm getting this error: [paste error]. The code is supposed to compare different retrieval strategies for research PDFs. What's going wrong?"

**Why this works:** AI assistants are surprisingly good at debugging, especially when you give them context. They won't judge you for that typo in line 47.

### The Nuclear Option: Start Fresh
```bash
# Kill all Docker containers
docker kill $(docker ps -q)

# Clear Python cache (sometimes helps with import issues)
find . -type d -name __pycache__ -exec rm -rf {} +

# Start over with containers
# (Re-run the docker commands from above)
```

---

## 🎉 Victory Conditions

### You'll Know You've Won When:
- ✅ The script runs without errors
- ✅ You see a DataFrame with 6 different responses
- ✅ Phoenix UI at `http://localhost:6006` shows your traces (click on a trace to see the full execution flow)
- ✅ PostgreSQL has tables full of research document embeddings
- ✅ You feel like a wizard who just summoned 6 different search spirits

### What You've Actually Built:
- ✅ **Complete 3-stage evaluation pipeline** from infrastructure to automated metrics
- ✅ **6 retrieval strategies** implemented and ready for comparison
- ✅ **RAGAS golden test sets** for consistent, repeatable evaluation
- ✅ **Automated scoring** with QA correctness and relevance metrics
- ✅ **Phoenix observability** tracking every operation and experiment
- ✅ **Production-ready foundation** that you can extend for your use case

🔍 **Deep dive into retrieval strategies:** See [DeepWiki's technical comparison](https://deepwiki.com/donbr/rag-eval-foundations) for detailed performance analysis and architecture insights.

### Your Complete Toolkit Includes:
- **Stage 1**: Infrastructure setup and manual strategy comparison
- **Stage 2**: RAGAS-powered golden test set generation
- **Stage 3**: Automated experiments with objective metrics
- **Validation Scripts**: Interactive tools for deeper analysis
- **Phoenix Integration**: Full observability and experiment tracking

---

## 🎓 Graduate-Level Encouragement

Remember: every AI engineer has been exactly where you are right now. The difference between a beginner and an expert isn't that experts don't encounter errors—it's that they've learned to Google them more effectively.

**You're building a complete evaluation pipeline** while learning the vocabulary. Understanding how BM25 differs from semantic search, why ensemble methods matter, and what Phoenix traces tell you about retriever performance. This hands-on experience is what separates engineers who can copy-paste code from those who can architect real solutions. ([Learn more about retrieval strategies →](https://deepwiki.com/donbr/rag-eval-foundations))

You're not just running code; you're learning to **think in retrievers** with a complete toolkit at your disposal.

**What's next for you:** Now that you have the complete pipeline, you can:
- Customize evaluation metrics for your domain
- Add new retrieval strategies to compare
- Scale to larger datasets
- Integrate into CI/CD for continuous evaluation

*Now go forth and retrieve! The vectors are waiting.* 🎯

---

## 📚 Additional Resources

### 📍 Documentation Map

Navigate our comprehensive documentation based on your needs:

| Document | Purpose | Best For |
|----------|---------|----------|
| **[This README](README.md)** | Quick start, setup, troubleshooting | Getting started, practical usage |
| **[Technical Journey](docs/technical/langchain_eval_learning_journey.md)** | Detailed 3-stage progression guide | Understanding the complete pipeline |
| **[Blog Post](docs/blog/langchain_eval_foundations_e2e_blog.md)** | Deep dive into Stage 1 implementation | Learning retrieval strategies in depth |
| **[DeepWiki](https://deepwiki.com/donbr/rag-eval-foundations)** | Interactive Q&A and exploration | Quick answers, architecture insights |

### Understanding the Code
- **Main Scripts:**
  - `langchain_eval_foundations_e2e.py` - Stage 1: Foundation & infrastructure
  - `langchain_eval_golden_testset.py` - Stage 2: RAGAS golden test set generation
  - `langchain_eval_experiments.py` - Stage 3: Automated evaluation & metrics
  - `data_loader.py` - Utilities for loading data

### Architecture Diagrams
- **Location:** `diagrams/` folder contains Excalidraw source files
- **Viewing:** Use VS Code Excalidraw extension or [excalidraw.com](https://excalidraw.com/)
- **Exports:** PNG/SVG versions in `diagrams/exports/` (when available)
- **Current Status:** Work in progress - see `diagrams/README.md` for details
- **Interactive Diagrams:** View system architecture and data flow diagrams on [DeepWiki](https://deepwiki.com/donbr/rag-eval-foundations)

## 🔍 Validation & Analysis Tools

The `validation/` directory contains interactive scripts for exploring and validating the RAG system components.

### Prerequisites for Validation Scripts
```bash
# 1. Ensure services are running
docker-compose up -d

# 2. Run the main pipeline first to populate data
python claude_code_scripts/run_rag_evaluation_pipeline.py
```

### Available Validation Scripts

💡 **Tip**: These scripts provide hands-on exploration of concepts covered in the [DeepWiki technical documentation](https://deepwiki.com/donbr/rag-eval-foundations).

#### 1. PostgreSQL Data Analysis
```bash
python validation/postgres_data_analysis.py
```
**Purpose:** Comprehensive analysis of the vector database
- Analyzes document distribution and metadata
- Compares baseline vs semantic chunking strategies
- Generates PCA visualization of embeddings
- **Outputs:** Creates 3 PNG charts in `outputs/charts/postgres_analysis/`

#### 2. Phoenix Telemetry Validation  
```bash
python validation/validate_telemetry.py
```
**Purpose:** Demonstrates Phoenix OpenTelemetry tracing integration
- Tests various LLM chain patterns with tracing
- Shows streaming responses with real-time trace updates
- Validates token usage and latency tracking
- **View traces:** http://localhost:6006

#### 3. Interactive Retrieval Strategy Comparison
```bash
python validation/retrieval_strategy_comparison.py  
```
**Purpose:** Interactive comparison of all 6 retrieval strategies
- Compares naive, semantic, BM25, compression, multiquery, and ensemble strategies
- Runs performance benchmarks across strategies
- Demonstrates query-specific strategy strengths
- **Outputs:** Performance visualization in `outputs/charts/retrieval_analysis/`

### Validation Script Features
- ✅ **Phoenix Integration:** All scripts include OpenTelemetry tracing
- 📊 **Visualization:** Generates charts and performance metrics  
- 🔧 **Interactive:** Real-time comparison and analysis capabilities
- 📝 **Documentation:** Each script includes detailed output explanations

**📖 Detailed Instructions:** See [`validation/README.md`](validation/README.md) for comprehensive usage guide and troubleshooting.

### Cost Estimates
- **OpenAI:** ~$0.50-$2.00 per full run (depending on data size)
- **Cohere:** ~$0.10-$0.50 for reranking
- **Total:** Budget $5 for experimentation

### Performance Benchmarks
- Data loading: 30-60 seconds
- Embedding generation: 2-4 minutes for ~269 PDF documents
- Retrieval comparison: 30-60 seconds
- Total runtime: 3-6 minutes (varies with PDF count)

📈 **For detailed performance analysis**: Check out [DeepWiki's performance insights](https://deepwiki.com/donbr/rag-eval-foundations) including:
- Strategy-by-strategy latency comparisons
- Token usage optimization techniques
- Scaling recommendations for larger datasets
- Cost-performance trade-offs for each retrieval method

### Glossary
- **RAG**: Retrieval-Augmented Generation - enhancing LLM responses with retrieved context
- **Embeddings**: Vector representations of text for semantic search
- **BM25**: Best Matching 25 - a keyword-based ranking algorithm
- **Semantic Search**: Finding similar content based on meaning, not just keywords
- **Phoenix**: Open-source LLM observability platform by Arize
- **pgvector**: PostgreSQL extension for vector similarity search
- **RAGAS**: Framework for evaluating RAG pipelines

🔍 **Want to understand these concepts in depth?** Visit [DeepWiki](https://deepwiki.com/donbr/rag-eval-foundations) for interactive explanations and examples.

---

## 📞 Emergency Contacts
- **Docker Issues:** `docker logs container-name`
- **Python Issues:** Your friendly neighborhood AI assistant
- **Existential Crisis:** Remember, even PostgreSQL had bugs once
- **Success Stories:** Share them! The community loves a good victory lap

*P.S. If this guide helped you succeed, pay it forward by helping the next intrepid adventurer who's staring at the same error messages you just conquered.*

---

## 📚 Appendix: Useful Links
- **[uv Documentation](https://docs.astral.sh/uv/)**: Learn more about the fast Python package and project manager used in this guide.

---

## 🌐 DeepWiki: Your Interactive Documentation Portal

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/donbr/rag-eval-foundations)

**DeepWiki provides an AI-powered interface to explore this project in depth.** Ask questions, get instant answers, and discover:

- 🏗️ **System Architecture**: Interactive diagrams showing how all components connect
- 📊 **Performance Analysis**: Detailed benchmarks comparing all 6 retrieval strategies
- 🔧 **Configuration Deep Dives**: Advanced settings and optimization techniques
- 💡 **Implementation Insights**: Code explanations and design decisions
- 🚀 **Scaling Strategies**: How to adapt this foundation for production use

Perfect for when you need quick answers or want to explore specific technical aspects without diving through all the code.