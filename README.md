# 🚀 RAG Evaluation Foundations: Survival Guide for Intrepid Adventurers

*Welcome, brave soul, to the wild world of Retrieval-Augmented Generation evaluation! You're about to embark on a journey that's part software engineering, part data science, and part digital archaeology. Don't worry—we've got your back.*

For a deep-dive into the technical concepts and a complete walkthrough of the code, see the [full technical guide](langchain_eval_foundations_e2e.md).

## 🎯 What You're Getting Into

**This is NOT the final evaluation pipeline yet!** Think of this as "RAG Evaluation Bootcamp" - you're learning to crawl before you sprint.

You're building a comparison engine for 6 different ways to find relevant information, then asking an AI to answer questions based on what it finds. Think of it as speed-dating for search algorithms, with John Wick movie reviews as your wingman.

**The real magic happens later** when we add RAGAS for golden test sets and structured evaluation metrics. Right now, we're just getting comfortable with LangChain retrievers and observability tooling.

---

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
```

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

---

## 🚦 The Moment of Truth

### Launch Sequence
```bash
python langchain_eval_foundations_e2e.py
```

**What should happen:**
1. 📥 Downloads John Wick review data (because who doesn't love Keanu?)
2. 🗄️ Creates fancy vector tables in PostgreSQL
3. 🔍 Tests 6 different ways to search for information
4. 📊 Shows you which method gives the best answers
5. 🕵️ Sends traces to Phoenix at `http://localhost:6006`

**Expected runtime:** 2-5 minutes (perfect time to practice your John Wick pencil tricks)

---

## 🚨 When Things Go Sideways (And They Will)

### The Greatest Hits of Failure

**"ModuleNotFoundError: No module named 'whatever'"**
- *Translation:* You forgot to install something
- *Solution:* Re-run that massive pip install command
- *Encouragement:* Even senior developers copy-paste pip commands wrong

**"Connection refused" or "Port already in use"**
- *Translation:* Docker containers aren't happy
- *Solution:* `docker ps` to check what's running, `docker kill` the troublemakers
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
> "I'm running this RAG evaluation foundations setup with LangChain, PostgreSQL, and Phoenix. I'm getting this error: [paste error]. The code is supposed to compare different retrieval strategies for John Wick movie reviews. What's going wrong?"

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
- ✅ Phoenix UI at `http://localhost:6006` shows your traces
- ✅ PostgreSQL has tables full of John Wick wisdom
- ✅ You feel like a wizard who just summoned 6 different search spirits

### What You've Actually Accomplished:
- Built the **foundation** for a production-ready RAG evaluation system
- Learned how different retrieval strategies behave (the real education!)
- Compared naive vector search against ensemble methods *by feel*
- Set up observability that shows you what's happening under the hood
- **Most importantly:** Got comfortable with LangChain's retriever ecosystem

### What's Coming Next (The Real Fun):
- **Golden test sets** with RAGAS (ground truth questions and answers)
- **Structured evaluation metrics** (precision, recall, hallucination detection)
- **Automated scoring** that doesn't require human eyeballs
- **Production-ready pipelines** that run evaluation on every code change

---

## 🎓 Graduate-Level Encouragement

Remember: every AI engineer has been exactly where you are right now. The difference between a beginner and an expert isn't that experts don't encounter errors—it's that they've learned to Google them more effectively.

**You're not building the final pipeline yet** - you're learning the vocabulary. Understanding how BM25 differs from semantic search, why ensemble methods matter, and what Phoenix traces tell you about retriever performance. This foundational knowledge is what separates engineers who can copy-paste code from those who can architect real solutions.

You're not just running code; you're learning to **think in retrievers**. John Wick would be proud.

**Next mission:** Once this foundation is solid, you'll be ready for the real magic - generating golden test sets with RAGAS and building automated evaluation foundations that tell you *objectively* which retrieval strategy wins.

*Now go forth and retrieve! The vectors are waiting.* 🎯

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