#!/usr/bin/env python3
"""
Phoenix Telemetry Validation and LLM Observability

This script demonstrates how to use Phoenix for LLM observability and tracing.
It includes examples of simple chains, RAG components, error handling, and streaming.
"""

import os

from dotenv import load_dotenv

# Set Phoenix endpoint before importing
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://localhost:4317"

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from phoenix.otel import register


def setup_phoenix():
    """Configure Phoenix tracing"""
    print("Setting up Phoenix tracing...")
    tracer_provider = register(
        project_name="telemetry-validation",
        auto_instrument=True
    )
    print("✅ Phoenix tracing configured")
    print("📊 View traces at: http://localhost:6006")
    return tracer_provider

def test_simple_chain():
    """Test simple chain with Phoenix tracing"""
    print("\n" + "=" * 80)
    print("SIMPLE CHAIN EXAMPLE")
    print("=" * 80)

    # Create a simple prompt and chain
    prompt = ChatPromptTemplate.from_template("{x} {y} {z}?").partial(x="why is", z="blue")
    chain = prompt | ChatOpenAI(model_name="gpt-4.1-mini")

    result = chain.invoke(dict(y="sky"))
    print("Question: why is sky blue?")
    print(f"Answer: {result.content[:200]}...")

    return result

def test_complex_chains():
    """Test more complex chains with custom span names"""
    print("\n" + "=" * 80)
    print("COMPLEX CHAIN EXAMPLES")
    print("=" * 80)

    # Example 1: Math calculator
    print("\n1. Math Calculator Chain:")
    math_chain = (
        ChatPromptTemplate.from_template("What is {x} + {y}?")
        | ChatOpenAI(model_name="gpt-4.1-mini")
        | StrOutputParser()
    ).with_config({"run_name": "math_calculator"})

    result = math_chain.invoke({"x": 15, "y": 27})
    print(f"Result: {result}")

    # Example 2: Text analyzer
    print("\n2. Text Analyzer Chain:")
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that analyzes text."),
        ("user", "Analyze this text and provide key insights: {text}")
    ])

    analysis_chain = (
        {"text": RunnablePassthrough()}
        | analysis_prompt
        | ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)
        | StrOutputParser()
    ).with_config({"run_name": "text_analyzer"})

    sample_text = "John Wick is an action movie franchise known for its choreographed fight scenes."
    analysis = analysis_chain.invoke(sample_text)
    print(f"Analysis: {analysis[:200]}...")

def test_rag_components():
    """Test RAG pipeline components"""
    print("\n" + "=" * 80)
    print("RAG PIPELINE SIMULATION")
    print("=" * 80)

    # Example 3: Embedding generation
    print("\n3. Embedding Generation:")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    sample_docs = [
        "John Wick is a retired assassin who comes back for revenge.",
        "The action sequences in John Wick are beautifully choreographed.",
        "Keanu Reeves delivers an excellent performance as John Wick."
    ]

    print("Generating embeddings...")
    doc_embeddings = embeddings.embed_documents(sample_docs)
    print(f"✅ Generated {len(doc_embeddings)} embeddings, each with {len(doc_embeddings[0])} dimensions")

    # Example 4: Simulated RAG chain
    print("\n4. Mock RAG Pipeline:")
    rag_prompt = ChatPromptTemplate.from_template("""
Use the following context to answer the question.

Context: {context}

Question: {question}

Answer:""")

    def mock_retriever(question):
        """Mock retriever function"""
        return "\n".join(sample_docs)

    rag_chain = (
        {
            "context": lambda x: mock_retriever(x["question"]),
            "question": lambda x: x["question"]
        }
        | rag_prompt
        | ChatOpenAI(model_name="gpt-4.1-mini")
        | StrOutputParser()
    ).with_config({
        "run_name": "rag_pipeline",
        "metadata": {"pipeline_type": "mock_rag"}
    })

    question = "What is John Wick about?"
    answer = rag_chain.invoke({"question": question})
    print(f"Question: {question}")
    print(f"Answer: {answer[:300]}...")

def test_error_handling():
    """Test error handling and debugging with Phoenix"""
    print("\n" + "=" * 80)
    print("ERROR HANDLING EXAMPLES")
    print("=" * 80)

    # Example 5: Division calculator with edge cases
    print("\n5. Division Calculator (with error cases):")
    risky_prompt = ChatPromptTemplate.from_template(
        "Calculate the result of {number} divided by {divisor}"
    )

    risky_chain = (
        risky_prompt
        | ChatOpenAI(model_name="gpt-4.1-mini")
        | StrOutputParser()
    ).with_config({"run_name": "division_calculator"})

    test_cases = [
        {"number": 100, "divisor": 5},
        {"number": 42, "divisor": 0},
        {"number": 7, "divisor": 3}
    ]

    for test in test_cases:
        try:
            result = risky_chain.invoke(test)
            print(f"{test['number']} ÷ {test['divisor']} = {result}")
        except Exception as e:
            print(f"Error with {test}: {e}")

def test_streaming():
    """Test streaming responses with Phoenix"""
    print("\n" + "=" * 80)
    print("STREAMING EXAMPLE")
    print("=" * 80)
    print("Watch the traces update in real-time at http://localhost:6006")
    print("-" * 80)

    streaming_chain = (
        ChatPromptTemplate.from_template("Tell me 3 facts about {topic}")
        | ChatOpenAI(model_name="gpt-4.1-mini", streaming=True)
    ).with_config({"run_name": "streaming_facts"})

    print("Streaming response:")
    for chunk in streaming_chain.stream({"topic": "Phoenix observability"}):
        print(chunk.content, end="", flush=True)
    print("\n")

def print_analysis_tips():
    """Print tips for analyzing traces in Phoenix"""
    print("\n" + "=" * 80)
    print("ANALYZING TRACES IN PHOENIX")
    print("=" * 80)
    print("""
After running these examples, go to http://localhost:6006 to explore:

1. **Traces View**: See all the traces from our examples
2. **Span Details**: Click on any trace to see the detailed breakdown
3. **Latency Analysis**: Identify which operations take the longest
4. **Token Usage**: Monitor how many tokens each operation uses
5. **Error Tracking**: Find failed operations and their error messages

Key Things to Look For:
- Trace Hierarchy: How operations are nested (prompts → LLM calls → parsing)
- Timing Information: Which steps are slowest
- Input/Output: Exact prompts sent and responses received
- Metadata: Custom run names and tags we added

Pro Tips:
1. Use descriptive run_name values to make traces easier to find
2. Add metadata to categorize different types of operations
3. Filter traces by time range or metadata in the Phoenix UI
4. Export traces for further analysis or sharing with your team
""")

def main():
    """Main execution function"""
    print("Phoenix Telemetry Validation and LLM Observability Demo")
    print("=" * 80)

    # Load environment variables
    load_dotenv()

    # Setup Phoenix
    setup_phoenix()

    try:
        # Run all test examples
        test_simple_chain()
        test_complex_chains()
        test_rag_components()
        test_error_handling()
        test_streaming()

        # Print analysis tips
        print_analysis_tips()

        print("\n✅ All examples completed successfully!")
        print("📊 Check your traces at: http://localhost:6006")

    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        print("\nMake sure:")
        print("1. Phoenix container is running (docker-compose up -d)")
        print("2. Your .env file contains valid API keys")
        print("3. Port 6006 is not blocked by firewall")

if __name__ == "__main__":
    main()
