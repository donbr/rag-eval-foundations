#!/usr/bin/env python3
"""
Interactive Retrieval Strategy Comparison

This script compares all 6 retrieval strategies used in our RAG evaluation pipeline.
You can test different queries and see how each strategy performs.

Output files are saved to: outputs/charts/retrieval_analysis/
- retrieval_performance.png: Performance benchmarks for all strategies
"""

import os
import asyncio
import time
from typing import List, Dict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

# Phoenix setup
from phoenix.otel import register

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGEngine, PGVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

def ensure_output_directory():
    """Ensure the output directory exists for saving charts"""
    output_dir = "outputs/charts/retrieval_analysis"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def setup_environment():
    """Setup environment and Phoenix tracing"""
    load_dotenv()
    
    # Setup Phoenix tracing
    os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://localhost:6006"
    tracer_provider = register(
        project_name="retrieval-comparison-script",
        auto_instrument=True
    )
    
    print("✅ Environment loaded and Phoenix tracing configured")
    print("📊 View traces at: http://localhost:6006")
    
    return tracer_provider

async def initialize_models_and_stores():
    """Initialize models and connect to vector stores"""
    print("\nInitializing models and database connections...")
    
    # Initialize models
    llm = ChatOpenAI(model="gpt-4.1-mini")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Database connection
    connection_string = "postgresql+asyncpg://langchain:langchain@localhost:6024/langchain"
    
    # Create engine
    pg_engine = PGEngine.from_connection_string(url=connection_string)
    
    # Connect to vector stores
    baseline_vectorstore = await PGVectorStore.create(
        engine=pg_engine,
        table_name="mixed_baseline_documents",
        embedding_service=embeddings,
    )
    
    semantic_vectorstore = await PGVectorStore.create(
        engine=pg_engine,
        table_name="mixed_semantic_documents",
        embedding_service=embeddings,
    )
    
    print("✅ Connected to vector stores")
    
    return llm, embeddings, baseline_vectorstore, semantic_vectorstore

def load_documents_for_bm25(baseline_vectorstore):
    """Load documents for BM25 retriever"""
    print("\nLoading documents for BM25...")
    
    # Retrieve documents using a broad query
    dummy_query = "financial aid"
    all_docs = baseline_vectorstore.similarity_search(dummy_query, k=100)
    
    print(f"✅ Loaded {len(all_docs)} documents for BM25")
    return all_docs

def create_retrievers(llm, baseline_vectorstore, semantic_vectorstore, all_docs):
    """Create all retrieval strategies"""
    print("\nCreating retrieval strategies...")
    
    retrievers = {}
    
    # 1. Naive Vector Search
    retrievers["naive"] = baseline_vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # 2. Semantic Chunking Vector Search
    retrievers["semantic"] = semantic_vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # 3. BM25 Retriever
    retrievers["bm25"] = BM25Retriever.from_documents(all_docs)
    retrievers["bm25"].k = 5
    
    # 4. Contextual Compression
    cohere_rerank = CohereRerank(model="rerank-english-v3.0", top_n=5)
    retrievers["compression"] = ContextualCompressionRetriever(
        base_compressor=cohere_rerank,
        base_retriever=retrievers["naive"]
    )
    
    # 5. Multi-Query Retriever
    retrievers["multiquery"] = MultiQueryRetriever.from_llm(
        retriever=retrievers["naive"],
        llm=llm
    )
    
    # 6. Ensemble Retriever
    retrievers["ensemble"] = EnsembleRetriever(
        retrievers=[
            retrievers["bm25"], 
            retrievers["naive"], 
            retrievers["compression"], 
            retrievers["multiquery"]
        ],
        weights=[0.25, 0.25, 0.25, 0.25]
    )
    
    print("✅ All retrievers created successfully")
    print(f"Available strategies: {list(retrievers.keys())}")
    
    return retrievers

async def compare_retrievers(query: str, retrievers: Dict) -> pd.DataFrame:
    """Compare all retrieval strategies for a given query"""
    results = []
    
    for name, retriever in retrievers.items():
        try:
            # Retrieve documents
            docs = await retriever.ainvoke(query)
            
            # Extract relevant information
            for i, doc in enumerate(docs[:3]):  # Top 3 results
                results.append({
                    "Strategy": name,
                    "Rank": i + 1,
                    "Content": doc.page_content[:200] + "...",
                    "Document": doc.metadata.get("document_name", "Unknown"),
                    "Source": doc.metadata.get("source_type", "N/A"),
                    "Length": len(doc.page_content)
                })
        except Exception as e:
            results.append({
                "Strategy": name,
                "Rank": 1,
                "Content": f"Error: {str(e)}",
                "Document": "Error",
                "Source": "Error",
                "Length": 0
            })
    
    return pd.DataFrame(results)

def display_results(df: pd.DataFrame):
    """Display retrieval results in a formatted way"""
    for strategy in df["Strategy"].unique():
        print(f"\n{'='*80}")
        print(f"Strategy: {strategy.upper()}")
        print('='*80)
        
        strategy_results = df[df["Strategy"] == strategy]
        for _, row in strategy_results.iterrows():
            print(f"\nRank {row['Rank']}:")
            print(f"Document: {row['Document']} | Source: {row['Source']} | Length: {row['Length']} chars")
            print(f"Content: {row['Content']}")
            print("-" * 40)

async def compare_rag_responses(query: str, retrievers: Dict, llm) -> Dict[str, str]:
    """Generate RAG responses using each retrieval strategy"""
    rag_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Use the context below to answer the question.
If you don't know the answer, say you don't know.

Context: {context}

Question: {question}

Answer:""")
    
    responses = {}
    
    for name, retriever in retrievers.items():
        try:
            # Retrieve documents
            docs = await retriever.ainvoke(query)
            
            # Combine documents into context
            context = "\n\n".join([doc.page_content for doc in docs[:5]])
            
            # Generate response
            messages = rag_prompt.format_messages(context=context, question=query)
            response = await llm.ainvoke(messages)
            
            responses[name] = response.content
        except Exception as e:
            responses[name] = f"Error: {str(e)}"
    
    return responses

async def benchmark_retrievers(query: str, retrievers: Dict, runs: int = 3) -> pd.DataFrame:
    """Benchmark retrieval strategies for speed"""
    results = []
    
    for name, retriever in retrievers.items():
        for run in range(runs):
            start = time.time()
            try:
                docs = await retriever.ainvoke(query)
                duration = time.time() - start
                num_docs = len(docs)
            except Exception as e:
                duration = time.time() - start
                num_docs = 0
            
            results.append({
                "Strategy": name,
                "Run": run + 1,
                "Duration (s)": duration,
                "Docs Retrieved": num_docs
            })
    
    return pd.DataFrame(results)

async def run_tests(retrievers, llm):
    """Run various tests on the retrievers"""
    print("\n" + "=" * 80)
    print("RUNNING RETRIEVAL COMPARISONS")
    print("=" * 80)
    
    # Test 1: General eligibility query
    print("\n1. General Eligibility Query")
    query1 = "What are the eligibility requirements for Federal Pell Grants?"
    print(f"Query: {query1}")
    results1 = await compare_retrievers(query1, retrievers)
    display_results(results1)
    
    # Test 2: Specific process query
    print("\n\n2. Specific Process Query")
    query2 = "How does the Direct Loan Program work?"
    print(f"Query: {query2}")
    results2 = await compare_retrievers(query2, retrievers)
    display_results(results2)
    
    # Test 3: RAG pipeline comparison
    print("\n\n3. RAG Pipeline Comparison")
    test_query = "What is the process for verifying financial aid applications?"
    print(f"Question: {test_query}\n")
    
    rag_responses = await compare_rag_responses(test_query, retrievers, llm)
    
    for strategy, response in rag_responses.items():
        print(f"\n{'='*80}")
        print(f"Strategy: {strategy.upper()}")
        print('='*80)
        print(response[:500] + "..." if len(response) > 500 else response)
        print(f"\n(Response length: {len(response)} characters)")
    
    # Test 4: Performance benchmark
    print("\n\n4. Performance Benchmark")
    benchmark_query = "What documents are needed for financial aid verification?"
    print(f"Benchmarking with query: {benchmark_query}")
    
    benchmark_df = await benchmark_retrievers(benchmark_query, retrievers)
    
    # Create performance visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Average duration by strategy
    avg_duration = benchmark_df.groupby('Strategy')['Duration (s)'].mean().sort_values()
    avg_duration.plot(kind='barh', ax=ax1)
    ax1.set_title('Average Retrieval Time by Strategy')
    ax1.set_xlabel('Duration (seconds)')
    
    # Box plot of durations
    benchmark_df.boxplot(column='Duration (s)', by='Strategy', ax=ax2)
    ax2.set_title('Retrieval Time Distribution')
    ax2.set_ylabel('Duration (seconds)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save to proper output directory
    output_dir = ensure_output_directory()
    output_path = os.path.join(output_dir, 'retrieval_performance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Saved performance visualization as '{output_path}'")
    
    # Summary statistics
    print("\nPerformance Summary:")
    print(benchmark_df.groupby('Strategy').agg({
        'Duration (s)': ['mean', 'std', 'min', 'max'],
        'Docs Retrieved': 'mean'
    }).round(3))

async def main():
    """Main execution function"""
    print("Interactive Retrieval Strategy Comparison")
    print("=" * 80)
    
    try:
        # Setup environment
        setup_environment()
        
        # Initialize models and stores
        llm, embeddings, baseline_vectorstore, semantic_vectorstore = await initialize_models_and_stores()
        
        # Load documents for BM25
        all_docs = load_documents_for_bm25(baseline_vectorstore)
        
        # Create retrievers
        retrievers = create_retrievers(llm, baseline_vectorstore, semantic_vectorstore, all_docs)
        
        # Run tests
        await run_tests(retrievers, llm)
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print("\nKey Observations:")
        print("1. Naive Vector Search: Good for semantic similarity in financial aid content")
        print("2. Semantic Chunking: Provides more coherent document chunks with better context")
        print("3. BM25: Excellent for specific financial aid term queries")
        print("4. Contextual Compression: Reduces noise and improves precision for aid documents")
        print("5. Multi-Query: Helps with ambiguous financial aid queries by exploring variations")
        print("6. Ensemble: Balances all approaches but may be slower")
        
        print("\n✅ All tests completed successfully!")
        print("📊 Check your traces at: http://localhost:6006")
        print("📈 Performance visualization saved in: outputs/charts/retrieval_analysis/")
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        print("\nMake sure:")
        print("1. Docker containers are running (docker-compose up -d)")
        print("2. You've run langchain_eval_foundations_e2e.py to populate the database")
        print("3. Your .env file contains valid API keys")

if __name__ == "__main__":
    asyncio.run(main())