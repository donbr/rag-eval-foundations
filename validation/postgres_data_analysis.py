#!/usr/bin/env python3
"""
PostgreSQL Data Analysis for RAG Evaluation

This script analyzes the vector database used in our RAG evaluation pipeline.
It examines table structure, data distribution, embedding analysis, and
compares baseline vs semantic chunking strategies.

Output files are saved to: outputs/charts/postgres_analysis/
- document_distribution.png: Document distribution by type/source
- chunking_comparison.png: Baseline vs semantic chunking comparison 
- embedding_visualization.png: 2D PCA visualization of embeddings
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, text
from sklearn.decomposition import PCA
from dotenv import load_dotenv
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='pandas')

def ensure_output_directory():
    """Ensure the output directory exists for saving charts"""
    output_dir = "outputs/charts/postgres_analysis"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def setup_connection():
    """Setup database connection"""
    POSTGRES_USER = "langchain"
    POSTGRES_PASSWORD = "langchain"
    POSTGRES_HOST = "localhost"
    POSTGRES_PORT = "6024"
    POSTGRES_DB = "langchain"
    
    sync_conn_str = (
        f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@"
        f"{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )
    
    return create_engine(sync_conn_str)

def analyze_baseline_table(engine):
    """Analyze the baseline documents table"""
    print("=" * 80)
    print("ANALYZING BASELINE DOCUMENTS TABLE")
    print("=" * 80)
    
    table_name = "mixed_baseline_documents"
    df = pd.read_sql_table(table_name, engine)
    
    print(f"\nTable Info:")
    print(f"Total documents: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Parse metadata - it might already be a dict
    df['metadata'] = df['langchain_metadata'].apply(lambda x: x if isinstance(x, dict) else json.loads(x))
    df['document_name'] = df['metadata'].apply(lambda x: x.get('document_name', 'Unknown'))
    df['source_type'] = df['metadata'].apply(lambda x: x.get('source_type', 'unknown'))
    df['last_accessed'] = df['metadata'].apply(lambda x: x.get('last_accessed_at', ''))
    
    print(f"\nDocuments per source:")
    print(df['document_name'].value_counts())
    print(f"\nDocument types:")
    print(df['source_type'].value_counts())
    
    # Document distribution
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='document_name')
    plt.title('Document Distribution by Source')
    plt.xlabel('Document Name')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save to proper output directory
    output_dir = ensure_output_directory()
    output_path = os.path.join(output_dir, 'document_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Saved document distribution plot as '{output_path}'")
    
    return df

def analyze_content(df):
    """Analyze document content"""
    print("\n" + "=" * 80)
    print("CONTENT ANALYSIS")
    print("=" * 80)
    
    df['content_length'] = df['content'].str.len()
    
    print("\nContent Length Statistics:")
    print(df['content_length'].describe())
    
    print("\n\nExample Documents:")
    print("-" * 80)
    for idx, row in df.head(3).iterrows():
        print(f"Document: {row['document_name']}")
        print(f"Source Type: {row['source_type']}")
        print(f"Last Accessed: {row['last_accessed']}")
        print(f"Content Preview: {row['content'][:200]}...")
        print("-" * 80)

def compare_chunking_strategies(engine, df_baseline):
    """Compare baseline vs semantic chunking"""
    print("\n" + "=" * 80)
    print("SEMANTIC VS BASELINE CHUNKING COMPARISON")
    print("=" * 80)
    
    semantic_table_name = "mixed_semantic_documents"
    df_semantic = pd.read_sql_table(semantic_table_name, engine)
    
    print(f"Baseline documents: {len(df_baseline)}")
    print(f"Semantic chunks: {len(df_semantic)}")
    print(f"Chunking ratio: {len(df_semantic) / len(df_baseline):.2f}x")
    
    # Parse semantic metadata
    df_semantic['metadata'] = df_semantic['langchain_metadata'].apply(lambda x: x if isinstance(x, dict) else json.loads(x))
    df_semantic['document_name'] = df_semantic['metadata'].apply(lambda x: x.get('document_name', 'Unknown'))
    df_semantic['content_length'] = df_semantic['content'].str.len()
    
    # Compare content lengths
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.hist(df_baseline['content_length'], bins=30, alpha=0.7, label='Baseline')
    ax1.set_title('Baseline Document Lengths')
    ax1.set_xlabel('Character Count')
    ax1.set_ylabel('Frequency')
    
    ax2.hist(df_semantic['content_length'], bins=30, alpha=0.7, label='Semantic', color='orange')
    ax2.set_title('Semantic Chunk Lengths')
    ax2.set_xlabel('Character Count')
    ax2.set_ylabel('Frequency')
    
    plt.tight_layout()
    
    # Save to proper output directory
    output_dir = ensure_output_directory()
    output_path = os.path.join(output_dir, 'chunking_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Saved chunking comparison plot as '{output_path}'")
    
    print("\nBaseline content length stats:")
    print(df_baseline['content_length'].describe())
    print("\nSemantic chunk length stats:")
    print(df_semantic['content_length'].describe())

def analyze_embeddings(df):
    """Analyze and visualize embeddings"""
    print("\n" + "=" * 80)
    print("EMBEDDING ANALYSIS")
    print("=" * 80)
    
    def parse_embedding(embedding_str):
        clean_str = embedding_str.strip('[]')
        return np.array([float(x) for x in clean_str.split(',')])
    
    # Sample for visualization
    sample_size = min(100, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    
    # Parse embeddings
    embeddings = np.array([parse_embedding(emb) for emb in df_sample['embedding']])
    
    # PCA for visualization
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(12, 8))
    # Color by document type
    unique_docs = df_sample['document_name'].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_docs)))
    
    for i, doc in enumerate(unique_docs):
        mask = df_sample['document_name'] == doc
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                   c=[colors[i]], label=doc, s=100, alpha=0.6)
    
    plt.title('Document Embeddings in 2D Space (PCA)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add document labels at centroids
    for doc in unique_docs:
        mask = df_sample['document_name'] == doc
        if mask.sum() > 0:  # Check if there are any samples for this document
            center = embeddings_2d[mask].mean(axis=0)
            plt.annotate(doc[:15] + '...', center, fontsize=10, fontweight='bold', 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    
    plt.tight_layout()
    
    # Save to proper output directory
    output_dir = ensure_output_directory()
    output_path = os.path.join(output_dir, 'embedding_visualization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Saved embedding visualization as '{output_path}'")
    
    print(f"Total variance explained by 2 components: {sum(pca.explained_variance_ratio_):.2%}")

def run_sample_queries(engine):
    """Run sample SQL queries"""
    print("\n" + "=" * 80)
    print("SAMPLE QUERIES")
    print("=" * 80)
    
    # Simplified queries to avoid metadata parsing issues
    queries = {
        "Documents mentioning 'Pell'": """
            SELECT content
            FROM mixed_baseline_documents
            WHERE lower(content) LIKE '%pell%'
            LIMIT 3
        """,
        
        "Documents mentioning 'loan'": """
            SELECT content
            FROM mixed_baseline_documents
            WHERE lower(content) LIKE '%loan%'
            LIMIT 3
        """,
        
        "Documents mentioning 'eligibility'": """
            SELECT content
            FROM mixed_baseline_documents
            WHERE lower(content) LIKE '%eligibility%'
            LIMIT 2
        """
    }
    
    for query_name, query in queries.items():
        print(f"\n{'='*80}")
        print(f"{query_name}:")
        print('='*80)
        try:
            results = pd.read_sql_query(query, engine)
            for idx, row in results.iterrows():
                print(f"\nResult {idx + 1}:")
                print(f"Content: {row['content'][:300]}...")
                print("-"*40)
        except Exception as e:
            print(f"Error running query: {e}")

def main():
    """Main execution function"""
    print("PostgreSQL Data Analysis for RAG Evaluation")
    print("=" * 80)
    
    # Setup connection
    engine = setup_connection()
    print("✅ Connected to PostgreSQL database")
    
    try:
        # Analyze baseline table
        df_baseline = analyze_baseline_table(engine)
        
        # Analyze content
        analyze_content(df_baseline)
        
        # Compare chunking strategies
        compare_chunking_strategies(engine, df_baseline)
        
        # Analyze embeddings
        analyze_embeddings(df_baseline)
        
        # Note: Sample queries disabled due to SQLAlchemy compatibility issues with vector type
        print("\n" + "=" * 80)
        print("SAMPLE QUERIES (SKIPPED)")
        print("=" * 80)
        print("Sample queries are disabled due to SQLAlchemy compatibility issues.")
        print("The main analysis above provides comprehensive insights into the PDF documents.")
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print("\nGenerated files in outputs/charts/postgres_analysis/:")
        print("- document_distribution.png")
        print("- chunking_comparison.png")
        print("- embedding_visualization.png")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        print("Make sure:")
        print("1. PostgreSQL container is running")
        print("2. You've run langchain_eval_foundations_e2e.py to populate the database")
    
    finally:
        # Close connection
        engine.dispose()
        print("\n✅ Database connection closed")

if __name__ == "__main__":
    main()