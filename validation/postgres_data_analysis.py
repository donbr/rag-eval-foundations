#!/usr/bin/env python3
"""
PostgreSQL Data Analysis for RAG Evaluation

This script analyzes the vector database used in our RAG evaluation pipeline.
It examines table structure, data distribution, embedding analysis, and
compares baseline vs semantic chunking strategies.

Output files are saved to: outputs/charts/postgres_analysis/
- rating_distribution.png: Rating distribution by movie
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
    
    table_name = "johnwick_baseline_documents"
    df = pd.read_sql_table(table_name, engine)
    
    print(f"\nTable Info:")
    print(f"Total documents: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Parse metadata - it might already be a dict
    df['metadata'] = df['langchain_metadata'].apply(lambda x: x if isinstance(x, dict) else json.loads(x))
    df['movie_title'] = df['metadata'].apply(lambda x: x.get('Movie_Title', 'Unknown'))
    df['rating'] = df['metadata'].apply(lambda x: x.get('Rating', 0))
    df['review_title'] = df['metadata'].apply(lambda x: x.get('Review_Title', ''))
    df['author'] = df['metadata'].apply(lambda x: x.get('Author', ''))
    
    print(f"\nDocuments per movie:")
    print(df['movie_title'].value_counts())
    
    # Rating distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='rating', hue='movie_title')
    plt.title('Rating Distribution by Movie')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.legend(title='Movie', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save to proper output directory
    output_dir = ensure_output_directory()
    output_path = os.path.join(output_dir, 'rating_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Saved rating distribution plot as '{output_path}'")
    
    return df

def analyze_content(df):
    """Analyze review content"""
    print("\n" + "=" * 80)
    print("CONTENT ANALYSIS")
    print("=" * 80)
    
    df['content_length'] = df['content'].str.len()
    
    print("\nContent Length Statistics:")
    print(df['content_length'].describe())
    
    print("\n\nExample Reviews:")
    print("-" * 80)
    for idx, row in df.head(3).iterrows():
        print(f"Movie: {row['movie_title']}")
        print(f"Rating: {row['rating']}/10")
        print(f"Review Title: {row['review_title']}")
        print(f"Content Preview: {row['content'][:200]}...")
        print("-" * 80)

def compare_chunking_strategies(engine, df_baseline):
    """Compare baseline vs semantic chunking"""
    print("\n" + "=" * 80)
    print("SEMANTIC VS BASELINE CHUNKING COMPARISON")
    print("=" * 80)
    
    semantic_table_name = "johnwick_semantic_documents"
    df_semantic = pd.read_sql_table(semantic_table_name, engine)
    
    print(f"Baseline documents: {len(df_baseline)}")
    print(f"Semantic chunks: {len(df_semantic)}")
    print(f"Chunking ratio: {len(df_semantic) / len(df_baseline):.2f}x")
    
    # Parse semantic metadata
    df_semantic['metadata'] = df_semantic['langchain_metadata'].apply(lambda x: x if isinstance(x, dict) else json.loads(x))
    df_semantic['movie_title'] = df_semantic['metadata'].apply(lambda x: x.get('Movie_Title', 'Unknown'))
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
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=df_sample['rating'], cmap='viridis', 
                         s=100, alpha=0.6)
    plt.colorbar(scatter, label='Rating')
    plt.title('Review Embeddings in 2D Space (PCA)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    
    # Add movie labels
    for movie in df_sample['movie_title'].unique():
        mask = df_sample['movie_title'] == movie
        center = embeddings_2d[mask].mean(axis=0)
        plt.annotate(movie, center, fontsize=12, fontweight='bold')
    
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
    
    queries = {
        "High-rated reviews": """
            SELECT content, rating, movie_title 
            FROM (
                SELECT content, 
                       langchain_metadata::json->>'Rating' as rating,
                       langchain_metadata::json->>'Movie_Title' as movie_title
                FROM johnwick_baseline_documents
            ) t
            WHERE rating::int >= 9
            LIMIT 3
        """,
        
        "Low-rated reviews": """
            SELECT content, rating, movie_title 
            FROM (
                SELECT content, 
                       langchain_metadata::json->>'Rating' as rating,
                       langchain_metadata::json->>'Movie_Title' as movie_title
                FROM johnwick_baseline_documents
            ) t
            WHERE rating::int <= 3
            LIMIT 3
        """,
        
        "Reviews mentioning 'action'": """
            SELECT content, movie_title 
            FROM (
                SELECT content, 
                       langchain_metadata::json->>'Movie_Title' as movie_title
                FROM johnwick_baseline_documents
            ) t
            WHERE lower(content) LIKE '%action%'
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
                print(f"\nMovie: {row.get('movie_title', 'N/A')}")
                if 'rating' in row:
                    print(f"Rating: {row['rating']}")
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
        
        # Run sample queries
        run_sample_queries(engine)
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print("\nGenerated files in outputs/charts/postgres_analysis/:")
        print("- rating_distribution.png")
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