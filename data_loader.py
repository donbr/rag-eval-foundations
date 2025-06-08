import os
import logging
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, TYPE_CHECKING

from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine, text
from langchain_core.documents import Document
from langchain_community.document_loaders.csv_loader import CSVLoader

if TYPE_CHECKING:
    from langchain_eval_foundations_e2e import Config

logger = logging.getLogger(__name__)

def load_docs_from_postgres(table_name: str = "johnwick_baseline_documents") -> list[Document]:
    """
    Loads documents from a PostgreSQL table into a list of LangChain Documents.
    """
    load_dotenv()

    POSTGRES_USER = os.getenv("POSTGRES_USER", "langchain")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "langchain")
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT", "6024")
    POSTGRES_DB = os.getenv("POSTGRES_DB", "langchain")

    sync_conn_str = (
        f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@"
        f"{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )

    engine = create_engine(sync_conn_str)

    try:
        with engine.connect() as connection:
            query = text(f'SELECT content, langchain_metadata FROM "{table_name}"')
            df = pd.read_sql_query(query, connection)
    except Exception as e:
        print(f"Error executing query: {e}")
        print("Please ensure the table name is correct and that the source script has run successfully.")
        return []

    documents = [
        Document(
            page_content=row["content"],
            metadata=row["langchain_metadata"],
        )
        for _, row in df.iterrows()
    ]

    print(f"Successfully loaded {len(documents)} documents from the '{table_name}' table.")
    return documents
