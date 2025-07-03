import os
import logging
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from langchain_core.documents import Document
from typing import List

logger = logging.getLogger(__name__)

def load_docs_from_postgres(table_name: str = "mixed_baseline_documents") -> List[Document]:
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
    documents = []

    try:
        with engine.connect() as connection:
            query = text(f'SELECT content, langchain_metadata FROM "{table_name}"')
            result = connection.execute(query)
            for row in result:
                doc = Document(
                    page_content=row._mapping["content"],
                    metadata=row._mapping["langchain_metadata"]
                )
                documents.append(doc)
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        logger.error("Please ensure the table name is correct and that the source script has run successfully.")
        return []

    logger.info(f"Successfully loaded {len(documents)} documents from the '{table_name}' table.")
    return documents