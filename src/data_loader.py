import logging
import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)


def load_docs_from_postgres(
    table_name: str = "mixed_baseline_documents",
) -> list[Document]:
    """
    Loads documents from a PostgreSQL table into a list of LangChain Documents.
    """
    load_dotenv()

    postgres_user = os.getenv("POSTGRES_USER", "langchain")
    postgres_password = os.getenv("POSTGRES_PASSWORD", "langchain")
    postgres_host = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port = os.getenv("POSTGRES_PORT", "6024")
    postgres_db = os.getenv("POSTGRES_DB", "langchain")

    sync_conn_str = (
        f"postgresql://{postgres_user}:{postgres_password}@"
        f"{postgres_host}:{postgres_port}/{postgres_db}"
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
                    metadata=row._mapping["langchain_metadata"],
                )
                documents.append(doc)
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        logger.error(
            "Please ensure the table name is correct and that the source "
            "script has run successfully."
        )
        return []

    logger.info(
        f"Successfully loaded {len(documents)} documents from the '{table_name}' table."
    )
    return documents
