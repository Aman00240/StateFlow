from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_postgres.vectorstores import PGVector

from app.core.config import settings

embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

raw_db_url = settings.database_url

if raw_db_url.startswith("postgresql://"):
    psycopg_url = raw_db_url.replace("postgresql://", "postgresql+psycopg://")
else:
    psycopg_url = raw_db_url

vector_store = PGVector(
    embeddings=embeddings,
    collection_name="internal_company_policies",
    connection=psycopg_url,
    use_jsonb=True,
)

_existing_docs = vector_store.similarity_search("test", k=1)
if not _existing_docs:
    print("--- Seeding Vector Database with Internal Documents ---")
    docs = [
        Document(
            page_content="Hardware Policy: Employees must return damaged laptops to the IT department on the 3rd floor. Do not go to external repair centers. Replacements take 24 hours."
        ),
        Document(
            page_content="Refund Policy: Software subscription refunds are only processed if requested within 14 days of purchase. Use the billing dashboard to initiate."
        ),
        Document(
            page_content="Remote Work: Employees may work remotely up to 3 days a week. Core hours are 10 AM to 3 PM EST."
        ),
    ]
    vector_store.add_documents(docs)


@tool
def search_internal_docs(query: str) -> str:
    """Use this tool to search internal company documents, policies, and private knowledge base articles."""
    try:
        results = vector_store.similarity_search(query, 2)
        if not results:
            return "No relevant internal documents found."

        return "\n\n".join([doc.page_content for doc in results])

    except Exception as e:
        return f"Internal search failed: {str(e)}"
