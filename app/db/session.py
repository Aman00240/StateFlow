import psycopg
from app.core.config import settings

DB_URI = settings.database_url


async def verify_db_tables():
    print("--- Connecting to PostgreSQL & Verifying Tables ---")
    async with await psycopg.AsyncConnection.connect(DB_URI) as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS workflow_configs (
                thread_id TEXT PRIMARY KEY,
                config_json JSONB NOT NULL
            )
            """)
        await conn.commit()
