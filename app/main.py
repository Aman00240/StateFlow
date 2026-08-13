from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1.routers import router as workflow_router
from app.db.session import verify_db_tables


@asynccontextmanager
async def lifespan(app: FastAPI):
    await verify_db_tables()
    yield


app = FastAPI(lifespan=lifespan)

app.include_router(workflow_router)
