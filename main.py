import json
from contextlib import asynccontextmanager

import psycopg
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent.graph import DB_URI, get_workflow_status, run_dynamic_graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- Connecting to PostgreSQL & Verifying Tables ---")
    async with await psycopg.AsyncConnection.connect(DB_URI) as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS workflow_configs (
                thread_id TEXT PRIMARY KEY,
                config_json JSONB NOT NULL
            )
            """)
        await conn.commit()

    yield


app = FastAPI(lifespan=lifespan)


class NodeConfig(BaseModel):
    name: str
    prompt: str


class EdgeConfig(BaseModel):
    source: str
    target: str


class ConditionalEdgeConfig(BaseModel):
    source: str
    path_map: dict[str, str]


class ContinueRequest(BaseModel):
    inject_message: str | None = None


class WorkFlowRequest(BaseModel):
    thread_id: str
    initial_message: str | None = None
    nodes: list[NodeConfig]
    edges: list[EdgeConfig]
    conditional_edges: list[ConditionalEdgeConfig] = Field(default_factory=list)
    interrupt_before: list[str] = Field(default_factory=list)


@app.post("/execute")
async def execute_workflow(req: WorkFlowRequest):
    print(f"--- Building Graph for: {req.thread_id} ---")

    async with await psycopg.AsyncConnection.connect(DB_URI) as conn:
        await conn.execute(
            """
                INSERT INTO workflow_configs (thread_id,config_json)
                VALUES (%s,%s)
                ON CONFLICT (thread_id) DO UPDATE SET config_json=EXCLUDED.config_json
            """,
            (req.thread_id, req.model_dump_json()),
        )
        await conn.commit()

    return StreamingResponse(
        run_dynamic_graph(
            thread_id=req.thread_id,
            initial_message=req.initial_message,
            nodes_config=req.nodes,
            edges_config=req.edges,
            conditional_edges_config=req.conditional_edges,
            interrupt_before=req.interrupt_before,
        ),
        media_type="text/event-stream",
    )


@app.get("/status/{thread_id}")
async def get_status(thread_id: str):
    print(f"--- Fetching status for thread: {thread_id} ---")

    async with await psycopg.AsyncConnection.connect(DB_URI) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT config_json FROM workflow_configs WHERE thread_id = %s",
                (thread_id,),
            )
            row = await cur.fetchone()

    if not row:
        raise HTTPException(
            status_code=404, detail=f"Configuration for '{thread_id}' not found."
        )

    config_dict = row[0]

    if isinstance(config_dict, str):
        config_dict = json.loads(config_dict)

    req = WorkFlowRequest(**config_dict)

    state = await get_workflow_status(
        thread_id=req.thread_id,
        nodes_config=req.nodes,
        edges_config=req.edges,
        conditional_edges_config=req.conditional_edges,
        interrupt_before=req.interrupt_before,
    )

    if not state:
        raise HTTPException(
            status_code=404,
            detail=f"State for '{thread_id}' not found in LangGraph checkpointer.",
        )

    return {
        "status": "success",
        "thread_id": req.thread_id,
        "next_node": state["next_node"],
        "messages": state["messages"],
    }


@app.post("/continue/{thread_id}")
async def continue_workflow(thread_id: str, req_body: ContinueRequest | None = None):
    async with await psycopg.AsyncConnection.connect(DB_URI) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT config_json FROM workflow_configs WHERE thread_id = %s",
                (thread_id,),
            )
            row = await cur.fetchone()

    if not row:
        raise HTTPException(
            status_code=404, detail=f"Configuration for '{thread_id}' not found."
        )

    config_dict = row[0]
    if isinstance(config_dict, str):
        config_dict = json.loads(config_dict)

    req = WorkFlowRequest(**config_dict)

    injection = req_body.inject_message if req_body else None

    return StreamingResponse(
        run_dynamic_graph(
            thread_id=req.thread_id,
            initial_message=None,
            nodes_config=req.nodes,
            edges_config=req.edges,
            conditional_edges_config=req.conditional_edges,
            interrupt_before=req.interrupt_before,
            inject_message=injection,
        ),
        media_type="text/event-stream",
    )
