from fastapi import FastAPI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from pydantic import BaseModel

from app.agent.graph import run_dynamic_graph

app = FastAPI()


class NodeConfig(BaseModel):
    name: str
    prompt: str


class EdgeConfig(BaseModel):
    source: str
    target: str


class WorkFlowRequest(BaseModel):
    thread_id: str
    initial_message: str | None = None
    nodes: list[NodeConfig]
    edges: list[EdgeConfig]


@app.post("/execute")
async def execute_workflow(req: WorkFlowRequest):
    print(f"--- Building Graph for: {req.thread_id} ---")

    final_state = await run_dynamic_graph(
        thread_id=req.thread_id,
        initial_message=req.initial_message,
        nodes_config=req.nodes,
        edges_config=req.edges,
    )

    return {"status": "success", "thread_id": req.thread_id, "final_state": final_state}
