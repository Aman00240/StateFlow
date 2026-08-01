from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.agent.graph import run_dynamic_graph

app = FastAPI()


class NodeConfig(BaseModel):
    name: str
    prompt: str


class EdgeConfig(BaseModel):
    source: str
    target: str


class ConditionalEdgeConfig(BaseModel):
    source: str
    path_map: dict[str, str]


class WorkFlowRequest(BaseModel):
    thread_id: str
    initial_message: str | None = None
    nodes: list[NodeConfig]
    edges: list[EdgeConfig]
    conditional_edges: list[ConditionalEdgeConfig] = Field(default_factory=list)


@app.post("/execute")
async def execute_workflow(req: WorkFlowRequest):
    print(f"--- Building Graph for: {req.thread_id} ---")

    final_state = await run_dynamic_graph(
        thread_id=req.thread_id,
        initial_message=req.initial_message,
        nodes_config=req.nodes,
        edges_config=req.edges,
        conditional_edges_config=req.conditional_edges,
    )

    return {"status": "success", "thread_id": req.thread_id, "final_state": final_state}
