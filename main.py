from fastapi import FastAPI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from pydantic import BaseModel

from app.agent.graph import run_dynamic_graph

app = FastAPI()


class NodeConfig(BaseModel):
    name: str
    prompt: str


class WorkFlowRequest(BaseModel):
    thread_id: str
    initial_message: str | None = None
    nodes: list[NodeConfig]


@app.post("/execute")
async def execute_workflow(req: WorkFlowRequest):
    print(f"--- Building Graph for: {req.thread_id} ---")

    final_state = await run_dynamic_graph(
        thread_id=req.thread_id,
        initial_message=req.initial_message,
        nodes_config=req.nodes,
    )

    return {"status": "success", "thread_id": req.thread_id, "final_state": final_state}


"""
@app.post("/execute")
async def execute_workflow(req: ExecutionRequest):
    async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
        await checkpointer.setup()

        workflow = builder.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": req.thread_id}}

        payload = {"messages": [req.message]} if req.message else None

        final_state = await workflow.ainvoke(payload, config=config)  # type: ignore

        return {
            "status": "success",
            "thread_id": req.thread_id,
            "current_state": final_state,
        }
"""
