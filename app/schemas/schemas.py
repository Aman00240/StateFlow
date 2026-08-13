from pydantic import BaseModel, Field


class NodeConfig(BaseModel):
    name: str
    prompt: str
    tools: list[str] = []


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
