from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph

from app.agent.state import State


def create_node_function(name: str, prompt: str):
    async def dynamic_node(state: State):
        response = f"[{name}] successfully processed prompt: '{prompt}'"

        return {"messages": [response]}

    return dynamic_node


async def run_dynamic_graph(
    thread_id: str, initial_message: str | None, nodes_config: list
):
    builder = StateGraph(State)

    previous_node = START

    for node in nodes_config:
        node_name = node.name

        builder.add_node(node_name, create_node_function(node_name, node.prompt))

        builder.add_edge(previous_node, node_name)

        previous_node = node_name

    DB_URI = "postgresql://postgres:postgres@localhost:5432/stateflow"

    async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
        await checkpointer.setup()

        app = builder.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}

        if initial_message:
            input_state = {"messages": [initial_message]}
        else:
            input_state = None

        final_state = await app.ainvoke(input_state, config=config)  # type: ignore
        return final_state
