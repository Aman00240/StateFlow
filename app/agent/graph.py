from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph

from app.agent.state import State
from app.agent.tools.web_search import AVAILABLE_TOOLS
from app.config import settings

llm = ChatGroq(api_key=settings.groq_key, model=settings.model)
llm_with_tools = llm.bind_tools(AVAILABLE_TOOLS)
MAX_ITERATIONS = 3
DB_URI = "postgresql://postgres:postgres@localhost:5432/stateflow"


def create_node_function(name: str, prompt: str, route_keys: list[str] | None = None):
    async def dynamic_node(state: State):
        system_rules = f"""{prompt}
        SYSTEM EXECUTION RULES:
        1. You have access to tools. Use them ONLY if necessary to fulfill the user's request.
        2. If you use a tool, evaluate the output. If the output contains the answer, you MUST immediately stop using tools and provide your final response to the user.
        3. NEVER call the same tool with the exact same arguments more than once. If a search fails to yield results, synthesize the best answer you can from the available data and stop.
        4. Do not apologize or explain your process. Just deliver the final output.
        """
        if route_keys:
            options_str = ", ".join(route_keys)
            system_rules += f"4. IMPORTANT: To finish your task, you MUST use the 'route_workflow' tool. The destination argument MUST be exactly one of these: {options_str}"

        current_messages: list[BaseMessage] = []
        current_messages.append(SystemMessage(content=system_rules))

        for msg in state.get("messages", []):
            current_messages.append(msg)

            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc["name"] == "route_workflow":
                        current_messages.append(
                            ToolMessage(
                                content="Route successful.", tool_call_id=tc["id"]
                            )
                        )

        node_tools = AVAILABLE_TOOLS.copy()

        if route_keys:

            @tool
            def route_workflow(destination: str):
                """Use this tool to route the workflow to the next step based on your analysis."""
                pass

            node_tools.append(route_workflow)

        node_llm = llm.bind_tools(node_tools)

        iterations = 0
        response = None

        while iterations < MAX_ITERATIONS:
            iterations += 1
            response = await node_llm.ainvoke(current_messages)

            if response.tool_calls:
                for tc in response.tool_calls:
                    if tc["name"] == "route_workflow":
                        print(
                            f"-- Agent [{name}] selected route tool: {tc['args'].get('destination')} --- "
                        )
                        return {"messages": [response]}

            current_messages.append(response)

            if not response.tool_calls:
                break

            for tool_call in response.tool_calls:
                print(f"--- Agent [{name}] is executing tool: {tool_call['name']} ---")

                selected_tool = next(
                    t for t in AVAILABLE_TOOLS if t.name == tool_call["name"]
                )
                tool_result = selected_tool.invoke(tool_call["args"])

                current_messages.append(
                    ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"])
                )

        if response is None or (response.tool_calls and iterations >= MAX_ITERATIONS):
            if iterations >= MAX_ITERATIONS:
                print(
                    f"--- Agent [{name}] hit max iterations limit. Forcing final response. ---"
                )

                current_messages.append(
                    HumanMessage(
                        content="System Override: You have reached the maximum allowed searches. You MUST now provide a final plain text answer using ONLY the information gathered so far. Do NOT output JSON or call any tools."
                    )
                )
            response = await llm.ainvoke(current_messages)

        formatted_output = f"[{name}]: {response.content}"

        return {"messages": [AIMessage(formatted_output)]}

    return dynamic_node


def create_routing_function(path_map: dict[str, str]):
    def routing_function(state: State):
        messages = state.get("messages", [])
        if not messages:
            return "fallback"

        last_message = messages[-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            for tc in last_message.tool_calls:
                if tc["name"] == "route_workflow":
                    destination = tc["args"].get("destination", "").lower()
                    if destination in [k.lower() for k in path_map.keys()]:
                        matched_key = next(
                            k for k in path_map.keys() if k.lower() == destination
                        )
                        print(
                            f"--- JSON Routing triggered. Destination: {path_map[matched_key]} ---"
                        )
                        return matched_key

        if hasattr(last_message, "content") and last_message.content:
            last_text = last_message.content.strip().lower()
            for key in path_map.keys():
                if key.lower() in last_text:
                    print(
                        f"--- Text Routing triggered. Destination: {path_map[key]} ---"
                    )
                    return key

        print(
            f"--- Routing failed: No keywords matched in '{last_message}'. Forcing END. ---"
        )
        return "fallback"

    return routing_function


async def get_workflow_status(
    thread_id: str,
    nodes_config: list,
    edges_config: list,
    conditional_edges_config: list,
    interrupt_before: list[str] = [],
):
    builder = build_graph_blueprint(
        nodes_config, edges_config, conditional_edges_config
    )

    async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
        app = builder.compile(
            checkpointer=checkpointer, interrupt_before=interrupt_before
        )
        config = {"configurable": {"thread_id": thread_id}}

        state_snapshot = await app.aget_state(config)  # type: ignore

        if not state_snapshot:
            return None

        return {
            "next_node": state_snapshot.next,
            "messages": state_snapshot.values.get("messages", []),
        }


def build_graph_blueprint(
    nodes_config,
    edges_config,
    conditional_edges_config,
):
    builder = StateGraph(State)

    routing_map = {}
    for c_edge in conditional_edges_config:
        routing_map[c_edge.source] = list(c_edge.path_map.keys())

    for node in nodes_config:
        route_keys = routing_map.get(node.name)
        builder.add_node(
            node.name, create_node_function(node.name, node.prompt, route_keys)
        )

    for edge in edges_config:
        source = START if edge.source == "START" else edge.source
        target = END if edge.target == "END" else edge.target
        builder.add_edge(source, target)

    for c_edge in conditional_edges_config:
        mapped_path = {
            k: (END if v == "END" else v) for k, v in c_edge.path_map.items()
        }
        mapped_path["fallback"] = END
        builder.add_conditional_edges(
            c_edge.source, create_routing_function(c_edge.path_map), mapped_path
        )

    return builder


async def run_dynamic_graph(
    thread_id: str,
    initial_message: str | None,
    nodes_config: list,
    edges_config: list,
    conditional_edges_config: list,
    interrupt_before: list[str] = [],
):

    builder = build_graph_blueprint(
        nodes_config, edges_config, conditional_edges_config
    )

    async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
        await checkpointer.setup()

        app = builder.compile(
            checkpointer=checkpointer, interrupt_before=interrupt_before
        )
        config = {"configurable": {"thread_id": thread_id}}

        if initial_message:
            input_state = {"messages": [initial_message]}
        else:
            input_state = None

        final_state = await app.ainvoke(input_state, config=config)  # type: ignore
        return final_state
