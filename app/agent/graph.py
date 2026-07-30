from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_groq import ChatGroq
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph

from app.agent.state import State
from app.agent.tools.web_search import AVAILABLE_TOOLS
from app.config import settings

llm = ChatGroq(api_key=settings.groq_key, model=settings.model)
llm_with_tools = llm.bind_tools(AVAILABLE_TOOLS)
MAX_ITERATIONS = 3


def create_node_function(name: str, prompt: str):
    async def dynamic_node(state: State):
        system_rules = f"""{prompt}
        *** SYSTEM EXECUTION RULES ***
        1. You have access to tools. Use them ONLY if necessary to fulfill the user's request.
        2. If you use a tool, evaluate the output. If the output contains the answer, you MUST immediately stop using tools and provide your final response to the user.
        3. NEVER call the same tool with the exact same arguments more than once. If a search fails to yield results, synthesize the best answer you can from the available data and stop.
        4. Do not apologize or explain your process. Just deliver the final output.
        """

        current_messages: list[BaseMessage] = []
        current_messages.append(SystemMessage(content=system_rules))
        current_messages.extend(state.get("messages", []))

        iterations = 0
        response = None

        while iterations < MAX_ITERATIONS:
            iterations += 1
            response = await llm_with_tools.ainvoke(current_messages)
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

        return {"messages": [formatted_output]}

    return dynamic_node


async def run_dynamic_graph(
    thread_id: str, initial_message: str | None, nodes_config: list, edges_config: list
):
    builder = StateGraph(State)

    for node in nodes_config:
        builder.add_node(node.name, create_node_function(node.name, node.prompt))

    for edge in edges_config:
        source = START if edge.source == "START" else edge.source
        target = END if edge.target == "END" else edge.target

        builder.add_edge(source, target)

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
