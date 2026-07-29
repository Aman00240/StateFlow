from langchain_core.messages import SystemMessage

from app.agent.state import State


async def node1(state: State):

    return {"messages": ["We Shall be bllin my fellas"]}
