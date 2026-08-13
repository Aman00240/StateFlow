from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """Searches the web for up-to-date information. Use this when you need news, facts or data you dont know"""
    search = DuckDuckGoSearchRun()

    return search.invoke(query)

