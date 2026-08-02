from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    AIMessage,
)
from langchain_openai import ChatOpenAI


# ==========================================================
# Agent State
# ==========================================================

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ==========================================================
# LLM
# ==========================================================

llm = ChatOpenAI(
    model="gemma4",              # Replace with your model name
    base_url="http://192.168.50.78:8666/v1",
    api_key="none",
)

# ==========================================================
# Tools
# ==========================================================

@tool
def date_and_time():
    '''this tool will calculate current date and time'''
    from datetime import datetime
    current_dt = datetime.now()
    return current_dt  # Output: 2026-08-02 12:00:00.000000

search_tool = DuckDuckGoSearchRun()

tools = [search_tool,date_and_time]

llm_with_tools = llm.bind_tools(tools)

tool_node = ToolNode(tools)


# ==========================================================
# Chat Node
# ==========================================================

def chat_node(state: AgentState):

    system_prompt = SystemMessage(
        content=(
            "You are a research assistant. But you like to answer in rhyming couplets and rhyming iambic pentameter "
            "Answer the user's question. "
            "Use the search tool whenever current or factual information is needed."
        )
    )

    response = llm_with_tools.invoke(
        [system_prompt] + state["messages"]
    )

    return {
        "messages": [response]
    }


# ==========================================================
# Build Graph
# ==========================================================

builder = StateGraph(AgentState)

builder.add_node("chat", chat_node)
builder.add_node("tools", tool_node)

builder.add_edge(START, "chat")

builder.add_conditional_edges(
    "chat",
    tools_condition,
)

builder.add_edge("tools", "chat")

graph = builder.compile()


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

     
    user_query = input("your question-->")

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content=user_query
                )
            ]
        }
    )

    print("\nConversation\n")

    for message in result["messages"]:

        print("=" * 70)
        print(type(message).__name__)
        print("=" * 70)

        if isinstance(message, AIMessage):
            print("Content:")
            print(message.content)

            if message.tool_calls:
                print("\nTool Calls:")
                print(message.tool_calls)

        else:
            print(message.content)

        print()