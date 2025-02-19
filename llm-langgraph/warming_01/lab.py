import openai
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatMaritalk
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
import os

load_dotenv("../credentials.env")

def llm_maritaca(state: MessagesState):
    model = ChatMaritalk(
        model="sabia-3",
        api_key = os.getenv("MARITACA_KEY"),
    )
    messages = state["messages"]
    response = model.invoke(messages[-1].content)
    return {
        "messages": [response]
    }


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# LANGGRAPH
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
workflow = StateGraph(MessagesState)

workflow.add_node("llm_maritaca", llm_maritaca)

workflow.add_edge(START, "llm_maritaca")
workflow.add_edge("llm_maritaca", END)

app = workflow.compile()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

input_messages = {
    "messages": [
        ("human", "Hello, what is your name?")
    ]
}

for chunk in app.stream(input_messages, stream_mode="values"):
    chunk["messages"][-1].pretty_print()