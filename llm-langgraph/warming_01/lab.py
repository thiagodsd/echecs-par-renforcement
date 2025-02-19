import openai
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatMaritalk
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

#
import os
import subprocess
import sys
from langchain_core.runnables.graph import MermaidDrawMethod, CurveStyle
import random

load_dotenv("../credentials.env")


def llm_maritaca(state: MessagesState):
    model = ChatMaritalk(
        model="sabia-3",
        api_key=os.getenv("MARITACA_KEY"),
    )
    messages = state["messages"]
    response = model.invoke(messages[-1].content)
    return {"messages": [response]}


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# BEGIN LANGGRAPH
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
workflow = StateGraph(MessagesState)

workflow.add_node("llm_maritaca", llm_maritaca)

workflow.add_edge(START, "llm_maritaca")
workflow.add_edge("llm_maritaca", END)

app = workflow.compile()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# END LANGGRAPH
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

input_messages = {"messages": [("human", "Hello, what is your name?")]}

for chunk in app.stream(input_messages, stream_mode="values"):
    chunk["messages"][-1].pretty_print()


def display_graph(graph, output_folder="output", file_name="graph"):
    mermaid_png = graph.get_graph(xray=1).draw_mermaid_png(
        draw_method=MermaidDrawMethod.API, curve_style=CurveStyle.NATURAL
    )
    output_folder = "./output"
    os.makedirs(output_folder, exist_ok=True)
    filename = os.path.join(output_folder, f"{file_name}_{random.randint(1, 100000)}.png")
    with open(filename, 'wb') as f:
        f.write(mermaid_png)
    if sys.platform.startswith('darwin'):
        subprocess.call(('open', filename))
    elif sys.platform.startswith('linux'):
        subprocess.call(('xdg-open', filename))
    elif sys.platform.startswith('win'):
        os.startfile(filename)

display_graph(app)