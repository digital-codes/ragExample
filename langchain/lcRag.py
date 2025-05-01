from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_core.tools import tool


import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../rag')))
import private_remote as pr 

from langchain.chat_models import init_chat_model
os.environ["OPENAI_API_KEY"] = pr.openAi["apiKey"]

# chat providers limited. openAi work. deepinfra not supported here. huggingface complicated or not working
llm = init_chat_model("gpt-4o-mini", model_provider="openai")

from langchain_community.embeddings import DeepInfraEmbeddings
embMdl = "BAAI/bge-m3"
embedder = DeepInfraEmbeddings(model_id=embMdl, deepinfra_api_token=pr.deepInfra["apiKey"])

from langchain_community.vectorstores import FAISS
# Load FAISS vector store
vector_store = FAISS.load_local(
    "test", embedder, allow_dangerous_deserialization=True
)



@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    try:
        llm_with_tools = llm.bind_tools([retrieve])
        response = llm_with_tools.invoke(state["messages"])
    except:
        response = "LLM Failed"
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}


# Step 2: Execute the retrieval.
tools = ToolNode([retrieve])


# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = llm.invoke(prompt)
    return {"messages": [response]}


# Build graph
graph_builder = StateGraph(MessagesState)

graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)


with open("graph1.png","wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())

print("Graph image written: graph1.png")

# Specify an ID for the thread
config = {"configurable": {"thread_id": "abc123"}}

###### 
input_message = "Hello"

for step in graph.stream(
    {"messages": [{"role": "user", "content": input_message}]},
    stream_mode="values",
    config=config,
):
    step["messages"][-1].pretty_print()

######

input_message = "What is main feature of docling?"

for step in graph.stream(
    {"messages": [{"role": "user", "content": input_message}]},
    stream_mode="values",
    config=config,
):
    step["messages"][-1].pretty_print()


config["configurable"]["thread_id"] = "abc1234"

######
input_message = "Can docling do ocr?"

for step in graph.stream(
    {"messages": [{"role": "user", "content": input_message}]},
    stream_mode="values",
    config=config,
):
    step["messages"][-1].pretty_print()

########

input_message = "Can docling convert tabular data?"

for step in graph.stream(
    {"messages": [{"role": "user", "content": input_message}]},
    stream_mode="values",
    config=config,
):
    step["messages"][-1].pretty_print()

########

chat_history = graph.get_state(config).values["messages"]
chat_history = graph.get_state(config).values["messages"]
with open("chat_history.txt", "w") as f:
    f.writelines([message.content + "\n\n" for message in chat_history])

print("Chat history written: chat_history.txt")
