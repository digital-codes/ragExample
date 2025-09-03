import sys
import os
import re
import json
import argparse


os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../rag")))
except NameError:
    sys.path.append(os.path.abspath(os.path.join("../rag")))
import private_remote as pr

from langchain_core.messages import SystemMessage, AIMessage, BaseMessage, ToolCall
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool

from typing import List, Optional

# Add argument parsing for provider
parser = argparse.ArgumentParser(description="Run the RAG example with a specified provider.")
parser.add_argument(
    "--provider","-p",
    type=str,
    choices=["ollama", "openai", "local", "deepinfra"],
    default="ollama",
    help="Specify the provider to use (default: ollama).",
)
parser.add_argument(
    "--model","-m",
    type=str,
    default="granite3.3:2b",
    help="Specify the model to use (default: granite3.3:2b).",
)
parser.add_argument(
    "--debug","-d",
    action="store_true",
    help="Enable debug mode (default: False).",
)
parser.add_argument(
    "--baseurl","-b",
    type=str,
    default="http://localhost:8085/",
    help="Base URL for the model provider (default: http://localhost:8085/).",
)
args = parser.parse_args()
PROVIDER = args.provider

#PROVIDER = "ollama" # "openai"  # "ollama"   or "openai" "local" #

DEBUG = args.debug

if "granite3" in args.model:
    granitePatch = True
else:
    granitePatch = False

# maybe check https://python.langchain.com/docs/how_to/qa_chat_history_how_to/


if PROVIDER == "deepinfra":
    from langgraph.prebuilt import create_react_agent
    from langchain_community.chat_models import ChatDeepInfra

    # Setup DeepInfra LLM
    llm = ChatDeepInfra(
        model=args.model,
        deepinfra_api_token=pr.deepInfra["apiKey"],
    )
    llm_with_tools = None
elif PROVIDER == "local":
    import localChatModel as LC

    llm = LC.ChatLocal(
        model=args.model
    )  # parrot_buffer_length=3, model="my_custom_model")
    print("Using local model", llm)
    llm_with_tools = None
elif PROVIDER == "ollama":
    from langchain.chat_models import init_chat_model
    # from langchain_community.chat_models import ChatLlamaCpp

    # https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html
    # chat providers limited. openAi work. deepinfra not supported here. huggingface complicated or not working
    # if ollama runs on remote machine, use ssh tunnel like:
    # ssh -L 8085:localhost:11434 user@remote
    # seems like granite3.3 tool request not always recognized by langgraph... message like:
    # <|tool_call|>[{"arguments": {"query": "main feature of docling"}}]
    # appears in output
    # llama3.2  works normally
    # on pycontabo. small deeepseek and phi don't support tools
    os.environ["OPENAI_API_KEY"] = "ollama"
    llm = init_chat_model(
        args.model,
        model_provider="ollama",
        base_url=args.baseurl,
        temperature=0.1,
        max_tokens=10000,
    )

else:
    from langchain.chat_models import init_chat_model

    os.environ["OPENAI_API_KEY"] = pr.openAi["apiKey"]
    # chat providers limited. openAi work. deepinfra not supported here. huggingface complicated or not working
    llm = init_chat_model("gpt-4o-mini", model_provider="openai")

from langchain_community.embeddings import DeepInfraEmbeddings

embMdl = "BAAI/bge-m3"
embedder = DeepInfraEmbeddings(
    model_id=embMdl, deepinfra_api_token=pr.deepInfra["apiKey"]
)

from langchain_community.vectorstores import FAISS

# Load FAISS vector store
vector_store = FAISS.load_local("test", embedder, allow_dangerous_deserialization=True)


@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    if DEBUG:
        print(f"Retrieving for query: {query}")
    retrieved_docs = vector_store.similarity_search(query, k=5, threshold=0.5)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    if DEBUG:
        print(f"Search returned {len(retrieved_docs)} results")
    return serialized, retrieved_docs

# granite format patch
def patch_granite_tool_output(msg: BaseMessage) -> BaseMessage:
    """Convert <|tool_call|> response to proper tool call structure for LangChain."""
    if isinstance(msg, AIMessage) and isinstance(msg.content, str):
        match = re.search(r"<\|tool_call\|>(\[.*?\])", msg.content)
        if match:
            try:
                raw_tool_calls = json.loads(match.group(1))
                tool_calls = [
                    ToolCall(
                        name="retrieve",  # <-- use your actual tool name
                        args=tc["arguments"],
                        id=f"granite-tool-call-{i}"
                    )
                    for i, tc in enumerate(raw_tool_calls)
                ]
                if len(tool_calls) == 0:
                    return ""
                #return AIMessage(content="Calling tool...", tool_calls=tool_calls)
                return AIMessage(content="", tool_calls=tool_calls)
            except Exception as e:
                print("Failed to parse tool call:", e)
    return msg

# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    try:
        llm_with_tools = llm.bind_tools([retrieve])
        response = llm_with_tools.invoke(state["messages"])
        if granitePatch:
            if DEBUG:
                print("Patching Granite tool output")
            response = patch_granite_tool_output(response)
    except Exception as e:
        print(f"Error invoking LLM: {e}")
        # Handle the error gracefully
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
            # check artifacts
            if message.artifact:
                # print("Artifacts:", message.artifact)
                doc_ids = [doc.metadata["doc_id"] for doc in message.artifact]
                print("Docs:", doc_ids)
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

# graph_builder = StateGraph(MessagesState)
graph_builder = StateGraph(MessagesState)

graph_builder.add_node("query_or_respond", query_or_respond)
graph_builder.add_node("tools", tools)
graph_builder.add_node("generate", generate)

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


# with open("graph1.png","wb") as f:
#    f.write(graph.get_graph().draw_mermaid_png())
# print("Graph image written: graph1.png")
# print(graph.get_graph())

# Specify an ID for the thread
config = {"configurable": {"thread_id": "abc123"}}

######
input_message = "Hello"

for step in graph.stream(
    {"messages": [{"role": "user", "content": input_message}]},
    stream_mode="values",
    config=config,
):
    if DEBUG:
        step["messages"][-1].pretty_print()
    if "tool" in step["messages"][-1].type:
        print("Tool call detected:", step["messages"][-1].tool_calls)
    if step["messages"][-1].type == "ai":  # and not message.tool_calls:
        print("AI:",step["messages"][-1].content)
######

input_message = "What is main feature of ibm docling Document Conversion?"

for step in graph.stream(
    {"messages": [{"role": "user", "content": input_message}]},
    stream_mode="values",
    config=config,
):
    if DEBUG:
        step["messages"][-1].pretty_print()
    last_message = step["messages"][-1]
    if "tool" in last_message.type:
        if isinstance(last_message, AIMessage):
            print("Tool call detected:", last_message.tool_calls)
        elif hasattr(last_message, "tool_call_id"):
            print("Tool returned result for call ID:", last_message.tool_call_id)
            print("Tool result content:", last_message.content)
    if last_message.type == "ai":  # and not message.tool_calls:
        print("AI:",step["messages"][-1].content)


# config["configurable"]["thread_id"] = "abc1234"

######
input_message = "Can docling do ocr?"

for step in graph.stream(
    {"messages": [{"role": "user", "content": input_message}]},
    stream_mode="values",
    config=config,
):
    if DEBUG:
        step["messages"][-1].pretty_print()
    if step["messages"][-1].type == "ai":  # and not message.tool_calls:
        print("AI:",step["messages"][-1].content)

########

input_message = "Can docling convert tabular data?"

for step in graph.stream(
    {"messages": [{"role": "user", "content": input_message}]},
    stream_mode="values",
    config=config,
):
    if DEBUG:
        step["messages"][-1].pretty_print()
    if step["messages"][-1].type == "ai":  # and not message.tool_calls:
        print("AI:",step["messages"][-1].content)

########

input_message = "how does it compare to tabulate or tika?"

for step in graph.stream(
    {"messages": [{"role": "user", "content": input_message}]},
    stream_mode="values",
    config=config,
):
    if DEBUG:
        step["messages"][-1].pretty_print()
    if step["messages"][-1].type == "ai":  # and not message.tool_calls:
        print("AI:",step["messages"][-1].content)

########

chat_history = graph.get_state(config).values["messages"]
with open("chat_history.txt", "w") as f:
    f.writelines([message.content + "\n\n" for message in chat_history])

print("Chat history written: chat_history.txt")
