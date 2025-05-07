# from https://platform.openai.com/docs/guides/function-calling?api-mode=responses
import os
import sys
import argparse


# add path to rag modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "./rag")))

import openai
import json

import private_remote as pr
import ragConfig as cfg

# openai mode works for openai and deepinfra and localllama 
# localllama requires explicit hint to use tool.
# not for huggingface

# openai paramter type array not working, only simple types (string)

# deepinfra models 0 (default) and 2 are good.

providers = ["localllama", "deepInfra", "openAi","huggingface","ollama"]
provider = providers[4]

if provider == "openAi":
    url = cfg.openAi["lngUrl"].split("/chat")[0]
    model = cfg.openAi["lngMdl"]
    key = pr.openAi["apiKey"]
elif provider == "deepInfra":
    url = cfg.deepInfra["lngUrl"].split("/chat")[0]
    model = cfg.deepInfra["lngMdl"]
    key = pr.deepInfra["apiKey"]
elif provider == "localllama":
    url = cfg.localllama["lngUrl"].split("/chat")[0]
    model = "granite-3.3-2b-instruct"  # cfg.localllama["lngMdl"]
    key = "1234"
elif provider == "huggingface":
    url = cfg.huggingface["lngUrl"][0].split("/chat")[0]
    model = cfg.huggingface["lngMdl"][0]
    key = pr.huggingface["apiKey"]
elif provider == "ollama":
    url = "http://localhost:11434/api/chat"
    model = "mistral:latest"
    key = "1234"
else:
    raise ("Invalid provider")

print("Using provider:", provider, "model:", model)

client = openai.OpenAI(base_url=url, api_key=key)

temperature = 0.2


# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def get_current_weather(location):
    """Get the current weather in a given location"""
    print("Calling get_current_weather client side.")
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "75"})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "60"})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "70"})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})


def GetColorRank(colorSet):
    """ "Returns ranked list of colors. best first."""
    print("Calling getColorRank client side.")
    return json.dumps({"Best color is": colorSet[-1]})


# here is the definition of our function
tools1 = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    }
                },
                "required": ["location"],
            },
        },
    }
]

query1 = "What is the weather in San Francisco?"
documents1 = []


tools2 = [
    {
        "type": "function",
        "function": {
            "description": "Returns ranked list of colors. best first.",
            "name": "GetColorRank",
            "strict": False,
            "parameters": {
                "type": "object",
                "properties": {
                    "colorSet": {
                        "type": "array",
                        "description": "List of colors to check for. Returns best",
                    }
                },
                "required": ["colorSet"],
            },
        }
    }
]


#query2 = "need best color. Use only data from documents. Use tools only if required not to verify. Justify tool usage."
#query2 = "need best color. Use only data from documents. Use tool functions only if required, not to verify."
query2 = "need best color. Use only data from documents. Use tools if required."
#query2 = "need best color. Use only data from documents. High penalty to use tools, must be justified."

documents2 = ["document_1:\nblue is bad", "document_2:\ngreen is not good","document_3:\nyellow is unclear"]
#documents2 = ["document_1:\nblue is bad", "document_2:\ngreen is not good"]

tools = tools2
query = query2
documents = documents2


# here is the user request
messages = [
    {"role": "user", "content": query},
    {"role": "user", "content": "\n".join(["documents:","\n".join(documents)])},
]

#messages.append({"role": "user", "content": "Use tools only if required"})

print("First messages:", messages)
# let's send the request and print the response
response = client.chat.completions.create(
    model=model,
    messages=messages,
    tools=tools,
    tool_choice="auto",
)
print(response)

stop = response.choices[0].finish_reason
print("Finish reason:", stop)   

if stop != "tool_calls": 
    print("Response:", response.choices[0].message.content)
    if provider != "huggingface":
        sys.exit()
    # huggingface stops with tool call
    if len(response.choices[0].message.tool_calls) == 0: 
        print("No tool calls from huggingface either")
        sys.exit()  
    
tool_calls = response.choices[0].message.tool_calls
print("Tool calls:",tool_calls)

for tool_call in tool_calls:
    print(tool_call.model_dump())


# extend conversation with assistant's reply
messages.append(response.choices[0].message)

for tool_call in tool_calls:
    function_name = tool_call.function.name
    if function_name == "get_current_weather":
        function_args = json.loads(tool_call.function.arguments)
        function_response = get_current_weather(location=function_args.get("location"))
    elif function_name == "GetColorRank":
        function_args = json.loads(tool_call.function.arguments)
        function_response = GetColorRank(colorSet=function_args.get("colorSet"))
    else:
        raise ("Unknown function call")


# extend conversation with function response
messages.append(
    {
        "tool_call_id": tool_call.id,
        "role": "tool",
        "content": function_response,
    }
)


# get a new response from the model where it can see the function responses
print("Second messages:", messages)
second_response = client.chat.completions.create(
    model=model,
    messages=messages,
    tools=tools,
    tool_choice="auto",
)
print("Second response:")
print(second_response)

print(second_response.choices[0].message.content)


