# from https://platform.openai.com/docs/guides/function-calling?api-mode=responses
import os
import sys

# add path to rag modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "./rag")))

import openai
import json

import private_remote as pr
import ragConfig as cfg

# openai mode works for openai and deepinfra
# localllama requires explicit hint to use tool.
# not for huggingface

# deepinfra models 0 (default) and 2 are good.

providers = ["localllama", "deepInfra", "openAi"]
provider = providers[1]

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
    return json.dumps({"colorSet": colorSet[::-1]})


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
        "function": {
            "description": "Returns ranked list of colors. best first.",
            "name": "GetColorRank",
            "strict": False,
            "parameters": {
                "type": "object",
                "required": ["colorSet"],
                "properties": {
                    "colorSet": {
                        "description": "List of colors to check for",
                        "type": "array",
                    }
                },
            },
        },
        "type": "function",
    }
]


query2 = "need best color"

documents2 = ["document_1:\nblue is bad", "document_2:\ngreen is not good"]

tools = tools2
query = query2
documents = documents2


# here is the user request
messages = [
    {"role": "user", "content": query},
    {"role": "user", "content": "\n".join(documents)},
]

if provider == "localllama":
    messages.append({"role": "user", "content": "Use tool only if required"})

print("First messages:", messages)
# let's send the request and print the response
response = client.chat.completions.create(
    model=model,
    messages=messages,
    tools=tools,
    tool_choice="auto",
)
print(response)
tool_calls = response.choices[0].message.tool_calls
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

sys.exit()

#######################################


size = 200

hdrs = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {key}",
}

tools = [
    {
        "function": {
            "description": "Returns ranked list of colors. best first.",
            "name": "GetColorRank",
            "strict": False,
            "parameters": {
                "type": "object",
                "required": [],
                "properties": {
                    "colorSet": {
                        "description": "List of colors to check for",
                        "type": "array",
                    }
                },
            },
        },
        "type": "function",
    }
]


query = "need best color"

documents = ["document_1:\nblue is bad", "document_2:\ngreen is not good"]


richQuery = f"""
You are a helpful assistant with access to the following tools. 
When a tool is required to answer the user's query, respond only with <|tool_call|> followed by a JSON list of tools used. 
If a tool does not exist in the provided list of tools, notify the user that you do not have the ability to fulfill the request.
The question is:
{query}
Documents:
{"\n".join(documents)}
"""


rdata = {
    "tools": tools,
    "controls": ["tinking"],
    "model": model,
    "temperature": temperature,
    "messages": [{"role": "user", "content": richQuery}],
}


response = requests.post(url, headers=hdrs, json=rdata)
if response.status_code == 200:
    data = response.json()
    print(data)

else:
    print(response.status_code, response.content)
    raise Exception(f"Error: {response.status_code} - {response.content}")

finish = data["choices"][0]["finish_reason"]

if finish == "stop":
    text = data["choices"][0]["message"]["content"].strip()
    tokens = data["usage"]["total_tokens"]
    print("Text:\n", text)
    print("Tokens:\n", tokens)
elif finish == "tool_calls":
    print("function call\n")
    print(data["choices"][0]["message"]["tool_calls"])
    id = data["choices"][0]["message"]["tool_calls"][0]["id"]
    # assume dummy function call
    rdata["messages"] = [
        {
            "role": "tool",
            "content": "ranked list of colors. first is best, last is worst: green, blue",
            "tool_call_id": id,
        },
        {"role": "user", "content": query},
    ]
    response = requests.post(url, headers=hdrs, json=rdata)
    if response.status_code == 200:
        data = response.json()
        print(data)
        print("Text:\n", data["choices"][0]["message"]["content"].strip())
    else:
        print(response.status_code, response.content)
        raise Exception(f"Error: {response.status_code} - {response.content}")

else:
    raise Exception("Error: stopped due to:", data["choices"][0]["finish_reason"])
