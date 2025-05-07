# from https://platform.openai.com/docs/guides/function-calling?api-mode=responses
import os
import sys

# add path to rag modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "./rag")))

import requests
import private_remote as pr
import ragConfig as cfg

provider = "localllama"

if provider == "openAi":
    url = cfg.openAi["lngUrl"]
    key = pr.openAi["apiKey"]
    model = cfg.openAi["lngMdl"]
elif provider == "localllama":
    url = cfg.localllama["lngUrl"]
    model = "granite-3.3-2b-instruct"  # cfg.localllama["lngMdl"]
    key = ""
elif provider == "deepInfra":
    url = cfg.deepInfra["lngUrl"]
    key = pr.deepInfra["apiKey"]
    model = cfg.deepInfra["lngMdl_3"]
else:
    raise ("Invalid provider")

temperature = 0.2

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
            "strict": "False",
            "parameters": {
                "type": "object",
                "required": [],
                "properties": {
                    "colorSet": {
                        "description": "List of colors to check for. Defaults to [red]",
                        "type": "array",
                    }
                },
            },
        },
        "type": "function",
    }
]


query = "need best color"

documents = ["document_1:\nblue is bad","document_2:\ngreen is not good"]


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
    else:
        print(response.status_code, response.content)
        raise Exception(f"Error: {response.status_code} - {response.content}")

else:
    raise Exception("Error: stopped due to:", data["choices"][0]["finish_reason"])
