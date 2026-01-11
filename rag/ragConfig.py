# zilliz database
zilliz = {
    "region": "gcp-us-west1",
    "cluster": "in03-eb450554ac4fcc5"
}

# deep infra
deepInfra = {
    "embUrl":"https://api.deepinfra.com/v1/openai/embeddings",
    "lngUrl":"https://api.deepinfra.com/v1/openai/chat/completions",
    "embMdl_0": "sentence-transformers/all-MiniLM-L12-v2",
    "embMdl": "BAAI/bge-m3",
    "embToks": 8192,
    "embSize":1024,
    "lngMdl": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "lngToks": 128*1024,
    "lngMdl_1": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "lngMdl_2": "deepseek-ai/DeepSeek-R1-Turbo",
    "lngMdl_3": "microsoft/Phi-4-multimodal-instruct",
}

# openai
openAi = {
    "embUrl":"https://api.openai.com/v1/embeddings",
    "embMdl": "text-embedding-3-small",
    "embToks": 512,
    "embSize":384,
    "lngUrl":"https://api.openai.com/v1/chat/completions",
    "lngMdl":"gpt-4o-mini",
    "lngToks": 128*1024
}

# huggingface
huggingface = {
    "embUrl_1":"https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L12-v2",
    "embUrl":"https://api-inference.huggingface.co/models/BAAI/bge-small-en-v1.5",
    "embMdl": "BAAI/bge-small-en-v1.5",
    "embSize":384,
    "embToks" : 512,
    "lngUrl":[
        "https://api-inference.huggingface.co/models/meta-llama/Llama-3.3-70B-Instruct/v1/chat/completions",
        "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1/v1/chat/completions",
        "https://te28a6lwmtjobtrh.us-east4.gcp.endpoints.huggingface.cloud/v1/chat/completions",
        "https://dyw6aprg057hdxqc.eu-west-1.aws.endpoints.huggingface.cloud/v1/chat/completions"

    ],
    "lngMdl":[
        "meta-llama/Llama-3.3-70B-Instruct",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "Mixtral-8x7B-Instruct-v0.1",
        "llama-3-1-8b-instruct-byo"
    ],
    "endpoint":[
        "serverless-pro-only",
        "serverless",
        "dedicated",
        "dedicated"
    ]
}

# local llama.cpp server
# start like so with jina embeddings
# /opt/llama/cpu/bin/llama-server -m /opt/llama/models/jina-embeddings-v2-base-de-Q5_K_M.gguf -c 0 -b 1000 -ub 1000  --embeddings --port 8085   
# or 
# /opt/llama/cpu/bin/llama-server -m /opt/llama/models/jina-embeddings-v2-base-de-Q5_K_M.gguf -c 0 -b 1000 -ub 1000  --embeddings --port 8085 --log-disable
# local gpu server via ssh from this machine:
# ssh -L 8085:localhost:11434 user@gpu-machine 
# make sure to use proper models, e.g. bge-m3 and llama-3.2 or mistral-nemo 
localllama = {
    "embUrl":"http://localhost:8085/v1/embeddings",
    "embMdl":"bge-m3-Q4_K_M",
    "embSize":1024,
    "embToks": 8192,
    "lngUrl":"http://localhost:8080/v1/chat/completions",
    "lngMdl": "ibm-granite.granite-4.0-h-1b.Q4_K_M",
    "lngMdl_1" : "LLaMmlein_1B_chat_selected"
}

# ollama uses openai compatible embeddings and chat completions with potentially special url
# can be used remote with ssh tunnel like 
# ssh -L 8085:localhost:11434 <user>@host
# basically ollama can be used as openai, optionally with customized urls nad models
ollama = {
    "embUrl":"http://localhost:8085/v1/embeddings",
    "embMdl":"bge-m3:latest",
    "embSize":1024,
    "embToks": 8192,
    "lngUrl":"http://localhost:8080/v1/chat/completions",
    "lngMdl": "granite3.3:2b"
}


# local vector search server
localsearch = {
    "url":"http://localhost:9001/"
}

# local vector search server with py
pysearch = {
    "url":"http://localhost:9001/search"
}


# local database
database = {
    # possible options
    "options": ["sqlite", "mysql", "remote"],
    # selected option
    "selected": "sqlite"
}
