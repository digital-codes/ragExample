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
    "lngMdl": "meta-llama/Llama-3.3-70B-Instruct-Turbo"
}

# openai
openAi = {
    "embUrl":"https://api.openai.com/v1/embeddings",
    "embMdl": "text-embedding-3-small",
    "lngUrl":"https://api.openai.com/v1/chat/completions",
    "lngMdl":"gpt-4o-mini"
}

# huggingface
huggingface = {
    "embUrl_1":"https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L12-v2",
    "embUrl":"https://api-inference.huggingface.co/models/BAAI/bge-small-en-v1.5",
    "embMdl": "BAAI/bge-small-en-v1.5",
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
localllama = {
    "embUrl":"http://localhost:8085/v1/embeddings",
    "embMdl":"bge-m3-Q4_K_M",
    "lngUrl":"http://localhost:8080/v1/chat/completions",
    "lngMdl": "Llama-3.2-3B-Instruct-Q4_K_M"
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
