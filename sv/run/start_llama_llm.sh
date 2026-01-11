#!/bin/bash
hostname=$(hostname)
if [[ "$hostname" == "tux3" ]]; then
    source /opt/oneapi/setvars.sh
    export LD_LIBRARY_PATH="/opt/llama/gpu/lib64:$LD_LIBRARY_PATH"
    export PATH="/opt/llama/gpu/bin:$PATH"
else
    export LD_LIBRARY_PATH="/opt/llama/lib:$LD_LIBRARY_PATH"
    export PATH="/opt/llama/bin:$PATH"
fi

if [ -z "$RAG_LLM_MODEL" ]; then
    echo "Error: RAG_LLM_MODEL environment variable is not set."
    exit 1
fi

port=${RAG_LLM_PORT:-8080}
exec llama-server --jinja -m "$RAG_LLM_MODEL" --port "$port" -c 40000 --no-webui


