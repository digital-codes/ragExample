#!/bin/bash
hostname=$(hostname)

if [[ "$hostname" == "lap3" ]]; then
    export LD_LIBRARY_PATH="/opt/llama/lib:$LD_LIBRARY_PATH"
    export PATH="/opt/llama/bin:$PATH"
elif [[ "$hostname" == "tux" ]]; then
    source /opt/oneapi/setvars.sh
    export LD_LIBRARY_PATH="/opt/llama/gpu/lib64:$LD_LIBRARY_PATH"
    export PATH="/opt/llama/gpu/bin:$PATH"
else
    echo "Unknown hostname. Exiting."
    exit 1
fi
exec llama-server -m /opt/llama/models/bge-m3-Q4_K_M.gguf --embeddings --port 8085 --no-webui
