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
# physical batch size large enough! to prevent error 500
exec llama-server -m /opt/llama/models/bge-m3-Q4_K_M.gguf -ub 4096 --embeddings --port 8085 --no-webui
