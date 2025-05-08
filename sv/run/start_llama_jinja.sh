#!/bin/bash
source /opt/oneapi/setvars.sh
export LD_LIBRARY_PATH="/opt/llama/gpu/lib64:$LD_LIBRARY_PATH"
export PATH="/opt/llama/gpu/bin:$PATH"
exec llama-server --jinja -m /opt/llama/models/granite-3.3-2b-instruct-Q4_K_M.gguf --port 8080 -c 20000 --no-webui
