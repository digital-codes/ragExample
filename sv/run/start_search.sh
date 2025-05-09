#!/bin/bash
#exec /opt/llama/search/annService 1024 9001 /opt/llama/data/vectors/ksk_1024_*de.vec
if [ -z "$RAG_SEARCH_ARGS" ]; then
    echo "Error: RAG_SEARCH_ARGS environment variable is not set."
    exit 1
fi
exec /opt/llama/search/annService $RAG_SEARCH_ARGS

