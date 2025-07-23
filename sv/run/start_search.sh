#!/bin/bash
#exec /opt/llama/search/annService 1024 9001 /opt/llama/data/vectors/ksk_1024_*de.vec
if [ -z "$RAG_SEARCH_ARGS" ]; then
    echo "Error: RAG_SEARCH_ARGS environment variable is not set."
    exit 1
fi
if [ -z "$RAG_SEARCH_FILES" ]; then
    echo "Error: RAG_SEARCH_FILES environment variable is not set."
    exit 1
fi
# Expand the glob into actual files
FILES=($RAG_SEARCH_FILES)
echo "Files: ${FILES[*]}"

# Check if any files were found
if [ ${#FILES[@]} -eq 0 ]; then
    echo "Error: No vector files matched pattern."
    exit 1
fi

# Build RAG_SEARCH_ARGS
ARGS="$RAG_SEARCH_ARGS ${FILES[*]}"
echo "ARGS: $ARGS"

exec /opt/llama/search/annService $ARGS

