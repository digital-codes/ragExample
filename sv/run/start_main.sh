#!/bin/bash

# Resolve directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
SUPERVISOR_CONF="$BASE_DIR/supervisord.conf"

echo "SV base: $BASE_DIR"

# Start supervisord in background
supervisord -c "$SUPERVISOR_CONF"

# Wait for supervisor.sock to be created
echo "Waiting for supervisord socket to be available..."
while [ ! -S /tmp/supervisor.sock ]; do
    sleep 1
done

# Wait until all expected programs are RUNNING
echo "Waiting for all services to be RUNNING..."
while true; do
    statuses=$(supervisorctl -c "$SUPERVISOR_CONF" status | grep -v RUNNING)
    if [[ -z "$statuses" ]]; then
        echo "All services are up."
        break
    fi
    echo "Still waiting:"
    echo "$statuses"
    sleep 2
done

# Run main application
echo "Starting main RAG program..."
python rag/ragRemoteQuery.py -d localsearch -c ksk_1024 -P localllama -p localllama -s projects/ksk.db -b

# On exit, shut down supervisord
echo "Main program exited. Shutting down companion services..."
supervisorctl -c "$SUPERVISOR_CONF" shutdown


