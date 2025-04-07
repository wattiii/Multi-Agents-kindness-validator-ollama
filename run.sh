#!/bin/bash

# Check if Gel database is running
if ! gel instance status my_instance &>/dev/null; then
    echo "Starting Gel database..."
    gel instance start my_instance
    sleep 5  # Give it a few seconds to start
fi

# Run Streamlit
PYTHONPATH=$(pwd) streamlit run agents/main.py 2>&1 | tee streamlit_debug.log
