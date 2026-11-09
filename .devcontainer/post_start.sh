#!/usr/bin/env bash
# Runs every time the codespace starts.

# Start the Ollama server in the background (containers have no systemd).
if command -v ollama >/dev/null 2>&1 && ! pgrep -x ollama >/dev/null 2>&1; then
    nohup ollama serve > /tmp/ollama.log 2>&1 &
fi

if [ -f data/ikea_reviews.csv ]; then
    echo "Data found."
else
    echo "Data missing: download llm_fundamentals_data.zip from Moodle, drag it onto the folder data"
    echo "and run the first cells of 01_Explore_Yelp_Data. See README.md, section 'Data'."
fi

if [ -n "${OPENAI_API_KEY:-}" ]; then
    echo "API key found."
else
    echo "No OPENAI_API_KEY found. See README.md, section 'API key'."
fi
