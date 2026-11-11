#!/usr/bin/env bash
# Runs every time the codespace starts.

# Start the Ollama server in the background (containers have no systemd).
if command -v ollama >/dev/null 2>&1 && ! pgrep -x ollama >/dev/null 2>&1; then
    # setsid detaches the server, so it survives the end of this script.
    setsid nohup ollama serve > /tmp/ollama.log 2>&1 < /dev/null &
    sleep 2
fi

if [ -f data/ikea_reviews.csv ] && [ -f data/mcdonalds_reviews.csv ]; then
    echo "Data found."
elif [ -f data/llm_fundamentals_data.zip ]; then
    if unzip -l data/llm_fundamentals_data.zip | grep -q mcdonalds_reviews.csv; then
        echo "Data zip found: run the first cells of 01_Explore_Yelp_Data to unpack it."
    else
        echo "Outdated data zip (no mcdonalds_reviews.csv): download the current"
        echo "llm_fundamentals_data.zip from Moodle and replace the file in the folder data."
    fi
else
    echo "Data missing: download llm_fundamentals_data.zip from Moodle, drag it onto the folder data"
    echo "and run the first cells of 01_Explore_Yelp_Data. See README.md, section 'Data'."
fi

if [ -n "${OPENAI_API_KEY:-}" ]; then
    echo "API key found."
else
    echo "No OPENAI_API_KEY found. See README.md, section 'API key'."
fi
