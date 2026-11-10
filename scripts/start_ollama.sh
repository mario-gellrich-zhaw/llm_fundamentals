#!/usr/bin/env bash
# Install Ollama if it is missing, start the server and download a model.
# Used by 05_Local_Model:  bash scripts/start_ollama.sh qwen3:1.7b

MODEL="${1:-qwen3:1.7b}"

# 1. Install Ollama (only if missing; about 2 minutes)
if ! command -v ollama >/dev/null 2>&1; then
    echo ">>> Installing Ollama ..."
    # The installer needs zstd to unpack; the Python image does not include it
    command -v zstd >/dev/null 2>&1 || { sudo apt-get update -qq && sudo apt-get install -y -qq zstd; }
    curl -fsSL https://ollama.com/install.sh | sh
fi

# 2. Start the server in the background (only if it is not running)
if ! pgrep -x ollama >/dev/null 2>&1; then
    echo ">>> Starting Ollama ..."
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    sleep 3
fi

# 3. Download the model (only the first time) and list the installed models
echo ">>> Model: $MODEL"
ollama pull "$MODEL" 2>&1 | tail -n 1
ollama list
