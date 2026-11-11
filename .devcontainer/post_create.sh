#!/usr/bin/env bash
# Runs once when the codespace is created.
set -euo pipefail

echo ">>> Installing Python packages"
python -m pip install --upgrade pip
pip install -r requirements.txt -c constraints-tested.txt

echo ">>> Installing the local English privacy model"
python -m spacy download en_core_web_sm

echo ">>> Installing Ollama (exercise 5, optional; failures do not stop the setup)"
if ! command -v ollama >/dev/null 2>&1; then
    # The Ollama installer needs zstd to unpack; the Python image does not include it
    command -v zstd >/dev/null 2>&1 || { sudo apt-get update -qq && sudo apt-get install -y -qq zstd; } || true
    curl -fsSL https://ollama.com/install.sh | sh || echo "Ollama install failed; local_model.ipynb can install it later."
fi

echo ">>> Setup finished"
