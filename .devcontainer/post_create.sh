#!/usr/bin/env bash
# Runs once when the codespace is created.
set -euo pipefail

echo ">>> Installing Python packages"
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

echo ">>> Installing Ollama (exercise 4, optional; failures do not stop the setup)"
if ! command -v ollama >/dev/null 2>&1; then
    curl -fsSL https://ollama.com/install.sh | sh || echo "Ollama install failed; local_model.ipynb can install it later."
fi

echo ">>> Setup finished"
