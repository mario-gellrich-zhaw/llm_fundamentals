#!/usr/bin/env bash
# Runs once when the codespace is created.
set -euo pipefail

echo ">>> Installing Python packages"
python -m pip install --upgrade pip
# The default PyPI torch bundles ~4 GB of CUDA libraries; there is no GPU here.
pip install "torch>=2.2,<3" -c constraints-tested.txt \
    --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -c constraints-tested.txt

echo ">>> Installing the local English privacy model"
python -m spacy download en_core_web_sm

echo ">>> Installing Ollama (exercise 5, optional; failures do not stop the setup)"
if ! command -v ollama >/dev/null 2>&1; then
    # Run in a subshell so a failure here cannot abort the outer script (set -e).
    # The Ollama installer needs zstd to unpack; the Python image does not include it.
    (
        command -v zstd >/dev/null 2>&1 || sudo apt-get update -qq && sudo apt-get install -y -qq zstd
        curl -fsSL https://ollama.com/install.sh | sh
    ) || echo "Ollama install failed; local_model.ipynb can install it later."
fi

echo ">>> Setup finished"
