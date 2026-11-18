#!/usr/bin/env bash
# On the pod, from the repo root: install the project and, in a separate venv, vLLM.
set -euo pipefail
cd "$(dirname "$0")/.."
source gpu/env.sh

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync --extra train

# vLLM pins its own torch; keeping it in its own venv means it can never conflict with ours.
[ -x "$VLLM_VENV/bin/python" ] || uv venv --python 3.12 "$VLLM_VENV"
uv pip install --python "$VLLM_VENV/bin/python" --torch-backend=auto vllm

echo "--- versions"
uv --version
uv run python -c "import torch; print('project torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available())"
"$VLLM_VENV/bin/python" -c "import torch, vllm; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)"
nvidia-smi
