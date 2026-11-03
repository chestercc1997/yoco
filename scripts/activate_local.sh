# Source this file from any working directory; proxy settings are left untouched.
YOCO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$YOCO_ROOT/.venv/bin/activate"
export HF_HOME="$YOCO_ROOT/data/cache/huggingface"
export TIKTOKEN_CACHE_DIR="$YOCO_ROOT/data/cache/tiktoken"
export MPLCONFIGDIR="$YOCO_ROOT/data/cache/matplotlib"
export PIP_CACHE_DIR="$YOCO_ROOT/data/cache/pip"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
