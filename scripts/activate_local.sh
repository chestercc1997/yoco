# Source this file from any working directory; proxy settings are left untouched.
YOCO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${YOCO_ENV_PREFIX:-}" ]]; then
  export PATH="$YOCO_ENV_PREFIX/bin:$PATH"
elif [[ -d "$HOME/miniconda3/envs/${YOCO_CONDA_ENV:-yoco}" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "${YOCO_CONDA_ENV:-yoco}"
elif [[ -f "$YOCO_ROOT/.venv/bin/activate" ]]; then
  source "$YOCO_ROOT/.venv/bin/activate"
else
  printf 'Create the yoco Conda environment or set YOCO_ENV_PREFIX first.\n' >&2
  return 1
fi
export HF_HOME="$YOCO_ROOT/data/cache/huggingface"
export TIKTOKEN_CACHE_DIR="$YOCO_ROOT/data/cache/tiktoken"
export MPLCONFIGDIR="$YOCO_ROOT/data/cache/matplotlib"
export PIP_CACHE_DIR="$YOCO_ROOT/data/cache/pip"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
