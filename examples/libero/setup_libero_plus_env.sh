#!/usr/bin/env bash
# Run from anywhere inside the openpi fork. Writes a project-local LIBERO config
# pointing at the third_party/libero (= LIBERO-plus) submodule, so rollouts load
# the Plus BDDLs/assets. Non-destructive: does NOT touch ~/.libero/config.yaml
# (which may point at a different LIBERO install). Idempotent.
set -euo pipefail

FORK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # examples/libero/../.. = fork root
PLUS="$FORK_ROOT/third_party/libero/libero/libero"
CONFIG_DIR="$FORK_ROOT/examples/libero/libero_config"

# the submodule must be initialized AND be the Plus repo (525-line env_wrapper)
if [ ! -f "$PLUS/envs/env_wrapper.py" ]; then
  echo "ERROR: $PLUS/envs/env_wrapper.py missing." >&2
  echo "  Repoint third_party/libero to leo5470/LIBERO-plus and run:" >&2
  echo "    git submodule sync third_party/libero" >&2
  echo "    git submodule update --init --recursive third_party/libero" >&2
  exit 1
fi
LINES=$(wc -l < "$PLUS/envs/env_wrapper.py")
[ "$LINES" -lt 400 ] && echo "WARNING: env_wrapper.py is $LINES lines (Plus is ~525; vanilla is 278) -- submodule may still be upstream LIBERO." >&2

mkdir -p "$CONFIG_DIR"
cat > "$CONFIG_DIR/config.yaml" <<YAML
assets: $PLUS/assets
bddl_files: $PLUS/bddl_files
benchmark_root: $PLUS
datasets: $FORK_ROOT/third_party/libero/libero/datasets
init_states: $PLUS/init_files
YAML
echo "wrote $CONFIG_DIR/config.yaml"

if [ -z "$(ls -A "$PLUS/assets" 2>/dev/null)" ]; then
  echo "WARNING: $PLUS/assets is EMPTY. Download assets.zip from" >&2
  echo "         https://huggingface.co/datasets/Sylvest/LIBERO-plus and unzip here" >&2
  echo "         before running rollouts, or perturbed scenes will fail to load." >&2
fi

echo
echo "Setup done. Before running the client, export:"
echo "    export LIBERO_CONFIG_PATH=$CONFIG_DIR"
