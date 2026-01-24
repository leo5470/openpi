#!/bin/bash
#SBATCH --job-name=libero-pi
#SBATCH --output=libero_%j.log
#SBATCH --error=libero_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2            # Request 2 GPUs (1 for server, 1 for client)
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Exit on error
set -e

# Load necessary modules (adjust based on your cluster setup)
# module load singularity

# Define paths
REPO_ROOT=$(pwd)
LIBERO_SIF="${REPO_ROOT}/examples/libero/libero.sif"
SERVER_SIF="${REPO_ROOT}/scripts/docker/serve_policy.sif"

# Ensure images exist
if [ ! -f "$LIBERO_SIF" ] || [ ! -f "$SERVER_SIF" ]; then
    echo "Error: Singularity images not found."
    echo "Please build them first using:"
    echo "  sudo singularity build $LIBERO_SIF examples/libero/libero.def"
    echo "  sudo singularity build $SERVER_SIF scripts/docker/serve_policy.def"
    exit 1
fi

# Set default arguments if not provided
SERVER_ARGS="${SERVER_ARGS:-checkpoint=openpi/policy-model-name}"
CLIENT_ARGS="${CLIENT_ARGS:-}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$HOME/.cache/openpi}"
mkdir -p "$OPENPI_DATA_HOME"

# Rendering setup (EGL for headless GPU rendering)
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=egl

# Function to kill background processes on exit
cleanup() {
    echo "Cleaning up..."
    kill $(jobs -p) 2>/dev/null || true
}
trap cleanup EXIT

echo "Starting OpenPI Policy Server..."
# Run server in background
# We bind the repo root to /app and the data home to /openpi_assets
singularity exec --nv \
    --bind "${REPO_ROOT}:/app" \
    --bind "${OPENPI_DATA_HOME}:/openpi_assets" \
    --env "OPENPI_DATA_HOME=/openpi_assets" \
    --env "IS_DOCKER=true" \
    "$SERVER_SIF" \
    bash -c "cd /app && python scripts/serve_policy.py $SERVER_ARGS" &

# Wait for server to be ready (adjust sleep as needed or implement a health check)
echo "Waiting for server to initialize..."
sleep 30

echo "Starting LIBERO Client..."
# Run client in foreground
# We bind the repo root to /app and data to /data
singularity exec --nv \
    --bind "${REPO_ROOT}:/app" \
    --bind "${REPO_ROOT}/data:/data" \
    --env "DISPLAY=$DISPLAY" \
    --env "MUJOCO_GL=$MUJOCO_GL" \
    --env "MUJOCO_EGL_DEVICE_ID=0" \
    --env "PYOPENGL_PLATFORM=egl" \
    "$LIBERO_SIF" \
    bash -c "cd /app && python examples/libero/main.py $CLIENT_ARGS"

echo "Job completed successfully."
