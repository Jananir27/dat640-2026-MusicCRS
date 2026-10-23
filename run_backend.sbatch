#!/bin/bash
# Submit from the server checkout:
# sbatch run_backend.sbatch cache/g1-485101/final_retrieval.json
# Temporary browser testing: GPU backend, local frontend, SSH tunnel via ssh1.
#SBATCH --job-name=musiccrs-web
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/backend-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

export MUSICCRS_RETRIEVAL_ARTIFACT="${1:-$PWD/cache/final_retrieval.json}"
test -r "$MUSICCRS_RETRIEVAL_ARTIFACT"
export HF_HOME="$PWD/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$PWD/cache/torch"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MUSICCRS_PORT="${MUSICCRS_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"

backend_node="$(hostname -s)"
echo "Backend job: $SLURM_JOB_ID on $backend_node, port $MUSICCRS_PORT"
echo "After the HTTP server starts, run this in a Mac terminal and leave it open:"
printf 'ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:5000:%s:%s %s@ssh1.ux.uis.no\n' \
    "$backend_node" "$MUSICCRS_PORT" "$USER"
echo "Loading the saved configuration, catalog, and Qwen model (no G1 retuning)."

python -u - <<'PY'
import os
import torch
from dialoguekit.platforms.flask_socket_platform import ChatNamespace, FlaskSocketPlatform
from musiccrs.musiccrs import MusicCRS
from musiccrs.music_service import get_shared_music_service
from retrieval.final import load_artifact

if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot access the GPU allocated by Slurm")
print("Allocated GPU:", torch.cuda.get_device_name(0), flush=True)
load_artifact()
music_service = get_shared_music_service()
music_service._get_retriever()
music_service.warm_r5()


class BrowserMusicCRS(MusicCRS):
    def __init__(self, agent_id=None):
        # DialogueKit's platform passes the agent class name positionally.
        # R1–R8 use catalog/retrieval; the unrelated /ask_llm command is disabled.
        super().__init__(use_llm=False)


platform = FlaskSocketPlatform(BrowserMusicCRS)
platform.socketio.on_namespace(ChatNamespace("/", platform))
print("Retrieval initialized; starting the browser-test server.", flush=True)
# Listen on the compute node's network interface so ssh1 can reach it through
# the user's SSH tunnel. This is a temporary development server, not deployment.
# Explicitly permit Werkzeug in the non-interactive Slurm batch environment.
platform.socketio.run(
    platform.app,
    host="0.0.0.0",
    port=int(os.environ["MUSICCRS_PORT"]),
    debug=False,
    use_reloader=False,
    allow_unsafe_werkzeug=True,
)
PY
