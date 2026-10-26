#!/bin/bash

# ============================================================
# Argument Parsing
# ============================================================
###########################################################################################
# === Please modify the paths to Python executables in conda environments ===

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
starVLA_PYTHON=~/miniconda3/envs/starVLA/bin/python
ROBOCASA_PYTHON=~/miniconda3/envs/robocasa/bin/python
export PYTHONPATH=$(pwd):${PYTHONPATH}
CKPT_DEFAULT="StarVLA/Qwen3-VL-OFT-Robocasa/checkpoints/steps_90000_pytorch_model.pt"


# === End of environment variable configuration ===
###########################################################################################


N_ENVS_DEFAULT=1
MAX_EPISODE_STEPS_DEFAULT=720
N_ACTION_STEPS_DEFAULT=12

BASE_PORT=${BASE_PORT:-6398}
# GPU_IDS: space-separated physical GPU ids to use (default all 8)
GPU_IDS=(${GPU_IDS:-0 1 2 3 4 5 6 7})
NUM_GPUS=${#GPU_IDS[@]}

# Parse command-line arguments
CKPT_PATH=${1:-$CKPT_DEFAULT}
N_ENVS=${2:-$N_ENVS_DEFAULT}
MAX_EPISODE_STEPS=${3:-$MAX_EPISODE_STEPS_DEFAULT}
N_ACTION_STEPS=${4:-$N_ACTION_STEPS_DEFAULT}
# Extra flags forwarded to simulation_env.py. The default matches CKPT_DEFAULT
# (Qwen3-VL-OFT-Robocasa, trained without state input). Pass "" for checkpoints
# that consume state (e.g. Qwen3-VL-GR00T-Robocasa-gr1).
EXTRA_EVAL_ARGS=${5-"--args.no_send_state"}


echo "=== Evaluation Configuration ==="
echo "Checkpoint Path      : ${CKPT_PATH}"
echo "Number of Envs       : ${N_ENVS}"
echo "Max Episode Steps    : ${MAX_EPISODE_STEPS}"
echo "Action Chunk Length  : ${N_ACTION_STEPS}"
echo "Extra Eval Args      : ${EXTRA_EVAL_ARGS}"
echo "================================"

# ============================================================
# Evaluation Function
# ============================================================

EvalEnv() {
    local GPU_ID=$1
    local PORT=$2
    local ENV_NAME=$3
    local CKPT_PATH=$4
    local LOG_DIR=$5
    local ROBOCASA_PYTHON=$6
    local N_ENVS=$7
    local MAX_EPISODE_STEPS=$8
    local N_ACTION_STEPS=$9
    # Save root: parent directory of CKPT_PATH
    local SAVE_ROOT=$(dirname "$(dirname "$CKPT_PATH")")
    local ckpt_name=$(basename "$CKPT_PATH" .pt)
    # VIDEO_SUFFIX (env, default empty): keeps videos of a re-eval (e.g. another scene-seed set) apart
    local VIDEO_OUT_PATH="${SAVE_ROOT}/videos${VIDEO_SUFFIX:-}/${ckpt_name}/n_action_steps_${N_ACTION_STEPS}_max_episode_steps_${MAX_EPISODE_STEPS}_n_envs_${N_ENVS}_${ENV_NAME}"
    mkdir -p "${VIDEO_OUT_PATH}"

    echo "Launching evaluation | GPU ${GPU_ID} | Port ${PORT} | Env ${ENV_NAME}"

    CUDA_VISIBLE_DEVICES=${GPU_ID} \
    ${ROBOCASA_PYTHON} examples/simBenchmarks/Robocasa_tabletop/eval_files/simulation_env.py \
        --args.env_name "${ENV_NAME}" \
        --args.port "${PORT}" \
        --args.n_episodes 50 \
        --args.n_envs "${N_ENVS}" \
        --args.max_episode_steps "${MAX_EPISODE_STEPS}" \
        --args.n_action_steps "${N_ACTION_STEPS}" \
        --args.video_out_path "${VIDEO_OUT_PATH}" \
        --args.pretrained_path "${CKPT_PATH}" \
        ${EXTRA_EVAL_ARGS} \
        > "${LOG_DIR}/eval_env_${ENV_NAME//\//_}_gpu${GPU_ID}.log" 2>&1
}

# ============================================================
# Environment List
# ============================================================

# EVAL_ENVS: optional whitespace-separated env-name override (e.g. only the 6 *Close tasks).
# Names may be given bare ("PnPCupToDrawerClose") or fully qualified; bare names are expanded to
# gr1_unified/<name>_GR1ArmsAndWaistFourierHands_Env. Unset -> the full 24-task list below.
ENV_NAMES=(
  gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env
)

if [[ -n "${EVAL_ENVS:-}" ]]; then
    ENV_NAMES=()
    for NAME in ${EVAL_ENVS}; do
        if [[ "${NAME}" == *"/"* ]]; then
            ENV_NAMES+=("${NAME}")
        else
            ENV_NAMES+=("gr1_unified/${NAME}_GR1ArmsAndWaistFourierHands_Env")
        fi
    done
fi

# ============================================================
# Runtime Configuration
# ============================================================



LOG_DIR="${CKPT_PATH}.log/eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

echo "=== Launching Multi-GPU Evaluation ==="
echo "GPUs            : ${NUM_GPUS}"
echo "Num Environments: ${#ENV_NAMES[@]}"
echo "Log Directory   : ${LOG_DIR}"

# ============================================================
# Step 1: Launch Policy Servers
# ============================================================

SERVER_PIDS=()

for SLOT in $(seq 0 $((NUM_GPUS - 1))); do
    GPU_ID=${GPU_IDS[$SLOT]}
    PORT=$((BASE_PORT + SLOT))
    echo "Starting policy server | GPU ${GPU_ID} | Port ${PORT}"

    CUDA_VISIBLE_DEVICES=${GPU_ID} \
    ${starVLA_PYTHON} deployment/model_server/server_policy.py \
        --ckpt_path "${CKPT_PATH}" \
        --port "${PORT}" \
        --use_bf16 \
        > "${LOG_DIR}/server_gpu${GPU_ID}_port${PORT}.log" 2>&1 &

    SERVER_PIDS[$SLOT]=$!

    sleep 10
done

sleep 30

# ============================================================
# Step 2: Dispatch Environments to GPUs
# ============================================================

COUNT=0
ENV_PIDS=()
for ENV_NAME in "${ENV_NAMES[@]}"; do
    SLOT=$((COUNT % NUM_GPUS))
    GPU_ID=${GPU_IDS[$SLOT]}
    PORT=$((BASE_PORT + SLOT))

    if (( (COUNT + 1) % NUM_GPUS == 0 )); then
        EvalEnv "${GPU_ID}" "${PORT}" "${ENV_NAME}" "${CKPT_PATH}" "${LOG_DIR}" \
                "${ROBOCASA_PYTHON}" "${N_ENVS}" "${MAX_EPISODE_STEPS}" "${N_ACTION_STEPS}"
    else
        EvalEnv "${GPU_ID}" "${PORT}" "${ENV_NAME}" "${CKPT_PATH}" "${LOG_DIR}" \
                "${ROBOCASA_PYTHON}" "${N_ENVS}" "${MAX_EPISODE_STEPS}" "${N_ACTION_STEPS}" &
        ENV_PIDS+=($!)
    fi

    COUNT=$((COUNT + 1))

    sleep 2
done

# ============================================================
# Step 3: Cleanup
# ============================================================

# Wait for THIS run's evaluation envs only. (A machine-wide `pgrep simulation_env.py` also waited for any
# other concurrent eval, e.g. a second checkpoint on other GPUs, keeping this run's servers/GPUs busy.)
echo "Waiting for all evaluation environments to finish..."
for PID in "${ENV_PIDS[@]}"; do
    wait "${PID}"
done

echo ""
echo "Shutting down policy servers..."

for PID in "${SERVER_PIDS[@]}"; do
    kill "${PID}" 2>/dev/null && echo "Killed server PID ${PID}"
done

echo "=== Evaluation Finished ==="
