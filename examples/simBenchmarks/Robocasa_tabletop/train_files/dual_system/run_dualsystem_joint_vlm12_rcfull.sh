#!/bin/bash
# QwenGR00T_DualSystem one-stage joint training with the VLM 0.6 s behind (stale_input vlm, video_offsets [-12, 0]) on the
# FULL RoboCasa GR1 set -> seeded 24-task sim eval with the same lag, back to back on ONE GPU group.
#
#   GPUS=4,5,6,7 TRAIN_PORT=29791 setsid nohup bash examples/simBenchmarks/Robocasa_tabletop/train_files/dual_system/run_dualsystem_joint_vlm12_rcfull.sh \
#     > dualsystem_joint_vlm12_rcfull_chain.log 2>&1 < /dev/null &
#
# Eval: ROBOCASA_VLM_PREV_FRAME=1 makes the client send [previous call's frame, frame t] (12 env steps apart = the
# training lag; frame t twice at an episode's first call), predict_action runs Qwen on the first one.
# JOINT_YAML / EVAL_TAG override the defaults.
set -u
cd "$(dirname "$0")/../../../../.."   # repo root
GPUS=${GPUS:-4,5,6,7}
TRAIN_PORT=${TRAIN_PORT:-29791}
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)
DIR=examples/simBenchmarks/Robocasa_tabletop/train_files/dual_system
YAML=${JOINT_YAML:-$DIR/joint_system12_plancond_vlm12_rcfull_50k.yaml}
RID=$(grep '^run_id:' "$YAML" | awk '{print $2}')
STEPS=$(grep -E '^\s+max_train_steps:' "$YAML" | awk '{print $2}')
CKPT=playground/Checkpoints/$RID/checkpoints/steps_${STEPS}_pytorch_model.pt

port=$TRAIN_PORT
while ss -ltn | grep -q ":$port "; do port=$((port + 1)); done
log=$PWD/regressor_dualsystem_$(basename "$YAML" .yaml)_$(date +%m%d_%H%M).log
echo "[dual-vlm12] TRAIN START $RID GPUs=$GPUS port=$port log=$log $(date)"
env PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=$GPUS \
    ACCELERATE_DEEPSPEED_ZERO_STAGE=0 NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=1 STARVLA_MAIN_TORCH_THREADS=8 \
    "$HOME/miniconda3/envs/starVLA/bin/torchrun" --nproc_per_node="$NPROC" --master_port="$port" \
    starVLA/training/train_starvla.py --config_yaml "$YAML" >> "$log" 2>&1
rc=$?
echo "[dual-vlm12] TRAIN END rc=$rc $(date)"
[ $rc -eq 0 ] || { echo "[dual-vlm12] training failed - stop"; exit 1; }
sleep 30
[ -f "$CKPT" ] || { echo "[dual-vlm12] checkpoint missing: $CKPT - stop"; exit 1; }
grep -q ROBOCASA_VLM_PREV_FRAME examples/simBenchmarks/Robocasa_tabletop/eval_files/model2robocasa_interface.py || {
    echo "[dual-vlm12] eval client lacks ROBOCASA_VLM_PREV_FRAME (VLM lag would not be emulated) - stop"; exit 1; }
ROBOCASA_VLM_PREV_FRAME=1 GPUS="$(echo "$GPUS" | tr ',' ' ')" bash "$DIR/eval_dualsystem_rc100.sh" "$CKPT" \
    "${EVAL_TAG:-$(echo "${RID#starvla_}" | sed "s/_robocasa_gr1$//")_${STEPS}}"
echo "[dual-vlm12] eval rc=$? $(date)"
echo "[dual-vlm12] ALL DONE $(date)"
