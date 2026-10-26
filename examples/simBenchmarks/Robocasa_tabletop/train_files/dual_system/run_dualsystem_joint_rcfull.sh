#!/bin/bash
# QwenGR00T_DualSystem one-stage joint training (System 2 + System 1 together) on the FULL RoboCasa GR1 set -> seeded
# 24-task sim eval, back to back on ONE GPU group.
#
#   GPUS=4,5,6,7 TRAIN_PORT=29790 setsid nohup bash examples/simBenchmarks/Robocasa_tabletop/train_files/dual_system/run_dualsystem_joint_rcfull.sh \
#     > dualsystem_joint_rcfull_chain.log 2>&1 < /dev/null &
#
# JOINT_YAML / EVAL_TAG override the defaults.
set -u
cd "$(dirname "$0")/../../../../.."   # repo root
GPUS=${GPUS:-4,5,6,7}
TRAIN_PORT=${TRAIN_PORT:-29790}
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)
DIR=examples/simBenchmarks/Robocasa_tabletop/train_files/dual_system
YAML=${JOINT_YAML:-$DIR/joint_system12_plancond_rcfull_50k.yaml}
RID=$(grep '^run_id:' "$YAML" | awk '{print $2}')
STEPS=$(grep -E '^\s+max_train_steps:' "$YAML" | awk '{print $2}')
CKPT=playground/Checkpoints/$RID/checkpoints/steps_${STEPS}_pytorch_model.pt

port=$TRAIN_PORT
while ss -ltn | grep -q ":$port "; do port=$((port + 1)); done
log=$PWD/regressor_dualsystem_$(basename "$YAML" .yaml)_$(date +%m%d_%H%M).log
echo "[dual-joint] TRAIN START $RID GPUs=$GPUS port=$port log=$log $(date)"
env PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=$GPUS \
    ACCELERATE_DEEPSPEED_ZERO_STAGE=0 NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=1 STARVLA_MAIN_TORCH_THREADS=8 \
    "$HOME/miniconda3/envs/starVLA/bin/torchrun" --nproc_per_node="$NPROC" --master_port="$port" \
    starVLA/training/train_starvla.py --config_yaml "$YAML" >> "$log" 2>&1
rc=$?
echo "[dual-joint] TRAIN END rc=$rc $(date)"
[ $rc -eq 0 ] || { echo "[dual-joint] training failed - stop"; exit 1; }
sleep 30
[ -f "$CKPT" ] || { echo "[dual-joint] checkpoint missing: $CKPT - stop"; exit 1; }
GPUS="$(echo "$GPUS" | tr ',' ' ')" bash "$DIR/eval_dualsystem_rc100.sh" "$CKPT" "${EVAL_TAG:-$(echo "${RID#starvla_}" | sed "s/_robocasa_gr1$//")_${STEPS}}"
echo "[dual-joint] eval rc=$? $(date)"
echo "[dual-joint] ALL DONE $(date)"
