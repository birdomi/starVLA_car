#!/bin/bash
# QwenGR00T_DualSystem with the VLAct Qwen3-VL-4B VLM (last 4 decoder layers trained), 0.25 s System 2 keyframes, an ActionMap System 1 and the
# episode memory (joint_system12_vlact4b_l4_kf025_mem_am5_vlm12_rcfull_50k.yaml): one-stage joint training -> seeded 24-task
# sim eval with the VLM 0.6 s behind and the memory window (client ROBOCASA_VLM_PREV_FRAMES=4: the last 4 calls' frames
# + frame t, frame_valid) -> paired comparisons vs the Qwen3.5-0.8B DualSystem runs on the same scenes.
#
#   GPUS=4,5,6,7 setsid nohup bash examples/simBenchmarks/Robocasa_tabletop/train_files/dual_system/run_dualsystem_vlact4b_mem.sh \
#     [pid to wait for] > dualsystem_vlact4b_mem_chain.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../../../../.."   # repo root
WAIT_PID=${1:-}
GPUS=${GPUS:-4,5,6,7}
TRAIN_PORT=${TRAIN_PORT:-29797}
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)
DIR=examples/simBenchmarks/Robocasa_tabletop/train_files/dual_system
YAML=${YAML:-$DIR/joint_system12_vlact4b_l4_kf025_mem_am5_vlm12_rcfull_50k.yaml}
RID=$(grep '^run_id:' "$YAML" | awk '{print $2}')
STEPS=$(grep -E '^\s+max_train_steps:' "$YAML" | awk '{print $2}')
CKPT=playground/Checkpoints/$RID/checkpoints/steps_${STEPS}_pytorch_model.pt
TAG=$(echo "${RID#starvla_}" | sed "s/_robocasa_gr1$//")_${STEPS}
PY=$HOME/miniconda3/envs/starVLA/bin/python
if [ -n "$WAIT_PID" ]; then
    echo "[vlact4b-mem] waiting for pid $WAIT_PID $(date)"
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi

port=$TRAIN_PORT
while ss -ltn | grep -q ":$port "; do port=$((port + 1)); done
log=$PWD/regressor_dualsystem_$(basename "$YAML" .yaml)_$(date +%m%d_%H%M).log
echo "[vlact4b-mem] TRAIN START $RID GPUs=$GPUS port=$port log=$log $(date)"
env PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=$GPUS \
    ACCELERATE_DEEPSPEED_ZERO_STAGE=0 NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=1 STARVLA_MAIN_TORCH_THREADS=8 \
    "$HOME/miniconda3/envs/starVLA/bin/torchrun" --nproc_per_node="$NPROC" --master_port="$port" \
    starVLA/training/train_starvla.py --config_yaml "$YAML" >> "$log" 2>&1
rc=$?
echo "[vlact4b-mem] TRAIN END rc=$rc $(date)"
[ $rc -eq 0 ] || { echo "[vlact4b-mem] training failed - stop"; exit 1; }
sleep 30
[ -f "$CKPT" ] || { echo "[vlact4b-mem] checkpoint missing: $CKPT - stop"; exit 1; }
EG="$(echo "$GPUS" | tr ',' ' ')"
ROBOCASA_VLM_PREV_FRAMES=4 GPUS="$EG" bash "$DIR/eval_dualsystem_rc100.sh" "$CKPT" "$TAG"
echo "[vlact4b-mem] eval x1 rc=$? $(date)"
N1=$(ls -dt "$CKPT.log"/eval_2* | head -1)
grep -l "vlm_prev_frames\] 5 images" "$N1"/eval_env_*.log > /dev/null 2>&1 && echo "[vlact4b-mem] clients confirm the 5-frame memory window" \
    || echo "[vlact4b-mem] WARNING: eval logs do not show the 5-frame memory window"
O=playground/Checkpoints/starvla_dualsystem_joint_plancond_vlm12_s2delta_s1res10_h16_rcfull_robocasa_gr1/checkpoints/steps_50000_pytorch_model.pt.log
O1=$O/eval_20261015_042524; O8=$O/eval_20261015_080006
J=results/eval_logs/starvla_dualsystem_joint_plancond_h16_rcfull_robocasa_gr1/eval_20261014_071215
S=results/eval_logs/starvla_dualsystem_s1plancond_h16_rcfull_robocasa_gr1/eval_20261013_030712
echo "===== s1res10 (Qwen3.5-0.8B, VLM lag) 1 sample vs VLAct-4B memory"; $PY $DIR/paired_eval_compare.py "$O1" "$N1" s1res10_x1 vlact4b_mem | tail -4
echo "===== s1res10 mean of 8 vs VLAct-4B memory"; $PY $DIR/paired_eval_compare.py "$O8" "$N1" s1res10_mean8 vlact4b_mem | tail -4
echo "===== joint plancond (no lag) vs VLAct-4B memory"; $PY $DIR/paired_eval_compare.py "$J" "$N1" joint vlact4b_mem | tail -4
echo "===== staged plancond (no lag) vs VLAct-4B memory"; $PY $DIR/paired_eval_compare.py "$S" "$N1" staged vlact4b_mem | tail -4
echo "[vlact4b-mem] ALL DONE $(date)"
