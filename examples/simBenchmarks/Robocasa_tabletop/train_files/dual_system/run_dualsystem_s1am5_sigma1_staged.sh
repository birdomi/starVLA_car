#!/bin/bash
# Control for the low-noise ActionMap System 1 (run_dualsystem_s1am5_staged.sh): stage2_system1_am5_on_s1res10_50k.yaml =
# the same stage-2 run (s1res10 50k Qwen + C-RADIO + System 2 frozen, new 5-column ActionMap System 1, 10k steps) with the
# default flow source noise N(0, 1) on every column -> seeded 24-task sim eval with the VLM lag, System 1 = one sample
# -> paired comparisons vs the low-noise ActionMap System 1 and the s1res10 System 1 (same frozen System 2).
#
#   GPUS=4,5,6,7 setsid nohup bash examples/simBenchmarks/Robocasa_tabletop/train_files/dual_system/run_dualsystem_s1am5_sigma1_staged.sh \
#     [pid to wait for] > dualsystem_s1am5_sigma1_staged_chain.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../../../../.."   # repo root
WAIT_PID=${1:-}
GPUS=${GPUS:-4,5,6,7}
TRAIN_PORT=${TRAIN_PORT:-29795}
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)
DIR=examples/simBenchmarks/Robocasa_tabletop/train_files/dual_system
YAML=$DIR/stage2_system1_am5_on_s1res10_50k.yaml
RID=$(grep '^run_id:' "$YAML" | awk '{print $2}')
STEPS=$(grep -E '^\s+max_train_steps:' "$YAML" | awk '{print $2}')
CKPT=playground/Checkpoints/$RID/checkpoints/steps_${STEPS}_pytorch_model.pt
TAG=$(echo "${RID#starvla_}" | sed "s/_robocasa_gr1$//")_${STEPS}
PY=$HOME/miniconda3/envs/starVLA/bin/python
if [ -n "$WAIT_PID" ]; then
    echo "[s1am5-sigma1] waiting for pid $WAIT_PID $(date)"
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi

port=$TRAIN_PORT
while ss -ltn | grep -q ":$port "; do port=$((port + 1)); done
log=$PWD/regressor_dualsystem_$(basename "$YAML" .yaml)_$(date +%m%d_%H%M).log
echo "[s1am5-sigma1] TRAIN START $RID GPUs=$GPUS port=$port log=$log $(date)"
env PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=$GPUS \
    ACCELERATE_DEEPSPEED_ZERO_STAGE=0 NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=1 STARVLA_MAIN_TORCH_THREADS=8 \
    "$HOME/miniconda3/envs/starVLA/bin/torchrun" --nproc_per_node="$NPROC" --master_port="$port" \
    starVLA/training/train_starvla.py --config_yaml "$YAML" >> "$log" 2>&1
rc=$?
echo "[s1am5-sigma1] TRAIN END rc=$rc $(date)"
[ $rc -eq 0 ] || { echo "[s1am5-sigma1] training failed - stop"; exit 1; }
sleep 30
[ -f "$CKPT" ] || { echo "[s1am5-sigma1] checkpoint missing: $CKPT - stop"; exit 1; }
EG="$(echo "$GPUS" | tr ',' ' ')"
ROBOCASA_VLM_PREV_FRAME=1 GPUS="$EG" bash "$DIR/eval_dualsystem_rc100.sh" "$CKPT" "$TAG"
echo "[s1am5-sigma1] eval x1 rc=$? $(date)"
N1=$(ls -dt "$CKPT.log"/eval_2* | head -1)
# low-noise ActionMap System 1, 1 sample (run_dualsystem_s1am5_staged.sh)
L1=playground/Checkpoints/starvla_dualsystem_s1am5_lownoise_on_s1res10_50k_h16_rcfull_robocasa_gr1/checkpoints/steps_10000_pytorch_model.pt.log/eval_20261015_150804
O=playground/Checkpoints/starvla_dualsystem_joint_plancond_vlm12_s2delta_s1res10_h16_rcfull_robocasa_gr1/checkpoints/steps_50000_pytorch_model.pt.log
O1=$O/eval_20261015_042524
J=results/eval_logs/starvla_dualsystem_joint_plancond_h16_rcfull_robocasa_gr1/eval_20261014_071215
S=results/eval_logs/starvla_dualsystem_s1plancond_h16_rcfull_robocasa_gr1/eval_20261013_030712
echo "===== low-noise ActionMap S1 vs sigma-1 ActionMap S1, 1 sample"; $PY $DIR/paired_eval_compare.py "$L1" "$N1" am_lownoise_x1 am_sigma1_x1
echo "===== s1res10 S1 vs sigma-1 ActionMap S1, 1 sample"; $PY $DIR/paired_eval_compare.py "$O1" "$N1" s1res10_x1 am_sigma1_x1 | tail -4
echo "===== vs joint plancond (no lag)"; $PY $DIR/paired_eval_compare.py "$J" "$N1" joint am_sigma1_x1 | tail -4
echo "===== vs staged plancond (no lag)"; $PY $DIR/paired_eval_compare.py "$S" "$N1" staged am_sigma1_x1 | tail -4
echo "[s1am5-sigma1] ALL DONE $(date)"
