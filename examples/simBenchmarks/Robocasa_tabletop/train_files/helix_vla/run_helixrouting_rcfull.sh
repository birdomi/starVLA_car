#!/bin/bash
# [optional: waits for a pid to exit] then, on GPUS once they are free: Helix_Routing_VLA lag12 50k
# (helix_routing_vla_lag12_rcfull_50k.yaml) smoke test (abort on failure) -> training (same launch as
# dual_system/run_dualsystem_joint_rcfull.sh) -> FAST seeded 24-task sim eval (fast_eval_rcfull.sh: one server + one
# 1-thread sim client per task, WPG per GPU at once; identical trajectories to eval_dualsystem_rc100.sh, ~5x faster;
# ROBOCASA_VLM_PREV_FRAME=1) -> paired comparisons vs Helix lag12 (the only difference = depth routing), am5reg, staged,
# joint.
#
#   GPUS=4,5,6,7 setsid nohup bash examples/simBenchmarks/Robocasa_tabletop/train_files/helix_vla/run_helixrouting_rcfull.sh \
#     [pid to wait for] > helixrouting_chain.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../../../../.."   # repo root
WAIT_PID=${1:-}
GPUS=${GPUS:-4,5,6,7}
WPG=${WPG:-6}
HD=examples/simBenchmarks/Robocasa_tabletop/train_files/helix_vla
DD=examples/simBenchmarks/Robocasa_tabletop/train_files/dual_system
YAML=${YAML:-$HD/helix_routing_vla_lag12_rcfull_50k.yaml}
PY=$HOME/miniconda3/envs/starVLA/bin/python
RID=$(grep '^run_id:' "$YAML" | awk '{print $2}')
STEPS=$(grep -E '^\s+max_train_steps:' "$YAML" | awk '{print $2}')
CKPT=playground/Checkpoints/$RID/checkpoints/steps_${STEPS}_pytorch_model.pt
A=playground/Checkpoints/starvla_dualsystem_joint_plancond_kf025_am5reg_h16_rcfull_robocasa_gr1/checkpoints/steps_100000_pytorch_model.pt.log/eval_20261018_032315
H=playground/Checkpoints/starvla_helix_vla_lag12_h16_rcfull_robocasa_gr1/checkpoints/steps_50000_pytorch_model.pt.log/eval_20261020_142413
S=results/eval_logs/starvla_dualsystem_s1plancond_h16_rcfull_robocasa_gr1/eval_20261013_030712
J=results/eval_logs/starvla_dualsystem_joint_plancond_h16_rcfull_robocasa_gr1/eval_20261014_071215

wait_gpus() {  # never start on top of another job: every GPU of the group under 3 GB, 3 checks in a row 1 min apart
    local i=0 ok=0 busy
    while :; do
        busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i "$GPUS" | awk -F', ' '$2 > 3000' | wc -l)
        if [ "$busy" -eq 0 ]; then ok=$((ok + 1)); [ "$ok" -ge 3 ] && break; else ok=0; fi
        [ $((i % 10)) -eq 0 ] && echo "[routing] waiting: $busy GPU(s) of $GPUS above 3 GB $(date)"
        i=$((i + 1)); sleep 60
    done
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i "$GPUS"
}

if [ -n "$WAIT_PID" ]; then
    echo "[routing] waiting for pid $WAIT_PID $(date)"
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi
wait_gpus

echo "[routing] smoke test on GPU ${GPUS%%,*} $(date)"
SLOG=$HD/smoke_routing_$(date +%m%d_%H%M).log
PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=${GPUS%%,*} timeout 3600 $PY $HD/smoke_helix.py "$YAML" > "$SLOG" 2>&1
grep -E "^(=====|built|trainable|step|predict|wrong)" "$SLOG"
if [ "$(grep -c '^predict:' "$SLOG")" -ne 1 ] || [ "$(grep -c 'ValueError OK' "$SLOG")" -ne 1 ] \
   || grep -qE "Traceback|\(bad\)|loss nan" "$SLOG"; then
    echo "[routing] SMOKE FAILED (log $SLOG) - stop $(date)"; tail -30 "$SLOG"; exit 1
fi
echo "[routing] smoke OK $(date)"
wait_gpus

NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)
port=${TRAIN_PORT:-29797}
while ss -ltn | grep -q ":$port "; do port=$((port + 1)); done
log=$PWD/regressor_helix_$(basename "$YAML" .yaml)_$(date +%m%d_%H%M).log
echo "[routing] TRAIN START $RID GPUs=$GPUS port=$port log=$log $(date)"
env PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=$GPUS \
    ACCELERATE_DEEPSPEED_ZERO_STAGE=0 NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=1 STARVLA_MAIN_TORCH_THREADS=8 \
    "$HOME/miniconda3/envs/starVLA/bin/torchrun" --nproc_per_node="$NPROC" --master_port="$port" \
    starVLA/training/train_starvla.py --config_yaml "$YAML" >> "$log" 2>&1
rc=$?
echo "[routing] TRAIN END rc=$rc $(date)"
[ $rc -eq 0 ] || { echo "[routing] training failed - stop"; exit 1; }
sleep 30
[ -f "$CKPT" ] || { echo "[routing] checkpoint missing: $CKPT - stop"; exit 1; }

EV=$CKPT.log/eval_$(date +%Y%m%d_%H%M%S)
TASKS=$(ls "$A" | grep '^eval_env_' | sed 's/eval_env_gr1_unified_//;s/_GR1Arms.*//' | tr '\n' ' ')
echo "[routing] FAST EVAL $(echo $TASKS | wc -w) tasks -> $EV $(date)"
CK=$CKPT EVAL_DIR=$EV GPUS="$(echo "$GPUS" | tr ',' ' ')" WPG=$WPG PORT0=${PORT0:-6960} SERVER_THREADS=1 \
    ROBOCASA_VLM_PREV_FRAME=1 bash $HD/fast_eval_rcfull.sh $TASKS
for f in "$EV"/eval_env_*.log; do grep -cE '^\[episode\]' "$f"; done | sort | uniq -c
grep -h "vlm_prev_frame\]" "$EV"/eval_env_*.log | sort | uniq -c
echo "===== helix lag12 (50k) vs helix routing lag12 (per task)"; $PY $DD/paired_eval_compare.py "$H" "$EV" helix_lag12 helix_routing
echo "===== am5reg DualSystem (49.8%, 100k) vs helix routing"; $PY $DD/paired_eval_compare.py "$A" "$EV" am5reg helix_routing | tail -4
echo "===== staged plancond vs helix routing"; $PY $DD/paired_eval_compare.py "$S" "$EV" staged helix_routing | tail -4
echo "===== joint plancond (50k) vs helix routing"; $PY $DD/paired_eval_compare.py "$J" "$EV" joint helix_routing | tail -4
echo "[routing] ALL DONE $(date)"
