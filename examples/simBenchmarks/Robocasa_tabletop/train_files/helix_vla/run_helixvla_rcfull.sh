#!/bin/bash
# [optional: waits for a pid to exit] then, on GPUS once they are free: Helix_VLA smoke test on every yaml (abort on
# failure) -> for each yaml in YAMLS (default: lag12 main, lag0 synchronous control, lag12 System-2-only control; 50k
# each) train -> seeded 24-task sim eval via dual_system/run_dualsystem_joint_rcfull.sh (absolute actions, state,
# 256 x 256, 12 of 16 actions per call; ROBOCASA_VLM_PREV_FRAME=1 when helix.s2_lag > 0: the client sends
# [previous call's frame, frame t]) -> paired comparisons on the same scenes vs the am5reg DualSystem (49.8%), staged,
# joint, and (controls) vs the main lag12 run.
#
#   GPUS=4,5,6,7 setsid nohup bash examples/simBenchmarks/Robocasa_tabletop/train_files/helix_vla/run_helixvla_rcfull.sh \
#     [pid to wait for] > helixvla_chain.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../../../../.."   # repo root
WAIT_PID=${1:-}
GPUS=${GPUS:-4,5,6,7}
DD=examples/simBenchmarks/Robocasa_tabletop/train_files/dual_system
HD=examples/simBenchmarks/Robocasa_tabletop/train_files/helix_vla
YAMLS=${YAMLS:-"$HD/helix_vla_lag12_rcfull_50k.yaml $HD/helix_vla_lag0_rcfull_50k.yaml $HD/helix_vla_lag12_s2only_rcfull_50k.yaml"}
SMOKE=${SMOKE:-/tmp/claude-1000/-mnt-DATA2-yyg-starVLA/173021a5-53cc-4501-8767-e288986d7589/scratchpad/helix_vla/smoke_helix.py}
PY=$HOME/miniconda3/envs/starVLA/bin/python
A=playground/Checkpoints/starvla_dualsystem_joint_plancond_kf025_am5reg_h16_rcfull_robocasa_gr1/checkpoints/steps_100000_pytorch_model.pt.log/eval_20261018_032315
S=results/eval_logs/starvla_dualsystem_s1plancond_h16_rcfull_robocasa_gr1/eval_20261013_030712
J=results/eval_logs/starvla_dualsystem_joint_plancond_h16_rcfull_robocasa_gr1/eval_20261014_071215

wait_gpus() {  # never start on top of another job: every GPU of the group under 3 GB, 3 checks in a row 1 min apart
    local i=0 ok=0 busy
    while :; do
        busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i "$GPUS" | awk -F', ' '$2 > 3000' | wc -l)
        if [ "$busy" -eq 0 ]; then ok=$((ok + 1)); [ "$ok" -ge 3 ] && break; else ok=0; fi
        [ $((i % 10)) -eq 0 ] && echo "[helix] waiting: $busy GPU(s) of $GPUS above 3 GB $(date)"
        i=$((i + 1)); sleep 60
    done
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i "$GPUS"
}

if [ -n "$WAIT_PID" ]; then
    echo "[helix] waiting for pid $WAIT_PID $(date)"
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi
wait_gpus

echo "[helix] smoke test on GPU ${GPUS%%,*} $(date)"
SLOG=$HD/smoke_$(date +%m%d_%H%M).log
PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=${GPUS%%,*} timeout 3600 $PY "$SMOKE" $YAMLS > "$SLOG" 2>&1
n_yaml=$(echo $YAMLS | wc -w)
grep -E "^(=====|built|trainable|step|predict|wrong)" "$SLOG"
if [ "$(grep -c '^predict:' "$SLOG")" -ne "$n_yaml" ] || [ "$(grep -c 'ValueError OK' "$SLOG")" -ne "$n_yaml" ] \
   || grep -qE "Traceback|\(bad\)|loss nan" "$SLOG"; then
    echo "[helix] SMOKE FAILED (log $SLOG) - stop $(date)"; tail -30 "$SLOG"; exit 1
fi
echo "[helix] smoke OK $(date)"

MAIN_EVAL=""
for YAML in $YAMLS; do
    wait_gpus
    RID=$(grep '^run_id:' "$YAML" | awk '{print $2}')
    STEPS=$(grep -E '^\s+max_train_steps:' "$YAML" | awk '{print $2}')
    LAG=$(grep -E '^\s+s2_lag:' "$YAML" | awk '{print $2}')
    CKPT=playground/Checkpoints/$RID/checkpoints/steps_${STEPS}_pytorch_model.pt
    unset ROBOCASA_VLM_PREV_FRAME ROBOCASA_VLM_PREV_FRAMES ROBOCASA_ACTION_MODE ROBOCASA_ACTION_MODE_KEYS
    if [ "$LAG" -gt 0 ]; then
        [ "$LAG" -eq 12 ] || { echo "[helix] $RID: s2_lag $LAG != the eval's 12 env steps per call - skip"; continue; }
        export ROBOCASA_VLM_PREV_FRAME=1
    fi
    echo "[helix] ===== $RID (s2_lag $LAG, ROBOCASA_VLM_PREV_FRAME=${ROBOCASA_VLM_PREV_FRAME:-unset}) $(date)"
    JOINT_YAML=$YAML GPUS=$GPUS TRAIN_PORT=${TRAIN_PORT:-29797} bash "$DD/run_dualsystem_joint_rcfull.sh"
    echo "[helix] $RID train + eval chain rc=$? $(date)"
    N1=$(ls -dt "$CKPT.log"/eval_2* 2>/dev/null | head -1)
    [ -n "$N1" ] || { echo "[helix] no eval dir under $CKPT.log"; continue; }
    tag=${RID#starvla_helix_vla_}; tag=${tag%_h16_rcfull_robocasa_gr1}
    echo "===== am5reg DualSystem (49.8%, 100k) vs helix_$tag"; $PY $DD/paired_eval_compare.py "$A" "$N1" am5reg "helix_$tag" | tail -4
    echo "===== staged plancond vs helix_$tag"; $PY $DD/paired_eval_compare.py "$S" "$N1" staged "helix_$tag" | tail -4
    echo "===== joint plancond (50k) vs helix_$tag"; $PY $DD/paired_eval_compare.py "$J" "$N1" joint "helix_$tag" | tail -4
    if [ -z "$MAIN_EVAL" ]; then
        MAIN_EVAL=$N1; MAIN_TAG=$tag
    else
        echo "===== helix_$MAIN_TAG vs helix_$tag"; $PY $DD/paired_eval_compare.py "$MAIN_EVAL" "$N1" "helix_$MAIN_TAG" "helix_$tag" | tail -4
    fi
done
echo "[helix] ALL DONE $(date)"
