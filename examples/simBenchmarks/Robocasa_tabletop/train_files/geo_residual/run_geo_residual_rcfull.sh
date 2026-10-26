#!/bin/bash
# [optional: waits for a pid to exit] then, on GPUS once they are free: QwenGR00T_GeoResidual (YAML=, default
# geo_residual_qwengr00t_rcfull_20k.yaml): needs the frozen System 2 checkpoint (trainer.pretrained_checkpoint) ->
# GPU smoke (smoke_geo_residual.py, abort on failure) -> training -> two FAST seeded 24-task sim evals with the deployment
# schedule (client ROBOCASA_S2_EVERY=12 + n_action_steps 4: System 2 every 12 env steps, residual every 4 on the newest
# frame): residual ON, and OFF (STARVLA_GEORES_OFF=1: same schedule, System 2 only) -> paired comparisons ON vs OFF,
# ON / OFF vs the QwenGR00T System 2 baseline eval (12 actions per call).
#
#   GPUS=0,1,2,3 setsid nohup bash examples/simBenchmarks/Robocasa_tabletop/train_files/geo_residual/run_geo_residual_rcfull.sh \
#     [pid to wait for] > georesidual_chain.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../../../../.."   # repo root
WAIT_PID=${1:-}
GPUS=${GPUS:-0,1,2,3}
WPG=${WPG:-6}
GD=examples/simBenchmarks/Robocasa_tabletop/train_files/geo_residual
DD=examples/simBenchmarks/Robocasa_tabletop/train_files/dual_system
YAML=${YAML:-$GD/geo_residual_qwengr00t_rcfull_20k.yaml}
LBL=${LBL:-$(basename "$YAML" .yaml)}
PY=$HOME/miniconda3/envs/starVLA/bin/python
RID=$(grep '^run_id:' "$YAML" | awk '{print $2}')
STEPS=$(grep -E '^\s+max_train_steps:' "$YAML" | awk '{print $2}')
CKPT=playground/Checkpoints/$RID/checkpoints/steps_${STEPS}_pytorch_model.pt
S2CK=$(grep -E '^\s+pretrained_checkpoint:' "$YAML" | awk '{print $2}')
A=playground/Checkpoints/starvla_dualsystem_joint_plancond_kf025_am5reg_h16_rcfull_robocasa_gr1/checkpoints/steps_100000_pytorch_model.pt.log/eval_20261018_032315

wait_gpus() {  # never start on top of another job: every GPU of the group under 3 GB, 3 checks in a row 1 min apart
    local i=0 ok=0 busy
    while :; do
        busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i "$GPUS" | awk -F', ' '$2 > 3000' | wc -l)
        if [ "$busy" -eq 0 ]; then ok=$((ok + 1)); [ "$ok" -ge 3 ] && break; else ok=0; fi
        [ $((i % 10)) -eq 0 ] && echo "[$LBL] waiting: $busy GPU(s) of $GPUS above 3 GB $(date)"
        i=$((i + 1)); sleep 60
    done
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i "$GPUS"
}

if [ -n "$WAIT_PID" ]; then
    echo "[$LBL] waiting for pid $WAIT_PID $(date)"
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi
[ -f "$S2CK" ] || { echo "[$LBL] System 2 checkpoint missing: $S2CK - stop"; exit 1; }
wait_gpus

echo "[$LBL] smoke test on GPU ${GPUS%%,*} $(date)"
SLOG=$GD/smoke_${LBL}_$(date +%m%d_%H%M).log
PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=${GPUS%%,*} timeout 3600 $PY $GD/smoke_geo_residual.py "$YAML" > "$SLOG" 2>&1
grep -E "^(built|trainable|example|step|predict|GEORES|SMOKE)" "$SLOG"
if ! grep -q "^SMOKE DONE" "$SLOG" || grep -qE "Traceback|loss nan|output == plan: False|tail == plan\[8:16\]: False" "$SLOG"; then
    echo "[$LBL] SMOKE FAILED (log $SLOG) - stop $(date)"; tail -30 "$SLOG"; exit 1
fi
echo "[$LBL] smoke OK $(date)"
wait_gpus

NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)
port=${TRAIN_PORT:-29797}
while ss -ltn | grep -q ":$port "; do port=$((port + 1)); done
log=$PWD/regressor_georesidual_$(basename "$YAML" .yaml)_$(date +%m%d_%H%M).log
echo "[$LBL] TRAIN START $RID GPUs=$GPUS port=$port log=$log $(date)"
env PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=$GPUS \
    ACCELERATE_DEEPSPEED_ZERO_STAGE=0 NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=1 STARVLA_MAIN_TORCH_THREADS=8 \
    "$HOME/miniconda3/envs/starVLA/bin/torchrun" --nproc_per_node="$NPROC" --master_port="$port" \
    starVLA/training/train_starvla.py --config_yaml "$YAML" >> "$log" 2>&1
rc=$?
echo "[$LBL] TRAIN END rc=$rc $(date)"
[ $rc -eq 0 ] || { echo "[$LBL] training failed - stop"; exit 1; }
sleep 30
[ -f "$CKPT" ] || { echo "[$LBL] checkpoint missing: $CKPT - stop"; exit 1; }

TASKS=$(ls "$A" | grep '^eval_env_' | sed 's/eval_env_gr1_unified_//;s/_GR1Arms.*//' | tr '\n' ' ')
EV_ON=$CKPT.log/eval_$(date +%Y%m%d_%H%M%S)_on
echo "[$LBL] FAST EVAL residual ON $(echo $TASKS | wc -w) tasks -> $EV_ON (S2 every 12, residual every 4) $(date)"
CK=$CKPT EVAL_DIR=$EV_ON GPUS="$(echo "$GPUS" | tr ',' ' ')" WPG=$WPG PORT0=${PORT0:-6960} SERVER_THREADS=1 \
    NAS=4 ROBOCASA_S2_EVERY=12 VID_TAG=_on bash $GD/fast_eval_geores.sh $TASKS
EV_OFF=$CKPT.log/eval_$(date +%Y%m%d_%H%M%S)_off
echo "[$LBL] FAST EVAL residual OFF -> $EV_OFF (same schedule, System 2 only) $(date)"
CK=$CKPT EVAL_DIR=$EV_OFF GPUS="$(echo "$GPUS" | tr ',' ' ')" WPG=$WPG PORT0=${PORT0:-6960} SERVER_THREADS=1 \
    NAS=4 ROBOCASA_S2_EVERY=12 VID_TAG=_off STARVLA_GEORES_OFF=1 bash $GD/fast_eval_geores.sh $TASKS
for f in "$EV_ON"/eval_env_*.log "$EV_OFF"/eval_env_*.log; do grep -cE '^\[episode\]' "$f"; done | sort | uniq -c
pcmp() {  # ref_dir new_dir ref_label new_label
    if [ -n "$1" ] && [ -d "$1" ] && [ -d "$2" ]; then
        echo "===== $3 ($1) vs $4 ($2)"; $PY $DD/paired_eval_compare.py "$1" "$2" "$3" "$4" | tail -4
    else
        echo "===== $3 vs $4: eval dir missing - skipped"
    fi
}
B=$(ls -dt "$S2CK.log"/eval_2* 2>/dev/null | head -1)
pcmp "$EV_OFF" "$EV_ON" residual_off residual_on
pcmp "$B" "$EV_ON" qwengr00t_s2 residual_on
pcmp "$B" "$EV_OFF" qwengr00t_s2 residual_off
echo "[$LBL] ALL DONE $(date)"
