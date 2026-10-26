#!/bin/bash
# After the s1res10 chain (joint_system12_plancond_vlm12_s2delta_s1res10_rcfull_50k.yaml: train 50k + seeded 24-task sim
# eval with the VLM lag, System 1 = ONE sample per call) has finished: the SAME checkpoint and scenes with System 1 = the
# MEAN of K samples on the same plan (STARVLA_DUAL_S1_NUM_SAMPLES=K -> inference.s1_num_samples, inference only), then a
# paired comparison (paired_eval_compare.py) of 1 sample vs K, and of both vs the joint / staged plancond baselines.
#
#   setsid nohup bash examples/simBenchmarks/Robocasa_tabletop/train_files/dual_system/run_eval_s1mean_after_chain.sh \
#     <chain bash pid> <chain log> > dualsystem_s1res10_s1mean8_eval_chain.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../../../../.."   # repo root
CHAIN_PID=${1:?chain pid}; CHAIN_LOG=${2:?chain log}; K=${K:-8}
DIR=examples/simBenchmarks/Robocasa_tabletop/train_files/dual_system
YAML=$DIR/joint_system12_plancond_vlm12_s2delta_s1res10_rcfull_50k.yaml
RID=$(grep '^run_id:' "$YAML" | awk '{print $2}')
CKPT=playground/Checkpoints/$RID/checkpoints/steps_50000_pytorch_model.pt
TAG1=$(echo "${RID#starvla_}" | sed "s/_robocasa_gr1$//")_50000
PY=$HOME/miniconda3/envs/starVLA/bin/python
echo "[s1mean] waiting for chain pid $CHAIN_PID ($CHAIN_LOG) $(date)"
while kill -0 "$CHAIN_PID" 2>/dev/null; do sleep 120; done
grep -q "ALL DONE" "$CHAIN_LOG" && [ -f "results/robocasa_rc100_${TAG1}_eval_results.csv" ] || {
    echo "[s1mean] chain did not finish its 1-sample eval - stop $(date)"; tail -3 "$CHAIN_LOG"; exit 1; }
L1=$(ls -dt "$CKPT.log"/eval_2* | head -1)
echo "[s1mean] 1-sample eval: $L1 ($(tail -1 "$CHAIN_LOG"))"
STARVLA_DUAL_S1_NUM_SAMPLES=$K ROBOCASA_VLM_PREV_FRAME=1 GPUS="4 5 6 7" bash "$DIR/eval_dualsystem_rc100.sh" "$CKPT" "${TAG1}_s1mean$K"
echo "[s1mean] eval rc=$? $(date)"
LK=$(ls -dt "$CKPT.log"/eval_2* | head -1)
[ "$LK" != "$L1" ] || { echo "[s1mean] no new eval log dir - stop"; exit 1; }
grep -l "mean of $K samples" "$LK"/server_gpu*.log > /dev/null 2>&1 && echo "[s1mean] servers confirm mean of $K samples" \
    || echo "[s1mean] WARNING: server logs do not show 'mean of $K samples'"
J=results/eval_logs/starvla_dualsystem_joint_plancond_h16_rcfull_robocasa_gr1/eval_20261014_071215
S=results/eval_logs/starvla_dualsystem_s1plancond_h16_rcfull_robocasa_gr1/eval_20261013_030712
echo "===== 1 sample vs mean of $K (same ckpt, same scenes)"; $PY $DIR/paired_eval_compare.py "$L1" "$LK" s1x1 "s1mean$K"
echo "===== vs joint plancond (no lag, abs plan)"; $PY $DIR/paired_eval_compare.py "$J" "$L1" joint s1x1 | tail -4
$PY $DIR/paired_eval_compare.py "$J" "$LK" joint "s1mean$K" | tail -4
echo "===== vs staged plancond"; $PY $DIR/paired_eval_compare.py "$S" "$L1" staged s1x1 | tail -4
$PY $DIR/paired_eval_compare.py "$S" "$LK" staged "s1mean$K" | tail -4
echo "[s1mean] ALL DONE $(date)"
