#!/bin/bash
# Streamed episode-memory fine-tune (joint_e2e_vlm15_relplan_holdcap05_memstream_v2_carturn_3cam.yaml: memory carried
# across training steps as deployed) from the window memory run's steps_10000 (= 56k): waits for that run's chain to
# finish, trains 10k steps (outputs labelled memstream66k), then stride-30 rollouts with the memory updated every 0.5 s
# from the episode start (continuous = deployment) of all 60 v2 val episodes (GPUs 0-2, 20 each) and of eps 0,1,2,5,39,49
# + render (GPU 3), then the sequence / re-activation checks and the progress error vs the 48k / 56k rollouts.
# Usage: run_dualsystem_v2_holdcap05_memstream_carturn.sh [gpus=0,1,2,3] [port=29780]   (launch with setsid; do not edit while running)
GPUS=${1:-0,1,2,3}; PORT=${2:-29780}
ROOT=/mnt/DATA2/yyg/starVLA; cd $ROOT
H=$ROOT/examples/realRobots/H1_2_car_real; D=$H/train_files/dual_system
PY=/home/user/miniconda3/envs/starVLA/bin/python; RPY=/home/user/miniconda3/envs/libero310/bin/python
Y=$D/joint_e2e_vlm15_relplan_holdcap05_memstream_v2_carturn_3cam.yaml
PRE=$ROOT/playground/Checkpoints/starvla_dualsystem_v2_e2e_vlm15_relplan_holdcap05_mem_ft46k_carturn_3cam/checkpoints/steps_10000_pytorch_model.pt
CK=$ROOT/playground/Checkpoints/starvla_dualsystem_v2_e2e_vlm15_relplan_holdcap05_memstream_carturn_3cam/checkpoints/steps_10000_pytorch_model.pt
O=$ROOT/playground/eval_results/dualsystem_v2_e2e_vlm15_relplan_holdcap05_memstream_carturn_3cam
OM=$ROOT/playground/eval_results/dualsystem_v2_e2e_vlm15_relplan_holdcap05_mem_ft46k_carturn_3cam
ST=$ROOT/playground/Datasets/car-turn-signal-v2-ep/train/meta/stats_gr00t.abs.json
PREV=$ROOT/playground/dualsystem_v2_holdcap05_mem_ft46k_carturn_chain.log
SB=/tmp/claude-1000/-mnt-DATA2-yyg-starVLA/3429367a-a321-47c8-ab05-731243e5181a/scratchpad
echo "[$(date '+%m-%d %T')] waiting for the window memory chain ($PREV) to finish"
until grep -qE "\] done$|checkpoint missing" $PREV; do sleep 60; done
if [ ! -f "$PRE" ]; then echo "pretrained checkpoint missing: $PRE -> stop"; exit 1; fi
export STARVLA_CUDA_DECODER_CACHE=150
echo "[$(date '+%m-%d %T')] streamed memory fine-tune from 56k start"
bash $H/train_files/run_car_yaml.sh $Y $GPUS $PORT
echo "[$(date '+%m-%d %T')] streamed memory fine-tune exit"
if [ ! -f "$CK" ]; then echo "checkpoint missing: $CK -> stop"; exit 1; fi
IFS=, read -r G0 G1 G2 G3 <<< "$GPUS"
roll() {  # gpu episodes out
  CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$ROOT RANK=0 taskset -c 17-30 $PY $H/rollout_preds_dualsystem.py --ckpt $CK --config $Y \
    --dataset car-turn-signal-v2-ep --robot h1_2_car_turn_dual_s2_nohc --episodes $2 --s1-stride 30 --s2-stride 30 \
    --memory-mode continuous --out $3/dual --out-s2 $3/dual_s2only 2>&1 | grep -E "^ep|Error|Traceback|assert" > $4
}
mkdir -p $O
echo "[$(date '+%m-%d %T')] rollouts"
roll $G0 $(seq -s, 0 19) $O/rollout_memstream66k_s30_allval $O/rollout_memstream66k_s30_allval.part0.log &
roll $G1 $(seq -s, 20 39) $O/rollout_memstream66k_s30_allval $O/rollout_memstream66k_s30_allval.part1.log &
roll $G2 $(seq -s, 40 59) $O/rollout_memstream66k_s30_allval $O/rollout_memstream66k_s30_allval.part2.log &
(roll $G3 0,1,2,5,39,49 $O/rollout_memstream66k_s30 $O/rollout_memstream66k_s30.log; export MUJOCO_GL=egl
 for d in dual dual_s2only; do CUDA_VISIBLE_DEVICES=$G3 taskset -c 17-30 $RPY $H/render_rollout.py $O/rollout_memstream66k_s30/$d --stats $ST 2>&1 | tail -1; done) &
wait
echo "[$(date '+%m-%d %T')] checks"
A=$O/rollout_memstream66k_s30_allval/dual
(cd $SB/seqcheck && $PY seq_check.py GT=$A cont48k=$OM/rollout_mem48k_s30_allval/dual cont56k=$OM/rollout_mem56k_s30_allval/dual win56k=$OM/rollout_mem56k_s30_allval_window/dual stream66k=$A > $O/seq_check.txt 2>&1; cp seq_check_last.csv $O/seq_check.csv
 $PY reactivate_check.py cont56k=$OM/rollout_mem56k_s30_allval/dual stream66k=$A > $O/reactivate_check.txt 2>&1)
$PY $SB/memory/cmp_modes.py cont48k=$OM/rollout_mem48k_s30_allval/dual win48k=$OM/rollout_mem48k_s30_allval_window/dual \
  cont56k=$OM/rollout_mem56k_s30_allval/dual win56k=$OM/rollout_mem56k_s30_allval_window/dual stream66k=$A > $O/progress_err.txt 2>&1
echo "[$(date '+%m-%d %T')] done"
