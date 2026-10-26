#!/bin/bash
# Episode-memory fine-tune (joint_e2e_vlm15_relplan_holdcap05_mem_ft46k_v2_carturn_3cam.yaml) from the hold-capped fine-tune's
# steps_6000 (= 46k; that run stopped there by user request): 10k steps (outputs labelled mem56k), then in parallel on the 4
# GPUs: stride-30 rollout of all 60 v2 val episodes with the memory updated every 0.5 s from the episode start (continuous,
# GPU 0) and rebuilt from the last 2 s before every call (window = training, GPU 1); eps 0,1,2,5,39,49 continuous + render (GPU 3).
# Usage: run_dualsystem_v2_holdcap05_mem_ft46k_carturn.sh [gpus=0,1,2,3] [port=29779]   (launch with setsid; do not edit while running)
GPUS=${1:-0,1,2,3}; PORT=${2:-29779}
ROOT=/mnt/DATA2/yyg/starVLA; cd $ROOT
H=$ROOT/examples/realRobots/H1_2_car_real; D=$H/train_files/dual_system
PY=/home/user/miniconda3/envs/starVLA/bin/python; RPY=/home/user/miniconda3/envs/libero310/bin/python
Y=$D/joint_e2e_vlm15_relplan_holdcap05_mem_ft46k_v2_carturn_3cam.yaml
PRE=$ROOT/playground/Checkpoints/starvla_dualsystem_v2_e2e_vlm15_relplan_holdcap05_ft40k_carturn_3cam/checkpoints/steps_6000_pytorch_model.pt
CK=$ROOT/playground/Checkpoints/starvla_dualsystem_v2_e2e_vlm15_relplan_holdcap05_mem_ft46k_carturn_3cam/checkpoints/steps_10000_pytorch_model.pt
O=$ROOT/playground/eval_results/dualsystem_v2_e2e_vlm15_relplan_holdcap05_mem_ft46k_carturn_3cam
ST=$ROOT/playground/Datasets/car-turn-signal-v2-ep/train/meta/stats_gr00t.abs.json
if [ ! -f "$PRE" ]; then echo "pretrained checkpoint missing: $PRE -> stop"; exit 1; fi
export STARVLA_CUDA_DECODER_CACHE=150
echo "[$(date '+%m-%d %T')] memory fine-tune from 46k start"
bash $H/train_files/run_car_yaml.sh $Y $GPUS $PORT
echo "[$(date '+%m-%d %T')] memory fine-tune exit"
if [ ! -f "$CK" ]; then echo "checkpoint missing: $CK -> stop"; exit 1; fi
IFS=, read -r G0 G1 G2 G3 <<< "$GPUS"
roll() {  # gpu episodes memory-mode out
  CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$ROOT RANK=0 taskset -c 17-30 $PY $H/rollout_preds_dualsystem.py --ckpt $CK --config $Y \
    --dataset car-turn-signal-v2-ep --robot h1_2_car_turn_dual_s2_nohc --episodes $2 --s1-stride 30 --s2-stride 30 \
    --memory-mode $3 --out $4/dual --out-s2 $4/dual_s2only 2>&1 | grep -E "^ep|Error|Traceback|assert" > $4.log
}
mkdir -p $O
echo "[$(date '+%m-%d %T')] rollouts"
roll $G0 $(seq -s, 0 59) continuous $O/rollout_mem56k_s30_allval &
roll $G1 $(seq -s, 0 59) window $O/rollout_mem56k_s30_allval_window &
(roll $G3 0,1,2,5,39,49 continuous $O/rollout_mem56k_s30; export MUJOCO_GL=egl
 for d in dual dual_s2only; do CUDA_VISIBLE_DEVICES=$G3 taskset -c 17-30 $RPY $H/render_rollout.py $O/rollout_mem56k_s30/$d --stats $ST 2>&1 | tail -1; done) &
wait
echo "[$(date '+%m-%d %T')] done"
