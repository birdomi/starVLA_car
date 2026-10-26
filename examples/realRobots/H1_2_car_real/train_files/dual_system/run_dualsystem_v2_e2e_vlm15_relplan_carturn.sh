#!/bin/bash
# DualSystem on car-turn-signal-v2-ep, JOINT training with the VLM 0.5 s behind C-RADIO / state (joint_e2e_vlm15_relplan_v2_carturn_3cam.yaml,
# 50k, stale_input vlm, System 1 on the fresh plan) -> rollout videos (VLM lag 15 frames) on v2 val episodes 0,1,2,5 (left, left fixed-seat,
# right, right fixed-seat).
# Usage: run_dualsystem_v2_e2e_vlm15_relplan_carturn.sh [gpus=0,1,2,3] [port=29777]   (launch with setsid; do not edit while running)
GPUS=${1:-0,1,2,3}; PORT=${2:-29777}
ROOT=/mnt/DATA2/yyg/starVLA; cd $ROOT
H=$ROOT/examples/realRobots/H1_2_car_real; D=$H/train_files/dual_system
PY=/home/user/miniconda3/envs/starVLA/bin/python; RPY=/home/user/miniconda3/envs/libero310/bin/python
Y=$D/joint_e2e_vlm15_relplan_v2_carturn_3cam.yaml
CK=$ROOT/playground/Checkpoints/starvla_dualsystem_v2_e2e_vlm15_relplan_carturn_3cam/checkpoints/steps_50000_pytorch_model.pt
O=$ROOT/playground/eval_results/dualsystem_v2_e2e_vlm15_relplan_carturn_3cam/rollout_50k
ST=$ROOT/playground/Datasets/car-turn-signal-v2-ep/train/meta/stats_gr00t.abs.json
export STARVLA_CUDA_DECODER_CACHE=150
echo "[$(date '+%m-%d %T')] joint training (System 2 + System 1, v2, VLM lag 0.5 s, previous head command in the state, System 2 delta plan) start"
bash $H/train_files/run_car_yaml.sh $Y $GPUS $PORT
echo "[$(date '+%m-%d %T')] joint training exit"
if [ ! -f "$CK" ]; then echo "checkpoint missing: $CK -> stop"; exit 1; fi
G0=$(echo $GPUS | cut -d, -f1)
echo "[$(date '+%m-%d %T')] rollout on GPU $G0"
CUDA_VISIBLE_DEVICES=$G0 PYTHONPATH=$ROOT RANK=0 $PY $H/rollout_preds_dualsystem.py --ckpt $CK \
  --config $Y --dataset car-turn-signal-v2-ep --robot h1_2_car_turn_dual_s2_nohc --episodes 0,1,2,5 \
  --out $O/dual --out-s2 $O/dual_s2only 2>&1 | grep -E "^ep|Error|Traceback|assert"
echo "[$(date '+%m-%d %T')] rendering"
export MUJOCO_GL=egl
$RPY $H/render_rollout.py $O/dual --stats $ST 2>&1 | tail -1
echo "[$(date '+%m-%d %T')] done"
