#!/bin/bash
# DualSystem on car-turn-signal-v2-ep (prepare_car_v2_dataset.py): head action = raw head[t], 40-D state without
# head_cmd, all 1143 / 60 episodes, 2 instructions without " (fixed seat)".
#   stage 1 (System 2, 50k, ColorJitter, stage1_system2_v2_carturn_3cam.yaml)
#   -> stage 2 (original-layout System 1, 50-step chunk, keyframe token dropout 0.2, 20k, from stage-1 steps_50000)
#   -> rollout videos on v2 val episodes 0,1,2,5 (left, left fixed-seat, right, right fixed-seat). The v1 rollouts
#      (val 33,46,59,45) are v2 TRAIN episodes, so there is no side-by-side with the earlier runs.
# Usage: run_dualsystem_v2_carturn.sh [gpus=0,1,2,3] [port=29777]   (launch with setsid; do not edit while running)
GPUS=${1:-0,1,2,3}; PORT=${2:-29777}
ROOT=/mnt/DATA2/yyg/starVLA; cd $ROOT
H=$ROOT/examples/realRobots/H1_2_car_real; D=$H/train_files/dual_system
PY=/home/user/miniconda3/envs/starVLA/bin/python; RPY=/home/user/miniconda3/envs/libero310/bin/python
Y1=$D/stage1_system2_v2_carturn_3cam.yaml
Y2=$D/stage2_system1_h50_tdrop_v2_carturn_3cam.yaml
CK1=$ROOT/playground/Checkpoints/starvla_dualsystem_v2_s2_carturn_3cam/checkpoints/steps_50000_pytorch_model.pt
CK2=$ROOT/playground/Checkpoints/starvla_dualsystem_v2_s1h50drop_carturn_3cam/checkpoints/steps_20000_pytorch_model.pt
O=$ROOT/playground/eval_results/dualsystem_v2_carturn_3cam/rollout_s1_20k
ST=$ROOT/playground/Datasets/car-turn-signal-v2-ep/train/meta/stats_gr00t.abs.json
export STARVLA_CUDA_DECODER_CACHE=150
echo "[$(date '+%m-%d %T')] stage 1 (System 2, v2) start"
bash $H/train_files/run_car_yaml.sh $Y1 $GPUS $PORT
echo "[$(date '+%m-%d %T')] stage 1 exit"
if [ ! -f "$CK1" ]; then echo "stage-1 checkpoint missing: $CK1 -> stop"; exit 1; fi
echo "[$(date '+%m-%d %T')] stage 2 (System 1, h50 + token dropout, v2) start"
bash $H/train_files/run_car_yaml.sh $Y2 $GPUS $PORT
echo "[$(date '+%m-%d %T')] stage 2 exit"
if [ ! -f "$CK2" ]; then echo "stage-2 checkpoint missing: $CK2 -> stop"; exit 1; fi
G0=$(echo $GPUS | cut -d, -f1)
echo "[$(date '+%m-%d %T')] rollout on GPU $G0"
CUDA_VISIBLE_DEVICES=$G0 PYTHONPATH=$ROOT RANK=0 $PY $H/rollout_preds_dualsystem.py --ckpt $CK2 \
  --config $Y2 --dataset car-turn-signal-v2-ep --robot h1_2_car_turn_dual_s2_nohc --episodes 0,1,2,5 \
  --out $O/dual --out-s2 $O/dual_s2only 2>&1 | grep -E "^ep|Error|Traceback|assert"
echo "[$(date '+%m-%d %T')] rendering"
export MUJOCO_GL=egl
$RPY $H/render_rollout.py $O/dual --stats $ST 2>&1 | tail -1
echo "[$(date '+%m-%d %T')] done"
