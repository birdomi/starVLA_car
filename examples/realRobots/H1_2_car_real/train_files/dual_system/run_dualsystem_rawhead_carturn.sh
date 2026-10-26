#!/bin/bash
# Raw-head DualSystem on car-turn-signal-ep-rawhead: head action = raw head[t+1], 40-D state without head_cmd.
#   stage 1 (System 2, 50k, ColorJitter, stage1_system2_rawhead_carturn_3cam.yaml)
#   -> stage 2 (original-layout System 1, 50-step chunk, keyframe token dropout 0.2, 20k, from stage-1 steps_30000)
#   -> rollout videos on val episodes 33,46,59,45 (ground truth from car-turn-signal-ep-rawhead: head = raw head),
#      compared with the original DualSystem S1 and the ActionMap S1 (their ground-truth head is the recorded staircase
#      action; arms / hands are the same).
# Usage: run_dualsystem_rawhead_carturn.sh [gpus=0,1,2,3] [port=29777]   (launch with setsid; do not edit while running)
GPUS=${1:-0,1,2,3}; PORT=${2:-29777}
ROOT=/mnt/DATA2/yyg/starVLA; cd $ROOT
H=$ROOT/examples/realRobots/H1_2_car_real; D=$H/train_files/dual_system
PY=/home/user/miniconda3/envs/starVLA/bin/python; RPY=/home/user/miniconda3/envs/libero310/bin/python
Y1=$D/stage1_system2_rawhead_carturn_3cam.yaml
Y2=$D/stage2_system1_h50_tdrop_rawhead_carturn_3cam.yaml
CK1=$ROOT/playground/Checkpoints/starvla_dualsystem_rawhead_s2_carturn_3cam/checkpoints/steps_50000_pytorch_model.pt
CK2=$ROOT/playground/Checkpoints/starvla_dualsystem_rawhead_s1h50drop_carturn_3cam/checkpoints/steps_20000_pytorch_model.pt
O=$ROOT/playground/eval_results/dualsystem_rawhead_carturn_3cam/rollout_s1_20k
OLD=$ROOT/playground/eval_results/dualsystem_carturn_3cam/rollout_s1_20k/dual
AM=$ROOT/playground/eval_results/dualsystem_am_carturn_3cam/rollout_s1_20k/dual_am
ST=$ROOT/playground/Datasets/car-turn-signal-ep-rawhead/train/meta/stats_gr00t.abs.json
export STARVLA_CUDA_DECODER_CACHE=150
echo "[$(date '+%m-%d %T')] stage 1 (System 2, raw head) start"
bash $H/train_files/run_car_yaml.sh $Y1 $GPUS $PORT
echo "[$(date '+%m-%d %T')] stage 1 exit"
if [ ! -f "$CK1" ]; then echo "stage-1 checkpoint missing: $CK1 -> stop"; exit 1; fi
echo "[$(date '+%m-%d %T')] stage 2 (System 1, h50 + token dropout) start"
bash $H/train_files/run_car_yaml.sh $Y2 $GPUS $PORT
echo "[$(date '+%m-%d %T')] stage 2 exit"
if [ ! -f "$CK2" ]; then echo "stage-2 checkpoint missing: $CK2 -> stop"; exit 1; fi
G0=$(echo $GPUS | cut -d, -f1)
echo "[$(date '+%m-%d %T')] rollout on GPU $G0"
CUDA_VISIBLE_DEVICES=$G0 PYTHONPATH=$ROOT RANK=0 $PY $H/rollout_preds_dualsystem.py --ckpt $CK2 \
  --config $Y2 --dataset car-turn-signal-ep-rawhead --robot h1_2_car_turn_dual_s2_nohc --episodes 33,46,59,45 \
  --out $O/dual --out-s2 $O/dual_s2only 2>&1 | grep -E "^ep|Error|Traceback|assert"
echo "[$(date '+%m-%d %T')] rendering"
export MUJOCO_GL=egl
$RPY $H/render_rollout.py $O/dual --stats $ST 2>&1 | tail -1
$RPY $H/render_compare.py --a $O/dual --b $OLD --labels "DualSystem rawhead S1 h50+drop 20k,DualSystem S1 20k" --out $O/compare_old_s1 2>&1 | grep -E "compare_ep|all:|Error|Traceback"
$RPY $H/render_compare.py --a $O/dual --b $AM --labels "DualSystem rawhead S1 h50+drop 20k,DualSystem ActionMap-S1 20k" --out $O/compare_am_s1 2>&1 | grep -E "compare_ep|all:|Error|Traceback"
echo "[$(date '+%m-%d %T')] done"
