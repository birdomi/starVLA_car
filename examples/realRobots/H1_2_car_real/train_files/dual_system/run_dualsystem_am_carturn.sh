#!/bin/bash
# ActionMap System 1 stage 2 (from the same stage-1 steps_30000 as stage2_system1_carturn_3cam.yaml) -> rollout videos
# on val episodes 33,46,59,45, compared with the original DualSystem S1 and the grouprope ActionMap 50k.
# Usage: run_dualsystem_am_carturn.sh [gpus=0,1,2,3] [port=29775]   (launch with setsid; do not edit while running)
GPUS=${1:-0,1,2,3}; PORT=${2:-29775}
ROOT=/mnt/DATA2/yyg/starVLA; cd $ROOT
H=$ROOT/examples/realRobots/H1_2_car_real; D=$H/train_files/dual_system
PY=/home/user/miniconda3/envs/starVLA/bin/python; RPY=/home/user/miniconda3/envs/libero310/bin/python
CK=$ROOT/playground/Checkpoints/starvla_dualsystem_s1am_carturn_3cam/checkpoints/steps_20000_pytorch_model.pt
O=$ROOT/playground/eval_results/dualsystem_am_carturn_3cam/rollout_s1_20k
OLD=$ROOT/playground/eval_results/dualsystem_carturn_3cam/rollout_s1_20k/dual
G=$ROOT/playground/eval_results/actionmap_grouprope_carturn_0p8b_3cam_abs/rollout_50k/new_carturn
ST=$ROOT/playground/Datasets/car-turn-signal-ep/train/meta/stats_gr00t.abs.json
export STARVLA_CUDA_DECODER_CACHE=150
echo "[$(date '+%m-%d %T')] stage 2 (ActionMap S1) start"
bash $H/train_files/run_car_yaml.sh $D/stage2_system1_actionmap_carturn_3cam.yaml $GPUS $PORT
echo "[$(date '+%m-%d %T')] stage 2 exit"
if [ ! -f "$CK" ]; then echo "checkpoint missing: $CK -> stop"; exit 1; fi
G0=$(echo $GPUS | cut -d, -f1)
echo "[$(date '+%m-%d %T')] rollout on GPU $G0"
CUDA_VISIBLE_DEVICES=$G0 PYTHONPATH=$ROOT RANK=0 $PY $H/rollout_preds_dualsystem.py --ckpt $CK \
  --config $D/stage2_system1_actionmap_carturn_3cam.yaml --episodes 33,46,59,45 \
  --out $O/dual_am --out-s2 $O/dual_am_s2only 2>&1 | grep -E "^ep|Error|Traceback|assert"
echo "[$(date '+%m-%d %T')] rendering"
export MUJOCO_GL=egl
$RPY $H/render_rollout.py $O/dual_am --stats $ST 2>&1 | tail -1
$RPY $H/render_compare.py --a $O/dual_am --b $OLD --labels "DualSystem ActionMap-S1 20k,DualSystem S1 20k" --out $O/compare_old_s1 2>&1 | grep -E "compare_ep|all:|Error|Traceback"
$RPY $H/render_compare.py --a $O/dual_am --b $G --labels "DualSystem ActionMap-S1 20k,ActionMap grouprope 50k" --out $O/compare_grouprope 2>&1 | grep -E "compare_ep|all:|Error|Traceback"
echo "[$(date '+%m-%d %T')] done"
