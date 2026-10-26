#!/bin/bash
# When the grouprope ActionMap run saves its 50k checkpoint: rollout videos of the new model (car-turn-signal-ep
# val: left 33/46 held out from BOTH models, right 59/45) and a side-by-side comparison with the old leftturn_v2_mix
# 100k model on the 3 left-turn episodes neither model trained on (new val 33, 46; old car-left-turn-signal val 4).
set -u
ROOT=/mnt/DATA2/yyg/starVLA; cd $ROOT
PY=/home/user/miniconda3/envs/starVLA/bin/python; RPY=/home/user/miniconda3/envs/libero310/bin/python
CK=$ROOT/playground/Checkpoints/starvla_qwengr00t_actionmap_grouprope_carturn_0p8b_3cam_abs/checkpoints/steps_50000_pytorch_model.pt
CFG=$ROOT/examples/realRobots/H1_2_car_real/train_files/starvla_qwengr00t_actionmap_grouprope_carturn_0p8b_3cam_abs.yaml
OLDB=$ROOT/deploy_bundle_leftturn_100k
O=$ROOT/playground/eval_results/actionmap_grouprope_carturn_0p8b_3cam_abs/rollout_50k
RP=$ROOT/examples/realRobots/H1_2_car_real/rollout_preds.py
echo "[$(date '+%m-%d %T')] waiting for $CK"
while [ ! -f "$CK" ]; do sleep 60; done
prev=-1; while :; do s=$(stat -c %s "$CK"); [ "$s" = "$prev" ] && break; prev=$s; sleep 60; done
G=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits -i 4,5,6,7 | sort -t, -k2 -nr | head -1 | cut -d, -f1)
echo "[$(date '+%m-%d %T')] checkpoint ready, GPU $G"
export CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$ROOT RANK=0
$PY $RP --ckpt $CK --config $CFG --dataset car-turn-signal-ep --robot h1_2_car_turn_3cam --front-cam head_left_2 \
  --episodes 33,46,59,45 --tag-prefix new_ --out $O/new_carturn 2>&1 | grep -E "^new_|Error|Traceback"
$PY $RP --ckpt $CK --config $CFG --dataset car-left-turn-signal --robot h1_2_car_left --stats-dataset car-turn-signal-ep \
  --stats-robot h1_2_car_turn_3cam --front-cam front_view --episodes 4 --tag-prefix new_ --out $O/new_oldval 2>&1 | grep -E "^new_|Error|Traceback"
[ -d $O/old_oldval ] || $PY $RP --ckpt $OLDB/checkpoints/steps_100000_pytorch_model.pt --config $OLDB/config/starvla_qwengr00t_leftturn_v2_mix.yaml \
  --dataset car-left-turn-signal --robot h1_2_car_left --rel-arms --front-cam front_view --episodes 4 --tag-prefix old_ \
  --out $O/old_oldval 2>&1 | grep -E "^old_|Error|Traceback"
echo "[$(date '+%m-%d %T')] rendering"
export MUJOCO_GL=egl
$RPY examples/realRobots/H1_2_car_real/render_rollout.py $O/new_carturn --stats $ROOT/playground/Datasets/car-turn-signal-ep/train/meta/stats_gr00t.abs.json 2>&1 | tail -2
$RPY examples/realRobots/H1_2_car_real/render_compare.py --a $O/new_carturn --b $O/old_carturn --labels "ActionMap 50k,old DiT 100k" --out $O/compare 2>&1 | grep -E "compare_ep|Error|Traceback"
$RPY examples/realRobots/H1_2_car_real/render_compare.py --a $O/new_oldval --b $O/old_oldval --labels "ActionMap 50k,old DiT 100k" --out $O/compare_oldval 2>&1 | grep -E "compare_ep|Error|Traceback"
ls $O/compare/compare_ep*.mp4 $O/compare_oldval/compare_ep*.mp4 | sed "s|^|file '|; s|$|'|" > $O/compare_list.txt
ffmpeg -nostdin -loglevel error -y -f concat -safe 0 -i $O/compare_list.txt -c copy $O/compare_all_3eps.mp4
echo "[$(date '+%m-%d %T')] done: $O/new_carturn/all_tasks.mp4 and $O/compare_all_3eps.mp4"
