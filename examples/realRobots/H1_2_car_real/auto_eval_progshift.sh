#!/bin/bash
# Wait for a progress-shift checkpoint, then val-evaluate it twice on the freest GPU among 4-7 (CPU video decode):
#   val_<step>_fp16.json           with history ([t-30, t] per camera) -> progress-conditioned model
#   val_<step>_fp16_curronly.json  current frames only -> zero shift = frozen grouprope 100k, same noise (paired)
# Usage: auto_eval_progshift.sh <step> [<step> ...]
ROOT=/mnt/DATA2/yyg/starVLA
RUN_ID=starvla_qwengr00t_actionmap_grouprope_progshift_carturn_0p8b_3cam_abs
CFG=$ROOT/examples/realRobots/H1_2_car_real/train_files/$RUN_ID.yaml
OUT=$ROOT/playground/eval_results/${RUN_ID#starvla_qwengr00t_}
PY=/home/user/miniconda3/envs/starVLA/bin/python
mkdir -p $OUT; cd $ROOT
for STEP in "$@"; do
  CK=$ROOT/playground/Checkpoints/$RUN_ID/checkpoints/steps_${STEP}_pytorch_model.pt
  echo "[$(date +%T)] waiting for $CK"
  while [ ! -f "$CK" ]; do sleep 60; done
  prev=-1; while :; do s=$(stat -c %s "$CK"); [ "$s" = "$prev" ] && break; prev=$s; sleep 60; done
  GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits -i 4,5,6,7 | sort -t, -k2 -nr | head -1 | cut -d, -f1)
  echo "[$(date +%T)] checkpoint complete ($((prev/1024/1024)) MB), GPU $GPU"
  for MODE in "" "_curronly"; do
    FLAG=""; [ "$MODE" = "_curronly" ] && FLAG="--curr_only"
    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$ROOT RANK=0 $PY examples/realRobots/H1_2_car_real/eval_val_chunks.py \
      --ckpt $CK --config $CFG --dataset car-turn-signal-ep --robot h1_2_car_turn_3cam_prev30 \
      --out $OUT/val_${STEP}_fp16${MODE}.json --dtype float16 --n 1800 --video_backend torchcodec $FLAG \
      > $OUT/val_${STEP}_fp16${MODE}.log 2>&1
    echo "[$(date +%T)] eval${MODE:- (history)} exit $?"
  done
done
