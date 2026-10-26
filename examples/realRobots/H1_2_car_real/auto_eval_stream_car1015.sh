#!/bin/bash
# Streamed val eval (eval_val_stream.py: every val episode from frame 0, one query per STRIDE frames, memory carried for
# QwenGR00T_EpisodeMemory) + seqcheck on the turn-signal episodes, for one checkpoint; waits for it to be written.
# Usage: [GPU=4] [STRIDE=15] [BATCH=16] [ROLLOUT=0,1 | all] setsid nohup bash auto_eval_stream_car1015.sh <ckpt.pt> <config.yaml> <out_dir> > <log> 2>&1 &
CK=${1:?ckpt}; CFG=${2:?config}; OUT=${3:?out dir}
ROOT=/mnt/DATA2/yyg/starVLA; PY=/home/user/miniconda3/envs/starVLA/bin/python
STRIDE=${STRIDE:-15}; BATCH=${BATCH:-16}; ROLLOUT=${ROLLOUT:-0,1}; NAME=${NAME:-$(basename "$CK" _pytorch_model.pt)}
cd $ROOT; mkdir -p $OUT
echo "[$(date +%T)] waiting for $CK"
while [ ! -f "$CK" ]; do sleep 60; done
prev=-1; while :; do s=$(stat -c %s "$CK"); [ "$s" = "$prev" ] && break; prev=$s; sleep 60; done
GPU=${GPU:-$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits -i 4,5,6,7 | sort -t, -k2 -nr | head -1 | cut -d, -f1)}
echo "[$(date +%T)] checkpoint complete ($((prev / 1024 / 1024)) MB), GPU $GPU"
# one process, BATCH episodes x 3 cameras of decoders: keep the cache just above that (the 10-21 crash came from
# hundreds of cached decoders per GPU across dataloader workers)
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$ROOT RANK=0 STARVLA_CUDA_DECODER_CACHE=$((BATCH * 3)) nice -n 5 taskset -c 17-30 \
  $PY examples/realRobots/H1_2_car_real/eval_val_stream.py --ckpt $CK --config $CFG --robot h1_2_car_3cam_headstate \
  --out $OUT --stride $STRIDE --batch $BATCH --rollout_tasks $ROLLOUT > $OUT/stream.log 2>&1
echo "[$(date +%T)] stream eval exit $?"
SEQCHECK_DATA=$ROOT/playground/eval_results/seqcheck taskset -c 17-30 $PY examples/realRobots/H1_2_car_real/seqcheck/seq_check.py \
  GT=$OUT/rollout $NAME=$OUT/rollout > $OUT/seqcheck.txt 2>&1
echo "[$(date +%T)] seq check exit $?"
cp $ROOT/playground/eval_results/seqcheck/seq_check_last.csv $OUT/seqcheck.csv
echo "[$(date +%T)] done"
