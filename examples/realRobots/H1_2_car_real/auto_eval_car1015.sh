#!/bin/bash
# Wait for steps_<STEP> of a car1015 chunk-policy run (single-frame frameworks, e.g. QwenGR00T; NOT the streamed
# QwenGR00T_EpisodeMemory, whose memory needs whole-episode streaming), then on GPU $GPU (default: freest of 4-7):
#   1. eval_val_chunks.py: 1800 val steps (sorted, fp16), per-task chunk error + hold_cmd / hold_state baselines
#   2. rollout_val_seq.py: tasks 0/1 (left / right turn signal) val episodes, stride 30
#   3. seqcheck/seq_check.py: GT vs the rollout (signal ON -> mirror looks -> head back -> OFF)
# Restored 10-21 into the repo (the scratchpad copy was wiped by a reboot).
# Usage: [GPU=0] [RUN_ID=..] [CFG=..] [ROBOT=..] setsid nohup bash auto_eval_car1015.sh <step> > <log> 2>&1 &
STEP=${1:?step}
ROOT=/mnt/DATA2/yyg/starVLA
RUN_ID=${RUN_ID:-starvla_qwengr00t_dit_car1015_3cam_50k}
CFG=${CFG:-$ROOT/examples/realRobots/H1_2_car_real/train_files/starvla_qwengr00t_dit_car1015_3cam_50k.yaml}
ROBOT=${ROBOT:-h1_2_car_3cam_headstate}
CK=$ROOT/playground/Checkpoints/$RUN_ID/checkpoints/steps_${STEP}_pytorch_model.pt
OUT=$ROOT/playground/eval_results/${RUN_ID#starvla_}
OUT=${OUT/qwengr00t_dit_car1015_3cam_50k/car1015_qwengr00t_dit_3cam_50k}   # the baseline's existing result dir
PY=/home/user/miniconda3/envs/starVLA/bin/python
mkdir -p $OUT
cd $ROOT
echo "[$(date +%T)] waiting for $CK"
while [ ! -f "$CK" ]; do sleep 60; done
prev=-1; while :; do s=$(stat -c %s "$CK"); [ "$s" = "$prev" ] && break; prev=$s; sleep 60; done
echo "[$(date +%T)] checkpoint complete ($((prev / 1024 / 1024)) MB)"
GPU=${GPU:-$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits -i 4,5,6,7 | sort -t, -k2 -nr | head -1 | cut -d, -f1)}
echo "[$(date +%T)] using GPU $GPU"
ENV="CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$ROOT RANK=0 STARVLA_CUDA_DECODER_CACHE=8"
env $ENV nice -n 5 taskset -c 17-30 $PY examples/realRobots/H1_2_car_real/eval_val_chunks.py --ckpt $CK --config $CFG \
  --dataset car-1015-v1-ep --robot $ROBOT --out $OUT/val_${STEP}_fp16.json --dtype float16 --n 1800 --sorted \
  > $OUT/val_${STEP}_fp16.log 2>&1
echo "[$(date +%T)] chunk eval exit $?"
env $ENV nice -n 5 taskset -c 17-30 $PY examples/realRobots/H1_2_car_real/rollout_val_seq.py --ckpt $CK --config $CFG \
  --dataset car-1015-v1-ep --robot $ROBOT --out $OUT/rollout_${STEP}_turnsignal --tasks 0,1 --stride 30 \
  > $OUT/rollout_${STEP}_turnsignal.log 2>&1
echo "[$(date +%T)] rollout exit $?"
SEQCHECK_DATA=$ROOT/playground/eval_results/seqcheck taskset -c 17-30 $PY examples/realRobots/H1_2_car_real/seqcheck/seq_check.py \
  GT=$OUT/rollout_${STEP}_turnsignal ${STEP}=$OUT/rollout_${STEP}_turnsignal > $OUT/seqcheck_${STEP}.txt 2>&1
echo "[$(date +%T)] seq check exit $?"
cp $ROOT/playground/eval_results/seqcheck/seq_check_last.csv $OUT/seqcheck_${STEP}.csv
echo "[$(date +%T)] done"
