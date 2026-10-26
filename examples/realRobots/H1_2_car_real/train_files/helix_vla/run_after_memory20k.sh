#!/bin/bash
# 10-21 (user: "memory 20K 학습이 끝나면 HelixVLA-DINOv3-ConvNext 계열로 50K 학습하자"), on GPUs 4-7:
#  1. wait for the car1015 memory run to exit (its train log's EXIT line) and for the memory 10k all-task eval (GPU 4)
#  2. memory 20k all-task streamed eval split over GPUs 4-7 (eval_val_stream.py --part i/4: the batches / seeds of a
#     full run, ~1/4 of its time), merged, + turn-signal seq check -> playground/eval_results/car1015_stream_s15/memory_20k
#  3. Helix lag15 DINOv3 ConvNeXt-S car1015 50k (helix_vla_lag15_dinov3cnxs3_car1015_3cam_50k.yaml) on GPUs 4-7
#     (b32 measured on synthetic batches: 26.5 / 40.0 GB peak at b8 / b16 -> ~67 GB at b32 + DDP / decoder contexts)
# Launch: setsid nohup bash <this> > <log> 2>&1 &
cd /mnt/DATA2/yyg/starVLA
PY=/home/user/miniconda3/envs/starVLA/bin/python
H=examples/realRobots/H1_2_car_real
E=playground/eval_results/car1015_stream_s15
ML=train_starvla_qwengr00t_dit_episodememory_car1015_3cam_1021_1025.log
CK=playground/Checkpoints/starvla_qwengr00t_dit_episodememory_stream_from30k_car1015_3cam/checkpoints/steps_20000_pytorch_model.pt
CFG=$H/train_files/starvla_qwengr00t_dit_episodememory_car1015_3cam.yaml
HY=$H/train_files/helix_vla/helix_vla_lag15_dinov3cnxs3_car1015_3cam_50k.yaml
log() { echo "[$(date +%T)] $*"; }

log "waiting for the memory run to exit ($ML)"
until grep -q "^EXIT" $ML; do sleep 60; done
log "memory run: $(grep '^EXIT' $ML)"
until grep -qE "done|exit [1-9]" $E/auto_memory_10k_all.log; do sleep 30; done
log "memory 10k all-task eval finished"
while pgrep -f "train_starvla.py --config_yaml $H/train_files/starvla_qwengr00t_dit_episodememory" > /dev/null; do sleep 10; done
sleep 30

if [ -f $CK ]; then
  OUT=$E/memory_20k; mkdir -p $OUT
  CPUS=(17-28 29-40 41-52 53-63)
  for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$((4 + i)) PYTHONPATH=$PWD RANK=0 STARVLA_CUDA_DECODER_CACHE=48 nice -n 5 taskset -c ${CPUS[$i]} \
      $PY $H/eval_val_stream.py --ckpt $CK --config $CFG --robot h1_2_car_3cam_headstate --out $OUT --stride 15 \
      --batch 16 --rollout_tasks all --part $i/4 > $OUT/stream_part$i.log 2>&1 &
  done
  wait
  log "memory 20k eval parts: $(grep -c '^(' $OUT/stream_part*.log | tr '\n' ' ')"
  $PY $H/eval_val_stream.py --ckpt $CK --config $CFG --robot h1_2_car_3cam_headstate --out $OUT --merge_parts 4 > $OUT/stream.log 2>&1
  log "merge exit $?"
  SEQCHECK_DATA=$PWD/playground/eval_results/seqcheck taskset -c 17-30 $PY $H/seqcheck/seq_check.py \
    GT=$OUT/rollout mem20k=$OUT/rollout > $OUT/seqcheck.txt 2>&1
  cp playground/eval_results/seqcheck/seq_check_last.csv $OUT/seqcheck.csv
  log "memory 20k eval done"
else
  log "no $CK: memory 20k eval skipped"
fi

log "launching Helix DINOv3 car1015 50k on GPUs 4-7"
STARVLA_CUDA_DECODER_CACHE=16 setsid nohup bash $H/train_files/run_car_yaml.sh $HY 4,5,6,7 29821 < /dev/null > /dev/null 2>&1 &
sleep 5
log "launched: $(ls -t train_helix_vla_lag15_dinov3cnxs3_car1015_3cam_50k_*.log 2>/dev/null | head -1)"
