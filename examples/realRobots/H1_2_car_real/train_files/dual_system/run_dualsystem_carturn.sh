#!/bin/bash
# Chain: stage 1 (System 2, 30k) -> stage 2 (System 1, 20k, from stage-1 steps_30000) on the given GPUs.
# Usage: run_dualsystem_carturn.sh [gpus=0,1,2,3] [port=29773]   (launch with setsid; do not edit while running)
GPUS=${1:-0,1,2,3}; PORT=${2:-29773}
cd /mnt/DATA2/yyg/starVLA
D=examples/realRobots/H1_2_car_real/train_files/dual_system
RUN=examples/realRobots/H1_2_car_real/train_files/run_car_yaml.sh
CK=playground/Checkpoints/starvla_dualsystem_s2_carturn_3cam/checkpoints/steps_30000_pytorch_model.pt
export STARVLA_CUDA_DECODER_CACHE=150
echo "[$(date '+%m-%d %T')] stage 1 start"
bash $RUN $D/stage1_system2_carturn_3cam.yaml $GPUS $PORT
echo "[$(date '+%m-%d %T')] stage 1 exit"
if [ ! -f "$CK" ]; then echo "stage-1 checkpoint missing: $CK -> stop"; exit 1; fi
echo "[$(date '+%m-%d %T')] stage 2 start"
bash $RUN $D/stage2_system1_carturn_3cam.yaml $GPUS $PORT
echo "[$(date '+%m-%d %T')] stage 2 exit"
