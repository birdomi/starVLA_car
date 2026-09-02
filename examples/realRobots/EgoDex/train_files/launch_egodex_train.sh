#!/bin/bash
# Detached launcher for EgoDex QwenGR00T training (GPUs 4-7).
cd /mnt/DATA2/yyg/starVLA
LOG=${1:-/mnt/DATA2/yyg/starVLA/egodex_train_$(date +%m%d_%H%M).log}
setsid nohup env PYTHONPATH=/mnt/DATA2/yyg/starVLA CUDA_VISIBLE_DEVICES=4,5,6,7 \
  /home/user/miniconda3/envs/starVLA/bin/torchrun --nproc_per_node=4 --master_port=29617 \
  starVLA/training/train_starvla.py \
  --config_yaml examples/realRobots/EgoDex/train_files/starvla_qwengr00t_egodex.yaml \
  >> "$LOG" 2>&1 < /dev/null &
echo "launched, log: $LOG"
