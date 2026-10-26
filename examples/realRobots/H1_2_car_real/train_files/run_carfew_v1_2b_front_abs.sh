#!/bin/bash
# car-few-tasks-v1, Qwen3.5-2B, front camera only, abs actions, 200k steps. Launch inside tmux (nice 0).
cd /mnt/DATA2/yyg/starVLA
env PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2,3} NCCL_P2P_DISABLE=1 ACCELERATE_DEEPSPEED_ZERO_STAGE=0 \
  OMP_NUM_THREADS=1 STARVLA_MAIN_TORCH_THREADS=8 \
  /home/user/miniconda3/envs/starVLA/bin/torchrun --nproc_per_node=4 --master_port=${PORT:-29751} \
  starVLA/training/train_starvla.py \
  --config_yaml examples/realRobots/H1_2_car_real/train_files/starvla_qwengr00t_carfew_v1_2b_front_abs.yaml "$@"
