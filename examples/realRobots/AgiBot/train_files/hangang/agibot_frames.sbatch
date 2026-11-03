#!/bin/bash
# AgiBot World 2026 frames for the episode-memory pretraining (examples/realRobots/AgiBot/tools/agibot_frames.py):
# top_head 448x256 + both hand cameras 312x256 at 10 fps, every episode of flora/index/index_agibot.jsonl, node k on
# shard k of NN -> /scratch/t509a04/data/AgiBotWorld2026/frames448 (index_agibot_{train,test}_<k>of<NN>.jsonl).
# Resumable: a resubmit with the same node count only re-indexes converted episodes.
#   sbatch examples/realRobots/AgiBot/train_files/hangang/agibot_frames.sbatch
#SBATCH -J agibot_frames
#SBATCH -p natStr
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --exclusive
#SBATCH -t 03:00:00
#SBATCH --comment="field=robotics;appl=pytorch"
#SBATCH -o /scratch/t509a04/data/AgiBotWorld2026/logs/%x_%j.out

module load miniconda3 >/dev/null 2>&1
export NN=${SLURM_NNODES}
srun --kill-on-bad-exit=0 bash -c 'OMP_NUM_THREADS=1 /home01/t509a04/envs/starVLA/bin/python \
  /scratch/t509a04/workspace/starVLA_car/examples/realRobots/AgiBot/tools/agibot_frames.py \
  /scratch/t509a04/data/AgiBotWorld2026/flora/index/index_agibot.jsonl /scratch/t509a04/data/AgiBotWorld2026/frames448 \
  --shard ${SLURM_NODEID} --num-shards ${NN} --workers 240 2>&1 | grep -v jemalloc'
