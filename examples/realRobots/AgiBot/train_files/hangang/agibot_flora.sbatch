#!/bin/bash
# Flora subtasks (dex profile: top_head + both hand cameras) of every AgiBot World 2026 episode of
# /scratch/t509a04/data/AgiBotWorld2026/flora/index/index_agibot.jsonl (agibot_flora_node.sh, 4 GPU processes per node)
# -> flora/subtasks_flora_dex/subtasks_s<k>of<NN*4>.jsonl. Resumes: a resubmit with the same node count redoes only
# what is missing or failed on the engine.
#   sbatch examples/realRobots/AgiBot/train_files/hangang/agibot_flora.sbatch
#SBATCH -J agibot_flora
#SBATCH -p natStr
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --exclusive
#SBATCH -t 24:00:00
#SBATCH --comment="field=robotics;appl=pytorch"
#SBATCH -o /scratch/t509a04/data/AgiBotWorld2026/flora/logs/%x_%j.out

module load miniconda3 cuda/12.8
export NN=8
srun --kill-on-bad-exit=0 bash /scratch/t509a04/workspace/starVLA_car/examples/realRobots/AgiBot/train_files/hangang/agibot_flora_node.sh
