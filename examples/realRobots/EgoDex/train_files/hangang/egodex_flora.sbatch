#!/bin/bash
# Flora subtasks of every EgoDex episode (egodex_flora_node.sh), after egodex_prep.sbatch converted the frames. Run on as
# many nodes as the frames were converted with (their index shards, NN): node k annotates index_<split>_<k>of<NN>.
# Resumes: a resubmit redoes only what is missing or failed on the engine.
#   sbatch examples/realRobots/EgoDex/train_files/hangang/egodex_flora.sbatch          (from a /scratch directory)
#SBATCH -J egodex_flora
#SBATCH -p natStr
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --exclusive
#SBATCH -t 24:00:00
#SBATCH --comment="field=robotics;appl=pytorch"
#SBATCH -o /scratch/t509a04/workspace/logs/%x_%j.out

module load miniconda3 cuda/12.8
export NN=16
srun --kill-on-bad-exit=0 bash /scratch/t509a04/workspace/starVLA_car/examples/realRobots/EgoDex/train_files/hangang/egodex_flora_node.sh
