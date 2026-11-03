#!/bin/bash
# EgoDex preprocessing for the memory pretraining on KISTI Hangang: every node converts its share of the episodes to
# 10 fps 448x256 frames and annotates them with Flora subtasks (egodex_prep_node.sh). Outputs:
#   /scratch/t509a04/data/EgoDex/frames448/index_<split>_<k>of<n>.jsonl + <split>/<task>/<N>.npz
#   /scratch/t509a04/data/EgoDex/subtasks_flora/subtasks_n<k>of<n>g<gpu>.jsonl     logs: .../EgoDex/prep_logs/
# Submit from a /scratch directory:  sbatch [--nodes N] examples/realRobots/EgoDex/train_files/hangang/egodex_prep.sbatch
# A resubmit with the same node count resumes.
#SBATCH -J egodex_prep
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
srun --kill-on-bad-exit=0 bash /scratch/t509a04/workspace/starVLA_car/examples/realRobots/EgoDex/train_files/hangang/egodex_prep_node.sh
