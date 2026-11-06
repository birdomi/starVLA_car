#!/bin/bash
# RoboCasa train -> 24-task sim eval of the final checkpoint in ONE job (one QOS submit slot instead of two; natStr allows 5).
# Runs train_rc100.sbatch then eval_rc100.sbatch inside this allocation (~9-13h train + ~4h eval < 24h walltime).
# The eval runs even when training exits non-zero: Hangang runs finish all steps and save the last checkpoint, then die
# with an NCCL error at teardown (so an afterok chain never fires); eval_rc100.sbatch exits if the checkpoint is missing.
#   [submit] -J <name> examples/simBenchmarks/Robocasa_tabletop/train_files/hangang/train_eval_rc100.sbatch <yaml>
#SBATCH -J rc_train_eval
#SBATCH -p natStr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH -t 24:00:00
#SBATCH --comment="field=robotics;appl=deepspeed"
#SBATCH -o /scratch/t509a04/workspace/logs/%x_%j.out

YAML=${1:?usage: sbatch train_eval_rc100.sbatch <yaml>}
HERE=/scratch/t509a04/workspace/starVLA_car/examples/simBenchmarks/Robocasa_tabletop/train_files/hangang

bash ${HERE}/train_rc100.sbatch "$YAML"
echo "[$(date)] training exited rc=$? -> eval"
# torchrun can leave stray workers after the teardown error; free the GPUs before the policy servers start
pkill -f "starVLA/training/train_starvla.py" 2>/dev/null; sleep 30
bash ${HERE}/eval_rc100.sbatch "$YAML"
