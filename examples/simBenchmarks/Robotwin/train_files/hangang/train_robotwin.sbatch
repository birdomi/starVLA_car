#!/bin/bash
# RoboTwin 2.0 training (one yaml) on KISTI Hangang (GH200, 4 GPU/node), torchrun + ZeRO-0 like run_rc100_train_eval.sh.
# Submit from /scratch (repo root):
#   [submit] examples/simBenchmarks/Robotwin/train_files/hangang/train_robotwin.sbatch \
#     examples/simBenchmarks/Robotwin/train_files/starvla_qwen35_2b_gr00t_rt50k_actionmap_v3_robotwin.yaml
# Resume after the 24h walltime: EXTRA_ARGS="--trainer.is_resume True" [submit] ... (same yaml)
#SBATCH -J rt50k
#SBATCH -p natStr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH -t 24:00:00
#SBATCH --comment="field=robotics;appl=deepspeed"
#SBATCH -o /scratch/t509a04/workspace/logs/%x_%j.out

set -eo pipefail

YAML=${1:?usage: sbatch train_robotwin.sbatch <yaml>}
SAVE_INTERVAL=${SAVE_INTERVAL:-5000}   # yaml says 25000; save more often so a 24h timeout loses little
EXTRA_ARGS=${EXTRA_ARGS:-}

REPO=/scratch/t509a04/workspace/starVLA_car

module load miniconda3 cuda/12.8
source activate /home01/t509a04/envs/starVLA

export HF_HUB_OFFLINE=1 RAYON_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=${WANDB_MODE:-disabled}
# torchvision_av decoding warns once per frame read (~5GB of log per 50k run)
export PYTHONWARNINGS="ignore:The video decoding and encoding capabilities of torchvision are deprecated"
export ACCELERATE_DEEPSPEED_ZERO_STAGE=0 OMP_NUM_THREADS=1 STARVLA_MAIN_TORCH_THREADS=8
export PYTHONPATH=${REPO}

cd ${REPO}
echo "[$(date)] $(hostname) yaml=${YAML} save_interval=${SAVE_INTERVAL} extra=${EXTRA_ARGS}"

torchrun --nproc_per_node=4 --master_port=$((20000 + SLURM_JOB_ID % 20000)) \
  starVLA/training/train_starvla.py \
  --config_yaml ${YAML} \
  --trainer.save_interval ${SAVE_INTERVAL} \
  ${EXTRA_ARGS}
