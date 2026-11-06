#!/bin/bash
# Multi-VLM episode-memory pretraining (starVLA/training/train_memory_pretrain.py) on KISTI Hangang (GH200, 4 GPU/node).
# Submit from the repo root:
#   sbatch [--nodes N] examples/realRobots/EgoDex/train_files/hangang/train_memory_pretrain.sbatch \
#     examples/realRobots/EgoDex/train_files/memory_pretrain_multivlm_egodex.yaml [key=value ...]
# Resumes from the run's newest checkpoint automatically (trainer.resume: steps_<k>.pt or the rolling latest.pt), so a
# run longer than the 24 h limit continues by submitting the same command again after the job ends. Hangang policy
# (2026-11-05) forbids jobs that resubmit themselves. A job finding <run dir>/DONE (written at max_steps) exits.
#SBATCH -J mem_pretrain
#SBATCH -p natStr
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --exclusive
#SBATCH -t 24:00:00
#SBATCH --comment="field=robotics;appl=pytorch"
#SBATCH -o /scratch/t509a04/workspace/logs/%x_%j.out

set -eo pipefail
YAML=${1:?usage: sbatch train_memory_pretrain.sbatch <yaml> [key=value ...]}
shift
export OVERRIDES="$*"
REPO=/scratch/t509a04/workspace/starVLA_car

module load miniconda3 cuda/12.8 aws-ofi-nccl/1.20.0 libfabric/2.3.1
source activate /home01/t509a04/envs/starVLA
# overlay venv on the starVLA env with transformers 5.7 (Gemma4 >= 5.5, MiniCPM-V-4.6 = 5.7); the base env stays 5.3
export PY=/home01/t509a04/envs/mempre/bin/python
if [ "${SLURM_NNODES}" -gt 1 ]; then
  export NCCL_NET="AWS Libfabric" FI_PROVIDER=cxi NCCL_IB_DISABLE=1 NCCL_PROTO=^LL128
fi
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 PYTHONPATH=${REPO}
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export YAML
cd ${REPO}
echo "[$(date)] ${SLURM_NNODES} nodes, master ${MASTER_ADDR}:${MASTER_PORT}, yaml=${YAML} overrides=${OVERRIDES}"

RUN_DIR=$(${PY} -c "import sys; from omegaconf import OmegaConf as O
c = O.merge(O.load(sys.argv[1]), O.from_dotlist([a.lstrip('-') for a in sys.argv[2:]])); print(f'{c.run_root_dir}/{c.run_id}')" ${YAML} ${OVERRIDES})
if [ -f "${RUN_DIR}/DONE" ]; then
  echo "[$(date)] ${RUN_DIR}/DONE: training finished, nothing to do"; exit 0
fi
# one torchrun per node: all 288 cores to that task (Hangang: srun binds a task to one core unless told otherwise)
srun --kill-on-bad-exit=1 --cpus-per-task=288 bash -c '
export TRITON_CACHE_DIR=/tmp/triton_${SLURM_JOB_ID}_${SLURM_NODEID}
${PY} -m torch.distributed.run --nnodes ${SLURM_NNODES} --nproc_per_node 4 --node_rank ${SLURM_NODEID} \
  --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} \
  starVLA/training/train_memory_pretrain.py --config_yaml ${YAML} ${OVERRIDES}
'
