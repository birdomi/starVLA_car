#!/bin/bash
# H1-2 car training (one yaml) on KISTI Hangang (GH200, 4 GPU/node), accelerate + DeepSpeed ZeRO-2 over 8 nodes (32 GPU).
# Submit from /scratch (repo root):
#   [submit] examples/realRobots/H1_2_car_real/train_files/hangang/train_car.sbatch \
#     examples/realRobots/H1_2_car_real/train_files/starvla_qwen35_4b_gr00t_dit_car1015_v3_3cam_50k.yaml
# Resume after the 24h walltime: EXTRA_ARGS="--trainer.is_resume True" [submit] ... (same yaml)
#SBATCH -J car_dit
#SBATCH -p natStr
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH -t 24:00:00
#SBATCH --comment="field=robotics;appl=deepspeed"
#SBATCH -o /scratch/t509a04/workspace/logs/%x_%j.out

set -eo pipefail

YAML=${1:?usage: sbatch train_car.sbatch <yaml>}
EXTRA_ARGS=${EXTRA_ARGS:-}

REPO=/scratch/t509a04/workspace/starVLA_car

module load miniconda3 cuda/12.8 aws-ofi-nccl/1.20.0 libfabric/2.3.1
source activate /home01/t509a04/envs/starVLA

# Slingshot (CXI) for inter-node NCCL, per Hangang docs; a 1-node job gets no CXI domain, so NVLink only
if [ "${SLURM_NNODES}" -gt 1 ]; then
  export NCCL_NET="AWS Libfabric" FI_PROVIDER=cxi NCCL_IB_DISABLE=1 NCCL_PROTO=^LL128
else
  # the aws-ofi-nccl module itself sets NCCL_NET="AWS Libfabric", whose plugin cannot start without CXI
  # ("Failed to initialize any NET plugin"): no net plugin on one node
  unset NCCL_NET
  export NCCL_NET_PLUGIN=none NCCL_IB_DISABLE=1
fi
export NCCL_DEBUG=${NCCL_DEBUG:-INFO} NCCL_DEBUG_SUBSYS=INIT,NET   # check log for "Using network AWS Libfabric"
export HF_HUB_OFFLINE=1 RAYON_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# torchvision_av decoding warns once per frame read (GBs of log per run)
export PYTHONWARNINGS="ignore:The video decoding and encoding capabilities of torchvision are deprecated"
export WANDB_MODE=${WANDB_MODE:-disabled}
export OMP_NUM_THREADS=1 STARVLA_MAIN_TORCH_THREADS=8
export PYTHONPATH=${REPO}

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export NUM_PROCESSES=$((SLURM_NNODES * 4))
export YAML EXTRA_ARGS

cd ${REPO}
echo "[$(date)] ${SLURM_NNODES} nodes, master ${MASTER_ADDR}:${MASTER_PORT}, yaml=${YAML} extra=${EXTRA_ARGS}"

# One accelerate launcher per node; $SLURM_NODEID is expanded per task inside srun.
# Triton cache per node on local /tmp: 32 ranks compiling into the shared ~/.triton on /home01 race and read
# half-written metadata (JSONDecodeError at the first step).
# Node 0 logs its 4 GPUs (memory / utilization) once a minute next to the job log: <log stem>_gpu.csv.
# Processes / threads: 4 ranks x 16 dataloader workers + Triton / LLVM compile threads sit close to the soft
# RLIMIT_NPROC (16384); a resume died at its first step on "pthread_create failed" (10-30) -> raise it to the hard limit.
srun --kill-on-bad-exit=1 bash -c '
ulimit -u $(ulimit -Hu) 2>/dev/null || true
[ "${SLURM_NODEID}" = 0 ] && echo "[node0] max user processes: $(ulimit -u) (hard $(ulimit -Hu))"
export TRITON_CACHE_DIR=/tmp/triton_${SLURM_JOB_ID}_${SLURM_NODEID}
if [ "${SLURM_NODEID}" = 0 ]; then
  nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv,noheader -l 60 \
    > /scratch/t509a04/workspace/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_gpu.csv &
fi
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_machines ${SLURM_NNODES} \
  --num_processes ${NUM_PROCESSES} \
  --machine_rank ${SLURM_NODEID} \
  --main_process_ip ${MASTER_ADDR} \
  --main_process_port ${MASTER_PORT} \
  starVLA/training/train_starvla.py \
  --config_yaml ${YAML} \
  ${EXTRA_ARGS}
'
