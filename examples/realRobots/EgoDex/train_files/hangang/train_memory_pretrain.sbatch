#!/bin/bash
# Multi-VLM episode-memory pretraining (starVLA/training/train_memory_pretrain.py) on KISTI Hangang (GH200, 4 GPU/node).
# Submit from the repo root:
#   sbatch [--nodes N] examples/realRobots/EgoDex/train_files/hangang/train_memory_pretrain.sbatch \
#     examples/realRobots/EgoDex/train_files/memory_pretrain_multivlm_egodex.yaml [key=value ...]
# Resumes from the run's newest checkpoint automatically (trainer.resume: steps_<k>.pt or the rolling latest.pt).
# Chain: a job queues its successor only when it ends, so no job waits in the queue meanwhile: 10 min before the 24 h
# limit (--signal: the batch shell gets USR1), or when training stops earlier (a crash; training jobs end FAILED at
# NCCL teardown even when they finish). The successor depends on afterany:<this job>, resumes from the newest
# checkpoint, and so on for up to CHAIN_LEFT more jobs (default 6); a job finding <run dir>/DONE (written at max_steps)
# exits and queues nothing. CHAIN_LEFT=0 sbatch ... for one job.
#SBATCH -J mem_pretrain
#SBATCH -p natStr
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH -t 24:00:00
#SBATCH --comment="field=robotics;appl=pytorch"
#SBATCH -o /scratch/t509a04/workspace/logs/%x_%j.out
#SBATCH --signal=B:USR1@600

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
CHAIN_LEFT=${CHAIN_LEFT:-6}
QUEUED=0
queue_next() {  # once: the next job of the chain, starting after this one ends
  if [ "${QUEUED}" = 1 ] || [ "${CHAIN_LEFT}" -le 0 ] || [ -f "${RUN_DIR}/DONE" ]; then return 0; fi
  QUEUED=1
  local i
  for i in $(seq 16); do  # the QOS caps submitted jobs (natStr: 5, pending ones included): retry for ~8 min
    NEXT=$(cd ${REPO} && CHAIN_LEFT=$((CHAIN_LEFT - 1)) sbatch --parsable --export=ALL --dependency=afterany:${SLURM_JOB_ID} \
      ${REPO}/examples/realRobots/EgoDex/train_files/hangang/train_memory_pretrain.sbatch ${YAML} ${OVERRIDES}) && break
    NEXT=
    sleep 30
  done
  echo "[$(date)] queued the successor ${NEXT:-(none: sbatch kept failing)} (afterany:${SLURM_JOB_ID}, ${CHAIN_LEFT} left before it)"
}
trap queue_next USR1

set +e
srun --kill-on-bad-exit=1 bash -c '
export TRITON_CACHE_DIR=/tmp/triton_${SLURM_JOB_ID}_${SLURM_NODEID}
${PY} -m torch.distributed.run --nnodes ${SLURM_NNODES} --nproc_per_node 4 --node_rank ${SLURM_NODEID} \
  --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} \
  starVLA/training/train_memory_pretrain.py --config_yaml ${YAML} ${OVERRIDES}
' &
SRUN=$!
while kill -0 ${SRUN} 2>/dev/null; do wait ${SRUN}; done  # a trapped USR1 interrupts wait: keep waiting
wait ${SRUN}
RC=$?
queue_next  # stopped before the limit (a crash, or max_steps: DONE then queues nothing)
exit ${RC}
