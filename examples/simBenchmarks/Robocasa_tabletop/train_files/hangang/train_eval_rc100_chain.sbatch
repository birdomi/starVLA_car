#!/bin/bash
# RoboCasa training longer than the 24 h limit (e.g. 100k steps on 1 node) as a chain of jobs, then the 24-task sim eval.
#   * The first job trains from scratch; each later one resumes (--trainer.is_resume True: weights of the newest steps_N
#     checkpoint, LR schedule fast-forwarded; the optimizer state and the stream memories restart) with the seed shifted
#     by the chain position, so its data streams do not replay the first job's episodes.
#   * A job queues its successor only when it ends, so no job waits in the queue meanwhile: 10 min before the limit
#     (--signal: the batch shell gets USR1), or when training stops without the final checkpoint (a crash), up to
#     CHAIN_LEFT more jobs (default 3).
#   * The job that finds steps_<max_train_steps> runs eval_rc100.sbatch (the memory is carried for episode-memory
#     models) and queues nothing.
# Submit from the repo root:
#   (sbatch) -J <name> examples/simBenchmarks/Robocasa_tabletop/train_files/hangang/train_eval_rc100_chain.sbatch <yaml>
#SBATCH -J rc_chain
#SBATCH -p natStr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH -t 24:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --comment="field=robotics;appl=deepspeed"
#SBATCH -o /scratch/t509a04/workspace/logs/%x_%j.out

YAML=${1:?usage: sbatch train_eval_rc100_chain.sbatch <yaml>}
REPO=/scratch/t509a04/workspace/starVLA_car
HERE=${REPO}/examples/simBenchmarks/Robocasa_tabletop/train_files/hangang
cd ${REPO}

RID=$(grep '^run_id:' "$YAML" | awk '{print $2}')
STEPS=$(grep -E '^\s+max_train_steps:' "$YAML" | awk '{print $2}')
SEED=$(grep -E '^seed:' "$YAML" | awk '{print $2}')
CKDIR=playground/Checkpoints/${RID}/checkpoints
FINAL=${CKDIR}/steps_${STEPS}_pytorch_model.pt
CHAIN_POS=${CHAIN_POS:-0}     # 0 = the first job of the chain
CHAIN_LEFT=${CHAIN_LEFT:-3}
echo "[$(date)] $(hostname) chain position ${CHAIN_POS} (${CHAIN_LEFT} more allowed), yaml=${YAML}"

QUEUED=0
queue_next() {  # once: the next job of the chain, starting after this one ends
  if [ "${QUEUED}" = 1 ] || [ "${CHAIN_LEFT}" -le 0 ] || [ -f "${FINAL}" ]; then return 0; fi
  QUEUED=1
  local i
  for i in $(seq 16); do  # the QOS caps submitted jobs (natStr: 5, pending ones included): retry for ~8 min
    NEXT=$(CHAIN_POS=$((CHAIN_POS + 1)) CHAIN_LEFT=$((CHAIN_LEFT - 1)) EXTRA_ARGS= \
      sbatch --parsable --export=ALL --dependency=afterany:${SLURM_JOB_ID} -J ${SLURM_JOB_NAME} \
      ${HERE}/train_eval_rc100_chain.sbatch ${YAML}) && break
    NEXT=
    sleep 30
  done
  echo "[$(date)] queued the successor ${NEXT:-(none: sbatch kept failing)} (afterany:${SLURM_JOB_ID})"
}
trap queue_next USR1

if [ ! -f "${FINAL}" ]; then
  EXTRA_ARGS=""
  if [ "${CHAIN_POS}" -gt 0 ]; then
    # a job killed at the limit while writing a checkpoint leaves it short: set it aside so the resume takes the previous one
    NEWEST=$(ls -t ${CKDIR}/steps_*_pytorch_model.pt 2>/dev/null | head -1)
    PREV=$(ls -t ${CKDIR}/steps_*_pytorch_model.pt 2>/dev/null | sed -n 2p)
    if [ -n "${PREV}" ] && [ "$(stat -c %s ${NEWEST})" -lt "$(stat -c %s ${PREV})" ]; then
      echo "[$(date)] ${NEWEST} is shorter than ${PREV}: moved aside (.partial)"; mv ${NEWEST} ${NEWEST}.partial
    fi
    EXTRA_ARGS="--trainer.is_resume True --seed $((SEED + CHAIN_POS))"
  fi
  export EXTRA_ARGS
  bash ${HERE}/train_rc100.sbatch "${YAML}" &
  TRAIN=$!
  while kill -0 ${TRAIN} 2>/dev/null; do wait ${TRAIN}; done  # a trapped USR1 interrupts wait: keep waiting
  echo "[$(date)] training exited"
  pkill -f "starVLA/training/train_starvla.py" 2>/dev/null; sleep 30
fi

if [ -f "${FINAL}" ]; then
  bash ${HERE}/eval_rc100.sbatch "${YAML}"
else
  queue_next  # stopped before max_train_steps (a crash): resume in the next job
fi
