#!/bin/bash
# Watches egodex_prep.sbatch's Flora processes (one per node x GPU) from the login node and reruns the caption stage
# of any whose output holds engine-error rows (a vLLM engine that died, e.g. CUDA OOM) once that process has ended:
# as an --overlap step on the same node and GPU, with a lower --gpu-memory-utilization. The rerun skips the episodes
# already annotated and redoes the engine-error ones (flora_subtasks.py read_paths); it stops on a new engine death,
# so the next pass tries again.
#   nohup bash flora_watchdog.sh <prep job id> > .../prep_logs/watchdog.log 2>&1 &
JOB=${1:?usage: flora_watchdog.sh <egodex_prep job id>}
REPO=/scratch/t509a04/workspace/starVLA_car
TOOLS=${REPO}/examples/realRobots/EgoDex/tools
D=/scratch/t509a04/data/EgoDex
LOG=${D}/prep_logs
OUT=${D}/subtasks_flora
SPLITS="test,part1,part2,part3,part4,part5,extra"
MEM=${MEM:-0.75}
NN=$(squeue -j ${JOB} -h -o %D)
mapfile -t HOSTS < <(scontrol show hostnames "$(squeue -j ${JOB} -h -o %N)")

needs_repair() {  # $1 node $2 gpu: the shard's process ended and its output still holds engine-error rows
  local tag=n$1of${NN}g$2 orig=${LOG}/flora_n$1g$2.log rep=${LOG}/flora_repair_n$1g$2.log
  [ -f ${LOG}/repair_n$1g$2.pid ] && kill -0 "$(cat ${LOG}/repair_n$1g$2.pid)" 2>/dev/null && return 1  # repair running
  if [ -f ${rep} ]; then grep -q "done:" ${rep} || [ "$(tail -c 300 ${rep} | grep -c EngineDead)" -gt 0 ] || return 1
  else grep -q "\[flora\] ${tag} done:" ${orig} 2>/dev/null || return 1; fi
  grep -q '"error": "[^"]*\(EngineDeadError\|OutOfMemoryError\|EngineCore\)' ${OUT}/subtasks_${tag}.jsonl 2>/dev/null || return 1
  # an earlier repair that finished cleanly appended good rows after the bad ones: compare counts of paths
  python3 - "${OUT}/subtasks_${tag}.jsonl" <<'EOF' || return 1
import json, sys
last = {}
for l in open(sys.argv[1]):
    try:
        r = json.loads(l)
    except json.JSONDecodeError:
        continue
    last[r["path"]] = r
bad = sum(any(k in (r.get("error") or "") for k in ("EngineDeadError", "OutOfMemoryError", "EngineCore")) for r in last.values())
sys.exit(0 if bad else 1)
EOF
}

echo "[$(date)] watching job ${JOB} (${NN} nodes: ${HOSTS[*]})"
while squeue -j ${JOB} -h | grep -q " R "; do
  for ((k = 0; k < NN; k++)); do
    for g in 0 1 2 3; do
      if needs_repair ${k} ${g}; then
        echo "[$(date)] repair n${k}g${g} on ${HOSTS[$k]} (gpu_memory_utilization ${MEM})"
        setsid nohup srun --jobid=${JOB} --overlap -w ${HOSTS[$k]} -N1 -n1 --cpus-per-task=32 --mem=200G bash -c "
          export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=${g}
          cd ${REPO}
          /home01/t509a04/envs/vllm129/bin/python ${TOOLS}/flora_subtasks.py ${D}/frames448 ${D}/zips ${OUT} \
            --split ${SPLITS} --index 'index_{split}_${k}of${NN}.jsonl' --shard ${g} --num-shards 4 --tag n${k}of${NN}g${g} \
            --instruction description --backend vllm --stage caption --threads 64 --gpu-memory-utilization ${MEM} \
            --tmp /dev/shm/flora_repair_n${k}g${g}" >> ${LOG}/flora_repair_n${k}g${g}.log 2>&1 < /dev/null &
        echo $! > ${LOG}/repair_n${k}g${g}.pid
      fi
    done
  done
  sleep 300
done
echo "[$(date)] job ${JOB} ended"
