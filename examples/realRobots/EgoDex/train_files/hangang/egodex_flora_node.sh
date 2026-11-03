#!/bin/bash
# One node of egodex_flora.sbatch: Flora subtasks (flora_subtasks.py --backend vllm, task description as instruction) of
# this node's share of the episodes (the frames of egodex_prep_node.sh: index_<split>_<NODE>of<NN>.jsonl), one process
# per GPU over a quarter of them: TempSeg, then the Captioner. Each GPU retries the pair up to RETRIES times; both
# stages resume (done episodes skipped, engine-failure rows redone), so a died engine costs only its own work.
# What the first runs taught:
#  * Triton / Inductor / vLLM compile caches per process: shared ones race between processes (KeyError: 'cubin',
#    "Cubin file saved by TritonBundler not found" at engine start)
#  * --gpu-memory-utilization 0.70: the 27B captioner at 0.85 ran out of GPU memory on large video batches
#  * --mem-budget-gb, MALLOC_ARENA_MAX=2 and TMPDIR on /scratch: the captioner decodes every frame of an episode's
#    caption blocks at full resolution at once and writes them as PNGs to TMPDIR (/tmp is RAM here); unbounded, a
#    process reached ~250 GB of host RAM and the node ran out of memory (bounded: ~25 GB, 1.1 s / episode / GPU)
set -o pipefail
NODE=${SLURM_NODEID:-0}
NN=${NN:-${SLURM_NNODES:-1}}   # the node count the frames were converted with (index_<split>_<k>of<NN>.jsonl)
REPO=/scratch/t509a04/workspace/starVLA_car
TOOLS=${REPO}/examples/realRobots/EgoDex/tools
D=/scratch/t509a04/data/EgoDex
SPLITS=test,part1,part2,part3,part4,part5,extra
THREADS=${THREADS:-32}
MEM=${MEM:-0.70}
BUDGET=${BUDGET:-64}
RETRIES=${RETRIES:-6}
LOG=${D}/prep_logs
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn MALLOC_ARENA_MAX=2

echo "[$(date)] node ${NODE}/${NN} $(hostname): flora"
for g in 0 1 2 3; do
  (
    tag=n${NODE}of${NN}g${g}
    for attempt in $(seq 1 ${RETRIES}); do
      ok=1
      for stage in tempseg caption; do
        c=/tmp/cc_${tag}_${attempt}
        mkdir -p /scratch/t509a04/tmp_flora/${tag}
        CUDA_VISIBLE_DEVICES=${g} TRITON_CACHE_DIR=${c}/triton TORCHINDUCTOR_CACHE_DIR=${c}/inductor VLLM_CACHE_ROOT=${c}/vllm \
          TMPDIR=/scratch/t509a04/tmp_flora/${tag} \
          /home01/t509a04/envs/vllm129/bin/python ${TOOLS}/flora_subtasks.py ${D}/frames448 ${D}/zips ${D}/subtasks_flora \
          --split ${SPLITS} --index "index_{split}_${NODE}of${NN}.jsonl" --shard ${g} --num-shards 4 --tag ${tag} \
          --instruction description --backend vllm --stage ${stage} --threads ${THREADS} \
          --gpu-memory-utilization ${MEM} --mem-budget-gb ${BUDGET} --tmp /dev/shm/flora_${tag} || { ok=0; break; }
      done
      [ ${ok} = 1 ] && { echo "[$(date)] ${tag} complete (attempt ${attempt})"; break; }
      echo "[$(date)] ${tag} attempt ${attempt} failed; retrying"
      pkill -9 -u "$(id -u)" -f "VLLM::EngineCore" -P 1 2>/dev/null  # engines left without their process
      sleep 30
    done
  ) >> ${LOG}/flora2_n${NODE}g${g}.log 2>&1 &
done
wait
echo "[$(date)] node ${NODE}: done"
