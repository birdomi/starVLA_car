#!/bin/bash
# One node of agibot_flora.sbatch: Flora subtasks (flora_subtasks.py --backend vllm --profile dex: top_head above both
# hand cameras, the episode instruction as text) of every AgiBot World 2026 episode of the index
# (examples/realRobots/AgiBot/tools/agibot_index.py), one process per GPU on shard NODE * 4 + g of NN * 4: TempSeg, then
# the Captioner, each GPU retrying the pair up to RETRIES times (both stages resume). Settings as
# examples/realRobots/EgoDex/train_files/hangang/egodex_flora_node.sh (per-process compile caches, GPU memory 0.70, a
# host-RAM budget, TMPDIR on /scratch); --max-duration-s 430: AgiBot episodes reach 429 s (Flora's own guard is 120 s).
set -o pipefail
NODE=${SLURM_NODEID:-0}
NN=${NN:-${SLURM_NNODES:-1}}
REPO=/scratch/t509a04/workspace/starVLA_car
TOOLS=${REPO}/examples/realRobots/EgoDex/tools
F=/scratch/t509a04/data/AgiBotWorld2026/flora
OUT=${OUT:-${F}/subtasks_flora_dex}
THREADS=${THREADS:-32}
MEM=${MEM:-0.70}
BUDGET=${BUDGET:-64}
RETRIES=${RETRIES:-6}
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn MALLOC_ARENA_MAX=2
mkdir -p ${F}/logs ${OUT}
echo "[$(date)] node ${NODE}/${NN} $(hostname): flora dex"
for g in 0 1 2 3; do
  (
    shard=$((NODE * 4 + g)); tag=s${shard}of$((NN * 4))
    for attempt in $(seq 1 ${RETRIES}); do
      ok=1
      for stage in tempseg caption; do
        c=/tmp/cc_agibot_${tag}_${attempt}
        mkdir -p /scratch/t509a04/tmp_flora/agibot_${tag}
        CUDA_VISIBLE_DEVICES=${g} TRITON_CACHE_DIR=${c}/triton TORCHINDUCTOR_CACHE_DIR=${c}/inductor VLLM_CACHE_ROOT=${c}/vllm \
          TMPDIR=/scratch/t509a04/tmp_flora/agibot_${tag} \
          /home01/t509a04/envs/vllm129/bin/python ${TOOLS}/flora_subtasks.py ${F}/index /scratch/t509a04/data/AgiBotWorld2026/raw ${OUT} \
          --split agibot --shard ${shard} --num-shards $((NN * 4)) --tag ${tag} --profile dex --max-duration-s 430 \
          --instruction description --backend vllm --stage ${stage} --threads ${THREADS} \
          --gpu-memory-utilization ${MEM} --mem-budget-gb ${BUDGET} --tmp /dev/shm/flora_agibot_${tag} || { ok=0; break; }
      done
      [ ${ok} = 1 ] && { echo "[$(date)] ${tag} complete (attempt ${attempt})"; break; }
      echo "[$(date)] ${tag} attempt ${attempt} failed; retrying"
      pkill -9 -u "$(id -u)" -f "VLLM::EngineCore" -P 1 2>/dev/null  # engines left without their process
      sleep 30
    done
  ) >> ${F}/logs/flora_dex_${tag}.log 2>&1 &
done
wait
echo "[$(date)] node ${NODE}: done"
