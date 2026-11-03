#!/bin/bash
# Flora subtasks (human profile, head camera, task text as instruction) of car-1015-v3 val, one process per GPU
module load miniconda3 cuda/12.8 >/dev/null 2>&1
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn MALLOC_ARENA_MAX=2
C=/scratch/t509a04/data/car1015_val
cd /scratch/t509a04/workspace/starVLA_car
for g in 0 1 2 3; do
  (
    tag=carval_g${g}; cc=/tmp/cc_${tag}; mkdir -p /scratch/t509a04/tmp_flora/${tag}
    for stage in tempseg caption; do
      CUDA_VISIBLE_DEVICES=${g} TRITON_CACHE_DIR=${cc}/triton TORCHINDUCTOR_CACHE_DIR=${cc}/inductor VLLM_CACHE_ROOT=${cc}/vllm \
        TMPDIR=/scratch/t509a04/tmp_flora/${tag} /home01/t509a04/envs/vllm129/bin/python \
        examples/realRobots/EgoDex/tools/flora_subtasks.py ${C}/frames448 ${C}/raw ${C}/subtasks_flora --split carval \
        --shard ${g} --num-shards 4 --tag ${tag} --instruction description --backend vllm --stage ${stage} --threads 32 \
        --gpu-memory-utilization 0.70 --mem-budget-gb 48 --tmp /dev/shm/flora_${tag} || exit 1
    done
  ) > ${C}/logs/flora_${g}.log 2>&1 &
done
wait
echo "[$(date)] done"
