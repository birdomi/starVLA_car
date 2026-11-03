#!/bin/bash
# Flora profile comparison (sbatch examples/realRobots/AgiBot/train_files/hangang/agibot_flora_ab.sbatch; index_agibotab.jsonl:
# a subset of index_agibot.jsonl) on 15 AgiBot sample episodes: human (top_head), dex and tianji (top_head + both hand
# cameras), one profile per GPU, TempSeg then Captioner; outputs in ab/<profile>/
#SBATCH -J flora_ab
#SBATCH -p natStr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH -t 02:00:00
#SBATCH --comment="field=robotics;appl=pytorch"
#SBATCH -o /scratch/t509a04/data/AgiBotWorld2026/flora/logs/%x_%j.out
module load miniconda3 cuda/12.8 >/dev/null 2>&1
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn MALLOC_ARENA_MAX=2
F=/scratch/t509a04/data/AgiBotWorld2026/flora
cd /scratch/t509a04/workspace/starVLA_car
g=0
for prof in human dex tianji; do
  (
    cc=/tmp/cc_ab_${prof}; mkdir -p /scratch/t509a04/tmp_flora/ab_${prof}
    for stage in tempseg caption; do
      CUDA_VISIBLE_DEVICES=${g} TRITON_CACHE_DIR=${cc}/triton TORCHINDUCTOR_CACHE_DIR=${cc}/inductor VLLM_CACHE_ROOT=${cc}/vllm \
        TMPDIR=/scratch/t509a04/tmp_flora/ab_${prof} /home01/t509a04/envs/vllm129/bin/python \
        examples/realRobots/EgoDex/tools/flora_subtasks.py ${F}/index /scratch/t509a04/data/AgiBotWorld2026/raw ${F}/ab/${prof} \
        --split agibotab --tag ab --profile ${prof} --instruction description --backend vllm --stage ${stage} --threads 16 \
        --max-duration-s 430 --gpu-memory-utilization 0.70 --mem-budget-gb 48 --tmp /dev/shm/flora_ab_${prof} || exit 1
    done
  ) > ${F}/logs/ab_${prof}.log 2>&1 &
  g=$((g + 1))
done
wait
echo "[$(date)] done"
