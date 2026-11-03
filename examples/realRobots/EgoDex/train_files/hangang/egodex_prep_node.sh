#!/bin/bash
# One node of egodex_prep.sbatch: this node's share (episode NODE, NODE + NNODES, ...) of every EgoDex split
#  1. -> 10 fps 448x256 frames (egodex_frames.py, CPU; index_<split>_<NODE>of<NNODES>.jsonl), straight from the zips
#  2. -> Flora subtasks with the task description as instruction (flora_subtasks.py --backend vllm), one process per
#     GPU over a quarter of those episodes: TempSeg (merged LoRA), then the Captioner (bf16 copy)
# Both steps resume: converted episodes and annotated paths are skipped on a rerun.
set -eo pipefail
NODE=${SLURM_NODEID:-0}
NN=${SLURM_NNODES:-1}
REPO=/scratch/t509a04/workspace/starVLA_car
TOOLS=${REPO}/examples/realRobots/EgoDex/tools
D=/scratch/t509a04/data/EgoDex
FRAMES=${D}/frames448
OUT=${D}/subtasks_flora
LOG=${D}/prep_logs
SPLITS="test part1 part2 part3 part4 part5 extra"
mkdir -p ${FRAMES} ${OUT} ${LOG}
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "[$(date)] node ${NODE}/${NN} $(hostname): frames"
nice -n 5 /home01/t509a04/envs/starVLA/bin/python ${TOOLS}/egodex_frames.py ${D}/zips ${FRAMES} --splits ${SPLITS} \
  --shard ${NODE} --num-shards ${NN} --workers 200 > ${LOG}/frames_n${NODE}.log 2>&1

echo "[$(date)] node ${NODE}: flora"
for g in 0 1 2 3; do
  (
    for stage in tempseg caption; do
      CUDA_VISIBLE_DEVICES=${g} /home01/t509a04/envs/vllm129/bin/python ${TOOLS}/flora_subtasks.py ${FRAMES} ${D}/zips ${OUT} \
        --split "$(echo ${SPLITS} | tr ' ' ',')" --index "index_{split}_${NODE}of${NN}.jsonl" --shard ${g} --num-shards 4 \
        --tag n${NODE}of${NN}g${g} --instruction description --backend vllm --stage ${stage} --threads 64 \
        --tmp /dev/shm/flora_n${NODE}g${g}
    done
  ) > ${LOG}/flora_n${NODE}g${g}.log 2>&1 &
done
wait
echo "[$(date)] node ${NODE}: done"
