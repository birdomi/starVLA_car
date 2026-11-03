#!/bin/bash
# Memory pretraining checkpoint -> held-out streams of car-1015-v3 val (head camera, Flora subtasks), all 7 VLMs,
# 4 GPUs, as an --overlap step: train_memory_pretrain.py eval.only in its own run dir (EVAL_RUN, the checkpoint
# linked in), so the pretraining run's eval.jsonl and W&B run stay untouched.
module load miniconda3 cuda/12.8 >/dev/null 2>&1
source activate /home01/t509a04/envs/starVLA
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
export PYTHONPATH=/scratch/t509a04/workspace/starVLA_car WANDB_MODE=disabled TRITON_CACHE_DIR=/tmp/triton_evalcar
C=/scratch/t509a04/data/car1015_val
cd /scratch/t509a04/workspace/starVLA_car
/home01/t509a04/envs/mempre/bin/python -m torch.distributed.run --standalone --nproc_per_node 4 starVLA/training/train_memory_pretrain.py \
  --config_yaml examples/realRobots/EgoDex/train_files/memory_pretrain_multivlm_egodex.yaml \
  run_id=${EVAL_RUN:-memory_eval_carval_10k} data.root=${C}/frames448 data.splits=[carval] data.subtasks_dir=${SUBTASKS:-${C}/subtasks_flora} \
  data.num_workers=4 eval.only=true eval.splits=[carval] eval.max_episodes=0 eval.batch_size=16 eval.max_updates=60 \
  gpu_memory_fraction=0.85
echo "[$(date)] exit $?"
