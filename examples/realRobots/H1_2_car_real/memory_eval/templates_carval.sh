#!/bin/bash
# car-1015-v3 val subtask labels with the VLM task templates (examples/realRobots/EgoDex/tools/subtask_templates.py,
# Qwen3.8-Flash-Next-FP8, 4 GPUs), boundaries snapped to the Flora labels of flora_carval.sh
module load miniconda3 cuda/12.8 >/dev/null 2>&1
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
export TRITON_CACHE_DIR=/tmp/cc_tpl/triton TORCHINDUCTOR_CACHE_DIR=/tmp/cc_tpl/inductor VLLM_CACHE_ROOT=/tmp/cc_tpl/vllm
C=/scratch/t509a04/data/car1015_val
cd /scratch/t509a04/workspace/starVLA_car/examples/realRobots/EgoDex/tools
/home01/t509a04/envs/vllm129/bin/python subtask_templates.py ${C}/frames448 ${C}/subtasks_flora ${C}/subtasks_template \
  --split carval --model /scratch/t509a04/workspace/starVLA/playground/Pretrained_models/Qwen3.8-Flash-Next-FP8 --tp 4 \
  --gpu-memory-utilization 0.78
echo "[$(date)] exit $?"
