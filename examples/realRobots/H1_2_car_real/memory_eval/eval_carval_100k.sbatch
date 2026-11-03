#!/bin/bash
# car-1015-v3 val evaluation of the 100K memory pretraining checkpoint: all 7 VLMs under the Flora and the template
# labels (eval_carval.sh, eval.only in its own run dir), then per-step dumps (qwen35_2b, 2 episodes per task).
#SBATCH -J mem_eval_carval
#SBATCH -p natStr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH -t 02:00:00
#SBATCH --comment="field=robotics;appl=pytorch"
#SBATCH -o /scratch/t509a04/data/car1015_val/logs/%x_%j.out
C=/scratch/t509a04/data/car1015_val
R=/scratch/t509a04/workspace/starVLA_car
cd ${C}
EVAL_RUN=memory_eval_carval_100k bash eval_carval.sh > logs/eval_carval_100k.log 2>&1
EVAL_RUN=memory_eval_carval_100k_tpl SUBTASKS=${C}/subtasks_template bash eval_carval.sh > logs/eval_carval_100k_tpl.log 2>&1
module load miniconda3 cuda/12.8 >/dev/null 2>&1
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=${R} OMP_NUM_THREADS=8
cd ${R}
i=0
for lab in flora template; do
  CUDA_VISIBLE_DEVICES=${i} TRITON_CACHE_DIR=/tmp/triton_dump_${lab} /home01/t509a04/envs/mempre/bin/python \
    examples/realRobots/EgoDex/tools/memory_eval_dump.py playground/Checkpoints/memory_pretrain_multivlm_n32d8_t1024_egodex \
    subtask_review/memory_eval_carval_100k_${lab}_qwen35_2b.json --ckpt steps_100000.pt --vlm qwen35_2b \
    --root ${C}/frames448 --split carval --subtasks-dir ${C}/subtasks_${lab} --per-task 2 > ${C}/logs/dump_100k_${lab}.log 2>&1 &
  i=$((i + 1))
done
wait
echo "[$(date)] done"
