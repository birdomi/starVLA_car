#!/bin/bash
# Wait for a checkpoint of the carfew run, then: VLM-vs-base weight diff + fp16 val eval.
# Usage: auto_eval_ckpt.sh <step> [run_id] [config_yaml] [dataset] [robot_type]   (tmux; freest GPU among 4-7)
STEP=$1
RUN_ID=${2:-starvla_qwengr00t_dit_carfew_v1_2b_front_abs}
CFG=${3:-/mnt/DATA2/yyg/starVLA/examples/realRobots/H1_2_car_real/train_files/starvla_qwengr00t_carfew_v1_2b_front_abs.yaml}
DATASET=${4:-car-few-tasks-v1}
ROBOT=${5:-h1_2_car_left_front}
ROOT=/mnt/DATA2/yyg/starVLA
RUN=$ROOT/playground/Checkpoints/$RUN_ID
CK=$RUN/checkpoints/steps_${STEP}_pytorch_model.pt
OUT=$ROOT/playground/eval_results/${RUN_ID#starvla_qwengr00t_}
mkdir -p $OUT
PY=/home/user/miniconda3/envs/starVLA/bin/python
cd $ROOT
echo "[$(date +%T)] waiting for $CK"
while [ ! -f "$CK" ]; do sleep 60; done
prev=-1; while :; do s=$(stat -c %s "$CK"); [ "$s" = "$prev" ] && break; prev=$s; sleep 60; done
echo "[$(date +%T)] checkpoint complete ($((prev/1024/1024/1024)) GB)"
GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits -i 4,5,6,7 | sort -t, -k2 -nr | head -1 | cut -d, -f1)
echo "[$(date +%T)] using GPU $GPU"
$PY - "$CK" > $OUT/vlm_diff_${STEP}.txt 2>&1 <<'PYEOF'
import sys,glob,random,torch,collections
from safetensors import safe_open
sd=torch.load(sys.argv[1],map_location='cpu',mmap=True,weights_only=True)
print('dtypes',dict(collections.Counter((k.split('.')[0],str(v.dtype)) for k,v in sd.items())))
f=safe_open(glob.glob('/home/user/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/*/model.safetensors-00001-of-00001.safetensors')[0],'pt'); bk=list(f.keys())
vk=[k for k in sd if 'qwen_vl_interface' in k and sd[k].ndim==2]; random.seed(0); rows=[]
for k in random.sample(vk,16):
    tail=k.split('qwen_vl_interface.model.',1)[1]; c=[x for x in bk if x.split('.',1)[1]==tail.split('.',1)[1]]
    if not c: continue
    w=f.get_tensor(c[0]).float(); v=sd[k].float()
    rows.append((tail[:60],((v-w)==0).float().mean().item(),((v-w).norm()/w.norm()).item()))
for t,same,rel in rows: print(f'{t:60s} identical {100*same:6.2f}%  rel|dW| {rel:.2e}')
print('mean identical %.3f%%'%(100*sum(r[1] for r in rows)/len(rows)))
PYEOF
echo "[$(date +%T)] vlm diff done"
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$ROOT RANK=0 $PY examples/realRobots/H1_2_car_real/eval_val_chunks.py \
  --ckpt $CK --config $CFG --dataset $DATASET --robot $ROBOT --out $OUT/val_${STEP}_fp16.json --dtype float16 --n 1800 > $OUT/val_${STEP}_fp16.log 2>&1
echo "[$(date +%T)] eval exit $?"
