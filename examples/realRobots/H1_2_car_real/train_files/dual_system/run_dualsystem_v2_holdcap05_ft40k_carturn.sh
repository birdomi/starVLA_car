#!/bin/bash
# Fine-tune the relplan run from steps_40000 on the hold-capped train set (joint_e2e_vlm15_relplan_holdcap05_ft40k_v2_carturn_3cam.yaml,
# 10k = 50k total; outputs labelled ft50k) -> in parallel on the 4 GPUs: stride-30 rollout on all 60 v2 val episodes (GPU 0), stride-30 rollout + render of eps 0,1,2,5
# (GPU 3), head-hold timing probe (scratchpad headtiming/hazard_probe.py) on the new ft50k and on the 40k start point (GPUs 1, 2).
# Usage: run_dualsystem_v2_holdcap05_ft40k_carturn.sh [gpus=0,1,2,3] [port=29778]   (launch with setsid; do not edit while running)
GPUS=${1:-0,1,2,3}; PORT=${2:-29778}
ROOT=/mnt/DATA2/yyg/starVLA; cd $ROOT
H=$ROOT/examples/realRobots/H1_2_car_real; D=$H/train_files/dual_system
PY=/home/user/miniconda3/envs/starVLA/bin/python; RPY=/home/user/miniconda3/envs/libero310/bin/python
Y=$D/joint_e2e_vlm15_relplan_holdcap05_ft40k_v2_carturn_3cam.yaml
CK=$ROOT/playground/Checkpoints/starvla_dualsystem_v2_e2e_vlm15_relplan_holdcap05_ft40k_carturn_3cam/checkpoints/steps_10000_pytorch_model.pt
CK40=$ROOT/playground/Checkpoints/starvla_dualsystem_v2_e2e_vlm15_relplan_carturn_3cam/checkpoints/steps_40000_pytorch_model.pt
O=$ROOT/playground/eval_results/dualsystem_v2_e2e_vlm15_relplan_holdcap05_ft40k_carturn_3cam
ST=$ROOT/playground/Datasets/car-turn-signal-v2-ep/train/meta/stats_gr00t.abs.json
HT=/tmp/claude-1000/-mnt-DATA2-yyg-starVLA/3429367a-a321-47c8-ab05-731243e5181a/scratchpad/headtiming
ALL=$($PY -c "
import glob, pandas as pd
e = pd.concat(pd.read_parquet(p) for p in sorted(glob.glob('$ROOT/playground/Datasets/car-turn-signal-v2-ep/val/meta/episodes/*/*.parquet')))
print(','.join(str(i) for i in sorted(e.episode_index)))")
IFS=, read -r G0 G1 G2 G3 <<< "$GPUS"
export STARVLA_CUDA_DECODER_CACHE=150
echo "[$(date '+%m-%d %T')] fine-tune from 40k on the hold-capped (0.5 s) train set start"
bash $H/train_files/run_car_yaml.sh $Y $GPUS $PORT
echo "[$(date '+%m-%d %T')] fine-tune exit"
if [ ! -f "$CK" ]; then echo "checkpoint missing: $CK -> stop"; exit 1; fi
roll() {  # gpu episodes out
  CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$ROOT RANK=0 taskset -c 17-30 $PY $H/rollout_preds_dualsystem.py --ckpt $CK --config $Y \
    --dataset car-turn-signal-v2-ep --robot h1_2_car_turn_dual_s2_nohc --episodes $2 --s1-stride 30 --s2-stride 30 \
    --out $3/dual --out-s2 $3/dual_s2only 2>&1 | grep -E "^ep|Error|Traceback|assert" > $3.log
}
mkdir -p $O
echo "[$(date '+%m-%d %T')] rollouts + probes"
roll $G0 $ALL $O/rollout_ft50k_s30_allval &
(roll $G3 0,1,2,5 $O/rollout_ft50k_s30; export MUJOCO_GL=egl
 for d in dual dual_s2only; do CUDA_VISIBLE_DEVICES=$G3 taskset -c 17-30 $RPY $H/render_rollout.py $O/rollout_ft50k_s30/$d --stats $ST 2>&1 | tail -1; done) &
CUDA_VISIBLE_DEVICES=$G1 PYTHONPATH=$ROOT RANK=0 taskset -c 17-30 $PY $HT/hazard_probe.py $CK $O/hazard_ft50k.csv 8 > $O/hazard_ft50k.log 2>&1 &
CUDA_VISIBLE_DEVICES=$G2 PYTHONPATH=$ROOT RANK=0 taskset -c 17-30 $PY $HT/hazard_probe.py $CK40 $O/hazard_40k.csv 8 > $O/hazard_40k.log 2>&1 &
wait
echo "[$(date '+%m-%d %T')] done"
