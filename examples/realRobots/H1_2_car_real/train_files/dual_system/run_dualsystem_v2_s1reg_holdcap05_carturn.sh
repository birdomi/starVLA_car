#!/bin/bash
# DualSystem joint from scratch with System 1 as a DIRECT MSE REGRESSOR (joint_e2e_vlm15_relplan_s1reg_holdcap05_v2_carturn_3cam.yaml,
# 50k, hold-capped train set) -> stride-30 rollouts (VLM lag 15 frames, no memory) of all 60 v2 val episodes for steps 50k and
# 10k..40k, episodes 0,1,2,5,39,49 + render for 50k, then the sequence check vs GT and the flow-S1 baselines + the MAE table.
# Usage: run_dualsystem_v2_s1reg_holdcap05_carturn.sh [gpus=0,1,2,3] [port=29782]   (launch with setsid; do not edit while running)
GPUS=${1:-0,1,2,3}; PORT=${2:-29782}
ROOT=/mnt/DATA2/yyg/starVLA; cd $ROOT
H=$ROOT/examples/realRobots/H1_2_car_real; D=$H/train_files/dual_system
PY=/home/user/miniconda3/envs/starVLA/bin/python; RPY=/home/user/miniconda3/envs/libero310/bin/python
Y=$D/joint_e2e_vlm15_relplan_s1reg_holdcap05_v2_carturn_3cam.yaml
CKD=$ROOT/playground/Checkpoints/starvla_dualsystem_v2_e2e_vlm15_relplan_s1reg_holdcap05_carturn_3cam/checkpoints
O=$ROOT/playground/eval_results/dualsystem_v2_e2e_vlm15_relplan_s1reg_holdcap05_carturn_3cam
E=$ROOT/playground/eval_results
ST=$ROOT/playground/Datasets/car-turn-signal-v2-ep/train/meta/stats_gr00t.abs.json
SB=/tmp/claude-1000/-mnt-DATA2-yyg-starVLA/3429367a-a321-47c8-ab05-731243e5181a/scratchpad
export STARVLA_CUDA_DECODER_CACHE=150
echo "[$(date '+%m-%d %T')] joint training from scratch (System 1 = MSE regression, hold-capped data) start"
bash $H/train_files/run_car_yaml.sh $Y $GPUS $PORT
echo "[$(date '+%m-%d %T')] joint training exit"
if [ ! -f "$CKD/steps_50000_pytorch_model.pt" ]; then echo "checkpoint missing: $CKD/steps_50000_pytorch_model.pt -> stop"; exit 1; fi
IFS=, read -r G0 G1 G2 G3 <<< "$GPUS"
roll() {  # gpu step episodes out log
  CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$ROOT RANK=0 taskset -c 17-30 $PY $H/rollout_preds_dualsystem.py --ckpt $CKD/steps_$2_pytorch_model.pt \
    --config $Y --dataset car-turn-signal-v2-ep --robot h1_2_car_turn_dual_s2_nohc --episodes $3 --s1-stride 30 --s2-stride 30 \
    --out $4/dual --out-s2 $4/dual_s2only 2>&1 | grep -E "^ep|Error|Traceback|assert" > $5
}
mkdir -p $O
echo "[$(date '+%m-%d %T')] rollouts 50k"
R=$O/rollout_s1reg50k_s30_allval
roll $G0 50000 $(seq -s, 0 19) $R $R.part0.log &
roll $G1 50000 $(seq -s, 20 39) $R $R.part1.log &
roll $G2 50000 $(seq -s, 40 59) $R $R.part2.log &
(roll $G3 50000 0,1,2,5,39,49 $O/rollout_s1reg50k_s30 $O/rollout_s1reg50k_s30.log; export MUJOCO_GL=egl
 for d in dual dual_s2only; do CUDA_VISIBLE_DEVICES=$G3 taskset -c 17-30 $RPY $H/render_rollout.py $O/rollout_s1reg50k_s30/$d --stats $ST 2>&1 | tail -1; done) &
wait
for k in 10 20 30 40; do
  echo "[$(date '+%m-%d %T')] rollouts ${k}k"
  R=$O/rollout_s1reg${k}k_s30_allval
  roll $G0 ${k}000 $(seq -s, 0 14) $R $R.part0.log &
  roll $G1 ${k}000 $(seq -s, 15 29) $R $R.part1.log &
  roll $G2 ${k}000 $(seq -s, 30 44) $R $R.part2.log &
  roll $G3 ${k}000 $(seq -s, 45 59) $R $R.part3.log &
  wait
done
echo "[$(date '+%m-%d %T')] checks"
F=$E/dualsystem_v2_e2e_vlm15_relplan_carturn_3cam; FT=$E/dualsystem_v2_e2e_vlm15_relplan_holdcap05_ft40k_carturn_3cam
MS=$E/dualsystem_v2_e2e_vlm15_relplan_holdcap05_memstream_carturn_3cam/rollout_memstream66k_s30_allval
BASE="flow10k=$F/rollout_10k_s30_allval flow20k=$F/rollout_20k_s30_allval flow30k=$F/rollout_30k_s30_allval flow40k=$F/rollout_40k_s30_allval ft42k=$FT/rollout_ft42k_s30_allval stream66k=$MS"
NEW="reg10k=$O/rollout_s1reg10k_s30_allval reg20k=$O/rollout_s1reg20k_s30_allval reg30k=$O/rollout_s1reg30k_s30_allval reg40k=$O/rollout_s1reg40k_s30_allval reg50k=$O/rollout_s1reg50k_s30_allval"
DUAL=$(for x in GT=$O/rollout_s1reg50k_s30_allval $BASE $NEW; do echo -n "${x}/dual "; done)
(cd $SB/seqcheck && $PY seq_check.py $DUAL > $O/seq_check.txt 2>&1; cp seq_check_last.csv $O/seq_check.csv)
$PY $SB/s1reg/mae_dirs.py $BASE $NEW > $O/mae.txt 2>&1
echo "[$(date '+%m-%d %T')] done"
