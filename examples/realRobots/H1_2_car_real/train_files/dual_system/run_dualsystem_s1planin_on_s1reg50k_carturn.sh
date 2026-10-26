#!/bin/bash
# Stage 2 on the S1-regression joint run's 50k (frozen Qwen / C-RADIO / System 2): a new System 1 whose only plan input is
# a smoothed 1 s chunk of the plan (stage2_system1_planin_on_s1reg50k_carturn_3cam.yaml, 20k) -> stride-30 rollouts (VLM lag
# 15 frames) of all 60 v2 val episodes for 20k and 10k, episodes 0,1,2,5,39,49 + render for 20k, then the sequence check /
# MAE table vs the joint run's 50k System 1 (same System 2) and its plan alone, and how far System 1 moves off its plan.
# Usage: run_dualsystem_s1planin_on_s1reg50k_carturn.sh [gpus=0,1,2,3] [port=29783]   (launch with setsid; do not edit while running)
GPUS=${1:-0,1,2,3}; PORT=${2:-29783}
ROOT=/mnt/DATA2/yyg/starVLA; cd $ROOT
H=$ROOT/examples/realRobots/H1_2_car_real; D=$H/train_files/dual_system
PY=/home/user/miniconda3/envs/starVLA/bin/python; RPY=/home/user/miniconda3/envs/libero310/bin/python
Y=$D/stage2_system1_planin_on_s1reg50k_carturn_3cam.yaml
CKD=$ROOT/playground/Checkpoints/starvla_dualsystem_s1planin_on_s1reg50k_carturn_3cam/checkpoints
O=$ROOT/playground/eval_results/dualsystem_s1planin_on_s1reg50k_carturn_3cam
E=$ROOT/playground/eval_results
ST=$ROOT/playground/Datasets/car-turn-signal-v2-ep/train/meta/stats_gr00t.abs.json
SB=/tmp/claude-1000/-mnt-DATA2-yyg-starVLA/3429367a-a321-47c8-ab05-731243e5181a/scratchpad
export STARVLA_CUDA_DECODER_CACHE=150
echo "[$(date '+%m-%d %T')] System 1 (smoothed 1 s plan chunk input) on the frozen s1reg 50k start"
bash $H/train_files/run_car_yaml.sh $Y $GPUS $PORT
echo "[$(date '+%m-%d %T')] training exit"
if [ ! -f "$CKD/steps_20000_pytorch_model.pt" ]; then echo "checkpoint missing: $CKD/steps_20000_pytorch_model.pt -> stop"; exit 1; fi
IFS=, read -r G0 G1 G2 G3 <<< "$GPUS"
roll() {  # gpu step episodes out log
  CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$ROOT RANK=0 taskset -c 17-30 $PY $H/rollout_preds_dualsystem.py --ckpt $CKD/steps_$2_pytorch_model.pt \
    --config $Y --dataset car-turn-signal-v2-ep --robot h1_2_car_turn_dual_s2_nohc --episodes $3 --s1-stride 30 --s2-stride 30 \
    --out $4/dual --out-s2 $4/dual_s2only 2>&1 | grep -E "^ep|Error|Traceback|assert" > $5
}
mkdir -p $O
echo "[$(date '+%m-%d %T')] rollouts 20k"
R=$O/rollout_planin20k_s30_allval
roll $G0 20000 $(seq -s, 0 19) $R $R.part0.log &
roll $G1 20000 $(seq -s, 20 39) $R $R.part1.log &
roll $G2 20000 $(seq -s, 40 59) $R $R.part2.log &
(roll $G3 20000 0,1,2,5,39,49 $O/rollout_planin20k_s30 $O/rollout_planin20k_s30.log; export MUJOCO_GL=egl
 for d in dual dual_s2only; do CUDA_VISIBLE_DEVICES=$G3 taskset -c 17-30 $RPY $H/render_rollout.py $O/rollout_planin20k_s30/$d --stats $ST 2>&1 | tail -1; done) &
wait
echo "[$(date '+%m-%d %T')] rollouts 10k"
R=$O/rollout_planin10k_s30_allval
roll $G0 10000 $(seq -s, 0 14) $R $R.part0.log &
roll $G1 10000 $(seq -s, 15 29) $R $R.part1.log &
roll $G2 10000 $(seq -s, 30 44) $R $R.part2.log &
roll $G3 10000 $(seq -s, 45 59) $R $R.part3.log &
wait
echo "[$(date '+%m-%d %T')] checks"
RG=$E/dualsystem_v2_e2e_vlm15_relplan_s1reg_holdcap05_carturn_3cam/rollout_s1reg50k_s30_allval
MS=$E/dualsystem_v2_e2e_vlm15_relplan_holdcap05_memstream_carturn_3cam/rollout_memstream66k_s30_allval
(cd $SB/seqcheck && $PY seq_check.py GT=$RG/dual reg50k=$RG/dual reg50k_plan=$RG/dual_s2only stream66k=$MS/dual \
   planin10k=$O/rollout_planin10k_s30_allval/dual planin20k=$O/rollout_planin20k_s30_allval/dual \
   planin20k_plan=$O/rollout_planin20k_s30_allval/dual_s2only > $O/seq_check.txt 2>&1; cp seq_check_last.csv $O/seq_check.csv)
$PY $SB/s1reg/mae_dirs.py reg50k=$RG stream66k=$MS planin10k=$O/rollout_planin10k_s30_allval planin20k=$O/rollout_planin20k_s30_allval > $O/mae.txt 2>&1
$PY $SB/memory/action_diff.py "reg50k S1-vs-plan=$RG/dual:$RG/dual_s2only" \
  "planin20k S1-vs-plan=$O/rollout_planin20k_s30_allval/dual:$O/rollout_planin20k_s30_allval/dual_s2only" \
  "planin10k S1-vs-plan=$O/rollout_planin10k_s30_allval/dual:$O/rollout_planin10k_s30_allval/dual_s2only" \
  "plan reg50k-vs-planin20k=$RG/dual_s2only:$O/rollout_planin20k_s30_allval/dual_s2only" > $O/s1_vs_plan.txt 2>&1
echo "[$(date '+%m-%d %T')] done"
