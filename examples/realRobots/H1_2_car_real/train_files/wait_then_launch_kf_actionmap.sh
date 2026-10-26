#!/bin/bash
# Wait for the DiT car run to finish and GPUs 0-3 to free up, then start the Keyframe ActionMap 200k run.
ROOT=/mnt/DATA2/yyg/starVLA
PREV_LOG=$ROOT/train_carfew_v1_2b_front_abs_0929_1621.log
YAML=$ROOT/examples/realRobots/H1_2_car_real/train_files/starvla_qwengr00t_kf_actionmap_carfew_v1_2b_front_abs.yaml
echo "[$(date '+%m-%d %T')] waiting for EXIT in $PREV_LOG"
until grep -aqE "EXIT [0-9]+$" "$PREV_LOG"; do sleep 60; done
echo "[$(date '+%m-%d %T')] previous run: $(grep -aoE "EXIT [0-9]+$" $PREV_LOG | tail -1)"
until [ "$(nvidia-smi -i 0,1,2,3 --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>2000' | wc -l)" = 0 ]; do sleep 30; done
echo "[$(date '+%m-%d %T')] GPUs 0-3 free, launching $(basename $YAML)"
bash $ROOT/examples/realRobots/H1_2_car_real/train_files/run_car_yaml.sh "$YAML" 0,1,2,3 29752
echo "[$(date '+%m-%d %T')] kf_actionmap run ended"
