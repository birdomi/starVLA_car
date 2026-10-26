#!/bin/bash
# Chain the QwenGR00T_TriSystem RoboCasa-100 stages: wait until a stage has written its final checkpoint, its
# torchrun has exited and the GPUs are free, then launch the next stage (each stage yaml points
# trainer.pretrained_checkpoint at the previous stage's final checkpoint).
# Usage: bash chain_rc100_stages.sh [first_stage_to_launch=2] [gpus=4,5,6,7] [port=29781]
# Run it in tmux; the trainings themselves are setsid-detached by launch_regressor_sim_local.sh.
set -u
FIRST=${1:-2}; GPUS=${2:-4,5,6,7}; PORT=${3:-29781}
cd /mnt/DATA2/yyg/starVLA
Y=examples/simBenchmarks/Robocasa_tabletop/train_files/trisystem
LAUNCH=examples/simBenchmarks/launch_regressor_sim_local.sh
CK=playground/Checkpoints
#        yaml                      run_id                                    final step
STAGES=("stage1_system15_rc100.yaml starvla_trisystem_s1p5_rc100_robocasa_gr1 30000"
        "stage2_system1_rc100.yaml  starvla_trisystem_s1_rc100_robocasa_gr1   50000"
        "stage3_joint_rc100.yaml    starvla_trisystem_joint_rc100_robocasa_gr1 20000")

log() { echo "[chain $(date '+%m-%d %H:%M:%S')] $*"; }

wait_stage() {  # yaml run_id step
  local ckpt=$CK/$2/checkpoints/steps_$3_pytorch_model.pt
  log "waiting for $ckpt and the end of $1"
  until [ -f "$ckpt" ] && ! pgrep -u "$USER" -f "[t]rain_starvla.py --config_yaml $Y/$1" > /dev/null; do sleep 120; done
  log "$1 finished"
}

wait_gpus_free() {
  local ids; ids=$(echo "$GPUS" | tr ',' ' ')
  while true; do
    local busy=0
    for g in $ids; do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
      [ "$used" -gt 2000 ] && busy=1
    done
    [ $busy -eq 0 ] && return
    sleep 60
  done
}

free_port() {
  local p=$PORT
  while ss -tln | grep -q ":$p "; do p=$((p + 1)); done
  echo $p
}

for i in "${!STAGES[@]}"; do
  n=$((i + 1))
  read -r yaml run_id step <<< "${STAGES[$i]}"
  if [ $n -ge "$FIRST" ]; then
    if [ $n -gt 1 ]; then
      read -r pyaml prun pstep <<< "${STAGES[$((i - 1))]}"
      if [ ! -f "$CK/$prun/checkpoints/steps_${pstep}_pytorch_model.pt" ]; then
        log "ABORT: stage $((n - 1)) checkpoint missing, not launching stage $n"; exit 1
      fi
    fi
    wait_gpus_free
    port=$(free_port)
    log "launching stage $n ($yaml) on GPUs $GPUS port $port"
    bash "$LAUNCH" "$Y/$yaml" "$GPUS" "$port"
    sleep 300
  fi
  wait_stage "$yaml" "$run_id" "$step"
done
log "all stages done"
