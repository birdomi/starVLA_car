#!/bin/bash
# Fast seeded RoboCasa eval: one fresh policy server + one 1-thread sim client (OPENBLAS/OMP/MKL_NUM_THREADS=1: same
# trajectories bit for bit, ~1/3 the CPU; tested 10-20) per task, WPG tasks per GPU at once, same args as
# eval_dualsystem_rc100.sh / batch_eval_args.sh (50 eps, seed 7, 720 steps, 12 actions per call, 256 x 256).
#   CK=<ckpt .pt> EVAL_DIR=<dir for eval_env_*.log> GPUS="4 5 6 7" WPG=5 PORT0=6960 [extra env, e.g. ROBOCASA_VLM_PREV_FRAME=1]
#   bash fast_eval.sh <task> [<task> ...]
cd /mnt/DATA2/yyg/starVLA
PYS=$HOME/miniconda3/envs/starVLA/bin/python; PYR=$HOME/miniconda3/envs/robocasa/bin/python
G=($GPUS); WPG=${WPG:-5}; PORT0=${PORT0:-6960}; TASKS=("$@")
ST=${SERVER_THREADS:-1}  # server CPU threads (default torch = all cores: 20 servers -> load 240)
VID=$(dirname "$CK")/../videos/$(basename "$CK" .pt)/n_action_steps_12_max_episode_steps_720_n_envs_1_gr1_unified
mkdir -p "$EVAL_DIR"
one() {  # task gpu port
    local t=$1 g=$2 p=$3 env=gr1_unified/${1}_GR1ArmsAndWaistFourierHands_Env
    OPENBLAS_NUM_THREADS=$ST OMP_NUM_THREADS=$ST MKL_NUM_THREADS=$ST \
        CUDA_VISIBLE_DEVICES=$g PYTHONPATH=/mnt/DATA2/yyg/starVLA: $PYS deployment/model_server/server_policy.py --ckpt_path "$CK" \
        --port $p --use_bf16 > "$EVAL_DIR/server_gpu${g}_port${p}.log" 2>&1 &
    local sp=$!
    for i in $(seq 1 180); do ss -ltn | grep -q ":$p " && break; kill -0 $sp 2>/dev/null || break; sleep 5; done
    echo "[fast] $t server gpu $g port $p up $(date +%T)"
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$g PYTHONPATH=/mnt/DATA2/yyg/starVLA: \
        PYTHONHASHSEED=0 $PYR examples/simBenchmarks/Robocasa_tabletop/eval_files/simulation_env.py --args.env_name $env \
        --args.port $p --args.n_episodes 50 --args.n_envs 1 --args.max_episode_steps 720 --args.n_action_steps 12 \
        --args.video_out_path "$VID/${t}_GR1ArmsAndWaistFourierHands_Env" --args.pretrained_path "$CK" --args.resize_size 256 256 \
        > "$EVAL_DIR/eval_env_gr1_unified_${t}_GR1ArmsAndWaistFourierHands_Env_gpu${g}.log" 2>&1
    echo "[fast] $t done rc=$? $(grep -cE '^\[episode\]' "$EVAL_DIR/eval_env_gr1_unified_${t}_GR1ArmsAndWaistFourierHands_Env_gpu${g}.log") eps $(date +%T)"
    kill $sp; wait $sp 2>/dev/null
}
worker() {  # worker index -> tasks idx, idx + n_workers, ...
    local w=$1 n=$2 g=${G[$(( $1 % ${#G[@]} ))]} p=$((PORT0 + $1)) k
    for ((k = w; k < ${#TASKS[@]}; k += n)); do one "${TASKS[$k]}" $g $p; done
}
NW=$(( ${#G[@]} * WPG ))
echo "[fast] ${#TASKS[@]} tasks, $NW workers on GPUs ${G[*]} ($WPG per GPU) $(date)"
for ((w = 0; w < NW; w++)); do worker $w $NW & sleep 2; done
wait
echo "[fast] ALL DONE $(date)"
