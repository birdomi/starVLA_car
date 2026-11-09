#!/usr/bin/env bash
# Evaluate a starVLA checkpoint of THIS repo on RMBench through the XPolicyLab adapter `starVLA_dev`
# (eval_files/xpolicylab/starVLA_dev, symlinked into $RMBENCH_ROOT/XPolicyLab/policy/).
#
#   bash examples/simBenchmarks/RMBench/eval_files/eval_rmbench.sh <ckpt_path> <tasks> [gpu] [test_num] [seed]
#     tasks    : a task, a comma list, or a group: all (10) | 9 (paper) | m1 | mn
#     gpu      : GPU for the simulator, and for the starVLA server unless STARVLA_SERVER_GPU is set (default 0)
#     test_num : episodes per task (default 100 = the RMBench protocol)
#     seed     : eval seed; episodes use sim seeds from 100000 * (1 + seed) (default 0)
#
# Processes: starVLA websocket server (starVLA env, GPU) <- XPolicyLab policy server running starVLA_dev (RMBench env,
# CPU) <- RMBench simulator client (RMBench env, GPU). Started once per task by XPolicyLab's eval.sh; the starVLA server
# is started once here and reused.
#
# Env: RMBENCH_ROOT (/mnt/DATA2/yyg/RMBench), RMBENCH_ENV (RMBench), STARVLA_PYTHON (starVLA env python),
#      STARVLA_SERVER_GPU (default: gpu), STARVLA_SERVER_PORT (default: a free port), RMBENCH_LOG_DIR (default results/rmbench/<ckpt tag>),
#      render knobs ROBOTWIN_SHADER_DIR / ROBOTWIN_RT_SPP / ROBOTWIN_RT_DEPTH / ROBOTWIN_RT_DENOISER (patched
#      envs/_base_task.py), adapter overrides STARVLA_* (see starVLA_dev/model.py).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

if [[ $# -lt 2 ]]; then
    sed -n 2,20p "$0" >&2
    exit 1
fi
ckpt_path="$(readlink -m "$1")"
tasks_arg="$2"
gpu="${3:-0}"
test_num="${4:-100}"
seed="${5:-0}"
server_gpu="${STARVLA_SERVER_GPU:-${gpu}}"

RMBENCH_ROOT="${RMBENCH_ROOT:-/mnt/DATA2/yyg/RMBench}"
RMBENCH_ENV="${RMBENCH_ENV:-RMBench}"
export STARVLA_PYTHON="${STARVLA_PYTHON:-$HOME/miniconda3/envs/starVLA/bin/python}"

M1="observe_and_pickup rearrange_blocks put_back_block swap_blocks swap_T"
MN="battery_try blocks_ranking_try cover_blocks press_button"
case "${tasks_arg}" in
    all) tasks="${M1} ${MN} place_block_mat" ;;
    9) tasks="${M1} ${MN}" ;;
    m1) tasks="${M1}" ;;
    mn) tasks="${MN}" ;;
    *) tasks="${tasks_arg//,/ }" ;;
esac

# run tag: <run dir>_<ckpt file stem>, e.g. starvla_..._robotwin_clean50_v2_pytorch_model
run_dir="$(dirname "${ckpt_path}")"
[[ "$(basename "${run_dir}")" =~ ^(checkpoints|final_model)$ ]] && run_dir="$(dirname "${run_dir}")"
ckpt_tag="$(basename "${run_dir}")_$(basename "${ckpt_path%.*}")"
log_dir="$(readlink -m "${RMBENCH_LOG_DIR:-${REPO_ROOT}/results/rmbench/${ckpt_tag}}")"
mkdir -p "${log_dir}"

# stream episode-memory checkpoints run as trained: VLM in its config dtype (bf16), memory + DiT fp32 (no --use_bf16)
if grep -qs 'name: QwenGR00T_EpisodeMemory' "${run_dir}/config.yaml"; then
    export ROBOTWIN_USE_BF16="${ROBOTWIN_USE_BF16:-0}"
fi

port="${STARVLA_SERVER_PORT:-$("${STARVLA_PYTHON}" -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1])')}"
export STARVLA_SERVER_PORT="${port}"

echo "[rmbench] ckpt=${ckpt_path} tasks=(${tasks}) gpu=${gpu} server_gpu=${server_gpu} test_num=${test_num} seed=${seed} port=${port}"
echo "[rmbench] logs: ${log_dir}"

server_pid=""
cleanup() {
    if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
        echo "[rmbench] stopping starVLA server pid ${server_pid}"
        kill -- "-${server_pid}" 2>/dev/null || kill "${server_pid}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

setsid bash "${REPO_ROOT}/examples/simBenchmarks/Robotwin/eval_files/run_policy_server.sh" "${ckpt_path}" "${server_gpu}" "${port}" \
    > "${log_dir}/server.log" 2>&1 &
server_pid=$!
for _ in $(seq 1 180); do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
        echo "[rmbench][ERROR] starVLA server exited; see ${log_dir}/server.log" >&2
        tail -20 "${log_dir}/server.log" >&2
        exit 1
    fi
    if "${STARVLA_PYTHON}" -c "import socket,sys; s=socket.socket(); sys.exit(s.connect_ex(('127.0.0.1', ${port})))"; then
        break
    fi
    sleep 5
done
echo "[rmbench] starVLA server up (pid ${server_pid})"

args_file="${log_dir}/eval_args.txt"
printf -- '--test_num\n%s\n' "${test_num}" > "${args_file}"
export ROBOTWIN_EVAL_ARGS_FILE="${args_file}"

summary="${log_dir}/summary.txt"
for task in ${tasks}; do
    task_log="${log_dir}/${task}.log"
    echo "[rmbench] ${task} -> ${task_log}"
    (
        cd "${RMBENCH_ROOT}"
        CUDA_VISIBLE_DEVICES="${gpu}" bash XPolicyLab/policy/starVLA_dev/eval.sh \
            RMBench "${task}" "${ckpt_tag}" aloha_agilex joint "${seed}" "${gpu}" 0 "${RMBENCH_ENV}" "${RMBENCH_ENV}"
    ) > "${task_log}" 2>&1 || echo "[rmbench][WARN] ${task} exited non-zero" >&2
    # no match (crashed / killed task) must not abort the remaining tasks under set -e -o pipefail
    result="$(grep -a 'Final success rate' "${task_log}" | tail -1 | sed 's/\x1b\[[0-9;]*m//g' || true)"
    echo "${task}: ${result:-NO RESULT}" | tee -a "${summary}"
done
echo "[rmbench] done; summary: ${summary}"
