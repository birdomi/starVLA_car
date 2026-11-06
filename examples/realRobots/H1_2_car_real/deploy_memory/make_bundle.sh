#!/bin/bash
# Assemble a self-contained deploy bundle for an H1-2 car QwenGR00T / QwenGR00T_EpisodeMemory checkpoint (Hangang paths):
#   model/       run-dir layout for baseframework.from_pretrained: config.yaml (= the run's config.full.yaml),
#                dataset_statistics.json (TRAIN normalization), checkpoints/steps_<STEP>_pytorch_model.pt
#   base_vlm/    the Qwen3.5 base model folder (processor / tokenizer / weights) for an offline build
#   starVLA_src/ the repo's starVLA + deployment packages and the car data registry, as in this working tree
#   stats/       modality.json (the robot's 63-D state / 42-D action layout, car-1015-v3-ep), stats_gr00t.abs.json and
#                tasks.json: every instruction the run trained on (the union over its data mixture's datasets)
#   sample/      car-1015-v3-ep val episodes (parquet + the 3 model cameras' mp4) for verify_bundle.py
#   offline_eval/ <eval dir name>.json for every EVAL_DIRS entry (result.json of eval_val_car_v3.py, or stream.json)
#   deploy_infer.py / serve_policy.py / verify_bundle.py / README.md (from this folder; README filled per model)
# Usage: [EVAL_DIRS="<dir> ..."] [SAMPLE_TASKS=0,1,30] [TITLE="..."] make_bundle.sh <run_dir> <step> <out_dir>
set -eo pipefail
RUN=${1:?run dir}; STEP=${2:?step}; OUT=${3:?out dir}
REPO=$(cd "$(dirname "$0")/../../../.." && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-/home01/t509a04/envs/starVLA/bin/python}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1   # scipy (pulled in by the data registry) hangs on the login node otherwise
CK=${RUN}/checkpoints/steps_${STEP}_pytorch_model.pt
ROOT=$(${PY} -c "from omegaconf import OmegaConf; print(OmegaConf.load('${RUN}/config.full.yaml').datasets.vla_data.data_root_dir)")
DATA=${ROOT}/car-1015-v3-ep
VLM=$(${PY} -c "from omegaconf import OmegaConf; print(OmegaConf.load('${RUN}/config.full.yaml').framework.qwenvl.base_vlm)")
[ -f "${CK}" ] || { echo "missing ${CK}"; exit 1; }
[ -e "${OUT}" ] && { echo "${OUT} exists; remove it first"; exit 1; }
cd ${REPO}
echo "[$(date +%T)] ${CK} -> ${OUT}"

mkdir -p ${OUT}/model/checkpoints ${OUT}/stats ${OUT}/base_vlm ${OUT}/sample ${OUT}/offline_eval ${OUT}/starVLA_src
cp ${RUN}/config.full.yaml ${OUT}/model/config.yaml
cp ${RUN}/dataset_statistics.json ${OUT}/model/
cp ${CK} ${OUT}/model/checkpoints/
cp -rL ${VLM} ${OUT}/base_vlm/$(basename ${VLM})
cp ${DATA}/train/meta/modality.json ${DATA}/train/meta/stats_gr00t.abs.json ${OUT}/stats/

# tasks.json: the instructions of every dataset in the run's mixture (v3.0: tasks with train episodes; v2.0: tasks.jsonl)
PYTHONPATH=${REPO} ${PY} - "${RUN}/config.full.yaml" "${ROOT}" "${OUT}/stats/tasks.json" <<'EOF' 2>&1 | grep -v -iE "registry\]|warning"
import json, sys
from pathlib import Path
import pandas as pd
from omegaconf import OmegaConf
from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES
cfg, root, out = OmegaConf.load(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
tasks = []
for d_name, _, _ in DATASET_NAMED_MIXTURES[str(cfg.datasets.vla_data.data_mix)]:
    meta = root / d_name / "meta"
    if (meta / "tasks.jsonl").exists():
        found = [json.loads(l)["task"] for l in open(meta / "tasks.jsonl")]
    else:
        t = pd.read_parquet(meta / "tasks.parquet")
        used = {int(x[0]) for x in pd.read_parquet(meta / "episodes/chunk-000/file-000.parquet")["tasks"]}
        found = [s for s, i in sorted(t["task_index"].items(), key=lambda kv: kv[1]) if int(i) in used]
    tasks += [s for s in found if s not in tasks]
json.dump(tasks, open(out, "w"), indent=1, ensure_ascii=False)
print(f"tasks.json: {len(tasks)} instructions")
EOF

# source as trained (this working tree), without caches
tar --exclude=__pycache__ -cf - starVLA deployment requirements.txt pyproject.toml \
    examples/realRobots/H1_2_car_real/train_files/data_registry | tar -xf - -C ${OUT}/starVLA_src

# one car-1015-v3-ep val episode of each SAMPLE_TASKS task (the robot's 63-D state layout)
${PY} - "${DATA}/val" "${OUT}/sample" "${SAMPLE_TASKS:-0,1,30}" <<'EOF'
import json, shutil, sys
from pathlib import Path
import pandas as pd
val, out, want = Path(sys.argv[1]), Path(sys.argv[2]), [int(t) for t in sys.argv[3].split(",")]
eps = pd.read_parquet(val / "meta/episodes/chunk-000/file-000.parquet")
tasks = {int(i): s for s, i in pd.read_parquet(val / "meta/tasks.parquet")["task_index"].items()}
slug = {0: "left-turn-signal", 1: "right-turn-signal", 30: "sunroof-open", 31: "sunroof-close", 32: "engine-start"}
for t in want:
    row = eps[eps["tasks"].map(lambda x: int(x[0]) == t)].iloc[0]
    d = out / slug.get(t, f"task{t}")
    d.mkdir(parents=True)
    shutil.copy(val / row["data_path"], d / "data.parquet")
    for cam in ("head_left_2", "wrist_left_2", "wrist_right_2"):
        k = f"videos/observation.images.{cam}"
        src = val / k / f"chunk-{int(row[k + '/chunk_index']):03d}" / f"file-{int(row[k + '/file_index']):03d}.mp4"
        assert row[k + "/from_timestamp"] == 0.0, "per-episode mp4 expected"
        shutil.copyfile(src.resolve(), d / f"{cam}.mp4")
    json.dump({"episode_index": int(row["episode_index"]), "task_index": t, "task": tasks[t], "length": int(row["length"]),
               "fps": 30}, open(d / "meta.json", "w"), indent=1)
    print("sample", d.name, int(row["episode_index"]), int(row["length"]))
EOF

for e in ${EVAL_DIRS}; do
  if [ -f ${e}/result.json ]; then cp ${e}/result.json ${OUT}/offline_eval/$(basename ${e}).json
  elif [ -f ${e}/stream.json ]; then cp ${e}/stream.json ${OUT}/offline_eval/$(basename ${e}).json; fi
done
cp ${HERE}/deploy_infer.py ${HERE}/serve_policy.py ${HERE}/verify_bundle.py ${OUT}/

# README: the template with this model's description block and its offline eval
${PY} - "${RUN}/config.full.yaml" "${OUT}" "${HERE}/README.md" "${STEP}" "${TITLE:-}" <<'EOF'
import json, sys
from pathlib import Path
from omegaconf import OmegaConf
cfg, out, tpl, step, title = OmegaConf.load(sys.argv[1]), Path(sys.argv[2]), open(sys.argv[3]).read(), sys.argv[4], sys.argv[5]
fw, tr, vd = cfg.framework, cfg.trainer, cfg.datasets.vla_data
vlm = Path(str(fw.qwenvl.base_vlm)).name
mem = fw.get("episode_memory", None)
frozen = [p.strip() for p in str(tr.get("freeze_modules", "") or "").split(",") if p.strip()]
bs = int(vd.per_device_batch_size)
lines = [f"- **Framework:** `{fw.name}`; VLM {vlm} ({'frozen' if 'qwen_vl_interface' in frozen else 'fine-tuned'}); "
         f"action model {fw.action_model.action_model_type}, {fw.action_model.action_horizon} x {fw.action_model.action_dim} absolute actions at 30 Hz.",
         f"- **Data:** `{vd.data_mix}`; {tr.max_train_steps} steps at {bs} per GPU x 32 GPUs (global {bs * 32}); this bundle: steps {step}.",
         f"- **Starting point:** {tr.get('pretrained_checkpoint') or 'the base VLM and a new action model'}."]
if mem is None:
    lines.append("- **Memory:** none. Each predict() is independent: only the current frame and state are used.")
else:
    dec = mem.get("decision", None)
    lines.append(f"- **Episode memory:** {mem.num_tokens} tokens x d_model {mem.d_model}, depth {mem.depth}, {mem.attention} attention, "
                 f"token_dim {mem.get('token_dim')}; streamed, one update per ~1 s, BPTT over {mem.stream.tbptt} updates in training; "
                 f"DiT condition: {mem.get('condition', 'cross_attn')}.")
    if dec is not None:
        lines.append("- **Decision transformer (EgoDex pretraining):** episode-progress output only -> progress estimate and the progress token.")
    lines.append(f"- **Frozen in training:** {', '.join(frozen) if frozen else 'nothing'}.")
evals = sorted((out / "offline_eval").glob("*.json"))
if evals:
    lines += ["", "Offline car val (eval_val_car_v3.py: 296 episodes, one query per 30 frames, chunk mean error; head / arms in deg, hands raw):", "",
              "| eval | head | arms | hands | progress MAE |", "|---|---|---|---|---|"]
    for e in evals:
        r = json.load(open(e))
        a = r.get("ALL") or r.get("per_task", {}).get("ALL")
        if a:
            pm = a.get("progress_mae")
            lines.append(f"| {e.stem}{' (memory reset at every query)' if r.get('memory_reset') else ''} | {a['head']:.2f} | {a['arm']:.2f} | {a['hand']:.1f} | {'' if pm is None else f'{pm:.3f}'} |")
text = tpl.replace("@TITLE@", title or f"{fw.name}, {vlm}").replace("@MODEL_BLOCK@", "\n".join(lines)).replace("@VLM@", vlm).replace("@STEP@", step)
open(out / "README.md", "w").write(text)
EOF
{
  echo "built $(date '+%F %T') from ${REPO} @ $(git rev-parse HEAD) (+ uncommitted: $(git status --short starVLA deployment | awk '{print $2}' | tr '\n' ' '))"
  echo "run ${RUN}  step ${STEP}"
  (cd ${OUT} && sha256sum model/checkpoints/*.pt model/config.yaml model/dataset_statistics.json)
} > ${OUT}/MANIFEST.txt
du -sh ${OUT}
echo "[$(date +%T)] done"
