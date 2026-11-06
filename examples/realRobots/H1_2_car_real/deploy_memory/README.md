# H1-2 car deploy bundle: @TITLE@

starVLA policy for the H1-2 car inspection tasks: 3 cameras (head, left wrist, right wrist) + 42-D state + instruction
-> 50 x 42 absolute actions at 30 Hz (1.67 s).

@MODEL_BLOCK@

## Contents

| Path | What it is |
|---|---|
| `deploy_infer.py` | `CarPolicy`: robot-side wrapper (raw images + 63-D state + time -> absolute (50, 42) chunk). Handles the episode memory when the model has one. |
| `serve_policy.py` | The same policy over the starVLA websocket protocol (msgpack-numpy); a memory, if any, is kept on the server. |
| `verify_bundle.py` | Offline check on `sample/` (no robot): streams the episodes as the robot would and compares with the recorded actions; its output on the build machine is `VERIFY.txt`. |
| `model/` | `config.yaml` (full training config), `dataset_statistics.json` (TRAIN normalization, required), `checkpoints/*.pt`. |
| `base_vlm/@VLM@/` | Processor, tokenizer and base weights for an offline model build. |
| `starVLA_src/` | starVLA source as trained; `deploy_infer.py` puts it on `sys.path` itself. |
| `stats/` | `modality.json` (63-D state / 42-D action layout), `stats_gr00t.abs.json`, `tasks.json` (every instruction the model was trained on). |
| `sample/<task>/` | One val episode each: `data.parquet` + `head_left_2.mp4`, `wrist_left_2.mp4`, `wrist_right_2.mp4`. |
| `offline_eval/` | Offline car val results of this checkpoint (summarised above). |
| `MANIFEST.txt` | Source commit, uncommitted files, checkpoint sha256. |

## Setup

```bash
pip install -r starVLA_src/requirements.txt av   # torch / transformers / flash-attn as in the training env
python verify_bundle.py                          # chunk errors on the samples + latency report
python serve_policy.py --port 10093              # or use deploy_infer.CarPolicy in-process
```

Training env: torch 2.9 + CUDA 12.8, transformers 5.x (needs `Qwen3_5ForConditionalGeneration`), flash-attn 2, GH200
aarch64. Build the same versions on the robot PC.

## Interface

```python
from deploy_infer import CarPolicy
policy = CarPolicy()                           # cuda, VLM bf16 / memory + DiT fp32
policy.reset(policy.tasks[0])                  # at EVERY episode start: instruction (+ an empty memory)
out = policy.predict({"head_left_2": head, "wrist_left_2": lw, "wrist_right_2": rw}, state63, t=sec_since_start)
out["actions"]          # (50, 42) absolute @30 Hz: head(2) Larm(7) Lhand(13) Rarm(7) Rhand(13); rad / raw Inspire
out["progress"], out["memory_updated"], out["latency_ms"]   # progress: None without a memory
```

- **Images:** RGB uint8, any size; each is resized to 224 x 224 with PIL's default filter (aspect ratio not kept), as in
  training. Camera order: head (`head_left_2`), left wrist, right wrist.
- **State:** the recorded 63-D `observation.state` layout; the model reads head 0:2, left arm 15:22, right arm 22:29,
  left hand 37:50, right hand 50:63 (`stats/modality.json`), or pass `state42=`.
- **Instruction:** one of `stats/tasks.json`, word for word.
- **Memory models:** the memory advances on the first call and then once per `update_interval_s` (1.0 s; trained on
  0.7 - 1.3 s intervals); calls in between use the last memory plus the current frame without advancing it. Call
  `reset()` between episodes: the memory carries the whole episode. A model without memory ignores `t`.

## Validation

`offline_eval/` holds the car val results (per task, per chunk step, hold-command / hold-state baselines, progress for
memory models); `VERIFY.txt` the bundle's own check on the samples. Closed-loop behaviour on the robot is unverified.
