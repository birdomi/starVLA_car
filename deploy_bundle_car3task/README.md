# H1-2 car 3-task deploy bundle (starVLA QwenGR00T)

Trained 2026-09-04~07 on `0820_car_data` {left-turn-signal, right-turn-signal,
car-interiorlight-onoff}/train. Qwen3.5-0.8B VLM (unfrozen, AdamW 1e-5) +
GR00T flow-matching DiT (Muon 1e-4, WSD schedule), global batch 64,
action_mode=rel, 3 camera views, horizon 16 @ 20 Hz.

## Contents
- `checkpoints/steps_20000_pytorch_model.pt`  — 20K run (v5)
- `checkpoints/steps_100000_pytorch_model.pt` — 100K run (offline metrics equal to 20K; both included)
- `config/` — train yaml + full config snapshots + data registry (layout reference)
- `stats/<task>.stats_gr00t.rel.json` — TRAIN normalization statistics (required at inference)
- `stats/<task>.modality.json`, `<task>.tasks.jsonl` — layouts and instruction strings
- `deploy_infer.py` — robot-side wrapper: (3 images, raw state63) -> absolute (16,42) chunk
- `sim/` — MuJoCo kinematic replay + gravity_comp.py (sol_tauff feedforward)
- `sample/<task>/` — one val episode each (parquet+videos) for on-site sanity checks
- `offline_eval/` — val rollout videos/preds (GT vs prediction) + rollout script
- `starvla_src.tar.gz` — starVLA source (needed on PYTHONPATH)
- `hf_cache/` — Qwen3.5-0.8B processor/base weights for OFFLINE model build

## Setup on site
```bash
tar xzf starvla_src.tar.gz                     # -> ./starVLA
pip install -r starVLA/requirements.txt mujoco decord
export PYTHONPATH=$PWD/starVLA
export HF_HOME=$PWD/hf_cache TRANSFORMERS_OFFLINE=1   # offline model build
python deploy_infer.py left-turn-signal checkpoints/steps_100000_pytorch_model.pt
# expect: chunk (16,42), MAE ~0.04 rad on the sample, latency report
```

## Interface (deploy_infer.CarPolicy)
```python
policy = CarPolicy(task="left-turn-signal",
                   ckpt="checkpoints/steps_100000_pytorch_model.pt")
chunk = policy.predict(front_rgb, left_wrist_rgb, right_wrist_rgb, state63)
```
- state63: raw robot state, dataset layout = head(2) legs(12) waist(1)
  Larm(7) Rarm(7) pad(8) Lhand(13) Rhand(13). Normalization is internal.
- output (16,42) absolute @20Hz: head(2 rad) Larm(7 rad) Lhand(13 raw
  Inspire units) Rarm(7 rad) Rhand(13 raw). Arms clipped to train delta range
  +-0.15 rad around current state; hands clipped to train raw range.
- interiorlight has TWO prompts (on/off) — pass `prompt=` to choose
  (see stats/car-interiorlight-onoff.tasks.jsonl).

## Deployment notes
- Latency (RTX PRO 6000, shared): mean 505 ms, max 648 ms per predict.
  16-step chunk @20Hz = 800 ms budget -> run one predict per FULL chunk, or
  predict asynchronously. An 8-step receding horizon (400 ms) does NOT fit
  synchronously.
- Send arm targets WITH gravity feedforward (sim/gravity_comp.py, kp 140/50)
  or arms droop 2-4 cm.
- Chunk-boundary discontinuities exist (independent diffusion draws); the PD
  low-passes them, but linear-blending the first ~3 steps of each new chunk
  onto the previous chunk's tail is a cheap extra smoothing.
- Start pose should match the dataset home pose (see sample episodes frame 0).
- Offline val (teacher-forced, full-16 chunks): arm/head MAE 0.041-0.050 rad,
  hands 1.4-6% of range, both checkpoints equivalent. Closed-loop behavior is
  unverified — that is what this eval is for.

## Sim sanity check (no robot needed)
```bash
cd sim
MUJOCO_GL=egl python replay_mujoco.py \
  --episode ../sample/left-turn-signal --ep 0 \
  --pred ../offline_eval/pred_left_ep0_s16.npy --compare --video check.mp4 --size 640x480
```

## Not in git (transported by disk)
`checkpoints/`, `hf_cache/`, `starvla_src.tar.gz`, and `sample/*/videos/` are
not in git (size limits) — copy them alongside this directory when moving to
the robot PC.
