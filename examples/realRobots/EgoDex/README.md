# Multi-VLM episode-memory pretraining on EgoDex

This pretrains the recurrent `EpisodeMemory` (32 tokens, depth 8, token_dim 1024) on EgoDex human videos, with no action
model. One memory serves seven VLMs: Qwen3.5-0.8B/2B/4B, Qwen3-VL-4B, Gemma4-E2B, MiniCPM-V-4.6 and Molmo2-4B. Each VLM
has its own token-resize MLP in front of the shared memory. The memory then goes into the car VLA, for example
`QwenGR00T_EpisodeMemory` with `episode_memory.init_from`.

- **Model:** `starVLA/model/modules/dual_system/memory_pretrain.py` (`MultiVLMMemory`).
  - The memory update is M_t = f(M_{t-1}, VLM tokens of the frame and the task text).
  - A Joint Decision Transformer reads only M_t through four queries: episode progress, current subtask, finished
    subtasks and subtask progress.
  - Subtask candidates are the episode's subtask texts, encoded by a frozen flan-t5-base.
  - `export_for_vlm` writes the `episode_memory.*` tensors for one VLM.
- **Training:** `starVLA/training/train_memory_pretrain.py`.
  - Streams walk each episode from frame 0 about once per second (stride 10 ± 3 at 10 fps). TBPTT is 8 updates, and the
    memory is carried over the whole episode.
  - It runs all seven VLMs every step and evaluates on held-out test streams (`eval.only` for an evaluation-only run).
  - Logs go to W&B. It saves `steps_<k>.pt` plus a rolling `latest.pt` and resumes from the newest.
- **Data:** `starVLA/dataloader/egodex_stream.py` reads the frames written by `tools/egodex_frames.py`: 10 fps, 448×256
  JPEG in npz, plus `index_<split>.jsonl`. Subtasks come from `subtasks_*.jsonl`.
- **Config:** `train_files/memory_pretrain_multivlm_egodex.yaml`. Hangang job: `train_files/hangang/train_memory_pretrain.sbatch`,
  which queues its successor only when it ends.
- **Tests:** `tests/test_memory_decision.py`.
- **Inspection:** `tools/memory_eval_dump.py` writes per-step predictions of a checkpoint to JSON.

Results after 100K steps on 512 held-out EgoDex test episodes: progress MAE 0.047, current-subtask accuracy 77.7%,
finished-subtask accuracy 88.1%, subtask-progress MAE 0.145. On car-1015-v3 val, which was not trained on, it barely
beats the baselines (see `examples/realRobots/H1_2_car_real/memory_eval/`).

## Subtask labels

| tool | what it does |
|---|---|
| `tools/flora_subtasks.py` | NVIDIA Flora (TempSeg segmentation, then Captioner captions) through Flora's own reference pipeline on local weights. HF or vLLM backend (`flora_vllm.py`, ~20x faster). Profiles: human, or dex / tianji (three cameras). Optional episode instruction. Resumable and sharded. Videos come from `<raw>/<split>/...`, a zip, or the index row's own paths. |
| `tools/flora_merge_tempseg.py`, `tools/flora_captioner_bf16.py` | Prepare the weights the vLLM backend serves: the TempSeg LoRA merged into Qwen3.5-9B, and a bf16 captioner. |
| `tools/subtask_templates.py` | One 2–6 step template per task (vLLM VLM), aligned to every episode, with boundaries snapped to Flora's. |
| `tools/annotate_subtasks.py` | VLM subtask annotation, and `clean()`, the contiguous-cover helper used above. |

Hangang jobs:
- `train_files/hangang/egodex_prep.sbatch`: frame conversion.
- `egodex_flora.sbatch`: Flora on 16 nodes, with `flora_watchdog.*`.
- `examples/realRobots/AgiBot/` covers AgiBot.
