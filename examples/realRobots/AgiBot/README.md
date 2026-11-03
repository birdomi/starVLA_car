# AgiBot World 2026 for episode-memory pretraining

[AgiBot World 2026](https://huggingface.co/datasets/agibot-world/AgiBotWorld2026) is real-world data of the AgiBot G2
dual-arm robot in LeRobot v2.1 format: 30 fps AV1 video and per-episode step labels. It is licensed CC BY-NC-SA 4.0.
This directory covers downloading it, building an episode index, writing subtask labels in the format that
`starVLA/dataloader/egodex_stream.py` reads, and running Flora on it.

## Data (downloaded 2026-11-02)

| theme | episodes | hours | kept on disk |
|---|---|---|---|
| ImitationLearning | 11,694 | 273 | 168 GB |
| ReinforcementLearning (expert, HG-DAgger, rollouts) | 10,293 | 178 | 399 GB |
| PhysicalReasoning | 8,272 | 159 | 482 GB |
| RichInteraction | not downloaded: its only label is one "random interaction" segment per episode | | |

- **What is kept:** meta, parquet, and the RGB video of `top_head` (640×400), `hand_left` and `hand_right`
  (640×528). Depth (about 93% of each archive) and the fisheye cameras are skipped.
- **Location:** `/scratch/t509a04/data/AgiBotWorld2026/raw/<HF path>`.
- **Episode length:** median 48–77 s; the longest is 429 s.
- **Labels in `meta/info.json`:**
  - `instruction_segments`: step labels on the `default` track, contiguous and with accurate timing.
  - `key_frame`: (Sub)Task Frames, plus Error, Success and Intervention frames in RL.
  - The frame-type names vary between tasks ("Task Frame", "SubTask Frame", "Sub-task frame").
  - In ImitationLearning, `tasks.jsonl` holds only a collection name, so the instruction comes from the Task Frame comments.

## Pipeline

```bash
T=examples/realRobots/AgiBot/tools
D=/scratch/t509a04/data/AgiBotWorld2026
# 1. download (login node; one process per archive, see fetch_agibot.py)
python $T/fetch_agibot.py list && python $T/fetch_agibot.py plan --workers 3
for k in 0 1 2; do setsid nohup $T/fetch_driver.sh tars $D/logs/list_w$k.txt > $D/logs/fetch_w$k.log 2>&1 & done
setsid nohup $T/fetch_driver.sh pr $D/logs/list_pr.txt > $D/logs/fetch_pr.log 2>&1 &
# 2. episode index: one row per episode with an id, the three video paths and the instruction
python $T/agibot_index.py $D/raw $D/flora/index
# 3. native step labels, repeated steps suffixed "(first)", "(second)", ...
python $T/agibot_subtasks.py native $D/flora/index/index_agibot.jsonl $D/flora/subtasks_native
# 4. Flora, dex profile (top_head above both hand cameras), 8 nodes x 4 GPUs
sbatch examples/realRobots/AgiBot/train_files/hangang/agibot_flora.sbatch
python $T/agibot_subtasks.py ordinals $D/flora/subtasks_flora_dex $D/flora/subtasks_flora_dex_ord
```

`examples/realRobots/EgoDex/tools/flora_subtasks.py` runs Flora with the vLLM backend and the TempSeg and Captioner
stages. For AgiBot it reads each episode's own video paths from the index, takes `--profile dex|tianji` for three
cameras, and takes `--max-duration-s 430` to lift Flora's 120 s input guard. Flora reads the AV1 videos directly, so no
transcoding is needed.

## Flora profile choice

`train_files/hangang/agibot_flora_ab.sbatch` compares profiles on 15 sample episodes. The score is boundary agreement
with the native step labels at ±1 s:

| profile | recall | precision | s / episode (TempSeg + captions, 1 GPU) |
|---|---|---|---|
| human (top_head only) | 49% | 58% | 3.7 + 12.9 |
| **dex** (top_head + hands) | **59%** | **61%** | 7.2 + 21.7 |
| tianji (top_head + hands) | 53% | 56% | 8.2 + 22.7 |

- None of the profiles segments body motions ("bend over", "raise the waist"), which the native labels list as steps.
- dex captions say which hand holds what. They also split failed grasps into retries, which the native labels merge
  into one step.

## Repeated steps

63–72% of episodes repeat a step text, for example three coins each going through "slide / pick / adjust / place".
`agibot_subtasks.py` makes repeated candidates distinct by appending the occurrence in time order: "The left arm places
the coin into the piggy bank (second)". Texts count as the same after lowercasing and dropping punctuation and
articles. For Flora output the ordinal counts occurrences of the caption, so retries within one round also count.
