# Episode-memory evaluation on car-1015-v3 val

This evaluates the EgoDex-pretrained memory (`examples/realRobots/EgoDex/README.md`) on car val with
the head camera.

1. `../lerobot_episodes_mp4.py`: cut the LeRobot v3 val split into one mp4 per episode in the EgoDex layout, for one
   camera (`observation.images.head_left_2`).
2. `examples/realRobots/EgoDex/tools/egodex_frames.py`: frames at 10 fps, 448×256 (`frames448/`).
3. `flora_carval.sh`: Flora subtasks with the human profile and the task text as instruction, 4 GPUs (`subtasks_flora/`).
4. `templates_carval.sh`: task templates aligned per episode (`subtasks_template/`). These were judged clearly better
   than Flora's per-episode labels on car data.
5. `eval_carval.sh` (or `eval_carval_100k.sbatch`): `train_memory_pretrain.py eval.only` for all 7 VLMs, in its own run dir
   that links the checkpoint (`EVAL_RUN`, `SUBTASKS`).

Averaged over the 7 VLMs, from 10K to 100K steps: template-label subtask accuracy went from 22.4% to 28.4% (the
always-first baseline is 20.2%) and progress MAE went from 0.189 to 0.192 (the constant-0.5 baseline is 0.259). The gains
on EgoDex did not transfer to car data.
