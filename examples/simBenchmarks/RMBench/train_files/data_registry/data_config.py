"""RMBench (memory-dependent manipulation on RoboTwin 2.0, aloha-agilex) — mixtures.

Datasets are the LeRobot conversions written by ../convert_rmbench_to_lerobot.py into <data_root_dir>/<task>
(data_root_dir: playground/Datasets/RMBench -> /mnt/DATA2/yyg/RMBench/lerobot). They share the RoboTwin 14-D layout
and cameras, so every RoboTwin robot type applies (robotwin, robotwin_pf32, robotwin_dual, ... from
../../Robotwin/train_files/data_registry); this file only adds mixtures.

Task groups follow the RMBench paper (arXiv 2603.01229, Task Memory Complexity): M(1) needs one past observation,
M(n) several. place_block_mat ships with the repo/data but is not in the paper's 9-task table.
"""

M1_TASKS = ["observe_and_pickup", "rearrange_blocks", "put_back_block", "swap_blocks", "swap_T"]
MN_TASKS = ["battery_try", "blocks_ranking_try", "cover_blocks", "press_button"]
EXTRA_TASKS = ["place_block_mat"]


def _mix(tasks, robot_type="robotwin"):
    return [(task, 1.0, robot_type) for task in tasks]


DATASET_NAMED_MIXTURES = {
    "rmbench_all": _mix(M1_TASKS + MN_TASKS + EXTRA_TASKS),  # 10 tasks x 50 demo_clean episodes
    "rmbench_9": _mix(M1_TASKS + MN_TASKS),  # the paper's 9-task benchmark
    "rmbench_9_h50": _mix(M1_TASKS + MN_TASKS, "robotwin_h50"),  # same, 50-step action chunks
    "rmbench_m1": _mix(M1_TASKS),
    "rmbench_mn": _mix(MN_TASKS),
    **{f"rmbench_{task}": _mix([task]) for task in M1_TASKS + MN_TASKS + EXTRA_TASKS},
}
