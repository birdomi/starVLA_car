"""Per-task side-by-side eval video montage for the rc100 RoboCasa runs.

Grid per task: columns = models, row 1 = a SUCCESS episode, row 2 = a FAILURE episode (first by name).
Episodes never terminate early (always 720 steps), and the video filename flag is the FINAL-step
success while the eval log counts success at ANY step -- so the label shows both, and
final_vs_any_success.csv counts episodes that succeeded and then undid it.

python examples/simBenchmarks/Robocasa_tabletop/eval_files/make_task_compare_videos.py [out_dir] [speed]
"""
import glob
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = "playground/Checkpoints"
SUB = "videos/steps_10000_pytorch_model/n_action_steps_12_max_episode_steps_720_n_envs_1_gr1_unified"
MODELS = [
    ("DiT", "starvla_qwen35_0_8b_dit_rc100_10k_robocasa_gr1"),
    ("Keyframe DiT", "starvla_qwen35_0_8b_kf_dit_rc100_10k_robocasa_gr1"),
    ("ActionMap", "starvla_qwen35_0_8b_actionmap_rc100_10k_robocasa_gr1"),
    ("ActionMap v2", "starvla_qwen35_0_8b_actionmap_rc100_10k_robocasa_gr1_v2"),
    ("KF ActionMap", "starvla_qwen35_0_8b_kf_actionmap_rc100_10k_robocasa_gr1"),
]
W, H = 384, 240
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
OUT = sys.argv[1] if len(sys.argv) > 1 else "results/robocasa_rc100_task_videos"
SPEED = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0


def any_step_rate(run, task):
    """Success rate the eval log reports (success at ANY step of the episode)."""
    logs = sorted(glob.glob(f"{ROOT}/{run}/checkpoints/steps_10000_pytorch_model.pt.log/eval_2*/eval_env_*_{task}_gpu*.log"))
    for f in reversed(logs):
        rates = [l.rsplit(" ", 1)[-1] for l in open(f, errors="ignore") if "Success rate" in l]
        if rates:
            return float(rates[-1])
    return float("nan")


def pick(task_dir):
    succ = sorted(glob.glob(f"{task_dir}/*_success1.mp4"))
    fail = sorted(glob.glob(f"{task_dir}/*_success0.mp4"))
    return (succ[0] if succ else None), (fail[0] if fail else None), len(succ), len(succ) + len(fail)


def esc(t):
    return t.replace(":", r"\:").replace("%", r"\%")


def build(task):
    cells, stats = [], {}
    for name, run in MODELS:
        s, f, ns, n = pick(f"{ROOT}/{run}/{SUB}/{task}")
        rate = any_step_rate(run, task)
        stats[name] = (ns, n, rate)
        cells.append((name, s, f, ns, n, rate))
    inputs, filt, labels = [], [], []
    k = n_in = 0
    for row, tag in ((1, "success"), (2, "failure")):
        for name, s, f, ns, n, rate in cells:
            p = s if row == 1 else f
            txt = esc(f"{name}  any {round(rate * n)} / final {ns} of {n}" if row == 1 else f"{name}  (final-step {tag})")
            color = "0x2e7d32" if row == 1 else "0xc62828"
            if p:
                inputs += ["-i", p]
            else:
                inputs += ["-f", "lavfi", "-i", f"color=c=black:s={W}x{H}:r=10:d=1"]
                txt = esc(f"{name}  (no {tag})")
            src = f"[{n_in}:v]"
            n_in += 1
            filt.append(
                f"{src}setpts=PTS/{SPEED},scale={W}:{H},tpad=stop_mode=clone:stop_duration=400,"
                f"drawbox=x=0:y=0:w=iw:h=26:color={color}@0.85:t=fill,"
                f"drawtext=fontfile={FONT}:text='{txt}':x=6:y=5:fontsize=14:fontcolor=white[c{k}]")
            labels.append(f"[c{k}]")
            k += 1
    ncol = len(MODELS)
    layout = "|".join(f"{c * W}_{r * H}" for r in range(2) for c in range(ncol))
    title = esc(task.replace("_GR1ArmsAndWaistFourierHands_Env", ""))
    filt.append(f"{''.join(labels)}xstack=inputs={len(labels)}:layout={layout}[g]")
    filt.append(f"[g]pad=iw:ih+34:0:34:color=0x202020,"
                f"drawtext=fontfile={FONT}:text='{title}   (x{SPEED:g}, top\\: final-step success, bottom\\: final-step failure)':"
                f"x=10:y=8:fontsize=18:fontcolor=white[v]")
    # clip length = longest real input at SPEED (failures run the full 720-step budget)
    dur = 360 / 10 / SPEED + 0.5
    out = f"{OUT}/{title}.mp4"
    cmd = ["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", ";".join(filt), "-map", "[v]",
           "-t", f"{dur}", "-r", "10", "-c:v", "libx264", "-crf", "28", "-preset", "veryfast",
           "-pix_fmt", "yuv420p", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print(task, r.stderr[-800:])
    return title, stats


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    only = sys.argv[3:]
    tasks = sorted(os.path.basename(d) for d in glob.glob(f"{ROOT}/{MODELS[0][1]}/{SUB}/*")
                   if not only or any(o in d for o in only))
    with ThreadPoolExecutor(12) as ex:
        results = list(ex.map(build, tasks))
    with open(f"{OUT}/final_vs_any_success.csv", "w") as fh:
        fh.write("task," + ",".join(f"{n} any,{n} final,{n} undone" for n, _ in MODELS) + "\n")
        for title, stats in results:
            row = [title]
            for n, _ in MODELS:
                ns, tot, rate = stats[n]
                row += [f"{round(rate * tot)}", f"{ns}", f"{round(rate * tot) - ns}"]
            fh.write(",".join(row) + "\n")
    print(f"done: {len(results)} tasks -> {OUT}")
