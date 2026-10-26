"""MuJoCo comparison videos of eval_val_stream.py rollouts: head camera | GT action | model A | model B (| ...).

Reads the stitched trajectories of eval_val_stream.py (<run>/rollout/ep{id:05d}/{pred.npy, gt_action.npy, meta.json}),
the recorded head camera and the 63-D state (legs / waist, seated) of the val episode, and replays every action track
kinematically with the deploy bundle's MuJoCo helpers (sim/replay_mujoco.py: H1-2 + Inspire scene, head yaw/pitch
joints added). Each panel shows where the head looks (|yaw| > 20 deg = side look), how many side looks it has made
so far and a top-down gaze dial (up = forward, left = left; ticks at +-20 deg; gray needle = GT); a strip below plots head yaw / pitch and the most-moving task-arm joint with a time cursor. The run's
seqcheck.csv (auto_eval_stream_car1015.sh), if present, adds its pass / fail to the title.
Open loop: every 0.5 s the model re-plans from the RECORDED observation, so the tracks only show what it would command.

Needs mujoco + cv2 + matplotlib (libero310 env) and MUJOCO_GL=egl.
Usage: MUJOCO_GL=egl python render_stream_compare.py --run "baseline 30k=<run dir>" --run "memory 10k=<run dir>" \
           --eps 83,129 --out <dir>
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", action="append", required=True, help="'label=<eval_val_stream out dir>' (repeat)")
ap.add_argument("--eps", required=True, help="comma-separated val episode ids")
ap.add_argument("--out", required=True)
ap.add_argument("--dataset", default="/mnt/DATA2/yyg/starVLA/playground/Datasets/car-1015-v1-ep")
ap.add_argument("--camera", default="head_left_2")
ap.add_argument("--sim", default="/mnt/DATA2/yyg/starVLA/deploy_bundle_dualsystem_mem_48k/sim")
ap.add_argument("--panel", default="480x360")
ap.add_argument("--eventcheck", default="", help="'<event_check.py csv>=<model name per --run, comma-separated>': "
                "title verdicts from the all-channel event check instead of the runs' turn-signal seqcheck.csv")
args = ap.parse_args()

sys.path.insert(0, args.sim)
sys.path.insert(0, str(Path(__file__).resolve().parent / "seqcheck"))
import mujoco  # noqa: E402
import replay_mujoco as rm  # noqa: E402
from cap_lib import looks  # noqa: E402

runs = [r.split("=", 1) for r in args.run]
# GT ink + categorical slots 1-2 of the dataviz default palette (validated light: CVD dE 24.7)
COLORS = ["#eb6834", "#2a78d6", "#1baf7a", "#eda100"]
GT_INK = "#2b2b29"
bgr = lambda h: tuple(int(h[i:i + 2], 16) for i in (5, 3, 1))

W, H = (int(x) for x in args.panel.split("x"))
STRIP_H = 230
model = mujoco.MjModel.from_xml_path(str(Path(args.sim) / "assets" / "scene.xml"))
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=H, width=W)
cam = mujoco.MjvCamera()
cam.lookat[:] = [0.1, 0.0, 1.42]  # head at z 1.65 (seated), both arms in view
cam.distance, cam.azimuth, cam.elevation = 1.35, 160, -14
root = Path(args.dataset)
st = json.load(open(root / "train/meta/stats_gr00t.abs.json"))["statistics"]["action"]
amin, amax = np.array(st["min"]), np.array(st["max"])
hand_bounds = {("action", k): (amin[a:b], amax[a:b]) for k, (a, b) in (("lhand", rm.ACTION["lhand"]), ("rhand", rm.ACTION["rhand"]))}
episodes = pd.read_parquet(root / "val/meta/episodes/chunk-000/file-000.parquet").set_index("episode_index")
seq = {}
for lab, d in runs:
    f = Path(d) / "seqcheck.csv"
    if f.exists():
        s = pd.read_csv(f)
        seq[lab] = s[s.model != "GT"].set_index("ep")
CHECK = "seq check"
if args.eventcheck:
    f, names = args.eventcheck.rsplit("=", 1)
    s = pd.read_csv(f)
    for (lab, _), nm in zip(runs, names.split(",")):
        seq[lab] = s[s.model == nm].set_index("ep").rename(columns={"n_yaw": "n_looks"})
    CHECK = "event check"


def sim(vec, state):
    rm.set_frame(model, data, vec, "action", hand_bounds=hand_bounds, hand_flip=False, legs_from=state)
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=cam)
    return cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)


def label(img, text, y=26, scale=0.62, swatch=None):
    x = 8
    if swatch is not None:
        cv2.rectangle(img, (8, y - 15), (26, y + 3), swatch, -1); x = 34
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)


def state_of(e):
    row = episodes.loc[e]; T = int(row["length"])
    files = sorted((root / "val/data").glob("chunk-*/*.parquet"))
    for f in files:
        df = pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state"])
        df = df[df.episode_index == e]
        if len(df):
            return np.stack(df.sort_values("frame_index")["observation.state"].values).astype(np.float64)[:T]
    raise SystemExit(f"episode {e}: no state rows")


def dial(img, yaw, color, gt_yaw=None, r=46):
    cx, cy = img.shape[1] - r - 14, r + 16
    cv2.ellipse(img, (cx, cy), (r, r), 0, 180, 360, (60, 60, 60), -1)
    cv2.ellipse(img, (cx, cy), (r, r), 0, 180, 360, (200, 200, 200), 1)
    for a in (-20, 20):
        x, y = -np.sin(np.radians(a)), -np.cos(np.radians(a))
        cv2.line(img, (int(cx + 0.8 * r * x), int(cy + 0.8 * r * y)), (int(cx + r * x), int(cy + r * y)), (200, 200, 200), 1)
    for v, c, w in ([(gt_yaw, (170, 170, 170), 2)] if gt_yaw is not None else []) + [(yaw, color, 3)]:
        x, y = -np.sin(np.radians(v)), -np.cos(np.radians(v))
        cv2.line(img, (cx, cy), (int(cx + r * x), int(cy + r * y)), c, w, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 4, (230, 230, 230), -1)


def look_text(yaw_deg):
    return "LEFT look" if yaw_deg > 20 else "RIGHT look" if yaw_deg < -20 else "center"


out_dir = Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)
for e in [int(x) for x in args.eps.split(",")]:
    dirs = [Path(d) / "rollout" / f"ep{e:05d}" for _, d in runs]
    meta = json.load(open(dirs[0] / "meta.json"))
    gt = np.load(dirs[0] / "gt_action.npy")
    preds = [np.load(d / "pred.npy") for d in dirs]
    for d in dirs[1:]:
        assert np.allclose(np.load(d / "gt_action.npy"), gt), f"ep{e}: GT differs between runs"
    state = state_of(e)
    T = min(len(gt), len(state), *(len(p) for p in preds))
    tracks = [("GT action", gt, GT_INK)] + [(lab, p, COLORS[i]) for i, ((lab, _), p) in enumerate(zip(runs, preds))]
    yaws = [np.degrees(tr[1][:T, 0]) for tr in tracks]
    onsets = [[s for s, _, _ in looks(y)] for y in yaws]
    turn = "turn signal" in meta["task"]
    arm = slice(2, 9) if turn or np.ptp(gt[:T, 2:9]) > np.ptp(gt[:T, 22:29]) else slice(22, 29)
    j = int(np.argmax(np.ptp(gt[:T, arm], axis=0)))
    t = np.arange(T) / 30.0
    n_pan = 1 + len(tracks)
    fig, axes = plt.subplots(1, 3, figsize=(n_pan * W / 100, STRIP_H / 100), dpi=100)
    for ax, (name, col) in zip(axes, [("head yaw (deg)", 0), ("head pitch (deg)", 1),
                                      (f"{'left' if arm.start == 2 else 'right'} arm joint {j} (deg)", arm.start + j)]):
        for lab, tr, c in tracks:
            ax.plot(t, np.degrees(tr[:T, col]), color=c, lw=2.0 if lab == "GT action" else 1.3, label=lab)
        if col == 0:
            for y in (20, -20): ax.axhline(y, color="#8a8984", lw=0.7, ls=(0, (3, 3)))
        ax.set_title(name, fontsize=9, loc="left"); ax.tick_params(labelsize=7); ax.set_xlim(0, t[-1])
        ax.grid(axis="y", color="#e4e3dd", lw=0.7)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=8, loc="upper right", frameon=False, ncol=len(tracks))
    fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.9))
    fig.canvas.draw()
    strip_src = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    xmaps = [tuple(ax.transData.transform([(0, 0), (t[-1], 0)])[:, 0]) for ax in axes]
    plt.close(fig)
    sx = n_pan * W / strip_src.shape[1]
    strip = cv2.cvtColor(cv2.resize(strip_src, (n_pan * W, STRIP_H)), cv2.COLOR_RGB2BGR)
    fi = int(episodes.loc[e, f"videos/observation.images.{args.camera}/file_index"])
    cap = cv2.VideoCapture(str(root / f"val/videos/observation.images.{args.camera}/chunk-000/file-{fi:03d}.mp4"))
    tmp = out_dir / f"ep{e:05d}.tmp.mp4"
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), 30, (n_pan * W, H + STRIP_H))
    for f in range(T):
        ok, fr = cap.read()
        front = cv2.resize(fr, (W, H)) if ok else np.zeros((H, W, 3), np.uint8)
        label(front, f"{args.camera}  t={f / 30:5.1f}s")
        panels = [front]
        for k, (lab, tr, c) in enumerate(tracks):
            img = sim(tr[f], state[f])
            label(img, lab, swatch=bgr(c))
            dial(img, yaws[k][f], (255, 255, 255) if k == 0 else bgr(c), None if k == 0 else yaws[0][f])
            n_so_far = sum(s <= f for s in onsets[k])
            label(img, f"{look_text(yaws[k][f])} | side looks so far: {n_so_far}", y=H - 14, scale=0.55)
            panels.append(img)
        s = strip.copy()
        for x0, x1 in xmaps:
            x = int((x0 + (x1 - x0) * f / max(T - 1, 1)) * sx)
            cv2.line(s, (x, 0), (x, STRIP_H - 1), (0, 0, 0), 1)
        vw.write(np.concatenate([np.concatenate(panels, axis=1), s], axis=0))
    vw.release()
    verdict = " | ".join(f"{lab} {int(seq[lab].loc[e, 'n_looks'])} looks {'PASS' if seq[lab].loc[e, 'ok_all'] else 'FAIL'}"
                         for lab, _ in runs if lab in seq and e in seq[lab].index)
    title = f"val ep {e} | {meta['task'][:58]} | GT {len(onsets[0])} side looks" + (f" | {CHECK}: {verdict}" if verdict else "")
    title = title.replace(":", "\\:").replace("'", "")
    out = out_dir / f"stream_compare_ep{e:05d}.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(tmp),
                    "-vf", f"pad=iw:ih+32:0:32:black,drawtext=text='{title}':x=10:y=8:fontcolor=white:fontsize=16",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart", str(out)], check=True)
    tmp.unlink()
    print(f"{out}  ({T} frames)", flush=True)
