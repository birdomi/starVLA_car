"""Render rollout_preds.py output as comparison videos: front camera | GT action | predicted action.

Uses the MuJoCo kinematic replay helpers of the deploy bundle (sim/replay_mujoco.py: set_frame,
hand mapping, H1-2 + Inspire scene). Below the three panels, a strip plots head yaw and the
most-moving joint of the task arm, GT vs prediction, with a cursor; chunk boundaries (where the
model re-plans from the real observation) are ticked.

Needs mujoco + cv2 + matplotlib (e.g. the libero310 env) and MUJOCO_GL=egl when headless.
Usage: MUJOCO_GL=egl python render_rollout.py <rollout_dir> [--sim /path/to/deploy_bundle/sim]
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

ap = argparse.ArgumentParser()
ap.add_argument("rollout_dir")
ap.add_argument("--sim", default="/mnt/DATA2/yyg/starVLA/deploy_bundle_leftturn_100k/sim")
ap.add_argument("--stats", default="/mnt/DATA2/yyg/starVLA/playground/Datasets/car-few-tasks-v1/train/meta/stats_gr00t.abs.json")
ap.add_argument("--panel", default="480x360")
args = ap.parse_args()

sys.path.insert(0, args.sim)
import mujoco  # noqa: E402
import replay_mujoco as rm  # noqa: E402

W, H = (int(x) for x in args.panel.split("x"))
STRIP_H = 220
model = mujoco.MjModel.from_xml_path(str(Path(args.sim) / "assets" / "scene.xml"))
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=H, width=W)
cam = mujoco.MjvCamera()
cam.lookat[:] = [0.25, 0.0, 1.05]
cam.distance, cam.azimuth, cam.elevation = 1.9, 150, -12

st = json.load(open(args.stats))["statistics"]["action"]
amin, amax = np.array(st["min"]), np.array(st["max"])
hand_bounds = {("action", k): (amin[a:b], amax[a:b]) for k, (a, b) in (("lhand", rm.ACTION["lhand"]), ("rhand", rm.ACTION["rhand"]))}


def sim_frame(vec, state):
    rm.set_frame(model, data, vec, "action", hand_bounds=hand_bounds, hand_flip=False, legs_from=state)
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=cam)
    return renderer.render()


def label(img, text, y=26, scale=0.7, color=(255, 255, 255)):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def strip_image(gt, pred, task, stride, T):
    turn = "turn signal" in task
    arm = slice(2, 9) if turn else slice(22, 29)
    j = int(np.argmax(np.ptp(gt[:, arm], axis=0)))
    t = np.arange(T) / 30.0
    fig, axes = plt.subplots(1, 2, figsize=(3 * W / 100, STRIP_H / 100), dpi=100)
    for ax, (name, g, p) in zip(axes, [("head yaw [deg]", gt[:, 0], pred[:, 0]),
                                       (f"{'L' if turn else 'R'} arm joint {j} [deg]", gt[:, arm][:, j], pred[:, arm][:, j])]):
        ax.plot(t, np.degrees(g), color="#2b7bba", lw=1.6, label="GT")
        ax.plot(t, np.degrees(p), color="#e4572e", lw=1.2, label="pred")
        for b in range(0, T, stride):
            ax.axvline(b / 30.0, color="#bbbbbb", lw=0.5, zorder=0)
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_xlim(0, t[-1])
    axes[0].legend(fontsize=7, loc="upper right")
    fig.tight_layout(pad=0.4)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    # data-x -> pixel-x for the cursor, per axis
    xmaps = []
    for ax in axes:
        x0, x1 = ax.transData.transform([(0, 0), (t[-1], 0)])[:, 0]
        xmaps.append((x0, x1))
    plt.close(fig)
    return cv2.resize(img, (3 * W, STRIP_H)), xmaps, img.shape[1]


outs = []
for d in sorted(p for p in Path(args.rollout_dir).iterdir() if (p / "pred.npy").exists()):
    meta = json.load(open(d / "meta.json"))
    pred, gt, state = np.load(d / "pred.npy"), np.load(d / "gt_action.npy"), np.load(d / "state.npy")
    T, stride = len(gt), meta["stride"]
    strip, xmaps, src_w = strip_image(gt, pred, meta["task"], stride, T)
    sx = 3 * W / src_w
    cap = cv2.VideoCapture(str(d / "front.mp4"))
    tmp = d / "render_tmp.mp4"
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), 30, (3 * W, H + STRIP_H))
    arm = np.r_[2:9, 22:29]
    for f in range(T):
        ok, fr = cap.read()
        front = cv2.resize(fr, (W, H)) if ok else np.zeros((H, W, 3), np.uint8)
        g = cv2.cvtColor(sim_frame(gt[f], state[f]), cv2.COLOR_RGB2BGR)
        p = cv2.cvtColor(sim_frame(pred[f], state[f]), cv2.COLOR_RGB2BGR)
        label(front, f"front  t={f / 30:5.1f}s")
        label(g, "GT action", color=(255, 200, 120))
        label(p, "predicted", color=(120, 160, 255))
        err = np.degrees(np.abs(pred[f, arm] - gt[f, arm]).mean())
        label(p, f"arm err {err:4.1f} deg  head err {np.degrees(np.abs(pred[f, :2] - gt[f, :2]).mean()):4.1f}",
              y=H - 14, scale=0.55)
        if f % stride == 0:
            cv2.rectangle(p, (0, 0), (W - 1, H - 1), (0, 220, 255), 6)  # re-plan from the real observation
        top = np.concatenate([front, g, p], axis=1)
        s = cv2.cvtColor(strip, cv2.COLOR_RGB2BGR).copy()
        for x0, x1 in xmaps:
            x = int((x0 + (x1 - x0) * f / max(T - 1, 1)) * sx)
            cv2.line(s, (x, 0), (x, STRIP_H - 1), (0, 0, 0), 1)
        frame = np.concatenate([top, s], axis=0)
        if f == 0:
            label(frame, meta["task"], y=H + STRIP_H - 10, scale=0.5)
        vw.write(frame)
    vw.release()
    out = d.parent / f"{d.name}.mp4"
    title = f"{meta['task']}  |  val ep {meta['episode']}  |  arm MAE {meta['arm_mae_deg']:.2f} deg, head {meta['head_mae_deg']:.2f} deg"
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(tmp),
                    "-vf", f"pad=iw:ih+30:0:30:black,drawtext=text='{title}':x=10:y=8:fontcolor=white:fontsize=16",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(out)], check=True)
    tmp.unlink()
    outs.append(out)
    print(f"{out}  ({T} frames)", flush=True)

lst = Path(args.rollout_dir) / "concat.txt"
lst.write_text("".join(f"file '{o.resolve()}'\n" for o in outs))
subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                "-c", "copy", str(Path(args.rollout_dir) / "all_tasks.mp4")], check=True)
print("all:", Path(args.rollout_dir) / "all_tasks.mp4")
