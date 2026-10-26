"""Side-by-side rollout comparison of two models on the same episodes: front | GT | model A | model B.

Takes two rollout_preds.py output dirs whose episode sub-dirs share the episode id (ep%03d_...) and
renders, per common episode, a 4-panel MuJoCo video plus a strip with head yaw and the most-moving
task-arm joint (GT, A, B). Needs mujoco + cv2 + matplotlib (libero310 env) and MUJOCO_GL=egl.

Usage: MUJOCO_GL=egl python render_compare.py --a <dirA> --b <dirB> --labels new,old --out <dir> [--prefix-a X --prefix-b Y]
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True)
ap.add_argument("--b", required=True)
ap.add_argument("--labels", default="A,B")
ap.add_argument("--out", required=True)
ap.add_argument("--sim", default="/mnt/DATA2/yyg/starVLA/deploy_bundle_leftturn_100k/sim")
ap.add_argument("--stats", default="/mnt/DATA2/yyg/starVLA/playground/Datasets/car-turn-signal-ep/train/meta/stats_gr00t.abs.json")
ap.add_argument("--panel", default="400x300")
args = ap.parse_args()
la, lb = args.labels.split(",")

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
ARM = np.r_[2:9, 22:29]


def sim(vec, state):
    rm.set_frame(model, data, vec, "action", hand_bounds=hand_bounds, hand_flip=False, legs_from=state)
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=cam)
    return cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)


def label(img, text, y=24, scale=0.6, color=(255, 255, 255)):
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def by_ep(d):
    out = {}
    for p in Path(d).iterdir():
        m = re.search(r"ep(\d{3})_", p.name)
        if m and (p / "pred.npy").exists():
            out[int(m.group(1))] = p
    return out


A, B = by_ep(args.a), by_ep(args.b)
out_dir = Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)
outs = []
for ep in sorted(set(A) & set(B)):
    da, db = A[ep], B[ep]
    ma, mb = json.load(open(da / "meta.json")), json.load(open(db / "meta.json"))
    gt, state = np.load(da / "gt_action.npy"), np.load(da / "state.npy")
    pa, pb = np.load(da / "pred.npy"), np.load(db / "pred.npy")
    T = min(len(gt), len(pa), len(pb))
    assert np.allclose(gt[:T], np.load(db / "gt_action.npy")[:T]), f"ep{ep}: GT differs between the two dirs"
    turn = "turn signal" in ma["task"]
    arm = slice(2, 9) if turn else slice(22, 29)
    j = int(np.argmax(np.ptp(gt[:T, arm], axis=0)))
    t = np.arange(T) / 30.0
    fig, axes = plt.subplots(1, 2, figsize=(4 * W / 100, STRIP_H / 100), dpi=100)
    for ax, (name, g, x, y) in zip(axes, [("head yaw [deg]", gt[:T, 0], pa[:T, 0], pb[:T, 0]),
                                          (f"{'L' if turn else 'R'} arm joint {j} [deg]", gt[:T, arm][:, j], pa[:T, arm][:, j], pb[:T, arm][:, j])]):
        ax.plot(t, np.degrees(g), color="#2b7bba", lw=1.8, label="GT")
        ax.plot(t, np.degrees(x), color="#e4572e", lw=1.2, label=la)
        ax.plot(t, np.degrees(y), color="#3aa655", lw=1.2, label=lb)
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_xlim(0, t[-1])
    axes[0].legend(fontsize=7, loc="upper right")
    fig.tight_layout(pad=0.4)
    fig.canvas.draw()
    strip_src = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    xmaps = [tuple(ax.transData.transform([(0, 0), (t[-1], 0)])[:, 0]) for ax in axes]
    plt.close(fig)
    sx = 4 * W / strip_src.shape[1]
    strip = cv2.cvtColor(cv2.resize(strip_src, (4 * W, STRIP_H)), cv2.COLOR_RGB2BGR)
    cap = cv2.VideoCapture(str(da / "front.mp4"))
    tmp = out_dir / f"ep{ep:03d}.tmp.mp4"
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), 30, (4 * W, H + STRIP_H))
    ea = np.degrees(np.abs(pa[:T, ARM] - gt[:T, ARM]).mean())
    eb = np.degrees(np.abs(pb[:T, ARM] - gt[:T, ARM]).mean())
    for f in range(T):
        ok, fr = cap.read()
        front = cv2.resize(fr, (W, H)) if ok else np.zeros((H, W, 3), np.uint8)
        g, a, b = sim(gt[f], state[f]), sim(pa[f], state[f]), sim(pb[f], state[f])
        label(front, f"front t={f / 30:5.1f}s")
        label(g, "GT action", color=(255, 200, 120))
        label(a, la, color=(120, 160, 255))
        label(b, lb, color=(140, 230, 140))
        label(a, f"arm err {np.degrees(np.abs(pa[f, ARM] - gt[f, ARM]).mean()):4.1f} deg", y=H - 12, scale=0.5)
        label(b, f"arm err {np.degrees(np.abs(pb[f, ARM] - gt[f, ARM]).mean()):4.1f} deg", y=H - 12, scale=0.5)
        s = strip.copy()
        for x0, x1 in xmaps:
            x = int((x0 + (x1 - x0) * f / max(T - 1, 1)) * sx)
            cv2.line(s, (x, 0), (x, STRIP_H - 1), (0, 0, 0), 1)
        vw.write(np.concatenate([np.concatenate([front, g, a, b], axis=1), s], axis=0))
    vw.release()
    title = (f"val ep {ep} | {ma['task'][:60]} | arm MAE {la} {ea:.2f} / {lb} {eb:.2f} deg, "
             f"head {la} {np.degrees(np.abs(pa[:T, :2] - gt[:T, :2]).mean()):.2f} / {lb} {np.degrees(np.abs(pb[:T, :2] - gt[:T, :2]).mean()):.2f}")
    title = title.replace(":", "\\:").replace("'", "")
    out = out_dir / f"compare_ep{ep:03d}.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(tmp),
                    "-vf", f"pad=iw:ih+30:0:30:black,drawtext=text='{title}':x=10:y=8:fontcolor=white:fontsize=15",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(out)], check=True)
    tmp.unlink()
    outs.append(out.resolve())
    print(f"{out}  ({T} frames)  arm {la} {ea:.2f} / {lb} {eb:.2f} deg", flush=True)
if outs:
    lst = out_dir / "concat.txt"
    lst.write_text("".join(f"file '{o}'\n" for o in outs))
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c", "copy", str(out_dir / "compare_all.mp4")], check=True)
    print("all:", out_dir / "compare_all.mp4")
