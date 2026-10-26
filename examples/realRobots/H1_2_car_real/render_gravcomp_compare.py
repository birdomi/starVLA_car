"""MuJoCo video: the same task recorded with the HIGH vs the LOW gravity compensation, side by side.

At PD rest q_meas - q_cmd = (tau_ff - g(q)) / kp, so the compensation setting shows up as a steady offset of the MEASURED
arm joints (observation.state) from the COMMANDED ones (action): high comp holds the arm at / above the command (elbow
pitch about -1 deg), low comp lets it sag (elbow pitch about +1.6 deg; playground/eval_results/gravity_comp_scan).
Per episode: the head camera | MuJoCo (deploy bundle sim/replay_mujoco: H1-2 + Inspire) with the measured arm solid and
the commanded arm as a yellow outline. A 1-3 deg offset is invisible at true scale, so the drawn arm is
q_cmd + gain * (q_meas - q_cmd) (gain --gain, offset clipped to +-30 deg); head, hands, legs and waist are the measured
state in both poses, so only the arms differ. The strip plots the true (unscaled) elbow / wrist pitch offsets.

Needs mujoco + cv2 + matplotlib (libero310 env) and MUJOCO_GL=egl.
Usage: MUJOCO_GL=egl python render_gravcomp_compare.py --high 240 --low 210 --out <file.mp4> [--gain 8] [--test_frame 300]
"""
import argparse
import glob
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
ap.add_argument("--high", type=int, required=True, help="val episode recorded with the high compensation")
ap.add_argument("--low", type=int, required=True, help="val episode recorded with the low compensation")
ap.add_argument("--out", required=True)
ap.add_argument("--gain", type=float, default=8.0, help="drawn arm offset = gain x the measured - commanded offset")
ap.add_argument("--dataset", default="/mnt/DATA2/yyg/starVLA/playground/Datasets/car-1015-v1-ep")
ap.add_argument("--camera", default="head_left_2")
ap.add_argument("--sim", default="/mnt/DATA2/yyg/starVLA/deploy_bundle_dualsystem_mem_48k/sim")
ap.add_argument("--panel", default="480x360")
ap.add_argument("--view", default="0.2,0,1.25,1.3,90,-5", help="lookat x,y,z, distance, azimuth, elevation")
ap.add_argument("--measured_only", action="store_true",
                help="draw the measured joint state only (true scale, no command outline); the strip plots the measured angles")
ap.add_argument("--test_frame", type=int, default=-1, help="write one composited frame as png instead of the video")
args = ap.parse_args()

sys.path.insert(0, args.sim)
import mujoco  # noqa: E402
import replay_mujoco as rm  # noqa: E402

W, H = (int(x) for x in args.panel.split("x"))
STRIP_H = 250
ARM_S, ARM_A = np.r_[15:22, 22:29], np.r_[2:9, 22:29]  # state / action arm joints (L, R; 3 = elbow pitch, 5 = wrist pitch)
CLIP = np.radians(30)
INK, MUTED, GRID = "#2b2b29", "#6b6a64", "#e4e3dd"
L_COL, R_COL = "#2a78d6", "#eb6834"  # dataviz default categorical slots 1-2
GHOST = (0, 190, 255)  # BGR yellow outline: the commanded arm

model = mujoco.MjModel.from_xml_path(str(Path(args.sim) / "assets" / "scene.xml"))
d_draw, d_cmd = mujoco.MjData(model), mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=H, width=W)
cam = mujoco.MjvCamera()
v = [float(x) for x in args.view.split(",")]
cam.lookat[:] = v[:3]
cam.distance, cam.azimuth, cam.elevation = v[3:]
side_of = lambda n: (1 if n.startswith(("left_shoulder", "left_elbow", "left_wrist", "L_")) else
                     2 if n.startswith(("right_shoulder", "right_elbow", "right_wrist", "R_")) else 0)
geom_arm = np.array([side_of(model.body(b).name) for b in model.geom_bodyid])  # 0 body / legs / head, 1 left arm, 2 right arm

root = Path(args.dataset)
st = json.load(open(root / "train/meta/stats_gr00t.abs.json"))["statistics"]["action"]
amin, amax = np.array(st["min"]), np.array(st["max"])
hand_bounds = {("state", k): (amin[a:b], amax[a:b]) for k, (a, b) in (("lhand", rm.ACTION["lhand"]), ("rhand", rm.ACTION["rhand"]))}
episodes = pd.read_parquet(root / "val/meta/episodes/chunk-000/file-000.parquet").set_index("episode_index")
task_text = {int(i): t for t, i in pd.read_parquet(root / "val/meta/tasks.parquet")["task_index"].items()}


def load(e):
    for f in sorted(glob.glob(str(root / "val/data/chunk-*/*.parquet"))):
        df = pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state", "action"])
        df = df[df.episode_index == e].sort_values("frame_index")
        if len(df):
            return (np.stack(df["observation.state"].values).astype(np.float64),
                    np.stack(df["action"].values).astype(np.float64))
    raise SystemExit(f"episode {e}: no rows")


def static_frames(S, A):
    """gravity_comp_scan rule: command and measurement both still over +-3 frames on every arm joint."""
    pad = lambda x: np.pad(x, ((3, 3), (0, 0)), mode="edge")
    qs, qc = S[:, ARM_S], A[:, ARM_A]
    vc = np.abs(pad(qc)[6:] - pad(qc)[:-6]).max(1) / 6
    vs = np.abs(pad(qs)[6:] - pad(qs)[:-6]).max(1) / 6
    return (vc < 0.0017) & (vs < 0.0017)


def render(S, A, f):
    s = S[f]
    if args.measured_only:
        rm.set_frame(model, d_draw, s, "state", hand_bounds=hand_bounds, hand_flip=False)
        mujoco.mj_forward(model, d_draw)
        renderer.update_scene(d_draw, camera=cam)
        return cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
    cmd = s.copy(); cmd[ARM_S] = A[f, ARM_A]
    draw = s.copy(); draw[ARM_S] = A[f, ARM_A] + np.clip(args.gain * (s[ARM_S] - A[f, ARM_A]), -CLIP, CLIP)
    for d, vec in ((d_draw, draw), (d_cmd, cmd)):
        rm.set_frame(model, d, vec, "state", hand_bounds=hand_bounds, hand_flip=False)
        mujoco.mj_forward(model, d)
    renderer.update_scene(d_draw, camera=cam)
    img = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
    renderer.enable_segmentation_rendering()
    renderer.update_scene(d_cmd, camera=cam)
    seg = renderer.render()
    renderer.disable_segmentation_rendering()
    gid = seg[..., 0]
    arm = np.where((seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (gid >= 0), geom_arm[np.clip(gid, 0, None)], 0)
    for side in (1, 2):  # one outline per arm (a union would merge the two arms where they overlap in the view)
        cnts, _ = cv2.findContours((arm == side).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, GHOST, 2, cv2.LINE_AA)
    return img


def label(img, text, y=26, scale=0.62, color=(255, 255, 255)):
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


eps = [("HIGH comp", args.high), ("LOW comp", args.low)]
data = {}
for name, e in eps:
    S, A = load(e)
    T = min(len(S), int(episodes.loc[e, "length"]))
    S, A = S[:T], A[:T]
    r = np.degrees(S[:, ARM_S] - A[:, ARM_A])
    stc = static_frames(S, A)
    data[e] = dict(S=S, A=A, T=T, r=r, static=stc, rest=r[stc].mean(0))
    print(f"{name} ep {e} ({task_text[int(np.asarray(episodes.loc[e, 'tasks'])[0])]}): {T} frames, {stc.sum()} static; "
          f"mean offset at rest elbow L {r[stc, 3].mean():+.2f} R {r[stc, 10].mean():+.2f}, wrist pitch L {r[stc, 5].mean():+.2f} "
          f"R {r[stc, 12].mean():+.2f} deg", flush=True)
T = max(dd["T"] for dd in data.values())

# strip: per episode, elbow pitch and wrist pitch offsets (true scale)
fig, axes = plt.subplots(1, 4, figsize=(4 * W / 100, STRIP_H / 100), dpi=100)
for k, (name, e) in enumerate(eps):
    dd = data[e]; t = np.arange(dd["T"]) / 30
    for j, (ax, jn, ji) in enumerate(((axes[2 * k], "elbow pitch", 3), (axes[2 * k + 1], "wrist pitch", 5))):
        if args.measured_only:
            for side, off, c in (("L", 0, L_COL), ("R", 7, R_COL)):
                ax.plot(t, np.degrees(dd["S"][:, ARM_S[ji + off]]), color=c, lw=1.3, label=f"{side} arm")
            ax.set_xlim(0, T / 30); ax.tick_params(labelsize=7, colors=MUTED)
            ax.set_title(f"{name} ep {e} · measured {jn} (deg)", fontsize=8.5, loc="left", color=INK)
            ax.grid(axis="y", color=GRID, lw=0.7)
            for s in ("top", "right"): ax.spines[s].set_visible(False)
            continue
        ax.fill_between(t, -4, 4, where=dd["static"], color="#f0efec", step="mid", lw=0)
        for side, off, c in (("L", 0, L_COL), ("R", 7, R_COL)):
            ax.plot(t, dd["r"][:, ji + off], color=c, lw=1.3, label=f"{side} arm")
            ax.axhline(dd["rest"][ji + off], color=c, lw=0.9, ls=(0, (3, 3)))
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_ylim(-4, 4); ax.set_xlim(0, T / 30); ax.tick_params(labelsize=7, colors=MUTED)
        ax.set_title(f"{name} ep {e} · {jn} measured − command (deg)", fontsize=8.5, loc="left", color=INK)
        ax.grid(axis="y", color=GRID, lw=0.7)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
axes[0].legend(fontsize=7, loc="upper right" if args.measured_only else "lower left", frameon=False, ncol=2)
axes[1].text(0.99, 0.03, "" if args.measured_only else "shaded: arm at rest · dashed: mean offset at rest", transform=axes[1].transAxes, ha="right", fontsize=7, color=MUTED)
fig.tight_layout(pad=0.4)
fig.canvas.draw()
strip_src = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
xmaps = [tuple(ax.transData.transform([(0, 0), (T / 30, 0)])[:, 0]) for ax in axes]
plt.close(fig)
sx = 4 * W / strip_src.shape[1]
strip = cv2.cvtColor(cv2.resize(strip_src, (4 * W, STRIP_H)), cv2.COLOR_RGB2BGR)

caps = {}
for _, e in eps:
    fi = int(episodes.loc[e, f"videos/observation.images.{args.camera}/file_index"])
    caps[e] = cv2.VideoCapture(str(root / f"val/videos/observation.images.{args.camera}/chunk-000/file-{fi:03d}.mp4"))
last = {e: np.zeros((H, W, 3), np.uint8) for _, e in eps}


def frame(f):
    row = []
    for name, e in eps:
        dd = data[e]; g = min(f, dd["T"] - 1)
        if f < dd["T"]:
            ok, fr = caps[e].read()
            if ok:
                last[e] = cv2.resize(fr, (W, H))
        cam_img = last[e].copy()
        label(cam_img, f"{name} | ep {e} | t={g / 30:5.1f}s", color=(80, 220, 255) if name.startswith("HIGH") else (255, 200, 90))
        sim = render(dd["S"], dd["A"], g)
        if args.measured_only:
            q = np.degrees(dd["S"][g, ARM_S])
            label(sim, f"{name}: measured joint state (true scale)", scale=0.5)
            label(sim, f"elbow pitch  L {q[3]:+.1f}  R {q[10]:+.1f} deg", y=H - 36, scale=0.5)
            label(sim, f"wrist pitch  L {q[5]:+.1f}  R {q[12]:+.1f} deg", y=H - 12, scale=0.5)
            row += [cam_img, sim]
            continue
        label(sim, f"{name}: solid = measured " + ("(true scale)" if args.gain == 1 else f"(offset x{args.gain:g})"), scale=0.5)
        label(sim, "yellow outline = command", y=48, scale=0.5, color=(0, 190, 255))
        label(sim, "solid above outline = held up, below = sagging", y=70, scale=0.5)
        r = dd["r"][g]
        label(sim, f"elbow  L {r[3]:+.1f}  R {r[10]:+.1f} deg   wrist  L {r[5]:+.1f}  R {r[12]:+.1f} deg",
              y=H - 36, scale=0.5)
        label(sim, f"mean at rest: elbow L {dd['rest'][3]:+.1f} R {dd['rest'][10]:+.1f} deg", y=H - 12, scale=0.5)
        row += [cam_img, sim]
    s = strip.copy()
    for x0, x1 in xmaps:
        x = int((x0 + (x1 - x0) * f / max(T - 1, 1)) * sx)
        cv2.line(s, (x, 0), (x, STRIP_H - 1), (0, 0, 0), 1)
    return np.concatenate([np.concatenate(row, axis=1), s], axis=0)


out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
if args.test_frame >= 0:
    for f in range(args.test_frame):
        for _, e in eps:
            if f < data[e]["T"]:
                caps[e].grab()
    for _, e in eps:
        if args.test_frame < data[e]["T"]:
            ok, fr = caps[e].read(); last[e] = cv2.resize(fr, (W, H))
            caps[e].set(cv2.CAP_PROP_POS_FRAMES, args.test_frame)
    cv2.imwrite(str(out), frame(args.test_frame)); print(out); sys.exit(0)
tmp = out.with_suffix(".tmp.mp4")
vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), 30, (4 * W, H + STRIP_H))
for f in range(T):
    vw.write(frame(f))
vw.release()
title = (f"{task_text[int(np.asarray(episodes.loc[args.high, 'tasks'])[0])][:60]} · gravity compensation HIGH (ep {args.high}) "
         f"vs LOW (ep {args.low}) · " + ("measured joint state only, true scale" if args.measured_only else
                                          ("measured vs command, true scale" if args.gain == 1 else f"arm offset drawn x{args.gain:g}, strip = true scale"))).replace(":", "\\:").replace("'", "")
subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(tmp),
                "-vf", f"pad=iw:ih+32:0:32:black,drawtext=text='{title}':x=10:y=8:fontcolor=white:fontsize=16",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart", str(out)], check=True)
tmp.unlink()
print(f"{out}  ({T} frames)", flush=True)
