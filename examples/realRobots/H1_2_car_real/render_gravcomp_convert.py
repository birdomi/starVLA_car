"""MuJoCo video of the high -> low gravity-compensation command conversion on HIGH-comp episodes (true scale).

a_low = a_high - shift (per-joint constant, playground/eval_results/gravity_comp_scan/convert_check/convert_coeffs.json M1,
fitted on train static frames; grav_convert_check.py). Panels: head camera | MuJoCo side view of the working arm | close-up
that follows its hand. Solid robot = the measured joint state (observation.state); yellow outline = the recorded command
(high comp); cyan outline = the converted command, i.e. what a LOW-comp robot would have to be sent to end up at the measured
pose (it sags below its command, so the cyan arm sits above the solid one). Head, hands, legs, waist = measured in all three.
Strip: measured - command for the elbow / wrist pitch of the working arm, recorded (yellow) and converted (cyan), with the
low-comp episodes' static range on val (median, IQR band) as the target.

Needs mujoco + cv2 + matplotlib (libero310 env) and MUJOCO_GL=egl.
Usage: MUJOCO_GL=egl python render_gravcomp_convert.py --eps 240,712 --out <dir> [--test_frame 200]
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

G = "/mnt/DATA2/yyg/starVLA/playground/eval_results/gravity_comp_scan"
ap = argparse.ArgumentParser()
ap.add_argument("--eps", required=True, help="comma-separated HIGH-comp val episode ids")
ap.add_argument("--out", required=True)
ap.add_argument("--coeffs", default=f"{G}/convert_check/convert_coeffs.json")
ap.add_argument("--frames", default=f"{G}/convert_check/static_frames.npz", help="grav_convert_check.py extract output")
ap.add_argument("--dataset", default="/mnt/DATA2/yyg/starVLA/playground/Datasets/car-1015-v1-ep")
ap.add_argument("--camera", default="head_left_2")
ap.add_argument("--sim", default="/mnt/DATA2/yyg/starVLA/deploy_bundle_dualsystem_mem_48k/sim")
ap.add_argument("--panel", default="480x360")
ap.add_argument("--test_frame", type=int, default=-1)
args = ap.parse_args()

sys.path.insert(0, args.sim)
import mujoco  # noqa: E402
import replay_mujoco as rm  # noqa: E402

W, H = (int(x) for x in args.panel.split("x"))
STRIP_H = 230
ARM_S, ARM_A = np.r_[15:22, 22:29], np.r_[2:9, 22:29]
SHIFT = np.radians(json.load(open(args.coeffs))["M1_shift_deg"])
INK, MUTED, GRID = "#2b2b29", "#6b6a64", "#e4e3dd"
REC, CONV = (0, 190, 255), (255, 229, 0)  # BGR: yellow = recorded command, cyan = converted command
REC_HEX, CONV_HEX = "#d99a00", "#1baf7a"  # strip lines (dataviz yellow / aqua; readable on white)

model = mujoco.MjModel.from_xml_path(str(Path(args.sim) / "assets" / "scene.xml"))
dm, dr, dc = mujoco.MjData(model), mujoco.MjData(model), mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=H, width=W)
side_of = lambda n: (1 if n.startswith(("left_shoulder", "left_elbow", "left_wrist", "L_")) else
                     2 if n.startswith(("right_shoulder", "right_elbow", "right_wrist", "R_")) else 0)
geom_arm = np.array([side_of(model.body(b).name) for b in model.geom_bodyid])
hand_body = {1: model.body("L_middle_proximal").id, 2: model.body("R_middle_proximal").id}

root = Path(args.dataset)
st = json.load(open(root / "train/meta/stats_gr00t.abs.json"))["statistics"]["action"]
amin, amax = np.array(st["min"]), np.array(st["max"])
hand_bounds = {("state", k): (amin[a:b], amax[a:b]) for k, (a, b) in (("lhand", rm.ACTION["lhand"]), ("rhand", rm.ACTION["rhand"]))}
episodes = pd.read_parquet(root / "val/meta/episodes/chunk-000/file-000.parquet").set_index("episode_index")
task_text = {int(i): t for t, i in pd.read_parquet(root / "val/meta/tasks.parquet")["task_index"].items()}
Z = np.load(args.frames)
r_low = np.degrees(Z["qs"] - Z["qc"])[Z["split"] & Z["low"]]  # val low-comp static residuals (target range)


def load(e):
    for f in sorted(glob.glob(str(root / "val/data/chunk-*/*.parquet"))):
        df = pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state", "action"])
        df = df[df.episode_index == e].sort_values("frame_index")
        if len(df):
            return (np.stack(df["observation.state"].values).astype(np.float64),
                    np.stack(df["action"].values).astype(np.float64))
    raise SystemExit(f"episode {e}: no rows")


def outline(img, d, cam, side, color):
    renderer.enable_segmentation_rendering()
    renderer.update_scene(d, camera=cam)
    seg = renderer.render()
    renderer.disable_segmentation_rendering()
    gid = seg[..., 0]
    arm = np.where((seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (gid >= 0), geom_arm[np.clip(gid, 0, None)], 0)
    cnts, _ = cv2.findContours((arm == side).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cnts, -1, color, 2, cv2.LINE_AA)


def panels(S, A, f, side, cams):
    s = S[f]
    rec = s.copy(); rec[ARM_S] = A[f, ARM_A]
    conv = s.copy(); conv[ARM_S] = A[f, ARM_A] - SHIFT
    for d, vec in ((dm, s), (dr, rec), (dc, conv)):
        rm.set_frame(model, d, vec, "state", hand_bounds=hand_bounds, hand_flip=False)
        mujoco.mj_forward(model, d)
    cams[1].lookat[:] = dm.xpos[hand_body[side]]  # close-up follows the measured hand
    out = []
    for cam in cams:
        renderer.update_scene(dm, camera=cam)
        img = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        outline(img, dr, cam, side, REC)
        outline(img, dc, cam, side, CONV)
        out.append(img)
    hm, hr, hc = (d.xpos[hand_body[side]] * 100 for d in (dm, dr, dc))
    return out, hm, hr, hc


def label(img, text, y=26, scale=0.55, color=(255, 255, 255)):
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


out_dir = Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)
for e in [int(x) for x in args.eps.split(",")]:
    S, A = load(e)
    T = min(len(S), int(episodes.loc[e, "length"]))
    S, A = S[:T], A[:T]
    side = 2 if np.ptp(A[:, 22:29], axis=0).max() >= np.ptp(A[:, 2:9], axis=0).max() else 1
    off = 0 if side == 1 else 7
    task = task_text[int(np.asarray(episodes.loc[e, "tasks"])[0])]
    cam_side, cam_close = mujoco.MjvCamera(), mujoco.MjvCamera()
    az = 90 if side == 1 else 270  # look at the working arm from its own side
    cam_side.lookat[:] = [0.2, 0.0, 1.25]; cam_side.distance, cam_side.azimuth, cam_side.elevation = 1.3, az, -5
    cam_close.distance, cam_close.azimuth, cam_close.elevation = 0.42, az, -5
    r_rec = np.degrees(S[:, ARM_S] - A[:, ARM_A]); r_conv = r_rec + np.degrees(SHIFT)

    # strip: elbow / wrist pitch of the working arm, recorded vs converted, target = val low-comp static range
    fig, axes = plt.subplots(1, 2, figsize=(3 * W / 100, STRIP_H / 100), dpi=100)
    t = np.arange(T) / 30
    for ax, (jn, j) in zip(axes, (("elbow pitch", 3 + off), ("wrist pitch", 5 + off))):
        q1, q2, q3 = np.percentile(r_low[:, j], [25, 50, 75])
        ax.axhspan(q1, q3, color="#cde2fb", lw=0, label="low-comp episodes (val, at rest): IQR")
        ax.axhline(q2, color="#2a78d6", lw=1, ls=(0, (3, 3)), label="  … median")
        ax.plot(t, r_rec[:, j], color=REC_HEX, lw=1.3, label="recorded command (high comp)")
        ax.plot(t, r_conv[:, j], color=CONV_HEX, lw=1.3, label="converted command")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_ylim(-4, 5); ax.set_xlim(0, t[-1]); ax.tick_params(labelsize=7, colors=MUTED)
        ax.set_title(f"{'L' if side == 1 else 'R'} {jn}: measured − command (deg)", fontsize=8.5, loc="left", color=INK)
        ax.grid(axis="y", color=GRID, lw=0.7)
        for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    axes[1].legend(fontsize=7, loc="upper right", frameon=False, ncol=2)
    fig.tight_layout(pad=0.4)
    fig.canvas.draw()
    src = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    xmaps = [tuple(ax.transData.transform([(0, 0), (t[-1], 0)])[:, 0]) for ax in axes]
    plt.close(fig)
    sx = 3 * W / src.shape[1]
    strip = cv2.cvtColor(cv2.resize(src, (3 * W, STRIP_H)), cv2.COLOR_RGB2BGR)

    fi = int(episodes.loc[e, f"videos/observation.images.{args.camera}/file_index"])
    cap = cv2.VideoCapture(str(root / f"val/videos/observation.images.{args.camera}/chunk-000/file-{fi:03d}.mp4"))
    frames = range(T) if args.test_frame < 0 else [args.test_frame]
    if args.test_frame >= 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.test_frame)
    tmp = out_dir / f"ep{e:05d}.tmp.mp4"
    vw = None if args.test_frame >= 0 else cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), 30, (3 * W, H + STRIP_H))
    for f in frames:
        ok, fr = cap.read()
        front = cv2.resize(fr, (W, H)) if ok else np.zeros((H, W, 3), np.uint8)
        label(front, f"HIGH comp | ep {e} | t={f / 30:5.1f}s", scale=0.62, color=(80, 220, 255))
        (side_img, close), hm, hr, hc = panels(S, A, f, side, [cam_side, cam_close])
        label(side_img, "solid = measured (state), true scale", scale=0.5)
        label(side_img, "yellow = recorded command (high comp)", y=48, scale=0.5, color=REC)
        label(side_img, "cyan = converted command (for low comp)", y=70, scale=0.5, color=CONV)
        label(close, f"close-up, {'left' if side == 1 else 'right'} hand", scale=0.5)
        label(close, "command hand height - measured:", y=H - 58, scale=0.5)
        label(close, f"recorded {hr[2] - hm[2]:+.1f} cm", y=H - 36, scale=0.5, color=REC)
        label(close, f"converted {hc[2] - hm[2]:+.1f} cm  (low comp at rest ~ +2.3)", y=H - 12, scale=0.5, color=CONV)
        s_ = strip.copy()
        for x0, x1 in xmaps:
            x = int((x0 + (x1 - x0) * f / max(T - 1, 1)) * sx)
            cv2.line(s_, (x, 0), (x, STRIP_H - 1), (0, 0, 0), 1)
        img = np.concatenate([np.concatenate([front, side_img, close], axis=1), s_], axis=0)
        if vw is None:
            p = out_dir / f"convert_ep{e:05d}_f{f}.png"; cv2.imwrite(str(p), img); print(p)
        else:
            vw.write(img)
    if vw is None:
        continue
    vw.release()
    title = (f"{task[:60]} · HIGH-comp ep {e} · converted command a_low = a_high - shift "
             f"(elbow {np.degrees(SHIFT[3 + off]):.2f}, wrist pitch {np.degrees(SHIFT[5 + off]):.2f} deg)").replace(":", "\\:").replace("'", "")
    out = out_dir / f"convert_ep{e:05d}.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(tmp),
                    "-vf", f"pad=iw:ih+32:0:32:black,drawtext=text='{title}':x=10:y=8:fontcolor=white:fontsize=16",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart", str(out)], check=True)
    tmp.unlink()
    print(f"{out}  ({T} frames, {'left' if side == 1 else 'right'} arm)", flush=True)
