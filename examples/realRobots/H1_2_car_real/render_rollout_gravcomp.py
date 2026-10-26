"""MuJoCo video of a streamed open-loop rollout's ARM COMMANDS against the measured robot (true scale, gravity-comp view).

Panels: head camera | MuJoCo side view of the working arm | close-up that follows its hand. Solid robot = the measured joint
state (observation.state); yellow outline = the model's predicted command (pred.npy of eval_val_stream.py); white outline =
the recorded command (GT). Under PD + gravity comp the arm rests off its command (low comp: sags ~2 cm below it), so a model
that learned the regime puts its yellow arm where the white one is, not on the solid one. Head, hands, legs, waist =
measured in all three. Strip: head yaw command (GT vs model), measured - command for the working arm's elbow pitch, and
the hand height command - measured (FK), GT vs model.

Needs mujoco + cv2 + matplotlib (libero310 env) and MUJOCO_GL=egl.
Usage: MUJOCO_GL=egl python render_rollout_gravcomp.py --rollout <eval dir>/rollout --eps 10,537 --out <dir> \
           --name "Helix v2 20k" [--eventcheck <csv>=<model>] [--test_frame 200]
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
ap.add_argument("--rollout", required=True, help="eval_val_stream.py rollout dir (ep%05d/pred.npy, gt_action.npy)")
ap.add_argument("--eps", required=True, help="comma-separated val episode ids")
ap.add_argument("--out", required=True)
ap.add_argument("--name", default="model")
ap.add_argument("--eventcheck", default="", help="<event_check csv>=<model column value>: pass/fail in the title")
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
INK, MUTED, GRID = "#2b2b29", "#6b6a64", "#e4e3dd"
PRED, GT = (0, 190, 255), (255, 255, 255)  # BGR: yellow = model command, white = recorded command
PRED_HEX = "#d99a00"  # strip line for the model (dataviz yellow, readable on white); GT = ink

model = mujoco.MjModel.from_xml_path(str(Path(args.sim) / "assets" / "scene.xml"))
dm, dg, dp = mujoco.MjData(model), mujoco.MjData(model), mujoco.MjData(model)
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
check = {}
if args.eventcheck:
    csv, mname = args.eventcheck.split("=")
    ev = pd.read_csv(csv)
    check = ev[ev.model == mname].set_index("ep").ok_all.to_dict()


def load(e):
    for f in sorted(glob.glob(str(root / "val/data/chunk-*/*.parquet"))):
        df = pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state", "action"])
        df = df[df.episode_index == e].sort_values("frame_index")
        if len(df):
            return (np.stack(df["observation.state"].values).astype(np.float64),
                    np.stack(df["action"].values).astype(np.float64))
    raise SystemExit(f"episode {e}: no rows")


def outline(img, d, cam, side, color, width=2):
    renderer.enable_segmentation_rendering()
    renderer.update_scene(d, camera=cam)
    seg = renderer.render()
    renderer.disable_segmentation_rendering()
    gid = seg[..., 0]
    arm = np.where((seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (gid >= 0), geom_arm[np.clip(gid, 0, None)], 0)
    cnts, _ = cv2.findContours((arm == side).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cnts, -1, color, width, cv2.LINE_AA)


def set_arm(d, s, q):
    v = s.copy(); v[ARM_S] = q
    rm.set_frame(model, d, v, "state", hand_bounds=hand_bounds, hand_flip=False)
    mujoco.mj_forward(model, d)


def panels(S, A, P, f, side, cams):
    set_arm(dm, S[f], S[f, ARM_S]); set_arm(dg, S[f], A[f, ARM_A]); set_arm(dp, S[f], P[f, ARM_A])
    cams[1].lookat[:] = dm.xpos[hand_body[side]]  # close-up follows the measured hand
    out = []
    for cam in cams:
        renderer.update_scene(dm, camera=cam)
        img = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        outline(img, dg, cam, side, GT)
        outline(img, dp, cam, side, PRED)
        out.append(img)
    return out


def hand_z(S, Q, side):
    z = []
    for s, q in zip(S, Q):
        set_arm(dm, s, q); z.append(dm.xpos[hand_body[side], 2] * 100)
    return np.array(z)


def label(img, text, y=26, scale=0.55, color=(255, 255, 255)):
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


out_dir = Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)
for e in [int(x) for x in args.eps.split(",")]:
    S, A = load(e)
    P, G_ = np.load(f"{args.rollout}/ep{e:05d}/pred.npy").astype(np.float64), np.load(f"{args.rollout}/ep{e:05d}/gt_action.npy")
    T = min(len(S), len(P), int(episodes.loc[e, "length"]))
    S, A, P = S[:T], A[:T], P[:T]
    assert np.abs(G_[:T] - A).max() < 1e-4, f"ep {e}: rollout GT != dataset action"
    side = 2 if np.ptp(A[:, 22:29], axis=0).max() >= np.ptp(A[:, 2:9], axis=0).max() else 1
    off = 0 if side == 1 else 7
    task = task_text[int(np.asarray(episodes.loc[e, "tasks"])[0])]
    cam_side, cam_close = mujoco.MjvCamera(), mujoco.MjvCamera()
    az = 90 if side == 1 else 270  # look at the working arm from its own side
    cam_side.lookat[:] = [0.2, 0.0, 1.25]; cam_side.distance, cam_side.azimuth, cam_side.elevation = 1.3, az, -5
    cam_close.distance, cam_close.azimuth, cam_close.elevation = 0.42, az, -5
    zm, zg, zp = (hand_z(S, Q, side) for Q in (S[:, ARM_S], A[:, ARM_A], P[:, ARM_A]))
    dz_g, dz_p = zg - zm, zp - zm

    # strip: head yaw | elbow pitch measured - command | hand height command - measured; GT (ink) vs model (yellow)
    fig, axes = plt.subplots(1, 3, figsize=(3 * W / 100, STRIP_H / 100), dpi=100)
    t = np.arange(T) / 30
    j = 3 + off
    series = (("head yaw command (deg)", np.degrees(A[:, 0]), np.degrees(P[:, 0]), (-45, 45)),
              (f"{'L' if side == 1 else 'R'} elbow pitch: measured − command (deg)",
               np.degrees(S[:, ARM_S[j]] - A[:, ARM_A[j]]), np.degrees(S[:, ARM_S[j]] - P[:, ARM_A[j]]), (-6, 6)),
              (f"{'left' if side == 1 else 'right'} hand height: command − measured (cm)", dz_g, dz_p, (-4, 4)))
    for ax, (title, yg, yp, yl) in zip(axes, series):
        ax.plot(t, yg, color=INK, lw=1.3, label="recorded command (GT)")
        ax.plot(t, yp, color=PRED_HEX, lw=1.3, label=f"{args.name} command")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_ylim(*yl); ax.set_xlim(0, t[-1]); ax.tick_params(labelsize=7, colors=MUTED)
        ax.set_title(title, fontsize=8.5, loc="left", color=INK)
        ax.grid(axis="y", color=GRID, lw=0.7)
        for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    axes[2].legend(fontsize=7, loc="upper right", frameon=False)
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
        label(front, f"ep {e} | t={f / 30:5.1f}s", scale=0.62)
        label(front, "head camera (recorded)", y=H - 12, scale=0.5)
        side_img, close = panels(S, A, P, f, side, [cam_side, cam_close])
        label(side_img, "solid = measured (state), true scale", scale=0.5)
        label(side_img, f"yellow = {args.name} command", y=48, scale=0.5, color=PRED)
        label(side_img, "white = recorded command (GT)", y=70, scale=0.5, color=GT)
        label(close, f"close-up, {'left' if side == 1 else 'right'} hand", scale=0.5)
        label(close, "command hand height - measured:", y=H - 36, scale=0.5)
        label(close, f"GT {dz_g[f]:+.1f} cm", y=H - 12, scale=0.5, color=GT)
        cv2.putText(close, f"{args.name} {dz_p[f]:+.1f} cm", (130, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
        cv2.putText(close, f"{args.name} {dz_p[f]:+.1f} cm", (130, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, PRED, 2)
        s_ = strip.copy()
        for x0, x1 in xmaps:
            x = int((x0 + (x1 - x0) * f / max(T - 1, 1)) * sx)
            cv2.line(s_, (x, 0), (x, STRIP_H - 1), (0, 0, 0), 1)
        img = np.concatenate([np.concatenate([front, side_img, close], axis=1), s_], axis=0)
        if vw is None:
            p = out_dir / f"rollout_ep{e:05d}_f{f}.png"; cv2.imwrite(str(p), img); print(p)
        else:
            vw.write(img)
    if vw is None:
        continue
    vw.release()
    res = "" if e not in check else (" | event check PASS" if check[e] else " | event check FAIL")
    title = f"{task[:70]} | {args.name} | ep {e}{res}".replace(":", "\\:").replace("'", "")
    out = out_dir / f"rollout_ep{e:05d}.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(tmp),
                    "-vf", f"pad=iw:ih+32:0:32:black,drawtext=text='{title}':x=10:y=8:fontcolor=white:fontsize=16",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart", str(out)], check=True)
    tmp.unlink()
    print(f"{out}  ({T} frames, {'left' if side == 1 else 'right'} arm)", flush=True)
