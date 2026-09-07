#!/usr/bin/env python
"""Kinematic replay of H1-2 car-task episodes in MuJoCo.

Replays observation.state (63-dim) or action (42-dim) from a bundled one-episode
LeRobot v2.0 dataset — or a predicted action trajectory from a .npy file — by
writing joint positions directly into qpos (no dynamics). Renders to an
interactive viewer or an mp4.

Vector layouts (H1-2 + Inspire DG2):
  state  (63) = head(2) + body(35) + left_hand(13) + right_hand(13)
                body: 0-11 legs, 12 waist, 13-19 left arm, 20-26 right arm, 27-34 padding
  action (42) = head(2) + left_arm(7) + left_hand(13) + right_arm(7) + right_hand(13)
Arms/head are radians. Hands are raw Inspire DG2 motor units (~1030..2040,
larger = open); channels 3 and 5 are unused (always 0) and channel 12's meaning
is unknown — the hand pose shown is an approximation for sanity-checking, not
exact kinematics.

Examples:
  # ground-truth state replay, save video
  python replay_mujoco.py --episode ../episodes/left-turn-signal --source state --video left_state.mp4
  # dataset action replay in the interactive viewer
  python replay_mujoco.py --episode ../episodes/car-sunroof-onoff --source action --viewer
  # side-by-side: GT state (left) vs model prediction (right)
  python replay_mujoco.py --episode ../episodes/left-turn-signal --pred preds.npy --compare --video cmp.mp4
"""

import argparse
import json
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- layouts
ARM_JOINTS = [
    "{side}_shoulder_pitch_joint",
    "{side}_shoulder_roll_joint",
    "{side}_shoulder_yaw_joint",
    "{side}_elbow_pitch_joint",
    "{side}_elbow_roll_joint",
    "{side}_wrist_pitch_joint",
    "{side}_wrist_yaw_joint",
]
# head channels are [yaw, pitch]: ch0 yaw (negative = right, matches the +z
# hinge as-is), ch1 pitch with positive = up, so it is negated for the +y hinge
# (positive rotation about +y tilts down).
HEAD_JOINTS = ["head_yaw_joint", "head_pitch_joint"]
HEAD_SIGN = [1.0, -1.0]
LEG_JOINTS = [
    "{side}_hip_yaw_joint",
    "{side}_hip_pitch_joint",
    "{side}_hip_roll_joint",
    "{side}_knee_joint",
    "{side}_ankle_pitch_joint",
    "{side}_ankle_roll_joint",
]
# Inspire DG2 raw channel -> MJCF finger joints. Empirically derived from the
# 0820_car_data episodes (channel correlation + which finger stays extended when
# pressing buttons): ch6-12 is a per-finger block [pinky, ring, middle, index,
# thumb_bend, thumb_mid, thumb_yaw]; ch0-5 mostly duplicates it (ch3/ch5 always
# 0) and is ignored.
HAND_CH_MAP = {
    6: ["{p}_pinky_proximal_joint", "{p}_pinky_intermediate_joint"],
    7: ["{p}_ring_proximal_joint", "{p}_ring_intermediate_joint"],
    8: ["{p}_middle_proximal_joint", "{p}_middle_intermediate_joint"],
    9: ["{p}_index_proximal_joint", "{p}_index_intermediate_joint"],
    10: ["{p}_thumb_proximal_pitch_joint"],
    11: ["{p}_thumb_intermediate_joint", "{p}_thumb_distal_joint"],
    12: ["{p}_thumb_proximal_yaw_joint"],
}

STATE = dict(head=(0, 2), legs=(2, 14), waist=14, larm=(15, 22), rarm=(22, 29), lhand=(37, 50), rhand=(50, 63))
ACTION = dict(head=(0, 2), larm=(2, 9), lhand=(9, 22), rarm=(22, 29), rhand=(29, 42))

# xr_teleoperate H1_2_ArmController position gains: non-wrist arm motors kp=140,
# wrist motors kp=50. Used by --sag to show the steady-state droop when joint
# targets are commanded WITHOUT the sol_tauff gravity feedforward.
ARM_KP = {
    "shoulder_pitch": 140.0,
    "shoulder_roll": 140.0,
    "shoulder_yaw": 140.0,
    "elbow_pitch": 140.0,
    "elbow_roll": 140.0,
    "wrist_pitch": 50.0,
    "wrist_yaw": 50.0,
}


def apply_arm_sag(mujoco, model, data, iters=3, gain=1.0):
    """Shift arm joints to the PD steady state q = q_cmd - g(q)/kp (no tau_ff).

    Uses MuJoCo's own bias force (gravity at zero velocity) so the droop is
    consistent with the model's inertias. Fixed-point iteration converges in
    a couple of steps since gravity torque varies slowly with the sag angle.
    """
    joints = [
        (model.joint(f"{side}_{name}_joint"), kp)
        for side in ("left", "right")
        for name, kp in ARM_KP.items()
    ]
    cmd = {int(j.qposadr[0]): float(data.qpos[j.qposadr[0]]) for j, _ in joints}
    data.qvel[:] = 0
    for _ in range(iters):
        mujoco.mj_forward(model, data)
        for j, kp in joints:
            adr, dof = int(j.qposadr[0]), int(j.dofadr[0])
            data.qpos[adr] = cmd[adr] - gain * data.qfrc_bias[dof] / kp
    mujoco.mj_forward(model, data)


def hand_qpos(model, raw13, prefix, lo, hi, flip):
    """raw motor units -> finger joint angles via per-channel linear map (approximate).

    lo/hi are per-channel arrays (13,) — the raw range mapped to closed..open.
    """
    raw = np.asarray(raw13, dtype=np.float64)
    span = np.maximum(np.asarray(hi) - np.asarray(lo), 1e-9)
    norm = np.clip((raw - lo) / span, 0.0, 1.0)
    # nearly-static channels: hold at their normalized start rather than flickering
    norm = np.where(span < 30, 0.5, norm)
    if not flip:  # raw high = open = joint at range-min (straight)
        norm = 1.0 - norm
    out = []
    for ch, joints in HAND_CH_MAP.items():
        for tmpl in joints:
            j = model.joint(tmpl.format(p=prefix))
            r0, r1 = j.range
            out.append((int(j.qposadr[0]), r0 + norm[ch] * (r1 - r0)))
    return out


def set_frame(model, data, vec, source, hand_bounds, hand_flip, legs_from=None):
    L = STATE if source == "state" else ACTION

    def put(names, values):
        for name, v in zip(names, values):
            data.qpos[model.joint(name).qposadr[0]] = v

    put(HEAD_JOINTS, [s * v for s, v in zip(HEAD_SIGN, vec[L["head"][0] : L["head"][1]])])
    put([n.format(side="left") for n in ARM_JOINTS], vec[L["larm"][0] : L["larm"][1]])
    put([n.format(side="right") for n in ARM_JOINTS], vec[L["rarm"][0] : L["rarm"][1]])
    for prefix, key in (("L", "lhand"), ("R", "rhand")):
        lo, hi = hand_bounds[(source, key)]
        for adr, val in hand_qpos(model, vec[L[key][0] : L[key][1]], prefix, lo, hi, hand_flip):
            data.qpos[adr] = val
    if source == "state":
        legs = vec[L["legs"][0] : L["legs"][1]]
        put([n.format(side="left") for n in LEG_JOINTS], legs[0:6])
        put([n.format(side="right") for n in LEG_JOINTS], legs[6:12])
        data.qpos[model.joint("torso_joint").qposadr[0]] = vec[L["waist"]]
    elif legs_from is not None:  # action has no legs/waist: hold them from state
        legs = legs_from[STATE["legs"][0] : STATE["legs"][1]]
        put([n.format(side="left") for n in LEG_JOINTS], legs[0:6])
        put([n.format(side="right") for n in LEG_JOINTS], legs[6:12])
        data.qpos[model.joint("torso_joint").qposadr[0]] = legs_from[STATE["waist"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", required=True, help="LeRobot v2.0 dataset dir (contains data/ meta/ videos/)")
    ap.add_argument("--ep", type=int, default=0, help="episode index within the dataset")
    ap.add_argument("--source", choices=["state", "action"], default="state")
    ap.add_argument("--pred", help=".npy file with predicted actions, shape (T, 42)")
    ap.add_argument("--compare", action="store_true", help="side-by-side: state (left) vs action/pred (right)")
    ap.add_argument("--viewer", action="store_true", help="interactive viewer instead of video")
    ap.add_argument("--video", default="replay.mp4", help="output mp4 path (video mode)")
    ap.add_argument("--size", default="960x720", help="render size WxH per panel")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier (viewer mode)")
    ap.add_argument("--hand-lo", type=float, default=None, help="override: raw hand unit mapped to fully closed (default: per-channel stats.json min/max)")
    ap.add_argument("--hand-hi", type=float, default=None, help="override: raw hand unit mapped to fully open")
    ap.add_argument("--hand-flip", action="store_true", help="invert open/close direction of the hands")
    ap.add_argument(
        "--sag",
        action="store_true",
        help="show action/pred tracks at the PD steady state without gravity feedforward "
        "(sol_tauff): arm joints droop by gravity_torque/kp (kp 140 arm / 50 wrist, "
        "from xr_teleoperate H1_2). GT state panels are never sagged.",
    )
    ap.add_argument(
        "--sag-gain",
        type=float,
        default=1.0,
        help="exaggeration factor for --sag (1.0 = physically accurate droop; >1 amplifies it for visibility)",
    )
    ap.add_argument("--camera", default="", help="named MJCF camera (e.g. head_cam); default free camera")
    ap.add_argument(
        "--views",
        default="",
        help="comma-separated sim camera presets for multi-view rendering: "
        "front, threequarter, left, right, top, hand_r, hand_l, head_cam. "
        "Multiple views render the --source track from each angle in a grid "
        "(disables --compare/--pred panels).",
    )
    ap.add_argument(
        "--cam-view",
        default="front_view",
        help="episode camera to show as the first panel (front_view / left_wrist_view / right_wrist_view / none)",
    )
    args = ap.parse_args()

    import mujoco

    ep = Path(args.episode)
    scene = Path(__file__).resolve().parent / "assets" / "scene.xml"
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    import pandas as pd

    pqs = sorted(ep.glob(f"data/chunk-*/episode_{args.ep:06d}.parquet"))
    if not pqs:
        raise SystemExit(f"episode {args.ep} not found under {ep}/data/")
    df = pd.read_parquet(pqs[0])
    states = np.stack(df["observation.state"].to_numpy())
    actions = np.stack(df["action"].to_numpy())
    fps = json.load(open(ep / "meta" / "info.json"))["fps"]
    task = "?"
    for line in (ep / "meta" / "episodes.jsonl").read_text().splitlines():
        rec = json.loads(line)
        if rec["episode_index"] == args.ep:
            task = rec["tasks"][0]
            break

    if args.pred is not None:
        preds = np.load(args.pred)
        assert preds.ndim == 2 and preds.shape[1] == 42, f"expected (T,42), got {preds.shape}"
        right = ("action", preds)
    else:
        right = ("action", actions)

    if args.compare:
        tracks = [("state", states), right]
    elif args.pred is not None:
        tracks = [right]
    else:
        tracks = [(args.source, states if args.source == "state" else actions)]

    T = min(len(t[1]) for t in tracks + [("state", states)])

    # per-channel hand normalization bounds from dataset stats (fallback: CLI/global)
    hand_bounds = {}
    try:
        stats = json.load(open(ep / "meta" / "stats.json"))
    except FileNotFoundError:
        stats = {}
    for source, feat, L in (("state", "observation.state", STATE), ("action", "action", ACTION)):
        for key in ("lhand", "rhand"):
            a, b = L[key]
            if args.hand_lo is not None and args.hand_hi is not None:
                lo = np.full(b - a, args.hand_lo)
                hi = np.full(b - a, args.hand_hi)
            elif feat in stats:
                lo = np.asarray(stats[feat]["min"][a:b], dtype=np.float64)
                hi = np.asarray(stats[feat]["max"][a:b], dtype=np.float64)
            else:
                lo = np.full(b - a, 1000.0)
                hi = np.full(b - a, 2050.0)
            hand_bounds[(source, key)] = (lo, hi)
    hand_kw = dict(hand_bounds=hand_bounds, hand_flip=args.hand_flip)
    print(f"episode: {ep.name} ep{args.ep} | task: {task!r} | {T} frames @ {fps} fps | tracks: {[t[0] for t in tracks]}")

    if args.viewer:
        if len(tracks) > 1:
            print("viewer mode shows one track; using the second (action/pred). Use --video for side-by-side.")
        src, arr = tracks[-1]
        import time

        import mujoco.viewer

        with mujoco.viewer.launch_passive(model, data) as v:
            t = 0
            while v.is_running():
                set_frame(model, data, arr[t], src, legs_from=states[t], **hand_kw)
                if args.sag and src != "state":
                    apply_arm_sag(mujoco, model, data, gain=args.sag_gain)
                mujoco.mj_forward(model, data)
                v.sync()
                time.sleep(1.0 / (fps * args.speed))
                t = (t + 1) % T
        return

    w, h = (int(x) for x in args.size.lower().split("x"))
    renderer = mujoco.Renderer(model, height=h, width=w)

    # camera presets: (azimuth, elevation, distance, lookat) or dynamic body track
    def free_cam(az, el, dist, lookat):
        c = mujoco.MjvCamera()
        c.lookat[:] = lookat
        c.distance, c.azimuth, c.elevation = dist, az, el
        return c

    VIEW_PRESETS = {
        "front": lambda: free_cam(180, -12, 1.9, [0.25, 0.0, 1.05]),
        "threequarter": lambda: free_cam(150, -12, 1.9, [0.25, 0.0, 1.05]),
        "left": lambda: free_cam(90, -10, 1.7, [0.15, 0.0, 1.05]),
        "right": lambda: free_cam(-90, -10, 1.7, [0.15, 0.0, 1.05]),
        "top": lambda: free_cam(180, -55, 2.0, [0.25, 0.0, 0.9]),
        "hand_r": ("track", "R_index_proximal", free_cam(120, -20, 0.5, [0, 0, 0])),
        "hand_l": ("track", "L_index_proximal", free_cam(-120, -20, 0.5, [0, 0, 0])),
        "head_cam": "head_cam",
    }

    views = [v.strip() for v in args.views.split(",") if v.strip()]
    for v in views:
        if v not in VIEW_PRESETS:
            raise SystemExit(f"unknown view '{v}', choose from {list(VIEW_PRESETS)}")
    if views and len(tracks) > 1:
        tracks = tracks[:1]  # multi-view renders a single track

    if args.camera:
        cam = args.camera
    else:
        cam = VIEW_PRESETS["threequarter"]()

    import cv2

    cap = None
    if args.cam_view and args.cam_view != "none":
        vids = sorted(ep.glob(f"videos/chunk-*/observation.images.{args.cam_view}/episode_{args.ep:06d}.mp4"))
        if vids:
            cap = cv2.VideoCapture(str(vids[0]))
        else:
            print(f"warning: no video found for cam '{args.cam_view}', skipping camera panel")

    def render_view(view_name):
        preset = VIEW_PRESETS[view_name]
        if isinstance(preset, str):  # named MJCF camera
            renderer.update_scene(data, camera=preset)
        elif isinstance(preset, tuple):  # body-tracking close-up
            _, body, c = preset
            c.lookat[:] = data.xpos[model.body(body).id]
            renderer.update_scene(data, camera=c)
        else:
            renderer.update_scene(data, camera=preset())
        return renderer.render()

    n_panels = (len(views) if views else len(tracks)) + (1 if cap is not None else 0)
    cols = n_panels if n_panels <= 3 else (n_panels + 1) // 2
    rows = (n_panels + cols - 1) // cols
    out_path = Path(args.video)
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w * cols, h * rows))
    for t in range(T):
        panels = []
        if cap is not None:
            ok, cam_frame = cap.read()
            if not ok:
                cam_frame = np.zeros((h, w, 3), np.uint8)
            else:
                cam_frame = cv2.cvtColor(cv2.resize(cam_frame, (w, h)), cv2.COLOR_BGR2RGB)
            cv2.putText(cam_frame, args.cam_view, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            panels.append(cam_frame)
        if views:
            src, arr = tracks[0]
            set_frame(model, data, arr[t], src, legs_from=states[t], **hand_kw)
            sagged = args.sag and src != "state"
            if sagged:
                apply_arm_sag(mujoco, model, data, gain=args.sag_gain)
            mujoco.mj_forward(model, data)
            label_src = f"{src}+sag" if sagged else src
            for v in views:
                frame = render_view(v).copy()
                cv2.putText(frame, f"{label_src}:{v}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                panels.append(frame)
        else:
            for src, arr in tracks:
                set_frame(model, data, arr[t], src, legs_from=states[t], **hand_kw)
                sagged = args.sag and src != "state"
                if sagged:
                    apply_arm_sag(mujoco, model, data, gain=args.sag_gain)
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=cam)
                frame = renderer.render()
                cv2.putText(frame, f"{src}+sag" if sagged else src, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                panels.append(frame)
        while len(panels) < cols * rows:
            panels.append(np.zeros((h, w, 3), np.uint8))
        grid = np.vstack([np.hstack(panels[r * cols : (r + 1) * cols]) for r in range(rows)])
        vw.write(cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    vw.release()
    if cap is not None:
        cap.release()
    # re-encode to H.264/yuv420p so players like VS Code can render it
    import shutil
    import subprocess

    if shutil.which("ffmpeg"):
        tmp = out_path.with_suffix(".h264.mp4")
        r = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(out_path), "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(tmp)],
        )
        if r.returncode == 0:
            tmp.replace(out_path)
    print(f"wrote {out_path.resolve()} ({T} frames, {len(tracks)} panel(s))")


if __name__ == "__main__":
    main()
