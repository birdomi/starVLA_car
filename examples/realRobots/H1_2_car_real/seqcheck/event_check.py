"""Task-agnostic event-sequence check of stitched rollouts (eval_val_stream.py --rollout_tasks all), every car-1015 task.

The same detectors run on the GT action and on the prediction of each episode; an episode passes a channel when the
prediction has the same events in the same order (yaw: same sides), each starting within TOL s of its GT event.
Channels (car-1015 action layout: head 0:2 yaw/pitch, L arm 2:9, L hand 9:22, R arm 22:29, R hand 29:42):
  yaw     head side looks, |yaw| > 20 deg entering, > 15 leaving (cap_lib.looks rule plus hysteresis: interior light
          tasks hold yaw at -12..-25), >= 9 frames, same-side gaps < 9 merged; turn signal / hazard looks 40-70 deg
  up      head up looks, pitch > 15 deg entering, < 10 leaving (neutral -7..5; interior light / sunroof 15-40, turn
          signal looks 10-25), >= 9 frames, gaps < 9 merged
  L/R_grasp  finger closures: hand mean (13 joints, raw) below the GT episode's open level (p90) - 200 entering,
          - 120 leaving (full grasp drops ~300-380: open ~1450-1500, closed ~1100-1150; lever touches ~100)
  L/R_stroke arm motions: speed of the 9-frame smoothed joints over +-5 frames (the stitched prediction jitters
          ~0.25 rad/s frame to frame at rest, GT ~0.03) > 0.15 rad/s, gaps < 0.5 s merged, >= 1 s, some joint sweeping
          >= 10 deg (lever pushes: wiper once 1 / twice 2 R strokes in GT)
Matching: yaw / up / grasp (level events: a split is the head / hand really leaving the pose) need the same number of
events in the same order (yaw: same sides), each starting within TOL s of its GT event; strokes (speed events: a pause
splits or merges them without any behaviour change) need every GT stroke overlapped by a predicted one and every
predicted stroke by a GT one (spans widened by TOL), i.e. no missing and no extra lever motion.
Channels with no GT event still count: the prediction must not add any. HOLD = the GT command held over each `stride`
window (pred[k*s : (k+1)*s] = gt[k*s - 1]), a lag-only reference for what the open-loop stitching alone gives.
usage: event_check.py [--tol 1.0] [--csv out.csv] <name>=<rollout dir with ep*/pred.npy, gt_action.npy, meta.json> ...
       (name GT scores gt_action against itself, HOLD the held-command reference)"""
import argparse, glob, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FPS = 30
CHANNELS = ["yaw", "up", "L_grasp", "R_grasp", "L_stroke", "R_stroke"]


def runs(mask, minlen=9, gap=9, keep=None):
    """on-runs of `mask` (or hysteresis: enter on `mask`, stay on while `keep`), gaps < `gap` merged, >= `minlen` kept"""
    if keep is not None:
        on = np.zeros(len(mask), bool); st = False
        for i in range(len(mask)): st = mask[i] or (st and keep[i]); on[i] = st
        mask = on
    idx = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0])); out = []
    for s, e in zip(idx[::2], idx[1::2]):
        if out and s - out[-1][1] < gap: out[-1] = (out[-1][0], e)
        else: out.append((s, e))
    return [(s, e) for s, e in out if e - s >= minlen]


def side_looks(yaw):
    return [(s, e, sd) for sd in (1, -1) for s, e in runs(sd * yaw > 20, keep=sd * yaw > 15)]


def strokes(x, thr=0.15, gap=15, minlen=30, rom=10, half=5):
    k = np.ones(9) / 9; xs = np.stack([np.convolve(np.pad(c, 4, mode="edge"), k, "valid") for c in x.T], 1)
    xp = np.pad(xs, ((half, half), (0, 0)), mode="edge")
    sp = np.linalg.norm(xp[2 * half:] - xp[:-2 * half], axis=1) * FPS / (2 * half)
    return [(s, e) for s, e in runs(sp > thr, minlen=1, gap=gap)
            if e - s >= minlen and np.degrees(np.ptp(x[s:e], axis=0)).max() >= rom]


def events(a, gt):
    """{channel: [(start, end, label)]}; hand thresholds from the GT episode so pred and GT share them."""
    pitch = np.degrees(a[:, 1])
    ev = {"yaw": sorted(side_looks(np.degrees(a[:, 0]))), "up": [(s, e, 0) for s, e in runs(pitch > 15, keep=pitch > 10)]}
    for nm, sl in [("L", slice(9, 22)), ("R", slice(29, 42))]:
        h, op = a[:, sl].mean(1), np.percentile(gt[:, sl].mean(1), 90)
        ev[f"{nm}_grasp"] = [(s, e, 0) for s, e in runs(h < op - 200, keep=h < op - 120)]
    for nm, sl in [("L", slice(2, 9)), ("R", slice(22, 29))]:
        ev[f"{nm}_stroke"] = [(s, e, 0) for s, e in strokes(a[:, sl])]
    return ev


def match(p, g, tol):
    if len(p) != len(g): return False
    return all(pi[2] == gi[2] and abs(pi[0] - gi[0]) <= tol * FPS for pi, gi in zip(p, g))


def overlap(p, g, tol):
    w = tol * FPS
    hit = lambda x, ys: any(x[0] < y[1] + w and y[0] - w < x[1] for y in ys)
    return all(hit(x, p) for x in g) and all(hit(x, g) for x in p)


def hold(gt, stride):
    out = np.empty_like(gt)
    for a in range(0, len(gt), stride): out[a:a + stride] = gt[max(a - 1, 0)]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--tol", type=float, default=1.0)
    ap.add_argument("--csv", default="")
    args = ap.parse_args()
    rows = []
    for arg in args.runs:
        name, d = arg.split("=", 1)
        for f in sorted(glob.glob(f"{d}/ep*/meta.json")):
            m = json.load(open(f)); ep_dir = os.path.dirname(f)
            gt = np.load(f"{ep_dir}/gt_action.npy")
            a = gt if name == "GT" else hold(gt, int(m.get("exec", m["stride"]))) if name == "HOLD" else np.load(f"{ep_dir}/pred.npy")
            ep, eg = events(a, gt), events(gt, gt)
            r = dict(model=name, ep=m["episode"], task=m["task_index"], task_text=m["task"])
            for c in CHANNELS:
                r[f"n_{c}"], r[f"gt_{c}"] = len(ep[c]), len(eg[c]); r[f"ok_{c}"] = (overlap if c.endswith("stroke") else match)(ep[c], eg[c], args.tol)
            r["ok_all"] = all(r[f"ok_{c}"] for c in CHANNELS)
            rows.append(r)
    r = pd.DataFrame(rows); models = list(dict.fromkeys(r.model))
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    oks = [f"ok_{c}" for c in CHANNELS] + ["ok_all"]
    print(f"pass rate per channel ({r.ep.nunique()} episodes, tol {args.tol} s):")
    print(r.groupby("model", sort=False)[oks].mean().round(3).to_string())
    print("\npredicted - GT event count (mean over episodes; + = extra events):")
    print(pd.DataFrame({c: (r[f"n_{c}"] - r[f"gt_{c}"]).groupby(r.model, sort=False).mean() for c in CHANNELS}).round(3).to_string())
    print("\nok_all per task:")
    t = r.pivot_table(index=["task"], columns="model", values="ok_all", aggfunc="mean")[models]
    t.insert(0, "n", r[r.model == models[0]].groupby("task").size())
    t.insert(0, "name", r.drop_duplicates("task").set_index("task").task_text.str[:44])
    print(t.round(2).to_string())
    if args.csv: r.to_csv(args.csv, index=False)
