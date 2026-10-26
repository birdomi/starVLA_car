"""Does a rollout do signal ON -> look signal-side mirror -> look other side -> head back -> signal OFF, per episode?
Arm phases from left-arm speed alone (head-independent): long segments (smoothed joint speed > 0.15 rad/s, gaps < 0.5 s merged,
>= 1.5 s, some joint sweeps >= 20 deg; GT ON/OFF p5 27 deg, arm sway during looks p90 14 deg); ON = first one, OFF = the last group (segments after ON with gaps < 2 s). Each phase, as pose relative to frame 0,
resampled to 20 points x 7 joints, goes to a ridge classifier trained on the v2 train GT (5-fold CV: ON 99.2%, OFF 98.9%) that
says which signal's motion it is. Order: ON starts before look 1; OFF starts no earlier than 1.5 s before look 2 ends (train GT
p1 = -1.2 s).
usage: seq_check.py <name>=<rollout dir with ep*/pred.npy, gt_action.npy, meta.json> ...   (name 'GT' scores gt_action)"""
import glob, json, os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cap_lib import looks
# classifier training data (make_train_gt.py) + the last result csv; restored 10-21 from the session transcript after
# the /tmp scratchpad was wiped by a reboot
HERE = os.environ.get("SEQCHECK_DATA", "/mnt/DATA2/yyg/starVLA/playground/eval_results/seqcheck")
FPS = 30; EARLY = 3.0
def segs(a, thr=0.15, gap=15, minlen=45, rom=20):
    arm = a[:, 2:9]; sp = np.r_[0, np.linalg.norm(np.diff(arm, axis=0), axis=1)] * 30; sm = np.convolve(sp, np.ones(9) / 9, "same")
    idx = np.flatnonzero(np.diff(np.r_[0, (sm > thr).astype(np.int8), 0])); out = []
    for s, e in zip(idx[::2], idx[1::2]):
        if out and s - out[-1][1] < gap: out[-1] = (out[-1][0], e)
        else: out.append((s, e))
    return [(s, e) for s, e in out if e - s >= minlen and np.degrees(np.ptp(arm[s:e], axis=0)).max() >= rom]
def phases(a, G=60):
    S = segs(a)
    if not S: return None, None
    rest = S[1:]
    if not rest: return S[0], None
    i = len(rest) - 1
    while i > 0 and rest[i][0] - rest[i - 1][1] < G: i -= 1
    return S[0], (rest[i][0], rest[-1][1])
def feat(a, s, e, n=20):
    seg = np.degrees(a[s:e, 2:9] - a[0, 2:9]); return seg[np.linspace(0, len(seg) - 1, n).round().astype(int)].ravel()
def fit():
    A = np.load(f"{HERE}/train_actions.npy", allow_pickle=True); Tk = np.load(f"{HERE}/train_tasks.npy"); cls = []
    for wi in [0, 1]:
        X, y = [], []
        for a, t in zip(A, Tk):
            ph = phases(a)[wi]
            if ph: X.append(feat(a, *ph)); y.append(t)
        X, y = np.array(X), np.array(y); mu, sd = X.mean(0), X.std(0) + 1e-6; Z = np.c_[(X - mu) / sd, np.ones(len(X))]
        w = np.linalg.solve(Z.T @ Z + 100 * np.eye(Z.shape[1]), Z.T @ (2 * y - 1)); cls.append((mu, sd, w))
    return cls
CLS = fit()
def score(x, wi):  # > 0 -> right-signal motion, < 0 -> left-signal motion
    mu, sd, w = CLS[wi]; return float(np.r_[(x - mu) / sd, 1] @ w)
def check(a, task, gt=None):
    """gt given (rollout): arm phases are cut at the GT's own ON/OFF spans (the offline rollout feeds recorded images, so the arm
    follows the recording's timing; small S1 jitter must not merge/split the speed segments) and the classifier says whose
    motion the prediction makes there; order = head back from look 2 no later than 1 s after the GT OFF motion starts."""
    yaw = np.degrees(a[:, 0]); side = 1 if task == 0 else -1; T = len(a)
    ph = phases(a if gt is None else gt)
    for nm, wi in [("on", 0), ("off", 1)]:
        p = ph[wi]; r = r if wi else {}; r[f"{nm}_t"] = p[0] / FPS if p else np.nan; r[f"{nm}_end"] = p[1] / FPS if p else np.nan
        r[f"{nm}_score"] = score(feat(a, *p), wi) if p else np.nan
        r[f"ok_{nm}"] = bool(p) and (r[f"{nm}_score"] > 0) == (task == 1)
    # a look starting < EARLY s after the arm starts switching ON comes before the lever is pushed (train GT: first look >= 5.3 s)
    lk_all = looks(yaw); t_on = r["on_t"] if np.isfinite(r["on_t"]) else 0.0
    early = [l for l in lk_all if l[0] / FPS < t_on + EARLY]
    # looks starting after the OFF motion has ended belong to after the task (reported, not judged)
    t_done = r["off_end"] if np.isfinite(r["off_end"]) else np.inf
    post = [l for l in lk_all if l[0] / FPS >= t_done]; lk = [l for l in lk_all if t_on + EARLY <= l[0] / FPS < t_done]
    r["n_looks"] = len(lk); r["n_early"] = len(early); r["n_after_task"] = len(post)
    r["early_t"] = early[0][0] / FPS if early else np.nan; r["ok_not_early"] = not early
    r["look1_on"] = lk[0][0] / FPS if lk else np.nan; r["look1_end"] = lk[0][1] / FPS if lk else np.nan
    r["look2_on"] = lk[1][0] / FPS if len(lk) > 1 else np.nan; r["look2_end"] = lk[1][1] / FPS if len(lk) > 1 else np.nan
    r["ok_look1"] = bool(lk) and lk[0][2] == side
    r["ok_look2"] = len(lk) > 1 and lk[1][2] == -side
    r["ok_center"] = len(lk) > 1 and lk[1][1] < min(T, t_done * FPS) - 5 and np.abs(yaw[lk[1][1]:int(min(T, t_done * FPS))]).max() < 20
    tol = 1.0 if gt is not None else 1.5
    r["ok_order"] = bool(ph[0] and ph[1] and len(lk) > 1) and r["on_t"] < r["look1_on"] and r["look2_end"] <= r["off_t"] + tol
    r["ok_all"] = all(r[k] for k in ["ok_on", "ok_not_early", "ok_look1", "ok_look2", "ok_center", "ok_off", "ok_order"]) and len(lk) == 2
    return r
if __name__ == "__main__":
    out = []
    for arg in sys.argv[1:]:
        name, d = arg.split("=", 1)
        for e in sorted(glob.glob(f"{d}/ep*/meta.json")):
            m = json.load(open(e)); ep_dir = e.rsplit("/", 1)[0]; tl = m["task"].lower()
            if "turn signal" not in tl or "three-flash" in tl: continue  # all-task rollout dirs: plain ON/OFF signal tasks only
            task = 0 if "left" in tl else 1
            gt = np.load(f"{ep_dir}/gt_action.npy"); a = gt if name == "GT" else np.load(f"{ep_dir}/pred.npy")
            r = check(a, task, None if name == "GT" else gt); g = check(gt, task)
            for k in ["on_end", "look1_on", "look1_end", "look2_end", "off_t"]: r[f"d_{k}"] = r[k] - g[k]
            out.append(dict(model=name, ep=m["episode"], task=task, **r))
    r = pd.DataFrame(out); oks = [c for c in r if c.startswith("ok_")]
    pd.set_option("display.width", 250)
    print("pass rate per step (60 val eps):"); print(r.groupby("model", sort=False)[oks].mean().round(2).to_string())
    print("\nlooks found:", r.groupby("model", sort=False).n_looks.value_counts().unstack(fill_value=0).to_string())
    print("\nepisodes with a look after the task (after OFF ended):", r.groupby("model", sort=False).n_after_task.apply(lambda x: int((x > 0).sum())).to_dict())
    print("\nmedian timing vs GT (s, + = later than GT):")
    print(r.groupby("model", sort=False)[[c for c in r if c.startswith("d_")]].median().round(2).to_string())
    r.to_csv(f"{HERE}/seq_check_last.csv", index=False)
    bad = r[(r.model != "GT") & ~r.ok_all]
    print("\nfailing episodes:"); print(bad[["model", "ep", "task", "n_looks"] + oks[:-1]].to_string(index=False))
