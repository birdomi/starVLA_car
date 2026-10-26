# Exact statistics of System 2's delta action window on the full RoboCasa GR1 set (24 tasks x 1000 demos).
#   window k = -40 .. 80 around every step t (20 Hz), 29-D layout = loader action keys
#   [left_arm 7, right_arm 7, left_hand 6, right_hand 6, waist 3]; delta dims = arms + waist (hands stay absolute):
#     d[0] = a[t] - s[t];  k > 0: d[k] = a[t+k] - a[t+k-1];  k < 0: d[k] = a[t+k] - a[t+k+1]
#   only slots inside the episode count (as the loader's action_valid mask). Groups: anchor (k = 0) / step (k != 0).
# Also: raw state / action min / max (atan2 range check of the sin/cos state).
#   python s2_delta_stats.py <datasets_root> <out.json>
import json, sys, glob, os
from multiprocessing import Pool
import numpy as np
import pandas as pd

ROOT, OUT = sys.argv[1], sys.argv[2]
K_MIN, K_MAX = -40, 80
SL = [(0, 7), (22, 29), (7, 13), (29, 35), (41, 44)]  # modality.json slices in loader key order
DELTA = np.zeros(29, bool); DELTA[0:14] = True; DELTA[26:29] = True


def take(x):
    return np.concatenate([x[:, a:b] for a, b in SL], axis=1)


def one_dataset(d):
    acc = {g: [np.zeros(29), np.zeros(29), 0] for g in ("anchor", "step")}
    mn = {k: np.full(29, np.inf) for k in ("state", "action", "anchor", "step")}
    mx = {k: np.full(29, -np.inf) for k in ("state", "action", "anchor", "step")}
    for p in sorted(glob.glob(os.path.join(d, "data", "chunk-*", "episode_*.parquet"))):
        df = pd.read_parquet(p, columns=["observation.state", "action"])
        s = take(np.stack(df["observation.state"].to_numpy()).astype(np.float64))
        a = take(np.stack(df["action"].to_numpy()).astype(np.float64))
        L = len(a)
        for k, v in (("state", s), ("action", a)):
            mn[k] = np.minimum(mn[k], v.min(0)); mx[k] = np.maximum(mx[k], v.max(0))
        anc = a - s  # d[0] at every t
        fwd = np.diff(a, axis=0)  # a[j] - a[j-1], j = 1 .. L-1
        # forward diff j (= a[j] - a[j-1], j = 1 .. L-1) is the future slot k = j - t of steps t = j - k >= 0,
        # k in 1 .. K_MAX -> min(K_MAX, j) windows; negated it is the past slot k = j - 1 - t of steps t <= L - 1,
        # -k in 1 .. -K_MIN -> min(-K_MIN, L - j) windows
        j = np.arange(1, L)
        n_fut = np.minimum(K_MAX, j)
        n_past = np.minimum(-K_MIN, L - j)
        for g, vals, w in (("anchor", anc, np.ones(L)), ("step_f", fwd, n_fut), ("step_p", -fwd, n_past)):
            G = "anchor" if g == "anchor" else "step"
            acc[G][0] += (vals * w[:, None]).sum(0); acc[G][1] += (vals ** 2 * w[:, None]).sum(0); acc[G][2] += w.sum()
            m = w > 0
            if m.any():
                mn[G] = np.minimum(mn[G], vals[m].min(0)); mx[G] = np.maximum(mx[G], vals[m].max(0))
    return acc, mn, mx


if __name__ == "__main__":
    ds = sorted(glob.glob(os.path.join(ROOT, "gr1_unified.*_1000")))
    assert len(ds) == 24, len(ds)
    with Pool(int(os.environ.get("NPROC", "6"))) as pool:
        res = pool.map(one_dataset, ds)
    out = {"k_range": [K_MIN, K_MAX], "delta_dims": DELTA.tolist(),
           "layout": {"left_arm": 7, "right_arm": 7, "left_hand": 6, "right_hand": 6, "waist": 3}}
    for G in ("anchor", "step"):
        s1 = sum(r[0][G][0] for r in res); s2 = sum(r[0][G][1] for r in res); n = sum(r[0][G][2] for r in res)
        mean = s1 / n; std = np.sqrt(np.maximum(s2 / n - mean ** 2, 0))
        out[G] = {"mean": mean.tolist(), "std": std.tolist(), "count": float(n),
                  "min": np.min([r[1][G] for r in res], 0).tolist(), "max": np.max([r[2][G] for r in res], 0).tolist()}
    for k in ("state", "action"):
        out[f"raw_{k}"] = {"min": np.min([r[1][k] for r in res], 0).tolist(), "max": np.max([r[2][k] for r in res], 0).tolist()}
    json.dump(out, open(OUT, "w"), indent=1)
    print("written", OUT)
