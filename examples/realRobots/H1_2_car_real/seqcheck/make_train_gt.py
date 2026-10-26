"""Training data of seq_check.py's ON / OFF signal classifier: the GT action trajectories + task index (0 left, 1 right) of
car-turn-signal-v2-ep/train (as built 10-15 in the session scratchpad; restored 10-21).
Usage: python make_train_gt.py [out_dir = $SEQCHECK_DATA or playground/eval_results/seqcheck]"""
import glob, os, sys, numpy as np, pandas as pd
D = "/mnt/DATA2/yyg/starVLA/playground/Datasets/car-turn-signal-v2-ep/train"
out = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SEQCHECK_DATA", "/mnt/DATA2/yyg/starVLA/playground/eval_results/seqcheck")
os.makedirs(out, exist_ok=True)
df = pd.concat(pd.read_parquet(p, columns=["episode_index", "frame_index", "action", "task_index"]) for p in sorted(glob.glob(D + "/data/*/*.parquet")))
A = {ep: (np.stack(g.sort_values("frame_index")["action"].values), g.task_index.iloc[0]) for ep, g in df.groupby("episode_index")}
np.save(f"{out}/train_actions.npy", np.array([A[e][0] for e in sorted(A)], dtype=object), allow_pickle=True)
np.save(f"{out}/train_tasks.npy", np.array([A[e][1] for e in sorted(A)]))
print(f"{len(A)} episodes -> {out}")
