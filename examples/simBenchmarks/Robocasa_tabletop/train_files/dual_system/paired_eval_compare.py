# Paired comparison of two seeded RoboCasa sim evals (same scenes): per task success, discordant pairs, exact McNemar.
#   python paired_eval_compare.py <eval_log_dir A> <eval_log_dir B> [name A] [name B]
# eval_log_dir = <ckpt>.log/eval_YYYYmmdd_HHMMSS (or its copy under results/eval_logs/<run_id>/) with eval_env_*.log
# carrying "[episode] idx= seed= success=" lines.
import glob, math, os, re, sys


def load(d):
    out = {}
    for f in glob.glob(os.path.join(d, "eval_env_*.log")):
        task = re.sub(r"_GR1.*", "", os.path.basename(f).replace("eval_env_gr1_unified_", ""))
        for line in open(f, errors="ignore"):
            m = re.search(r"\[episode\] idx=\d+ seed=(\d+) success=(\d)", line)
            if m:
                out[(task, int(m.group(1)))] = int(m.group(2))
    return out


a, b = load(sys.argv[1]), load(sys.argv[2])
na = sys.argv[3] if len(sys.argv) > 3 else "A"
nb = sys.argv[4] if len(sys.argv) > 4 else "B"
keys = sorted(set(a) & set(b))
tasks = sorted({t for t, _ in keys})
print(f"{'task':42s} {na:>10s} {nb:>10s}  (episodes)")
for t in tasks:
    k = [x for x in keys if x[0] == t]
    print(f"{t:42s} {100 * sum(a[x] for x in k) / len(k):9.1f}% {100 * sum(b[x] for x in k) / len(k):9.1f}%  ({len(k)})")
groups = {"all": tasks, "Close": [t for t in tasks if "Close" in t], "Novel": [t for t in tasks if "Novel" in t]}
for g, ts in groups.items():
    k = [x for x in keys if x[0] in ts]
    if not k:
        continue
    # per-task mean (as the CSV means), then the paired episode test
    ma = sum(sum(a[x] for x in keys if x[0] == t) / sum(1 for x in keys if x[0] == t) for t in ts) / len(ts)
    mb = sum(sum(b[x] for x in keys if x[0] == t) / sum(1 for x in keys if x[0] == t) for t in ts) / len(ts)
    n10 = sum(1 for x in k if a[x] == 1 and b[x] == 0)
    n01 = sum(1 for x in k if a[x] == 0 and b[x] == 1)
    n = n10 + n01
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(n10, n01) + 1)) / 2 ** n) if n else 1.0
    print(f"[{g}] {len(ts)} tasks / {len(k)} episodes: {na} {100 * ma:.1f}%  {nb} {100 * mb:.1f}%  "
          f"only {na} {n10}, only {nb} {n01}, exact McNemar p={p:.3g}")
print(f"(episodes in only one eval: {len(set(a) ^ set(b))})")
