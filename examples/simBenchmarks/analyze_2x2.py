"""2x2 (ActionMap x Keyframe) comparison of the 50k runs on episode-paired eval results.

    python examples/simBenchmarks/analyze_2x2.py robotwin [--n 20]
    python examples/simBenchmarks/analyze_2x2.py robocasa [--seed-sets 7 1000]

Episodes are paired across models by (task, scene seed): RoboCasa evals are seeded (simulation_env.py, seed + i,
PYTHONHASHSEED pinned) so every model sees identical episodes. RoboTwin uses fixed scene seeds 100000+i but skips seeds
whose expert demo fails, and that expert check is NOT deterministic (planner), so the first N valid seeds differ a little
between runs: success rates use each model's own N episodes, episode-level McNemar uses only the seeds common to the
compared models. Models with no results yet are skipped.
Prints success rates, the 2x2 contrasts (task-paired t over tasks) and pairwise exact McNemar tests, and writes
results/<bench>_2x2_summary.md.
"""
import argparse
import glob
import json
import math
import os
import re
import statistics as st

MODELS = ["DiT", "KF-DiT", "AM", "KF-AM"]   # (ActionMap, Keyframe) = (0,0) (0,1) (1,0) (1,1)
RUN_IDS = {
    "robotwin": {
        "DiT": "starvla_qwen35_0_8b_dit_rt_50k_b128_robotwin_clean50_v2",
        "KF-DiT": "starvla_qwen35_0_8b_kf_dit_rt_50k_b128_robotwin_clean50_v3",
        "AM": "starvla_qwen35_0_8b_actionmap_rt_50k_b128_robotwin_clean50_v3",
        "KF-AM": "starvla_qwen35_0_8b_kf_actionmap_rt_50k_b128_robotwin_clean50_v3",
    },
    "robocasa": {
        "DiT": "starvla_qwen35_0_8b_dit_rc100_50k_b128_robocasa_gr1_v2",
        "KF-DiT": "starvla_qwen35_0_8b_kf_dit_rc100_50k_b128_robocasa_gr1_v3",
        "AM": "starvla_qwen35_0_8b_actionmap_rc100_50k_b128_robocasa_gr1_v3",
        "KF-AM": "starvla_qwen35_0_8b_kf_actionmap_rc100_50k_b128_robocasa_gr1_v3",
    },
}
TOKENS = {"robotwin": {"DiT": 16, "KF-DiT": 4, "AM": 32, "KF-AM": 8}, "robocasa": {"DiT": 16, "KF-DiT": 4, "AM": 48, "KF-AM": 12}}


def load_robotwin(run_id, n):
    """{task: {seed: success}} from the per-seed resume cache, first n valid seeds per task."""
    out = {}
    for f in glob.glob(f"results/robotwin_resume/{run_id}/*_demo_clean_*_st*.json"):
        d = json.load(open(f))
        valid = [int(k) for k in sorted(d, key=int) if d[k]["valid"]][:n]
        if len(valid) == n:
            out[os.path.basename(f).split("_demo_clean_")[0]] = {s: int(d[str(s)]["succ"]) for s in valid}
    return out


def load_robocasa(run_id, seed_sets):
    """{task: {seed: success}} from the copied eval logs ([episode] lines), all requested scene-seed sets merged."""
    out = {}
    for base in seed_sets:
        rid = run_id if base == 7 else f"{run_id}_seed{base}"
        for f in glob.glob(f"results/eval_logs/{rid}/eval_*/eval_env_*.log"):
            t = f.split("eval_env_gr1_unified_")[1].split("_GR1")[0]
            eps = {int(s): int(v) for _, s, v in re.findall(r"\[episode\] idx=(\d+) seed=(\d+) success=(\d)", open(f, errors="ignore").read())}
            if len(eps) == 50:
                out.setdefault(t, {}).update(eps)
    return out


def mcnemar_p(b, c):
    n, k = b + c, min(b, c)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench", choices=["robotwin", "robocasa"])
    ap.add_argument("--n", type=int, default=20, help="RoboTwin episodes per task")
    ap.add_argument("--seed-sets", type=int, nargs="+", default=[7, 1000], help="RoboCasa scene-seed bases to pool")
    a = ap.parse_args()
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))  # repo root

    R = {}
    for m in MODELS:
        X = load_robotwin(RUN_IDS[a.bench][m], a.n) if a.bench == "robotwin" else load_robocasa(RUN_IDS[a.bench][m], a.seed_sets)
        if X:
            R[m] = X
    if not R:
        raise SystemExit("no results yet")
    tasks = sorted(set.intersection(*(set(X) for X in R.values())))
    common = {t: set.intersection(*(set(X[t]) for X in R.values())) for t in tasks}   # seeds evaluated by every model
    if a.bench == "robocasa":
        assert all(len(common[t]) == len(R[m][t]) for m in R for t in tasks), "RoboCasa episode sets should be identical"
    n_common = [len(common[t]) for t in tasks]
    groups = {"all": tasks}
    if a.bench == "robocasa":
        groups["Close"] = [t for t in tasks if "Close" in t]
        groups["Novel"] = [t for t in tasks if "Close" not in t]
    rate = lambda m, t: st.mean(R[m][t].values())

    lines = [f"# {a.bench} 2x2 (50k) -- {len(tasks)} tasks, models: {', '.join(R)}",
             f"rates: each model's own episodes ({len(R[next(iter(R))][tasks[0]])}/task); McNemar: seeds common to the compared pair "
             f"(all-model common: {sum(n_common)} eps, {min(n_common)}-{max(n_common)}/task)", ""]
    lines.append("| model | tokens | " + " | ".join(groups) + " |")
    lines.append("|---|---|" + "---|" * len(groups))
    for m in R:
        lines.append(f"| {m} | {TOKENS[a.bench][m]} | " + " | ".join(f"{100 * st.mean(rate(m, t) for t in ts):.1f}" for ts in groups.values()) + " |")

    contrasts = {
        "ActionMap main ((AM+KF-AM)-(DiT+KF-DiT))/2": {"AM": .5, "KF-AM": .5, "DiT": -.5, "KF-DiT": -.5},
        "Keyframe main ((KF-DiT+KF-AM)-(DiT+AM))/2": {"KF-DiT": .5, "KF-AM": .5, "DiT": -.5, "AM": -.5},
        "Interaction (KF-AM-AM)-(KF-DiT-DiT)": {"KF-AM": 1, "AM": -1, "KF-DiT": -1, "DiT": 1},
    }
    if all(m in R for m in MODELS):
        lines += ["", "| contrast (pp, task-paired t) | " + " | ".join(groups) + " |", "|---|" + "---|" * len(groups)]
        for name, w in contrasts.items():
            cells = []
            for ts in groups.values():
                x = [sum(c * rate(k, t) for k, c in w.items()) for t in ts]
                se = st.stdev(x) / math.sqrt(len(x))
                cells.append(f"{100 * st.mean(x):+.1f} (t={st.mean(x) / se:+.2f})")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")

    pairs = [("AM", "DiT"), ("KF-AM", "KF-DiT"), ("KF-DiT", "DiT"), ("KF-AM", "AM"), ("KF-AM", "DiT"), ("AM", "KF-DiT")]
    lines += ["", "| pair (episode McNemar) | " + " | ".join(groups) + " |", "|---|" + "---|" * len(groups)]
    for x, y in pairs:
        if x not in R or y not in R:
            continue
        cells = []
        for ts in groups.values():
            pair_seeds = {t: set(R[x][t]) & set(R[y][t]) for t in ts}
            b = sum(R[x][t][s] == 1 and R[y][t][s] == 0 for t in ts for s in pair_seeds[t])
            c = sum(R[x][t][s] == 0 and R[y][t][s] == 1 for t in ts for s in pair_seeds[t])
            d = [rate(x, t) - rate(y, t) for t in ts]
            npair = sum(len(v) for v in pair_seeds.values())
            cells.append(f"{100 * st.mean(d):+.1f}pp p={mcnemar_p(b, c):.4f} ({sum(v > 0 for v in d)}/{sum(v < 0 for v in d)} tasks, {b}/{c} of {npair} paired eps)")
        lines.append(f"| {x} - {y} | " + " | ".join(cells) + " |")

    lines += ["", "| task | " + " | ".join(R) + " |", "|---|" + "---|" * len(R)]
    for t in tasks:
        lines.append(f"| {t} | " + " | ".join(f"{100 * rate(m, t):.0f}" for m in R) + " |")
    text = "\n".join(lines)
    print(text)
    out = f"results/{a.bench}_2x2_summary.md"
    open(out, "w").write(text + "\n")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
