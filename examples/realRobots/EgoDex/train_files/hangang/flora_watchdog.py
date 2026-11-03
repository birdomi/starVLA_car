#!/usr/bin/env python3
"""Keeps egodex_prep.sbatch's Flora shards (node k x GPU g) going until every episode has a usable row.

Every --every seconds, from the login node:
  * a shard is complete when every episode of it (index_<split>_<k>of<n>.jsonl, every 4th from g) has a TempSeg row and
    a subtasks row that is no engine failure (flora_subtasks.read_paths: dead vLLM engine, CUDA OOM, "missing");
  * a process whose latest log ends on a dead engine (it may hang or write fallbacks) is killed;
  * an incomplete shard with no flora_subtasks.py process of its tag left on its node is rerun there as an --overlap
    step: orphaned vLLM engine processes (parent gone) on that node are killed first, then the TempSeg and caption
    stages run again with a per-process Triton cache (a shared one races: KeyError 'cubin') and a lower
    --gpu-memory-utilization; they skip what is done.
  python3 flora_watchdog.py <egodex_prep job id> [--mem 0.75] [--every 300]   (log: prep_logs/watchdog.log)
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/scratch/t509a04/workspace/starVLA_car")
TOOLS = REPO / "examples/realRobots/EgoDex/tools"
D = Path("/scratch/t509a04/data/EgoDex")
FRAMES, OUT, LOG = D / "frames448", D / "subtasks_flora", D / "prep_logs"
SPLITS = "test,part1,part2,part3,part4,part5,extra"
sys.path.insert(0, str(TOOLS))
from flora_subtasks import read_paths  # noqa: E402


def sh(cmd, timeout=120):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout).stdout


def shard_paths(k, n, g):
    paths = []
    for sp in SPLITS.split(","):
        f = FRAMES / f"index_{sp}_{k}of{n}.jsonl"
        if f.exists():
            paths += [json.loads(l)["path"] for l in open(f) if l.strip()]
    return paths[g::4]


def latest_log(k, g, tag):
    logs = [p for p in (LOG / f"flora_n{k}g{g}.log", LOG / f"flora_repair_n{k}g{g}.log", LOG / f"flora_rerun_{tag}.log")
            if p.exists()]
    return max(logs, key=lambda p: p.stat().st_mtime) if logs else None


def engine_dead(log) -> bool:
    if log is None:
        return False
    with open(log, "rb") as f:
        f.seek(max(0, log.stat().st_size - 4000))
        tail = f.read().decode(errors="replace")
    return any(k in tail for k in ("EngineDeadError", "Engine core initialization failed", "CUDA out of memory"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("--mem", type=float, default=0.75)
    ap.add_argument("--every", type=int, default=300)
    a = ap.parse_args()
    n = int(sh(f"squeue -j {a.job} -h -o %D").strip())
    hosts = sh(f'scontrol show hostnames "$(squeue -j {a.job} -h -o %N)"').split()
    print(f"[{time.ctime()}] watching {a.job}: {n} nodes", flush=True)
    while " R " in f" {sh(f'squeue -j {a.job} -h -o %t').strip()} ":
        todo = {}
        for k in range(n):
            for g in range(4):
                tag = f"n{k}of{n}g{g}"
                want = set(shard_paths(k, n, g))
                good = read_paths(OUT / f"flora_segments_{tag}.jsonl") & read_paths(OUT / f"subtasks_{tag}.jsonl")
                if want and not want <= good:
                    todo.setdefault(k, []).append((g, tag, len(want - good)))
        if not todo:
            print(f"[{time.ctime()}] every shard complete", flush=True)
            break
        for k, shards in todo.items():
            alive = sh(f"srun --jobid={a.job} --overlap -w {hosts[k]} -N1 -n1 --mem=1G bash -c "
                       f"'ps -eo args | grep \"[f]lora_subtasks.py\" | grep -o \"\\-\\-tag [a-z0-9]*\"'")
            running = alive.split("\n")
            for g, tag, left in shards:
                if f"--tag {tag}" in running and engine_dead(latest_log(k, g, tag)):
                    print(f"[{time.ctime()}] kill {tag} on {hosts[k]}: its engine died", flush=True)
                    sh(f"srun --jobid={a.job} --overlap -w {hosts[k]} -N1 -n1 --mem=1G bash -c "
                       f"'p=$(pgrep -f \"[f]lora_subtasks.py.*--tag {tag} \"); [ -n \"$p\" ] && kill -9 $p; sleep 3'")
                    running.remove(f"--tag {tag}")
            dead = [(g, tag, left) for g, tag, left in shards if f"--tag {tag}" not in running]
            if not dead:
                continue
            # engines whose flora process is gone hold their GPU: kill them (a live one's parent is alive)
            sh(f"srun --jobid={a.job} --overlap -w {hosts[k]} -N1 -n1 --mem=1G bash -c "
               f"'o=$(ps -eo pid,ppid,args | awk \"\\$2==1 && /VLLM::EngineCore/ {{print \\$1}}\"); "
               f"[ -n \"$o\" ] && kill -9 $o; sleep 5'")
            for g, tag, left in dead:
                print(f"[{time.ctime()}] rerun {tag} on {hosts[k]} GPU {g}: {left} episodes left", flush=True)
                stages = " && ".join(
                    f"/home01/t509a04/envs/vllm129/bin/python {TOOLS}/flora_subtasks.py {FRAMES} {D}/zips {OUT} "
                    f"--split {SPLITS} --index 'index_{{split}}_{k}of{n}.jsonl' --shard {g} --num-shards 4 --tag {tag} "
                    f"--instruction description --backend vllm --stage {st} --threads 64 "
                    f"--gpu-memory-utilization {a.mem} --tmp /dev/shm/flora_rerun_{tag}" for st in ("tempseg", "caption"))
                cmd = (f"export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn "
                       f"CUDA_VISIBLE_DEVICES={g} TRITON_CACHE_DIR=/tmp/triton_{tag}; cd {REPO}; {stages}")
                subprocess.Popen(["setsid", "srun", f"--jobid={a.job}", "--overlap", "-w", hosts[k], "-N1", "-n1",
                                  "--cpus-per-task=32", "--mem=200G", "bash", "-c", cmd],
                                 stdout=open(LOG / f"flora_rerun_{tag}.log", "a"), stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL)
        time.sleep(a.every)
    print(f"[{time.ctime()}] watchdog ends", flush=True)


if __name__ == "__main__":
    main()
