"""Parallel RMBench evaluation of one starVLA checkpoint on a pool of GPUs (eval_rmbench.sh runs the tasks one by one).

Every task's episodes are split into seed shards (ROBOTWIN_EVAL_SHARD, a local patch of RMBench
scripts/eval_policy_xpolicylab.py: shard k of K keeps the seeds st_seed + k mod K and runs test_num / K episodes) that
run as independent XPolicyLab eval.sh jobs (own adapter server + simulator client, so every job carries its own
episode-memory slot). One starVLA policy server per GPU serves the jobs on that GPU (--server_gpus puts it on another
GPU, e.g. while those GPUs still train). A job whose simulator hangs (the render readback never returns: its websocket
to the adapter sits in CLOSE-WAIT, or nothing it writes grows) is killed and resumed after its last finished episode
(ROBOTWIN_EVAL_START_SEED, same patch), so a shard keeps the seeds of an uninterrupted run. A seed the simulator hangs
on in --skip_after attempts is skipped like an unstable seed (listed in results.json and summary.txt).

  python examples/simBenchmarks/RMBench/eval_files/eval_rmbench_parallel.py <ckpt_path> [--tasks 9] [--gpus 4,5,6,7]
      [--server_gpus 4,5,6,7] [--slots_per_gpu 4] [--shards 10] [--test_num 100] [--seed 0] [--log_dir DIR]

Logs in <log_dir> (default results/rmbench/<ckpt tag>_parallel): driver.log, server_gpu<g>.log, jobs/<task>_s<k>.log.
Results: summary.txt (eval_rmbench.sh lines + average), results.json (per-episode seed / success / steps).
Re-running with the same log_dir keeps the finished shards.
"""
import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
M1 = ["observe_and_pickup", "rearrange_blocks", "put_back_block", "swap_blocks", "swap_T"]
MN = ["battery_try", "blocks_ranking_try", "cover_blocks", "press_button"]
GROUPS = {"all": M1 + MN + ["place_block_mat"], "9": M1 + MN, "m1": M1, "mn": MN}
EPISODE = re.compile(r"\[episode\] seed=(\d+) success=(\d) steps=(\d+)")
EPISODE_START = re.compile(r"\[episode-start\] seed=(\d+)")


def log(msg):
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def free_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def port_open(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def group_pids(pgid):
    pids = []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            stat = open(f"/proc/{d}/stat").read()
        except OSError:
            continue
        if int(stat[stat.rindex(")") + 2:].split()[2]) == pgid:  # state, ppid, pgrp
            pids.append(int(d))
    return pids


def kill_group(proc, grace=15):
    """SIGTERM the job's process group (it was started in its own session), SIGKILL whatever is left."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            break
        deadline = time.time() + grace
        while time.time() < deadline and group_pids(proc.pid):
            time.sleep(1)
        if not group_pids(proc.pid):
            break
    try:
        proc.wait(5)
    except subprocess.TimeoutExpired:
        pass


def client_pids(pids):
    out = []
    for pid in pids:
        try:
            if b"eval_policy_xpolicylab.py" in open(f"/proc/{pid}/cmdline", "rb").read():
                out.append(pid)
        except OSError:
            pass
    return out


def close_wait_pids():
    out = subprocess.run(["ss", "-tnpH", "state", "close-wait"], capture_output=True, text=True).stdout
    return {int(p) for p in re.findall(r"pid=(\d+)", out)}


def written_bytes(pids, log_path):
    """Job log + the eval videos its ffmpeg children are writing: grows every few frames while the sim steps."""
    total = log_path.stat().st_size if log_path.exists() else 0
    for pid in pids:
        try:
            for fd in os.listdir(f"/proc/{pid}/fd"):
                target = os.readlink(f"/proc/{pid}/fd/{fd}")
                if target.endswith(".mp4"):
                    total += os.stat(target).st_size
        except OSError:
            pass
    return total


def shard_state(job_log, skip_after=2):
    """Finished episodes of a shard over all its attempts, by seed (every attempt runs the same seeds; first result
    kept), and the seeds skipped because an attempt died inside their episode skip_after times."""
    eps, stuck = {}, {}
    text = job_log.read_text(errors="replace") if job_log.exists() else ""
    for attempt in text.split("\n===== attempt "):
        finished = EPISODE.findall(attempt)
        for a, b, c in finished:
            eps.setdefault(int(a), {"seed": int(a), "success": int(b), "steps": int(c)})
        started = EPISODE_START.findall(attempt)
        if started and started[-1] not in {a for a, _, _ in finished}:
            stuck[int(started[-1])] = stuck.get(int(started[-1]), 0) + 1
    skipped = sorted(s for s, n in stuck.items() if n >= skip_after and s not in eps)
    return [eps[s] for s in sorted(eps)], skipped


class Server:
    """starVLA websocket policy server on one GPU (examples/simBenchmarks/Robotwin/eval_files/run_policy_server.sh)."""

    def __init__(self, ckpt, gpu, log_dir, env):
        self.ckpt, self.gpu, self.env = ckpt, gpu, env
        self.log_path = log_dir / f"server_gpu{gpu}.log"
        self.port = free_port()
        self.proc = None

    def start(self):
        script = REPO_ROOT / "examples/simBenchmarks/Robotwin/eval_files/run_policy_server.sh"
        self.proc = subprocess.Popen(["bash", str(script), self.ckpt, str(self.gpu), str(self.port)], env=self.env,
                                     stdout=open(self.log_path, "a"), stderr=subprocess.STDOUT,
                                     start_new_session=True)
        log(f"server gpu {self.gpu}: pid {self.proc.pid} port {self.port}")

    def wait_up(self, timeout=1200):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"starVLA server on gpu {self.gpu} exited; see {self.log_path}")
            if port_open(self.port):
                log(f"server gpu {self.gpu}: up")
                return
            time.sleep(5)
        raise RuntimeError(f"starVLA server on gpu {self.gpu} not up after {timeout} s")


class Job:
    def __init__(self, task, shard, n_shards, n_eps):
        self.task, self.shard, self.n_shards, self.n_eps = task, shard, n_shards, n_eps
        self.key = f"{task}_s{shard}"
        self.attempts = self.stalls = self.n_before = 0
        self.minutes = 0.0
        self.proc = self.gpu = None
        self.started = self.last_active = self.close_wait_since = 0.0
        self.last_bytes = -1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpt_path")
    p.add_argument("--tasks", default="9", help="a group (all | 9 | m1 | mn) or a comma list of tasks")
    p.add_argument("--gpus", default="4,5,6,7", help="simulator GPUs")
    p.add_argument("--server_gpus", default=None, help="policy server GPU per simulator GPU (default: the same)")
    p.add_argument("--slots_per_gpu", type=int, default=4, help="concurrent simulator jobs per GPU")
    p.add_argument("--shards", type=int, default=10, help="seed shards per task (must divide test_num)")
    p.add_argument("--test_num", type=int, default=100, help="episodes per task (RMBench protocol: 100)")
    p.add_argument("--seed", type=int, default=0, help="eval seed; episodes use sim seeds from 100000 * (1 + seed)")
    p.add_argument("--log_dir", default=None)
    p.add_argument("--no_video_grid", action="store_true", help="stock third-view video instead of the 2x2 grid")
    p.add_argument("--idle_min", type=float, default=20, help="kill a job when nothing it writes grows this long")
    p.add_argument("--close_wait_min", type=float, default=5,
                   help="kill a job whose client socket sits in CLOSE-WAIT with no output growth this long")
    p.add_argument("--max_attempts", type=int, default=3,
                   help="give a shard up after this many attempts in a row that finish (or skip) no episode")
    p.add_argument("--skip_after", type=int, default=2,
                   help="skip a seed after this many attempts died inside its episode (deterministic sim hang)")
    args = p.parse_args()
    assert args.test_num % args.shards == 0, "--shards must divide --test_num"

    rmbench_root = Path(os.environ.get("RMBENCH_ROOT", "/mnt/DATA2/yyg/RMBench"))
    rmbench_env = os.environ.get("RMBENCH_ENV", "RMBench")
    starvla_python = os.environ.get("STARVLA_PYTHON", str(Path.home() / "miniconda3/envs/starVLA/bin/python"))
    ckpt = str(Path(args.ckpt_path).resolve())
    tasks = GROUPS.get(args.tasks, args.tasks.split(","))
    gpus = [int(g) for g in args.gpus.split(",")]
    server_gpus = [int(g) for g in (args.server_gpus or args.gpus).split(",")]
    assert len(server_gpus) == len(gpus), "--server_gpus needs one GPU per --gpus entry"

    # run tag as eval_rmbench.sh: <run dir>_<ckpt file stem>
    run_dir = Path(ckpt).parent
    if run_dir.name in ("checkpoints", "final_model"):
        run_dir = run_dir.parent
    ckpt_tag = f"{run_dir.name}_{Path(ckpt).stem}"
    log_dir = Path(args.log_dir or REPO_ROOT / "results/rmbench" / f"{ckpt_tag}_parallel").resolve()
    (log_dir / "jobs").mkdir(parents=True, exist_ok=True)
    results_path = log_dir / "results.json"
    done = json.loads(results_path.read_text()) if results_path.exists() else {}

    env = dict(os.environ, STARVLA_PYTHON=starvla_python, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1")
    # stream episode-memory checkpoints run as trained: VLM bf16 (config dtype), memory + DiT fp32 (no --use_bf16)
    if "name: QwenGR00T_EpisodeMemory" in (run_dir / "config.yaml").read_text():
        env.setdefault("ROBOTWIN_USE_BF16", "0")
    n_eps = args.test_num // args.shards

    # longest step limit first, so the slowest tasks do not trail at the end
    limits = {}
    for line in (rmbench_root / "env_cfg/task_config/_eval_step_limit.yml").read_text().splitlines():
        m = re.match(r"(\w+):\s*(\d+)", line)
        if m:
            limits[m.group(1)] = int(m.group(2))
    order = sorted(tasks, key=lambda t: -limits.get(t, 0))
    queue = [Job(t, k, args.shards, n_eps) for t in order for k in range(args.shards)
             if f"{t}_s{k}" not in done]
    log(f"ckpt={ckpt} tag={ckpt_tag} tasks={order} gpus={gpus} slots/gpu={args.slots_per_gpu} "
        f"servers={server_gpus} shards={args.shards} x {n_eps} eps, {len(queue)} jobs to run ({len(done)} done), logs {log_dir}")

    servers = {g: Server(ckpt, sg, log_dir, env) for g, sg in zip(gpus, server_gpus)}
    running = []

    def shutdown(*_):
        log("shutting down")
        procs = [job.proc for job in running] + [s.proc for s in servers.values() if s.proc is not None]
        for proc in procs:  # SIGTERM every group at once: kill_group one by one waits ~15 s per hung job
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        time.sleep(15)
        for proc in procs:
            kill_group(proc, grace=5)
        sys.exit(1)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    if queue:
        for s in servers.values():
            s.start()
        for s in servers.values():
            s.wait_up()

    def finish(job, reason=None):
        if job.started:
            job.minutes += (time.time() - job.started) / 60
        # the log keeps every attempt; a resumed attempt continues where the killed one stopped
        episodes, skipped = shard_state(log_dir / "jobs" / f"{job.key}.log", args.skip_after)
        episodes = episodes[:job.n_eps]
        if len(episodes) == job.n_eps:
            succ = sum(e["success"] for e in episodes)
            done[job.key] = {"task": job.task, "shard": job.shard, "success": succ, "n": job.n_eps,
                             "episodes": episodes, "skipped_seeds": skipped, "attempts": job.attempts,
                             "minutes": round(job.minutes, 1)}
            results_path.write_text(json.dumps(done, indent=1))
            log(f"{job.key}: {succ}/{job.n_eps} in {done[job.key]['minutes']} min, {job.attempts} attempts "
                f"(gpu {job.gpu}); {len(done)} shards done, {len(queue)} queued, {len(running)} running")
            return
        job.stalls = 0 if len(episodes) + len(skipped) > job.n_before else job.stalls + 1
        if job.stalls < args.max_attempts:
            log(f"{job.key}: {reason or 'no final result'} at {len(episodes)}/{job.n_eps} episodes"
                f"{f', skipped seeds {skipped}' if skipped else ''} (attempt {job.attempts}, gpu {job.gpu}) -> resumed")
            queue.insert(0, job)
        else:
            log(f"{job.key}: {reason or 'no final result'}, no new episode in {job.stalls} attempts -> given up")

    last_launch = 0.0
    while queue or running:
        now = time.time()
        for g, s in servers.items():
            if s.proc is not None and s.proc.poll() is not None:
                log(f"server gpu {s.gpu} (sims on gpu {g}) died (exit {s.proc.returncode}); restarting")
                s.start()
                s.wait_up()
        cw = close_wait_pids() if running else set()
        for job in list(running):
            if job.proc.poll() is not None:
                running.remove(job)
                finish(job)
                continue
            pids = group_pids(job.proc.pid)
            nbytes = written_bytes(pids, log_dir / "jobs" / f"{job.key}.log")
            if nbytes != job.last_bytes:
                job.last_bytes, job.last_active = nbytes, now
            if cw & set(client_pids(pids)):
                job.close_wait_since = job.close_wait_since or now
            else:
                job.close_wait_since = 0.0
            idle = now - job.last_active
            hung = None
            if job.close_wait_since and min(now - job.close_wait_since, idle) > 60 * args.close_wait_min:
                hung = f"client socket in CLOSE-WAIT, idle {idle / 60:.0f} min"
            elif idle > 60 * args.idle_min:
                hung = f"idle {idle / 60:.0f} min"
            if hung:
                log(f"{job.key}: hung ({hung}), killing pgid {job.proc.pid}")
                kill_group(job.proc)
                running.remove(job)
                finish(job, reason=f"killed: {hung}")
        # launch into free slots, one every 5 s (XPolicyLab picks its adapter port with a racy free-port probe)
        if queue and now - last_launch > 5:
            load = {g: sum(j.gpu == g for j in running) for g in gpus}
            g = min(gpus, key=lambda x: load[x])
            if load[g] < args.slots_per_gpu:
                job = queue.pop(0)
                eps, skipped = shard_state(log_dir / "jobs" / f"{job.key}.log", args.skip_after)
                if len(eps) >= job.n_eps:  # killed (or the driver stopped) after its last episode
                    finish(job)
                    continue
                job.attempts += 1
                job.n_before = len(eps) + len(skipped)
                job.gpu, job.started, job.last_active, job.last_bytes, job.close_wait_since = g, now, now, -1, 0.0
                args_file = log_dir / "jobs" / f"{job.key}.args"
                args_file.write_text(f"--test_num\n{job.n_eps - len(eps)}\n")
                job_env = dict(env, CUDA_VISIBLE_DEVICES=str(g), STARVLA_SERVER_PORT=str(servers[g].port),
                               ROBOTWIN_EVAL_ARGS_FILE=str(args_file),
                               ROBOTWIN_EVAL_SHARD=f"{job.shard}/{job.n_shards}")
                if eps or skipped:  # the shard filter skips to this shard's next seed
                    job_env["ROBOTWIN_EVAL_START_SEED"] = str(max([e["seed"] for e in eps] + skipped) + 1)
                if not args.no_video_grid:
                    job_env["ROBOTWIN_EVAL_VIDEO_GRID"] = "1"
                job_log = open(log_dir / "jobs" / f"{job.key}.log", "a")
                job_log.write(f"\n===== attempt {job.attempts} gpu {g} {time.strftime('%m-%d %H:%M:%S')}, "
                              f"{len(eps)}/{job.n_eps} episodes done, skipped seeds {skipped} =====\n")
                job_log.flush()
                job.proc = subprocess.Popen(
                    ["bash", "XPolicyLab/policy/starVLA_dev/eval.sh", "RMBench", job.task, ckpt_tag, "aloha_agilex",
                     "joint", str(args.seed), str(g), "0", rmbench_env, rmbench_env],
                    cwd=rmbench_root, env=job_env, stdout=job_log, stderr=subprocess.STDOUT, start_new_session=True)
                job_log.close()
                running.append(job)
                last_launch = now
                continue  # fill the remaining slots on the next 5 s tick
        time.sleep(5 if queue and len(running) < args.slots_per_gpu * len(gpus) else 20)

    for s in servers.values():
        if s.proc is not None:
            kill_group(s.proc)

    lines, rates = [], []
    for t in tasks:
        shards = [done.get(f"{t}_s{k}") for k in range(args.shards)]
        if all(shards):
            succ, n = sum(s["success"] for s in shards), sum(s["n"] for s in shards)
            n_skip = sum(len(s.get("skipped_seeds", [])) for s in shards)
            rates.append(100 * succ / n)
            lines.append(f"{t}: Final success rate: {succ}/{n} = {100 * succ / n:.1f}%"
                         + (f" ({n_skip} seeds skipped: simulator hung)" if n_skip else ""))
        else:
            lines.append(f"{t}: INCOMPLETE ({sum(map(bool, shards))}/{args.shards} shards)")
    if len(rates) == len(tasks):
        lines.append(f"average over {len(tasks)} tasks: {sum(rates) / len(rates):.1f}%")
    (log_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    log("done\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
