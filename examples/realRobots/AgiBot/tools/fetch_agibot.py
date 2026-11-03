#!/usr/bin/env python3
"""AgiBot World 2026 (HF agibot-world/AgiBotWorld2026) without Rich Interaction, the RGB of the three main cameras only
(top_head, hand_left, hand_right), parquet and meta, into <root>/raw/<HF path> (downloaded 2026-11-02: 30,259 episodes,
610 h, 1.05 TB of 13 TB).

  list:  the repo tree (path, size) -> <root>/files.json, read by the other modes (the Hub tree of the 800k-file repo is
         listed once, not per call)
  plan:  the archives not yet done -> <root>/logs/list_w<k>.txt for --workers drivers, the PhysicalReasoning dirs ->
         <root>/logs/list_pr.txt
  tars:  ImitationLearning / ReinforcementLearning .tar.gz (245, ~9 TB; gzip, so each is read whole): downloaded to
         <root>/tmp_w<k>/, meta, parquet and the three cameras extracted to raw/<path without .tar.gz>/ (depth and
         fisheye skipped), checked (one parquet and one mp4 per camera per episode of meta/info.json), marked .done and
         the archive deleted; worker k of n takes every n-th, or --only one archive. Resumable (.done is skipped).
  pr:    PhysicalReasoning (unpacked LeRobot on the Hub): the same files of one --sub dir, files on disk with the
         listed size skipped.
Login nodes cap a process at 2 h of CPU time (ulimit -t 7200) and kill it silently, so fetch_driver.sh runs one process
per archive / PhysicalReasoning dir:
  python fetch_agibot.py list && python fetch_agibot.py plan --workers 3
  for k in 0 1 2; do setsid nohup fetch_driver.sh tars <root>/logs/list_w$k.txt > <root>/logs/fetch_w$k.log 2>&1 & done
  setsid nohup fetch_driver.sh pr <root>/logs/list_pr.txt > <root>/logs/fetch_pr.log 2>&1 &
"""
import argparse
import fnmatch
import glob
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "agibot-world/AgiBotWorld2026"
THEMES = ("ImitationLearning", "ReinforcementLearning", "PhysicalReasoning")
TAR_THEMES = ("ImitationLearning", "ReinforcementLearning")
CAMS = ("top_head", "hand_left", "hand_right")
KEEP = ["*/meta/*", "*.parquet"] + [f"*observation.images.{c}/*" for c in CAMS]


def log(msg):
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def list_files(root: Path):
    api, files = HfApi(), []
    for theme in THEMES:
        for x in api.list_repo_tree(REPO, repo_type="dataset", path_in_repo=theme, recursive=True):
            if type(x).__name__ == "RepoFile":
                files.append((x.path, x.size))
    (root / "files.json").write_text(json.dumps(files))
    log(f"{len(files)} files, {sum(s for _, s in files) / 1e12:.2f} TB -> {root / 'files.json'}")


def plan(root: Path, workers: int):
    files = json.loads((root / "files.json").read_text())
    todo = sorted(p for p, _ in files if p.endswith(".tar.gz") and p.split("/")[0] in TAR_THEMES)
    left = [p for p in todo if not (root / "raw" / p[: -len(".tar.gz")] / ".done").exists()]
    (root / "logs").mkdir(parents=True, exist_ok=True)
    for k in range(workers):
        (root / "logs" / f"list_w{k}.txt").write_text("\n".join(left[k::workers]) + "\n")
    subs = sorted({"/".join(p.split("/")[1:5]) for p, _ in files
                   if p.startswith("PhysicalReasoning/opensourcev4-") and "/lerobot/" in p})
    (root / "logs" / "list_pr.txt").write_text("\n".join(subs) + "\n")
    log(f"{len(left)} of {len(todo)} archives left over {workers} lists, {len(subs)} PhysicalReasoning dirs")


def check(dst: Path):
    """-> (ok, note): one parquet and one video per camera for every episode of each extracted dataset."""
    notes, ok = [], True
    infos = sorted(dst.glob("**/meta/info.json"))
    if not infos:
        return False, "no meta/info.json"
    for info in infos:
        d = info.parent.parent
        n = json.loads(info.read_text())["total_episodes"]
        counts = {"parquet": len(list(d.glob("data/chunk-*/*.parquet")))}
        counts.update({c: len(list(d.glob(f"videos/chunk-*/observation.images.{c}/*.mp4"))) for c in CAMS})
        bad = {k: v for k, v in counts.items() if v != n}
        if bad:
            ok = counts["parquet"] == n and counts["top_head"] == n  # a missing hand video is noted, not fatal
            notes.append(f"{d.relative_to(dst)}: {n} episodes, {bad}")
        else:
            notes.append(f"{d.relative_to(dst)}: {n} episodes")
    return ok, "; ".join(notes)


def tars(root: Path, worker: int, workers: int, only=None):
    files = json.loads((root / "files.json").read_text())
    todo = sorted(p for p, _ in files if p.endswith(".tar.gz") and p.split("/")[0] in TAR_THEMES)
    size = dict((p, s) for p, s in files)
    mine = [only] if only else todo[worker::workers]
    tmp = root / f"tmp_w{worker}"
    log(f"worker {worker}/{workers}: {len(mine)} archives, {sum(size[p] for p in mine) / 1e12:.2f} TB")
    t0, done_bytes = time.time(), 0
    for i, p in enumerate(mine):
        dst = root / "raw" / p[: -len(".tar.gz")]
        if (dst / ".done").exists():
            continue
        for attempt in range(3):
            try:
                if dst.exists():
                    shutil.rmtree(dst)
                dst.mkdir(parents=True)
                t1 = time.time()
                arc = Path(hf_hub_download(REPO, p, repo_type="dataset", local_dir=tmp))
                t2 = time.time()
                r = subprocess.run(["tar", "-I", "pigz", "-xf", str(arc), "-C", str(dst), "--wildcards"] + KEEP,
                                   capture_output=True, text=True)
                errs = [l for l in r.stderr.splitlines()
                        if "Not found in archive" not in l and "due to previous errors" not in l]
                if r.returncode != 0 and errs:  # a camera absent from an archive is left to check()
                    raise RuntimeError(f"tar exit {r.returncode}: {errs[:3]}")
                t3 = time.time()
                ok, note = check(dst)
                if not ok:
                    raise RuntimeError(f"incomplete: {note}")
                (dst / ".done").write_text(json.dumps({"archive": p, "bytes": size[p], "note": note}) + "\n")
                arc.unlink()
                done_bytes += size[p]
                rate = done_bytes / (time.time() - t0)
                log(f"[{i + 1}/{len(mine)}] {p} {size[p] / 1e9:.1f} GB: download {t2 - t1:.0f}s, extract {t3 - t2:.0f}s; "
                    f"{note[:200]} | worker {rate / 1e6:.0f} MB/s")
                break
            except Exception as e:  # noqa: BLE001 - retried, then logged and skipped
                log(f"[{i + 1}/{len(mine)}] {p} attempt {attempt + 1} failed: {e!r}")
                for a in glob.glob(str(tmp / p)):
                    os.remove(a)
                time.sleep(30)
        else:
            log(f"[{i + 1}/{len(mine)}] {p} GAVE UP")
    log(f"worker {worker} finished")


def pr(root: Path, sub="opensourcev4-*/lerobot/*/*"):
    keep = [f"PhysicalReasoning/{sub}/{k}" for k in
            ["meta/*", "data/*/*.parquet"] + [f"videos/*/observation.images.{c}/*" for c in CAMS]]
    files = [(p, s) for p, s in json.loads((root / "files.json").read_text()) if any(fnmatch.fnmatch(p, k) for k in keep)]
    todo = [p for p, s in files if not ((root / "raw" / p).exists() and (root / "raw" / p).stat().st_size == s)]
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(lambda p: hf_hub_download(REPO, p, repo_type="dataset", local_dir=root / "raw"), todo))
    log(f"PhysicalReasoning {sub}: {len(files)} files, {len(todo)} fetched")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["list", "plan", "tars", "pr"])
    ap.add_argument("--root", type=Path, default=Path("/scratch/t509a04/data/AgiBotWorld2026"))
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--only", default=None, help="tars: one archive")
    ap.add_argument("--sub", default="opensourcev4-*/lerobot/*/*", help="pr: one <export>/lerobot/<task>/<n> dir")
    a = ap.parse_args()
    {"list": lambda: list_files(a.root), "plan": lambda: plan(a.root, a.workers),
     "tars": lambda: tars(a.root, a.worker, a.workers, a.only), "pr": lambda: pr(a.root, a.sub)}[a.mode]()
