"""Faster variant of download_gr00t_ft_data.py: lists each target folder with list_repo_tree
(seconds) instead of list_repo_files on the whole repo (very slow), then downloads in parallel.
Usage: python download_gr00t_ft_data_fast.py [local_dir] [workers]
"""
import sys, time, random
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import HfApi, hf_hub_download
from download_gr00t_ft_data import REPO_ID, REPO_TYPE, FOLDERS, LOCAL_DIR

local_dir = sys.argv[1] if len(sys.argv) > 1 else LOCAL_DIR
workers = int(sys.argv[2]) if len(sys.argv) > 2 else 16
api = HfApi()
files = []
for folder in FOLDERS:
    tree = list(api.list_repo_tree(REPO_ID, repo_type=REPO_TYPE, path_in_repo=folder, recursive=True))
    fl = [e.path for e in tree if getattr(e, "size", None) is not None]
    print(f"{folder}: {len(fl)} files", flush=True)
    files += fl
print(f"total {len(files)} files", flush=True)

def dl(fn):
    for attempt in range(1, 31):
        try:
            hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE, filename=fn, local_dir=local_dir)
            return True
        except Exception as exc:
            time.sleep(random.uniform(1, 5) * attempt / 5)
    print("GIVE UP", fn, flush=True); return False

failed = 0
with ThreadPoolExecutor(workers) as ex:
    futs = {ex.submit(dl, f): f for f in files}
    for i, fu in enumerate(as_completed(futs), 1):
        if not fu.result(): failed += 1
        if i % 500 == 0 or i == len(files): print(f"{i}/{len(files)} done, {failed} failed", flush=True)
print("ALL_DOWNLOAD_DONE failed=%d" % failed, flush=True)
