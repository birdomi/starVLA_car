# Parse rich-printed metric dicts from a starVLA train log -> {key: [(step, value)]}; print window means.
#   python parse_trainlog_any.py <train log> "<key1,key2 or empty = all>" "0-0.5,0.5-1,..." (windows in k steps)
import re, sys, collections
t = open(sys.argv[1], errors="ignore").read()
flat = re.sub(r"\s+", " ", t)
flat = re.sub(r"'([^'{}]*)'", lambda m: "'" + m.group(1).replace(" ", "") + "'", flat)
out = collections.defaultdict(list)
step = 0
for m in re.finditer(r"(\d+)/(?:100000|50000) \[|\{('[^{}]*?)\}\)", flat):
    if m.group(1):
        step = int(m.group(1)); continue
    for k, v in re.findall(r"'([\w/.\-]+)': ?([-0-9.e]+|nan)", m.group(2)):
        try: out[k].append((step, float(v)))
        except ValueError: pass
keys = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] else sorted(out)
wins = [tuple(int(float(x)*1000) for x in w.split("-")) for w in sys.argv[3].split(",")]
print("key".ljust(42), *[f"{a/1000:g}-{b/1000:g}k".rjust(9) for a, b in wins])
for k in keys:
    if k not in out or k.startswith(("timing", "learning_rate")): continue
    row = []
    for a, b in wins:
        v = [x for s, x in out[k] if a <= s < b]
        row.append(f"{sum(v)/len(v):9.4f}" if v else " " * 9)
    print(k[:42].ljust(42), *row, f"(n={len(out[k])})")
