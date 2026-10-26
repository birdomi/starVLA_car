"""Head-look segmentation for seq_check.py (the looks() part of the 10-15 scratchpad headtiming/cap_lib.py, restored
10-21 from the session transcript). looks(yaw_deg) -> [(start, end, side)], side +1 = |yaw| > 20 deg to the left (+),
-1 = right; runs >= 9 frames, same-side runs < 9 frames apart merged."""
import numpy as np


def looks(yaw):
    s = np.where(yaw > 20, 1, np.where(yaw < -20, -1, 0)); out = []; i = 0; T = len(s)
    while i < T:
        if s[i]:
            j = i
            while j < T and s[j] == s[i]: j += 1
            if out and out[-1][2] == s[i] and i - out[-1][1] < 9: out[-1] = (out[-1][0], j, s[i])
            elif j - i >= 9: out.append((i, j, s[i]))
            i = j
        else: i += 1
    return out
