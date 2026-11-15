"""The prediction engine: turns a model projection and the sportsbook's line into a chance for every bet.

Three steps, each one fit on real results (research/fit_blend.py):

1. The models give every player a full range of outcomes, not one number: 101 points from his worst
   realistic game to his best, shaped by how players with that kind of projection have actually scattered.
2. The sportsbook's line is a second opinion on the same player. The model's chance of clearing that line
   is blended with it on the log-odds scale. The blend weights say how far to trust the model when it
   disagrees with the line, and they are measured on this season's lines and results.
3. The whole range is then slid so it agrees with the blended chance, and widened or narrowed by a
   measured factor. Every other number for that player ("40 or more", "60 or more") is read off that range.
"""
import math
import numpy as np


def logit(p):
    p = min(max(float(p), 1e-3), 1 - 1e-3)
    return math.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def grid(m, loc, sc):
    """101 outcome points (0th to 100th percentile) for a projection `loc` with spread `sc`."""
    t = int(sum(loc > c for c in m["cuts"]))
    yq = loc + sc * np.array(m["zq"][t]) if m["spread_model"] else loc * np.array(m["rq"][t])
    return np.maximum.accumulate(np.clip(yq, -10, None))


def p_over(yq, line):
    """Chance the result lands above `line`. For 'n or more' pass n - 0.5."""
    if line < yq[0]:
        return 0.998
    if line >= yq[-1]:
        return 0.002
    i = int(np.searchsorted(yq, line, side="right")) - 1
    lo, hi = yq[i], yq[i + 1]
    frac = 0.5 if hi == lo else (line - lo) / (hi - lo)
    return float(min(0.998, max(0.002, 1 - (i + frac) / (len(yq) - 1))))


def blend(p_model, w):
    """Final chance of going over the sportsbook's line: the model's read, pulled toward the line."""
    return sigmoid(w["base"] + w["model"] * logit(p_model))


def reshape(yq, shift, scale):
    """Slide the range by `shift` yards and stretch it around its middle by `scale`."""
    mid = yq[(len(yq) - 1) // 2]
    return np.maximum.accumulate(np.clip(mid + shift + (yq - mid) * scale, -10, None))


def anchor(yq, line, target, scale=1.0):
    """The range, slid until its chance of clearing `line` equals `target` (and stretched by `scale`)."""
    lo, hi = -(yq[-1] - yq[0]) - 50, (yq[-1] - yq[0]) + 50
    for _ in range(40):
        mid = (lo + hi) / 2
        if p_over(reshape(yq, mid, scale), line) < target:
            lo = mid
        else:
            hi = mid
    return reshape(yq, (lo + hi) / 2, scale)


def with_line(m, w, loc, sc, line):
    """Model range and the final range once the sportsbook's line is taken into account."""
    raw = grid(m, loc, sc)
    if line is None or not w:
        return raw, raw, None
    pm = p_over(raw, line)
    pf = blend(pm, w)
    return raw, anchor(raw, line, pf, w.get("scale", 1.0)), pm
