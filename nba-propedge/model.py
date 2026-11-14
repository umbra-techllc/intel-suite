"""Loads the trained models and turns a projection into chances.

The saved bundle holds, per stat: a projection model, a second model for how widely results swing,
the measured shape of misses for 12 projection sizes, and the weights that blend the model with the
sportsbook's price. Used by the daily build and by the research scripts.
"""
import gzip, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")
NOFAIR = 0.47  # overs on opening lines win about 47% of the time; used when a two-sided price is missing
SFEATS = ["loc", "min_m", "min_sd", "min_s", "start_m", "n_prev", "abs_spread", "n_out", "new_min", "last_min", "rest"]


def logit(p):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def implied(american):
    a = float(american)
    return -a / (-a + 100.0) if a < 0 else 100.0 / (a + 100.0)


def decimal(american):
    a = float(american)
    return 1 + a / 100.0 if a > 0 else 1 + 100.0 / abs(a)


def tier_of(cuts, loc):
    return int(np.searchsorted(cuts, loc))


def p_over(loc, sc, cuts, zq, line, tier=None):
    """Chance the result lands above `line`. For 'n or more' pass n - 0.5."""
    q = zq[tier_of(cuts, loc) if tier is None else tier]
    r = (line - loc) / sc
    if r < q[0]:
        return 0.999
    if r >= q[-1]:
        return 0.001
    i = int(np.searchsorted(q, r, side="right")) - 1
    lo, hi = q[i], q[i + 1]
    frac = 0.5 if hi == lo else (r - lo) / (hi - lo)
    return float(np.clip(1 - (i + frac) / (len(q) - 1), 0.001, 0.999))


def blend(p_model, fair, w):
    """Final chance of the over on the book's main line: the model's read weighed against the book's price."""
    f = NOFAIR if fair is None or (isinstance(fair, float) and np.isnan(fair)) else fair
    return float(sigmoid(w["base"] + w["model"] * logit(p_model) + w["book"] * logit(f)))


def anchor(loc, sc, cuts, zq, line, target):
    """Slide the projection until its chance of clearing the main line equals the blended chance.
    Every other number for that player (the 'n or more' ladder) is then read off the moved projection."""
    tier = tier_of(cuts, loc)
    lo, hi = loc - 5 * sc, loc + 5 * sc
    for _ in range(40):
        mid = (lo + hi) / 2
        if p_over(mid, sc, cuts, zq, line, tier) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2, tier


def median(loc, sc, zq, tier):
    q = zq[tier]
    return loc + sc * q[(len(q) - 1) // 2]


def load_bundle():
    import lightgbm as lgb
    b = json.load(open(os.path.join(MODEL_DIR, "bundle.json")))
    def booster(name):
        with gzip.open(os.path.join(MODEL_DIR, name + ".txt.gz"), "rt") as fh:
            return lgb.Booster(model_str=fh.read())
    for t, m in b["stats"].items():
        m["center"] = booster(f"{t}_center")
        m["spread"] = booster(f"{t}_spread")
        m["cuts"] = np.array(m["cuts"]); m["zq"] = [np.array(q) for q in m["zq"]]
    return b


def predict(b, feats):
    """feats: DataFrame from features.build. Adds loc_<stat> and sc_<stat> columns."""
    out = feats.copy()
    for t, m in b["stats"].items():
        loc = np.clip(m["center"].predict(out[b["features"]]), 0.05, None)
        x = out[SFEATS[1:]].copy(); x.insert(0, "loc", loc)
        out[f"loc_{t}"] = loc
        out[f"sc_{t}"] = np.sqrt(np.clip(m["spread"].predict(x[SFEATS]), 0.05, None))
    return out
