"""Shared training pieces, plus the held-out comparisons when run directly.

  python research/train.py     train on 2024-25 and 2025-26, test on 2026-27 so far
"""
import os, sys
import numpy as np, pandas as pd, lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import dataset
import features as F
from model import p_over, SFEATS

FEATS, TARGETS = F.FEATS, F.TARGETS
MIN_PREV = 3  # a player needs a few games on record before he is used


def load():
    p = F.build(dataset.load_games())
    return p[(p["min"] > 0) & (p.n_prev >= MIN_PREV)].reset_index(drop=True)


def load_lines():
    l = dataset.load_lines()
    imp = lambda a: np.where(a < 0, -a / (-a + 100.0), 100.0 / (a + 100.0))
    po, pu = imp(l.over), imp(l.under)
    hold = po + pu
    # Only DraftKings openers are kept. ESPN BET's stored openers (2024-25 and early 2025-26) are unreliable:
    # both sides often carry the same price and many lines are far from where the player was actually listed.
    l = l[(l.book == "DraftKings") & (hold > 1.02) & (hold < 1.13)].copy()
    l["fair"] = (po / hold)[l.index]
    return l.reset_index(drop=True)


def params(obj="huber"):
    d = dict(objective=obj, learning_rate=0.06, num_leaves=16, min_data_in_leaf=60, feature_fraction=0.7,
             bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=7, num_threads=8)
    if obj == "huber":
        d["alpha"] = 6.0
    return d


def obj_for(target):
    return "poisson" if target in ("tpm", "ast", "reb", "ra") else "huber"


def fit_center(tr, target, rounds=None):
    obj = obj_for(target)
    if rounds is None:
        cut = tr.t.quantile(0.85)
        a, b = tr[tr.t <= cut], tr[tr.t > cut]
        m = lgb.train(params(obj), lgb.Dataset(a[FEATS], a[target]), 2500, valid_sets=[lgb.Dataset(b[FEATS], b[target])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        rounds = int(max(150, m.best_iteration) * 1.1)
    return lgb.train(params(obj), lgb.Dataset(tr[FEATS], tr[target]), rounds), rounds


def spread_params():
    return dict(objective="regression", learning_rate=0.03, num_leaves=8, min_data_in_leaf=150, feature_fraction=0.8,
                bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=11, num_threads=8)



def oof_center(d, target):
    """Predictions for each season from a model that never saw that season."""
    out = pd.Series(index=d.index, dtype=float)
    for s in sorted(d.season.unique()):
        m, _ = fit_center(d[d.season != s], target)
        out[d.season == s] = m.predict(d.loc[d.season == s, FEATS])
    return out


NT = 12  # outcome shapes are measured separately for 12 projection sizes


def fit_shape(loc, y, sfe):
    """Second model: how far results usually land from the projection, then the measured shape of those misses."""
    d = sfe.copy(); d["loc"] = loc
    sq = (y - loc) ** 2
    sm = lgb.train(spread_params(), lgb.Dataset(d[SFEATS], sq), 300)
    sc = np.sqrt(np.clip(sm.predict(d[SFEATS]), 0.05, None))
    z = (y - loc) / sc
    cuts = np.quantile(loc, np.linspace(0, 1, NT + 1)[1:-1])
    tier = np.searchsorted(cuts, loc)
    zq = [np.quantile(z[tier == k], np.linspace(0, 1, 201)) for k in range(NT)]
    return sm, cuts, zq


def scale(sm, d, loc):
    x = d[SFEATS[1:]].copy(); x.insert(0, "loc", loc)
    return np.sqrt(np.clip(sm.predict(x[SFEATS]), 0.05, None))


if __name__ == "__main__":
    df = load(); lines = load_lines()
    TEST = 2027
    tr, te = df[df.season < TEST], df[df.season == TEST].copy()
    print(f"train {len(tr)} player-games, test {len(te)} ({te.gid.nunique()} games this season)\n")
    print(f"{'stat':5} {'model':>7} {'last-10 avg':>12} {'season avg':>11} {'book line':>10}   (average miss, lower is better)")
    res = {}
    for t in TARGETS:
        oof = oof_center(tr, t)
        sm, cuts, zq = fit_shape(oof.values, tr[t].values, tr)
        m, _ = fit_center(tr, t)
        te["loc"] = m.predict(te[FEATS]); te["sc"] = scale(sm, te, te["loc"].values)
        parts = F.__dict__.get("PARTS") or {"pts": ["pts"], "reb": ["reb"], "ast": ["ast"], "tpm": ["tpm"], "pra": ["pts", "reb", "ast"],
                                              "pr": ["pts", "reb"], "pa": ["pts", "ast"], "ra": ["reb", "ast"]}
        l10 = sum(te[f"{c}_m"] for c in parts[t]); seas = sum(te[f"{c}_l"] for c in parts[t])
        ln = te.merge(lines[lines.stat == t], on=["gid", "pid"])
        mae = lambda a, b: float(np.mean(np.abs(a - b)))
        print(f"{t:5} {mae(te[t], te['loc']):7.2f} {mae(te[t], l10):12.2f} {mae(te[t], seas):11.2f} "
              f"{mae(ln[t], ln.line):10.2f} | model on lined players {mae(ln[t], ln['loc']):.2f}")
        ln["p"] = [p_over(a, b, cuts, zq, c) for a, b, c in zip(ln["loc"], ln.sc, ln.line)]
        ln["y"] = (ln[t] > ln.line).astype(int); ln["stat"] = t
        res[t] = ln[["gid", "pid", "stat", "line", "over", "under", "fair", "loc", "sc", "p", "y", t]].rename(columns={t: "actual"})
    r = pd.concat(res.values())
    r.to_pickle("/tmp/nba_test_preds.pkl")
    print("\nDoes the model's over chance match what happened? (test season, opening lines)")
    print(r.groupby(pd.cut(r.p, [0, .3, .4, .45, .5, .55, .6, .7, 1])).agg(said=("p", "mean"), happened=("y", "mean"), n=("y", "size")).round(3))
    b = lambda p: float(np.mean((p - r.y) ** 2))
    f = r.fair.fillna(0.47)
    print(f"\nBrier score (lower is better): model {b(r.p):.4f} | book price {b(f):.4f} | coin flip {b(0.5):.4f} | always 'under' lean {b(r.y.mean()):.4f}")
