"""Location + spread model: one model predicts the center, a second predicts how widely each player's
results swing; outcome shapes come from out-of-fold standardized errors. Compared on 2025 Brier scores."""
import numpy as np, pandas as pd, lightgbm as lgb
from train import df, FEATS, fit_gbm, fit_ridge, params
from score_probs import fit_ranges, p_over

BLEND_RIDGE = {"pass": True, "rush": False, "rec": True}

def loc_model(tr, kind):
    h, _ = fit_gbm(tr, kind, "huber")
    r = fit_ridge(tr) if BLEND_RIDGE[kind] else None
    return lambda X: (h.predict(X) + r.predict(X)) / 2 if r is not None else h.predict(X)

def oof_loc(tr, kind):
    """Leave-one-season-out predictions for every training row."""
    out = pd.Series(index=tr.index, dtype=float)
    for s in sorted(tr.season.unique()):
        f = loc_model(tr[tr.season != s], kind)
        out[tr.season == s] = f(tr[tr.season == s][FEATS])
    return out

SFEATS = FEATS + ["loc"]
def scale_params():
    return dict(objective="regression_l1", learning_rate=0.03, num_leaves=7, min_data_in_leaf=80, feature_fraction=0.8,
                bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=11)

def fit_dist(tr, kind):
    tr = tr.copy(); tr["loc"] = oof_loc(tr, kind).clip(lower=1)
    tr["absr"] = (tr.y - tr["loc"]).abs()
    sm = lgb.train(scale_params(), lgb.Dataset(tr[SFEATS], tr.absr), 400)
    sc = np.clip(sm.predict(tr[SFEATS]), 3, None)
    z = (tr.y - tr["loc"]) / sc
    cuts = np.quantile(tr["loc"], [1 / 3, 2 / 3]); tier = np.searchsorted(cuts, tr["loc"])
    zq = [np.quantile(z[tier == t], np.linspace(0, 1, 101)) for t in range(3)]
    return loc_model(tr, kind), sm, cuts, zq

def p_over_ls(loc, sc, cuts, zq, line):
    q = zq[int(np.searchsorted(cuts, loc))]; r = (line - loc) / sc
    if r < q[0]: return 1.0
    if r >= q[-1]: return 0.0
    i = np.searchsorted(q, r, side="right") - 1
    lo, hi = q[i], q[i + 1]
    return 1 - (i + (0 if hi == lo else (r - lo) / (hi - lo))) / 100

if __name__ == "__main__":
    rows = []
    for kind in ("pass", "rush", "rec"):
        d = df[df.kind == kind]; tr, te = d[d.season < 2025], d[d.season == 2025].copy()
        f, sm, cuts, zq = fit_dist(tr, kind)
        te["loc"] = np.clip(f(te[FEATS]), 1, None); te["sc"] = np.clip(sm.predict(te[SFEATS]), 3, None)
        for r in te.itertuples():
            base = max(1.0, r.mu_struct)
            for mult in (0.7, 0.85, 1.0, 1.15, 1.3):
                L = np.floor(base * mult) + 0.5
                rows.append((kind, (p_over_ls(r.loc, r.sc, cuts, zq, L) - (r.y > L)) ** 2))
        print(kind, "done", flush=True)
    t = pd.DataFrame(rows, columns=["kind", "b"])
    g = t.groupby("kind").b.mean()
    print(f"location+spread  pass {g['pass']:.4f}  rush {g['rush']:.4f}  rec {g['rec']:.4f}  all {t.b.mean():.4f}")
