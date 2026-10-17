"""Score each model on what matters for betting: the over/under chance at lines near the projection.

Outcome ranges for each model are fit on its 2024 out-of-sample predictions, then the 2025 over/under
chances are scored (Brier score: lower is better; 0.25 is a coin flip)."""
import numpy as np, pandas as pd
from train import df, FEATS, fit_gbm, fit_ridge

def preds(test_season):
    out = []
    for kind in ("pass", "rush", "rec"):
        d = df[df.kind == kind]; tr, te = d[d.season < test_season], d[d.season == test_season].copy()
        g, _ = fit_gbm(tr, kind); h, _ = fit_gbm(tr, kind, "huber"); r = fit_ridge(tr)
        te["engine"] = te.mu_struct; te["gbm"] = g.predict(te[FEATS]); te["huber"] = h.predict(te[FEATS])
        te["ridge"] = r.predict(te[FEATS]); te["huber_ridge"] = (te.huber + te.ridge) / 2; te["gbm_ridge"] = (te.gbm + te.ridge) / 2
        out.append(te)
    return pd.concat(out)

def fit_ranges(d, col):
    cal = {}
    for kind, g in d.groupby("kind"):
        mu = g[col].clip(lower=1); cuts = np.quantile(mu, [1 / 3, 2 / 3])
        tier = np.searchsorted(cuts, mu)
        cal[kind] = (cuts, [np.quantile((g.y / mu)[tier == t], np.linspace(0, 1, 101)) for t in range(3)])
    return cal

def p_over(cal, kind, mu, line):
    cuts, qs = cal[kind]; q = qs[int(np.searchsorted(cuts, mu))]; r = line / mu
    if r < q[0]: return 1.0
    if r >= q[-1]: return 0.0
    i = np.searchsorted(q, r, side="right") - 1
    lo, hi = q[i], q[i + 1]
    return 1 - (i + (0 if hi == lo else (r - lo) / (hi - lo))) / 100

if __name__ == "__main__":
    a, b = preds(2024), preds(2025)
    models = ["engine", "ridge", "gbm", "huber", "gbm_ridge", "huber_ridge"]
    res = {m: [] for m in models}
    for m in models:
        cal = fit_ranges(a, m)
        for r in b.itertuples():
            mu = max(1.0, getattr(r, m))
            # score at a common set of lines around the current engine's projection, so every model faces the same lines
            base = max(1.0, r.engine)
            for mult in (0.7, 0.85, 1.0, 1.15, 1.3):
                L = np.floor(base * mult) + 0.5
                res[m].append((r.kind, (p_over(cal, r.kind, mu, L) - (r.y > L)) ** 2))
    print("Brier score on 2025 over/under calls (lower is better):")
    for m in models:
        t = pd.DataFrame(res[m], columns=["kind", "b"]).groupby("kind").b.mean()
        print(f"  {m:12} pass {t['pass']:.4f}  rush {t['rush']:.4f}  rec {t['rec']:.4f}  all {np.mean([x[1] for x in res[m]]):.4f}")
