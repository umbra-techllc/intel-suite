import numpy as np, pandas as pd
from train import df, FEATS
from score_probs import preds, fit_ranges, p_over
from dist import fit_dist, p_over_ls, SFEATS
a, b = preds(2023), preds(2024)
res = {}
for m in ("engine", "huber_ridge", "huber"):
    cal = fit_ranges(a, m); rows = []
    for r in b.itertuples():
        mu = max(1.0, getattr(r, m)); base = max(1.0, r.engine)
        for mult in (0.7, 0.85, 1.0, 1.15, 1.3):
            L = np.floor(base * mult) + 0.5; rows.append((r.kind, (p_over(cal, r.kind, mu, L) - (r.y > L)) ** 2))
    res[m] = pd.DataFrame(rows, columns=["kind", "b"]).groupby("kind").b.mean()
rows = []
for kind in ("pass", "rush", "rec"):
    d = df[df.kind == kind]; tr, te = d[d.season < 2024], d[d.season == 2024].copy()
    f, sm, cuts, zq = fit_dist(tr, kind)
    te["loc"] = np.clip(f(te[FEATS]), 1, None); te["sc"] = np.clip(sm.predict(te[SFEATS]), 3, None)
    for r in te.itertuples():
        base = max(1.0, r.mu_struct)
        for mult in (0.7, 0.85, 1.0, 1.15, 1.3):
            L = np.floor(base * mult) + 0.5; rows.append((kind, (p_over_ls(r.loc, r.sc, cuts, zq, L) - (r.y > L)) ** 2))
res["location+spread"] = pd.DataFrame(rows, columns=["kind", "b"]).groupby("kind").b.mean()
print("Brier on 2024 (lower is better):")
for m, t in res.items(): print(f"  {m:16} pass {t['pass']:.4f}  rush {t['rush']:.4f}  rec {t['rec']:.4f}")
