"""Train the production models on every season and save them to ../model/.

Per market:
  center  = gradient-boosted model (huber loss), averaged with a ridge regression for passing and receiving
  spread  = for rushing and receiving, a second model predicts how widely this player's results swing;
            for passing, measured ranges by projection size
Outcome shapes are fit on leave-one-season-out predictions, so they reflect real out-of-sample errors.
"""
import gzip, json, os
import numpy as np, pandas as pd, lightgbm as lgb
from train import df, FEATS, params, fit_ridge
from dist import scale_params, SFEATS

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model")
os.makedirs(OUT, exist_ok=True)
RIDGE = {"pass": True, "rush": False, "rec": True}
SPREAD_MODEL = {"pass": False, "rush": True, "rec": True}
FULL = [s for s in sorted(df.season.unique()) if s < 2026]

def save(model, name):
    with gzip.open(os.path.join(OUT, name + ".txt.gz"), "wt", compresslevel=9) as fh:
        fh.write(model.model_to_string())
    old = os.path.join(OUT, name + ".txt")
    if os.path.exists(old):
        os.remove(old)

def gbm(tr, kind):
    val = max(s for s in tr.season.unique() if s in FULL)
    a, b = tr[tr.season != val], tr[tr.season == val]
    m = lgb.train(params(kind, "huber"), lgb.Dataset(a[FEATS], a.y), 3000, valid_sets=[lgb.Dataset(b[FEATS], b.y)],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
    n = max(100, m.best_iteration)
    return lgb.train(params(kind, "huber"), lgb.Dataset(tr[FEATS], tr.y), n)

def center(tr, kind):
    g = gbm(tr, kind); r = fit_ridge(tr) if RIDGE[kind] else None
    f = (lambda X: (g.predict(X) + r.predict(X)) / 2) if r is not None else g.predict
    return f, g, r

def ridge_export(r):
    imp, sc, rr = r.named_steps["simpleimputer"], r.named_steps["standardscaler"], r.named_steps["ridgecv"]
    return {"median": imp.statistics_.tolist(), "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
            "coef": rr.coef_.tolist(), "intercept": float(rr.intercept_)}

bundle = {"features": FEATS, "markets": {}, "trained_on": f"{min(df.season)}-{max(df.season)} regular seasons, {len(df)} player-games"}
for kind in ("pass", "rush", "rec"):
    d = df[df.kind == kind].copy()
    oof = pd.Series(index=d.index, dtype=float)
    for s in sorted(d.season.unique()):
        f, *_ = center(d[d.season != s], kind)
        oof[d.season == s] = f(d[d.season == s][FEATS])
    d["loc"] = oof.clip(lower=1)
    f, g, r = center(d, kind)
    save(g, f"{kind}_center")
    m = {"ridge": ridge_export(r) if r is not None else None, "spread_model": SPREAD_MODEL[kind]}
    cuts = np.quantile(d["loc"], [1 / 3, 2 / 3]); tier = np.searchsorted(cuts, d["loc"])
    m["cuts"] = cuts.round(2).tolist()
    if SPREAD_MODEL[kind]:
        d["absr"] = (d.y - d["loc"]).abs()
        sm = lgb.train(scale_params(), lgb.Dataset(d[SFEATS], d.absr), 400)
        save(sm, f"{kind}_spread")
        sc = np.clip(sm.predict(d[SFEATS]), 3, None)
        z = (d.y - d["loc"]) / sc
        m["zq"] = [np.round(np.quantile(z[tier == t], np.linspace(0, 1, 101)), 4).tolist() for t in range(3)]
    else:
        m["rq"] = [np.round(np.quantile((d.y / d["loc"])[tier == t], np.linspace(0, 1, 101)), 4).tolist() for t in range(3)]
    m["oof_mae"] = round(float((d.y - d["loc"]).abs().mean()), 2)
    m["engine_mae"] = round(float((d.y - d.mu_struct).abs().mean()), 2)
    m["n"] = int(len(d))
    bundle["markets"][kind] = m
    print(kind, "OOF MAE", m["oof_mae"], "engine", m["engine_mae"], "rows", m["n"], flush=True)
json.dump(bundle, open(os.path.join(OUT, "bundle.json"), "w"), separators=(",", ":"))
print("saved to", OUT, os.listdir(OUT))
