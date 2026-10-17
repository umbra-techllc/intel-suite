"""Train gradient-boosted yardage models per market and test them walk-forward against the current engine."""
import json, os, sys, warnings
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "cache", "dataset.csv"))
META = ["season", "week", "pid", "name", "team", "kind", "game", "y"]
FEATS = [c for c in df.columns if c not in META]
MONO_UP = {"pass": ["mu_struct", "itt", "qb_ypa"], "rush": ["mu_struct", "cs", "itt"], "rec": ["mu_struct", "ts", "itt", "qb_ypa"]}

def params(kind, obj="regression"):
    mono = [1 if f in MONO_UP[kind] else 0 for f in FEATS]
    return dict(objective=obj, learning_rate=0.03, num_leaves=15, min_data_in_leaf=40, feature_fraction=0.8,
                bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, monotone_constraints=mono, verbose=-1, seed=7)

def fit_gbm(tr, kind, obj="regression"):
    # pick the number of rounds on the last training season, then refit on everything
    last = tr.season.max()
    a, b = tr[tr.season < last], tr[tr.season == last]
    m = lgb.train(params(kind, obj), lgb.Dataset(a[FEATS], a.y), 3000, valid_sets=[lgb.Dataset(b[FEATS], b.y)],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
    n = max(100, m.best_iteration)
    return lgb.train(params(kind, obj), lgb.Dataset(tr[FEATS], tr.y), n), n

def fit_ridge(tr):
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))
    return m.fit(tr[FEATS], tr.y)

if __name__ == "__main__":
    out = []
    for test_season in (2024, 2025):
        for kind in ("pass", "rush", "rec"):
            d = df[df.kind == kind]
            tr, te = d[d.season < test_season], d[d.season == test_season]
            g, n = fit_gbm(tr, kind)
            g1, n1 = fit_gbm(tr, kind, "huber")
            r = fit_ridge(tr)
            pg, pg1, pr = g.predict(te[FEATS]), g1.predict(te[FEATS]), r.predict(te[FEATS])
            ens = (pg + pr) / 2
            row = {"test": test_season, "kind": kind, "n": len(te),
                   "engine": abs(te.y - te.mu_struct).mean(), "ridge": abs(te.y - pr).mean(),
                   "gbm": abs(te.y - pg).mean(), "gbm_huber": abs(te.y - pg1).mean(), "gbm+ridge": abs(te.y - ens).mean(),
                   "bias_engine": (te.y - te.mu_struct).mean(), "bias_gbm": (te.y - pg).mean(), "rounds": n}
            out.append(row)
            print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    t = pd.DataFrame(out)
    print("\nMAE in yards (lower is better), tested on seasons the models never saw:")
    print(t[["test", "kind", "n", "engine", "ridge", "gbm", "gbm_huber", "gbm+ridge"]].round(2).to_string(index=False))
