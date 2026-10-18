"""Train the production touchdown models on every season and save them to ../model/.

Anytime TD: gradient-boosted classifier, then a calibration curve fit on leave-one-season-out predictions.
QB touchdown passes: gradient-boosted Poisson count model.
Also measures how often touchdown outcomes move together (for parlays), from out-of-sample errors.
"""
import gzip, json, os, sys
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_td import df, F, QF, p_any, fit_any, fit_qb
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model")
os.makedirs(OUT, exist_ok=True)

def save(m, name):
    with gzip.open(os.path.join(OUT, name + ".txt.gz"), "wt", compresslevel=9) as fh:
        fh.write(m.model_to_string())

full = [s for s in sorted(df.season.unique()) if s < 2026]
oof = pd.Series(index=df.index, dtype=float)
for s in sorted(df.season.unique()):
    tr = df[(df.season != s) & df.season.isin(full)]
    oof[df.season == s] = fit_any(tr).predict(df[df.season == s][F])
# Platt scaling: a smooth two-number correction on the log-odds scale, fit on out-of-sample predictions
lo = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
pl = LogisticRegression(C=1e6).fit(lo(oof.values).reshape(-1, 1), df.any_td)
cal = lambda p: pl.predict_proba(lo(np.asarray(p)).reshape(-1, 1))[:, 1]
grid = np.round(np.linspace(0, 0.95, 191), 4)
curve = np.round(cal(grid), 4)
df["oof"] = cal(oof.values)
print("Platt scaling: slope", round(float(pl.coef_[0][0]), 3), "intercept", round(float(pl.intercept_[0]), 3))
save(fit_any(df[df.season.isin(full) | (df.season == 2026)]), "anytime")

q = df[(df.position == "QB") & (df.is_starter == 1)].copy()
qoof = pd.Series(index=q.index, dtype=float)
for s in sorted(q.season.unique()):
    qoof[q.season == s] = fit_qb(q[(q.season != s) & q.season.isin(full)]).predict(q[q.season == s][QF])
q["lam"] = qoof
save(fit_qb(q), "qb_pass_td")

# how touchdown outcomes move together, from out-of-sample errors
df["res"] = df.any_td - df.oof
q["res2"] = (q.passing_tds >= 2).astype(float) - (1 - np.exp(-q.lam) * (1 + q.lam))
pairs = {"same team": [], "same team, same position": [], "opponents": [], "QB 2+ TD passes + own receiver": [], "QB 2+ TD passes + own running back": []}
qb_by_game = q.set_index(["game_id", "team"]).res2.to_dict()
for gid, g in df[df.season.isin(full)].groupby("game_id"):
    recs = g[["team", "position", "res", "player_id"]].to_dict("records")
    for i, a in enumerate(recs):
        if a["position"] != "QB":
            r2 = qb_by_game.get((gid, a["team"]))
            if r2 is not None:
                pairs["QB 2+ TD passes + own receiver" if a["position"] in ("WR", "TE") else "QB 2+ TD passes + own running back"].append((r2, a["res"]))
        for b in recs[i + 1:]:
            if a["position"] == "QB" or b["position"] == "QB":
                continue
            if a["team"] == b["team"]:
                pairs["same team"].append((a["res"], b["res"]))
                if a["position"] == b["position"]:
                    pairs["same team, same position"].append((a["res"], b["res"]))
            else:
                pairs["opponents"].append((a["res"], b["res"]))
corr = {}
for k, v in pairs.items():
    x, y = np.array(v).T
    corr[k] = round(float(spearmanr(x, y)[0]), 3)
    print(f"  {k:36} {corr[k]:+.3f}  (n={len(v)})")

bundle = {"features": F, "qb_features": QF, "calibration": {"x": grid.tolist(), "y": curve.tolist()}, "correlations": corr,
          "trained_on": f"{int(df.season.min())}-{int(df.season.max())} regular seasons, {len(df)} player-games",
          "base_rate": round(float(df.any_td.mean()), 4)}
json.dump(bundle, open(os.path.join(OUT, "bundle.json"), "w"), separators=(",", ":"))
print("saved", sorted(os.listdir(OUT)))
c = pd.DataFrame({"p": df.oof, "y": df.any_td}); c["bin"] = pd.cut(c.p, [0, .05, .1, .2, .3, .4, .5, .7, 1])
print("calibrated, out of sample, all seasons:"); print(c.groupby("bin", observed=True).agg(said=("p", "mean"), happened=("y", "mean"), n=("y", "size")).round(3).T.to_string())
