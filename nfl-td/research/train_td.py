"""Walk-forward tests for the touchdown models against simpler approaches."""
import os, sys, warnings
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import td_engine as E
C = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
df = pd.read_parquet(f"{C}/td_dataset.parquet")
F, QF = E.FEATS, E.QB_FEATS
MONO = {"e_rz_car_sh": 1, "e_rz_tgt_sh": 1, "e_g10_sh": 1, "itt": 1}

def p_any(kind="binary"):
    return dict(objective=kind, learning_rate=0.02, num_leaves=15, min_data_in_leaf=60, feature_fraction=0.8, bagging_fraction=0.8,
                bagging_freq=1, lambda_l2=2.0, verbose=-1, seed=3, monotone_constraints=[MONO.get(f, 0) for f in F])

def fit_any(tr):
    last = tr.season.max(); a, b = tr[tr.season < last], tr[tr.season == last]
    m = lgb.train(p_any(), lgb.Dataset(a[F], a.any_td), 4000, valid_sets=[lgb.Dataset(b[F], b.any_td)], callbacks=[lgb.early_stopping(200, verbose=False)])
    return lgb.train(p_any(), lgb.Dataset(tr[F], tr.any_td), max(150, m.best_iteration))

def fit_lr(tr):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.3, max_iter=2000)).fit(tr[F], tr.any_td)

def rate_baseline(tr, te):
    pos = tr.groupby("position").any_td.mean()
    prior = te.position.map(pos)
    prev = te.p_any.where(te.p_g >= 6)
    prior = prev.fillna(prior)
    n = te.n_games
    return ((te.e_any_td.fillna(0) * n + prior * 4) / (n + 4)).clip(0.01, 0.95)

def qb_params():
    return dict(objective="poisson", learning_rate=0.02, num_leaves=7, min_data_in_leaf=40, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=5)

def fit_qb(tr):
    last = tr.season.max(); a, b = tr[tr.season < last], tr[tr.season == last]
    m = lgb.train(qb_params(), lgb.Dataset(a[QF], a.passing_tds), 3000, valid_sets=[lgb.Dataset(b[QF], b.passing_tds)], callbacks=[lgb.early_stopping(200, verbose=False)])
    return lgb.train(qb_params(), lgb.Dataset(tr[QF], tr.passing_tds), max(100, m.best_iteration))

if __name__ == "__main__":
    print("ANYTIME TOUCHDOWN (rushing or receiving) - lower log loss / Brier is better, higher AUC is better")
    for test in (2024, 2025):
        tr, te = df[df.season < test], df[df.season == test]
        g = fit_any(tr); l = fit_lr(tr)
        pg, pl, pr = g.predict(te[F]), l.predict_proba(te[F])[:, 1], rate_baseline(tr, te)
        y = te.any_td
        base = np.full(len(te), tr.any_td.mean())
        for name, p in (("everyone same rate", base), ("player's own TD rate", pr), ("logistic regression", pl), ("boosted model", pg), ("boosted + logistic", (pg + pl) / 2)):
            print(f"  {test} {name:22} logloss {log_loss(y, p):.4f}  brier {brier_score_loss(y, p):.4f}  AUC {roc_auc_score(y, p):.3f}")
        cal = pd.DataFrame({"p": pg, "y": y}); cal["bin"] = pd.cut(cal.p, [0, .05, .1, .2, .3, .4, .5, .7, 1])
        print(cal.groupby("bin", observed=True).agg(model=("p", "mean"), actual=("y", "mean"), n=("y", "size")).round(3).T.to_string())
    print("\nQB TOUCHDOWN PASSES (starting QBs)")
    q = df[(df.position == "QB") & (df.is_starter == 1)]
    for test in (2024, 2025):
        tr, te = q[q.season < test], q[q.season == test]
        m = fit_qb(tr); lam = m.predict(te[QF])
        n = te.n_games; prior = te.p_pass_td.where(te.p_g >= 6).fillna(tr.passing_tds.mean())
        lam0 = ((te.e_passing_tds.fillna(0) * n + prior * 3) / (n + 3)).clip(0.2, 4)
        y = te.passing_tds
        for name, L in (("QB's own average", lam0), ("boosted model", lam)):
            ll = -poisson.logpmf(y, L).mean()
            p2 = 1 - poisson.cdf(1, L)
            print(f"  {test} {name:18} avg miss {np.mean(np.abs(y - L)):.3f}  count log loss {ll:.4f}  2+ TD passes: brier {brier_score_loss(y >= 2, p2):.4f} AUC {roc_auc_score(y >= 2, p2):.3f}")
