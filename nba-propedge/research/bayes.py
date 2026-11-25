"""A Bayesian player model, kept as a standing test next to the production engine.

  python research/bayes.py     compare it with the current blend on every DraftKings opening line on file

What it is: tonight's number = projected minutes x a per-minute rate x a matchup adjustment. The rate is a
gamma-Poisson posterior. It starts from the league rate for the player's position, is updated after every game
he plays, forgets old games gradually, and stays close to the starting rate while his sample is small.

Why it is not in the chances shown on the page: the production engine already carries recency-weighted form
at three speeds, and the sportsbook's line is weighed in on top of that. Tested on games the models had not
seen, this model on its own lands about level with the production one, and adding it to the blend does not
move the score by a measurable amount (the range of the difference straddles zero). The weekly retrain reruns
the comparison and writes it to the accuracy report, so if that changes as lines pile up it will show.

Also tried and measured in November 2026, none of which helped on held-out games:
  per-player updating of the production model's misses (a running, shrunk bias and consistency per player)
  feeding the posterior rates and a minutes projection into the production models as extra inputs
  separate blend weights per stat
"""
import os, sys
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.linear_model import LogisticRegression, PoissonRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import dataset, features as F
import train as T
from model import p_over, logit, NOFAIR

TG = F.TARGETS
PARTS = {"pts": ["pts"], "reb": ["reb"], "ast": ["ast"], "tpm": ["tpm"], "pra": ["pts", "reb", "ast"],
         "pr": ["pts", "reb"], "pa": ["pts", "ast"], "ra": ["reb", "ast"]}
HALF_LIFE, PRIOR_MIN, CARRY = 6.0, 100.0, 0.5  # games, minutes of prior weight, share of the record kept over a summer


def rates(p):
    """Posterior per-minute rate for every stat going into each game. p: output of features.build (time sorted)."""
    lam = 0.5 ** (1 / HALF_LIFE)
    played = (p["min"] > 0).values
    r0 = {g: (d.loc[d["min"] > 0, TG].sum() / d.loc[d["min"] > 0, "min"].sum()).values for g, d in p.groupby("posg")}
    Y = p[TG].fillna(0).values.astype(float); M = p["min"].fillna(0).values.astype(float)
    day = p["t"].values.astype("datetime64[s]").astype(np.int64) / 86400.0
    pos = p["posg"].values
    post = np.zeros((len(p), len(TG)))
    for ix in p.groupby("pid", sort=False).indices.values():
        sy = np.zeros(len(TG)); sm = 0.0; last = None; prior = r0[pos[ix[0]]]
        for i in ix:
            if last is not None and day[i] - last > 60:
                sy *= CARRY; sm *= CARRY
            post[i] = (PRIOR_MIN * prior + sy) / (PRIOR_MIN + sm)
            if played[i]:
                sy = lam * sy + Y[i]; sm = lam * sm + M[i]; last = day[i]
    return pd.DataFrame(post, columns=[f"br_{t}" for t in TG], index=p.index)


def minutes_oof(d):
    """Projected minutes for each season from a model that never saw that season."""
    out = pd.Series(index=d.index, dtype=float)
    for s in sorted(d.season.unique()):
        tr = d[d.season != s]
        cut = tr.t.quantile(0.85); a, b = tr[tr.t <= cut], tr[tr.t > cut]
        m = lgb.train(T.params("huber"), lgb.Dataset(a[F.FEATS], a["min"]), 2500, valid_sets=[lgb.Dataset(b[F.FEATS], b["min"])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        m = lgb.train(T.params("huber"), lgb.Dataset(tr[F.FEATS], tr["min"]), int(max(150, m.best_iteration) * 1.1))
        out[d.season == s] = m.predict(d.loc[d.season == s, F.FEATS])
    return out.clip(lower=1)


def matchup(d, t):
    dvp = sum(d[f"dvp_{c}"] for c in PARTS[t])
    return pd.DataFrame({
        "dvp": np.log((dvp / dvp.groupby(d["posg"]).transform("mean")).clip(0.5, 2)).fillna(0),
        "pace": np.log((d.pace_pair / d.pace_pair.mean()).clip(0.8, 1.25)).fillna(0),
        "total": np.log((d.own_total / d.own_total.mean()).clip(0.7, 1.4)).fillna(0),
        "home": d.home - 0.5, "b2b": d.b2b,
        "new_pts": (d.new_pts.fillna(0) / 20).clip(0, 3), "miss_min": (d.miss_min_pos.fillna(0) / 30).clip(0, 3),
        "blowout": (d.abs_spread.fillna(5) / 10).clip(0, 2)})


def project(d, t):
    """Minutes x posterior rate, times a small matchup adjustment fit on the other seasons."""
    base = np.clip((d.min_hat * d[f"br_{t}"]).values, 0.05, None)
    X, y, out = matchup(d, t), d[t].values, np.zeros(len(d))
    for s in sorted(d.season.unique()):
        tr = (d.season != s).values
        g = PoissonRegressor(alpha=1e-4, max_iter=300).fit(X[tr], y[tr] / base[tr], sample_weight=base[tr])
        out[~tr] = base[~tr] * g.predict(X[~tr])
    return out


def _cv(dk, cols, k=5):
    """Chances for every line from blend weights fit on other time blocks."""
    gids = dk.gid.drop_duplicates().tolist()
    fold = dk.gid.map({g: i * k // len(gids) for i, g in enumerate(gids)}).values
    X = np.column_stack([logit(np.where(np.isnan(dk[c].values), NOFAIR, dk[c].values)) for c in cols])
    pf = np.zeros(len(dk))
    for f in range(k):
        pf[fold == f] = LogisticRegression(C=1.0).fit(X[fold != f], dk.y.values[fold != f]).predict_proba(X[fold == f])[:, 1]
    return pf, LogisticRegression(C=1.0).fit(X, dk.y.values).coef_[0]


def compare(df, lined, log=print):
    """df: train.load(). lined: the production model's out-of-sample chances on every line (gid, pid, stat, line, fair, p, y, t).
    Returns the scores with and without the Bayesian model in the blend, and a resampled range for the difference."""
    p = F.build(dataset.load_games())
    br = pd.concat([p[["gid", "pid"]], rates(p)], axis=1)
    d = df.merge(br, on=["gid", "pid"], how="left")
    d["min_hat"] = minutes_oof(d).values
    rows, miss = [], {}
    for t in TG:
        loc = project(d, t)
        sm, cuts, zq = T.fit_shape(loc, d[t].values, d)
        x = d[["gid", "pid"]].assign(loc_b=loc, sc_b=T.scale(sm, d, loc))
        ln = lined[lined.stat == t].merge(x, on=["gid", "pid"])
        ln["pb"] = [p_over(a, b, cuts, zq, c) for a, b, c in zip(ln.loc_b, ln.sc_b, ln.line)]
        rows.append(ln)
        miss[t] = round(float(np.abs(d[t].values - loc).mean()), 2)
    dk = pd.concat(rows).sort_values("t").reset_index(drop=True)
    two, _ = _cv(dk, ["p", "fair"]); three, w = _cv(dk, ["p", "pb", "fair"])
    sq = lambda v: (v - dk.y.values) ** 2
    diff = pd.Series(sq(three) - sq(two))
    by_game = diff.groupby(dk.gid.values).agg(["sum", "size"]).values
    rng = np.random.default_rng(7)
    draws = [by_game[ix, 0].sum() / by_game[ix, 1].sum() for ix in rng.integers(0, len(by_game), (500, len(by_game)))]
    out = {"blend": round(float(sq(two).mean()), 4), "with_bayes": round(float(sq(three).mean()), 4), "diff": round(float(diff.mean()), 5),
           "low": round(float(np.percentile(draws, 2.5)), 5), "high": round(float(np.percentile(draws, 97.5)), 5),
           "weights": {"model": round(float(w[0]), 3), "bayes": round(float(w[1]), 3), "book": round(float(w[2]), 3)},
           "avg_miss": miss, "lines": int(len(dk))}
    log("Bayesian player model vs current blend:", out)
    return out


if __name__ == "__main__":
    from market import lined_oof
    df = T.load(); lines = T.load_lines()
    compare(df, pd.concat([lined_oof(df, lines, t)[0] for t in TG]))
