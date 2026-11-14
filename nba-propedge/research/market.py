"""Blend the model's read with the sportsbook's line.

The book's line already reflects news and money, and on its own it predicts about as well as the model.
So for any player with a line, the final chance comes from a small second-stage fit that weighs
(a) the model's chance of going over that line and (b) the book's own price. Measured on the test season.

  python research/market.py
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from train import load, load_lines, oof_center, fit_shape, fit_center, scale, p_over, FEATS, TARGETS

from model import logit, NOFAIR


def meta_X(p, fair):
    f = np.where(np.isnan(fair), NOFAIR, fair)
    return np.column_stack([logit(p), logit(f)])


def lined_oof(d, lines, t):
    """Out-of-sample model chances for every lined player-game in d."""
    loc = oof_center(d, t)
    sm, cuts, zq = fit_shape(loc.values, d[t].values, d)
    x = d.assign(loc=loc.values); x["sc"] = scale(sm, x, x["loc"].values)
    ln = x.merge(lines[lines.stat == t], on=["gid", "pid"])
    ln["p"] = [p_over(a, b, cuts, zq, c) for a, b, c in zip(ln["loc"], ln.sc, ln.line)]
    ln["y"] = (ln[t] > ln.line).astype(int); ln["actual"] = ln[t]; ln["stat"] = t
    return ln[["gid", "pid", "t", "season", "stat", "line", "over", "under", "fair", "loc", "sc", "p", "y", "actual"]], (sm, cuts, zq)


def fit_meta(r):
    return LogisticRegression(C=1.0).fit(meta_X(r.p.values, r.fair.values), r.y.values)


def cv_blend(dk, k=5):
    """Blend chances for every DraftKings row from weights fit on other time blocks."""
    dk = dk.sort_values("t").reset_index(drop=True)
    gids = dk.gid.drop_duplicates().tolist()
    fold = dk.gid.map({g: i * k // len(gids) for i, g in enumerate(gids)})
    pf = np.zeros(len(dk))
    for f in range(k):
        m = fit_meta(dk[fold != f])
        pf[(fold == f).values] = m.predict_proba(meta_X(dk.p[fold == f].values, dk.fair[fold == f].values))[:, 1]
    dk["pf"] = pf
    return dk


def bet_table(r):
    """Take whichever side the blend favors at the opening price; group by how far the chance clears break-even."""
    dec = lambda a: np.where(a > 0, 1 + a / 100.0, 1 + 100.0 / np.abs(a))
    r = r.copy()
    r["side_p"] = np.where(r.pf >= 0.5, r.pf, 1 - r.pf)
    r["won"] = np.where(r.pf >= 0.5, r.y, 1 - r.y)
    r["pay"] = np.where(r.pf >= 0.5, dec(r.over), dec(r.under))
    r["edge"] = r.side_p - 1 / r.pay
    out = []
    for lo in (-1, 0.0, 0.02, 0.04):
        x = r[r.edge > lo]
        out.append({"min_edge": lo, "bets": int(len(x)), "won": round(float(x.won.mean()), 3),
                    "ret": round(float((x.won * x.pay).mean() - 1), 3), "under_share": round(float((x.pf < 0.5).mean()), 2)})
    return out


if __name__ == "__main__":
    df = load(); lines = load_lines()
    dk = cv_blend(pd.concat([lined_oof(df, lines, t)[0] for t in TARGETS]))
    b = lambda p: float(np.mean((p - dk.y) ** 2))
    print(f"{len(dk)} DraftKings opening lines across {dk.gid.nunique()} games")
    print(f"Brier score (lower is better): blended {b(dk.pf):.4f} | model alone {b(dk.p):.4f} | book price {b(dk.fair):.4f} | coin flip 0.2500")
    print(dk.groupby(pd.cut(dk.pf, [0, .35, .4, .45, .5, .55, .6, 1])).agg(said=("pf", "mean"), happened=("y", "mean"), n=("y", "size")).round(3))
    for row in bet_table(dk):
        print(row)
