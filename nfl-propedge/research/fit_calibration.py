"""Fit outcome ranges (actual / projected average) by market and size of projection.

Usage: python research/fit_calibration.py            -> writes calibration.json from 2024 + 2025
       python research/fit_calibration.py --check    -> fits on 2024, reports calibration on 2025
"""
import json, sys
import numpy as np, pandas as pd

def fit(df):
    out = {}
    for kind, g in df.groupby("kind"):
        cuts = np.quantile(g.mu, [1 / 3, 2 / 3]).round(1).tolist()
        tier = np.searchsorted(cuts, g.mu, side="left")
        qs = []
        for t in range(3):
            r = (g.y / g.mu)[tier == t].values
            qs.append(np.round(np.quantile(r, np.linspace(0, 1, 101)), 4).tolist())
        out[kind] = {"cuts": cuts, "q": qs, "n": int(len(g))}
    return out

def p_over(cal, kind, mu, line):
    c = cal[kind]; t = int(sum(mu > x for x in c["cuts"]))
    q = np.array(c["q"][t]); r = line / mu
    # interpolate the CDF from the quantile grid
    if r <= q[0]: return 1.0
    if r >= q[-1]: return 0.0
    i = np.searchsorted(q, r, side="right") - 1
    lo, hi = q[i], q[i + 1]
    frac = 0 if hi == lo else (r - lo) / (hi - lo)
    return 1 - (i + frac) / 100

if __name__ == "__main__":
    a, b = pd.read_csv("backtest_2024.csv"), pd.read_csv("backtest_2025.csv")
    if "--check" in sys.argv:
        cal = fit(a)
        recs = []
        for mult in [0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.4]:
            for r in b.itertuples():
                L = np.floor(r.mu * mult) + 0.5
                recs.append((p_over(cal, r.kind, r.mu, L), r.y > L))
        c = pd.DataFrame(recs, columns=["said", "over"])
        c["bin"] = pd.cut(c.said, [0, .2, .35, .45, .55, .65, .8, 1])
        t = c.groupby("bin", observed=True).agg(said=("said", "mean"), happened=("over", "mean"), n=("said", "size"))
        print("fit on 2024, tested on 2025:"); print(t.round(3).to_string())
        print("avg calibration error", round(float((abs(t.said - t.happened) * t.n).sum() / t.n.sum()), 4))
    else:
        cal = fit(pd.concat([a, b]))
        cal["fitted_on"] = "2024-2025 regular seasons, replayed week by week"
        json.dump(cal, open("calibration.json", "w"), separators=(",", ":"))
        print({k: (v["cuts"], v["n"], v["q"][1][50]) for k, v in cal.items() if isinstance(v, dict)})
