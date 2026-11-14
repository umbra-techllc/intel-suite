"""Train the production models on everything and save them to ../model/.

Per stat (points, rebounds, assists, threes, and the four combinations):
  center  gradient-boosted projection from about 70 pre-game inputs
  spread  second model for how widely this player's results swing tonight
  shape   measured distribution of misses for 12 projection sizes, from predictions on seasons the model had not seen
Then, across stats:
  blend   weights that combine the model's chance with the sportsbook's price (fit on DraftKings opening lines)
  links   how results move together for the same player, for teammates, and for opponents (used for parlay math)
  report  held-out accuracy numbers shown on the page
"""
import gzip, json, os
import numpy as np, pandas as pd
from train import load, load_lines, oof_center, fit_shape, fit_center, scale, FEATS, TARGETS, ROOT
from market import cv_blend, fit_meta, bet_table
from model import p_over, anchor, SFEATS

OUT = os.path.join(ROOT, "model")
os.makedirs(OUT, exist_ok=True)


def save(model, name):
    with gzip.GzipFile(os.path.join(OUT, name + ".txt.gz"), "wb", compresslevel=9, mtime=0) as fh:
        fh.write(model.model_to_string().encode())


df = load(); lines = load_lines()
PARTS = {"pts": ["pts"], "reb": ["reb"], "ast": ["ast"], "tpm": ["tpm"], "pra": ["pts", "reb", "ast"],
         "pr": ["pts", "reb"], "pa": ["pts", "ast"], "ra": ["reb", "ast"]}
bundle = {"features": FEATS, "stats": {}, "trained_on": f"{len(df):,} player-games from {df.gid.nunique():,} games, "
          f"{pd.to_datetime(df.t.min()).strftime('%b %Y')} to {pd.to_datetime(df.t.max()).strftime('%b %-d, %Y')}"}
z = df[["gid", "pid", "tid", "oid", "min_m"]].copy()
lined, mae = [], {}
for t in TARGETS:
    loc = oof_center(df, t)
    sm, cuts, zq = fit_shape(loc.values, df[t].values, df)
    sc = scale(sm, df, loc.values)
    z[t] = (df[t].values - loc.values) / sc
    m, rounds = fit_center(df, t)
    save(m, f"{t}_center"); save(sm, f"{t}_spread")
    bundle["stats"][t] = {"cuts": np.round(cuts, 3).tolist(), "zq": [np.round(q, 4).tolist() for q in zq], "rounds": rounds}
    x = df.assign(loc=loc.values, sc=sc)
    ln = x.merge(lines[lines.stat == t], on=["gid", "pid"])
    ln["p"] = [p_over(a, b, cuts, zq, c) for a, b, c in zip(ln["loc"], ln.sc, ln.line)]
    ln["y"] = (ln[t] > ln.line).astype(int); ln["actual"] = ln[t]; ln["stat"] = t
    lined.append(ln[["gid", "pid", "t", "stat", "line", "over", "under", "fair", "loc", "sc", "p", "y", "actual"]])
    l10 = sum(df[f"{c}_m"] for c in PARTS[t])
    mae[t] = {"model": round(float((df[t] - loc).abs().mean()), 2), "recent_avg": round(float((df[t] - l10).abs().mean()), 2),
              "book_line": round(float((ln[t] - ln.line).abs().mean()), 2), "model_on_lined": round(float((ln[t] - ln["loc"]).abs().mean()), 2)}
    print(t, mae[t], "rounds", rounds, flush=True)

dk = cv_blend(pd.concat(lined))
meta = fit_meta(dk)
bundle["blend"] = {"model": round(float(meta.coef_[0][0]), 4), "book": round(float(meta.coef_[0][1]), 4), "base": round(float(meta.intercept_[0]), 4)}
brier = lambda p: round(float(np.mean((p - dk.y) ** 2)), 4)

# 'n or more' ladder: slide each projection to the blended chance on the main line, then check lower rungs
rows = []
for r in dk.itertuples():
    s = bundle["stats"][r.stat]
    cuts, zq = np.array(s["cuts"]), [np.array(q) for q in s["zq"]]
    loc2, tier = anchor(r.loc, r.sc, cuts, zq, r.line, r.pf)
    for f in (0.45, 0.6, 0.75, 0.9):
        n = int(np.ceil(r.line * f))
        if n >= 1:
            rows.append((p_over(loc2, r.sc, cuts, zq, n - 0.5, tier), int(r.actual >= n), r.gid))
lad = pd.DataFrame(rows, columns=["p", "y", "gid"]).drop_duplicates()
ladder = [{"said": round(float(g.p.mean()), 3), "happened": round(float(g.y.mean()), 3), "n": int(len(g))}
          for _, g in lad.groupby(pd.cut(lad.p, [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]), observed=True)]
main = [{"said": round(float(g.pf.mean()), 3), "happened": round(float(g.y.mean()), 3), "n": int(len(g))}
        for _, g in dk.groupby(pd.cut(dk.pf, [0, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 1]), observed=True)]

# how results move together
z = z[z.min_m >= 18]
def corr(pairs):
    out = {}
    for a in TARGETS:
        out[a] = {}
        for b in TARGETS:
            v = pairs[[a + "_x", b + "_y"]].dropna()
            w = pairs[[b + "_x", a + "_y"]].dropna()
            c = (np.corrcoef(v.iloc[:, 0], v.iloc[:, 1])[0, 1] + np.corrcoef(w.iloc[:, 0], w.iloc[:, 1])[0, 1]) / 2
            out[a][b] = round(float(c), 3)
    return out
same = z[TARGETS].corr().round(3).to_dict()
mate = z.merge(z, on=["gid", "tid"]); mate = mate[mate.pid_x < mate.pid_y]
opp = z.merge(z, left_on=["gid", "tid"], right_on=["gid", "oid"])
bundle["links"] = {"same": same, "mate": corr(mate), "opp": corr(opp)}

bundle["report"] = {
    "avg_miss": mae, "lines": int(len(dk)), "line_games": int(dk.gid.nunique()),
    "brier": {"blend": brier(dk.pf), "model": brier(dk.p), "book": brier(dk.fair), "coin": 0.25},
    "main": main, "ladder": ladder, "bets": bet_table(dk), "over_rate": round(float(dk.y.mean()), 3),
}
json.dump(bundle, open(os.path.join(OUT, "bundle.json"), "w"), separators=(",", ":"))
print(json.dumps({k: bundle[k] for k in ("blend", "report", "trained_on")}, indent=1))
print("links same pts:", same["pts"], "\nmate pts:", bundle["links"]["mate"]["pts"], "\nopp pts:", bundle["links"]["opp"]["pts"])
print("saved", sorted(os.listdir(OUT)), sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT)) // 1024, "KB")
