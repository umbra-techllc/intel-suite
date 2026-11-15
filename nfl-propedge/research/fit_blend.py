"""Fit the blend between the models and the DraftKings line, and write the accuracy report the page shows.

Runs after fetch_lines.py and make_dataset.py. Steps:
  1. Train the projection models on every season before this one and predict this season's games,
     so every prediction here is for a game the model had not seen.
  2. Match those predictions to the DraftKings lines saved in ../data/ and to what each player did.
  3. Fit how much weight the model's read gets against the line (logistic regression on the log-odds scale,
     one weight for the model plus a starting point per market, since rushing lines have gone under more
     often than over). Both the opening line and the last line before kickoff are used.
  4. Pick, per market, how wide the range of outcomes around the line should be so that the chances for
     "this many or more" bets match what happened.
  5. Save the weights and a report (average miss, how often overs won, said-vs-happened table) to ../model/blend.json.
"""
import json, os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from train import df, FEATS, fit_gbm, fit_ridge
import build_data as B, engine as E

KINDS = ["pass", "rush", "rec"]
RIDGE = {"pass": True, "rush": False, "rec": True}
STEPS = (0.4, 0.55, 0.7, 0.85, 1.15, 1.3, 1.5, 1.8)   # "or more" numbers tested, as a share of the line
SCALES = (0.8, 0.9, 1.0, 1.1, 1.2, 1.3)
SEASON = int(df.season.max())
bundle = json.load(open(os.path.join(B.MODEL_DIR, "bundle.json")))


def ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def unseen_predictions():
    out = []
    for kind in KINDS:
        d = df[df.kind == kind]
        tr, te = d[d.season < SEASON], d[d.season == SEASON].copy()
        g, _ = fit_gbm(tr, kind, "huber")
        p = g.predict(te[FEATS])
        if RIDGE[kind]:
            p = (p + fit_ridge(tr).predict(te[FEATS])) / 2
        te["loc"] = np.clip(p, 1, None)
        m = bundle["markets"][kind]
        te["sc"] = (np.clip(B.load_booster(f"{kind}_spread").predict(np.column_stack([te[FEATS].astype(float).values, te["loc"].values])), 3, None)
                    if m["spread_model"] else np.nan)
        out.append(te[["week", "pid", "kind", "y", "loc", "sc", "recent_avg"]])
        print(kind, len(te), "games predicted without having seen them", flush=True)
    return pd.concat(out)


def design(d, col):
    X = pd.DataFrame({"model": d["lg_" + col].values})
    for k in KINDS:
        X[k] = (d.kind.values == k).astype(float)
    return X


def step_number(kind, line, mult):
    return round(line * mult / 25) * 25 if kind == "pass" else max(5, round(line * mult / 5) * 5)


def ladder(d, w, scale):
    """Chance and result for a spread of 'this many or more' numbers around each line."""
    rows = []
    for r, g in zip(d.itertuples(), d.grid):
        L = r.close
        a = E.anchor(g, L, E.blend(E.p_over(g, L), {"base": w[r.kind], "model": w["model"]}), scale[r.kind])
        for mult in STEPS:
            n = step_number(r.kind, L, mult)
            rows.append((r.kind, E.p_over(a, n - 0.5), float(r.y >= n)))
    return pd.DataFrame(rows, columns=["kind", "p", "y"])


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(HERE), "data", f"lines_{SEASON}.csv.gz")
    if not os.path.exists(path) or (df.season == SEASON).sum() < 300:
        print("Too early in the season to refit. Keeping the current blend.")
        sys.exit(0)
    lines = pd.read_csv(path)
    lines = lines[lines.kind.isin(KINDS) & lines.pid.notna() & lines.open.notna() & lines.close.notna()]
    d = unseen_predictions().merge(lines[["week", "pid", "kind", "open", "close"]], on=["week", "pid", "kind"])
    if len(d) < 400:
        print(f"Only {len(d)} lines matched so far this season. Keeping the current blend.")
        sys.exit(0)
    d["grid"] = [E.grid(bundle["markets"][r.kind], r.loc, r.sc) for r in d.itertuples()]
    d["mid"] = [g[50] for g in d.grid]
    for col in ("open", "close"):
        d["over_" + col] = (d.y > d[col]).astype(int)
        d["lg_" + col] = [E.logit(E.p_over(g, l)) for g, l in zip(d.grid, d[col])]
    print(len(d), "player lines matched to a prediction and a result, weeks", int(d.week.min()), "to", int(d.week.max()))

    X = pd.concat([design(d, "open"), design(d, "close")])
    y = np.r_[d.over_open.values, d.over_close.values]
    lr = LogisticRegression(C=0.2, fit_intercept=False).fit(X, y)
    w = dict(zip(X.columns, (float(c) for c in lr.coef_[0])))
    print("weights", {k: round(v, 3) for k, v in w.items()})

    scale = {}
    for k in KINDS:
        dk = d[d.kind == k]
        res = [(ll(t.y.values, t.p.values), s) for s in SCALES for t in [ladder(dk, w, {k: s})]]
        scale[k] = min(res)[1]
    print("range width", scale)

    # week-by-week check of the over/under call: fit on earlier weeks only, then score the next week
    wf = []
    for wk in sorted(d.week.unique())[2:]:
        tr, te = d[d.week < wk], d[d.week == wk]
        m = LogisticRegression(C=0.2, fit_intercept=False).fit(pd.concat([design(tr, "open"), design(tr, "close")]),
                                                              np.r_[tr.over_open.values, tr.over_close.values])
        p = m.predict_proba(design(te, "close"))[:, 1]
        wf.append(pd.DataFrame({"p": p, "y": te.over_close.values}))
    wf = pd.concat(wf)
    side = np.where(wf.p >= 0.5, wf.y, 1 - wf.y)
    conf = np.where(wf.p >= 0.5, wf.p, 1 - wf.p)

    t = ladder(d, w, scale)
    t["bin"] = pd.cut(t.p, [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
    tab = t.groupby("bin", observed=True).agg(said=("p", "mean"), happened=("y", "mean"), n=("y", "size"))
    names = {"pass": "Passing yards", "rush": "Rushing yards", "rec": "Receiving yards"}
    report = {
        "season": SEASON, "weeks": [int(d.week.min()), int(d.week.max())], "lines": int(len(d)),
        "avg_miss": {k: {"model": round(float((g.y - g.mid).abs().mean()), 1), "line": round(float((g.y - g.close).abs().mean()), 1),
                         "recent_avg": round(float((g.y - g.recent_avg).abs().mean()), 1), "n": int(len(g)), "name": names[k]}
                     for k, g in d.groupby("kind")},
        "over_rate": {k: round(float(g.over_close.mean()), 3) for k, g in d.groupby("kind")},
        "ladder": [{"said": round(float(r.said), 3), "happened": round(float(r.happened), 3), "n": int(r.n)} for r in tab.itertuples()],
        "ladder_bets": int(len(t)),
        "main_line": {"bets": int(len(wf)), "won": round(float(side.mean()), 3), "said": round(float(conf.mean()), 3),
                      "strong": {"bets": int((conf >= 0.55).sum()), "won": round(float(side[conf >= 0.55].mean()), 3) if (conf >= 0.55).any() else None}},
    }
    out = {"weights": {k: {"base": round(w[k], 4), "model": round(w["model"], 4), "scale": scale[k]} for k in KINDS}, "report": report}
    json.dump(out, open(os.path.join(B.MODEL_DIR, "blend.json"), "w"), separators=(",", ":"))
    print(json.dumps(report, indent=1))
