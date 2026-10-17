"""How do teams redistribute an absent regular's carries and targets? Fit simple rules on 2024-25."""
import itertools, numpy as np, pandas as pd
rows = []
for yr in (2024, 2025):
    d = pd.read_csv(f"research/cache/ps{yr}.csv", low_memory=False)  # run backtest.py first to download
    d = d[(d.season_type == "REG") & d.position.isin(["QB", "RB", "WR", "TE"])].copy()
    tm = d.groupby(["team", "week"]).agg(car=("carries", "sum"), tgt=("targets", "sum")).reset_index()
    d = d.merge(tm, on=["team", "week"])
    d["cs"] = d.carries / d.car.clip(lower=1); d["ts"] = d.targets / d.tgt.clip(lower=1)
    d["yr"] = yr
    rows.append(d)
d = pd.concat(rows)
weeks = d.groupby(["yr", "team"]).week.apply(lambda s: sorted(s.unique())).to_dict()
events = []  # one per team-week
for (yr, team), wks in weeks.items():
    td = d[(d.yr == yr) & (d.team == team)]
    for i, w in enumerate(wks):
        if i < 2: continue
        prev_w = wks[max(0, i - 3):i]
        prev = td[td.week.isin(prev_w)].groupby(["player_id", "position"]).agg(cs=("cs", "sum"), ts=("ts", "sum"), g=("week", "nunique")).reset_index()
        prev["cs"] /= len(prev_w); prev["ts"] /= len(prev_w)
        cur = td[td.week == w].set_index("player_id")
        present = set(cur.index)
        prev["present"] = prev.player_id.isin(present)
        prev["cs_now"] = prev.player_id.map(cur.cs).fillna(0); prev["ts_now"] = prev.player_id.map(cur.ts).fillna(0)
        events.append((yr, team, w, prev))

def evaluate(kind, pos_ok, k, p, fill):
    """kind 'cs' or 'ts'. Vacated share from absent players: fraction `fill` is re-absorbed by present players;
    of that, `k` goes to same-position players (weighted by share**p), the rest to all present players by share."""
    errs, errs0, n_ev = [], [], 0
    for yr, team, w, prev in events:
        prev = prev[prev.position.isin(pos_ok)]
        out = prev[~prev.present & (prev[kind] >= (0.25 if kind == "cs" else 0.12))]
        if out.empty: continue
        pres = prev[prev.present & (prev.position != "QB")]
        if pres.empty: continue
        n_ev += 1
        new = pres[kind].copy()
        for o in out.itertuples():
            V = getattr(o, kind) * fill
            same = pres[pres.position == o.position]
            ws = same[kind] ** p if len(same) else None
            if ws is not None and ws.sum() > 0:
                new.loc[same.index] += V * k * ws / ws.sum()
                rest = V * (1 - k)
            else:
                rest = V
            wa = pres[kind]
            if wa.sum() > 0: new += rest * wa / wa.sum()
        m = pres[kind] >= 0.03
        errs += list((new[m] - pres[kind + "_now"][m]).abs())
        errs0 += list((pres[kind][m] - pres[kind + "_now"][m]).abs())
    return np.mean(errs), np.mean(errs0), n_ev, len(errs)

for kind, pos in (("cs", ["RB", "QB"]), ("ts", ["WR", "TE", "RB"])):
    res = []
    for k, p, fill in itertools.product([0, .3, .5, .7, .9], [.5, 1, 2, 3], [.5, .7, .85, 1.0]):
        mae, mae0, n_ev, n = evaluate(kind, pos if kind == "ts" else ["RB"], k, p, fill)
        res.append((mae, k, p, fill))
    res.sort()
    mae0 = evaluate(kind, pos if kind == "ts" else ["RB"], 0, 1, 0)[0]
    _, _, n_ev, n = evaluate(kind, pos if kind == "ts" else ["RB"], *res[0][1:])
    prop = [r for r in res if r[1] == 0 and r[3] == 1.0 and r[2] == 1][0]
    print(f"{'carries' if kind == 'cs' else 'targets'}: {n_ev} team-games with a regular missing, {n} player-games")
    print(f"  no redistribution MAE {mae0:.4f} | spread to everyone by share {prop[0]:.4f} | best {res[0][0]:.4f} with same-position k={res[0][1]} p={res[0][2]} fill={res[0][3]}")
    for r in res[1:4]: print(f"    next: {r[0]:.4f} k={r[1]} p={r[2]} fill={r[3]}")
