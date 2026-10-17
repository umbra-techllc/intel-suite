"""Replay the projection engine over a past season, one week at a time, using only data before each week.

Injury report stand-in: players who actually recorded a stat that week are treated as active
(perfect injury info, so real-world results will be a little worse). No weather adjustment.
Writes backtest_<season>.csv with one row per graded prop. Data is downloaded to research/cache on first run.
Refit the outcome ranges afterwards with: python research/fit_calibration.py
"""
import os, sys, math, urllib.request
import numpy as np, pandas as pd
from scipy.stats import gamma
import build_data as B

season = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
SK = ["QB", "RB", "WR", "TE"]
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research", "cache")
os.makedirs(CACHE, exist_ok=True)
def cached(name, url):
    path = os.path.join(CACHE, name)
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return path
def load(y):
    d = pd.read_csv(cached(f"ps{y}.csv", B.NFLVERSE_STATS.format(season=y)), low_memory=False)
    return d[(d.season_type == "REG") & d.position.isin(SK)].copy()
cur_all, prev = load(season), load(season - 1)
sched = pd.read_csv(cached("games.csv", "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"))
sched = sched[(sched.season == season) & (sched.game_type == "REG")]

rows = []
for w in range(4, int(cur_all.week.max()) + 1):
    cur = cur_all[cur_all.week < w]
    wk = cur_all[cur_all.week == w]
    games = []
    for g in sched[sched.week == w].itertuples():
        hs = -float(g.spread_line); tot = float(g.total_line)
        games.append({"id": g.game_id, "home": g.home_team, "away": g.away_team, "total": tot, "homeSpread": hs,
                      "homeITT": tot / 2 - hs / 2, "awayITT": tot / 2 + hs / 2, "wet": False})
    # Depth-chart stand-in: the QB who threw the most that week is the listed starter
    qb = wk[wk.position == "QB"].sort_values("attempts", ascending=False).drop_duplicates("team")
    depth = {r.team: {"QB": [r.player_display_name]} for r in qb.itertuples() if r.attempts >= 5}
    props, league, _, inj = B.project(cur, prev, games, w, active=set(wk.player_id), depth=depth)
    act = wk.set_index("player_id")[["passing_yards", "rushing_yards", "receiving_yards", "attempts", "carries", "targets"]].to_dict("index")
    for p in props:
        a = act.get(p["pid"])
        if not a: continue
        y = a[{"pass": "passing_yards", "rush": "rushing_yards", "rec": "receiving_yards"}[p["kind"]]]
        logs = [l[p["kind"]] for l in p["logs"]]
        wts = [B.week_weight(l["wk"], w) for l in p["logs"]]
        recent = B.wavg(logs, wts)
        d = p["drivers"]
        rows.append({"week": w, "pid": p["pid"], "name": p["name"], "pos": p["pos"], "kind": p["kind"], "n": p["n"],
                     "mu": p["mu"], "median": p["median"], "k": p["k"], "theta": p["theta"], "estLine": p["estLine"],
                     "recent": recent, "season_avg": float(np.mean(logs)), "y": y,
                     "oppF": d.get("oppF", 1), "scriptPass": d["scriptPass"], "scriptRush": d["scriptRush"],
                     "ittF": d.get("ittF", 1), "spreadFor": d["spreadFor"],
                     "vol_proj": d.get("att") or d.get("car") or d.get("tgt"),
                     "vol_act": a["attempts"] if p["kind"] == "pass" else a["carries"] if p["kind"] == "rush" else a["targets"],
                     "boosted": bool(p["role"] and p["role"].get("from")), "first_start": bool(p["role"] and p["role"].get("firstStart")),
                     "replacing": bool(p["role"] and p["role"].get("replacing")),
                     "share_now": d.get("tshare") if p["kind"] == "rec" else d.get("cshare"),
                     "share_base": d.get("tshareBase") if p["kind"] == "rec" else d.get("cshareBase")})
    print(f"week {w}: {len(props)} props", file=sys.stderr)

df = pd.DataFrame(rows)
df.to_csv(f"backtest_{season}.csv", index=False)
print(len(df), "graded props")
