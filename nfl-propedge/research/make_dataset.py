"""Replay every week from 2019 on and write one row per prop: pre-game inputs + what actually happened.

Uses only data available before each game. Players who recorded a stat that week stand in for the
injury report, and the QB who threw the most stands in for the depth chart.
"""
import os, sys, urllib.request
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import build_data as B

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
SK = ["QB", "RB", "WR", "TE"]
YCOL = {"pass": "passing_yards", "rush": "rushing_yards", "rec": "receiving_yards"}

def cached(name, url):
    path = os.path.join(CACHE, name)
    if not os.path.exists(path):
        os.makedirs(CACHE, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path

GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"

def load(y):
    d = pd.read_csv(cached(f"ps{y}.csv", B.NFLVERSE_STATS.format(season=y)), low_memory=False)
    return d[(d.season_type == "REG") & d.position.isin(SK)].copy()

def season_rows(season):
    cur_all, prev = load(season), load(season - 1)
    sched = pd.read_csv(cached("games.csv", GAMES_URL))
    sched = sched[(sched.season == season) & (sched.game_type == "REG")]
    last = int(cur_all.week.max())
    if season == 2026:
        done = sched[sched.home_score.notna()].groupby("week").size()
        full = sched.groupby("week").size()
        last = max(w for w in done.index if done[w] == full[w])
    rows = []
    for w in range(2, last + 1):
        cur, wk = cur_all[cur_all.week < w], cur_all[cur_all.week == w]
        if cur.empty or wk.empty:
            continue
        games = []
        for g in sched[(sched.week == w) & sched.spread_line.notna()].itertuples():
            hs, tot = -float(g.spread_line), float(g.total_line)
            games.append({"id": g.game_id, "home": g.home_team, "away": g.away_team, "total": tot, "homeSpread": hs,
                          "homeITT": tot / 2 - hs / 2, "awayITT": tot / 2 + hs / 2, "wet": False,
                          "indoor": str(g.roof) in ("dome", "closed")})
        qb = wk[wk.position == "QB"].sort_values("attempts", ascending=False).drop_duplicates("team")
        depth = {r.team: {"QB": [r.player_display_name]} for r in qb.itertuples() if r.attempts >= 5}
        props, *_ = B.project(cur, prev, games, w, active=set(wk.player_id), depth=depth)
        act = wk.set_index("player_id")
        for p in props:
            if p["pid"] not in act.index:
                continue
            a = act.loc[p["pid"]]
            if isinstance(a, pd.DataFrame):
                a = a.iloc[0]
            rows.append({"season": season, "week": w, "pid": p["pid"], "name": p["name"], "team": p["team"], "kind": p["kind"],
                         "game": p["game"], "y": float(a[YCOL[p["kind"]]]), **p["feat"]})
    print(f"{season}: {len(rows)} rows", file=sys.stderr, flush=True)
    return rows

if __name__ == "__main__":
    # Fresh copies of the current season's stats and the schedule on every run
    for f in ("games.csv", f"ps{B.datetime.now().year}.csv"):
        try: os.remove(os.path.join(CACHE, f))
        except FileNotFoundError: pass
    this = B.datetime.now().year if B.datetime.now().month >= 9 else B.datetime.now().year - 1
    seasons = [int(x) for x in sys.argv[1:]] or list(range(2019, this + 1))
    for y in range(min(seasons) - 1, max(seasons) + 1):  # download sequentially before the workers start
        cached(f"ps{y}.csv", B.NFLVERSE_STATS.format(season=y))
    cached("games.csv", GAMES_URL)
    with Pool(min(len(seasons), os.cpu_count() or 2)) as pool:
        parts = pool.map(season_rows, seasons)
    df = pd.DataFrame([r for part in parts for r in part])
    for c in df.columns:
        if df[c].dtype == bool:
            df[c] = df[c].astype(float)
    df.to_csv(os.path.join(CACHE, "dataset.csv"), index=False)
    print(df.shape, df.groupby(["season", "kind"]).size().unstack().to_string())
