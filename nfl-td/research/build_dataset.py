"""Build the touchdown training set: every player-game since 2019 with pre-game features and results.
Downloads nflverse weekly stats and play-by-play into research/cache on first run (about 200 MB)."""
import os, sys, time, urllib.request
from datetime import datetime
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import td_engine as E
C = os.path.join(HERE, "cache"); os.makedirs(C, exist_ok=True)
STATS = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{y}.csv"
PBP = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{y}.parquet"
GAMES = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"

def cached(url, name, fresh=False):
    p = os.path.join(C, name)
    if fresh and os.path.exists(p):
        os.remove(p)
    if not os.path.exists(p):
        urllib.request.urlretrieve(url, p)
    return p

if __name__ == "__main__":
    t0 = time.time()
    now = datetime.now()
    this = now.year if now.month >= 9 else now.year - 1
    years = range(2018, this + 1)
    stats = pd.concat([pd.read_csv(cached(STATS.format(y=y), f"ps{y}.csv", fresh=(y == this)), low_memory=False) for y in years])
    pbp = pd.concat([pd.read_parquet(cached(PBP.format(y=y), f"pbp_{y}.parquet", fresh=(y == this)), columns=E.PBP_COLS) for y in years])
    sched = pd.read_csv(cached(GAMES, "games.csv", fresh=True))
    base = E.build_base(stats, pbp, sched)
    base["result_known"] = 1
    st = base[base.position == "QB"].sort_values("attempts", ascending=False).drop_duplicates(["game_id", "team"])
    base["is_starter"] = base.index.isin(st.index).astype(int)
    d = E.features(base)
    d = d[d.season >= 2019]
    d.to_parquet(os.path.join(C, "td_dataset.parquet"))
    print("dataset", d.shape, round(time.time() - t0), "s")
