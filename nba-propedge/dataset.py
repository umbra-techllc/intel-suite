"""Reads and tops up the box-score history in data/ (one file per season so old seasons never change)."""
import datetime as dt, glob, os
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import espn

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIRST = "20241022"  # opening night 2024-25
IDS = {"gid": str, "pid": str, "tid": str, "oid": str}


def load_games():
    files = sorted(glob.glob(os.path.join(DATA, "games_*.csv.gz")))
    return pd.concat([pd.read_csv(f, dtype=IDS) for f in files], ignore_index=True) if files else pd.DataFrame()


def load_lines():
    files = sorted(glob.glob(os.path.join(DATA, "lines_*.csv.gz")))
    return pd.concat([pd.read_csv(f, dtype=IDS) for f in files], ignore_index=True) if files else pd.DataFrame()


def _one(ev, with_lines=True):
    s = espn.get_json(f"{espn.SITE}/summary?event={ev['id']}")
    rows = espn.parse_box(s, ev) if s and s.get("boxscore", {}).get("players") else []
    if rows and rows[0]["total"] is None:
        total, spread = espn.game_odds(ev["id"])
        for r in rows:
            r["total"] = total
            r["spread"] = None if spread is None else (spread if r["home"] else -spread)
    lines = []
    if with_lines:
        main, _, book = espn.props(ev["id"])
        # After tip-off ESPN's "current" numbers are in-game, so history keeps the opening line and prices.
        # Only DraftKings is kept: the ESPN BET openers stored for 2024-25 and early 2025-26 are unreliable.
        lines = [{"gid": ev["id"], "pid": m["pid"], "stat": m["stat"], "line": m["oline"], "over": m["oover"],
                  "under": m["ounder"], "book": book, "season": ev["season"]}
                 for m in main if book == "DraftKings" and m["oline"] is not None and m["oover"] is not None]
    return rows, lines


def fetch_new(games, since=None, with_lines=True, log=print):
    """Finished games not yet in `games`. Returns (player rows, line rows) as DataFrames."""
    have = set(games.gid.unique()) if len(games) else set()
    start = since or (str(games.date.max())[:10].replace("-", "") if len(games) else FIRST)
    d = dt.datetime.strptime(start, "%Y%m%d").date() - dt.timedelta(days=2)
    days = []
    while d <= dt.date.today() + dt.timedelta(days=1):
        days.append(f"{d:%Y%m%d}"); d += dt.timedelta(days=1)
    evs = {}
    with ThreadPoolExecutor(10) as ex:
        for day in ex.map(espn.scoreboard, days):
            for e in day:
                if e["final"] and e["stype"] in (2, 3) and e["id"] not in have:
                    evs[e["id"]] = e
    log(f"{len(evs)} new finished games since {start}")
    rows, lines = [], []
    with ThreadPoolExecutor(10) as ex:
        for i, (r, l) in enumerate(ex.map(lambda e: _one(e, with_lines), list(evs.values())), 1):
            rows += r; lines += l
            if i % 200 == 0:
                log(f"{i} games")
    return pd.DataFrame(rows), pd.DataFrame(lines)


def save(games, lines=None):
    os.makedirs(DATA, exist_ok=True)
    gz = {"method": "gzip", "compresslevel": 9, "mtime": 0}
    games = games.drop_duplicates(["gid", "pid"]).sort_values(["date", "gid", "tid", "pid"])
    for s, d in games.groupby("season"):
        d.to_csv(os.path.join(DATA, f"games_{s}.csv.gz"), index=False, compression=gz)
    if lines is not None and len(lines):
        lines = lines.drop_duplicates(["gid", "pid", "stat"]).sort_values(["gid", "pid", "stat"])
        for s, d in lines.groupby("season"):
            d.to_csv(os.path.join(DATA, f"lines_{s}.csv.gz"), index=False, compression=gz)
