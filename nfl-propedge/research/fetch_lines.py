"""Save the DraftKings lines for every finished game this season to ../data/lines_<season>.csv.gz.

One row per player and market: the opening line and the last line before kickoff. These are what the
blend between the models and the sportsbook is fit on (fit_blend.py). Weeks already saved are skipped,
so the Tuesday retrain only fetches the week that just ended.

  python research/fetch_lines.py            current season
  python research/fetch_lines.py 2026 --all re-fetch every week
"""
import os, sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import dk_lines as DK

DATA = os.path.join(os.path.dirname(HERE), "data")


def fetch_week(season, week, ids):
    rows = []
    evs = [e for e in DK.week_events(season, week) if e[3]]
    with ThreadPoolExecutor(4) as ex:
        for (eid, home, away, _), gl in zip(evs, ex.map(lambda e: DK.game_lines(e[0]), evs)):
            for (a, kind), m in gl["main"].items():
                rows.append({"season": season, "week": week, "event": eid, "espn_id": a, "pid": ids.get(a), "kind": kind,
                             "open": m["open"], "close": m["line"], "book": gl["book"]})
            for a in gl["td1"]:
                rows.append({"season": season, "week": week, "event": eid, "espn_id": a, "pid": ids.get(a), "kind": "td1",
                             "open": None, "close": None, "book": gl["book"]})
    return rows


if __name__ == "__main__":
    now = datetime.now()
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    season = int(args[0]) if args else (now.year if now.month >= 9 else now.year - 1)
    path = os.path.join(DATA, f"lines_{season}.csv.gz")
    os.makedirs(DATA, exist_ok=True)
    have = pd.read_csv(path, dtype={"espn_id": str, "event": str}) if os.path.exists(path) and "--all" not in sys.argv else pd.DataFrame()
    done = set(have.week.unique()) if len(have) else set()
    ids = DK.id_map(os.path.join(HERE, "cache", "player_ids.csv"))
    new = []
    for week in range(1, 19):
        if week in done:
            continue
        evs = DK.week_events(season, week)
        if not evs or not all(e[3] for e in evs):
            break  # this week is not finished yet
        rows = fetch_week(season, week, ids)
        print(f"week {week}: {len(rows)} lines from {len(evs)} games", flush=True)
        new += rows
    if new:
        have = pd.concat([have, pd.DataFrame(new)], ignore_index=True)
        have.to_csv(path, index=False)
    print("saved", path, len(have), "rows; unmatched players:", int(have.pid.isna().sum()) if len(have) else 0)
