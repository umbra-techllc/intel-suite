"""DraftKings player lines for NFL games, read from the feed ESPN publishes. No key needed.

For the NFL, ESPN passes along the numbers DraftKings posts (the over/under line for each player, the
"this many or more" steps, and who is listed to score a touchdown) but not the prices. That is all this
project needs: the line is the sportsbook's own estimate of a player's typical game, and the engine blends
it with the models. Used by NFL PropEdge and the NFL TD Predictor.
"""
import io, os, time
import pandas as pd
import requests

CORE = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"
SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?seasontype=2&week={week}&dates={season}"
PLAYERS = "https://github.com/nflverse/nflverse-data/releases/download/players/players.csv"
BOOKS = {"100": "DraftKings", "58": "ESPN BET"}

# ESPN's market name -> our short name
MAIN = {"Total Passing Yards (incl. overtime)": "pass", "Total Rushing Yards (incl. overtime)": "rush",
        "Total Receiving Yards (incl. overtime)": "rec", "Total Passing Touchdowns (incl. overtime)": "ptd"}
STEPS = {"Passing Yards Milestones": "pass", "Rushing Yards Milestones": "rush", "Receiving Yards Milestones": "rec",
         "Passing Touchdown Milestones": "ptd"}
LISTED = {"Anytime Touchdown Scorer": "td1", "Player to score 2 or more touchdowns": "td2"}

_s = requests.Session()
_s.headers["User-Agent"] = "Mozilla/5.0 (umbra-intel-suite)"


def get_json(url, tries=4, timeout=40):
    last = None
    for i in range(tries):
        try:
            r = _s.get(url, timeout=timeout)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except Exception as e:  # network hiccup or a half-written response
            last = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"failed: {url} ({last})")


def id_map(cache=None):
    """ESPN athlete id -> nflverse player id (the id every stats file here uses)."""
    if cache and os.path.exists(cache) and time.time() - os.path.getmtime(cache) < 6 * 86400:
        d = pd.read_csv(cache, dtype=str)
    else:
        r = _s.get(PLAYERS, timeout=180)
        r.raise_for_status()
        d = pd.read_csv(io.StringIO(r.text), dtype=str, usecols=["gsis_id", "espn_id", "display_name"])
        d = d[d.gsis_id.notna() & d.espn_id.notna()]
        if cache:
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            d.to_csv(cache, index=False)
    return dict(zip(d.espn_id.str.replace(r"\.0$", "", regex=True), d.gsis_id))


def _athlete(item):
    ref = (item.get("athlete") or {}).get("$ref", "")
    return ref.split("athletes/")[1].split("?")[0] if "athletes/" in ref else None


def _target(item, when):
    v = ((item.get(when) or {}).get("target") or {}).get("value")
    return None if v is None else float(v)


def game_lines(event_id):
    """Lines for one game, keyed by ESPN athlete id.

    Returns {"book", "main": {(athlete, kind): {"line", "open"}}, "steps": {(athlete, kind): [numbers]},
             "td1": set of athletes listed to score, "td2": set listed to score twice}
    kind is pass / rush / rec (yards) or ptd (touchdown passes).
    """
    odds = get_json(f"{CORE}/events/{event_id}/competitions/{event_id}/odds?limit=20") or {}
    provs = [str(i["provider"]["id"]) for i in odds.get("items", []) if "propBets" in i
             and "live" not in i["provider"].get("name", "").lower()]
    provs.sort(key=lambda p: {"100": 0, "58": 1}.get(p, 5))
    out = {"book": None, "main": {}, "steps": {}, "td1": set(), "td2": set()}
    for prov in provs:
        items, page = [], 1
        while True:
            d = get_json(f"{CORE}/events/{event_id}/competitions/{event_id}/odds/{prov}/propBets?limit=1000&page={page}")
            if not d:
                break
            items += d.get("items", [])
            if page >= d.get("pageCount", 1):
                break
            page += 1
        for it in items:
            t, a = (it.get("type") or {}).get("name"), _athlete(it)
            if not a:
                continue
            if t in MAIN:
                line = _target(it, "current")
                if line is not None:
                    out["main"].setdefault((a, MAIN[t]), {"line": line, "open": _target(it, "open")})
            elif t in STEPS:
                n = _target(it, "current")
                if n is not None:
                    out["steps"].setdefault((a, STEPS[t]), set()).add(n)
            elif t in LISTED:
                out[LISTED[t]].add(a)
        if out["main"] or out["td1"]:
            out["book"] = BOOKS.get(prov, prov)
            out["steps"] = {k: sorted(v) for k, v in out["steps"].items()}
            return out
    out["steps"] = {}
    return out


def week_events(season, week):
    """[(event id, home abbreviation, away abbreviation, finished?)] for one regular-season week."""
    sb = get_json(SCOREBOARD.format(season=season, week=week)) or {}
    out = []
    for e in sb.get("events", []):
        c = e["competitions"][0]
        t = {x["homeAway"]: x["team"]["abbreviation"] for x in c["competitors"]}
        out.append((e["id"], t["home"], t["away"], bool(c.get("status", {}).get("type", {}).get("completed"))))
    return out
