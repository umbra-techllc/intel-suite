"""ESPN fetch and parse helpers shared by the dataset builder and the daily build.

Everything here is public ESPN data and needs no key: schedules, box scores, injuries, and the
sportsbook prop lines ESPN republishes (DraftKings from 2025-26 on, ESPN BET before that).
"""
import time
import requests

SITE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
CORE = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba"
_s = requests.Session()
_s.headers["User-Agent"] = "Mozilla/5.0 (umbra-intel-suite)"

# prop type name -> our stat key
MAIN = {
    "Total Points": "pts", "Total Rebounds": "reb", "Total Assists": "ast",
    "Total 3-Point Field Goals": "tpm", "Total Points, Rebounds, and Assists": "pra",
    "Total Points and Rebounds": "pr", "Total Points and Assists": "pa", "Total Assists and Rebounds": "ra",
}
ALT = {
    "Points Milestones": "pts", "Rebounds Milestones": "reb", "Assists Milestones": "ast",
    "3-Point Field Goals Milestones": "tpm", "Points + Assists + Rebounds Milestones": "pra",
    "Points + Rebounds Milestones": "pr", "Points + Assists Milestones": "pa", "Rebounds + Assists Milestones": "ra",
}
STATS = ["pts", "reb", "ast", "tpm", "pra", "pr", "pa", "ra"]
PARTS = {"pts": ["pts"], "reb": ["reb"], "ast": ["ast"], "tpm": ["tpm"], "pra": ["pts", "reb", "ast"],
         "pr": ["pts", "reb"], "pa": ["pts", "ast"], "ra": ["reb", "ast"]}


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


def scoreboard(dates):
    """dates: 'YYYYMMDD' or 'YYYYMMDD-YYYYMMDD'. Returns simplified events."""
    d = get_json(f"{SITE}/scoreboard?dates={dates}&limit=1000") or {}
    out = []
    for e in d.get("events", []):
        c = e["competitions"][0]
        if c.get("type", {}).get("abbreviation") == "ALLSTAR":
            continue
        home = next(x for x in c["competitors"] if x["homeAway"] == "home")
        away = next(x for x in c["competitors"] if x["homeAway"] == "away")
        odds = (c.get("odds") or [{}])[0]
        out.append({
            "id": e["id"], "date": e["date"], "state": e["status"]["type"]["state"],
            "final": e["status"]["type"]["name"] == "STATUS_FINAL",
            "status": e["status"]["type"]["name"],
            "stype": e.get("season", {}).get("type"), "season": e.get("season", {}).get("year"),
            "home": {"id": str(home["team"]["id"]), "abbr": home["team"].get("abbreviation", "?"), "name": home["team"].get("displayName", "")},
            "away": {"id": str(away["team"]["id"]), "abbr": away["team"].get("abbreviation", "?"), "name": away["team"].get("displayName", "")},
            "total": odds.get("overUnder"), "spread": odds.get("spread"),  # spread: negative = home favored
        })
    return out


def _num(x):
    try:
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return 0


def _pair(x):
    if isinstance(x, str) and "-" in x:
        a, b = x.split("-", 1)
        return _num(a), _num(b)
    return 0, 0


def parse_box(summary, ev):
    """Player rows for one finished game. ev is the scoreboard event (for date, teams, odds)."""
    rows = []
    pc = (summary.get("pickcenter") or [{}])[0]
    total = pc.get("overUnder", ev.get("total"))
    spread = pc.get("spread", ev.get("spread"))
    comps = summary["header"]["competitions"][0]["competitors"]
    score = {str(c["id"]): _num(c.get("score")) for c in comps}
    for tb in summary.get("boxscore", {}).get("players", []):
        tid = str(tb["team"]["id"])
        is_home = tid == ev["home"]["id"]
        me, opp = (ev["home"], ev["away"]) if is_home else (ev["away"], ev["home"])
        st = (tb.get("statistics") or [{}])[0]
        labels = st.get("labels") or st.get("names") or []
        ix = {k: i for i, k in enumerate(labels)}
        for a in st.get("athletes", []):
            ath = a.get("athlete") or {}
            if not ath.get("id"):
                continue
            s = a.get("stats") or []
            g = lambda k: s[ix[k]] if k in ix and ix[k] < len(s) else "0"
            fgm, fga = _pair(g("FG")); tpm, tpa = _pair(g("3PT")); ftm, fta = _pair(g("FT"))
            mins = _num(g("MIN")) if s else 0
            rows.append({
                "gid": ev["id"], "date": ev["date"], "season": ev["season"], "stype": ev["stype"],
                "team": me["abbr"], "tid": tid, "opp": opp["abbr"], "oid": opp["id"], "home": int(is_home),
                "pid": str(ath["id"]), "name": ath.get("displayName", ""),
                "pos": (ath.get("position") or {}).get("abbreviation", ""),
                "starter": int(bool(a.get("starter"))), "min": mins,
                "pts": _num(g("PTS")), "fgm": fgm, "fga": fga, "tpm": tpm, "tpa": tpa, "ftm": ftm, "fta": fta,
                "reb": _num(g("REB")), "ast": _num(g("AST")), "tov": _num(g("TO")), "stl": _num(g("STL")),
                "blk": _num(g("BLK")), "oreb": _num(g("OREB")),
                "total": total, "spread": (spread if is_home else (-spread if spread is not None else None)),
                "tm_score": score.get(tid, 0), "opp_score": score.get(opp["id"], 0),
            })
    return rows


def game_odds(event_id):
    """(total, home spread) from ESPN's odds feed, skipping in-game lines. Home spread is negative when home is favored."""
    d = get_json(f"{CORE}/events/{event_id}/competitions/{event_id}/odds?limit=20") or {}
    for i in d.get("items", []):
        if "live" in (i.get("provider", {}).get("name", "")).lower():
            continue
        if i.get("overUnder") is not None:
            return i.get("overUnder"), i.get("spread")
    return None, None


def _aid(item):
    ref = (item.get("athlete") or {}).get("$ref", "")
    return ref.split("athletes/")[1].split("?")[0] if "athletes/" in ref else None


def _price(item):
    try:
        return int(str(item["odds"]["american"]["value"]).replace("+", ""))
    except Exception:
        return None


def _imp(price):
    return -price / (-price + 100.0) if price < 0 else 100.0 / (price + 100.0)


def _clean_ladders(alt, main):
    """ESPN files some first-quarter ladders under the full-game name (rebounds, for one). The feed lists each
    ladder as a run of rising numbers, so split into runs, keep the run that agrees with the main line (or
    reaches the highest numbers), and only add rungs from other runs if they fit between its prices."""
    groups, out = {}, []
    for a in alt:
        groups.setdefault((a["pid"], a["stat"]), []).append(a)
    for key, items in groups.items():
        runs = [[items[0]]]
        for a in items[1:]:
            if a["n"] > runs[-1][-1]["n"]:
                runs[-1].append(a)
            else:
                runs.append([a])
        m = main.get(key)
        if m:
            near = lambda run: min(run, key=lambda a: abs(a["n"] - (m["line"] + 0.5)))
            base = min(runs, key=lambda run: abs(near(run)["n"] - (m["line"] + 0.5)) * 0.2 + abs(_imp(near(run)["price"]) - 0.5))
        else:
            base = max(runs, key=lambda run: (run[-1]["n"], len(run)))
        keep = {a["n"]: a for a in base}
        for run in runs:
            if run is base:
                continue
            for a in run:
                if a["n"] in keep:
                    continue
                lo = [k for k in keep if k < a["n"]]; hi = [k for k in keep if k > a["n"]]
                p = _imp(a["price"])
                if (not lo or _imp(keep[max(lo)]["price"]) >= p) and (not hi or _imp(keep[min(hi)]["price"]) <= p) and (lo or hi):
                    keep[a["n"]] = a
        out += [keep[n] for n in sorted(keep)]
    return out


def _open_price(item):
    try:
        return int(str(item["odds"]["american"]["open"]).replace("+", ""))
    except Exception:
        return None


def _open_line(item):
    try:
        return float(item["odds"]["total"]["open"])
    except Exception:
        return None


def prop_providers(event_id):
    d = get_json(f"{CORE}/events/{event_id}/competitions/{event_id}/odds?limit=20") or {}
    ids = [str(i["provider"]["id"]) for i in d.get("items", []) if "propBets" in i]
    return sorted(ids, key=lambda p: {"100": 0, "58": 1}.get(p, 5))


def props(event_id, provider=None):
    """Sportsbook player lines for one game.

    Returns (main, alt, book):
      main: [{pid, stat, line, over, under, oline, oover, ounder}]   American prices; the o* fields are the
            opening numbers. Once a game has started, line/over/under are in-game values, so history uses the openers.
      alt:  [{pid, stat, n, price}]                   'n or more' at that price
    """
    provs = [provider] if provider else prop_providers(event_id)
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
        if not items:
            continue
        main, alt, seen = [], [], {}
        for it in items:
            t = (it.get("type") or {}).get("name"); pid = _aid(it); price = _price(it)
            if not pid or price is None:
                continue
            tgt = ((it.get("current") or {}).get("target") or {}).get("value")
            if tgt is None:
                continue
            if t in MAIN:
                key = (pid, MAIN[t])
                if key not in seen:  # ESPN lists the over first, then the under
                    seen[key] = {"pid": pid, "stat": MAIN[t], "line": float(tgt), "over": price, "under": None,
                                 "oline": _open_line(it), "oover": _open_price(it), "ounder": None}
                    main.append(seen[key])
                elif seen[key]["under"] is None:
                    seen[key]["under"] = price
                    seen[key]["ounder"] = _open_price(it)
            elif t in ALT:
                alt.append({"pid": pid, "stat": ALT[t], "n": int(float(tgt)), "price": price})
        if main or alt:
            return main, _clean_ladders(alt, seen), {"100": "DraftKings", "58": "ESPN BET"}.get(str(prov), str(prov))
    return [], [], None


def injuries():
    """League injury report: {athlete id: {status, note, date, name, team}}."""
    d = get_json(f"{SITE}/injuries") or {}
    out = {}
    for t in d.get("injuries", []):
        for i in t.get("injuries", []):
            a = i.get("athlete") or {}
            pid = None
            for l in a.get("links", []):
                h = l.get("href", "")
                if "/id/" in h:
                    pid = h.split("/id/")[1].split("/")[0]
                    break
            if not pid:
                continue
            det = i.get("details") or {}
            out[pid] = {"status": i.get("status", ""), "note": i.get("shortComment", ""), "date": i.get("date", ""),
                        "name": a.get("displayName", ""), "team": t.get("displayName", ""),
                        "tag": (det.get("fantasyStatus") or {}).get("abbreviation", ""), "ret": det.get("returnDate", ""),
                        "what": " ".join(x for x in (det.get("side", ""), det.get("type", "")) if x and x != "Not Specified").strip()}
    return out
