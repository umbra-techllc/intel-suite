"""Builds tonight's NBA board: projections, chances for every posted bet, and the AI-picked parlay.

  python build_nba.py                 next slate with games that have not started
  python build_nba.py --no-ai         skip the AI step (model ranking only)

Steps: top up box scores -> find the slate -> read the injury report -> work out who is expected to play ->
compute the same inputs the models were trained on -> project every player -> pull the sportsbook's lines ->
blend model and line into a chance for each bet -> let the AI choose and order the parlay legs -> grade
earlier picks -> write ../nba_parlay_data.json, which nba_parlay_finder.html loads. Exit code 3 means nothing to build.
"""
import argparse, datetime as dt, json, os, sys
from concurrent.futures import ThreadPoolExecutor
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd

import dataset, espn, features as F, model as M, picker

HERE = os.path.dirname(os.path.abspath(__file__))
ET = ZoneInfo("America/New_York")
LOG = os.path.join(HERE, "data", "picks_log.json")
UNIT = {"pts": "points", "reb": "rebounds", "ast": "assists", "tpm": "threes", "pra": "points + rebounds + assists",
        "pr": "points + rebounds", "pa": "points + assists", "ra": "rebounds + assists"}
OUT_WORDS = ("ruled out", "will not play", "won't play", "is out for", "listed as out", "will miss", "will sit out",
             "not expected to play", "has been ruled out", "will be sidelined", "out for the season", "remains out")
OK_WORDS = ("probable", "is available", "will play", "expected to play", "will return", "cleared to play", "good to go")


def find_slate(now):
    for add in range(3):
        day = (now.astimezone(ET) + dt.timedelta(days=add)).strftime("%Y%m%d")
        evs = [e for e in espn.scoreboard(day) if e["stype"] in (2, 3)]
        if any(e["state"] == "pre" for e in evs):
            return day, evs
    return None, []


def read_status(i, game_time, now):
    """out / doubtful / questionable / probable / '' from ESPN's status, expected return date and note."""
    if not i:
        return ""
    note = (i.get("note") or "").lower()
    ret = i.get("ret") or ""
    game_day = game_time.astimezone(ET).strftime("%Y-%m-%d")
    fresh = False
    try:
        fresh = (now - dt.datetime.fromisoformat(i["date"].replace("Z", "+00:00"))).total_seconds() < 40 * 3600
    except Exception:
        pass
    if i.get("status") in ("Out", "Suspension"):
        if i.get("tag") == "GTD" and ret and ret <= game_day:
            return "questionable"
        return "out"
    if "doubtful" in note:
        return "doubtful"
    if fresh and any(w in note for w in OUT_WORDS) and not (ret and ret < game_day):
        return "out"
    if any(w in note for w in OK_WORDS):
        return "probable"
    return "questionable"


def expected_players(hist, slate, inj, now):
    """A row for every player likely to play tonight: on the team's floor in the last 5 games and not ruled out."""
    played = hist[hist["min"] > 0]
    last_team = played.sort_values("date").groupby("pid").tail(1).set_index("pid")["tid"]
    rows, status, absent = [], {}, {}
    for e in slate:
        if e["state"] != "pre":
            continue
        t = dt.datetime.fromisoformat(e["date"].replace("Z", "+00:00"))
        for me, opp, home in ((e["home"], e["away"], 1), (e["away"], e["home"], 0)):
            tg = played[played.tid == me["id"]]
            recent = tg.drop_duplicates("gid").sort_values("date").gid.tail(5)
            cand = tg[tg.gid.isin(recent)].sort_values("date").groupby("pid").tail(1)
            for r in cand.itertuples():
                if last_team.get(r.pid) != me["id"]:
                    continue  # traded or moved since
                st = read_status(inj.get(r.pid), t, now)
                status[r.pid] = st
                if st in ("out", "doubtful"):
                    absent.setdefault(me["abbr"], []).append(r.name)
                    continue
                spread = e["spread"]
                rows.append({"gid": e["id"], "date": e["date"], "season": e["season"], "stype": e["stype"],
                             "team": me["abbr"], "tid": me["id"], "opp": opp["abbr"], "oid": opp["id"], "home": home,
                             "pid": r.pid, "name": r.name, "pos": r.pos, "starter": np.nan, "min": np.nan,
                             "total": e["total"], "spread": None if spread is None else (spread if home else -spread),
                             "future": 1})
    return pd.DataFrame(rows), status, absent


def flags_for(r, st, absent, inj):
    f = []
    if st == "questionable":
        f.append("listed as questionable" + (f' ({inj["what"].lower()})' if inj and inj.get("what") else ""))
    elif st == "probable":
        f.append("on the injury report but expected to play")
    out = absent.get(r.team, [])
    if r.new_min >= 18 and out:
        f.append("teammates out: " + ", ".join(out[:4]) + ", so more minutes and shots to go around")
    elif out:
        f.append("teammates out: " + ", ".join(out[:4]))
    if r.b2b == 1:
        f.append("second game in two nights")
    if pd.notna(r.abs_spread) and r.abs_spread >= 11:
        f.append(f'lopsided game expected ({r.team} {"favored" if r.spread < 0 else "underdog"} by {r.abs_spread:g}); starters could sit late')
    if pd.notna(r.min_sd) and r.min_sd >= 7:
        f.append("minutes have swung a lot lately")
    if r.n_prev < 8:
        f.append(f"only {int(r.n_prev)} games on record")
    if pd.notna(r.rest) and r.rest >= 7:
        f.append(f"first game back after {int(r.rest)} days off")
    return f


def fair_over(over, under):
    if over is None or under is None:
        return None
    a, b = M.implied(over), M.implied(under)
    return a / (a + b) if 1.01 < a + b < 1.15 else None


def board(pred, hist, props_by_game, status, absent, inj, bundle):
    """Players with projections, and every posted bet with its chance."""
    players, legs = [], []
    logs = hist[hist["min"] > 0].sort_values("date")
    logs = {pid: d.tail(10).iloc[::-1] for pid, d in logs.groupby("pid")}
    for r in pred.itertuples():
        if not (r.min_m >= 12) or r.n_prev < 3:
            continue
        main, alt = props_by_game.get(r.gid, ({}, {}))
        lg = logs.get(r.pid)
        st = status.get(r.pid, "")
        p = {"id": r.pid, "name": r.name, "team": r.team, "opp": r.opp, "home": int(r.home), "pos": r.posg, "gid": r.gid,
             "status": st, "note": (inj.get(r.pid) or {}).get("note", "") if st else "", "min": round(float(r.min_m), 1),
             "n_prev": int(r.n_prev), "flags": flags_for(r, st, absent, inj.get(r.pid)), "stats": {},
             "log": [{"d": str(x.date)[:10], "opp": ("vs " if x.home else "@ ") + x.opp, "min": int(x.min), "pts": int(x.pts),
                      "reb": int(x.reb), "ast": int(x.ast), "tpm": int(x.tpm)} for x in lg.itertuples()]}
        for t in F.TARGETS:
            s = bundle["stats"][t]
            loc, sc = float(getattr(r, f"loc_{t}")), float(getattr(r, f"sc_{t}"))
            m = main.get((r.pid, t))
            rungs = sorted(alt.get((r.pid, t), []), key=lambda a: a["n"])
            entry = {}
            ref = None  # (line, fair) the projection is tied to
            if m and m["over"] is not None:
                ref = (m["line"], fair_over(m["over"], m["under"]))
                entry.update(line=m["line"], over=m["over"], under=m["under"])
            elif rungs:
                near = min(rungs, key=lambda a: abs(M.implied(a["price"]) - 0.5))
                if 0.3 <= M.implied(near["price"]) <= 0.7:
                    ref = (near["n"] - 0.5, M.implied(near["price"]) / 1.07)
            if ref:
                pf = M.blend(M.p_over(loc, sc, s["cuts"], s["zq"], ref[0]), ref[1], bundle["blend"])
                loc2, tier = M.anchor(loc, sc, s["cuts"], s["zq"], ref[0], pf)
            else:
                loc2, tier = loc, M.tier_of(s["cuts"], loc)
            q = s["zq"][tier]
            chance = lambda line: M.p_over(loc2, sc, s["cuts"], s["zq"], line, tier)
            entry.update(proj=round(max(0.0, loc2 + sc * q[100]), 1), lo=round(max(0.0, loc2 + sc * q[30]), 1),
                         hi=round(max(0.0, loc2 + sc * q[170]), 1), model=round(loc, 1))
            def leg(kind, num, price, pr):
                be = M.implied(price)
                if pr - be > 0.2 or price < -2500:
                    return  # a gap that big means a mislabeled market, not a gift; and a price that short pays nothing
                legs.append({"id": f"{r.pid}-{t}-{kind[0]}{num:g}", "pid": r.pid, "gid": r.gid, "stat": t, "kind": kind, "num": num,
                             "price": int(price), "p": round(pr, 3)})
            if m and m["over"] is not None:
                po = chance(m["line"])
                entry["pOver"] = round(po, 3)
                leg("over", m["line"], m["over"], po)
                if m["under"] is not None:
                    leg("under", m["line"], m["under"], 1 - po)
            if rungs:
                for a in rungs:
                    leg("alt", a["n"], a["price"], chance(a["n"] - 0.5))
            p["stats"][t] = entry
        players.append(p)
    return players, legs


def describe(l, p):
    """A leg with the readable extras (the page works these out itself from the player's game log)."""
    t, kind, num = l["stat"], l["kind"], l["num"]
    vals = [sum(g[c] for c in espn.PARTS[t]) for g in p["log"]]
    hit = (lambda v: v >= num) if kind == "alt" else ((lambda v: v > num) if kind == "over" else (lambda v: v < num))
    return {**l, "dec": round(M.decimal(l["price"]), 3), "be": round(M.implied(l["price"]), 3), "unit": UNIT[t],
            "text": f"{num}+ {UNIT[t]}" if kind == "alt" else f"{kind} {num:g} {UNIT[t]}", "proj": p["stats"][t]["proj"],
            "hits": sum(hit(v) for v in vals), "of": len(vals), "last": vals[:8]}


def grade(log, hist):
    """Mark earlier picks as hit or missed once their games are in the box-score history."""
    have = set(hist.gid.unique())
    stat = hist.set_index(["gid", "pid"])
    for day, d in log["days"].items():
        if d.get("graded"):
            continue
        legs = d.get("safest", []) + d.get("payout", [])
        if not legs or not all(l["gid"] in have for l in legs):
            continue
        for l in legs:
            key = (l["gid"], l["pid"])
            if key not in stat.index or stat.loc[key, "min"] == 0:
                l["res"] = "void"
                continue
            v = sum(int(stat.loc[key, c]) for c in espn.PARTS[l["stat"]])
            l["got"] = v
            l["res"] = "hit" if (v >= l["num"] if l["kind"] == "alt" else v > l["num"] if l["kind"] == "over" else v < l["num"]) else "miss"
        d["graded"] = True


def record(log):
    """Running results of the published picks."""
    out = {"days": 0, "safest": {"hit": 0, "n": 0}, "payout": {"hit": 0, "n": 0}, "parlays": {}, "recent": []}
    for day in sorted(log["days"], reverse=True):
        d = log["days"][day]
        if not d.get("graded"):
            continue
        out["days"] += 1
        for style in ("safest", "payout"):
            live = [l for l in d.get(style, []) if l.get("res") in ("hit", "miss")]
            out[style]["hit"] += sum(l["res"] == "hit" for l in live); out[style]["n"] += len(live)
        s = [l for l in d.get("safest", []) if l.get("res") in ("hit", "miss")]
        for n in range(2, 7):
            if len(s) >= n:
                k = out["parlays"].setdefault(str(n), {"hit": 0, "n": 0})
                k["n"] += 1; k["hit"] += int(all(l["res"] == "hit" for l in s[:n]))
        if len(out["recent"]) < 10:
            out["recent"].append({"day": day, "by": d.get("by"), "legs": [{"name": l["name"], "text": l["text"], "res": l.get("res"), "got": l.get("got")}
                                                                          for l in d.get("safest", [])]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-ai", action="store_true")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "nba_parlay_data.json"))
    args = ap.parse_args()
    if args.no_ai:
        os.environ.pop("ANTHROPIC_API_KEY", None)
    now = dt.datetime.now(dt.timezone.utc)

    hist = dataset.load_games()
    new, _ = dataset.fetch_new(hist, with_lines=False)
    if len(new):
        hist = pd.concat([hist, new], ignore_index=True).drop_duplicates(["gid", "pid"])

    day, slate = find_slate(now)
    if not day:
        print("No upcoming games in the next three days. Nothing built.")
        sys.exit(3)
    pre = [e for e in slate if e["state"] == "pre"]
    print(f"slate {day}: {len(pre)} games not started, {len(slate) - len(pre)} under way or finished")

    inj = espn.injuries()
    fut, status, absent = expected_players(hist, slate, inj, now)
    feats = F.build(pd.concat([hist, fut], ignore_index=True))
    bundle = M.load_bundle()
    pred = M.predict(bundle, feats[feats.future == 1])

    def game_props(e):
        main, alt, book = espn.props(e["id"])
        a = {}
        for x in alt:
            a.setdefault((x["pid"], x["stat"]), []).append(x)
        return e["id"], ({(m["pid"], m["stat"]): m for m in main}, a), book
    props_by_game, book = {}, None
    with ThreadPoolExecutor(6) as ex:
        for gid, pr, bk in ex.map(game_props, pre):
            props_by_game[gid] = pr; book = book or bk

    players, legs = board(pred, hist, props_by_game, status, absent, inj, bundle)
    P = {p["id"]: p for p in players}
    full = [describe(l, P[l["pid"]]) for l in legs]  # with text, recent results and break-even, for the picker
    injuries = {}
    by_team = {t: a for e in pre for t, a in ((e["home"]["name"], e["home"]["abbr"]), (e["away"]["name"], e["away"]["abbr"]))}
    for pid, i in inj.items():
        ab = by_team.get(i["team"])
        if not ab:
            continue
        t = dt.datetime.fromisoformat(pre[0]["date"].replace("Z", "+00:00"))
        st = status.get(pid) or read_status(i, t, now)
        injuries.setdefault(ab, []).append({"name": i["name"], "status": st or "listed", "what": i.get("what", ""), "note": i.get("note", "")})
    for ab in injuries:
        injuries[ab].sort(key=lambda x: {"out": 0, "doubtful": 1, "questionable": 2}.get(x["status"], 3))

    data = {
        "generated": now.strftime("%Y-%m-%dT%H:%MZ"),
        "day": dt.datetime.strptime(day, "%Y%m%d").strftime("%A, %B %-d, %Y"), "date": f"{day[:4]}-{day[4:6]}-{day[6:]}",
        "book": book,
        "games": [{"id": e["id"], "away": e["away"]["abbr"], "home": e["home"]["abbr"], "awayName": e["away"]["name"],
                   "homeName": e["home"]["name"], "time": e["date"], "total": e["total"], "spread": e["spread"]} for e in pre],
        "injuries": injuries, "players": players,
        # every priced bet as [player id, stat, o/u/a (over, under, "or more"), number, American price, chance]
        "legs": [[l["pid"], l["stat"], l["kind"][0], l["num"], l["price"], l["p"]] for l in legs],
        "links": bundle["links"], "report": bundle["report"], "trained_on": bundle["trained_on"], "blend": bundle["blend"],
    }
    safe, pay = picker.pools(full, players)
    picks, text = picker.pick(data, safe, pay)
    data["picks"] = picks
    data["pool"] = {"safest": [l["id"] for l in safe], "payout": [l["id"] for l in pay]}
    data["prompt"] = {"rules": picker.rules(), "data": text, "model": picker.MODEL}
    print(f'{len(players)} players, {len(legs)} priced bets, picked by {picks["by"]}: '
          f'{len(picks["safest"])} safest, {len(picks["payout"])} payout')

    # results log: keep the picks as they stood before the first game of the day tipped off
    log = json.load(open(LOG)) if os.path.exists(LOG) else {"days": {}}
    grade(log, hist)
    L = {l["id"]: l for l in full}
    if len(pre) == len(slate) and picks["safest"]:
        keep = lambda items: [{**{k: L[i["id"]][k] for k in ("id", "pid", "gid", "stat", "kind", "num", "price", "p", "text")},
                               "name": P[L[i["id"]]["pid"]]["name"]} for i in items]
        log["days"][data["date"]] = {"by": picks["by"], "at": picks["at"], "safest": keep(picks["safest"]), "payout": keep(picks["payout"])}
    log["days"] = {k: log["days"][k] for k in sorted(log["days"])[-200:]}
    json.dump(log, open(LOG, "w"), separators=(",", ":"))
    data["record"] = record(log)

    json.dump(data, open(args.out, "w"), separators=(",", ":"))
    print("wrote", args.out, os.path.getsize(args.out) // 1024, "KB")


if __name__ == "__main__":
    main()
