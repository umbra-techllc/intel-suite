"""Turns the week's projections and DraftKings lines into the board the page shows:
one entry per player, and one entry per posted bet with its chance of hitting.
"""
import json, os
import numpy as np
import engine as E

HERE = os.path.dirname(os.path.abspath(__file__))
UNIT = {"pass": "passing yards", "rush": "rushing yards", "rec": "receiving yards"}
# How two bets from the same game move together, measured on 2024-25 games (both written as "more yards").
# Pairs that measured near zero (opposing receivers, a back and his own receivers, two backs) are left out.
LINKS = {"qb_receiver": 0.30, "qb_back_catching": 0.23, "qb_own_runner": -0.12, "same_player_run_catch": 0.10, "two_qbs": 0.15}


def load_blend():
    path = os.path.join(HERE, "model", "blend.json")
    return json.load(open(path)) if os.path.exists(path) else {"weights": {}, "report": None}


def flags_for(p, g, line):
    """Short plain-English notes about anything that makes this player's week unusual."""
    f = []
    d = p["drivers"]
    if p["status"] == "Questionable":
        f.append("listed as questionable")
    elif p["status"] and p["status"] != "Active":
        f.append(f'on the injury report ({p["status"].lower()})')
    role = p.get("role") or {}
    if role.get("from"):
        f.append("bigger role with " + ", ".join(x["name"] for x in role["from"][:3]) + " out")
    if role.get("firstStart"):
        f.append("first start of the season")
    elif role.get("replacing"):
        f.append(f'starting in place of {role["replacing"]}')
    if p.get("qbChange") and p["kind"] == "rec":
        f.append("his quarterback is making his first start of the season" if p["qbChange"] == "first start"
                 else f'his usual quarterback, {p["qbChange"]}, is out')
    if d.get("wet"):
        f.append("rain or snow expected")
    sp = d.get("spreadFor") or 0
    if abs(sp) >= 9.5:
        f.append(f'lopsided game expected ({p["team"]} {"favored" if sp > 0 else "underdog"} by {abs(sp):g})')
    if p["n"] < 3:
        f.append(f'only {p["n"]} game{"" if p["n"] == 1 else "s"} this season')
    if line and line.get("open") is not None and abs(line["line"] - line["open"]) >= max(3.0, 0.08 * line["open"]):
        f.append(f'his {UNIT[p["kind"]]} line has moved from {line["open"]:g} to {line["line"]:g} since it opened')
    return f


def build(props, games, lines_by_game, blend, espn_id=None):
    """props: output of build_data.project + apply_models (each still carrying its outcome grid in "_grid").
    lines_by_game: {game id: {"main": {(pid, kind): {line, open}}, "steps": {(pid, kind): [n]}}} keyed by nflverse player id.
    Returns (players, legs)."""
    G = {g["id"]: g for g in games}
    W = blend.get("weights") or {}
    players, legs = {}, []
    for p in props:
        raw = p.get("_grid")
        if raw is None:
            continue
        gl = lines_by_game.get(p["game"]) or {"main": {}, "steps": {}}
        key = (p["pid"], p["kind"])
        line = gl["main"].get(key)
        steps = gl["steps"].get(key, [])
        w = W.get(p["kind"])
        if line and w:
            pm = E.p_over(raw, line["line"])
            final = E.anchor(raw, line["line"], E.blend(pm, w), w.get("scale", 1.0))
        else:
            pm, final = None, raw
        pl = players.setdefault(p["pid"], {
            "id": p["pid"], "name": p["name"], "team": p["team"], "opp": p["opp"], "home": bool(p["home"]), "pos": p["pos"],
            "gid": p["game"], "eid": (espn_id or {}).get(p["pid"]), "status": "" if p["status"] == "Active" else p["status"], "note": "" if p["status"] == "Active" else p["note"], "n": p["n"], "flags": [], "stats": {},
            "log": [{k: l[k] for k in ("wk", "opp", "pass", "att", "rush", "car", "rec", "tgt")} for l in reversed(p["logs"])]})
        for f in flags_for(p, G[p["game"]], line):
            if f not in pl["flags"]:
                pl["flags"].append(f)
        entry = {"proj": round(float(final[50]), 1), "lo": round(float(max(0, final[20])), 1), "hi": round(float(final[80]), 1),
                 "model": round(float(raw[50]), 1), "factors": (p.get("factors") or {}).get("top", [])[:4],
                 # the model's own range of outcomes, so the page can redo the chances when it loads newer lines
                 "g": [round(float(v), 1) for v in raw]}
        def leg(kind, num, chance, model_chance):
            if 0.03 <= chance <= 0.97:
                legs.append({"id": f'{p["pid"]}-{p["kind"]}-{kind[0]}{num:g}', "pid": p["pid"], "gid": p["game"], "stat": p["kind"],
                             "kind": kind, "num": num, "p": round(chance, 3), "pm": round(model_chance, 3)})
        if line:
            po = E.p_over(final, line["line"])
            entry.update(line=line["line"], open=line["open"], pOver=round(po, 3))
            if w:
                leg("over", line["line"], po, pm)
                leg("under", line["line"], 1 - po, 1 - pm)
        if w and line:
            for n in steps:
                if abs(n - 0.5 - line["line"]) < 0.01:
                    continue  # the same bet as the over on the main line
                leg("alt", n, E.p_over(final, n - 0.5), E.p_over(raw, n - 0.5))
        pl["stats"][p["kind"]] = entry
    return list(players.values()), legs


def describe(l, P):
    """A bet with the readable extras: its wording, the projection, and how often it happened this season."""
    p = P[l["pid"]]
    vals = [g[l["stat"]] for g in p["log"]]
    num, kind = l["num"], l["kind"]
    hit = (lambda v: v >= num) if kind == "alt" else ((lambda v: v > num) if kind == "over" else (lambda v: v < num))
    s = p["stats"][l["stat"]]
    return {**l, "unit": UNIT[l["stat"]], "text": f'{num:g}+ {UNIT[l["stat"]]}' if kind == "alt" else f'{kind} {num:g} {UNIT[l["stat"]]}',
            "proj": s["proj"], "model": s["model"], "line": s.get("line"), "hits": sum(hit(v) for v in vals), "of": len(vals), "last": vals[:8]}


def grade(log, stats, season, cur_week):
    """Mark saved picks as hit or missed once that week is over. stats: nflverse weekly rows for the season."""
    col = {"pass": "passing_yards", "rush": "rushing_yards", "rec": "receiving_yards"}
    for key, d in log["weeks"].items():
        if d.get("graded") or d.get("season") != season or d["week"] >= cur_week:
            continue  # graded from the following week's build on, when every game of that week is in
        wk = stats[stats.week == d["week"]]
        if wk.empty:
            continue
        got = wk.set_index("player_id")
        for l in d.get("safest", []) + d.get("payout", []):
            if l["pid"] not in got.index:
                l["res"] = "void"
                continue
            row = got.loc[l["pid"]]
            v = float(row[col[l["stat"]]].iloc[0] if hasattr(row[col[l["stat"]]], "iloc") else row[col[l["stat"]]])
            l["got"] = int(v)
            l["res"] = "hit" if (v >= l["num"] if l["kind"] == "alt" else v > l["num"] if l["kind"] == "over" else v < l["num"]) else "miss"
        d["graded"] = True


def record(log):
    """Running results of the published picks."""
    out = {"weeks": 0, "safest": {"hit": 0, "n": 0}, "payout": {"hit": 0, "n": 0}, "parlays": {}, "recent": []}
    for key in sorted(log["weeks"], reverse=True):
        d = log["weeks"][key]
        if not d.get("graded"):
            continue
        out["weeks"] += 1
        for style in ("safest", "payout"):
            live = [l for l in d.get(style, []) if l.get("res") in ("hit", "miss")]
            out[style]["hit"] += sum(l["res"] == "hit" for l in live); out[style]["n"] += len(live)
        s = [l for l in d.get("safest", []) if l.get("res") in ("hit", "miss")]
        for n in range(2, 7):
            if len(s) >= n:
                k = out["parlays"].setdefault(str(n), {"hit": 0, "n": 0})
                k["n"] += 1; k["hit"] += int(all(l["res"] == "hit" for l in s[:n]))
        if len(out["recent"]) < 8:
            out["recent"].append({"week": d["week"], "by": d.get("by"), "legs": [
                {"name": l["name"], "text": l["text"], "res": l.get("res"), "got": l.get("got"), "style": st}
                for st in ("safest", "payout") for l in d.get(st, [])]})
    return out
