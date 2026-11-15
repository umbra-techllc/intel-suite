"""
NFL TD Predictor weekly build.

Pulls nflverse weekly stats and play-by-play (this season and last), the week's schedule, game odds and weather
(ESPN), the injury report and depth charts (ESPN), and the touchdown bets DraftKings lists (from ESPN). Then
predicts every player's chance to score and every starting quarterback's touchdown passes, has the AI choose
and order the bets from the strongest candidates, grades earlier weeks' picks, and writes ../nfl_td_data.json,
which nfl_td_predictor.html loads.

Run:  python build_td.py            (current week, detected from ESPN)
      python build_td.py --week 7   (a specific week)
      python build_td.py --no-ai    (skip the AI step and rank by the model alone)
Needs: pandas, numpy, pyarrow, lightgbm, requests. Set ANTHROPIC_API_KEY for the AI step.
"""
import argparse, gzip, io, json, math, os, re, sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(os.path.dirname(HERE), "nfl-propedge"))  # shared: DraftKings lines and the AI pick step
import td_engine as E
import dk_lines as DK, picker

STATS_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.csv"
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?seasontype=2&week={week}"
ESPN_SCOREBOARD_NOW = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
ESPN_INJURIES = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"
ESPN_TEAMS = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams"
ESPN_DEPTH = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{id}/depthcharts"
ESPN_TO_NFLVERSE = {"WSH": "WAS", "LAR": "LA"}
OUT_STATUSES = {"Out", "Injured Reserve", "Doubtful", "Suspension", "Physically Unable to Perform"}
NOTHING_TO_BUILD = 3
MODEL_DIR = os.path.join(HERE, "model")

FACTOR_GROUPS = [
    ("Gets the ball near the end zone", ["e_rz_car", "e_rz_tgt", "e_g10_car", "e_g10_tgt", "e_g5_car", "e_g5_tgt", "e_ez_tgt", "e_rz_car_sh", "e_rz_tgt_sh", "e_g10_sh"]),
    ("Touchdowns so far this season", ["e_tds", "e_any_td", "e_rushing_tds", "season_tds"]),
    ("Overall role in the offense", ["e_car_sh", "e_tgt_sh", "e_carries", "e_targets", "e_ypg", "n_games"]),
    ("Last season", ["p_g", "p_tds", "p_any", "p_rz_car", "p_rz_tgt", "p_rz_car_sh", "p_rz_tgt_sh"]),
    ("Team's expected scoring this game", ["itt", "total_line", "spread_for", "home", "dome"]),
    ("Team's recent scoring", ["te_t_td", "te_t_pass_td_frac", "te_t_rz_pass_rate", "te_t_rz_plays", "te_t_g10", "te_t_att", "te_t_car"]),
    ("His quarterback's touchdown passing", ["qb_td_rate", "qb_td_pg", "qb_att_pg", "qb_rz_rate", "qb_n", "qb_p_td", "qb_p_att", "qb_l3"]),
    ("Opponent's defense", ["opp_allow_pos", "opp_allow_all"]),
    ("Injured teammates", ["vac_rz_car", "vac_rz_tgt"]),
    ("Position and week of season", ["pos_qb", "pos_rb", "pos_wr", "pos_te", "week"]),
]
QB_GROUPS = [
    ("His touchdown passing this season", ["e_passing_tds", "e_pass_td_rate", "qb_last3_td", "n_games"]),
    ("How he does near the end zone", ["e_qb_rz_att", "e_qb_rz_td_rate"]),
    ("How much he throws", ["e_attempts", "te_t_att", "te_t_car"]),
    ("Last season", ["p_pass_td", "p_att", "p_g"]),
    ("Team's expected scoring this game", ["itt", "total_line", "spread_for", "home", "dome"]),
    ("Team's recent scoring", ["te_t_td", "te_t_pass_td_frac", "te_t_rz_pass_rate", "te_t_rz_plays"]),
    ("Opponent's defense", ["o_allow_WR", "o_allow_TE", "o_allow_RB", "opp_allow_all"]),
    ("Week of season", ["week"]),
]


def get_json(url):
    return DK.get_json(url)  # with retries


def cached(url, path):
    if not os.path.exists(path):
        r = requests.get(url, timeout=180); r.raise_for_status()
        open(path, "wb").write(r.content)
    return path


def norm_name(s):
    s = re.sub(r"[.'’]", "", str(s)).lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def parse_slate(sb):
    games = []
    for e in sb.get("events", []):
        c = e["competitions"][0]
        if c.get("status", {}).get("type", {}).get("state", "pre") != "pre":
            continue  # under way or finished
        teams = {t["homeAway"]: ESPN_TO_NFLVERSE.get(t["team"]["abbreviation"], t["team"]["abbreviation"]) for t in c["competitors"]}
        odds = (c.get("odds") or [{}])[0]
        total, spread = odds.get("overUnder"), odds.get("spread")
        if total is None or spread is None:
            continue
        w = e.get("weather") or c.get("weather") or {}
        indoor = bool(c.get("venue", {}).get("indoor"))
        games.append({"id": e["id"], "kickoff": e["date"], "home": teams["home"], "away": teams["away"], "total": float(total),
                      "homeSpread": float(spread), "homeITT": round(total / 2 - float(spread) / 2, 1), "awayITT": round(total / 2 + float(spread) / 2, 1),
                      "indoor": indoor, "weather": "Dome" if indoor else (w.get("displayValue") or "").strip(), "temp": w.get("temperature")})
    return games


def booster(name):
    import lightgbm as lgb
    with gzip.open(os.path.join(MODEL_DIR, name + ".txt.gz"), "rt") as fh:
        return lgb.Booster(model_str=fh.read())


def group_effects(contrib, idx, groups, total, link):
    """Change in the prediction from each group of inputs (prediction with vs without that group)."""
    out = []
    for label, fs in groups:
        c = float(sum(contrib[idx[f]] for f in fs if f in idx))
        delta = link(total) - link(total - c)
        out.append({"label": label, "delta": round(delta, 4)})
    out.sort(key=lambda g: -abs(g["delta"]))
    return [g for g in out if abs(g["delta"]) >= 0.005][:5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--week", type=int)
    ap.add_argument("--no-ai", action="store_true")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "nfl_td_data.json"))
    args = ap.parse_args()
    if args.no_ai:
        os.environ.pop("ANTHROPIC_API_KEY", None)
    season, week = args.season, args.week
    auto = week is None
    if auto:
        now = get_json(ESPN_SCOREBOARD_NOW)
        if now.get("season", {}).get("type") != 2:
            print("ESPN is not showing a regular-season week right now. Nothing built."); sys.exit(NOTHING_TO_BUILD)
        week = int(now["week"]["number"])
    sb = get_json(ESPN_SCOREBOARD.format(week=week))
    games = parse_slate(sb)
    if not games and auto:
        week += 1; sb = get_json(ESPN_SCOREBOARD.format(week=week)); games = parse_slate(sb)
    if not games:
        print(f"No upcoming games with odds posted for week {week}. Nothing built."); sys.exit(NOTHING_TO_BUILD)
    started = len(sb.get("events", [])) - len(games)  # games of this week already under way or finished
    print(f"Week {week}: {len(games)} games")

    cache = os.path.join(HERE, "research", "cache"); os.makedirs(cache, exist_ok=True)
    for f in (f"ps{season}.csv", f"pbp_{season}.parquet"):  # always fresh for the current season
        p = os.path.join(cache, f)
        if os.path.exists(p):
            os.remove(p)
    stats = pd.concat([pd.read_csv(cached(STATS_URL.format(season=y), os.path.join(cache, f"ps{y}.csv")), low_memory=False) for y in (season - 1, season)])
    pbp = pd.concat([pd.read_parquet(cached(PBP_URL.format(season=y), os.path.join(cache, f"pbp_{y}.parquet")), columns=E.PBP_COLS) for y in (season - 1, season)])
    stats = stats[stats.season_type == "REG"]
    stats = stats[(stats.season < season) | (stats.week < week)]
    pbp = pbp[(pbp.season < season) | (pbp.week < week)]

    # schedule: past games from nflverse results, this week's lines from ESPN
    sched = pd.read_csv(cached("https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv", os.path.join(cache, "games.csv")))
    sched = sched[sched.season.isin([season - 1, season])]
    gid = {}
    up_sched = []
    for g in games:
        game_id = f"{season}_{week:02d}_{g['away']}_{g['home']}"
        gid[g["id"]] = game_id
        up_sched.append({"game_id": game_id, "season": season, "week": week, "game_type": "REG", "home_team": g["home"], "away_team": g["away"],
                         "spread_line": -g["homeSpread"], "total_line": g["total"], "roof": "dome" if g["indoor"] else "outdoors"})
    sched = pd.concat([sched[~sched.game_id.isin(gid.values())], pd.DataFrame(up_sched)], ignore_index=True)

    base = E.build_base(stats, pbp, sched)
    base["result_known"] = 1
    st = base[base.position == "QB"].sort_values("attempts", ascending=False).drop_duplicates(["game_id", "team"])
    base["is_starter"] = base.index.isin(st.index).astype(int)

    # injuries and depth charts
    inj = get_json(ESPN_INJURIES)
    status, note = {}, {}
    for t in inj.get("injuries", []):
        for x in t.get("injuries", []):
            a = x.get("athlete", {}); ab = ESPN_TO_NFLVERSE.get(a.get("team", {}).get("abbreviation", ""), a.get("team", {}).get("abbreviation", ""))
            status[(ab, norm_name(a.get("displayName", "")))] = x.get("status", ""); note[(ab, norm_name(a.get("displayName", "")))] = x.get("shortComment", "")
    depth = {}
    try:
        teams = get_json(ESPN_TEAMS)["sports"][0]["leagues"][0]["teams"]
        ids = {ESPN_TO_NFLVERSE.get(t["team"]["abbreviation"], t["team"]["abbreviation"]): t["team"]["id"] for t in teams}
        for team in sorted({g["home"] for g in games} | {g["away"] for g in games}):
            try:
                for chart in get_json(ESPN_DEPTH.format(id=ids[team])).get("depthchart", []):
                    for key, slot in chart.get("positions", {}).items():
                        if key == "qb":
                            depth.setdefault(team, [a.get("displayName", "") for a in slot.get("athletes", [])])
            except Exception:
                pass
    except Exception as e:
        print("Depth charts unavailable:", e)

    # this week's candidates: players from the team's last two games who aren't ruled out, plus the depth-chart starting QB
    cur = base[base.season == season]
    up_rows, out_list, q_list = [], {}, {}
    slate = {}
    for g in games:
        slate[g["home"]] = (g, g["away"]); slate[g["away"]] = (g, g["home"])
    last_wk = cur.groupby("team").week.max().to_dict()
    for team, (g, opp) in slate.items():
        tw = sorted(cur[cur.team == team].week.unique())[-2:]
        recent = cur[(cur.team == team) & cur.week.isin(tw)].sort_values("week").drop_duplicates("player_id", keep="last")
        starter_name = None
        for nm in depth.get(team, []):
            if status.get((team, norm_name(nm)), "") in OUT_STATUSES:
                continue
            starter_name = nm; break
        if starter_name is None:
            qbs = recent[(recent.position == "QB") & recent.player_display_name.map(lambda n: status.get((team, norm_name(n)), "") not in OUT_STATUSES)]
            if len(qbs):
                starter_name = qbs.sort_values(["week", "attempts"], ascending=False).player_display_name.iloc[0]
        for r in recent.itertuples():
            stt = status.get((team, norm_name(r.player_display_name)), "")
            if stt in OUT_STATUSES:
                if (r.position == "QB" and r.is_starter) or (r.position != "QB" and (r.carries or 0) + (r.targets or 0) >= 3):
                    out_list.setdefault(team, []).append({"name": r.player_display_name, "pos": r.position, "status": stt})
                continue
            if r.position == "QB" and starter_name and norm_name(r.player_display_name) != norm_name(starter_name):
                continue
            if stt == "Questionable":
                q_list.setdefault(team, []).append({"name": r.player_display_name, "pos": r.position})
            up_rows.append({"player_id": r.player_id, "player_display_name": r.player_display_name, "position": r.position, "team": team,
                            "opponent_team": opp, "season": season, "week": week, "game_id": gid[g["id"]], "season_type": "REG",
                            "result_known": 0, "is_starter": int(r.position == "QB"), "status": stt, "note": note.get((team, norm_name(r.player_display_name)), "") if stt else "",
                            "headshot_url": getattr(r, "headshot_url", "")})
        if starter_name and not any(u["team"] == team and u["position"] == "QB" for u in up_rows):
            prev = base[(base.position == "QB") & (base.player_display_name.map(norm_name) == norm_name(starter_name))]
            pid = prev.player_id.iloc[-1] if len(prev) else "dc-" + norm_name(starter_name)
            up_rows.append({"player_id": pid, "player_display_name": starter_name, "position": "QB", "team": team, "opponent_team": opp,
                            "season": season, "week": week, "game_id": gid[g["id"]], "season_type": "REG", "result_known": 0, "is_starter": 1,
                            "status": status.get((team, norm_name(starter_name)), ""), "note": "", "headshot_url": ""})
    up = E.add_games(pd.DataFrame(up_rows), sched)
    allrows = pd.concat([base.assign(status="", note=""), up], ignore_index=True)
    d = E.features(allrows)
    d = d[d.result_known == 0].reset_index(drop=True)
    d.to_parquet(os.path.join(cache, "live_features.parquet"))  # kept for checking the inputs

    bundle = json.load(open(os.path.join(MODEL_DIR, "bundle.json")))
    F, QF = bundle["features"], bundle["qb_features"]
    cx, cy = np.array(bundle["calibration"]["x"]), np.array(bundle["calibration"]["y"])
    calib = lambda p: np.interp(p, cx, cy)
    sig = lambda z: 1 / (1 + math.exp(-z))

    m_any = booster("anytime")
    X = d[F].astype(float).values
    raw = m_any.predict(X)
    contrib = m_any.predict(X, pred_contrib=True)
    p_any = calib(raw)
    idx = {f: i for i, f in enumerate(F)}
    m_qb = booster("qb_pass_td")
    qbmask = (d.position == "QB").values
    lam = np.full(len(d), np.nan); qcontrib = None
    if qbmask.any():
        XQ = d.loc[qbmask, QF].astype(float).values
        lam[qbmask] = m_qb.predict(XQ)
        qcontrib = m_qb.predict(XQ, pred_contrib=True)
    qidx = {f: i for i, f in enumerate(QF)}

    players = []
    qi = 0
    for i, r in d.iterrows():
        g, opp = slate[r.team]
        p = float(p_any[i])
        lam_any = -math.log(max(1e-6, 1 - p))
        rz_c, rz_t = float(np.nan_to_num(r.e_rz_car)), float(np.nan_to_num(r.e_rz_tgt))
        run_share = rz_c / (rz_c + rz_t) if rz_c + rz_t > 0 else (1.0 if r.position == "RB" else 0.0 if r.position in ("WR", "TE") else 0.7)
        total = float(contrib[i].sum())
        cal_ratio = p / max(1e-6, float(raw[i]))
        effects = group_effects(contrib[i], idx, FACTOR_GROUPS, total, lambda z: min(0.99, sig(z) * cal_ratio))
        hist = cur[cur.player_id == r.player_id].sort_values("week")
        logs = [{"wk": int(h.week), "opp": h.opponent_team, "td": int((h.rushing_tds or 0) + (h.receiving_tds or 0)), "rtd": int(h.rushing_tds or 0),
                 "ctd": int(h.receiving_tds or 0), "ptd": int(h.passing_tds or 0), "rz_car": int(h.rz_car), "rz_tgt": int(h.rz_tgt),
                 "g10": int(h.g10_car + h.g10_tgt), "ez": int(h.ez_tgt), "att": int(h.attempts or 0)} for h in hist.itertuples()]
        row = {"id": f"{r.player_id}-{g['id']}", "pid": r.player_id, "name": r.player_display_name, "pos": r.position, "team": r.team, "opp": opp,
               "home": bool(r.home), "game": g["id"], "status": r.status or "", "note": r.note or "", "n": int(r.n_games),
               "p": round(p, 4), "p2": round(1 - math.exp(-lam_any) * (1 + lam_any), 4), "runShare": round(run_share, 3),
               "role": {"rz_car": round(rz_c, 2), "rz_tgt": round(rz_t, 2), "g10": round(float(np.nan_to_num(r.e_g10_car) + np.nan_to_num(r.e_g10_tgt)), 2),
                        "ez": round(float(np.nan_to_num(r.e_ez_tgt)), 2), "rz_car_sh": round(float(np.nan_to_num(r.e_rz_car_sh)), 3),
                        "rz_tgt_sh": round(float(np.nan_to_num(r.e_rz_tgt_sh)), 3), "season_tds": int(r.season_tds or 0),
                        "prev_tds_pg": None if pd.isna(r.p_tds) else round(float(r.p_tds), 2)},
               "vac": {"car": round(float(r.vac_rz_car), 2), "tgt": round(float(r.vac_rz_tgt), 2)},
               "itt": round(float(r.itt), 1), "effects": effects, "logs": logs}
        if r.position == "QB" and qbmask[i]:
            L = float(lam[i])
            qtot = float(qcontrib[qi].sum())
            qe = group_effects(qcontrib[qi], qidx, QB_GROUPS, qtot, lambda z: math.exp(z))
            qi += 1
            pk = [math.exp(-L) * L ** k / math.factorial(k) for k in range(6)]
            row["qb"] = {"lam": round(L, 3), "p1": round(1 - pk[0], 4), "p2": round(1 - pk[0] - pk[1], 4), "p3": round(1 - sum(pk[:3]), 4),
                         "last3": None if pd.isna(r.qb_last3_td) else round(float(r.qb_last3_td), 2),
                         "season_pg": None if pd.isna(r.e_passing_tds) else round(float(r.e_passing_tds), 2),
                         "rz_rate": None if pd.isna(r.e_qb_rz_td_rate) else round(float(r.e_qb_rz_td_rate), 3),
                         "team_rz_pass": None if pd.isna(r.te_t_rz_pass_rate) else round(float(r.te_t_rz_pass_rate), 3),
                         "prev_pg": None if pd.isna(r.p_pass_td) else round(float(r.p_pass_td), 2), "effects": qe}
        players.append(row)
    # team expected touchdowns (sum of each player's expected touchdowns)
    for g in games:
        for side in ("home", "away"):
            t = g[side]
            g[side + "XTD"] = round(sum(-math.log(max(1e-6, 1 - pl["p"])) for pl in players if pl["team"] == t and pl["pos"] != "QB") +
                                    sum(-math.log(max(1e-6, 1 - pl["p"])) for pl in players if pl["team"] == t and pl["pos"] == "QB"), 2)
    # ---------- the touchdown bets DraftKings lists (from ESPN) ----------
    listed, book = {}, None
    try:
        from concurrent.futures import ThreadPoolExecutor
        to_pid = DK.id_map(os.path.join(cache, "player_ids.csv"))
        with ThreadPoolExecutor(5) as ex:
            for g, gl in zip(games, ex.map(lambda g: DK.game_lines(g["id"]), games)):
                book = book or gl["book"]
                listed[g["id"]] = {"td1": {to_pid.get(a) for a in gl["td1"]}, "td2": {to_pid.get(a) for a in gl["td2"]},
                                   "ptd": {to_pid.get(a): v for (a, k), v in gl["main"].items() if k == "ptd"},
                                   "steps": {to_pid.get(a): v for (a, k), v in gl["steps"].items() if k == "ptd"}}
    except Exception as e:
        print("Could not read the DraftKings lists:", e)

    # every bet as a dict; "td" = scores n or more touchdowns, "ptd" = throws n or more (or over/under the line)
    legs = []
    def leg(pl, stat, kind, num, chance):
        if 0.03 <= chance <= 0.97:
            legs.append({"id": f'{pl["pid"]}-{stat}-{kind[0]}{num:g}', "pid": pl["pid"], "gid": pl["game"], "stat": stat, "kind": kind,
                         "num": num, "p": round(float(chance), 3)})
    pois = lambda L, n: 1 - sum(math.exp(-L) * L ** k / math.factorial(k) for k in range(int(n)))
    for pl in players:
        ls = listed.get(pl["game"]) or {"td1": set(), "td2": set(), "ptd": {}, "steps": {}}
        if pl["pos"] != "QB" or pl["p"] >= 0.12:
            if not ls["td1"] or pl["pid"] in ls["td1"]:
                leg(pl, "td", "alt", 1, pl["p"])
            if (not ls["td2"] or pl["pid"] in ls["td2"]) and pl["p2"] >= 0.08:
                leg(pl, "td", "alt", 2, pl["p2"])
        if pl.get("qb"):
            L = pl["qb"]["lam"]
            main = ls["ptd"].get(pl["pid"])
            if main:
                pl["qb"]["line"] = main["line"]
                po = pois(L, math.floor(main["line"]) + 1)
                leg(pl, "ptd", "over", main["line"], po); leg(pl, "ptd", "under", main["line"], 1 - po)
            for n in (ls["steps"].get(pl["pid"]) or [1, 2, 3]):
                if not (main and abs(n - 0.5 - main["line"]) < 0.01):
                    leg(pl, "ptd", "alt", int(n), pois(L, n))
    P = {pl["pid"]: pl for pl in players}
    def describe(l):
        pl = P[l["pid"]]
        vals = [g["ptd" if l["stat"] == "ptd" else "td"] for g in reversed(pl["logs"])]
        num, kind = l["num"], l["kind"]
        hit = (lambda v: v >= num) if kind == "alt" else ((lambda v: v > num) if kind == "over" else (lambda v: v < num))
        what = "touchdown passes" if l["stat"] == "ptd" else "touchdowns"
        if kind != "alt":
            text = f"{kind} {num:g} touchdown passes"
        elif l["stat"] == "ptd":
            text = "throws a touchdown pass" if num == 1 else f"throws {num}+ touchdown passes"
        else:
            text = "scores a touchdown" if num == 1 else f"scores {num}+ touchdowns"
        return {**l, "text": text, "unit": what, "hits": sum(hit(v) for v in vals), "of": len(vals), "last": vals[:8]}
    full = [describe(l) for l in legs]

    # candidates for the AI: most likely bets, and longer shots that pay more
    safe = sorted([l for l in full if l["kind"] == "alt" and ((l["stat"] == "td" and l["num"] == 1 and l["p"] >= 0.38) or
                                                              (l["stat"] == "ptd" and 0.5 <= l["p"] <= 0.93))], key=lambda l: -l["p"])
    best = {}
    for l in safe:  # one bet per player: for a quarterback, the hardest touchdown-pass bet that is still better than 60%
        if l["stat"] == "ptd" and l["pid"] in best and not (l["p"] >= 0.6 and l["num"] > best[l["pid"]]["num"]):
            continue
        if l["stat"] == "td" and l["pid"] in best:
            continue
        best[l["pid"]] = l
    # players scoring come first; a quarterback's touchdown-pass bet has to be much more likely to rank alongside them
    safe = sorted(best.values(), key=lambda l: -(l["p"] - (0.3 if l["stat"] == "ptd" else 0)))[:36]
    pay = sorted([l for l in full if l["id"] not in {x["id"] for x in safe} and l["stat"] == "td" and
                  ((l["num"] == 1 and 0.22 <= l["p"] < 0.38) or (l["num"] == 2 and l["p"] >= 0.12))], key=lambda l: -l["p"])[:24]

    def row(l):
        pl = P[l["pid"]]
        r = pl["role"]
        extra = (f'expected touchdown passes {pl["qb"]["lam"]:.1f}, last three games {pl["qb"]["last3"]}' if l["stat"] == "ptd" else
                 f'per game: {r["rz_car"]:g} runs and {r["rz_tgt"]:g} passes thrown to him inside the 20, {r["g10"]:g} chances inside the 10; '
                 f'{r["season_tds"]} touchdowns this season')
        notes = [e["label"] + (" (helps)" if e["delta"] > 0 else " (hurts)") for e in (pl["qb"]["effects"] if l["stat"] == "ptd" else pl["effects"])[:3]]
        return (f'{l["id"]} | {pl["name"]}, {pl["pos"]} ({pl["team"]} {"vs" if pl["home"] else "at"} {pl["opp"]}) | {l["text"]} | chance {round(l["p"] * 100)}% '
                f'| his team is expected to score about {pl["itt"]:g} points | {extra} | did this in {l["hits"]} of {l["of"]} games this season '
                f'| {l["unit"]} by game, newest first: {", ".join(str(v) for v in l["last"])} | biggest factors: {"; ".join(notes)}'
                + (f' | listed as {pl["status"].lower()}' if pl["status"] and pl["status"] != "Active" else "")
                + (f' | injury note: {pl["note"]}' if pl["note"] and pl["status"] != "Active" else ""))
    text = [f"NFL week {week}, {season} season. Touchdown bets listed by {book or 'the sportsbook'}.", "", "GAMES"]
    for g in games:
        fav = g["home"] if g["homeSpread"] < 0 else g["away"]
        text.append(f'- {g["away"]} at {g["home"]}: {fav} favored by {abs(g["homeSpread"]):g}, expected points {g["away"]} {g["awayITT"]:g}, {g["home"]} {g["homeITT"]:g}, '
                    f'{"indoors" if g["indoor"] else "weather: " + (g["weather"] or "not posted")}')
        for t in (g["away"], g["home"]):
            if out_list.get(t):
                text.append(f"    {t} out: " + "; ".join(f'{x["name"]} ({x["pos"]}, {x["status"].lower()})' for x in out_list[t]))
            if q_list.get(t):
                text.append(f"    {t} questionable: " + "; ".join(f'{x["name"]} ({x["pos"]})' for x in q_list[t]))
    text += ["", "CANDIDATES FOR THE SAFEST LIST"] + [row(l) for l in safe] + ["", "CANDIDATES FOR THE PAYOUT LIST"] + [row(l) for l in pay]
    text = "\n".join(text)

    def fallback(l):
        pl = P[l["pid"]]
        r = pl["role"]
        why = (f'Expected to throw about {pl["qb"]["lam"]:.1f} touchdown passes and did this in {l["hits"]} of his {l["of"]} games this season.' if l["stat"] == "ptd" else
               f'Gets about {r["rz_car"] + r["rz_tgt"]:.1f} chances a game inside the 20 and did this in {l["hits"]} of his {l["of"]} games this season.')
        return why, f'His team is expected to score about {pl["itt"]:g} points.', pl["status"] in ("", "Active") and pl["n"] >= 3
    picks = picker.pick("td", text, safe, pay, fallback)
    print(f'{len(legs)} bets, picked by {picks["by"]}: {len(picks["safest"])} safest, {len(picks["payout"])} payout')

    # ---------- results log: the picks as they stood before the week's first kickoff, graded the week after ----------
    log_path = os.path.join(HERE, "data", "picks_log.json")
    log = json.load(open(log_path)) if os.path.exists(log_path) else {"weeks": {}}
    for d_ in log["weeks"].values():
        if d_.get("graded") or d_.get("season") != season or d_["week"] >= week:
            continue
        got = stats[(stats.season == season) & (stats.week == d_["week"])].drop_duplicates("player_id").set_index("player_id")
        if got.empty:
            continue
        for l in d_.get("safest", []) + d_.get("payout", []):
            if l["pid"] not in got.index:
                l["res"] = "void"; continue
            r = got.loc[l["pid"]]
            v = int(r.passing_tds or 0) if l["stat"] == "ptd" else int((r.rushing_tds or 0) + (r.receiving_tds or 0))
            l["got"] = v
            l["res"] = "hit" if (v >= l["num"] if l["kind"] == "alt" else v > l["num"] if l["kind"] == "over" else v < l["num"]) else "miss"
        d_["graded"] = True
    key = f"{season}-W{week:02d}"
    Lf = {l["id"]: l for l in full}
    if picks["safest"] and (started == 0 or key not in log["weeks"]):
        keep = lambda items: [{**{k: Lf[i["id"]][k] for k in ("id", "pid", "gid", "stat", "kind", "num", "p", "text")}, "name": P[Lf[i["id"]]["pid"]]["name"]} for i in items]
        log["weeks"][key] = {"season": season, "week": week, "by": picks["by"], "at": picks["at"], "safest": keep(picks["safest"]), "payout": keep(picks["payout"])}
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    json.dump(log, open(log_path, "w"), separators=(",", ":"))
    rec = {"weeks": 0, "safest": {"hit": 0, "n": 0}, "payout": {"hit": 0, "n": 0}, "recent": []}
    for k in sorted(log["weeks"], reverse=True):
        d_ = log["weeks"][k]
        if not d_.get("graded"):
            continue
        rec["weeks"] += 1
        for st_ in ("safest", "payout"):
            live = [l for l in d_.get(st_, []) if l.get("res") in ("hit", "miss")]
            rec[st_]["hit"] += sum(l["res"] == "hit" for l in live); rec[st_]["n"] += len(live)
        if len(rec["recent"]) < 8:
            rec["recent"].append({"week": d_["week"], "by": d_.get("by"), "legs": [{"name": l["name"], "text": l["text"], "res": l.get("res"), "got": l.get("got"), "style": st_}
                                                                                 for st_ in ("safest", "payout") for l in d_.get(st_, [])]})

    out = {"season": season, "week": week, "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "book": book,
           "statsThrough": int(cur.week.max()) if len(cur) else 0, "games": games, "players": players,
           "injuries": {"out": out_list, "questionable": q_list}, "corr": bundle["correlations"],
           "modelInfo": bundle["trained_on"], "baseRate": bundle["base_rate"],
           # every bet as [player id, td or ptd, o/u/a (over, under, "or more"), number, chance]
           "legs": [[l["pid"], l["stat"], l["kind"][0], l["num"], l["p"]] for l in legs],
           "picks": picks, "pool": {"safest": [l["id"] for l in safe], "payout": [l["id"] for l in pay]},
           "prompt": {"rules": picker.rules("td"), "data": text, "model": picker.MODEL}, "record": rec}
    json.dump(out, open(args.out, "w"), separators=(",", ":"))
    print(f"{len(games)} games, {len(players)} players ({int(qbmask.sum())} QBs) -> {args.out}", os.path.getsize(args.out) // 1024, "KB")


if __name__ == "__main__":
    main()
