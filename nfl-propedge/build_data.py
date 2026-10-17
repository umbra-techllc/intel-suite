"""
NFL PropEdge data build.

Pulls weekly player stats (nflverse), the current week's schedule, odds and
weather (ESPN scoreboard), and the injury report (ESPN), then writes
projections for passing, rushing and receiving yard props to data.json.

Run:  python build_data.py            (current week, detected from ESPN)
      python build_data.py --week 5   (a specific week)
Needs: pandas, scipy, requests
"""
import argparse, json, math, re, sys, io
from datetime import datetime, timezone

import pandas as pd
import requests
from scipy.stats import gamma

NFLVERSE_STATS = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.csv"
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?seasontype=2&week={week}"
ESPN_SCOREBOARD_NOW = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
ESPN_INJURIES = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"

ESPN_TO_NFLVERSE = {"WSH": "WAS", "LAR": "LA"}
NOTHING_TO_BUILD = 3  # exit code: no open games to build, keep the current page
OUT_STATUSES = {"Out", "Injured Reserve", "Doubtful", "Suspension", "Physically Unable to Perform"}

# Priors used when a player has no usable 2025 sample
POS_TGT_SHARE = {"WR": 0.12, "TE": 0.10, "RB": 0.08}
POS_YPT = {"WR": 8.3, "TE": 7.3, "RB": 5.9}
LEAGUE_YPC = 4.3

# Spread of outcomes as a share of the projection, plus a floor (yards)
SD_MODEL = {"pass": (0.24, 16.0), "rush": (0.47, 9.0), "rec": (0.56, 9.0)}


def get_csv(url):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text), low_memory=False)


def get_json(url):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def norm_name(s):
    s = re.sub(r"[.'’]", "", str(s)).lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def shrink(obs, n_obs, prior, n_prior):
    if n_obs <= 0:
        return prior
    return (obs * n_obs + prior * n_prior) / (n_obs + n_prior)


def wavg(values, weights):
    tw = sum(weights)
    return sum(v * w for v, w in zip(values, weights)) / tw if tw else 0.0


def week_weight(week, cur_week):
    # Newer games count more: last week 1.0, two weeks back 0.8, then 0.65, 0.55...
    back = cur_week - week
    return max(0.35, 1.0 - 0.2 * (back - 1) + 0.05 * max(0, back - 2))


def gamma_params(mu, kind):
    frac, floor = SD_MODEL[kind]
    sd = max(floor, frac * mu)
    k = (mu / sd) ** 2
    theta = sd * sd / mu
    return k, theta, sd


def half_line(x):
    return math.floor(x) + 0.5


def project(cur, prev, games, cur_week, status_of=lambda t, n: "", note_of=lambda t, n: "", active=None):
    """Projections for every skill player on the slate.

    cur/prev: nflverse weekly stats (REG, skill positions) for this season before cur_week, and last season.
    games: slate dicts with home/away/total/homeSpread/homeITT/awayITT/wet/id.
    active: optional set of player_ids known to be playing (used by the backtest in place of the injury report).
    Returns (props, league, defense).
    """
    # ---------- league baselines ----------
    tg = cur.groupby(["game_id", "team"]).agg(
        att=("attempts", "sum"), car=("carries", "sum"), pyd=("passing_yards", "sum"),
        ryd=("rushing_yards", "sum"), tgt=("targets", "sum"), week=("week", "first"),
        opp=("opponent_team", "first")).reset_index()
    lg_att = tg.att.mean()
    lg_car = tg.car.mean()
    lg_pyd = tg.pyd.mean()
    lg_ryd = tg.ryd.mean()
    lg_ypa = tg.pyd.sum() / tg.att.sum()
    lg_pts = sum(g["total"] for g in games) / len(games) / 2

    # Team volume (recency weighted, shrunk to league over 3 games)
    team_vol = {}
    for team, g in tg.groupby("team"):
        w = [week_weight(wk, cur_week) for wk in g.week]
        n = len(g)
        team_vol[team] = {
            "att": shrink(wavg(g.att, w), n, lg_att, 2),
            "car": shrink(wavg(g.car, w), n, lg_car, 2),
            "last_week": int(g.week.max()),
        }

    # Defense: yards allowed relative to league, by unit and receiving position
    rec_pos = cur.groupby(["game_id", "opponent_team", "position"]).receiving_yards.sum().reset_index()
    lg_rec_pos = rec_pos.groupby("position").receiving_yards.sum() / tg.shape[0]
    defense = {}
    for opp, g in tg.groupby("opp"):
        n = len(g)
        d = {
            "pass": shrink(g.pyd.mean() / lg_pyd, n, 1.0, 4),
            "rush": shrink((g.ryd.sum() / max(1, g.car.sum())) / (tg.ryd.sum() / tg.car.sum()), n, 1.0, 4),
        }
        rp = rec_pos[rec_pos.opponent_team == opp]
        for pos in ["WR", "TE", "RB"]:
            v = rp[rp.position == pos].receiving_yards.sum() / n
            d[pos] = shrink(v / lg_rec_pos.get(pos, 1), n, 1.0, 4)
        defense[opp] = d

    # ---------- player history ----------
    prev_agg = prev.groupby("player_id").agg(
        g=("week", "count"), att=("attempts", "sum"), pyd=("passing_yards", "sum"),
        car=("carries", "sum"), ryd=("rushing_yards", "sum"), tgt=("targets", "sum"),
        recyd=("receiving_yards", "sum"), tshare=("target_share", "mean")).to_dict("index")
    prev_team_car = prev.groupby(["game_id", "team"]).carries.sum().reset_index()
    prev_team_car_avg = prev_team_car.groupby("team").carries.mean().to_dict()
    prev_car_share = {}
    for pid, g in prev.groupby("player_id"):
        tm = g.team.mode().iloc[0]
        prev_car_share[pid] = (g.carries.sum() / max(1, len(g))) / prev_team_car_avg.get(tm, lg_car)

    team_car_by_game = tg.set_index(["game_id", "team"]).car.to_dict()

    slate_teams = {}
    for gm in games:
        slate_teams[gm["home"]] = (gm, gm["away"], True)
        slate_teams[gm["away"]] = (gm, gm["home"], False)

    players = []
    for pid, g in cur.groupby("player_id"):
        g = g.sort_values("week")
        team = g.team.iloc[-1]
        if team not in slate_teams:
            continue
        g = g[g.team == team]
        tv = team_vol.get(team)
        if tv is None:
            continue
        # must have appeared in one of the team's last two games
        if g.week.max() < tv["last_week"] - 1:
            continue
        name = g.player_display_name.iloc[-1]
        pos = g.position.iloc[-1]
        st = status_of(team, name)
        if st in OUT_STATUSES:
            continue
        if active is not None and pid not in active:
            continue
        w = [week_weight(wk, cur_week) for wk in g.week]
        car_share = [r.carries / max(1, team_car_by_game.get((r.game_id, team), lg_car)) for r in g.itertuples()]
        players.append({
            "pid": pid, "name": name, "pos": pos, "team": team, "status": st,
            "note": note_of(team, name) if st else "",
            "n": len(g), "w": w,
            "att": wavg(g.attempts, w), "att_tot": g.attempts.sum(), "pyd_tot": g.passing_yards.sum(),
            "tshare": wavg(g.target_share.fillna(0), w), "tgt_tot": g.targets.sum(), "recyd_tot": g.receiving_yards.sum(),
            "cshare": wavg(car_share, w), "car_tot": g.carries.sum(), "ryd_tot": g.rushing_yards.sum(),
            "logs": [{"wk": int(r.week), "opp": r.opponent_team, "pass": int(r.passing_yards), "att": int(r.attempts),
                      "rush": int(r.rushing_yards), "car": int(r.carries), "rec": int(r.receiving_yards),
                      "tgt": int(r.targets), "ts": round(float(r.target_share or 0), 3)} for r in g.itertuples()],
            "headshot": g.headshot_url.iloc[-1] if isinstance(g.headshot_url.iloc[-1], str) else "",
        })

    # Starting QB per team: most attempts in the team's latest game among available QBs
    starters = {}
    for team in slate_teams:
        qbs = [p for p in players if p["team"] == team and p["pos"] == "QB" and p["att_tot"] > 0]
        if not qbs:
            continue
        latest = tv_last = team_vol[team]["last_week"]
        def latest_att(p):
            return next((l["att"] for l in p["logs"] if l["wk"] == latest), 0)
        qbs.sort(key=lambda p: (latest_att(p), p["att_tot"]), reverse=True)
        starters[team] = qbs[0]["pid"]

    # Shares, shrunk to priors, then scaled when the active group falls short (vacated usage)
    for p in players:
        pr = prev_agg.get(p["pid"])
        pos = p["pos"]
        prior_ts = pr["tshare"] if pr and pr["g"] >= 6 and not math.isnan(pr["tshare"]) else POS_TGT_SHARE.get(pos, 0.03)
        if pos == "QB":
            prior_ts = 0.0
        p["ts_adj"] = shrink(p["tshare"], p["n"], prior_ts, 1.5)
        prior_cs = prev_car_share.get(p["pid"], 0.25 if pos == "RB" else 0.04) if pr and pr["g"] >= 6 else (0.25 if pos == "RB" else 0.04)
        p["cs_adj"] = shrink(p["cshare"], p["n"], prior_cs, 1.5)
        prior_ypt = (pr["recyd"] / pr["tgt"]) if pr and pr["tgt"] >= 30 else POS_YPT.get(pos, 6.5)
        p["ypt"] = shrink(p["recyd_tot"] / p["tgt_tot"], p["tgt_tot"], prior_ypt, 25) if p["tgt_tot"] else prior_ypt
        prior_ypc = (pr["ryd"] / pr["car"]) if pr and pr["car"] >= 40 else (LEAGUE_YPC if pos != "QB" else 5.0)
        p["ypc"] = shrink(p["ryd_tot"] / p["car_tot"], p["car_tot"], prior_ypc, 60) if p["car_tot"] else prior_ypc
        prior_ypa = (pr["pyd"] / pr["att"]) if pr and pr["att"] >= 120 else lg_ypa - 0.4
        p["ypa"] = shrink(p["pyd_tot"] / p["att_tot"], p["att_tot"], prior_ypa, 100) if p["att_tot"] else prior_ypa
        p["boost_t"] = 1.0
        p["boost_c"] = 1.0

    for team in slate_teams:
        grp = [p for p in players if p["team"] == team and p["pos"] != "QB"]
        st = sum(p["ts_adj"] for p in grp)
        if st > 0 and st < 0.90:
            f = min(0.90 / st, 1.25)
            for p in grp:
                p["boost_t"] = f
        grp_c = [p for p in players if p["team"] == team and (p["pos"] != "QB" or p["pid"] == starters.get(team))]
        sc = sum(p["cs_adj"] for p in grp_c)
        if sc > 0 and sc < 0.92:
            f = min(0.92 / sc, 1.25)
            for p in grp_c:
                if p["pos"] == "RB":
                    p["boost_c"] = f

    # ---------- projections ----------
    props = []
    for p in players:
        team = p["team"]
        gm, opp, is_home = slate_teams[team]
        spread_for = -gm["homeSpread"] if is_home else gm["homeSpread"]  # positive = this team favored
        itt = gm["homeITT"] if is_home else gm["awayITT"]
        tv = team_vol[team]
        dfn = defense.get(opp, {"pass": 1, "rush": 1, "WR": 1, "TE": 1, "RB": 1})

        script_pass = max(0.85, min(1.15, 1 - 0.011 * spread_for))
        script_rush = max(0.85, min(1.18, 1 + 0.014 * spread_for))
        itt_f = 1 + 0.30 * (itt - lg_pts) / lg_pts
        wet_f = 0.95 if gm["wet"] else 1.0

        proj_att = tv["att"] * script_pass
        proj_car = tv["car"] * script_rush

        drivers_common = {
            "spreadFor": spread_for, "itt": itt, "teamAtt": round(proj_att, 1), "teamCar": round(proj_car, 1),
            "scriptPass": round(script_pass, 3), "scriptRush": round(script_rush, 3), "wet": gm["wet"],
        }

        def add(kind, mu, drivers):
            if mu <= 0:
                return
            k, theta, sd = gamma_params(mu, kind)
            med = gamma.median(k, scale=theta)
            # Starting line: recency-weighted average of what he's actually done, on a half-yard
            field = {"pass": "pass", "rush": "rush", "rec": "rec"}[kind]
            recent = wavg([l[field] for l in p["logs"]], p["w"])
            # Stand-in for a book line until a real one is entered: halfway between recent form and the projection
            base = 0.5 * recent + 0.5 * med
            props.append({
                "id": f'{p["pid"]}-{kind}', "pid": p["pid"], "name": p["name"], "pos": p["pos"], "team": team,
                "opp": opp, "home": is_home, "game": gm["id"], "kind": kind,
                "mu": round(mu, 1), "median": round(med, 1), "k": round(k, 4), "theta": round(theta, 4),
                "estLine": half_line(base), "status": p["status"], "note": p["note"],
                "starter": starters.get(team) == p["pid"],
                "drivers": {**drivers_common, **drivers}, "logs": p["logs"], "n": p["n"],
            })

        if p["pos"] == "QB" and starters.get(team) == p["pid"]:
            att = proj_att * 0.97
            opp_f = dfn["pass"] ** 0.6
            mu = att * p["ypa"] * opp_f * itt_f * wet_f
            add("pass", mu, {"att": round(att, 1), "ypa": round(p["ypa"], 2), "oppF": round(opp_f, 3), "ittF": round(itt_f, 3)})
        if p["pos"] == "QB" and starters.get(team) != p["pid"]:
            continue

        cars = proj_car * p["cs_adj"] * p["boost_c"]
        opp_r = dfn["rush"] ** 0.6
        mu_r = cars * p["ypc"] * opp_r * (1 + 0.15 * (itt - lg_pts) / lg_pts)
        if (p["pos"] == "RB" and mu_r >= 28) or (p["pos"] == "QB" and mu_r >= 16):
            add("rush", mu_r, {"car": round(cars, 1), "cshare": round(p["cs_adj"] * p["boost_c"], 3),
                               "ypc": round(p["ypc"], 2), "oppF": round(opp_r, 3), "boost": round(p["boost_c"], 3)})

        if p["pos"] in ("WR", "TE", "RB"):
            tgts = proj_att * p["ts_adj"] * p["boost_t"]
            opp_p = dfn.get(p["pos"], 1.0) ** 0.6
            mu_c = tgts * p["ypt"] * opp_p * itt_f * wet_f
            if mu_c >= 24:
                add("rec", mu_c, {"tgt": round(tgts, 1), "tshare": round(p["ts_adj"] * p["boost_t"], 3),
                                  "ypt": round(p["ypt"], 2), "oppF": round(opp_p, 3), "ittF": round(itt_f, 3),
                                  "boost": round(p["boost_t"], 3)})


    league = {"att": round(lg_att, 1), "car": round(lg_car, 1), "ypa": round(lg_ypa, 2), "pts": round(lg_pts, 1)}
    return props, league, defense


def parse_slate(sb):
    """Games from an ESPN scoreboard that haven't finished and have a spread and total posted."""
    games = []
    for e in sb.get("events", []):
        c = e["competitions"][0]
        if c.get("status", {}).get("type", {}).get("completed"):
            continue
        teams = {}
        for t in c["competitors"]:
            ab = ESPN_TO_NFLVERSE.get(t["team"]["abbreviation"], t["team"]["abbreviation"])
            teams[t["homeAway"]] = ab
        odds = (c.get("odds") or [{}])[0]
        total = odds.get("overUnder")
        spread = odds.get("spread")  # negative = home favored
        if total is None or spread is None:
            continue
        home_spread = float(spread)
        w = e.get("weather") or c.get("weather") or {}
        indoor = bool(c.get("venue", {}).get("indoor"))
        cond = (w.get("displayValue") or "").strip()
        wet = (not indoor) and bool(re.search(r"rain|storm|shower|snow|thunder", cond, re.I))
        home_itt = total / 2 - home_spread / 2
        away_itt = total / 2 + home_spread / 2
        games.append({
            "id": e["id"], "kickoff": e["date"], "home": teams["home"], "away": teams["away"],
            "total": total, "homeSpread": home_spread,
            "homeITT": round(home_itt, 1), "awayITT": round(away_itt, 1),
            "indoor": indoor, "weather": cond if not indoor else "Dome",
            "temp": w.get("temperature"), "wet": wet,
            "venue": c.get("venue", {}).get("fullName", ""),
        })
    return games



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--week", type=int, help="Week to build. Leave out to use the current week from ESPN.")
    ap.add_argument("--out", default="data.json")
    args = ap.parse_args()
    season, cur_week = args.season, args.week
    auto_week = cur_week is None
    if auto_week:
        now = get_json(ESPN_SCOREBOARD_NOW)
        if now.get("season", {}).get("type") != 2:
            print("ESPN is not showing a regular-season week right now. Nothing built.")
            sys.exit(NOTHING_TO_BUILD)
        cur_week = int(now["week"]["number"])
        print(f"Current week from ESPN: {cur_week}")

    cur = get_csv(NFLVERSE_STATS.format(season=season))
    prev = get_csv(NFLVERSE_STATS.format(season=season - 1))
    sb = get_json(ESPN_SCOREBOARD.format(week=cur_week))
    inj = get_json(ESPN_INJURIES)

    skill = ["QB", "RB", "WR", "TE"]
    cur = cur[(cur.season_type == "REG") & (cur.position.isin(skill))].copy()
    prev = prev[(prev.season_type == "REG") & (prev.position.isin(skill))].copy()

    # ---------- games on the slate (not yet final) ----------
    games = parse_slate(sb)
    if not games and auto_week:
        # Early in the week ESPN can still be showing the finished week: move on to the next one
        cur_week += 1
        sb = get_json(ESPN_SCOREBOARD.format(week=cur_week))
        games = parse_slate(sb)
        if games:
            print(f"Week {cur_week - 1} is finished, building week {cur_week}")
    if not games:
        print(f"No upcoming games with odds posted for week {cur_week}. Nothing built.")
        sys.exit(NOTHING_TO_BUILD)

    # ---------- injuries ----------
    status_by = {}
    note_by = {}
    for t in inj.get("injuries", []):
        for x in t.get("injuries", []):
            a = x.get("athlete", {})
            ab = a.get("team", {}).get("abbreviation", "")
            ab = ESPN_TO_NFLVERSE.get(ab, ab)
            key = (ab, norm_name(a.get("displayName", "")))
            status_by[key] = x.get("status", "")
            note_by[key] = x.get("shortComment", "")

    def status_of(team, name):
        return status_by.get((team, norm_name(name)), "")

    def note_of(team, name):
        return note_by.get((team, norm_name(name)), "")

    props, league, defense = project(cur, prev, games, cur_week, status_of, note_of)


    out = {
        "season": season, "week": cur_week,
        "built": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        "statsThrough": int(cur.week.max()),
        "league": league,
        "games": games, "props": props,
        "defense": {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in defense.items()},
    }
    with open(args.out, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"{len(games)} games, {len(props)} props -> {args.out}")


if __name__ == "__main__":
    main()
