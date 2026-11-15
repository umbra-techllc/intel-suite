"""
NFL PropEdge data build.

Pulls weekly player stats (nflverse), the current week's schedule, game odds and weather (ESPN scoreboard),
the injury report and depth charts (ESPN), and the DraftKings player lines ESPN publishes. Then:

  1. projects passing, rushing and receiving yards for every player with the trained models,
  2. blends each projection with the DraftKings line and works out a chance for every posted bet (engine.py),
  3. has the AI choose and order the bets from the strongest candidates (picker.py),
  4. grades earlier weeks' picks, and writes ../nfl_propedge_data.json, which nfl_propedge.html loads.

Run:  python build_data.py            (current week, detected from ESPN)
      python build_data.py --week 5   (a specific week)
      python build_data.py --no-ai    (skip the AI step and rank by the engine alone)
Needs: pandas, scipy, requests, lightgbm. Set ANTHROPIC_API_KEY for the AI step. Exit code 3 means nothing to build.
"""
import argparse, json, math, os, re, sys, io, time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
from scipy.stats import gamma

NFLVERSE_STATS = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.csv"
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?seasontype=2&week={week}"
ESPN_SCOREBOARD_NOW = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
ESPN_INJURIES = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"
ESPN_TEAMS = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams"
ESPN_DEPTH = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{id}/depthcharts"

ESPN_TO_NFLVERSE = {"WSH": "WAS", "LAR": "LA"}
NOTHING_TO_BUILD = 3  # exit code: no open games to build, keep the current page
OUT_STATUSES = {"Out", "Injured Reserve", "Doubtful", "Suspension", "Physically Unable to Perform"}

# Priors used when a player has no usable 2025 sample
POS_TGT_SHARE = {"WR": 0.12, "TE": 0.10, "RB": 0.08}
POS_YPT = {"WR": 8.3, "TE": 7.3, "RB": 5.9}
LEAGUE_YPC = 4.3

# How an injured regular's work is passed on (fit on 2024-25 games where a regular sat out; see research/vacated.py).
# Teammates' shares grow by more than this, but the offense as a whole produces less without him, so the
# yardage bump is smaller. These values are fit on yards, and they leave fill-in projections unbiased.
VAC_TGT_FILL = 0.15  # receiving: how much of his target share shows up as extra yards for teammates
VAC_TGT_SAME = 0.9   # of that, the part that goes to his own position group
VAC_CAR_FILL = 0.35  # rushing: how much of a missing back's carries show up for the other backs
VAC_POW = 0.5        # flatter than current roles: smaller roles grow more than their size suggests
VAC_TGT_MIN = 0.12   # a regular in the passing game
VAC_CAR_MIN = 0.25   # a regular in the running game

# Outcome ranges measured from two seasons of results (written by research/fit_calibration.py)
CALIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")
CALIB = json.load(open(CALIB_PATH)) if os.path.exists(CALIB_PATH) else None


def calib_tier(kind, mu):
    if not CALIB:
        return 0
    return int(sum(mu > c for c in CALIB[kind]["cuts"]))


def calib_median(kind, mu):
    """Tier and median outcome for a projected average, from the measured outcome ranges."""
    if not CALIB:
        return 0, None
    t = calib_tier(kind, mu)
    q = CALIB[kind]["q"][t]  # ratio quantiles at 0, 1/100, ... 1
    return t, mu * q[50]


# ---------- learned models (trained by research/fit_models.py) ----------
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
FACTOR_GROUPS = [
    ("His recent games", ["recent_avg", "last_game", "max_game", "sd_games", "n_games", "played_last"]),
    ("How often he gets the ball", ["ts", "ts_base", "last_ts", "ay_share", "cs", "cs_base", "last_cs"]),
    ("What he does with it", ["ypt", "catch_rate", "rec_epa_t", "adot", "ypc", "rush_epa_c", "ypa", "pass_epa_a", "cpoe", "sack_rate", "pass_adot"]),
    ("Last season", ["prev_pg", "prev_g"]),
    ("Basic estimate from his share of plays", ["mu_struct"]),
    ("His team's play calling", ["team_att", "team_car", "pass_rate", "team_ypa"]),
    ("His quarterback", ["qb_ypa", "qb_new", "qb_n"]),
    ("Expected score and point spread", ["itt", "total", "spread_for", "home"]),
    ("Opponent's defense", ["opp_raw", "opp_pass", "opp_rush"]),
    ("Injured teammates", ["vac_ts_team", "vac_ts_pos", "vac_cs", "role_gain"]),
    ("Stadium and weather", ["dome", "wet"]),
    ("Position and week of season", ["pos_qb", "pos_rb", "pos_wr", "pos_te", "week"]),
]


def load_booster(name):
    import gzip, lightgbm as lgb
    gz = os.path.join(MODEL_DIR, name + ".txt.gz")
    if os.path.exists(gz):
        with gzip.open(gz, "rt") as fh:
            return lgb.Booster(model_str=fh.read())
    return lgb.Booster(model_file=os.path.join(MODEL_DIR, name + ".txt"))


def apply_models(props):
    """Replace each projection with the learned models' center, outcome range and factor breakdown."""
    path = os.path.join(MODEL_DIR, "bundle.json")
    if not props or not os.path.exists(path):
        return False
    import lightgbm as lgb
    bundle = json.load(open(path))
    feats = bundle["features"]
    idx = {f: i for i, f in enumerate(feats)}
    for kind, m in bundle["markets"].items():
        rows = [p for p in props if p["kind"] == kind]
        if not rows:
            continue
        X = np.array([[np.nan if p["feat"].get(f) is None else float(p["feat"][f]) for f in feats] for p in rows], dtype=float)
        center = load_booster(f"{kind}_center")
        gp = center.predict(X)
        gc = center.predict(X, pred_contrib=True)
        if m["ridge"]:
            r = m["ridge"]
            Xi = np.where(np.isnan(X), np.array(r["median"]), X)
            Z = (Xi - np.array(r["mean"])) / np.array(r["scale"])
            rc = Z * np.array(r["coef"])
            loc = (gp + rc.sum(axis=1) + r["intercept"]) / 2
            contrib, base = (gc[:, :-1] + rc) / 2, (gc[:, -1] + r["intercept"]) / 2
        else:
            loc, contrib, base = gp, gc[:, :-1], gc[:, -1]
        loc = np.clip(loc, 1, None)
        cuts = m["cuts"]
        if m["spread_model"]:
            spread = load_booster(f"{kind}_spread")
            sc = np.clip(spread.predict(np.column_stack([X, loc])), 3, None)
        for i, p in enumerate(rows):
            t = int(sum(loc[i] > c for c in cuts))
            if m["spread_model"]:
                yq = loc[i] + sc[i] * np.array(m["zq"][t])
            else:
                yq = loc[i] * np.array(m["rq"][t])
            yq = np.maximum.accumulate(np.clip(yq, -10, None))
            groups = []
            for label, fs in FACTOR_GROUPS:
                if kind == "pass" and label == "His quarterback":
                    continue
                fs2 = fs + (["qb_ypa", "qb_n", "qb_new"] if kind == "pass" and label == "What he does with it" else [])
                v = float(sum(contrib[i, idx[f]] for f in fs2 if f in idx))
                if abs(v) >= 0.5:
                    groups.append({"label": label, "yards": round(v, 1)})
            groups.sort(key=lambda g: -abs(g["yards"]))
            p["mu"] = round(float(loc[i]), 1)
            p["median"] = round(float(yq[50]), 1)
            p["estLine"] = half_line(yq[50])
            p["q"] = [round(float(v), 1) for v in yq[::2]]
            p["_grid"] = yq
            p["factors"] = {"base": round(float(base[i]), 1), "top": groups[:5]}
            p["model"] = "learned"
    return True


# Spread of outcomes as a share of the projection, plus a floor (yards)
SD_MODEL = {"pass": (0.24, 16.0), "rush": (0.47, 9.0), "rec": (0.56, 9.0)}


def get_csv(url):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text), low_memory=False)


def get_json(url):
    last = None
    for i in range(4):  # ESPN now and then refuses a request; a short wait fixes it
        try:
            r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0 (umbra-intel-suite)"})
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            time.sleep(2 * (i + 1))
    raise last


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


def project(cur, prev, games, cur_week, status_of=lambda t, n: "", note_of=lambda t, n: "", active=None, depth=None):
    """Projections for every skill player on the slate.

    cur/prev: nflverse weekly stats (REG, skill positions) for this season before cur_week, and last season.
    games: slate dicts with home/away/total/homeSpread/homeITT/awayITT/wet/id.
    active: optional set of player_ids known to be playing (used by the backtest in place of the injury report).
    depth: optional {team: {"QB": [names in depth-chart order], ...}} used to pick the starting QB.
    Returns (props, league, defense, injuries).
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

    # ---------- this week's roster: recent regulars, split into available and out ----------
    pool = []
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
        out = st in OUT_STATUSES or (active is not None and pid not in active)
        w = [week_weight(wk, cur_week) for wk in g.week]
        car_share = [r.carries / max(1, team_car_by_game.get((r.game_id, team), lg_car)) for r in g.itertuples()]
        pool.append({
            "pid": pid, "name": name, "pos": pos, "team": team, "status": st, "out": out,
            "note": note_of(team, name) if st else "",
            "n": len(g), "w": w,
            "att": wavg(g.attempts, w), "att_tot": g.attempts.sum(), "pyd_tot": g.passing_yards.sum(),
            "tshare": wavg(g.target_share.fillna(0), w), "tgt_tot": g.targets.sum(), "recyd_tot": g.receiving_yards.sum(),
            "cshare": wavg(car_share, w), "car_tot": g.carries.sum(), "ryd_tot": g.rushing_yards.sum(),
            "logs": [{"wk": int(r.week), "opp": r.opponent_team, "pass": int(r.passing_yards), "att": int(r.attempts),
                      "rush": int(r.rushing_yards), "car": int(r.carries), "rec": int(r.receiving_yards),
                      "tgt": int(r.targets), "ts": round(float(r.target_share or 0), 3)} for r in g.itertuples()],
            "last_att": int(g[g.week == tv["last_week"]].attempts.sum()),
            # extra history used by the learned model
            "ay_share": wavg(g.air_yards_share.fillna(0), w), "rec_ay": float(g.receiving_air_yards.fillna(0).sum()),
            "rec_epa": float(g.receiving_epa.fillna(0).sum()), "receptions": float(g.receptions.sum()),
            "rush_epa": float(g.rushing_epa.fillna(0).sum()), "pass_epa": float(g.passing_epa.fillna(0).sum()),
            "cpoe": float(g.passing_cpoe.mean()) if g.passing_cpoe.notna().any() else float("nan"),
            "sacks": float(g.sacks_suffered.fillna(0).sum()), "pass_ay": float(g.passing_air_yards.fillna(0).sum()),
            "last_ts": float(g.target_share.fillna(0).iloc[-1]), "last_cs": float(car_share[-1]),
            "played_last": bool(g.week.max() == tv["last_week"]),
        })

    prev_by_name = {}
    for pid, g in prev.groupby("player_id"):
        prev_by_name[norm_name(g.player_display_name.iloc[-1])] = pid

    def rates(p):
        """Usage and efficiency, each blended with last season (or a position baseline) until this season's sample is big enough."""
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
        # Backups who haven't thrown much get a lower baseline than an average starter
        prior_ypa = (pr["pyd"] / pr["att"]) if pr and pr["att"] >= 120 else lg_ypa - (0.6 if p["n"] == 0 else 0.4)
        p["ypa"] = shrink(p["pyd_tot"] / p["att_tot"], p["att_tot"], prior_ypa, 100) if p["att_tot"] else prior_ypa
        p["ts_base"], p["cs_base"] = p["ts_adj"], p["cs_adj"]
        p["gain_t"], p["gain_c"] = {}, {}

    for p in pool:
        rates(p)

    # ---------- starting QB: depth chart first, then whoever threw the most last game ----------
    starters, qb_note = {}, {}
    for team in slate_teams:
        qbs = [p for p in pool if p["team"] == team and p["pos"] == "QB"]
        avail = [p for p in qbs if not p["out"]]
        starter = None
        for nm in (depth or {}).get(team, {}).get("QB", []):
            if status_of(team, nm) in OUT_STATUSES:
                continue
            match = next((p for p in avail if norm_name(p["name"]) == norm_name(nm)), None)
            if match is None and any(norm_name(p["name"]) == norm_name(nm) for p in qbs):
                continue  # listed on the depth chart but ruled out
            if match is None:
                # A backup with no snaps this season: build him from last season or a backup baseline
                match = {"pid": prev_by_name.get(norm_name(nm), "dc-" + norm_name(nm)), "name": nm, "pos": "QB", "team": team,
                         "status": status_of(team, nm), "out": False, "note": "", "n": 0, "w": [], "att": 0, "att_tot": 0,
                         "pyd_tot": 0, "tshare": 0, "tgt_tot": 0, "recyd_tot": 0, "cshare": 0, "car_tot": 0, "ryd_tot": 0,
                         "logs": [], "last_att": 0}
                rates(match)
                pool.append(match)
            starter = match
            break
        if starter is None:
            cands = [p for p in avail if p["att_tot"] > 0]
            if not cands:
                continue
            cands.sort(key=lambda p: (p["last_att"], p["att_tot"]), reverse=True)
            starter = cands[0]
        starters[team] = starter["pid"]
        # Who he is replacing, if last game's main passer is out
        lead = max(qbs, key=lambda p: p["last_att"], default=None)
        if lead and lead["pid"] != starter["pid"] and lead["out"] and lead["last_att"] >= 10:
            qb_note[team] = {"replacing": lead["name"], "status": lead["status"] or "Out"}

    # ---------- injured players' work goes to teammates (fit on 2024-25 games with a regular missing) ----------
    team_out = {}
    for team in slate_teams:
        here = [p for p in pool if p["team"] == team and not p["out"] and p["pos"] != "QB"]
        gone = [p for p in pool if p["team"] == team and p["out"] and p["pos"] != "QB"]
        listed = []
        for o in gone:
            if o["ts_adj"] < 0.05 and o["cs_adj"] < 0.05:
                continue
            listed.append({"name": o["name"], "pos": o["pos"], "status": o["status"] or "Out",
                           "ts": round(o["ts_adj"], 3), "cs": round(o["cs_adj"], 3), "note": o["note"]})
            # Only a real regular's absence moves work around (the same cutoffs the fit used)
            # Targets: about half get re-absorbed, mostly by the same position
            V = o["ts_adj"] * VAC_TGT_FILL if o["ts_adj"] >= VAC_TGT_MIN else 0
            same = [p for p in here if p["pos"] == o["pos"] and p["ts_base"] > 0]
            if same:
                ws = [p["ts_base"] ** VAC_POW for p in same]
                for p, wt in zip(same, ws):
                    add = V * VAC_TGT_SAME * wt / sum(ws)
                    p["ts_adj"] += add; p["gain_t"][o["name"]] = p["gain_t"].get(o["name"], 0) + add
                rest = V * (1 - VAC_TGT_SAME)
            else:
                rest = V
            tot = sum(p["ts_base"] for p in here)
            for p in here:
                if tot > 0 and p["ts_base"] > 0:
                    add = rest * p["ts_base"] / tot
                    p["ts_adj"] += add; p["gain_t"][o["name"]] = p["gain_t"].get(o["name"], 0) + add
            # Carries: a missing back's carries mostly go to the other backs, spread more evenly than their current roles
            if o["pos"] == "RB" and o["cs_adj"] >= VAC_CAR_MIN:
                V = o["cs_adj"] * VAC_CAR_FILL
                backs = [p for p in here if p["pos"] == "RB" and p["cs_base"] > 0]
                ws = [p["cs_base"] ** VAC_POW for p in backs]
                for p, wt in zip(backs, ws):
                    add = V * wt / sum(ws)
                    p["cs_adj"] += add; p["gain_c"][o["name"]] = p["gain_c"].get(o["name"], 0) + add
        starter_out = qb_note.get(team)
        if starter_out:
            listed.insert(0, {"name": starter_out["replacing"], "pos": "QB", "status": starter_out["status"], "ts": 0, "cs": 0, "note": ""})
        if listed:
            team_out[team] = listed

    players = [p for p in pool if not p["out"]]

    # Team context for the learned model
    team_ctx = {}
    for team, g in tg.groupby("team"):
        wts = [week_weight(wk, cur_week) for wk in g.week]
        att, car = wavg(g.att, wts), wavg(g.car, wts)
        st = next((p for p in pool if p["pid"] == starters.get(team)), None)
        team_ctx[team] = {
            "pass_rate": att / max(1, att + car), "team_ypa": g.pyd.sum() / max(1, g.att.sum()),
            "qb_ypa": st["ypa"] if st else lg_ypa, "qb_n": st["n"] if st else 0,
            "qb_new": 1.0 if (team in qb_note or (st and st["n"] == 0)) else 0.0,
            "games": len(g),
        }
    vac = {}
    for team in slate_teams:
        gone = [p for p in pool if p["team"] == team and p["out"] and p["pos"] != "QB"]
        vac[team] = {"ts": sum(o["ts_adj"] for o in gone if o["ts_adj"] >= 0.05),
                     "cs": sum(o["cs_adj"] for o in gone if o["pos"] == "RB" and o["cs_adj"] >= 0.05),
                     "by_pos": {pos: sum(o["ts_adj"] for o in gone if o["pos"] == pos and o["ts_adj"] >= 0.05) for pos in ("WR", "TE", "RB")}}
    prev_pg = {}
    for pid, g in prev.groupby("player_id"):
        n = g.week.nunique()
        prev_pg[pid] = {"g": n, "pass": g.passing_yards.sum() / n, "rush": g.rushing_yards.sum() / n, "rec": g.receiving_yards.sum() / n}

    def gains(d):
        return [{"name": k, "share": round(v, 3)} for k, v in sorted(d.items(), key=lambda kv: -kv[1]) if v >= 0.005]

    # ---------- projections ----------
    props = []
    for p in players:
        team = p["team"]
        gm, opp, is_home = slate_teams[team]
        spread_for = -gm["homeSpread"] if is_home else gm["homeSpread"]  # positive = this team favored
        itt = gm["homeITT"] if is_home else gm["awayITT"]
        tv = team_vol[team]
        dfn = defense.get(opp, {"pass": 1, "rush": 1, "WR": 1, "TE": 1, "RB": 1})

        # The spread no longer moves passing volume: in the 2024-25 replay it made passing projections slightly worse
        script_pass = 1.0
        script_rush = max(0.85, min(1.18, 1 + 0.014 * spread_for))
        itt_f = 1 + 0.30 * (itt - lg_pts) / lg_pts
        wet_f = 0.95 if gm["wet"] else 1.0

        proj_att = tv["att"] * script_pass
        proj_car = tv["car"] * script_rush

        drivers_common = {
            "spreadFor": spread_for, "itt": itt, "teamAtt": round(proj_att, 1), "teamCar": round(proj_car, 1),
            "scriptPass": round(script_pass, 3), "scriptRush": round(script_rush, 3), "wet": gm["wet"],
        }

        def features(kind, mu):
            tc = team_ctx.get(team, {})
            logs = [l[kind] for l in p["logs"]]
            pr = prev_pg.get(p["pid"], {})
            raw_opp = {"pass": dfn["pass"], "rush": dfn["rush"], "rec": dfn.get(p["pos"], 1.0)}[kind]
            v = vac.get(team, {"ts": 0, "cs": 0, "by_pos": {}})
            return {
                "mu_struct": mu, "pos_qb": p["pos"] == "QB", "pos_rb": p["pos"] == "RB", "pos_wr": p["pos"] == "WR", "pos_te": p["pos"] == "TE",
                "n_games": p["n"], "week": cur_week, "played_last": p.get("played_last", False),
                "recent_avg": wavg(logs, p["w"]) if logs else float("nan"),
                "last_game": logs[-1] if logs else float("nan"),
                "max_game": max(logs) if logs else float("nan"),
                "sd_games": float(np.std(logs)) if len(logs) >= 2 else float("nan"),
                "prev_pg": pr.get(kind, float("nan")), "prev_g": pr.get("g", 0),
                "ts": p["ts_adj"], "ts_base": p["ts_base"], "last_ts": p.get("last_ts", 0), "ay_share": p.get("ay_share", 0),
                "adot": p.get("rec_ay", 0) / p["tgt_tot"] if p["tgt_tot"] else float("nan"),
                "catch_rate": p.get("receptions", 0) / p["tgt_tot"] if p["tgt_tot"] else float("nan"),
                "rec_epa_t": p.get("rec_epa", 0) / p["tgt_tot"] if p["tgt_tot"] else float("nan"),
                "ypt": p["ypt"], "cs": p["cs_adj"], "cs_base": p["cs_base"], "last_cs": p.get("last_cs", 0), "ypc": p["ypc"],
                "rush_epa_c": p.get("rush_epa", 0) / p["car_tot"] if p["car_tot"] else float("nan"),
                "ypa": p["ypa"], "pass_epa_a": p.get("pass_epa", 0) / p["att_tot"] if p["att_tot"] else float("nan"),
                "cpoe": p.get("cpoe", float("nan")), "sack_rate": p.get("sacks", 0) / max(1, p["att_tot"] + p.get("sacks", 0)),
                "pass_adot": p.get("pass_ay", 0) / p["att_tot"] if p["att_tot"] else float("nan"),
                "team_att": tv["att"], "team_car": tv["car"], "pass_rate": tc.get("pass_rate", 0.55),
                "team_ypa": tc.get("team_ypa", lg_ypa), "qb_ypa": tc.get("qb_ypa", lg_ypa), "qb_new": tc.get("qb_new", 0), "qb_n": tc.get("qb_n", 0),
                "itt": itt, "total": gm["total"], "spread_for": spread_for, "home": is_home, "dome": bool(gm.get("indoor")), "wet": bool(gm.get("wet")),
                "opp_raw": raw_opp, "opp_pass": dfn["pass"], "opp_rush": dfn["rush"],
                "vac_ts_team": v["ts"], "vac_ts_pos": v["by_pos"].get(p["pos"], 0), "vac_cs": v["cs"],
                "role_gain": (p["ts_adj"] - p["ts_base"]) if kind == "rec" else (p["cs_adj"] - p["cs_base"]) if kind == "rush" else 0,
            }

        def add(kind, mu, drivers, role=None):
            if mu <= 0:
                return
            k, theta, sd = gamma_params(mu, kind)
            tier, med = calib_median(kind, mu)
            if med is None:
                med = gamma.median(k, scale=theta)
            props.append({
                "id": f'{p["pid"]}-{kind}', "pid": p["pid"], "name": p["name"], "pos": p["pos"], "team": team,
                "opp": opp, "home": is_home, "game": gm["id"], "kind": kind,
                "mu": round(mu, 1), "median": round(med, 1), "k": round(k, 4), "theta": round(theta, 4), "tier": tier,
                "estLine": half_line(med), "status": p["status"], "note": p["note"],
                "starter": starters.get(team) == p["pid"],
                "drivers": {**drivers_common, **drivers}, "logs": p["logs"], "n": p["n"],
                "role": role, "feat": features(kind, mu),
            })

        if p["pos"] == "QB" and starters.get(team) == p["pid"]:
            att = proj_att * 0.97
            opp_f = dfn["pass"] ** 0.6
            mu = att * p["ypa"] * opp_f * itt_f * wet_f
            qn = qb_note.get(team)
            add("pass", mu, {"att": round(att, 1), "ypa": round(p["ypa"], 2), "oppF": round(opp_f, 3), "ittF": round(itt_f, 3)},
                {"replacing": qn["replacing"] if qn else None, "firstStart": p["n"] == 0} if qn or p["n"] == 0 else None)
        if p["pos"] == "QB" and starters.get(team) != p["pid"]:
            continue

        cars = proj_car * p["cs_adj"]
        opp_r = dfn["rush"] ** 0.6
        mu_r = cars * p["ypc"] * opp_r * (1 + 0.15 * (itt - lg_pts) / lg_pts)
        if (p["pos"] == "RB" and mu_r >= 28) or (p["pos"] == "QB" and mu_r >= 16):
            g_c = gains(p["gain_c"])
            add("rush", mu_r, {"car": round(cars, 1), "cshare": round(p["cs_adj"], 3), "cshareBase": round(p["cs_base"], 3),
                               "ypc": round(p["ypc"], 2), "oppF": round(opp_r, 3), "boost": round(p["cs_adj"] / p["cs_base"], 3) if p["cs_base"] else 1},
                {"kind": "carries", "base": round(p["cs_base"], 3), "now": round(p["cs_adj"], 3), "from": g_c} if g_c else None)

        if p["pos"] in ("WR", "TE", "RB"):
            tgts = proj_att * p["ts_adj"]
            opp_p = dfn.get(p["pos"], 1.0) ** 0.6
            mu_c = tgts * p["ypt"] * opp_p * itt_f * wet_f
            if mu_c >= 24:
                g_t = gains(p["gain_t"])
                add("rec", mu_c, {"tgt": round(tgts, 1), "tshare": round(p["ts_adj"], 3), "tshareBase": round(p["ts_base"], 3),
                                  "ypt": round(p["ypt"], 2), "oppF": round(opp_p, 3), "ittF": round(itt_f, 3),
                                  "boost": round(p["ts_adj"] / p["ts_base"], 3) if p["ts_base"] else 1},
                    {"kind": "targets", "base": round(p["ts_base"], 3), "now": round(p["ts_adj"], 3), "from": g_t} if g_t else None)

    # Questionable players who are projected (may not play)
    questionable = {}
    for p in players:
        if p["status"] == "Questionable" and (p["ts_adj"] >= 0.08 or p["cs_adj"] >= 0.15 or starters.get(p["team"]) == p["pid"]):
            questionable.setdefault(p["team"], []).append({"name": p["name"], "pos": p["pos"], "note": p["note"]})

    league = {"att": round(lg_att, 1), "car": round(lg_car, 1), "ypa": round(lg_ypa, 2), "pts": round(lg_pts, 1)}
    return props, league, defense, {"out": team_out, "questionable": questionable}


def parse_slate(sb):
    """Games from an ESPN scoreboard that haven't finished and have a spread and total posted."""
    games = []
    for e in sb.get("events", []):
        c = e["competitions"][0]
        if c.get("status", {}).get("type", {}).get("state", "pre") != "pre":
            continue  # under way or finished
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



def candidate_pools(full, P):
    """Narrow the full board to the bets worth the AI's attention."""
    by = {}
    for l in full:
        by.setdefault((l["pid"], l["stat"]), []).append(l)
    safe, pay = [], []
    for ls in by.values():
        # safest: the most demanding "this many or more" bet that still has at least a 76% chance
        s = [l for l in ls if l["kind"] == "alt" and 0.76 <= l["p"] <= 0.93]
        if s:
            safe.append(max(s, key=lambda l: l["num"]))
        # payout: whichever side of the main line the engine leans to
        m = [l for l in ls if l["kind"] in ("over", "under") and l["p"] >= 0.53]
        if m:
            pay.append(max(m, key=lambda l: l["p"]))
    # where the model and the line disagree a lot about a "safe" bet, trust it less
    score = lambda l: l["p"] - 0.5 * abs(l["p"] - l["pm"]) - (0.04 if P[l["pid"]]["status"] else 0)
    safe.sort(key=score, reverse=True); pay.sort(key=lambda l: -l["p"])
    def cap(pool, n, per_player):
        out, seen = [], {}
        for l in pool:
            if seen.get(l["pid"], 0) < per_player:
                seen[l["pid"]] = seen.get(l["pid"], 0) + 1; out.append(l)
            if len(out) >= n:
                break
        return out
    return cap(safe, 40, 1), cap(pay, 24, 1)


def board_text(data, safe, pay, P):
    """The board written out for the AI."""
    def row(l):
        p = P[l["pid"]]
        last = ", ".join(str(x) for x in l["last"])
        return (f'{l["id"]} | {p["name"]}, {p["pos"]} ({p["team"]} {"vs" if p["home"] else "at"} {p["opp"]}) | {l["text"]} '
                f'| chance {round(l["p"] * 100)}% (model alone {round(l["pm"] * 100)}%) | projection {l["proj"]}, model alone {l["model"]}'
                + (f', DraftKings line {l["line"]:g}' if l.get("line") is not None else "")
                + f' | did this in {l["hits"]} of {l["of"]} games this season | games, newest first: {last}'
                + (f' | notes: {"; ".join(p["flags"])}' if p["flags"] else "")
                + (f' | injury note: {p["note"]}' if p["note"] else ""))
    out = [f'NFL week {data["week"]}, {data["season"]} season. Player lines from {data["book"] or "the sportsbook"}.', "", "GAMES"]
    for g in data["games"]:
        fav = g["home"] if g["homeSpread"] < 0 else g["away"]
        out.append(f'- {g["away"]} at {g["home"]}: {fav} favored by {abs(g["homeSpread"]):g}, expected total points {g["total"]:g}, '
                   f'{"indoors" if g["indoor"] else "weather: " + (g["weather"] or "not posted")}')
        for t in (g["away"], g["home"]):
            o = [f'{x["name"]} ({x["pos"]}, {x["status"].lower()})' for x in data["injuries"]["out"].get(t, [])]
            q = [f'{x["name"]} ({x["pos"]})' for x in data["injuries"]["questionable"].get(t, [])]
            if o:
                out.append(f"    {t} out: " + "; ".join(o))
            if q:
                out.append(f"    {t} questionable: " + "; ".join(q))
    out += ["", "CANDIDATES FOR THE SAFEST LIST"] + [row(l) for l in safe]
    out += ["", "CANDIDATES FOR THE PAYOUT LIST"] + [row(l) for l in pay]
    return "\n".join(out)


def main():
    import board as BD, dk_lines as DK, picker
    from concurrent.futures import ThreadPoolExecutor
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--week", type=int, help="Week to build. Leave out to use the current week from ESPN.")
    ap.add_argument("--no-ai", action="store_true")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(here), "nfl_propedge_data.json"))
    args = ap.parse_args()
    if args.no_ai:
        os.environ.pop("ANTHROPIC_API_KEY", None)
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

    # ---------- games on the slate (not yet started) ----------
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
    started = len(sb.get("events", [])) - len(games)  # games of this week already under way or finished

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

    # ---------- depth charts: who is next up at each spot ----------
    depth = {}
    try:
        teams = get_json(ESPN_TEAMS)["sports"][0]["leagues"][0]["teams"]
        ids = {ESPN_TO_NFLVERSE.get(t["team"]["abbreviation"], t["team"]["abbreviation"]): t["team"]["id"] for t in teams}
        on_slate = {g["home"] for g in games} | {g["away"] for g in games}
        for team in sorted(on_slate):
            if team not in ids:
                continue
            try:
                dc = get_json(ESPN_DEPTH.format(id=ids[team])).get("depthchart", [])
            except Exception:
                continue
            spots = {}
            for chart in dc:
                for key, slot in chart.get("positions", {}).items():
                    if key in ("qb", "rb", "wr1", "wr2", "wr3", "te"):
                        spots.setdefault(key.upper(), [a.get("displayName", "") for a in slot.get("athletes", [])])
            if spots:
                depth[team] = spots
        print(f"Depth charts for {len(depth)} teams")
    except Exception as e:
        print(f"Depth charts unavailable ({e}); using last game's passer as the starter")

    # only games before this week count as history, even if this week's early games are already in the stats file
    props, league, defense, injuries = project(cur[cur.week < cur_week], prev, games, cur_week, status_of, note_of, depth=depth)
    if not apply_models(props):
        print("Trained models not found in model/. Nothing built.")
        sys.exit(1)
    change = {p["team"]: (p["role"] or {}).get("replacing") or "first start" for p in props
              if p["kind"] == "pass" and p.get("role") and (p["role"].get("replacing") or p["role"].get("firstStart"))}
    for p in props:
        if p["team"] in change:
            p["qbChange"] = change[p["team"]]

    # ---------- DraftKings lines (from ESPN), keyed to the same player ids as the stats ----------
    lines_by_game, book = {}, None
    try:
        espn_to_pid = DK.id_map(os.path.join(here, "research", "cache", "player_ids.csv"))
        with ThreadPoolExecutor(5) as ex:
            for g, gl in zip(games, ex.map(lambda g: DK.game_lines(g["id"]), games)):
                book = book or gl["book"]
                lines_by_game[g["id"]] = {
                    "main": {(espn_to_pid.get(a), k): v for (a, k), v in gl["main"].items() if espn_to_pid.get(a)},
                    "steps": {(espn_to_pid.get(a), k): v for (a, k), v in gl["steps"].items() if espn_to_pid.get(a)}}
    except Exception as e:
        print("Could not read the DraftKings lines:", e)
    blend = BD.load_blend()
    players, legs = BD.build(props, games, lines_by_game, blend)
    P = {p["id"]: p for p in players}
    full = [BD.describe(l, P) for l in legs]
    bundle = json.load(open(os.path.join(MODEL_DIR, "bundle.json")))

    data = {
        "season": season, "week": cur_week, "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "statsThrough": int(cur[cur.week < cur_week].week.max()), "book": book,
        "games": games, "injuries": injuries, "players": players,
        # every posted bet as [player id, stat, o/u/a (over, under, "or more"), number, chance, the model's own chance]
        "legs": [[l["pid"], l["stat"], l["kind"][0], l["num"], l["p"], l["pm"]] for l in legs],
        "links": BD.LINKS, "report": blend.get("report"), "weights": blend.get("weights"),
        "trained_on": bundle.get("trained_on"),
        "model_miss": {k: {"model": m["oof_mae"], "simple": m["engine_mae"], "n": m["n"]} for k, m in bundle["markets"].items()},
    }
    safe, pay = candidate_pools(full, P)
    text = board_text(data, safe, pay, P)
    def fallback(l):
        p = P[l["pid"]]
        why = f'Projected for {l["proj"]:g} {l["unit"]} and did this in {l["hits"]} of his {l["of"]} games this season.'
        return why, "; ".join(p["flags"][:2]), p["status"] == "" and p["n"] >= 3
    picks = picker.pick("yards", text, safe, pay, fallback)
    data["picks"] = picks
    data["pool"] = {"safest": [l["id"] for l in safe], "payout": [l["id"] for l in pay]}
    data["prompt"] = {"rules": picker.rules("yards"), "data": text, "model": picker.MODEL}
    print(f'{len(games)} games, {len(players)} players, {len(legs)} posted bets, picked by {picks["by"]}: '
          f'{len(picks["safest"])} safest, {len(picks["payout"])} payout')

    # ---------- results log: the picks as they stood before the week's first kickoff, graded the week after ----------
    log_path = os.path.join(here, "data", "picks_log.json")
    log = json.load(open(log_path)) if os.path.exists(log_path) else {"weeks": {}}
    BD.grade(log, cur, season, cur_week)
    key = f"{season}-W{cur_week:02d}"
    L = {l["id"]: l for l in full}
    if picks["safest"] and (started == 0 or key not in log["weeks"]):
        keep = lambda items: [{**{k: L[i["id"]][k] for k in ("id", "pid", "gid", "stat", "kind", "num", "p", "text")},
                               "name": P[L[i["id"]]["pid"]]["name"]} for i in items]
        log["weeks"][key] = {"season": season, "week": cur_week, "by": picks["by"], "at": picks["at"],
                             "safest": keep(picks["safest"]), "payout": keep(picks["payout"])}
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    json.dump(log, open(log_path, "w"), separators=(",", ":"))
    data["record"] = BD.record(log)

    with open(args.out, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print("wrote", args.out, os.path.getsize(args.out) // 1024, "KB")


if __name__ == "__main__":
    main()
