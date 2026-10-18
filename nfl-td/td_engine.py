"""
NFL TD Predictor feature engine.

Builds one row per player-game with information known before kickoff:
  - how often he gets the ball near the end zone (inside the 20, 10 and 5) and his share of his team's chances there
  - his touchdown history this season and last
  - his team's expected scoring (from the betting market), how often it scores, and how it splits passing and running TDs
  - his starting quarterback's touchdown-pass trends
  - the opponent's touchdowns allowed by position
  - work freed up by injured teammates
The same code builds past seasons (for training and testing) and this week's slate (for predictions):
upcoming games are appended as rows with no result, and every feature uses only earlier games.
"""
import numpy as np
import pandas as pd

SK = ["QB", "RB", "WR", "TE"]
HL = 3.0  # recency half-life, in games

PBP_COLS = ["game_id", "season", "week", "season_type", "posteam", "defteam", "yardline_100", "play_type",
            "rusher_player_id", "receiver_player_id", "passer_player_id", "rush_touchdown", "pass_touchdown",
            "two_point_attempt", "air_yards", "qb_dropback"]


def pbp_player_games(pbp):
    """Red-zone and goal-line work per player-game, and per team-game."""
    p = pbp[(pbp.season_type == "REG") & pbp.play_type.isin(["run", "pass"]) & (pbp.two_point_attempt != 1)].copy().reset_index(drop=True)
    y = p.yardline_100
    run = p[p.play_type == "run"]
    pas = p[(p.play_type == "pass") & p.receiver_player_id.notna()]
    out = []
    for lab, lim in (("rz", 20), ("g10", 10), ("g5", 5)):
        r = run[run.yardline_100 <= lim].groupby(["game_id", "posteam", "rusher_player_id"]).size().rename(f"{lab}_car")
        t = pas[pas.yardline_100 <= lim].groupby(["game_id", "posteam", "receiver_player_id"]).size().rename(f"{lab}_tgt")
        r.index.names = t.index.names = ["game_id", "team", "player_id"]
        out += [r, t]
    ez = pas[pas.air_yards.notna() & (pas.air_yards >= pas.yardline_100)].groupby(["game_id", "posteam", "receiver_player_id"]).size().rename("ez_tgt")
    ez.index.names = ["game_id", "team", "player_id"]
    out.append(ez)
    pg = pd.concat(out, axis=1).fillna(0).reset_index()
    # team chances near the end zone and quarterback red-zone passing
    p = p.reset_index(drop=True)
    p["is_rz_run"] = ((p.yardline_100 <= 20) & (p.play_type == "run")).astype(int)
    p["is_rz_pass"] = ((p.yardline_100 <= 20) & (p.play_type == "pass")).astype(int)
    p["is_g10"] = (p.yardline_100 <= 10).astype(int)
    tm = p.groupby(["game_id", "posteam"]).agg(t_rz_run=("is_rz_run", "sum"), t_rz_pass=("is_rz_pass", "sum"), t_g10=("is_g10", "sum")).reset_index().rename(columns={"posteam": "team"})
    qb = p[(p.play_type == "pass") & p.passer_player_id.notna()]
    qrz = qb[qb.yardline_100 <= 20].groupby(["game_id", "passer_player_id"]).agg(qb_rz_att=("play_type", "size"), qb_rz_td=("pass_touchdown", "sum")).reset_index().rename(columns={"passer_player_id": "player_id"})
    return pg, tm, qrz


def build_base(stats, pbp, sched):
    """One row per player-game played, with results and red-zone work."""
    s = stats[(stats.season_type == "REG") & stats.position.isin(SK)].copy()
    s["any_td"] = ((s.rushing_tds.fillna(0) + s.receiving_tds.fillna(0)) > 0).astype(float)
    s["tds"] = s.rushing_tds.fillna(0) + s.receiving_tds.fillna(0)
    pg, tm, qrz = pbp_player_games(pbp)
    s = s.merge(pg, on=["game_id", "team", "player_id"], how="left").merge(qrz, on=["game_id", "player_id"], how="left")
    for c in ["rz_car", "rz_tgt", "g10_car", "g10_tgt", "g5_car", "g5_tgt", "ez_tgt", "qb_rz_att", "qb_rz_td"]:
        s[c] = s[c].fillna(0)
    team = s.groupby(["game_id", "team"]).agg(t_car=("carries", "sum"), t_tgt=("targets", "sum"), t_att=("attempts", "sum"),
                                              t_rush_td=("rushing_tds", "sum"), t_rec_td=("receiving_tds", "sum"),
                                              t_rz_car=("rz_car", "sum"), t_rz_tgt=("rz_tgt", "sum"),
                                              t_g10_car=("g10_car", "sum"), t_g10_tgt=("g10_tgt", "sum")).reset_index()
    team = team.merge(tm, on=["game_id", "team"], how="left")
    s = s.merge(team, on=["game_id", "team"], how="left")
    s = add_games(s, sched)
    return s


def add_games(s, sched):
    g = sched[sched.game_type == "REG"][["game_id", "home_team", "away_team", "spread_line", "total_line", "roof"]]
    s = s.drop(columns=[c for c in ["home_team", "away_team", "spread_line", "total_line", "roof"] if c in s.columns]).merge(g, on="game_id", how="left")
    home = s.team == s.home_team
    s["home"] = home.astype(float)
    s["spread_for"] = np.where(home, s.spread_line, -s.spread_line)  # positive = this team favored
    s["itt"] = s.total_line / 2 + s.spread_for / 2
    s["dome"] = s.roof.isin(["dome", "closed"]).astype(float)
    return s


def _recent(d, keys, cols, hl=HL):
    """Recency-weighted average over earlier rows of each group (current row excluded)."""
    e = d.groupby(keys, sort=False)[cols].ewm(halflife=hl, min_periods=1).mean().reset_index(level=list(range(len(keys))), drop=True)
    e = e.reindex(d.index)
    return e.groupby([d[k] for k in keys], sort=False).shift(1)


def features(base):
    """Pre-game features. Rows with no result yet (upcoming games) get features too."""
    d = base.sort_values(["season", "week"], kind="mergesort").reset_index(drop=True).copy()
    d["rz_car_sh"] = d.rz_car / d.t_rz_car.where(d.t_rz_car > 0)
    d["rz_tgt_sh"] = d.rz_tgt / d.t_rz_tgt.where(d.t_rz_tgt > 0)
    d["g10_sh"] = (d.g10_car + d.g10_tgt) / (d.t_g10_car + d.t_g10_tgt).where((d.t_g10_car + d.t_g10_tgt) > 0)
    d["car_sh"] = d.carries / d.t_car.where(d.t_car > 0)
    d["tgt_sh"] = d.target_share
    d["ypg"] = d.rushing_yards + d.receiving_yards
    d["pass_td_rate"] = d.passing_tds / d.attempts.where(d.attempts > 0)
    d["qb_rz_td_rate"] = d.qb_rz_td / d.qb_rz_att.where(d.qb_rz_att > 0)
    pcols = ["rz_car", "rz_tgt", "g10_car", "g10_tgt", "g5_car", "g5_tgt", "ez_tgt", "rz_car_sh", "rz_tgt_sh", "g10_sh",
             "car_sh", "tgt_sh", "carries", "targets", "ypg", "tds", "any_td", "passing_tds", "attempts", "pass_td_rate",
             "qb_rz_att", "qb_rz_td_rate", "rushing_tds"]
    played = d.result_known == 1
    e = _recent(d, ["player_id", "season"], pcols)
    for c in pcols:
        d["e_" + c] = e[c].values
    g = d.groupby(["player_id", "season"], sort=False)
    d["n_games"] = g.cumcount()
    d["season_tds"] = g.tds.transform(lambda x: x.fillna(0).cumsum().shift(1)).fillna(0)
    d["qb_last3_td"] = g.passing_tds.transform(lambda x: x.fillna(0).rolling(3, min_periods=1).mean().shift(1))
    # last season's per-game rates
    hist = d[played]
    prev = hist.groupby(["player_id", "season"]).agg(p_g=("week", "count"), p_tds=("tds", "mean"), p_any=("any_td", "mean"),
                                                     p_rz_car=("rz_car", "mean"), p_rz_tgt=("rz_tgt", "mean"),
                                                     p_rz_car_sh=("rz_car_sh", "mean"), p_rz_tgt_sh=("rz_tgt_sh", "mean"),
                                                     p_pass_td=("passing_tds", "mean"), p_att=("attempts", "mean")).reset_index()
    prev["season"] += 1
    d = d.merge(prev, on=["player_id", "season"], how="left")
    d["p_g"] = d.p_g.fillna(0)
    # team scoring trends (earlier games this season)
    tg = d.drop_duplicates(["game_id", "team"])[["game_id", "team", "season", "week", "t_rush_td", "t_rec_td", "t_rz_run", "t_rz_pass", "t_g10", "t_att", "t_car"]].copy()
    tg = tg.sort_values(["season", "week"], kind="mergesort").reset_index(drop=True)
    tg["t_td"] = tg.t_rush_td + tg.t_rec_td
    tg["t_pass_td_frac"] = tg.t_rec_td / tg.t_td.where(tg.t_td > 0)
    tg["t_rz_pass_rate"] = tg.t_rz_pass / (tg.t_rz_pass + tg.t_rz_run).where((tg.t_rz_pass + tg.t_rz_run) > 0)
    tg["t_rz_plays"] = tg.t_rz_pass + tg.t_rz_run
    tcols = ["t_td", "t_pass_td_frac", "t_rz_pass_rate", "t_rz_plays", "t_g10", "t_att", "t_car"]
    tw = _recent(tg, ["team", "season"], tcols)
    for c in tcols:
        tg["te_" + c] = tw[c].values
    d = d.merge(tg[["game_id", "team"] + ["te_" + c for c in tcols]], on=["game_id", "team"], how="left")
    # opponent touchdowns allowed per game, by position (earlier games this season)
    og = d[played].groupby(["game_id", "opponent_team", "season", "week", "position"]).tds.sum().unstack("position", fill_value=0).reset_index()
    og.columns.name = None
    for k in SK:
        if k not in og:
            og[k] = 0.0
    og = og.rename(columns={k: "allow_" + k for k in SK})
    up = d[~played][["game_id", "opponent_team", "season", "week"]].drop_duplicates()
    ocols = ["allow_" + k for k in SK]
    og = pd.concat([og, up.assign(**{c: np.nan for c in ocols})], ignore_index=True)
    og = og.sort_values(["season", "week"], kind="mergesort").reset_index(drop=True)
    ow = _recent(og, ["opponent_team", "season"], ocols, hl=4)
    for c in ocols:
        og["o_" + c] = ow[c].values
    og = og.drop_duplicates(["game_id", "opponent_team"], keep="last")
    d = d.merge(og[["game_id", "opponent_team"] + ["o_" + c for c in ocols]], on=["game_id", "opponent_team"], how="left")
    d["opp_allow_pos"] = np.select([d.position == k for k in SK], [d["o_allow_" + k] for k in SK], np.nan)
    d["opp_allow_all"] = d[["o_allow_" + k for k in SK]].sum(axis=1, min_count=1)
    # starting quarterback's touchdown-pass trends, attached to every teammate
    qb = d[(d.position == "QB") & (d.is_starter == 1)][["game_id", "team", "player_id", "e_pass_td_rate", "e_passing_tds", "e_attempts", "e_qb_rz_td_rate", "n_games", "p_pass_td", "p_att", "qb_last3_td"]]
    qb = qb.rename(columns={"player_id": "qb_id", "e_pass_td_rate": "qb_td_rate", "e_passing_tds": "qb_td_pg", "e_attempts": "qb_att_pg",
                            "e_qb_rz_td_rate": "qb_rz_rate", "n_games": "qb_n", "p_pass_td": "qb_p_td", "p_att": "qb_p_att", "qb_last3_td": "qb_l3"}).drop_duplicates(["game_id", "team"])
    d = d.merge(qb, on=["game_id", "team"], how="left")
    d = vacated(d)
    for c in ["home", "dome"]:
        d[c] = d[c].astype(float)
    d["pos_qb"], d["pos_rb"], d["pos_wr"], d["pos_te"] = [(d.position == k).astype(float) for k in SK]
    return d


def vacated(d):
    """Sum of red-zone shares held by teammates who played in the team's last two games but are not in this one."""
    d = d.copy()
    tw = d[["team", "season", "week"]].drop_duplicates().sort_values(["team", "season", "week"])
    tw["prev1"] = tw.groupby(["team", "season"]).week.shift(1)
    tw["prev2"] = tw.groupby(["team", "season"]).week.shift(2)
    roster = d[["team", "season", "week", "player_id", "e_rz_car_sh", "e_rz_tgt_sh"]]
    latest = d[["team", "season", "week", "player_id", "rz_car_sh", "rz_tgt_sh"]].copy()
    rows = []
    present = roster.groupby(["team", "season", "week"]).player_id.apply(set).to_dict()
    by_week = {k: v for k, v in roster.groupby(["team", "season", "week"])}
    for r in tw.itertuples():
        here = present.get((r.team, r.season, r.week), set())
        prior = [by_week.get((r.team, r.season, w)) for w in (r.prev2, r.prev1) if pd.notna(w)]
        prior = [p for p in prior if p is not None]
        if not prior:
            rows.append((r.team, r.season, r.week, 0.0, 0.0)); continue
        rec = pd.concat(prior).drop_duplicates("player_id", keep="last")
        gone = rec[~rec.player_id.isin(here)]
        # their share going into their last game, updated with that game
        vc = gone.e_rz_car_sh.fillna(0); vt = gone.e_rz_tgt_sh.fillna(0)
        rows.append((r.team, r.season, r.week, float(vc[vc >= 0.1].sum()), float(vt[vt >= 0.08].sum())))
    v = pd.DataFrame(rows, columns=["team", "season", "week", "vac_rz_car", "vac_rz_tgt"])
    return d.drop(columns=[c for c in ["vac_rz_car", "vac_rz_tgt"] if c in d.columns]).merge(v, on=["team", "season", "week"], how="left")


FEATS = ["e_rz_car", "e_rz_tgt", "e_g10_car", "e_g10_tgt", "e_g5_car", "e_g5_tgt", "e_ez_tgt", "e_rz_car_sh", "e_rz_tgt_sh",
         "e_g10_sh", "e_car_sh", "e_tgt_sh", "e_carries", "e_targets", "e_ypg", "e_tds", "e_any_td", "e_rushing_tds", "n_games", "season_tds",
         "p_g", "p_tds", "p_any", "p_rz_car", "p_rz_tgt", "p_rz_car_sh", "p_rz_tgt_sh",
         "te_t_td", "te_t_pass_td_frac", "te_t_rz_pass_rate", "te_t_rz_plays", "te_t_g10", "te_t_att", "te_t_car",
         "itt", "total_line", "spread_for", "home", "dome", "opp_allow_pos", "opp_allow_all",
         "qb_td_rate", "qb_td_pg", "qb_att_pg", "qb_rz_rate", "qb_n", "qb_p_td", "qb_p_att",
         "vac_rz_car", "vac_rz_tgt", "qb_l3", "pos_qb", "pos_rb", "pos_wr", "pos_te", "week"]

QB_FEATS = ["e_passing_tds", "e_attempts", "e_pass_td_rate", "e_qb_rz_att", "e_qb_rz_td_rate", "qb_last3_td", "n_games", "p_pass_td", "p_att", "p_g",
            "te_t_td", "te_t_pass_td_frac", "te_t_rz_pass_rate", "te_t_rz_plays", "te_t_att", "te_t_car",
            "itt", "total_line", "spread_for", "home", "dome", "o_allow_WR", "o_allow_TE", "o_allow_RB", "opp_allow_all", "week"]
