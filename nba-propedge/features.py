"""Turns box-score history into the pre-game inputs the models use.

One code path for training and for the daily build: the daily build appends a row for every player
expected to play tonight (stats blank, future=1) and calls the same function, so tonight's inputs are
computed exactly the way the training inputs were. Every input only looks at games before the one it describes.
"""
import numpy as np
import pandas as pd

TARGETS = ["pts", "reb", "ast", "tpm", "pra", "pr", "pa", "ra"]
BASE = ["min", "pts", "reb", "ast", "tpm", "fga", "tpa", "fta", "tov", "oreb"]
POSG = {"G": "G", "PG": "G", "SG": "G", "F": "F", "SF": "F", "PF": "F", "C": "C"}


def _ewm_prev(g, cols, hl):
    """Recency-weighted average of earlier games for each player (the current game is excluded)."""
    return g[cols].transform(lambda s: s.shift(1).ewm(halflife=hl, min_periods=1).mean())


def build(df):
    df = df.copy()
    if "future" not in df:
        df["future"] = 0
    df["future"] = df["future"].fillna(0).astype(int)
    df["t"] = pd.to_datetime(df["date"], utc=True)
    df["posg"] = df["pos"].map(POSG).fillna("F")
    for c in ("pid", "tid", "oid", "gid"):
        df[c] = df[c].astype(str)
    df["pra"] = df.pts + df.reb + df.ast
    df["pr"] = df.pts + df.reb
    df["pa"] = df.pts + df.ast
    df["ra"] = df.reb + df.ast
    df = df.sort_values(["t", "gid", "tid", "pid"]).reset_index(drop=True)

    # ---- team-game table: pace, scoring, and what each defense gives up (overall and by position group)
    pl = df[(df["min"] > 0) | (df.future == 1)]
    tg = pl.groupby(["gid", "tid", "oid", "t"], as_index=False).agg(
        fga=("fga", "sum"), fta=("fta", "sum"), oreb=("oreb", "sum"), tov=("tov", "sum"),
        t_pts=("pts", "sum"), t_reb=("reb", "sum"), t_ast=("ast", "sum"), t_tpm=("tpm", "sum"), fut=("future", "max"))
    tg["poss"] = tg.fga + 0.44 * tg.fta - tg.oreb + tg.tov
    tg.loc[tg.fut == 1, ["poss", "t_pts", "t_reb", "t_ast", "t_tpm"]] = np.nan
    for s in ("pts", "reb", "ast", "tpm"):
        by = pl.pivot_table(index=["gid", "tid"], columns="posg", values=s, aggfunc="sum").reindex(columns=["G", "F", "C"]).fillna(0)
        by.columns = [f"{s}_{c}" for c in by.columns]
        tg = tg.merge(by.reset_index(), on=["gid", "tid"], how="left")
    posg_cols = [f"{s}_{c}" for s in ("pts", "reb", "ast", "tpm") for c in ("G", "F", "C")]
    tg.loc[tg.fut == 1, posg_cols] = np.nan
    tg = tg.sort_values(["t", "gid"]).reset_index(drop=True)
    own = ["poss", "t_pts", "t_reb", "t_ast", "t_tpm"]
    gt = tg.groupby("tid")
    for c in own:
        tg[f"tm_{c}"] = gt[c].transform(lambda s: s.shift(1).ewm(halflife=8, min_periods=1).mean())
    # what each team allowed = the opponent's production in the same game
    opp = tg[["gid", "tid"] + own + posg_cols].rename(columns={"tid": "oid", **{c: f"a_{c}" for c in own + posg_cols}})
    tg = tg.merge(opp, on=["gid", "oid"], how="left")
    gt = tg.groupby("tid")
    for c in own + posg_cols:
        tg[f"def_{c}"] = gt[f"a_{c}"].transform(lambda s: s.shift(1).ewm(halflife=10, min_periods=1).mean())
    tg["tgames"] = gt.cumcount()
    tm_cols = [f"tm_{c}" for c in own]
    def_cols = [f"def_{c}" for c in own + posg_cols]

    # ---- player form going into each game
    p = df[(df["min"] > 0) | (df.future == 1)].copy()
    g = p.groupby("pid", sort=False)
    for hl, tag in ((2, "s"), (6, "m"), (20, "l")):
        e = _ewm_prev(g, BASE, hl)
        e.columns = [f"{c}_{tag}" for c in BASE]
        p = pd.concat([p, e], axis=1)
    g = p.groupby("pid", sort=False)
    for c in ("min", "pts", "reb", "ast", "tpm"):
        p[f"{c}_sd"] = g[c].transform(lambda s: s.shift(1).rolling(10, min_periods=3).std())
    p["last_min"] = g["min"].shift(1)
    p["start_m"] = g["starter"].transform(lambda s: s.shift(1).ewm(halflife=3, min_periods=1).mean())
    p["n_prev"] = g.cumcount()
    p["rest"] = (p.t - g["t"].shift(1)).dt.total_seconds().div(86400).clip(upper=10)
    p["b2b"] = (p.rest < 1.6).astype(int)
    for c in ("pts", "reb", "ast", "tpm", "fga"):
        p[f"{c}_pm"] = p[f"{c}_m"] / p["min_m"].clip(lower=4)
    # state after each game (used to size up a teammate who then sits out)
    for c in ("min", "pts", "reb", "ast", "fga"):
        p[f"{c}_post"] = g[c].transform(lambda s: s.ewm(halflife=6, min_periods=1).mean())

    # ---- teammates missing tonight who were playing lately
    miss_rows = []
    tgi = {tid: d.reset_index(drop=True) for tid, d in tg.groupby("tid")}
    played = p[p.future == 0].groupby(["tid", "gid"])["pid"].apply(set).to_dict()
    listed = p.groupby(["tid", "gid"])["pid"].apply(set).to_dict()
    post = p[p.future == 0].set_index(["pid", "gid"])[["min_post", "pts_post", "reb_post", "ast_post", "fga_post", "posg"]]
    for tid, d in tgi.items():
        gids, times = d.gid.tolist(), d.t.tolist()
        last_seen = {}  # pid -> index of the last team game he played
        for i, gid in enumerate(gids):
            if i and (times[i] - times[i - 1]).days > 45:
                last_seen = {}  # new season: last year's roster says nothing about who is missing
            here = listed.get((tid, gid), set())
            acc = {"n_out": 0, "miss_min": 0.0, "miss_pts": 0.0, "miss_reb": 0.0, "miss_ast": 0.0, "miss_fga": 0.0,
                   "new_min": 0.0, "new_pts": 0.0, "miss_min_G": 0.0, "miss_min_F": 0.0, "miss_min_C": 0.0}
            for pid, j in last_seen.items():
                gap = i - j - 1  # team games already missed before this one
                if pid in here or gap > 6:
                    continue
                r = post.loc[(pid, gids[j])]
                if r.min_post < 10:
                    continue
                w = 0.7 ** gap
                acc["n_out"] += 1
                for c in ("min", "pts", "reb", "ast", "fga"):
                    acc[f"miss_{c}"] += r[f"{c}_post"]
                acc["new_min"] += w * r.min_post
                acc["new_pts"] += w * r.pts_post
                acc[f"miss_min_{r.posg}"] += r.min_post
            miss_rows.append({"tid": tid, "gid": gid, **acc})
            for pid in played.get((tid, gid), ()):
                last_seen[pid] = i
    miss = pd.DataFrame(miss_rows)

    p = p.merge(tg[["gid", "tid", "tgames"] + tm_cols], on=["gid", "tid"], how="left")
    opp_side = tg[["gid", "tid"] + tm_cols + def_cols].rename(
        columns={"tid": "oid", **{c: "opp_" + c for c in tm_cols + def_cols}})
    p = p.merge(opp_side, on=["gid", "oid"], how="left")
    p = p.merge(miss, on=["gid", "tid"], how="left")
    # opponent's defense against this player's position group
    for s in ("pts", "reb", "ast", "tpm"):
        p[f"dvp_{s}"] = np.select([p.posg == "G", p.posg == "F"], [p[f"opp_def_{s}_G"], p[f"opp_def_{s}_F"]], p[f"opp_def_{s}_C"])
    p["miss_min_pos"] = np.select([p.posg == "G", p.posg == "F"], [p.miss_min_G, p.miss_min_F], p.miss_min_C)
    p["own_total"] = p.total / 2 - p.spread / 2
    p["abs_spread"] = p.spread.abs()
    p["pace_pair"] = (p.tm_poss + p.opp_tm_poss) / 2
    p["is_G"] = (p.posg == "G").astype(int)
    p["is_C"] = (p.posg == "C").astype(int)
    return p


FEATS = (
    [f"{c}_{t}" for t in ("s", "m", "l") for c in BASE]
    + ["min_sd", "pts_sd", "reb_sd", "ast_sd", "tpm_sd", "last_min", "start_m", "n_prev", "rest", "b2b", "home",
       "pts_pm", "reb_pm", "ast_pm", "tpm_pm", "fga_pm",
       "tm_poss", "tm_t_pts", "opp_tm_poss", "pace_pair",
       "opp_def_t_pts", "opp_def_t_reb", "opp_def_t_ast", "opp_def_t_tpm", "opp_def_poss",
       "dvp_pts", "dvp_reb", "dvp_ast", "dvp_tpm",
       "n_out", "miss_min", "miss_pts", "miss_reb", "miss_ast", "miss_fga", "new_min", "new_pts", "miss_min_pos",
       "total", "spread", "own_total", "abs_spread", "is_G", "is_C"]
)

# plain-English names for the page and for the AI step
LABELS = {
    "min": "minutes", "pts": "points", "reb": "rebounds", "ast": "assists", "tpm": "three-pointers made",
    "pra": "points + rebounds + assists", "pr": "points + rebounds", "pa": "points + assists", "ra": "rebounds + assists",
}
