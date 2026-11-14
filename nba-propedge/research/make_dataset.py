"""Build or top up the history the models learn from.

  data/games_<season>.csv.gz   one row per player per game (box score line, plus the game's betting total and spread)
  data/lines_<season>.csv.gz   the sportsbook's opening over/under for each player and stat, where ESPN has it

Run with no arguments to add finished games that are missing. --since YYYYMMDD starts a fresh pull from that date.
"""
import argparse, os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dataset

ap = argparse.ArgumentParser()
ap.add_argument("--since")
args = ap.parse_args()
games = pd.DataFrame() if args.since else dataset.load_games()
lines = pd.DataFrame() if args.since else dataset.load_lines()
g, l = dataset.fetch_new(games, since=args.since)
games = pd.concat([games, g], ignore_index=True) if len(g) else games
lines = pd.concat([lines, l], ignore_index=True) if len(l) else lines
dataset.save(games, lines)
print(f"{len(games)} player rows from {games.gid.nunique()} games | {len(lines)} opening lines")
