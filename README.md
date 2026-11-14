# Intel Suite by Umbra Tech

Free, browser-based screens for markets and player props. Every tool is a single HTML file with no build step and no server: open it, and it pulls live data from public APIs. Watchlists, alerts, and any API keys are stored in your browser's local storage only.

Live site: https://umbra-techllc.github.io/intel-suite/

## Tools

| Tool | File | What it does |
|---|---|---|
| Crypto Intel | `crypto_intel_tracker.html` | Watchlist screener. Scores each coin 0 to 100 from momentum, trend, strength vs BTC, liquidity, ATH distance, and hourly RSI, then gives a Buy, Hold, or Sell signal with the breakdown shown. Market tape, price alerts, auto refresh, coin search, headlines, and an optional Umbra Read. |
| Defense Intel | `defense_stock_tracker.html` | Tracks 61 defense, aerospace and space stocks (every ITA ETF holding plus newer space, drone and government tech names). Each gets a 0 to 100 score from price trend, recent performance, performance against the defense ETF, how stretched the run is, trading activity and analyst ratings, with Buy, Hold or Sell shown from it. Plain-English Umbra Read notes on the whole sector and on every stock, group performance, leaders and laggers, upcoming earnings, price charts with 50 and 200-day averages, a filtered sector news feed, watchlist, notes and price alerts. Data is rebuilt automatically (see below). |
| NFL PropEdge | `nfl_propedge.html` | Projects passing, rushing, and receiving yards for the week's slate with gradient-boosted models trained on every player-game since 2019 (about 45 pre-game inputs, including confirmed injuries and backups' bigger roles), with a per-player outcome range and a factor breakdown in yards. Two views: Most likely (default) shows each player's most likely result, an 80% floor, and the bet most likely to hit, using alternate lines and a range anchored to the sportsbook's main line; Best value shows where the projection beats the book's price. Parlay chances are simulated with a Gaussian copula using measured correlations. Parlay builder with measured same-game correlations, live lines via The Odds API, and a plain-English Umbra Read. The slate is rebuilt automatically (see below). |
| NFL TD Predictor | `nfl_td_predictor.html` | Each player's chance to score a touchdown and each starting quarterback's expected touchdown passes (1+, 2+, 3+), from gradient-boosted models trained on every player-game since 2019. Built on red-zone and goal-line usage from play-by-play, team expected scoring, quarterback touchdown-pass trends, opponent and injuries. Touchdown parlay with linked-leg math. Rebuilt automatically (see below). |
| NBA Parlay Finder | `nba_parlay_finder.html` | Projects points, rebounds, assists, threes and the combined stats for every player on tonight's slate with gradient-boosted models trained on every player-game since October 2024 (about 70 pre-game inputs, including who is out and how that shifts minutes). Each projection is blended with the DraftKings line to give a chance for every posted bet: overs, unders and "this many or more". Umbra Read then chooses and orders the parlay legs from the strongest candidates, with a reason and the main risk for each. Parlay chances account for legs that move together. Each day's picks are saved before tip-off and graded from the box scores. Rebuilt automatically (see below). |
| WNBA Parlay Finder | `wnba_parlay_finder.html` | Same approach for the WNBA with a recency-weighted stats engine and a composite Play Score per prop. |

## Data sources

CoinGecko, alternative.me (Fear & Greed), Yahoo Finance, Google News, ESPN, DraftKings and FanDuel lines, nflverse. Optional keys: a free CoinGecko demo key lifts rate limits; an Anthropic key powers the Umbra Read feature; a key from The Odds API loads live NFL prop lines.

## Defense Intel refresh

`defense_data.json` holds a year of daily prices, quote details, scores, headlines and the written reads for every stock on the page. The `Defense Intel refresh` workflow rebuilds it about every hour during market hours on weekdays, plus once on Saturday. The written reads are redone before the open, after the close, and on Saturday; that step needs an `ANTHROPIC_API_KEY` repository secret (Settings > Secrets and variables > Actions). Without the secret, prices and news still refresh and the page shows the last reads with the time they were written. By hand:

```
cd defense
pip install requests
python build_defense.py                 # prices, news, and new reads if the saved ones are old
AI_MODE=force python build_defense.py   # rewrite the reads now
AI_MODE=off python build_defense.py     # prices and news only
```

## NFL PropEdge refresh

`nfl_propedge.html` has the week's slate, stats, odds, and injuries built into it. The `NFL PropEdge refresh` workflow in `.github/workflows/` rebuilds it every morning and evening, and again late Sunday morning after inactives are announced. It also publishes `nfl_propedge_data.json`, which the page's **Check for new slate** button reads, so a saved copy of the page can pull the newest week too. To rebuild by hand, open the Actions tab, choose the workflow, and click Run workflow. The build scripts live in `nfl-propedge/`:

```
cd nfl-propedge
pip install pandas scipy requests
python build_data.py            # current week, detected from ESPN
python make_page.py data.json --site ../nfl_propedge.html
```

The models in `nfl-propedge/model/` are retrained every Tuesday by the `NFL PropEdge retrain` workflow (`research/make_dataset.py`, then `research/fit_models.py`). `research/train.py`, `research/score_probs.py` and `research/check2024.py` reproduce the held-out comparisons against the earlier engine. To re-measure the older outcome ranges and injury rules: `python backtest.py 2024`, `python backtest.py 2025`, then `python research/fit_calibration.py` (writes `calibration.json`).

## NFL TD Predictor refresh

`nfl_td_predictor.html` and `nfl_td_data.json` are rebuilt by the `NFL TD Predictor refresh` workflow (twice a day, plus Sunday after inactives), and the models in `nfl-td/model/` are retrained every Tuesday by `NFL TD Predictor retrain`. By hand:

```
cd nfl-td
pip install pandas numpy pyarrow lightgbm requests scikit-learn scipy
python build_td.py                                   # this week's predictions
python make_page.py data.json --site ../nfl_td_predictor.html
python research/build_dataset.py && python research/fit_td_models.py   # retrain
python research/train_td.py                          # held-out tests against simpler approaches
```

## NBA Parlay Finder refresh

`nba_parlay_finder.html` is a fixed page that loads `nba_parlay_data.json`. The `NBA Parlay Finder refresh` workflow rebuilds that file four times a day (about 11am, 2pm, 4:30pm and 6pm Eastern). The step where Umbra Read picks the legs needs the `ANTHROPIC_API_KEY` repository secret, the same one Defense Intel uses. Without it the board still updates and the picks fall back to the model's own ranking. No odds key is needed: lines come from the DraftKings numbers ESPN publishes. The models in `nba-propedge/model/` are retrained every Monday by `NBA Parlay Finder retrain`, which also adds the week's games to `nba-propedge/data/`. By hand:

```
cd nba-propedge
pip install pandas numpy requests lightgbm scikit-learn
python build_nba.py                    # tonight's board (set ANTHROPIC_API_KEY for the AI pick, or add --no-ai)
python research/make_dataset.py        # add finished games to the history
cd research && python fit_models.py    # retrain and rewrite the accuracy report
python train.py                        # held-out comparison against simple averages
python market.py                       # held-out comparison against DraftKings opening lines
```

Because the page reads a data file, open it from the site or a local web server (`python -m http.server`), not straight from disk.

## Running locally

Clone the repo and open any `.html` file in a browser. No dependencies.

## Disclaimer

For information and entertainment only. Nothing here is financial or betting advice.

Umbra Technologies LLC
