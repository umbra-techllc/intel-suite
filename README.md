# Intel Suite by Umbra Tech

Free, browser-based screens for markets and player props. Every tool is a single HTML file with no build step and no server: open it, and it pulls live data from public APIs. Watchlists, alerts, and any API keys are stored in your browser's local storage only.

Live site: https://umbra-techllc.github.io/intel-suite/

## Tools

| Tool | File | What it does |
|---|---|---|
| Crypto Intel | `crypto_intel_tracker.html` | Watchlist screener. Scores each coin 0 to 100 from momentum, trend, strength vs BTC, liquidity, ATH distance, and hourly RSI, then gives a Buy, Hold, or Sell signal with the breakdown shown. Market tape, price alerts, auto refresh, coin search, headlines, and an optional Umbra Read. |
| Defense Intel | `defense_stock_tracker.html` | Small and mid-cap defense stock tracker with 52-week range, volume signals, multi-day returns, news, verdicts, and an ITA ETF holdings explorer. |
| NFL PropEdge | `nfl_propedge.html` | Projects passing, rushing, and receiving yards for the week's slate with gradient-boosted models trained on every player-game since 2019 (about 45 pre-game inputs, including confirmed injuries and backups' bigger roles), with a per-player outcome range and a factor breakdown in yards. Over/under chances are measured on the 2024 and 2025 seasons and blended with the sportsbook line; picks only appear on real lines. Parlay builder with measured same-game correlations, live lines via The Odds API, and a plain-English Umbra Read. The slate is rebuilt automatically (see below). |
| NBA Parlay Finder | `nba_parlay_finder.html` | Compares last-five averages against prop lines across points, rebounds, assists, threes, and PRA. Value scoring, ESPN box scores, parlay builder. |
| WNBA Parlay Finder | `wnba_parlay_finder.html` | Same approach for the WNBA with a recency-weighted stats engine and a composite Play Score per prop. |

## Data sources

CoinGecko, alternative.me (Fear & Greed), Google News via rss2json, ESPN, FanDuel lines, nflverse. Optional keys: a free CoinGecko demo key lifts rate limits; an Anthropic key powers the Umbra Read feature; a key from The Odds API loads live NFL prop lines.

## NFL PropEdge refresh

`nfl_propedge.html` has the week's slate, stats, odds, and injuries built into it. The `NFL PropEdge refresh` workflow in `.github/workflows/` rebuilds it every morning and evening, and again late Sunday morning after inactives are announced. It also publishes `nfl_propedge_data.json`, which the page's **Check for new slate** button reads, so a saved copy of the page can pull the newest week too. To rebuild by hand, open the Actions tab, choose the workflow, and click Run workflow. The build scripts live in `nfl-propedge/`:

```
cd nfl-propedge
pip install pandas scipy requests
python build_data.py            # current week, detected from ESPN
python make_page.py data.json --site ../nfl_propedge.html
```

The models in `nfl-propedge/model/` are retrained every Tuesday by the `NFL PropEdge retrain` workflow (`research/make_dataset.py`, then `research/fit_models.py`). `research/train.py`, `research/score_probs.py` and `research/check2024.py` reproduce the held-out comparisons against the earlier engine. To re-measure the older outcome ranges and injury rules: `python backtest.py 2024`, `python backtest.py 2025`, then `python research/fit_calibration.py` (writes `calibration.json`).

## Running locally

Clone the repo and open any `.html` file in a browser. No dependencies.

## Disclaimer

For information and entertainment only. Nothing here is financial or betting advice.

Umbra Technologies LLC
