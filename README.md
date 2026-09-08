# Intel Suite by Umbra Tech

Free, browser-based screens for markets and player props. Every tool is a single HTML file with no build step and no server: open it, and it pulls live data from public APIs. Watchlists, alerts, and any API keys are stored in your browser's local storage only.

Live site: https://umbra-techllc.github.io/intel-suite/

## Tools

| Tool | File | What it does |
|---|---|---|
| Crypto Intel | `crypto_intel_tracker.html` | Watchlist screener. Scores each coin 0 to 100 from momentum, trend, strength vs BTC, liquidity, ATH distance, and hourly RSI, then gives a Buy, Hold, or Sell signal with the breakdown shown. Market tape, price alerts, auto refresh, coin search, headlines, and an optional Umbra Read. |
| Defense Intel | `defense_stock_tracker.html` | Small and mid-cap defense stock tracker with 52-week range, volume signals, multi-day returns, news, verdicts, and an ITA ETF holdings explorer. |
| NBA Parlay Finder | `nba_parlay_finder.html` | Compares last-five averages against prop lines across points, rebounds, assists, threes, and PRA. Value scoring, ESPN box scores, parlay builder. |
| WNBA Parlay Finder | `wnba_parlay_finder.html` | Same approach for the WNBA with a recency-weighted stats engine and a composite Play Score per prop. |

## Data sources

CoinGecko, alternative.me (Fear & Greed), Google News via rss2json, ESPN, FanDuel lines. Optional keys: a free CoinGecko demo key lifts rate limits; an Anthropic key powers the Umbra Read feature.

## Running locally

Clone the repo and open any `.html` file in a browser. No dependencies.

## Disclaimer

For information and entertainment only. Nothing here is financial or betting advice.

Umbra Technologies LLC
