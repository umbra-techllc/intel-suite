"""
Builds defense_data.json for Defense Intel.

Pulls a year of daily prices, quote details and headlines for every stock the
page can show, scores each one 0 to 100, and (when ANTHROPIC_API_KEY is set)
writes plain-English insights for the market and for every stock.

    pip install requests
    python build_defense.py                    # writes ../defense_data.json
    python build_defense.py --out data.json
    AI_MODE=force python build_defense.py      # rewrite the insights now
    AI_MODE=off   python build_defense.py      # prices and news only

AI_MODE=auto (the default) rewrites insights only when the saved ones are more
than AI_MAX_AGE_HOURS old (default 5), so the frequent price refreshes stay cheap.
"""
import argparse
import datetime as dt
import email.utils
import html
import json
import math
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote_plus

import requests

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/129.0 Safari/537.36")
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("AI_MODEL", "claude-sonnet-5-5")

# ---------------------------------------------------------------------------
# Universe. Sector names are plain on purpose; they show up in the page as-is.
# "aka" is how headlines usually refer to the company (for tagging news).
# ---------------------------------------------------------------------------
U = [
    # ticker, name, sector, aka
    ("GE",   "GE Aerospace",            "Engines & parts",      ["GE Aerospace"]),
    ("RTX",  "RTX",                     "Big contractors",      ["RTX", "Raytheon", "Pratt & Whitney"]),
    ("BA",   "Boeing",                  "Big contractors",      ["Boeing"]),
    ("LMT",  "Lockheed Martin",         "Big contractors",      ["Lockheed"]),
    ("NOC",  "Northrop Grumman",        "Big contractors",      ["Northrop"]),
    ("GD",   "General Dynamics",        "Ships & vehicles",     ["General Dynamics", "Gulfstream"]),
    ("LHX",  "L3Harris",                "Electronics & sensors",["L3Harris"]),
    ("HII",  "Huntington Ingalls",      "Ships & vehicles",     ["Huntington Ingalls", "HII"]),
    ("HWM",  "Howmet Aerospace",        "Engines & parts",      ["Howmet"]),
    ("TDG",  "TransDigm",               "Engines & parts",      ["TransDigm"]),
    ("TXT",  "Textron",                 "Big contractors",      ["Textron", "Bell Textron"]),
    ("AXON", "Axon Enterprise",         "Software & data",      ["Axon"]),
    ("PLTR", "Palantir",                "Software & data",      ["Palantir"]),
    ("CACI", "CACI International",      "Software & data",      ["CACI"]),
    ("LDOS", "Leidos",                  "Software & data",      ["Leidos"]),
    ("SAIC", "Science Applications",    "Software & data",      ["SAIC", "Science Applications"]),
    ("BAH",  "Booz Allen Hamilton",     "Software & data",      ["Booz Allen"]),
    ("PSN",  "Parsons",                 "Software & data",      ["Parsons"]),
    ("VVX",  "V2X",                     "Software & data",      ["V2X"]),
    ("BBAI", "BigBear.ai",              "Software & data",      ["BigBear"]),
    ("RKLB", "Rocket Lab",              "Space",                ["Rocket Lab"]),
    ("LUNR", "Intuitive Machines",      "Space",                ["Intuitive Machines"]),
    ("FLY",  "Firefly Aerospace",       "Space",                ["Firefly"]),
    ("VOYG", "Voyager Technologies",    "Space",                ["Voyager Technologies"]),
    ("KRMN", "Karman Holdings",         "Space",                ["Karman"]),
    ("RDW",  "Redwire",                 "Space",                ["Redwire"]),
    ("PL",   "Planet Labs",             "Space",                ["Planet Labs"]),
    ("BKSY", "BlackSky",                "Space",                ["BlackSky"]),
    ("SPIR", "Spire Global",            "Space",                ["Spire Global"]),
    ("ASTS", "AST SpaceMobile",         "Space",                ["AST SpaceMobile"]),
    ("KTOS", "Kratos Defense",          "Drones & autonomy",    ["Kratos"]),
    ("AVAV", "AeroVironment",           "Drones & autonomy",    ["AeroVironment"]),
    ("RCAT", "Red Cat Holdings",        "Drones & autonomy",    ["Red Cat"]),
    ("ONDS", "Ondas Holdings",          "Drones & autonomy",    ["Ondas"]),
    ("UMAC", "Unusual Machines",        "Drones & autonomy",    ["Unusual Machines"]),
    ("AIRO", "AIRO Group",              "Drones & autonomy",    ["AIRO Group"]),
    ("DRS",  "Leonardo DRS",            "Electronics & sensors",["Leonardo DRS"]),
    ("MRCY", "Mercury Systems",         "Electronics & sensors",["Mercury Systems"]),
    ("CW",   "Curtiss-Wright",          "Electronics & sensors",["Curtiss-Wright"]),
    ("OSIS", "OSI Systems",             "Electronics & sensors",["OSI Systems"]),
    ("MOG-A","Moog",                    "Electronics & sensors",["Moog"]),
    ("ATRO", "Astronics",               "Electronics & sensors",["Astronics"]),
    ("BWXT", "BWX Technologies",        "Nuclear",              ["BWX"]),
    ("ACHR", "Archer Aviation",         "Air taxis",            ["Archer Aviation"]),
    ("JOBY", "Joby Aviation",           "Air taxis",            ["Joby"]),
    ("BETA", "Beta Technologies",       "Air taxis",            ["Beta Technologies"]),
    ("FTAI", "FTAI Aviation",           "Engines & parts",      ["FTAI"]),
    ("WWD",  "Woodward",                "Engines & parts",      ["Woodward"]),
    ("ATI",  "ATI",                     "Engines & parts",      ["ATI Inc"]),
    ("CRS",  "Carpenter Technology",    "Engines & parts",      ["Carpenter Technology"]),
    ("HEI",  "HEICO",                   "Engines & parts",      ["HEICO"]),
    ("HXL",  "Hexcel",                  "Engines & parts",      ["Hexcel"]),
    ("SARO", "StandardAero",            "Engines & parts",      ["StandardAero"]),
    ("LOAR", "Loar Holdings",           "Engines & parts",      ["Loar"]),
    ("DCO",  "Ducommun",                "Engines & parts",      ["Ducommun"]),
    ("VSEC", "VSE Corp",                "Engines & parts",      ["VSE Corp"]),
    ("AIR",  "AAR Corp",                "Engines & parts",      ["AAR Corp"]),
    ("CDRE", "Cadre Holdings",          "Firearms & gear",      ["Cadre Holdings"]),
    ("NPK",  "National Presto",         "Firearms & gear",      ["National Presto"]),
    ("RGR",  "Sturm Ruger",             "Firearms & gear",      ["Ruger"]),
    ("SWBI", "Smith & Wesson",          "Firearms & gear",      ["Smith & Wesson"]),
]
BENCH = [("ITA", "Defense ETF (ITA)"), ("SPY", "S&P 500"), ("QQQ", "Nasdaq 100")]

# Last known ITA holdings, used when the iShares file can't be fetched.
ITA_FALLBACK = [("GE",19.00),("RTX",16.44),("BA",8.65),("GD",4.91),("LMT",4.72),("LHX",4.69),("NOC",4.61),
    ("HWM",4.48),("TDG",4.47),("AXON",2.98),("RKLB",2.60),("FTAI",2.06),("CW",2.06),("WWD",1.78),("ATI",1.64),
    ("CRS",1.63),("BWXT",1.57),("HEI",2.27),("TXT",1.30),("HII",1.27),("KTOS",1.17),("MOG-A",0.69),("KRMN",0.55),
    ("AVAV",0.52),("HXL",0.51),("VSEC",0.42),("SARO",0.42),("AIR",0.36),("MRCY",0.34),("DRS",0.30),("ACHR",0.28),
    ("LUNR",0.18),("ATRO",0.17),("LOAR",0.17),("DCO",0.14),("RCAT",0.12),("VVX",0.12),("CDRE",0.07),("NPK",0.06),
    ("RDW",0.06),("RGR",0.06),("SWBI",0.06),("BETA",0.04)]
ITA_FALLBACK_ASOF = "2026-03-31"

EXCLUDE = {"RTX": r"geforce|nvidia|graphics|gpu", "GE": r"vernova|healthcare|appliances", "PSN": r"\bwru\b|rugby|union|hayley",
           "BA": r"\bBAE\b", "HEI": r"heineken", "AIR": r"airline|air india", "FLY": r"\bfly\b(?!.*firefly)", "BETA": r"beta version"}

NEWS_QUERIES = [
    "U.S. defense stocks", "Pentagon contract awarded", "defense budget Congress", "Army contract award",
    "Navy shipbuilding contract", "Space Force launch contract", "drone contract Pentagon",
    "NATO defense spending", "Golden Dome missile defense",
    "Pentagon site:breakingdefense.com", "contract site:defensenews.com", "launch site:spacenews.com",
    "Pentagon site:defensescoop.com", "defense contractor Reuters", "missile defense contract",
]
GOOD_SOURCES = {"reuters", "bloomberg", "the wall street journal", "wsj", "cnbc", "barron's", "barrons.com", "defense news",
    "breaking defense", "defensescoop", "spacenews", "space news", "marketwatch", "associated press", "ap news", "axios",
    "the information", "air & space forces magazine", "aviation week", "flightglobal", "payload", "financial times",
    "investor's business daily", "politico", "defense one", "military.com", "the war zone", "twz", "c4isrnet",
    "national defense magazine", "spacepolicyonline", "nasaspaceflight", "investopedia", "fortune"}
BAD_SOURCES = re.compile(r"ad-hoc-news|cryptorank|tradingview|marketbeat|finance canada|stocks down under|zacks|"
                         r"defenseworld\.net|etf daily news|the globe and mail|investing\.com india|gurufocus|stocktitan|kalkine|traders union|"
                         r"stock traders daily|siam\.in|\.gov\.|scanx|quiver|coincodex|ad hoc news|moomoo|chartmill|benzinga india", re.I)
JUNK = re.compile(r"latest stock news and headlines|interactive stock chart|archives\b|sitemap|\| stock price|latest news \||^[-(]|aktien|prognose|stock price, news, quote|estimates [–-] |tokenized stock|trades at eur|costs eur|stocks to research|"
                  r"stocks? to watch|quote & history|historical prices|\bstock forecast|price target \d+ ?%? upside$|quote price and forecast|"
                  r"price, quote|major holders|beneficial ownership|passive stake|donchian|vwap|risk zones|stock analysis$|"
                  r"(under|out)performs (\w+ )?(when compared to|competitors)|symbol__|declines more than market|gains more than market", re.I)
OPINION = re.compile(r"should you|is (it|now) (a good|the|finally)|undervalued|overvalued|here'?s why|time to buy|buy the dip|"
                     r"better buy|could (soar|double|be)|millionaire|prediction:|how investors may|might change the case|"
                     r"may be fully priced|i remain|my favorite|invested in|worth in \d+ years|key takeaways|whale activity|"
                     r"rotational strategy|\?$", re.I)

SPAM = re.compile(r"(shares (sold|bought|purchased|acquired) by|position in|stake in|holdings (raised|lowered|trimmed|boosted|cut)"
                  r"|stock position|invests? \$|acquires (new )?(shares|stake)|sells \d|buys \d|short interest|"
                  r"unusual options|stock price today|price prediction|\bsells? [\d,]+ shares)", re.I)

S = requests.Session()
S.headers.update({"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"})


def get(url, tries=3, **kw):
    for i in range(tries):
        try:
            r = S.get(url, timeout=20, **kw)
            if r.status_code == 429:
                time.sleep(2 + 3 * i)
                continue
            r.raise_for_status()
            return r
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(1.5 * (i + 1))
    raise RuntimeError("rate limited: " + url)


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------
def ysym(t):
    return t.replace(".", "-")


def chart(t):
    for host in ("query2", "query1"):
        try:
            r = get(f"https://{host}.finance.yahoo.com/v8/finance/chart/{ysym(t)}?interval=1d&range=1y&includePrePost=false")
            res = r.json()["chart"]["result"][0]
            ts = res.get("timestamp") or []
            q = res["indicators"]["quote"][0]
            rows = [(dt.datetime.fromtimestamp(a, dt.timezone.utc).date().isoformat(), c, v)
                    for a, c, v in zip(ts, q.get("close", []), q.get("volume", [])) if c is not None]
            return {"meta": res["meta"], "rows": rows}
        except Exception:
            continue
    return None


def quotes(tickers):
    """Market cap, P/E, earnings date and analyst rating. Needs a cookie crumb."""
    out = {}

    def crumb():
        S.get("https://fc.yahoo.com", timeout=15)
        return S.get("https://query2.finance.yahoo.com/v1/test/getcrumb", timeout=15).text.strip()

    try:
        c = crumb()
    except Exception as e:
        print("quote details unavailable:", e, file=sys.stderr)
        return out
    for i in range(0, len(tickers), 25):
        chunk = ",".join(ysym(t) for t in tickers[i:i + 25])
        for attempt in range(4):
            try:
                r = S.get(f"https://query2.finance.yahoo.com/v7/finance/quote?symbols={chunk}&crumb={quote_plus(c)}", timeout=20)
                if r.status_code in (401, 403):
                    c = crumb(); raise RuntimeError("crumb expired")
                r.raise_for_status()
                for q in r.json()["quoteResponse"]["result"]:
                    out[q["symbol"]] = q
                break
            except Exception as e:
                time.sleep(2 + 2 * attempt)
                if attempt == 3:
                    print("quote chunk failed:", chunk, e, file=sys.stderr)
        time.sleep(0.8)
    return out


def ita_holdings():
    url = ("https://www.ishares.com/us/products/239502/ishares-us-aerospace-defense-etf/"
           "1467271812596.ajax?tab=all&fileType=json")
    try:
        raw = get(url).content.decode("utf-8-sig")
        rows = json.loads(raw)["aaData"]
        eq = []
        for r in rows:
            if r[3] != "Equity":
                continue
            w = r[5]["raw"] if isinstance(r[5], dict) else float(r[5])
            tk = r[0].replace(" ", "-").replace(".", "-")
            if tk == "MOGA":
                tk = "MOG-A"
            eq.append({"t": tk, "name": r[1].title(), "w": round(w, 2)})
        if len(eq) > 20:
            return eq, dt.date.today().isoformat()
    except Exception as e:
        print("iShares holdings unavailable, using saved list:", e, file=sys.stderr)
    return [{"t": t, "w": w} for t, w in ITA_FALLBACK], ITA_FALLBACK_ASOF


# ---------------------------------------------------------------------------
# Metrics and score
# ---------------------------------------------------------------------------
def pct(a, b):
    return None if (a is None or not b) else round((a / b - 1) * 100, 2)


def sma(xs, n):
    return sum(xs[-n:]) / n if len(xs) >= n else None


def rsi(xs, n=14):
    if len(xs) < n + 1:
        return None
    gains = losses = 0.0
    for i in range(1, n + 1):
        d = xs[i] - xs[i - 1]
        gains += max(d, 0); losses += max(-d, 0)
    ag, al = gains / n, losses / n
    for i in range(n + 1, len(xs)):
        d = xs[i] - xs[i - 1]
        ag = (ag * (n - 1) + max(d, 0)) / n
        al = (al * (n - 1) + max(-d, 0)) / n
    return 100.0 if al == 0 else round(100 - 100 / (1 + ag / al), 1)


def clamp(x, lo=0, hi=100):
    return max(lo, min(hi, x))


def lin(x, x0, x1):
    """Map x from [x0, x1] onto [0, 100]."""
    if x is None:
        return 50
    return clamp((x - x0) / (x1 - x0) * 100)


def metrics(rows, meta, q, bench_ret):
    closes = [r[1] for r in rows]
    vols = [r[2] or 0 for r in rows]
    dates = [r[0] for r in rows]
    price = meta.get("regularMarketPrice") or (closes[-1] if closes else None)
    prev = q.get("regularMarketPreviousClose") or (closes[-2] if len(closes) > 1 else None)
    # Use the live price as the last close so returns line up with the quote
    if closes and price:
        closes[-1] = price

    def back(n):
        return closes[-1 - n] if len(closes) > n else None

    year = dates[-1][:4] if dates else None
    ytd_base = None
    for d, c in zip(dates, closes):
        if d[:4] == year:
            break
        ytd_base = c
    ma50, ma200 = sma(closes, 50), sma(closes, 200)
    rets = [math.log(closes[i] / closes[i - 1]) for i in range(max(1, len(closes) - 60), len(closes)) if closes[i - 1]]
    vol_ann = round(math.sqrt(sum(r * r for r in rets) / len(rets)) * math.sqrt(252) * 100, 1) if len(rets) > 10 else None
    hi52 = max(closes) if closes else None
    lo52 = min(closes) if closes else None
    hi52 = max(hi52 or 0, meta.get("fiftyTwoWeekHigh") or 0) or hi52
    lo52 = min(x for x in [lo52, meta.get("fiftyTwoWeekLow")] if x) if (lo52 or meta.get("fiftyTwoWeekLow")) else None
    v5 = sum(vols[-5:]) / 5 if len(vols) >= 5 else None
    v63 = sum(vols[-63:]) / min(63, len(vols)) if vols else None
    rel_vol = round(v5 / v63, 2) if v5 and v63 else None
    up_days = sum(1 for i in range(-5, 0) if len(closes) > 6 and closes[i] > closes[i - 1])

    m = {
        "price": round(price, 4) if price else None,
        "prev": round(prev, 4) if prev else None,
        "chgPct": pct(price, prev),
        "dayHi": meta.get("regularMarketDayHigh"), "dayLo": meta.get("regularMarketDayLow"),
        "vol": meta.get("regularMarketVolume"),
        "r": {"d5": pct(price, back(5)), "m1": pct(price, back(21)), "m3": pct(price, back(63)),
              "m6": pct(price, back(126)), "ytd": pct(price, ytd_base), "y1": pct(price, closes[0] if len(closes) > 200 else None)},
        "ma50": round(ma50, 4) if ma50 else None, "ma200": round(ma200, 4) if ma200 else None,
        "rsi": rsi(closes[-120:]),
        "volat": vol_ann,
        "hi52": round(hi52, 4) if hi52 else None, "lo52": round(lo52, 4) if lo52 else None,
        "offHigh": pct(price, hi52),
        "relVol": rel_vol, "upDays5": up_days,
        "mktCap": q.get("marketCap"),
        "pe": round(q["trailingPE"], 1) if q.get("trailingPE") else None,
        "fpe": round(q["forwardPE"], 1) if q.get("forwardPE") else None,
        "earnings": None, "analyst": None, "analystLabel": None,
        "divYield": q.get("dividendYield"),
    }
    et = q.get("earningsTimestamp") or q.get("earningsTimestampStart")
    if et:
        m["earnings"] = dt.datetime.fromtimestamp(et, dt.timezone.utc).date().isoformat()
        m["earningsEst"] = bool(q.get("isEarningsDateEstimate"))
    ar = q.get("averageAnalystRating")
    if ar:
        try:
            num, label = ar.split(" - ", 1)
            m["analyst"], m["analystLabel"] = float(num), label.strip()
        except ValueError:
            pass
    m["score"] = score(m, bench_ret)
    return m, closes


def score(m, bench):
    p, r = m["price"], m["r"]
    parts = {}

    # Trend: where the price sits against its 50- and 200-day averages
    t = 50
    notes = []
    if m["ma50"] and p:
        d50 = (p / m["ma50"] - 1) * 100
        t += clamp(d50 * 2.2, -25, 25)
        notes.append(f"{abs(d50):.0f}% {'above' if d50 >= 0 else 'below'} its 50-day average")
    if m["ma200"] and p:
        d200 = (p / m["ma200"] - 1) * 100
        t += clamp(d200 * 1.2, -20, 20)
        notes.append(f"{abs(d200):.0f}% {'above' if d200 >= 0 else 'below'} its 200-day average")
    if m["ma50"] and m["ma200"]:
        t += 5 if m["ma50"] > m["ma200"] else -5
    parts["trend"] = {"v": round(clamp(t)), "why": " and ".join(notes) or "Not enough history yet"}

    # Momentum: 1 and 3 month returns, weighted toward the recent month
    mo = None
    if r["m1"] is not None or r["m3"] is not None:
        mo = 0.6 * (r["m1"] or 0) + 0.4 * (r["m3"] or 0) / 2
    parts["momentum"] = {"v": round(lin(mo, -18, 18)),
                         "why": f"{fmt_pct(r['m1'])} over a month, {fmt_pct(r['m3'])} over three months"}

    # Versus the sector: 3 month return against the defense ETF
    rel = None
    if r["m3"] is not None and bench.get("m3") is not None:
        rel = r["m3"] - bench["m3"]
    parts["relative"] = {"v": round(lin(rel, -25, 25)),
                         "why": ("No 3-month history yet" if rel is None else
                                 f"{abs(rel):.0f} points {'ahead of' if rel >= 0 else 'behind'} the defense ETF over three months")}

    # Heat: how stretched the recent run is (14-day RSI, explained in plain words)
    h = m["rsi"]
    if h is None:
        hv, hw = 50, "Not enough history yet"
    elif h >= 78:
        hv, hw = 25, "Has run up very fast lately, pullbacks are common from here"
    elif h >= 68:
        hv, hw = 50, "Running hot after a strong stretch"
    elif h >= 45:
        hv, hw = 80, "Moving steadily, not stretched in either direction"
    elif h >= 32:
        hv, hw = 55, "Has been sliding, buyers have been quiet"
    else:
        hv, hw = 40, "Sold off hard recently, sometimes a bounce follows"
    parts["heat"] = {"v": hv, "why": hw, "raw": h}

    # Volume: are more shares changing hands than usual, and on up days or down days?
    rv = m["relVol"]
    if rv is None:
        vv, vw = 50, "No volume data"
    else:
        up = m["upDays5"] >= 3
        if rv >= 1.3:
            vv = 80 if up else 25
            vw = f"Trading {rv:.1f}x its usual volume this week, mostly on {'up' if up else 'down'} days"
        elif rv <= 0.7:
            vv, vw = 45, "Quieter than usual this week"
        else:
            vv, vw = 55 + (8 if up else -8), "Normal trading volume this week"
    parts["volume"] = {"v": round(vv), "why": vw}

    # Analysts: Yahoo's average rating, 1 (strong buy) to 5 (sell)
    a = m["analyst"]
    parts["analysts"] = {"v": round(lin(None if a is None else 4 - a, 0, 2.8)) if a else 50,
                         "why": f"Average Wall Street rating: {m['analystLabel']}" if a else "No analyst coverage listed"}

    W = {"trend": .24, "momentum": .20, "relative": .18, "heat": .12, "volume": .10, "analysts": .16}
    total = round(sum(parts[k]["v"] * w for k, w in W.items()))
    signal = "Buy" if total >= 65 else "Sell" if total <= 38 else "Hold"
    return {"total": total, "signal": signal, "parts": parts}


def fmt_pct(x):
    return "n/a" if x is None else f"{'+' if x >= 0 else ''}{x:.1f}%"


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------
def clean_title(t):
    t = html.unescape(t or "").strip()
    return re.sub(r"\s+", " ", t)


def google_news(query, days=10):
    url = f"https://news.google.com/rss/search?q={quote_plus(query + f' when:{days}d')}&hl=en-US&gl=US&ceid=US:en"
    out = []
    try:
        root = ET.fromstring(get(url).content)
        for it in root.iter("item"):
            title = clean_title(it.findtext("title"))
            src_el = it.find("source")
            src = src_el.text if src_el is not None else ""
            if src and title.endswith(" - " + src):
                title = title[: -(len(src) + 3)]
            pub = it.findtext("pubDate")
            ts = int(email.utils.parsedate_to_datetime(pub).timestamp()) if pub else None
            out.append({"title": title, "src": src or "Google News", "url": it.findtext("link"), "ts": ts})
    except Exception as e:
        print("google news failed:", query, e, file=sys.stderr)
    return out


def yahoo_news(t):
    try:
        r = get(f"https://query2.finance.yahoo.com/v1/finance/search?q={ysym(t)}&newsCount=8&quotesCount=0")
        return [{"title": clean_title(n["title"]), "src": n.get("publisher", "Yahoo Finance"),
                 "url": n.get("link"), "ts": n.get("providerPublishTime")} for n in r.json().get("news", [])]
    except Exception:
        return []


def norm(t):
    return re.sub(r"[^a-z0-9 ]", "", t.lower())[:70]


STOP = set("the a an of to in on for and or with by at as is are its from after over this that will be has have stock stocks shares".split())


def words(t):
    return {w for w in re.sub(r"[^a-z0-9 ]", " ", t.lower()).split() if w not in STOP and len(w) > 2}


RELEVANT = re.compile(r"defen[cs]e|pentagon|military|army|navy|naval|air force|space force|missile|drone|unmanned|contract|nato|"
                      r"weapon|munition|satellite|launch|rocket|aerospace|fighter|jet|shipbuild|submarine|hypersonic|golden dome|"
                      r"\bdod\b|marine corps|ukraine|israel|counter-|radar|nuclear|lunar|orbit|spacecraft|warship|budget|tank", re.I)


def keep(items, max_age_days, limit):
    cutoff = time.time() - max_age_days * 86400
    seen, out, seen_w = set(), [], []
    for n in sorted(items, key=lambda x: -(x["ts"] or 0)):
        if not n["title"] or len(n["title"]) < 18 or SPAM.search(n["title"]) or JUNK.search(n["title"]):
            continue
        if BAD_SOURCES.search(n["src"] or "") or re.search(r"[\u0600-\u06ff\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]", n["title"] + n["src"]):
            continue
        if n["ts"] and n["ts"] < cutoff:
            continue
        k = norm(n["title"])
        if k in seen:
            continue
        ws = words(n["title"])
        if any(len(ws & o) / max(1, min(len(ws), len(o))) > 0.6 for o in seen_w):
            continue
        if len(n["title"].split()) < 5:
            continue
        seen.add(k); seen_w.append(ws)
        out.append(n)
        if len(out) >= limit:
            break
    return out


TOPICS = [
    ("Analysts", r"price target|upgraded|downgraded|downgrades|upgrades \w+ to|initiat\w* (coverage|at)|outperform|overweight|underweight|analyst"),
    ("Contracts", r"\bcontracts?\b|\bawarded?s?\b|task order|\border(s|ed)? for|selected (by|to|for)|\bwins? .*(deal|order|bid)"),
    ("Earnings", r"earnings|revenue|quarterly|\bq[1-4]\b|guidance|results|profit|backlog"),
    ("Budget & policy", r"budget|congress|pentagon|appropriation|ndaa|tariff|export|nato|spending|administration|senate|house|white house|shutdown|golden dome"),
    ("Space", r"launch|satellite|space|orbit|lunar|moon|rocket"),
    ("Drones", r"drone|unmanned|uav|autonomous|counter-uas|loitering"),
]


def topic_of(title):
    if OPINION.search(title):
        return "Opinion"
    t = title.lower()
    for name, pat in TOPICS:
        if re.search(pat, t):
            return name
    return "General"


def tag_tickers(title, aliases):
    hits = []
    for tk, names in aliases.items():
        if re.search(r"\(" + re.escape(tk) + r"\)|\b(NYSE|NASDAQ|Nasdaq): ?" + re.escape(tk) + r"\b", title):
            hits.append(tk); continue
        for a in names:
            if re.search(r"(?<![A-Za-z])" + re.escape(a) + r"(?![A-Za-z])", title):
                hits.append(tk); break
    return hits[:4]


# ---------------------------------------------------------------------------
# AI insights
# ---------------------------------------------------------------------------
STYLE = """Write for a smart person who follows the stock market casually and does not know finance jargon.
Rules:
- Short, plain sentences. Concrete numbers and named events from the data, not vague phrases.
- Do not use these words or ideas: RSI, overbought, oversold, moving average (say "50-day average price" if needed), bullish, bearish, catalyst, headwind, tailwind, momentum play, multiple, consolidation, support, resistance, breakout, priced in, risk-on, risk-off, upside, downside.
- No em dashes. Do not use the "it's not X, it's Y" sentence pattern. No hype, no slogans, no exclamation points.
- Do not tell the reader to buy or sell. Describe what is happening and what matters.
- If the headlines are thin or unrelated, say the move looks driven by the wider market or by nothing specific, instead of inventing a reason.
- Only use facts from the data given. Do not make up contract values, dates or quotes."""


def ask_ai(prompt, max_tokens=4000):
    key = os.environ.get("ANTHROPIC_API_KEY")
    r = requests.post("https://api.anthropic.com/v1/messages", timeout=180, headers={
        "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
        json={"model": MODEL, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]})
    r.raise_for_status()
    text = "".join(b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text")
    return extract_json(text)


def extract_json(text):
    t = re.sub(r"```(?:json)?", "", text)
    s, e = t.find("{"), t.rfind("}")
    return json.loads(t[s:e + 1])


def stock_brief(tk, d):
    m = d
    sc = m["score"]
    lines = [
        f"{tk} | {m['name']} | group: {m['sector']}",
        f"price ${m['price']:.2f}, today {fmt_pct(m['chgPct'])}, 5 days {fmt_pct(m['r']['d5'])}, 1 month {fmt_pct(m['r']['m1'])}, "
        f"3 months {fmt_pct(m['r']['m3'])}, year to date {fmt_pct(m['r']['ytd'])}, 1 year {fmt_pct(m['r']['y1'])}",
        f"{fmt_pct(m['offHigh'])} from 52-week high ${m['hi52']}, 52-week low ${m['lo52']}",
        f"score {sc['total']}/100 ({sc['signal']}): " + "; ".join(f"{k} {v['v']} ({v['why']})" for k, v in sc["parts"].items()),
    ]
    if m.get("mktCap"):
        lines.append(f"market value ${m['mktCap']/1e9:.1f}B" + (f", P/E {m['pe']}" if m.get("pe") else "") +
                     (f", forward P/E {m['fpe']}" if m.get("fpe") else ""))
    if m.get("earnings") and m["earnings"] >= dt.date.today().isoformat():
        lines.append(f"next earnings report: {m['earnings']}{' (estimated)' if m.get('earningsEst') else ''}")
    heads = [f"- {dt.datetime.fromtimestamp(n['ts'], dt.timezone.utc):%b %d} {n['title']} ({n['src']})" for n in m["news"][:6] if n.get("ts")]
    lines.append("headlines:\n" + ("\n".join(heads) if heads else "- none in the last few weeks"))
    return "\n".join(lines)


def ai_takes(data):
    tks = list(data["tickers"].keys())
    takes = {}
    for i in range(0, len(tks), 8):
        batch = tks[i:i + 8]
        prompt = (f"{STYLE}\n\nToday is {dt.date.today():%A, %B %d, %Y}. For each defense or aerospace stock below, write:\n"
                  '- "take": 2 or 3 sentences on what is going on with the stock right now and the most likely reason.\n'
                  '- "good": 1 or 2 short points working in its favor.\n'
                  '- "risks": 1 or 2 short points to be careful about.\n'
                  '- "watch": one sentence on the next thing that could move it (an earnings date, a contract decision, a launch, a budget vote), only if the data supports it.\n'
                  'Reply with only JSON: {"TICKER": {"take": "...", "good": ["..."], "risks": ["..."], "watch": "..."}, ...}\n\n'
                  + "\n\n".join(stock_brief(t, data["tickers"][t]) for t in batch))
        try:
            res = ask_ai(prompt)
            for t in batch:
                if t in res:
                    takes[t] = {**res[t], "priceAt": data["tickers"][t]["price"]}
            print(f"insights {i + len(batch)}/{len(tks)}")
        except Exception as e:
            print("insight batch failed:", batch, e, file=sys.stderr)
            if isinstance(e, requests.HTTPError) and e.response is not None and e.response.status_code in (401, 403):
                print("The API key was rejected, stopping here", file=sys.stderr)
                break
    return takes


def ai_brief(data):
    tk = data["tickers"]
    groups = {}
    for t, d in tk.items():
        groups.setdefault(d["sector"], []).append(d)
    gl = []
    for g, ds in groups.items():
        w5 = [d["r"]["d5"] for d in ds if d["r"]["d5"] is not None]
        m1 = [d["r"]["m1"] for d in ds if d["r"]["m1"] is not None]
        gl.append(f"- {g}: avg 5-day {fmt_pct(sum(w5)/len(w5) if w5 else None)}, avg 1-month {fmt_pct(sum(m1)/len(m1) if m1 else None)}")
    movers = sorted(tk.items(), key=lambda kv: -(kv[1]["r"]["d5"] or 0))
    top = ", ".join(f"{t} {fmt_pct(d['r']['d5'])}" for t, d in movers[:6])
    bot = ", ".join(f"{t} {fmt_pct(d['r']['d5'])}" for t, d in movers[-6:])
    b = data["bench"]
    heads = "\n".join(f"- {n['title']} ({n['src']})" for n in data["news"][:30])
    sig = {s: sum(1 for d in tk.values() if d["score"]["signal"] == s) for s in ("Buy", "Hold", "Sell")}
    prompt = (f"{STYLE}\n\nToday is {dt.date.today():%A, %B %d, %Y}. Write a short market read for a dashboard that tracks "
              f"{len(tk)} defense, aerospace and space stocks.\n\n"
              f"Defense ETF (ITA): today {fmt_pct(b['ITA']['chgPct'])}, 5 days {fmt_pct(b['ITA']['r']['d5'])}, 1 month {fmt_pct(b['ITA']['r']['m1'])}, YTD {fmt_pct(b['ITA']['r']['ytd'])}\n"
              f"S&P 500 (SPY): today {fmt_pct(b['SPY']['chgPct'])}, 5 days {fmt_pct(b['SPY']['r']['d5'])}, 1 month {fmt_pct(b['SPY']['r']['m1'])}\n"
              f"Groups:\n" + "\n".join(gl) +
              f"\nBest 5-day: {top}\nWorst 5-day: {bot}\nScore signals: {sig}\n\nRecent headlines:\n{heads}\n\n"
              'Reply with only JSON: {"headline": "one plain sentence that sums up where defense stocks stand right now",'
              ' "summary": ["paragraph of 2-3 sentences", "paragraph of 2-3 sentences"],'
              ' "watch": ["three short items worth keeping an eye on, each one sentence"]}')
    return ask_ai(prompt, 1500)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "defense_data.json"))
    args = ap.parse_args()

    old = {}
    if os.path.exists(args.out):
        try:
            old = json.load(open(args.out))
        except Exception:
            old = {}

    meta_by = {t: (n, s, a) for t, n, s, a in U}
    ita, ita_asof = ita_holdings()
    for h in ita:  # any new ITA holding gets added to the universe
        if h["t"] not in meta_by:
            meta_by[h["t"]] = (h.get("name", h["t"]), "Engines & parts", [h.get("name", h["t"])])
    universe = list(meta_by.keys())
    all_syms = universe + [b for b, _ in BENCH]

    print(f"Fetching {len(all_syms)} charts")
    with ThreadPoolExecutor(6) as ex:
        charts = dict(zip(all_syms, ex.map(chart, all_syms)))
    qd = quotes(all_syms)

    data = {"version": 2, "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "bench": {}, "tickers": {}, "news": [], "ita": ita, "itaAsOf": ita_asof}

    # trading-day calendar shared by every series
    cal = sorted({r[0] for c in charts.values() if c for r in c["rows"]})
    data["dates"] = cal
    pos = {d: i for i, d in enumerate(cal)}

    def series(rows, closes):
        if not rows:
            return {"s": 0, "c": []}
        dg = 2 if closes[-1] >= 5 else 3
        return {"s": pos[rows[0][0]], "c": [round(c, dg) for c in closes]}

    latest_ts = 0
    for b, name in BENCH:
        c = charts.get(b)
        if not c:
            continue
        m, closes = metrics(c["rows"], c["meta"], qd.get(b, {}), {})
        m.pop("score", None)
        m["name"] = name
        m["hist"] = series(c["rows"], closes)
        data["bench"][b] = m
    bench_ret = data["bench"].get("ITA", {}).get("r", {})

    itaw = {h["t"]: h["w"] for h in ita}
    for t in universe:
        c = charts.get(t)
        if not c or len(c["rows"]) < 5:
            print("skipping", t, "(no price data)", file=sys.stderr)
            continue
        name, sector, aka = meta_by[t]
        q = qd.get(ysym(t))
        if not q and t in (old.get("tickers") or {}):
            o = old["tickers"][t]
            q = {"marketCap": o.get("mktCap"), "trailingPE": o.get("pe"), "forwardPE": o.get("fpe"),
                 "averageAnalystRating": f"{o['analyst']} - {o['analystLabel']}" if o.get("analyst") else None}
            if o.get("earnings"):
                q["earningsTimestamp"] = int(dt.datetime.fromisoformat(o["earnings"] + "T12:00:00+00:00").timestamp())
                q["isEarningsDateEstimate"] = o.get("earningsEst")
            print("using saved quote details for", t, file=sys.stderr)
        m, closes = metrics(c["rows"], c["meta"], q or {}, bench_ret)
        m.update({"name": name, "sector": sector, "ita": itaw.get(t), "hist": series(c["rows"], closes)})
        latest_ts = max(latest_ts, c["meta"].get("regularMarketTime") or 0)
        data["tickers"][t] = m
    if not data["tickers"]:
        print("No price data came back. Keeping the existing file.", file=sys.stderr)
        sys.exit(3)
    data["asOf"] = dt.datetime.fromtimestamp(latest_ts, dt.timezone.utc).isoformat(timespec="seconds") if latest_ts else None

    # News: per stock (Yahoo + Google), then a sector-wide feed
    print("Fetching headlines")
    tks = list(data["tickers"].keys())

    def per_stock(t):
        name, _, aka = meta_by[t]
        names = [name] + aka
        pat = re.compile(r"\(" + re.escape(t) + r"\)|(?<![A-Za-z])(" + "|".join(re.escape(a) for a in names) + r")(?![A-Za-z])"
                         + (r"|\b" + re.escape(t) + r"\b" if len(t) >= 3 and t not in ("AIR", "ATI", "PL", "GE") else ""))
        bad = EXCLUDE.get(t)
        items = yahoo_news(t) + google_news(f'"{name}" stock', 21)
        items = [n for n in items if pat.search(n["title"]) and not (bad and re.search(bad, n["title"], re.I))]
        return t, keep(items, 21, 8)

    with ThreadPoolExecutor(6) as ex:
        for t, items in ex.map(per_stock, tks):
            for n in items:
                n["topic"] = topic_of(n["title"])
            data["tickers"][t]["news"] = items

    aliases = {t: meta_by[t][2] for t in tks}
    pool = []
    with ThreadPoolExecutor(5) as ex:
        for items in ex.map(lambda q: google_news(q, 7), NEWS_QUERIES):
            pool += items
    for t in tks:
        pool += [n for n in data["tickers"][t]["news"] if n["ts"] and n["ts"] > time.time() - 5 * 86400]
    feed = keep(pool, 7, 200)
    feed = [n for n in feed if (RELEVANT.search(n["title"]) or tag_tickers(n["title"], aliases)) and not OPINION.search(n["title"])]
    def rank(n):
        src = (n["src"] or "").lower()
        boost = 1.0 if src in GOOD_SOURCES else 0.0
        if OPINION.search(n["title"]):
            boost -= 1.2
        age_h = (time.time() - (n["ts"] or 0)) / 3600
        return boost * 18 - age_h
    feed = sorted(feed, key=lambda n: -rank(n))[:90]
    for n in feed:
        n["topic"] = topic_of(n["title"])
        n["tickers"] = tag_tickers(n["title"], aliases)
        n.pop("why", None)
    data["news"] = feed

    # AI insights
    mode = os.environ.get("AI_MODE", "auto").lower()
    max_age = float(os.environ.get("AI_MAX_AGE_HOURS", "5"))
    old_ai = old.get("ai") or {}
    age_h = 1e9
    if old_ai.get("generated"):
        age_h = (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(old_ai["generated"])).total_seconds() / 3600
    want = mode == "force" or (mode == "auto" and age_h >= max_age)
    if want and os.environ.get("ANTHROPIC_API_KEY"):
        print("Writing insights with", MODEL)
        now = data["generated"]
        brief = None
        try:
            brief = ai_brief(data)
            brief["at"] = now
        except Exception as e:
            print("brief failed:", e, file=sys.stderr)
        takes = ai_takes(data)
        for v in takes.values():
            v["at"] = now
        if brief is None and not takes:
            print("No insights came back, keeping the saved ones")
            data["ai"] = old_ai
        else:
            ai = {"generated": now, "model": MODEL, "brief": brief or old_ai.get("brief"), "takes": takes}
            if brief is None and ai["brief"] and "at" not in ai["brief"]:
                ai["brief"]["at"] = old_ai.get("generated")
            for t, v in (old_ai.get("takes") or {}).items():  # keep older takes for anything that failed
                if t not in takes:
                    ai["takes"][t] = {**v, "at": v.get("at") or old_ai.get("generated")}
            data["ai"] = ai
    else:
        if want:
            print("ANTHROPIC_API_KEY not set, keeping the saved insights")
        data["ai"] = old_ai

    with open(args.out, "w") as f:
        json.dump(data, f, separators=(",", ":"), ensure_ascii=False)
    print(f"Wrote {args.out}: {len(data['tickers'])} stocks, {len(data['news'])} headlines, "
          f"{os.path.getsize(args.out)/1024:.0f} KB")


if __name__ == "__main__":
    main()
