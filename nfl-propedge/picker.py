"""The step that actually chooses the bets.

The engine produces a chance for every bet on the board. This step hands the strongest candidates to the
AI together with the things numbers alone miss (injury notes, who is questionable, a new quarterback,
weather, lopsided games, lines that have moved, bets that pull against each other) and asks it to choose
and order the bets and say why. Its answer is checked against the board, so it can only pick bets that exist.

If no ANTHROPIC_API_KEY is set, the same rules are applied to the engine's ranking so the page still works.
Shared by NFL PropEdge and the NFL TD Predictor: each passes its own rules text and candidate lines.
"""
import datetime as dt, json, os, re
import requests

MODEL = os.environ.get("AI_MODEL", "claude-sonnet-5-5")
MAX_SAFE, MAX_PAY = 8, 6

WRITING = """Writing rules for the text you produce:
- Plain English for someone who does not follow betting or know football terms. No slang and no jargon: do not
  say juice, vig, fade, lock, sharp, square, smash, chalk, hammer, value play, alt line, ladder, edge, correlation,
  regression, usage, game script, volume, ceiling, floor, bell cow, target share, snap share, red zone looks.
- One or two short sentences per reason. Use concrete facts from the board: recent results, who is out, the line.
- Do not just repeat the chance back. Say what makes the bet hold up, and name the main way it loses.
- No em dashes. Do not use the "it's not X, it's Y" pattern. No hype and no exclamation points.

Hard rules:
- Use only candidates from the board, by their id. Never change the number.
- At most one bet per player. At most two bets from the same game.
- Order each list so that the first 2, first 3, and so on each make a sensible parlay on their own. Best bet first.
- A bet can appear in only one list.

Answer with JSON only, in this shape:
{"safest": [{"id": "...", "why": "...", "risk": "..."}],
 "payout": [{"id": "...", "why": "...", "risk": "..."}],
 "skip": [{"id": "...", "why": "..."}],
 "note": "two or three sentences on this week's games overall: which to lean on and which to avoid"}
"safest" should have up to SAFE_N bets and "payout" up to PAY_N. "skip" lists up to 6 candidates that look good by
the numbers but that you left out on purpose, with the reason."""

YARDS_RULES = """You are choosing NFL player yardage bets for a parlay from a board of candidates. Each candidate already
has a chance of hitting. That chance comes from a statistical model trained on every player-game since 2019,
blended with the DraftKings line for the same player, and checked against this season's results: bets it gave
85% have hit about 85% of the time. So respect the chances. Your job is the part the numbers cannot do: read the
situation around each bet and decide which ones to trust this week.

Things to weigh:
- Injury notes. A player listed as questionable might sit or play less. Leave him out unless the bet is clearly
  safe even in a smaller role. If a bet depends on a teammate sitting and that teammate is only questionable, it is shaky.
- Role changes. If a teammate is newly out, the model has already raised the others. Trust that more when the
  recent games show the bigger role actually happening, less when it is a guess.
- A new or backup quarterback makes every receiver on that team harder to predict.
- Rain, snow or strong wind hurts passing and receiving and helps rushing.
- Lopsided games. With a spread near 10 or more, the favorite runs the ball late and its passing slows down, and
  the underdog throws more. That helps the favorite's lead runner and can hurt its receivers.
- When the line has moved a lot since it opened, the sportsbook has learned something. Do not bet against the move.
- Prefer bets where the recent games, the model and the sportsbook's line all point the same way. The board shows
  the model's own chance next to the final chance: a wide gap between them is a reason for suspicion first.
- Do not stack bets that fight each other, such as a quarterback going over on passing while his own runner goes
  over on rushing, or two receivers on the same team both needing big games.
- Players with only one or two games this season are less reliable.

"safest" is for bets most likely to hit: "this many or more" bets set below what the player usually does.
"payout" is for over or under bets on the sportsbook's main line, which pay close to even money. Those are close to
coin flips by design, so only choose ones with a real reason behind them.

"""

TD_RULES = """You are choosing NFL touchdown bets for a parlay from a board of candidates. Each candidate already has a
chance of hitting from a statistical model trained on every player-game since 2019. The model is built on how
often each player gets the ball close to the goal line, his team's expected points from the betting market, his
quarterback's touchdown passing, the opposing defense, and who is injured. Its chances have matched real results
closely in testing, so respect them. Your job is the part the numbers cannot do: read the situation around each
bet and decide which ones to trust this week.

Things to weigh:
- Touchdowns are rare events. Even the best running back scores in only about two games out of three. Say so
  honestly in the reasons and do not oversell.
- Injury notes. A player listed as questionable might sit or play less. If a teammate who usually gets the ball
  near the goal line is out, someone else inherits those chances. Trust that more when recent games show it.
- A team expected to score a lot gives all its players more chances. A big underdog expected to score little does not.
- A new or backup quarterback lowers every receiver's chance and usually the team's whole scoring.
- Rain, snow or wind pushes teams to run near the goal line, which helps running backs over receivers.
- A quarterback's touchdown passes and his own receivers scoring go together. Pairing them is fine and makes the
  parlay more likely than the separate chances suggest. Two receivers on the same team compete for the same scores.
- Prefer players whose chances near the goal line have been steady week after week over players who scored on a
  few long plays.

"safest" is for the bets most likely to hit: players most likely to score, and quarterbacks most likely to throw
at least one or two touchdown passes.
"payout" is for less likely bets that pay more: players with roughly a one in three chance who have a real reason
to score this week, or a player scoring twice.

"""


def rules(kind, extra=""):
    r = (YARDS_RULES if kind == "yards" else TD_RULES) + WRITING.replace("SAFE_N", str(MAX_SAFE)).replace("PAY_N", str(MAX_PAY))
    return r + ("\n\nExtra instructions from the user for this request:\n" + extra if extra else "")


def extract_json(text):
    t = re.sub(r"```(?:json)?", "", text)
    s, e = t.find("{"), t.rfind("}")
    return json.loads(t[s:e + 1])


def ask(prompt, max_tokens=20000):
    # The model reasons before it answers and that reasoning counts against max_tokens, so leave plenty of room.
    r = requests.post("https://api.anthropic.com/v1/messages", timeout=600, headers={
        "x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01", "content-type": "application/json"},
        json={"model": MODEL, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]})
    r.raise_for_status()
    j = r.json()
    text = "".join(b.get("text", "") for b in j.get("content", []) if b.get("type") == "text")
    if not text.strip():
        raise RuntimeError(f"no answer text came back (stop reason: {j.get('stop_reason')})")
    return extract_json(text)


def clean(text):
    return re.sub(r"\s*[—–]\s*", ", ", str(text or "")).strip()[:400]


def enforce(items, L, limit, used_players=None):
    """Apply the hard rules to an ordered list: real ids only, one bet per player, two per game."""
    out, players, games = [], set(used_players or ()), {}
    for it in items:
        l = L.get(it.get("id"))
        if not l or l["pid"] in players or games.get(l["gid"], 0) >= 2:
            continue
        players.add(l["pid"]); games[l["gid"]] = games.get(l["gid"], 0) + 1
        out.append({"id": l["id"], "why": clean(it.get("why")), "risk": clean(it.get("risk"))})
        if len(out) >= limit:
            break
    return out, players


def pick(kind, text, safe, pay, fallback):
    """kind: "yards" or "td". text: the board written out for the AI. safe/pay: candidate bets.
    fallback(leg) -> (why, risk, usable) gives the wording and filter for the engine-only ranking."""
    L = {l["id"]: l for l in safe + pay}
    by, resp, err = "model", None, None
    if os.environ.get("ANTHROPIC_API_KEY") and (safe or pay):
        try:
            resp = ask(rules(kind) + "\n\n" + text)
            by = "ai"
        except Exception as e:  # keep the page working if the AI call fails
            err = f"{type(e).__name__}: {e}"[:200]
            print("AI pick failed, using the engine's ranking:", err)
    def ranked(pool):
        out = []
        for l in pool:
            why, risk, ok = fallback(l)
            if ok:
                out.append({"id": l["id"], "why": why, "risk": risk})
        return out
    base = {"safest": ranked(safe), "payout": ranked(pay), "skip": [], "note": ""}
    if resp is None:
        resp = base
    safest, used = enforce(resp.get("safest") or [], L, MAX_SAFE)
    payout, used = enforce(resp.get("payout") or [], L, MAX_PAY, used)
    if by == "ai" and len(safest) < 4:  # top up from the engine's ranking if the AI returned a short list
        more, used = enforce(base["safest"], L, MAX_SAFE, used)
        safest += more[: 4 - len(safest)]
    skip = [{"id": s["id"], "why": clean(s.get("why"))} for s in (resp.get("skip") or []) if s.get("id") in L][:6]
    return {"by": by, "model": MODEL if by == "ai" else None, "at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
            "safest": safest, "payout": payout, "skip": skip, "note": clean(resp.get("note"))[:700], "error": err}
