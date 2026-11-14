"""The step that actually chooses the parlay.

The models produce a chance for every bet on the board. This step hands the strongest candidates to the
AI together with the things numbers alone miss (injury notes, who is questionable, lopsided games, roles
that just changed, legs that pull against each other) and asks it to choose and order the legs and say why.
Its answer is checked against the board, so it can only pick bets that exist at the price shown.

If no ANTHROPIC_API_KEY is set, the same rules are applied to the model's ranking so the page still works.
"""
import datetime as dt, json, os, re
import requests

MODEL = os.environ.get("AI_MODEL", "claude-sonnet-5-5")
MAX_SAFE, MAX_PAY = 8, 6

RULES = """You are choosing player bets for an NBA parlay from a board of candidates. Each candidate already has a
chance of hitting, worked out by a statistical model that was trained on two-plus seasons of box scores and
blended with the sportsbook's own line. Those chances are well calibrated on average, so respect them. Your job
is the part the model cannot do: read the situation around each bet and decide which ones to trust tonight.

Things to weigh:
- Injury notes. A player listed as questionable might sit or play limited minutes. Leave him out unless the bet is
  clearly safe even on reduced minutes. If a key teammate is questionable, a bet that depends on him sitting is shaky.
- Role changes. If a teammate is newly out, the model has already raised the others. Trust that more when the
  recent game log shows the bigger role actually happening, less when it is a guess.
- Lopsided games. With a spread of 11 or more, starters on both teams can sit the fourth quarter. That hurts
  "or more" bets for stars and helps "under" bets.
- Second night of a back-to-back, long absences, and players with few games on record are all less reliable.
- Do not stack bets that fight each other, such as big scoring nights from two teammates who share the ball.
- Prefer bets where the recent game log, the projection, and the sportsbook's line all point the same way.
- A candidate whose model chance is far above what its price implies deserves suspicion first: check the notes
  for a reason the book would know and the box scores would not.

Hard rules:
- Use only candidates from the board, by their id. Never change the number or the price.
- At most one bet per player. At most two bets from the same game.
- Order each list so that the first 2, first 3, and so on each make a sensible parlay on their own. Best bet first.
- "safest" is for bets most likely to hit. "payout" is for bets that pay close to even money or better and still
  have a real case. A bet can appear in only one list.

Writing rules for the text you produce:
- Plain English for someone who does not follow betting. No slang and no jargon: do not say juice, vig, fade,
  lock, sharp, square, smash, chalk, hammer, value play, alt line, ladder, edge, correlation, regression, usage.
- One or two short sentences per reason. Use concrete facts from the board: minutes, recent results, who is out.
- Do not just repeat the model's chance back. Say what makes the bet hold up, and name the main way it loses.
- No em dashes. Do not use the "it's not X, it's Y" pattern. No hype and no exclamation points.

Answer with JSON only, in this shape:
{"safest": [{"id": "...", "why": "...", "risk": "..."}],
 "payout": [{"id": "...", "why": "...", "risk": "..."}],
 "skip": [{"id": "...", "why": "..."}],
 "note": "two or three sentences on tonight's slate overall: which games to lean on and which to avoid"}
"safest" should have up to SAFE_N bets and "payout" up to PAY_N. "skip" lists up to 6 candidates that look good by
the numbers but that you left out on purpose, with the reason."""


def rules(extra=""):
    r = RULES.replace("SAFE_N", str(MAX_SAFE)).replace("PAY_N", str(MAX_PAY))
    return r + ("\n\nExtra instructions from the user for this request:\n" + extra if extra else "")


def pools(legs, players):
    """Narrow the full board to the candidates worth the AI's attention."""
    P = {p["id"]: p for p in players}
    safe, pay = [], []
    by_ps = {}
    for l in legs:
        by_ps.setdefault((l["pid"], l["stat"]), []).append(l)
    for (pid, stat), ls in by_ps.items():
        if P[pid]["status"] in ("out", "doubtful"):
            continue
        # safest: the rung of each player/stat with the best chance that still pays something
        s = [l for l in ls if l["p"] >= 0.70 and l["dec"] >= 1.13]
        if s:
            safe.append(max(s, key=lambda l: l["p"] + 0.6 * (l["p"] - l["be"])))
        v = [l for l in ls if l["dec"] >= 1.65 and l["p"] >= 0.50 and l["p"] - l["be"] >= 0.015]
        if v:
            pay.append(max(v, key=lambda l: l["p"] - l["be"]))
    def top(pool, key, per_player, n):
        out, seen = [], {}
        for l in sorted(pool, key=key, reverse=True):
            if seen.get(l["pid"], 0) >= per_player:
                continue
            seen[l["pid"]] = seen.get(l["pid"], 0) + 1
            out.append(l)
            if len(out) >= n:
                break
        return out
    safe = top(safe, lambda l: l["p"] + 0.6 * (l["p"] - l["be"]), 2, 40)
    pay = top(pay, lambda l: l["p"] - l["be"], 1, 24)
    return safe, pay


def _leg_line(l, p):
    last = ", ".join(str(x) for x in l["last"])
    return (f'{l["id"]} | {p["name"]} ({p["team"]} {"vs" if p["home"] else "at"} {p["opp"]}) | {l["text"]} | price {l["price"]:+d} '
            f'| model chance {round(l["p"] * 100)}%, price implies {round(l["be"] * 100)}% | projection {l["proj"]} '
            f'| hit in {l["hits"]} of last {l["of"]} | last games, newest first: {last} | minutes lately {p["min"]:.0f}'
            + (f' | notes: {"; ".join(p["flags"])}' if p["flags"] else ""))


def prompt_data(data, safe, pay):
    P = {p["id"]: p for p in data["players"]}
    out = [f'NBA slate for {data["day"]}. Lines from {data["book"] or "the sportsbook"}.', "", "GAMES"]
    for g in data["games"]:
        fav = g["home"] if (g["spread"] or 0) < 0 else g["away"]
        sp = f'{fav} favored by {abs(g["spread"])}' if g["spread"] else "no spread posted"
        out.append(f'- {g["away"]} at {g["home"]}: {sp}, expected total points {g["total"]}')
        for side in (g["away"], g["home"]):
            inj = [f'{i["name"]} ({i["status"]}{", " + i["what"] if i.get("what") else ""})' for i in data["injuries"].get(side, [])]
            if inj:
                out.append(f'    {side} injuries: ' + "; ".join(inj))
    out += ["", "CANDIDATES FOR THE SAFEST LIST"] + [_leg_line(l, P[l["pid"]]) for l in safe]
    out += ["", "CANDIDATES FOR THE PAYOUT LIST"] + [_leg_line(l, P[l["pid"]]) for l in pay]
    return "\n".join(out)


def extract_json(text):
    t = re.sub(r"```(?:json)?", "", text)
    s, e = t.find("{"), t.rfind("}")
    return json.loads(t[s:e + 1])


def ask(prompt, max_tokens=4000):
    r = requests.post("https://api.anthropic.com/v1/messages", timeout=240, headers={
        "x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01", "content-type": "application/json"},
        json={"model": MODEL, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]})
    r.raise_for_status()
    return extract_json("".join(b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text"))


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


def model_only(safe, pay, players):
    """Fallback ranking with the same hard rules, used when the AI step is unavailable."""
    P = {p["id"]: p for p in players}
    def why(l):
        p = P[l["pid"]]
        return (f'Projected for {l["proj"]} {l["unit"]} and did this in {l["hits"]} of his last {l["of"]} games.',
                "; ".join(p["flags"][:2]))
    ok = lambda l: P[l["pid"]]["status"] == "" and P[l["pid"]]["n_prev"] >= 8
    s = [{"id": l["id"], "why": why(l)[0], "risk": why(l)[1]} for l in sorted(safe, key=lambda l: -l["p"]) if ok(l)]
    v = [{"id": l["id"], "why": why(l)[0], "risk": why(l)[1]} for l in sorted(pay, key=lambda l: l["be"] - l["p"]) if ok(l)]
    return {"safest": s, "payout": v, "skip": [], "note": ""}


def pick(data, safe, pay):
    L = {l["id"]: l for l in safe + pay}
    text = prompt_data(data, safe, pay)
    by, resp, err = "model", None, None
    if os.environ.get("ANTHROPIC_API_KEY") and (safe or pay):
        try:
            resp = ask(rules() + "\n\n" + text)
            by = "ai"
        except Exception as e:  # keep the page working if the AI call fails
            err = f"{type(e).__name__}: {e}"[:200]
            print("AI pick failed, using the model's ranking:", err)
    base = model_only(safe, pay, data["players"])
    if resp is None:
        resp = base
    safest, used = enforce(resp.get("safest") or [], L, MAX_SAFE)
    payout, used = enforce(resp.get("payout") or [], L, MAX_PAY, used)
    if by == "ai":  # top up from the model's ranking if the AI returned a short list
        if len(safest) < 4:
            more, used = enforce(base["safest"], L, MAX_SAFE, used)
            safest += more[: 4 - len(safest)]
    skip = [{"id": s["id"], "why": clean(s.get("why"))} for s in (resp.get("skip") or []) if s.get("id") in L][:6]
    return {"by": by, "model": MODEL if by == "ai" else None, "at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
            "safest": safest, "payout": payout, "skip": skip, "note": clean(resp.get("note"))[:700], "error": err}, text
