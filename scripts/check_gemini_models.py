# -*- coding: utf-8 -*-
"""Which Gemini models can this key actually use, and which take tools?

CDH_MODEL has to be an exact model ID. A wrong one fails with 404 NOT_FOUND,
and because Gemini is now the *reserve* provider that error stays hidden until
OpenRouter runs dry — usually mid-run, which is the worst moment to discover it.
So check the name before setting it rather than after.

The agent drives eighteen tools, so a model without function calling is no use
here however fast it is. ListModels reports that per model, in
supportedGenerationMethods.

    $env:GOOGLE_API_KEY = "..."      # PowerShell
    python scripts/check_gemini_models.py
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
URL = "https://generativelanguage.googleapis.com/v1beta/models"
WANT = os.environ.get("CDH_MODEL", "gemini-3.1-flash-lite")


def fetch() -> list[dict]:
    """Every model this key can see, following pagination."""
    out, token = [], ""
    while True:
        url = f"{URL}?key={KEY}&pageSize=200" + (f"&pageToken={token}" if token else "")
        with urllib.request.urlopen(url, timeout=60) as r:
            data = json.loads(r.read())
        out += data.get("models", [])
        token = data.get("nextPageToken", "")
        if not token:
            return out


def main() -> int:
    if not KEY:
        print("Set GOOGLE_API_KEY first.")
        return 1
    if "--probe" in sys.argv:
        return run_probes()
    try:
        models = fetch()
    except urllib.error.HTTPError as exc:
        print(f"{exc.code}: {exc.read().decode(errors='ignore')[:200]}")
        return 1

    # "models/gemini-2.5-flash" is how the API names them; CDH_MODEL takes the
    # bare id, so strip the prefix to show exactly what you would paste.
    rows = []
    for m in models:
        mid = m["name"].split("/", 1)[-1]
        methods = m.get("supportedGenerationMethods") or []
        if "generateContent" not in methods:
            continue            # embedding and token-count models are not usable here
        rows.append((mid, m.get("inputTokenLimit", 0), m.get("displayName", "")))
    rows.sort()

    print(f"{len(rows)} models support generateContent\n")
    print(f"{'CDH_MODEL value':46} {'input limit':>12}  name")
    print("-" * 92)
    for mid, lim, disp in rows:
        mark = " <- currently configured" if mid == WANT else ""
        print(f"{mid:46} {lim:>12,}  {disp}{mark}")

    if not any(mid == WANT for mid, _, _ in rows):
        print(f"\nCDH_MODEL is set to {WANT!r}, which this key cannot reach.")
        print("Copy one of the ids above instead — the agent will 404 otherwise.")
    else:
        print(f"\n{WANT} is available to this key.")
    print("\nPick by tool-calling reliability, not speed: the agent drives eighteen "
          "tools behind a ~78k-token prompt, and a weaker model tends to narrate "
          "the steps rather than call them.")
    return 0




# ---------------------------------------------------------------------------
# Tool-calling probe
#
# ListModels says a model supports generateContent. It does not say whether it
# will actually emit a function call, and that is the only property this agent
# needs: it drives eighteen tools, and a model that prefers prose stalls without
# ever looking broken. So ask each candidate to call one, through the same SDK
# path the app uses, and see what comes back.
#
# Each probe spends one request against that model's daily allowance. Quotas are
# per model, not per key: a 429 from one model says nothing about the others,
# and a model with no free tier at all answers 429 on its first request.
#
# `--probe` tests the configured CDH_MODEL only; `--probe --all` also tries the
# shortlist of alternatives.
# ---------------------------------------------------------------------------
SHORTLIST = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-flash-lite-latest",
]

_TOOL = {
    "name": "get_yield",
    "description": "Return the simulated maize yield for a district.",
    "parameters": {
        "type": "object",
        "properties": {
            "district": {"type": "string", "description": "District name."},
            "year": {"type": "integer", "description": "Season, e.g. 2020."},
        },
        "required": ["district", "year"],
    },
}
_ASK = ("What was the maize yield in Nandom in 2020? "
        "Use the tool - you cannot know this without it.")


def probe(model: str) -> tuple[str, str]:
    """One real call. Returns (verdict, detail)."""
    from google import genai
    from google.genai import types as gt
    client = genai.Client(api_key=KEY)
    try:
        resp = client.models.generate_content(
            model=model,
            contents=[gt.Content(role="user", parts=[gt.Part(text=_ASK)])],
            config=gt.GenerateContentConfig(
                tools=[gt.Tool(function_declarations=[_TOOL])], temperature=0),
        )
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
            return "QUOTA", ("no free allowance left for this model today "
                             "(or none at all: limit 0 means it is paid-only)")
        if "NOT_FOUND" in msg or "404" in msg:
            return "404", "this key cannot reach that model"
        return "ERROR", f"{type(exc).__name__}: {msg[:90]}"

    cand = (resp.candidates or [None])[0]
    if cand is None or not cand.content:
        return "EMPTY", "no content returned"
    parts = cand.content.parts or []
    for part in parts:
        fc = getattr(part, "function_call", None)
        if fc:
            return "TOOLS", f"{fc.name}({dict(fc.args or {})})"
    text = "".join(getattr(p, "text", "") or "" for p in parts).strip()
    return ("prose", text.replace(chr(10), " ")[:70]) if text else ("EMPTY", "nothing")


def run_probes() -> int:
    models = [WANT] + ([m for m in SHORTLIST if m != WANT]
                       if "--all" in sys.argv else [])
    print(f"Probing {len(models)} model(s) - one request each, against that")
    print("model's own free-tier allowance.")
    print()
    print(f"{'model':30} {'result':7} detail")
    print("-" * 92)
    good, tested, stopped = [], 0, False
    for m in models:
        verdict, detail = probe(m)
        print(f"{m:30} {verdict:7} {detail}", flush=True)
        if verdict == "QUOTA":
            # Nothing was learned about this model, but quotas are per model,
            # so carry on to the next one rather than giving up on the day.
            stopped = True
            continue
        tested += 1
        if verdict == "TOOLS":
            good.append(m)
    print()
    if good:
        print("Models that emitted a tool call:")
        for g in good:
            print(f"   CDH_MODEL={g}")
        print()
        print("Prefer a pinned version over a *-latest alias: an alias can change")
        print("under a running Space, and a model swap shows up as odd behaviour")
        print("rather than as an error.")
    elif tested == 0:
        print("No model was actually tested - every probe hit a quota. This says")
        print("nothing about whether these models can call tools.")
        print()
        print("Free allowances reset daily (midnight Pacific). A model that is")
        print("QUOTA on its very first request of the day has no free tier.")
    else:
        print(f"Of the {tested} model(s) actually tested, none emitted a tool")
        print("call. The agent needs function calling, not prose.")
    if stopped and tested:
        print()
        print("Models marked QUOTA were not tested; try them again tomorrow.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
