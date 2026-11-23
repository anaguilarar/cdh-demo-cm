# -*- coding: utf-8 -*-
"""Which free OpenRouter models can actually call tools?

The agent is useless without tool calling — it has to drive seventeen MCP tools
to get anywhere. OpenRouter's free tier does not say which models support it, and
a model that quietly answers in prose instead of emitting a tool call looks like
it is working right up until nothing happens.

So ask each one to call a trivial tool and see what comes back.

    $env:OPENROUTER_API_KEY = "sk-or-..."    # PowerShell
    python scripts/check_openrouter_models.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

URL = os.environ.get("CDH_OPENROUTER_URL",
                     "https://openrouter.ai/api/v1/chat/completions")
KEY = os.environ.get("OPENROUTER_API_KEY")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36")

# Discovered rather than hard-coded: OpenRouter retires ":free" slugs often, and a
# stale list reports "404 unavailable for free" as though the model had declined to
# call a tool. /models is public, lists pricing, and flags which models accept
# `tools` — so ask it, then verify the claim by actually calling one.
def free_tool_models(limit: int = 8) -> list[str]:
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/models",
        headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        models = json.loads(r.read()).get("data", [])
    out = []
    for m in models:
        pr = m.get("pricing") or {}
        free = (str(pr.get("prompt", "1")) in ("0", "0.0")
                and str(pr.get("completion", "1")) in ("0", "0.0"))
        if free and "tools" in (m.get("supported_parameters") or []):
            out.append(m["id"])
    return out[:limit]


# Deliberately unanswerable without the tool, so a model that prefers prose is
# making a choice rather than lacking information.
TOOLS = [{
    "type": "function",
    "function": {
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
    },
}]
ASK = ("What was the maize yield in Nandom in 2020? "
       "Use the tool — you cannot know this without it.")


def probe(model: str) -> tuple[str, str]:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": ASK}],
        "tools": TOOLS,
        "temperature": 0,
        # Without a budget a reasoning model can spend the entire reply thinking
        # and return empty content with no tool call — which looks exactly like
        # a model that cannot use tools.
        "max_tokens": 512,
    }).encode()
    req = urllib.request.Request(
        URL, data=body, method="POST",
        headers={"Authorization": f"Bearer {KEY}",
                 "Content-Type": "application/json",
                 # Cloudflare fronts this endpoint and denies urllib's default
                 # agent with "error code: 1010" before any model sees the
                 # request — a uniform, instant 403 that looks like the models
                 # rejecting tools when nothing was ever asked of them.
                 "User-Agent": UA,
                 # OpenRouter attributes usage by these two.
                 "HTTP-Referer": "https://cgiar-climate-data-hub.github.io/skills/",
                 "X-Title": "CDH Crop Yield Explorer",
                 "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="ignore")[:120].replace("\n", " ")
        if exc.code == 403 and "1010" in detail:
            # Cloudflare refused the client, so no model was ever consulted.
            # Reporting this as "no tool support" would be a false verdict.
            return "BLOCK", "Cloudflare refused the client, not the model"
        if "FreeTierError" in detail:
            return "POLICY", "free tier is restricted to OpenCode's own client"
        return "HTTP", f"{exc.code} {detail}"
    except Exception as exc:  # noqa: BLE001
        return "ERROR", f"{type(exc).__name__}: {str(exc)[:90]}"

    # An upstream failure can arrive as HTTP 200 carrying an `error` body instead
    # of a completion. CDH-metadata-app's chat.js documents this exact case: it
    # reads as "(empty response)", indistinguishable from a model declining to
    # answer, so it has to be checked before looking at choices.
    if data.get("error"):
        err = data["error"]
        return "ERROR", str(err.get("message", err))[:80]

    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    calls = msg.get("tool_calls") or []
    if calls:
        fn = (calls[0].get("function") or {})
        return "TOOLS", f"{fn.get('name')}({str(fn.get('arguments'))[:60]})"
    text = (msg.get("content") or "").strip().replace("\n", " ")
    if not text:
        # Empty with no tool call is its own finding, not prose. finish_reason
        # separates "ran out of budget thinking" from "chose to say nothing".
        return "EMPTY", f"no content, no call (finish={choice.get('finish_reason')})"
    return "prose", text[:70]


def main() -> int:
    if not KEY:
        print("Set OPENROUTER_API_KEY first.")
        return 1
    print(URL)
    try:
        models = free_tool_models()
    except Exception as exc:  # noqa: BLE001
        print(f"could not list models: {type(exc).__name__}: {exc}")
        return 1
    if not models:
        print("No free model currently advertises tool support.")
        return 0
    print(f"{len(models)} free models advertise tool support — verifying each\n")
    print(f"{'model':38} {'result':6} detail")
    print("-" * 96)
    usable, verdicts = [], []
    for model in models:
        t = time.time()
        verdict, detail = probe(model)
        verdicts.append(verdict)
        if verdict == "TOOLS":
            usable.append(model)
        print(f"{model:38} {verdict:6} {detail}   [{time.time()-t:.1f}s]", flush=True)

    print()
    if usable:
        print("Free models that emitted a tool call:")
        for u in usable:
            print(f"   CDH_OPENROUTER_MODEL={u}")
        print("\nPick one of these — a model that answers in prose cannot drive the agent.")
    elif all(v == "BLOCK" for v in verdicts):
        print("Every request was blocked by Cloudflare before reaching a model, so "
              "this says nothing about tool support.")
        print("Either the endpoint does not accept server-side calls, or it wants "
              "headers this script is not sending. Worth checking whether OpenCode "
              "intends this endpoint for its own CLI rather than third-party apps.")
    else:
        print("None of the free models emitted a tool call.")
        print("They can still narrate a finished run — which covers the failures seen")
        print("so far, both of which hit after the tools had already executed — but")
        print("they cannot start one. Either keep a paid provider for the tool-driving")
        print("turns, or adopt the <fill>-style text protocol CDH-metadata-app uses,")
        print("which needs no function calling at all.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
