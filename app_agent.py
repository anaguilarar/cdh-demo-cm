# -*- coding: utf-8 -*-
"""Crop Yield Explorer — the agent version.

`app.py` is a form: the skill's logic written out in Python. This is the other
half of the argument — the skill *is* the behaviour. `SKILL.md` is fetched from
the repository and becomes the system prompt, and the work is done by the two
packages' own tools:

    aggeodata      download climate + soil, build datacubes
    ag-cube-cm     run DSSAT per pixel

Nothing here knows what CHIRPS is, which CRS SoilGrids ships in, or that soil
must be averaged onto the climate grid. That knowledge lives in the skill and in
the packages, so this file cannot drift from them.

Every visitor gets their own work directory. Results, configs and the yield map
are scoped to the browser session, so two people using the Space at once never
see each other's runs, and a restart never shows last week's files.
"""
from __future__ import annotations

import asyncio
import contextvars
import datetime as dt
import functools
import importlib
import inspect
import json
import logging
import os
import re
import pathlib
import shutil
import tempfile
import time
import traceback
import urllib.error
import urllib.request

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go   # imported here: lazily it adds seconds to the first draw

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("agent-space")
NL = chr(10)   # literal newline, kept out of f-strings

# --------------------------------------------------------------------------
# Configuration — every knob is an env var so the Space can be retargeted
# without editing code.
# --------------------------------------------------------------------------
REPO = os.environ.get("CDH_REPO", "CGIAR-Climate-Data-Hub/skills")
# Three skills, because the Space covers the whole chain. The crop modeller
# alone leaves the agent holding a dozen acquisition tools with no idea that
# SoilGrids ships in Interrupted Goode Homolosine, that wv* layers are 1 km while
# the rest are 250 m, or that soil has to be averaged onto the climate grid.
# Only the crop-modeler skill by default. The two download skills are 52k of
# text the agent cannot act on here (acquire_region does that work in one
# call), and on the free tier every character is resent on every request:
# with all three loaded, one turn of five tool calls hits Gemini's per-minute
# input-token limit. Add them back with CDH_SKILL if a richer prompt is wanted.
SKILLS = [s.strip() for s in os.environ.get(
    "CDH_SKILL", "spatial-crop-modeler"
).split(",") if s.strip()]
SKILL = SKILLS[0]                       # the one the page is named after
# Tracks `main` by default, so edits to the published skill reach this Space on
# its next restart. Pin to a tag or commit SHA when the demo needs to be stable:
#   CDH_SKILL_URL=https://raw.githubusercontent.com/<repo>/v0.4.1/.../SKILL.md
# Flash-Lite, not Flash: on the free tier it has the largest daily request
# allowance of the Gemini models, and this agent needs tool calling more than
# reasoning depth. Confirm the id with scripts/check_gemini_models.py.
MODEL = os.environ.get("CDH_MODEL", "gemini-3.5-flash-lite")
# The Gemini models tried in order, each with its own free quota: MODEL first,
# then these (exact ids from the Gemini API model list; no Gemma).
GEMINI_FALLBACKS = [m.strip() for m in os.environ.get(
    "CDH_GEMINI_FALLBACKS", "gemini-3.5-flash-lite,gemini-3.1-flash-lite").split(",") if m.strip()]
# Bring your own key: built, but off until it has passed its live tests (on a
# private duplicate Space). Off: no key panel, no settings accordion, and a run
# that finds every shared model out of quota ends with a plain message.
BYOK_ENABLED = os.environ.get("BYOK_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")
GEMINI_MODELS = list(dict.fromkeys([MODEL] + GEMINI_FALLBACKS))
MAX_STEPS = int(os.environ.get("CDH_MAX_STEPS", "30"))
# A tool that never returns is worse than one that fails: the chat shows
# "running x..." indefinitely and nothing says why. DSSAT over a district is well
# under a minute, so anything past this is wrong rather than slow.
TOOL_TIMEOUT = float(os.environ.get("CDH_TOOL_TIMEOUT", "300"))
# Acquisition is minutes by nature — boundary lookup, four climate variables,
# then one WCS request per soil layer and depth. Espinal 2020 measured ~3.5 min,
# so the 300 s that suits every other tool would abandon a run that was working.
ACQUIRE_TIMEOUT = float(os.environ.get("CDH_ACQUIRE_TIMEOUT", "1500"))

# Work tree. `sessions/` holds one folder per browser session and is wiped on
# every start, so a redeploy never shows a previous visitor's results. `shared/`
# keeps acquired cubes, which are data rather than results and expensive to
# rebuild, so a second visitor asking for the same district gets them for free.
WORK = pathlib.Path(os.environ.get("CDH_WORK", tempfile.gettempdir())) / "cdh_agent"
SESSIONS = WORK / "sessions"
SHARED = WORK / "shared"
shutil.rmtree(SESSIONS, ignore_errors=True)
SESSIONS.mkdir(parents=True, exist_ok=True)
SHARED.mkdir(parents=True, exist_ok=True)
# Sessions older than this are pruned on the next turn. Results are declared
# ephemeral in the UI, so this only bounds disk use on a long-running Space.
SESSION_TTL_S = float(os.environ.get("CDH_SESSION_TTL", str(6 * 3600)))

# Pre-built cubes. A district is ~1 MB in Zarr, so having two ready keeps the
# demo everyone actually clicks fast, while leaving the download path reachable
# for anyone who asks for somewhere else. They live in a Hub dataset, not in
# this repository, pinned to one commit so a rebuilt Space reads the same bytes,
# and are fetched at start-up into the layout the app reads
# (`<place>_cube_<kind>.zarr/` side by side). If that fails the app still
# starts: the examples say so, and fresh downloads work as before.
CUBES_REPO = os.environ.get("CDH_CUBES_REPO", "aaguilar90/cdh-demo-cubes")
CUBES_REVISION = os.environ.get("CDH_CUBES_REVISION", "cfc905dd1d078a837d261a051815a9c1e303f234")
PREBAKED = pathlib.Path(os.environ.get("CDH_CUBES_DIR", str(WORK / "cubes")))
CUBES_UNAVAILABLE = (
    "The pre-built data for the examples could not be loaded when the demo "
    "started, so this example is not available right now. You can still ask about "
    "any place and season: its data will be downloaded fresh, which takes a few "
    "minutes.")


def _fetch_cubes() -> bool:
    """Put the pinned cubes under PREBAKED; False if they could not be had.

    A revision is a commit and never changes, so a folder already holding it
    (the marker is written only after a complete download) is used as is."""
    marker = PREBAKED / ".revision"
    try:
        if marker.read_text().strip() == CUBES_REVISION:
            return True
    except OSError:
        pass
    shutil.rmtree(PREBAKED, ignore_errors=True)   # another revision, or a partial one
    # Three request lines per file otherwise (huggingface_hub logs as "httpx2").
    quiet = [logging.getLogger(n) for n in ("httpx", "httpx2")]
    levels = [q.level for q in quiet]
    for q in quiet:
        q.setLevel(logging.WARNING)
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(CUBES_REPO, repo_type="dataset", revision=CUBES_REVISION,
                          local_dir=PREBAKED, allow_patterns=["*_cube_*.zarr/*"])
        marker.write_text(CUBES_REVISION)
        log.info("pre-built cubes: %s@%s", CUBES_REPO, CUBES_REVISION[:10])
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("pre-built cubes unavailable (%s@%s): %s: %s", CUBES_REPO,
                    CUBES_REVISION[:10], type(exc).__name__, str(exc)[:200])
        shutil.rmtree(PREBAKED, ignore_errors=True)
        return False
    finally:
        for q, lv in zip(quiet, levels):
            q.setLevel(lv)


CUBES_READY = _fetch_cubes()

# Jobs = pixels x planting windows x seasons. DSSAT runs roughly ten a second
# on two cores, so this is about two minutes of simulation — the limit that
# matters is patience, not memory. Sixty was too tight to ask a planting-date
# question at all: ten dates over five seasons is five hundred jobs before a
# single pixel is chosen.
MAX_SIMULATIONS = int(os.environ.get("CDH_MAX_SIMULATIONS", "1200"))
# How often to look for a finished run. Long enough to be invisible in cost,
# short enough that the result appears while the user is still reading the
# reply that announced it.
POLL_SECONDS = float(os.environ.get("CDH_POLL_SECONDS", "4"))
NCORES = int(os.environ.get("CDH_NCORES", "2"))
SEP = "__"
TODAY = dt.date.today()


# --------------------------------------------------------------------------
# Credentials for solar radiation
# --------------------------------------------------------------------------
#: Secret holding the Copernicus CDS personal access token. On a Space set it
#: under Settings -> Secrets; locally, put it in a gitignored .env.
CDS_KEY_ENV = "CDSAPI_KEY"
CDS_URL_ENV = "CDSAPI_URL"
DEFAULT_CDS_URL = "https://cds.climate.copernicus.eu/api"


def configure_cds() -> bool:
    """Materialise ~/.cdsapirc from the environment. Returns True if configured.

    aggeodata's ARCO reader passes the key as a bearer token and reads it from
    that file, so writing it once at startup is enough. The key is never logged
    and never written into the repo — on a Space it arrives as an injected
    secret, and ``~`` is outside the git working tree.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    key = (os.getenv(CDS_KEY_ENV) or "").strip()
    if not key:
        return False
    url = (os.getenv(CDS_URL_ENV) or DEFAULT_CDS_URL).strip()
    rc = pathlib.Path.home() / ".cdsapirc"
    try:
        rc.write_text(f"url: {url}\nkey: {key}\n", encoding="utf-8")
        rc.chmod(0o600)
    except OSError as exc:
        log.warning("could not write %s: %s", rc, exc)
        return False
    return True


# A key in the environment (the Space's secret) or an existing ~/.cdsapirc
# (a workstation that already uses the CDS) both count.
HAS_CDS = configure_cds() or (pathlib.Path.home() / ".cdsapirc").is_file()
# Solar radiation is not optional: DSSAT reads SRAD from the weather file and
# does not derive it from the temperature range, so without it every pixel
# yields zero. AgERA5 (0.1°) whenever a CDS key is present; NASA POWER (0.5°,
# no credentials) only as the fallback so a fork of this Space still works.
# CDH_RSDS_SOURCE pins one or the other.
RSDS_SOURCE = (os.environ.get("CDH_RSDS_SOURCE") or
               ("agera5" if HAS_CDS else "nasa_power")).strip().lower()
if RSDS_SOURCE == "agera5" and not HAS_CDS:
    log.warning("CDH_RSDS_SOURCE=agera5 but no CDS key was found (CDSAPI_KEY or "
                "~/.cdsapirc); radiation downloads will fail")
elif not HAS_CDS:
    log.warning("no CDS key: solar radiation falls back to NASA POWER at 0.5 deg; "
                "set CDSAPI_KEY for AgERA5 at 0.1 deg")
log.info("solar radiation source: %s", RSDS_SOURCE)


# --------------------------------------------------------------------------
# Skill text
# --------------------------------------------------------------------------
def _skill_url(name: str) -> str:
    if os.environ.get("CDH_SKILL_URL") and name == SKILLS[0]:
        return os.environ["CDH_SKILL_URL"]
    return (f"https://raw.githubusercontent.com/{REPO}/main/"
            f".agents/skills/{name}/SKILL.md")


def load_skill() -> str:
    """Fetch each SKILL.md — the same files `npx skills add` installs.

    Fetched at startup rather than vendored so the Space tracks what is
    published. A failure becomes text rather than an exception: a Space that
    starts and explains it is misconfigured beats one that will not start.
    """
    parts = []
    for name in SKILLS:
        url = _skill_url(name)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "cdh-space/1.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                text = r.read().decode("utf-8")
            log.info("skill %-24s %6d chars", name, len(text))
            parts.append(f"===== SKILL: {name} =====" + NL + NL + text)
        except Exception as exc:  # noqa: BLE001
            log.error("skill %s failed: %s", name, exc)
            parts.append(f"(The {name} skill could not be fetched from {url}: "
                         f"{type(exc).__name__}: {exc}. Say so if asked to do "
                         "something that needs it.)")
    return (NL * 2).join(parts)


SKILL_TEXT = load_skill()


# --------------------------------------------------------------------------
# Sessions
# --------------------------------------------------------------------------
class Session:
    """One browser tab's work: its folder, its run manifest, its last result.

    The manifest is the source of truth for the result panel. Scanning the
    disk for anything with an HWAM variable found files from other sessions
    and from before the last restart, and opened every NetCDF under the tree
    to do it. A run is listed when the tool that produced it says so, with the
    crop, nitrogen rate and sowing date it was produced with.
    """

    def __init__(self, sid: str):
        self.id = sid
        self.dir = SESSIONS / sid
        self.dir.mkdir(parents=True, exist_ok=True)
        self.last_result = ""     # raw text of the last tool call, for salvage
        self.seen_sig = None      # what the result panel last drew
        self.status = ""          # one line for the result panel while a turn runs
        self.seen_status = None
        # Plan gate, kept here rather than read back from the chat text:
        # plan_shown -- the last turn ended with the agent's reply and nothing
        #               slow ran, so that reply was a plan (or a revision of one);
        # approved   -- this turn's message confirmed it; DSSAT may run;
        # ran_slow   -- this turn ran DSSAT or a real download.
        self.plan_shown = False
        self.approved = False
        self.ran_slow = False
        self.turn_tools: set[str] = set()   # tools that ran this turn
        self.places: dict = {}             # find_place answers, by (name, country)
        self.region: dict | None = None    # area and cubes fixed by acquire_region
        self.sowing: dict | None = None    # sowing dates fixed by acquire_region
        self.last_message = ""             # this turn's request, for a fallback plan
        # A run paused because every shared model is out of quota: the
        # conversation so far (tool results included) and the gate state, so a
        # visitor's own key resumes it at the same step. Never holds the key.
        self.paused: dict | None = None
        self.provider = ""                 # who answered last, e.g. "shared Gemini"
        # The conversation as the server saw it: one entry per finished turn,
        # with the reply as shown. A message sent while the previous turn was
        # still running arrives with the history from before that turn (the
        # plan missing), and the model then wrote the plan again after "yes".
        self.exchanges: list[dict] = []
        self.turn_lock: asyncio.Lock | None = None   # one turn at a time
        self.seen_keyui = None

    @property
    def manifest(self) -> pathlib.Path:
        return self.dir / "runs.json"

    def runs(self) -> list[dict]:
        """Recorded runs, newest first, dropping any whose file has gone."""
        try:
            recs = json.loads(self.manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        recs = [r for r in recs if pathlib.Path(r.get("file", "")).is_file()]
        return sorted(recs, key=lambda r: r.get("created", 0), reverse=True)

    def add_run(self, rec: dict) -> None:
        recs = [r for r in self.runs() if r["file"] != rec["file"]]
        recs.append(rec)
        self.manifest.write_text(json.dumps(recs, indent=1, default=str),
                                 encoding="utf-8")

    def touch(self) -> None:
        try:
            os.utime(self.dir)
        except OSError:
            pass


_SESSIONS: dict[str, Session] = {}


def session_of(request) -> Session:
    sid = getattr(request, "session_hash", None) or "local"
    sid = re.sub(r"[^A-Za-z0-9_-]", "", str(sid))[:40] or "local"
    sess = _SESSIONS.get(sid)
    if sess is None:
        sess = _SESSIONS[sid] = Session(sid)
        _prune_sessions()
    sess.touch()
    return sess


def _prune_sessions() -> None:
    cutoff = time.time() - SESSION_TTL_S
    for d in SESSIONS.iterdir():
        try:
            if d.is_dir() and d.stat().st_mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)
                _SESSIONS.pop(d.name, None)
        except OSError:
            continue


def _run_label(rec: dict) -> str:
    """How a run reads in the dropdown: what was grown, where, when, with what."""
    bits = [rec.get("crop") or "Crop", rec.get("region") or rec.get("country") or ""]
    seasons = rec.get("seasons") or []
    if seasons:
        bits.append(f"{seasons[0]}" if len(seasons) == 1 else f"{seasons[0]}–{seasons[-1]}")
    n = rec.get("n_kg_ha")
    if n is not None:
        bits.append(f"{n:g} kg N")
    if (rec.get("n_windows") or 1) > 1:
        # "dates", not "windows": shorter on a phone, and it is what the
        # "Sowing date" picker beside it calls them.
        bits.append(f"{rec['n_windows']} dates")
    return " · ".join(b for b in bits if b)


# --------------------------------------------------------------------------
# System prompt
# --------------------------------------------------------------------------
def prebaked_regions() -> dict[str, dict]:
    """The shipped cubes, keyed by region tag, with their exact paths."""
    regions: dict[str, dict] = {}
    if not CUBES_READY or not PREBAKED.is_dir():
        return regions
    for p in sorted(PREBAKED.glob("*_cube_*")):
        if not p.is_dir():
            continue
        tag, _, rest = p.name.partition("_cube_")
        kind = "weather" if rest.startswith("climate") else "soil"
        regions.setdefault(tag, {})[kind] = p
        if kind == "weather":
            regions[tag]["season"] = rest.replace("climate_", "").replace(".zarr", "")
    return {t: v for t, v in regions.items() if "weather" in v and "soil" in v}


def prebaked_note() -> str:
    """List the shipped cubes with their EXACT paths.

    Naming the region and the folder is not enough: given only "nandom (under
    cubes/)" a model will happily invent `cubes/nandom/soil_nandom.nc`, write it
    into a config, and the simulation then blocks on files that do not exist.
    Spelling out the full path is the difference between copying and guessing.
    """
    ready = prebaked_regions()
    if not ready:
        return "No pre-built cubes are shipped with this Space."
    lines = ["These cubes are already built. Use them verbatim — copy the paths "
             "exactly as written, do not construct your own:"]
    for tag, v in sorted(ready.items()):
        lines.append(f"  {tag} (season {v.get('season', '?')} only — the weather "
                     f"cube holds that calendar year and nothing else)")
        lines.append(f"      weather_path: {v['weather']}")
        lines.append(f"      soil_path:    {v['soil']}")
    return NL.join(lines)


def system_prompt(sess: Session) -> str:
    return f"""You are running the CGIAR Climate Data Hub `{SKILL}` skill for someone
who is not a programmer, is not at a terminal, and is waiting on a web page. The
full skill follows and you should follow it. What differs here is the environment.

TOOLS, NOT COMMANDS

- You have tools from two packages, named `data{SEP}*` (aggeodata: downloads and
  datacube building) and `crop{SEP}*` (ag-cube-cm: DSSAT). Use them.
- Never print a shell command, a YAML config or Python for the user to run. There is
  nowhere for them to run it. If a step cannot be done with a tool, say so plainly.
- Write every output under {sess.dir}. Always pass `save_to` to
  `crop{SEP}generate_config` and give `output_path` a descriptive file name such as
  `yield_maize_nandom_2020_n60.nc` — one file per run, never overwrite a previous
  run's file when comparing scenarios.
- Set `working_path` to {sess.dir / 'dssat'} and `ncores` to {NCORES}.

WHAT IS ALREADY HERE

{prebaked_note()}

THE FLOW (the page fixes the area, the weather and the sowing date)

1. `data{SEP}find_place` once per place, with the name and the ISO3 country
   code. It returns at most five matches with province and district; use
   `best`. Never call it twice for the same place, and never list a country's
   admin units.
2. `data{SEP}acquire_region` for every request, pre-built places included
   (instant there): the place and adm_level from `best`, the season ("2019",
   or "2018/19" for a season that runs into the next year), the crop, the
   number of sowing dates and the days between them, and `planting_date` ONLY
   if the user gave one. The page then fixes, in code, so the same request
   always gives the same result:
   - the area: the boundary of the place named (the map is clipped to it);
   - the weather window: from the sowing dates plus the crop's days to
     maturity, with fixed margins; you do not choose dates;
   - the default sowing date when the user gave none: the onset of rains by a
     fixed rule on CHIRPS rainfall. It returns `sowing_dates` and
     `sowing_rule`; use them as they are.
   For a pre-built place and season it returns the shipped cubes at once; for
   anywhere else it downloads (a few minutes), so it waits for the user's yes.
   It is one call: do not use the individual `download_*` or
   `build_*_datacube` tools, and do not look for a way to run code.
3. `crop{SEP}generate_config`, then `crop{SEP}run_simulation`. The page puts
   the area, cubes and sowing dates from step 2 into the config itself.
4. `crop{SEP}read_yield`, and answer.

- Simulate the place the user named, at the level they named it ("Huila" is
  the department, not Neiva). A large region is fine: the run spreads its
  pixels over the whole area as an even sample; say so in the plan.
- If a tool returns an ERROR, say which step failed and stop.

DATES

- Today is {TODAY.isoformat()}. A season can only be simulated once its weather
  exists, so never choose a season that is still running or in the future. If the
  user gives no year, use the pre-built season for a pre-built place, and otherwise
  the most recent *complete* calendar year, {TODAY.year - 1}.
- Write every date as YYYY-MM-DD.

PLAN, CONFIRM, THEN RUN

- Before the first slow step — a download or `crop{SEP}run_simulation` — show the
  plan: the place as find_place found it (district, province, country), the
  season, the data ("Data: pre-built data cubes (instant)" or "Data: fresh
  download (a few minutes)"), crop and cultivar, the sowing date(s), nitrogen,
  and roughly how long. For a pre-built place call acquire_region before the plan
  so it can show the sowing date. When the sowing date is the default, write it
  as "Sowing: <date> — default sowing date (rule: <sowing_rule>)"; for a
  download the date is computed after the download, so write "Sowing: default
  sowing date, computed from the rainfall after the download (rule: onset of
  rains, Sivakumar 1988)". End by asking them to reply "yes" or change anything.
  Do not run until they answer.
- Fill other gaps with defaults rather than questions: the cultivar from the
  skill's table for that crop and climate, one sowing date, no fertiliser, the
  pre-built season or the most recent complete one. Mark each value you chose.
- `crop{SEP}run_simulation`, and a download, are refused until the user has
  replied yes to your plan. If they reply with a change instead of a yes, send
  the revised plan and ask again; a new sowing date, place or season means a new
  acquire_region call.
- The answer of generate_config (pre-built places) carries a SEASON CHECK saying
  whether every sowing date reaches maturity inside the weather data. If it does
  not fit, say so, propose the dates it gives that fit (or a fresh download that
  covers them), and ask which they want.
- After a "yes", carry out every step without asking again. Say what each tool
  gave you in a line and move on — the user is watching a progress log that
  already names every tool as it runs, so restating the plan reads as stalling.
- A follow-up that changes one thing ("now with 60 kg of nitrogen", "try sowing
  in June") needs only a short plan: name the change in one line and ask them
  to reply yes.

THIS MACHINE IS SMALL

- {NCORES} CPU cores. Keep runs under {MAX_SIMULATIONS} jobs, where jobs =
  pixels x planting windows x seasons. Roughly ten jobs a second, so
  {MAX_SIMULATIONS} is a few minutes. Pass `max_pixels` to `crop{SEP}run_simulation`
  to stay inside it.
- Spend the budget on whatever the question is about. "Which planting date is
  best over five years" is a question about dates, not places: ten dates x five
  seasons x a dozen pixels answers it, and those pixels are a sample of the
  region rather than the point of the run. "Where is yield highest" is the other
  way round — many pixels, one date. Cap pixels, never seasons, because dropping
  a season silently changes what the average means.
- People ask decisions, not runs. Turn the question into the run that answers it:
  "when is the best time to sow", "which sowing date" -> a sowing-date
  comparison (one run, several windows, below); "how much nitrogen", "is
  fertiliser worth it" -> a nitrogen comparison (one run per rate; 0, 60 and
  120 kg N ha⁻¹ if they give no rates); "what yield can I expect", "how much
  will I harvest" -> one run, one sowing date. If they give no year or dates,
  propose them in the plan, say they are your choice, and ask.
- Comparing nitrogen rates means one config and one run per rate, each with its
  own `fertilizer_n_kg_ha` and its own output file. The page draws the response
  curve from those files once they all exist, so run every rate before reporting.
- Comparing sowing dates is the opposite: ONE config and ONE run. Set
  `planting_date` to the first date, `n_planting_windows` to the number of
  dates and `planting_window_days` to the gap between them — four dates two
  weeks apart is `n_planting_windows=4, planting_window_days=14`. Never one run
  per date: the page compares the windows inside a single result file.
- If a request genuinely does not fit, say what it would cost and offer a
  smaller version. Do not quietly shrink it to something that no longer answers
  the question, and do not report a single pixel as if it described a region.
- Nothing is kept after the session. When you finish, tell the user to download
  the results before closing the tab.

HOW TO TALK TO THEM

- Never write a file-system path in a message, a plan or a table. Name a file
  by its file name (`yield_maize_nandom_2020.nc`) and the work folder as
  "session workspace". Paths are for tool arguments only. The page manages
  the session workspace: never ask the user to create, check or choose a
  folder, and leave working paths out of the plan.
- Never write a tool name, a parameter name or an internal value (a mode such
  as `with_cubes`, a variable such as `HWAM`) to the user. Say what it means:
  "using the pre-built data cubes", "grain yield". The page already shows each
  step as it runs.
- Write units as kg ha⁻¹ for yield and kg N ha⁻¹ for nitrogen, never kg/ha.
- Never state a yield, a percentage change or a result file that did not come
  from a tool result in this session. A new rate, date, crop or place has no
  result until you have run it: send the plan for it and wait for yes. Quote
  yields exactly as `crop{SEP}read_yield` gave them. The page withholds any
  reply that quotes numbers no result backs up.
- Ask for what you need in plain language, several things at once, each with a
  sensible default they can simply accept. Never ask for a bounding box or a CRS;
  ask for a place.
- Say what you are about to do before anything slow, and roughly how long.
- After a run, call `crop{SEP}read_yield` and report from it. Answer the
  question first, in one sentence with the number, written like these
  templates with every <…> filled from read_yield: "In <season>, the <date>
  sowing gave the highest mean yield: <mean> kg ha⁻¹." / "In <season>, <rate>
  kg N ha⁻¹ gave the highest mean yield: <mean> kg ha⁻¹, against <mean at 0>
  kg ha⁻¹ with none." / "In <season>, maize in <place> averaged <mean> kg ha⁻¹."
  Then the details: yields
  in kg ha⁻¹, the place and the season, and which planting window did best when
  there is more than one. Give the pixel count as `pixels` from read_yield.
  If `pixels_no_yield` is above 0, say so: "14 of 48 pixels gave no yield,
  possibly too cold, steep or dry there" (they are greyed out on the map and
  included in the mean). If read_yield lists `seasons_left_out`, say which
  seasons and why. Name a season that runs into the next year as both years
  ("the 2018/19 season"), not by its sowing year alone. Quote yields exactly
  as read_yield gives them (whole kg ha⁻¹), as the page shows them. Over several seasons, also give that window's worst and best season
  (`season_means_kg_ha`) — a date that wins on average but fails one year in
  three is a different answer. The map and charts on the right update by themselves.
- Close with the honest limits: one season is not a recommendation, SoilGrids is a
  global model rather than a soil survey, and these runs are rainfed.

--- SKILL ---

{SKILL_TEXT}
"""


# --------------------------------------------------------------------------
# Tool declarations
# --------------------------------------------------------------------------
from google import genai                      # noqa: E402
from google.genai import types as gt          # noqa: E402


def _clean_schema(node):
    """Trim a JSON Schema to the subset Gemini's function declarations accept.

    Unknown keywords (`$schema`, `additionalProperties`, `anyOf`) make Gemini
    reject the whole tool, and the refusal is silent — the model simply never
    calls it and nothing explains why.
    """
    if not isinstance(node, dict):
        return None
    keep = {"type", "description", "enum", "items", "properties", "required"}
    out = {k: v for k, v in node.items() if k in keep}

    t = out.get("type")
    if isinstance(t, list):                      # ["string", "null"] -> "string"
        t = next((x for x in t if x != "null"), "string")
    if not t and isinstance(node.get("anyOf"), list):
        t = next((x.get("type") for x in node["anyOf"]
                  if isinstance(x, dict) and x.get("type") not in (None, "null")),
                 "string")
    out["type"] = t or "string"

    if out["type"] == "object":
        props = {k: _clean_schema(v) for k, v in (out.get("properties") or {}).items()}
        props = {k: v for k, v in props.items() if v}
        if not props:
            return None                          # unusable as a parameter; drop it
        out["properties"] = props
        out["required"] = [r for r in (out.get("required") or []) if r in props]
    elif out["type"] == "array":
        out["items"] = _clean_schema(out.get("items") or {"type": "string"}) \
            or {"type": "string"}
    return out


def _declare(tools, prefix):
    decls = []
    for t in tools:
        schema = _clean_schema(t.inputSchema or {})
        # A no-argument tool must declare NO parameters: an empty object is
        # rejected, and again the refusal is silent.
        decls.append(gt.FunctionDeclaration(
            name=f"{prefix}{SEP}{t.name}",
            description=(t.description or "")[:1024],
            **({"parameters": schema} if schema else {}),
        ))
    return decls


# Tools the agent never needs here. Every declaration costs context on a free
# model, and these invite detours (climate indices, GEE) the skill does not ask for.
HIDDEN = set(os.environ.get(
    "CDH_HIDE_TOOLS",
    f"data{SEP}download_gee,data{SEP}compute_climate_features,"
    f"data{SEP}run_summarization_pipeline,data{SEP}list_available_climate_indices,"
    f"data{SEP}reshape_flat_soil_cube,crop{SEP}run_simulation,"
    # Whole-country name lists; find_place returns the best few instead.
    f"data{SEP}list_admin_units").split(","))

# Tools run in-process rather than over MCP, keyed by their namespaced name.
#
# DSSAT through the MCP stdio transport deadlocks: the identical config finishes
# in 7.9 s called directly, and 7.9 s inside a worker thread like the one FastMCP
# uses, but never returns through the pipe. Rather than push a job that spawns
# hundreds of subprocesses down a channel that is also a protocol, everything
# runs here. The servers are still consulted for tool *names and schemas*, which
# is what they are good at.
MODULES = {"data": "aggeodata.mcp_server", "crop": "ag_cube_cm.mcp_server"}


def _server_functions() -> dict:
    """Every MCP tool, resolved to the plain function behind it."""
    out = {}
    for prefix, module in MODULES.items():
        try:
            mod = importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001
            log.error("cannot import %s: %s", module, exc)
            continue
        for name in dir(mod):
            fn = getattr(mod, name)
            if (callable(fn) and not name.startswith("_")
                    and getattr(fn, "__module__", "") == module):
                out[f"{prefix}{SEP}{name}"] = fn
    return out


def _server_declarations() -> list:
    """Tool schemas, read from each FastMCP server object without a subprocess."""
    async def _list():
        decls = []
        for prefix, module in MODULES.items():
            try:
                mod = importlib.import_module(module)
                tools = await mod.mcp.list_tools()
                decls += _declare(tools, prefix)
            except Exception as exc:  # noqa: BLE001
                log.error("cannot list tools of %s: %s", module, exc)
        return decls
    return asyncio.run(_list())


SERVER_FUNCS = _server_functions()
log.info("%d MCP tools resolved to in-process functions", len(SERVER_FUNCS))


# acquire_region has no MCP server behind it, so it has to be declared by hand or
# the model never learns it exists.
ACQUIRE_DECL = gt.FunctionDeclaration(
    name=f"data{SEP}acquire_region",
    description=(
        "Fix the area, the weather window and the sowing dates for a request, and "
        "return the two simulation-ready datacubes (pre-built ones instantly, "
        "otherwise downloaded and built). Call it for every request, pre-built "
        "places included, before generate_config. The page derives the weather "
        "window and, unless the user gave a sowing date, the default sowing date; "
        "use the values it returns."),
    parameters={
        "type": "object",
        "properties": {
            "country_code": {"type": "string",
                             "description": "ISO3 code, e.g. COL, GHA, HND."},
            "feature_name": {"type": "string",
                             "description": "The place, as find_place's 'best' names it."},
            "adm_level": {"type": "integer",
                          "description": "find_place's adm_level for that place."},
            "season": {"type": "string",
                       "description": "The season: '2019', or '2018/19' when it spans two years."},
            "crop": {"type": "string", "description": "e.g. maize, bean."},
            "planting_date": {"type": "string",
                              "description": "YYYY-MM-DD. ONLY if the user gave a sowing "
                                             "date; leave it out otherwise."},
            "n_planting_windows": {"type": "integer",
                                   "description": "Number of sowing dates to compare (1 if none)."},
            "planting_window_days": {"type": "integer",
                                     "description": "Days between sowing dates (14 if not said)."},
        },
        "required": ["country_code", "feature_name", "season", "crop"],
    },
)

FIND_DECL = gt.FunctionDeclaration(
    name=f"data{SEP}find_place",
    description=(
        "Find an administrative unit by name in one country. Returns at most five "
        "matches with country, province (adm1), district (adm2) and admin level, "
        "best first. Call it once per place and use 'best'."),
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "The place as the user wrote it."},
            "country_code": {"type": "string", "description": "ISO3 code, e.g. COL, RWA, ZMB."},
        },
        "required": ["name", "country_code"],
    },
)

RUN_DECL = gt.FunctionDeclaration(
    name=f"crop{SEP}run_simulation",
    description=("Run the spatial crop model from a saved YAML config. Returns the "
                 "output file, pixel counts and mean yield. Call read_yield "
                 "afterwards to see per-window results."),
    parameters={"type": "object", "properties": {
        "config_path": {"type": "string",
                        "description": "The YAML written by generate_config (save_to)."},
        "max_pixels": {"type": "integer",
                       "description": "Cap on pixels, spread over the region."}},
        "required": ["config_path"]},
)

READ_DECL = gt.FunctionDeclaration(
    name=f"crop{SEP}read_yield",
    description=("Read simulation results: mean, min, max and pixel count per "
                 "planting window, plus which window yielded best, together with "
                 "the crop, nitrogen rate and season each run used. Omit `path` to "
                 "summarise every run of this session, which is how to compare "
                 "separate runs such as different nitrogen rates. Use this before "
                 "reporting results — it is the only way to see what a simulation "
                 "produced."),
    parameters={"type": "object", "properties": {
        "path": {"type": "string",
                 "description": "One result file. Omit to summarise all of them."}}},
)

DECLARED = [d for d in _server_declarations() if d.name not in HIDDEN]
DECLARED += [FIND_DECL, ACQUIRE_DECL, RUN_DECL, READ_DECL]
log.info("declared %d tools", len(DECLARED))


# --------------------------------------------------------------------------
# Progress
#
# The terminal shows a tqdm bar while DSSAT runs; the page showed "running
# run_simulation…" for five minutes. The bar's counts are routed here: a tqdm
# subclass reports into whatever dict the current context names, and the chat
# loop redraws its status line from that dict every couple of seconds. The
# contextvar follows the call into `asyncio.to_thread`, so each session's run
# reports to its own dict even when two run at once.
# --------------------------------------------------------------------------
PROGRESS: contextvars.ContextVar = contextvars.ContextVar("cdh_progress", default=None)


def _install_progress_hook() -> None:
    import tqdm as _tqdm
    base = _tqdm.tqdm

    class _Reporting(base):          # type: ignore[misc,valid-type]
        def update(self, n=1):
            r = super().update(n)
            prog = PROGRESS.get()
            if prog is not None:
                # A disabled bar skips most of __init__, so read defensively.
                prog.update(n=int(getattr(self, "n", 0) or 0),
                            total=int(getattr(self, "total", 0) or 0),
                            desc=str(getattr(self, "desc", "") or ""))
            return r

    # The crop model does `from tqdm import tqdm` at call time, so replacing the
    # package attribute is enough; nothing else here uses tqdm's class identity.
    _tqdm.tqdm = _Reporting


try:
    _install_progress_hook()
except Exception as exc:  # noqa: BLE001
    log.warning("progress hook not installed: %s", exc)


def _stage(text: str) -> None:
    prog = PROGRESS.get()
    if prog is not None:
        prog["stage"] = text
        prog["n"] = prog["total"] = 0


def _fmt_s(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}" if seconds >= 60 else f"{seconds}s"


# What each tool is doing, in the words a visitor would use. The chat and the
# status line show these; the tool names are for the model and the logs.
TOOL_LABELS = {
    "list_admin_units": "Checking the location",
    "find_place": "Checking the location",
    "acquire_region": "Downloading climate and soil data",
    "list_supported_crops": "Checking supported crops",
    "generate_config": "Preparing the simulation",
    "run_simulation": "Running DSSAT",
    "read_yield": "Reading the results",
    "list_results": "Listing the results",
}
# Internal values the model sometimes repeats to the user.
PLAIN_TERMS = {"with_cubes": "pre-built data cubes", "full_pipeline": "fresh download"}


def _tool_label(bare: str) -> str:
    return TOOL_LABELS.get(bare) or (bare.replace("_", " ").capitalize() or "Working")


def _plain_terms(text: str) -> str:
    """Tool names and internal values in a reply, as plain words."""
    for k, v in PLAIN_TERMS.items():
        text = re.sub(rf"`?\b{re.escape(k)}\b`?", v, text)
    # One unit style across tiles, charts and replies.
    text = re.sub(r"\bkg(\s+N)?\s*/\s*ha\b", lambda m: f"kg{m.group(1) or ''} ha⁻¹", text)
    for k, v in TOOL_LABELS.items():
        text = re.sub(rf"`(?:\w+{re.escape(SEP)})?{re.escape(k)}`", v.lower(), text)
    # Yields with thousands separators, as the tiles show them: "8193 kg ha⁻¹"
    # -> "8,193 kg ha⁻¹". Only figures followed by a yield unit; years and
    # dates are left alone.
    text = _YIELD_NUM.sub(lambda m: f"{int(m.group(1)):,}", text)
    # ... and after a statistic's name, where the unit is implied:
    # "(min 483, max 3801)" -> "max 3,801".
    text = _STAT_NUM.sub(lambda m: f"{m.group(1)}{int(m.group(2)):,}", text)
    return _table_separators(text)


_STAT_NUM = re.compile(
    r"(\b(?:min|max|minimum|maximum|mean|average|median|p10|p90)\b[:=]?\s+)(\d{4,})(?![\d,.\-/])(?!\s*(?:season|sowing|harvest|crop))", re.I)


_YIELD_COL = re.compile(r"kg|yield|mean|min|max|p10|p90|range", re.I)
_NOT_YIELD_COL = re.compile(r"pixel|date|season|year|kg\s*N|nitrogen", re.I)


def _table_separators(text: str) -> str:
    """The same inside a markdown table, where the unit sits in the header:
    numbers in yield columns ("Mean yield (kg ha⁻¹)", "Min", "Max") get
    separators; date, season, pixel and nitrogen columns are left alone."""
    lines = text.split("\n")
    cols: list[bool] | None = None
    for i, line in enumerate(lines):
        if not line.lstrip().startswith("|"):
            cols = None
            continue
        cells = line.strip().strip("|").split("|")
        if cols is None:                              # the header row
            cols = [bool(_YIELD_COL.search(c)) and not _NOT_YIELD_COL.search(c) for c in cells]
            continue
        if all(re.fullmatch(r"\s*:?-+:?\s*", c) for c in cells):
            continue                                  # the |---| row
        out = [re.sub(r"(?<![\d,.])(\d{4,})(?=(?:\.\d+)?\s*$)", lambda m: f"{int(m.group(1)):,}", c)
               if j < len(cols) and cols[j] else c for j, c in enumerate(cells)]
        lead = line[:len(line) - len(line.lstrip())]
        lines[i] = lead + "|" + "|".join(out) + "|"
    return "\n".join(lines)


_YIELD_NUM = re.compile(
    r"(?<![\d,.])(\d{4,})(?=(?:\.\d+)?\s*(?:[–-]\s*[\d,]+(?:\.\d+)?\s*)?"
    r"(?:kg|kilograms?)\s*(?:ha⁻¹|ha-1|/\s*ha\b|per\s+hectare))", re.I)


# The progress bars of aggeodata and ag-cube-cm are labelled for developers
# ("CHIRPS (HF) pr", "Stacking dates"); the page says what they are.
_DESC_WORDS = (
    (r"\btasmax\b", "Downloading maximum temperature"),
    (r"\btasmin\b", "Downloading minimum temperature"),
    (r"\brsds\b|agera5|nasa", "Downloading solar radiation"),
    (r"chirps|\bpr\b", "Downloading rainfall"),
    (r"chirts", "Downloading temperature"),
    (r"soil depths|soil cube|soilgrids", "Building the soil cube"),
    (r"stacking dates|processing dates", "Building the climate cube"),
    (r"simulat", "Simulations"),
)


def _plain_desc(desc: str | None) -> str:
    low = str(desc or "").lower()
    for pat, words in _DESC_WORDS:
        if re.search(pat, low):
            return words
    return "Steps" if low else "Simulations"


def _progress_line(bare: str, prog: dict, t0: float) -> str:
    """One line of status for a tool that is still running."""
    elapsed = time.time() - t0
    bits = [f"⏳ {_tool_label(bare)}"]
    if prog.get("stage"):
        bits.append(prog["stage"])
    n, total = prog.get("n", 0), prog.get("total", 0)
    if total:
        k = int(round(10 * n / total))
        eta = (elapsed / n) * (total - n) if n else None
        bar = "▰" * k + "▱" * (10 - k)
        bits.append(f"{bar} {_plain_desc(prog.get('desc'))} {n}/{total} "
                    f"({100 * n // total}%)")
        if eta is not None and n >= 3:
            bits.append(f"~{_fmt_s(eta)} left")
    bits.append(f"{_fmt_s(elapsed)} elapsed")
    return " · ".join(bits)


def _status_text(step: int, bare: str, prog: dict, t0: float) -> str:
    """The progress line, without chat markup, for the result panel."""
    line = _progress_line(bare, prog, t0).replace("⏳ ", "")
    return f"Step {step + 1} · {line}"


def _failed_line(bare: str, detail: str) -> str:
    """A tool that failed: plain words first, the exception folded away.

    The agent explains the failure in its next message; the class name is for
    whoever debugs the Space, not for the person asking about maize.
    """
    detail = detail.replace("<", "&lt;").replace(">", "&gt;").replace("`", "'")[:600]
    return (f"✗ {_tool_label(bare)} did not complete"
            f"<details><summary>details</summary>{NL}{NL}`{detail}`{NL}{NL}</details>")


# --------------------------------------------------------------------------
# Local tools
# --------------------------------------------------------------------------
CLIMATE_SOURCES = {"pr": "chirps", "tasmax": "chirts",
                   "tasmin": "chirts", "rsds": RSDS_SOURCE}
SOIL_LAYERS = ["clay", "sand", "silt", "bdod", "cfvo", "nitrogen", "phh2o", "soc",
               "wv0010", "wv0033", "wv1500"]
SOIL_DEPTHS = ["0-5", "5-15", "15-30", "30-60"]


# Missing weather days a cube may still carry into DSSAT, where ag-cube-cm fills
# them by linear interpolation; it warns above five, and so do we, by refusing.
MAX_GAP_DAYS = 5


def _missing_days(cube_path) -> list[str]:
    """Days absent from inside a weather cube's own date range.

    aggeodata stacks the days in parallel and skips a day that fails, with a
    warning ("Skipping date 20250102: Reindexing only valid with uniquely valued
    Index objects", seen on the Space, not locally). ag-cube-cm then fills the
    day by linear interpolation when it writes the weather file, so a lost day
    is made-up weather rather than a failed run. The ends are the data's own:
    a source that has not published the last days yet is not a gap."""
    import xarray as xr
    p = pathlib.Path(str(cube_path))
    try:
        ds = xr.open_zarr(p) if p.suffix == ".zarr" or p.is_dir() else xr.open_dataset(p)
        with ds:
            days = pd.DatetimeIndex(ds["time"].values).normalize()
    except Exception as exc:  # noqa: BLE001
        log.warning("could not read the days of climate cube %s: %s", p, exc)
        return ["(unreadable)"]
    if len(days) == 0:
        return ["(no days)"]
    full = pd.date_range(days.min(), days.max(), freq="D")
    return [d.strftime("%Y-%m-%d") for d in full.difference(days)]


def _missing_climate_vars(cube_path) -> list[str]:
    """Which of the variables DSSAT needs are absent from a weather cube.

    Radiation is the one that goes missing in practice: its download can fail
    while the others succeed, and a cube without it still looks complete.
    """
    import xarray as xr
    p = pathlib.Path(str(cube_path))
    try:
        ds = xr.open_zarr(p) if p.suffix == ".zarr" or p.is_dir() else xr.open_dataset(p)
        with ds:
            present = set(ds.data_vars)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not open climate cube %s: %s", p, exc)
        return list(CLIMATE_SOURCES)
    return [v for v in CLIMATE_SOURCES if v not in present]


def _install_regular_sampling() -> None:
    """Make capped runs sample a regular stride, until ag-cube-cm ships it.

    The installed sampler picks uneven rows and columns; written back onto the
    union of their coordinates the result has an empty column every third or
    fourth one, and the map shows stripes. A regular stride (every sy-th row,
    sx-th column) gives a coarser grid with no holes. ag-cube-cm carries the
    same change behind `_REGULAR_SAMPLING`; once installed, this is a no-op.
    """
    from ag_cube_cm.spatial.spatial_cm import SpatialCM
    if getattr(SpatialCM, "_REGULAR_SAMPLING", False):
        return
    original = SpatialCM._representative_pixels

    def _regular(self, weather_ds, max_pixels):
        ys, xs = weather_ds.y.values, weather_ds.x.values
        ny, nx = len(ys), len(xs)
        valid = None
        for da in weather_ds.data_vars.values():
            if {"y", "x"}.issubset(da.dims) and da.ndim >= 2:
                probe = da.isel({d: slice(0, 5) for d in da.dims if d not in ("y", "x")})
                dims = [d for d in probe.dims if d not in ("y", "x")]
                ok = np.isfinite(probe).any(dim=dims) if dims else np.isfinite(probe)
                valid = np.asarray(ok.transpose("y", "x").values, dtype=bool)
                break
        if valid is None or not valid.any():
            valid = np.ones((ny, nx), dtype=bool)
        if max_pixels >= int(valid.sum()):
            return original(self, weather_ds, max_pixels)
        best = None
        for s in range(1, max(ny, nx) + 1):
            for sy, sx in ((s, s), (s, s + 1), (s + 1, s)):
                oy, ox = ((ny - 1) % sy) // 2, ((nx - 1) % sx) // 2
                n = int(valid[oy::sy, ox::sx].sum())
                if 0 < n <= max_pixels and (best is None or n > best[0]):
                    best = (n, sy, sx, oy, ox)
            if best is not None and int(valid[::s, ::s].sum()) * 2 < best[0]:
                break
        if best is None:
            return original(self, weather_ds, max_pixels)
        _, sy, sx, oy, ox = best
        picked = [(float(ys[i]), float(xs[j]))
                  for i in range(oy, ny, sy) for j in range(ox, nx, sx) if valid[i, j]]
        log.info("sampling %d of %d cells: every %d row(s), %d column(s)",
                 len(picked), int(valid.sum()), sy, sx)
        return picked

    SpatialCM._representative_pixels = _regular


# ag-cube-cm starts every DSSAT run this many days before sowing (SDATE in the
# experiment file), and names the weather station after that start year.
SIM_LEAD_DAYS = 30


def _install_season_slice() -> None:
    """Give each season job only its own weather, from the day DSSAT starts.

    Older ag-cube-cm swallowed an AttributeError here and gave every job the
    whole series, so DSSAT averaged several seasons into one. The pinned one
    slices by calendar year from the SOWING year, but DSSAT starts 30 days
    before sowing: a sowing in January starts in the previous December, which
    the slice has dropped, and DSSAT stops on every pixel (rc=99; seen live for
    Huila sown 3 Jan 2025 and Aipe sown 1 Jan 2025). Here the slice starts at
    the simulation's start year. Each job is one season (number_years = 1), so
    the extra December changes no average.
    """
    import pandas as pd
    from ag_cube_cm.spatial import spatial_cm
    span = getattr(spatial_cm, "_SEASON_SLICE_DAYS", 250)

    def _slice(ds, time_dim, year, pdate):
        times = pd.DatetimeIndex([pd.Timestamp(t) for t in ds[time_dim].values])
        start_yr = min(year, (pdate - dt.timedelta(days=SIM_LEAD_DAYS)).year)
        harvest_yr = (pdate + dt.timedelta(days=span)).year
        mask = np.asarray(times.year.isin(range(start_yr, harvest_yr + 1)))
        return ds.isel({time_dim: mask})

    spatial_cm._slice_season = _slice


def _fitting_seasons(base_pdate, n_windows: int, step_days: int, crop: str,
                     first, last) -> list[int]:
    """Seasons whose every simulation, from its start (SIM_LEAD_DAYS before
    sowing) to maturity, lies inside the weather record [first, last]. Dates
    are built as SpatialCM builds its jobs: each window's date moved to the
    season's year."""
    from ag_cube_cm.spatial.spatial_cm import _shift_year
    first, last = pd.Timestamp(first), pd.Timestamp(last)
    days = _maturity_days(crop)
    keep = []
    for yr in range(first.year, last.year + 1):
        ok = True
        for w in range(max(1, int(n_windows or 1))):
            p = pd.Timestamp(_shift_year(base_pdate + dt.timedelta(days=w * int(step_days or 0)), yr))
            if (p - pd.Timedelta(SIM_LEAD_DAYS, unit="D") < first
                    or p + pd.Timedelta(days, unit="D") > last):
                ok = False
                break
        if ok:
            keep.append(yr)
    return keep


def _install_season_years() -> None:
    """Simulate only the seasons the weather record can carry to maturity.

    ag-cube-cm keeps a season when sowing + 200 days lands in any calendar year
    present in the weather, so a download from December 2018 to January 2020
    ran three seasons: 2018 (only December's weather), 2019 and 2020 (only
    January's). The two phantom seasons yield 0 on every pixel, and the
    reported mean of a one-season request came out at half its real value
    (Palmira 2019: 974 kg/ha shown, 1,949 real). Here a season is kept only when
    every sowing window, plus the crop's days to maturity from ag-cube-cm's own
    table, falls between the first and last day of the weather."""
    from ag_cube_cm.spatial import spatial_cm

    def _simulation_years(self, weather_ds, time_dim):
        mgmt = self.config.MANAGEMENT
        base = mgmt.planting_date
        if time_dim is None:
            return [base.year]
        times = pd.to_datetime(weather_ds[time_dim].values)
        keep = _fitting_seasons(base, mgmt.n_planting_windows, mgmt.planting_window_days,
                                str(self.config.CROP.name), times.min(), times.max())
        if not keep:
            raise RuntimeError(
                f"no season fits inside the weather data ({times.min():%Y-%m-%d} to "
                f"{times.max():%Y-%m-%d}) for sowing on {base} plus "
                f"{_maturity_days(str(self.config.CROP.name))} days to maturity")
        log.info("seasons to simulate: %s (weather %s to %s)", keep,
                 f"{times.min():%Y-%m-%d}", f"{times.max():%Y-%m-%d}")
        return keep

    spatial_cm.SpatialCM._simulation_years = _simulation_years


for _install in (_install_regular_sampling, _install_season_slice, _install_season_years):
    try:
        _install()
    except Exception as exc:  # noqa: BLE001
        log.warning("%s not installed: %s", _install.__name__, exc)


def _cube_extent(cube_path) -> tuple[int, int]:
    """(calendar years, grid cells) of a weather cube; (1, 0) if unreadable."""
    import xarray as xr
    p = pathlib.Path(str(cube_path))
    try:
        ds = xr.open_zarr(p) if p.suffix == ".zarr" or p.is_dir() else xr.open_dataset(p)
        with ds:
            years = np.unique(ds["time"].dt.year.values)
            ny = ds.sizes.get("y", ds.sizes.get("lat", 0))
            nx = ds.sizes.get("x", ds.sizes.get("lon", 0))
        return max(1, int(len(years))), int(ny * nx)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not read %s: %s", p, exc)
        return 1, 0


def _acquire_sync(country_code, feature_name, adm_level, start_date, end_date):
    """Download climate + soil for one admin unit and return two aligned cubes.

    This exists because the Space's agent has no shell and no editor: it can only
    call tools. The documented flow — author a YAML, run the pipeline, align the
    grids — assumes a workstation, which is why the model reached for a `run_code`
    tool that does not exist here.

    Routing through run_download also buys the Zarr fast path. The per-variable
    MCP tools call CHIRPSDownloader.download() directly, which fetches one file
    per day at one connection: 366 round trips for a season, against a handful of
    chunk reads going direct. That is what exhausted the 300 s tool budget.

    The soil grid is reprojected onto the climate grid here rather than left to
    the model. Skipping it is the worst failure in this pipeline because it is
    silent — every pixel drops out and the run still reports success.

    Cubes land under the shared cache keyed by region and season, so a second
    request for the same place returns in seconds whoever makes it.
    """
    import yaml
    from aggeodata.mcp_server import _resolve_extent
    from aggeodata.pipelines.datacube import run_datacube
    from aggeodata.pipelines.download import run_download
    from aggeodata.transform.soil_cube import SoilDataCubeBuilder

    slug = "".join(c.lower() if c.isalnum() else "_"
                   for c in (feature_name or country_code)).strip("_")
    # The window is derived in code from the sowing dates, so the same request
    # always asks for the same dates; the cache is keyed by them exactly, so two
    # windows in the same years can never share a cube.
    tag = f"{slug}_{adm_level}_{start_date.replace('-', '')}_{end_date.replace('-', '')}"
    root = SHARED / "acquired" / tag
    done = root / "cubes.json"
    if done.is_file():
        try:
            cached = json.loads(done.read_text(encoding="utf-8"))
            if (all(pathlib.Path(cached[k]).exists() for k in ("climate_cube", "soil_cube"))
                    and not _missing_climate_vars(cached["climate_cube"])
                    and not _missing_days(cached["climate_cube"])):
                log.info("acquire %s: cache hit", tag)
                return cached
            log.warning("acquire %s: cached cubes incomplete, rebuilding", tag)
        except (OSError, ValueError, KeyError):
            pass
    root.mkdir(parents=True, exist_ok=True)

    _stage("finding the boundary")
    extent = _resolve_extent(country_code, feature_name, adm_level, None)
    cfg = {
        "DATES": {"starting_date": start_date, "ending_date": end_date},
        "SPATIAL_INFO": {"spatial_file": None, "extent": list(extent)},
        "CLIMATE": {"variables": {v: {"source": s}
                                  for v, s in CLIMATE_SOURCES.items()}},
        "SOIL": {"enabled": True, "layers": SOIL_LAYERS, "depths": SOIL_DEPTHS,
                 # SoilGrids v2 at 1 km from the Hub Zarr store: one read per
                 # variable instead of 44 WCS requests to ISRIC.
                 "backend": os.environ.get("CDH_SOIL_BACKEND", "hf")},
        "GENERAL": {"suffix": tag, "ncores": NCORES, "task": "datacube",
                    "reference_variable": "pr", "use_zarr": True,
                    "download_format": "zarr", "output_format": "zarr",
                    "zarr_codec": "zstd", "zarr_quantize": True,
                    "zarr_sharded": False, "zarr_chunks": None},
        "PATHS": {"output_path": str(root)},
    }
    cfg_path = root / f"{tag}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    log.info("acquire %s: extent=%s", tag, [round(float(e), 3) for e in extent])
    _stage("downloading climate and soil")
    run_download(str(cfg_path))
    _stage("building the climate cube")
    climate = run_datacube(str(cfg_path))
    # A day skipped while stacking would be filled by interpolation downstream.
    # The skip has been intermittent, so build the cube once more to get the
    # real weather; a short gap that survives is left to that interpolation,
    # a long one (made-up rain over weeks) is refused.
    gaps = _missing_days(climate)
    if gaps:
        log.warning("acquire %s: climate cube is missing %d day(s) %s; building it again",
                    tag, len(gaps), gaps[:5])
        shutil.rmtree(climate, ignore_errors=True)
        climate = run_datacube(str(cfg_path))
        gaps = _missing_days(climate)
        if len(gaps) > MAX_GAP_DAYS:
            raise RuntimeError(
                f"the climate cube is missing {len(gaps)} days ({', '.join(gaps[:5])}"
                f"{', …' if len(gaps) > 5 else ''}) after two attempts to build it. "
                "Filling that many days would invent the weather, so the season "
                "cannot be simulated. Tell the user the weather data for those "
                "days could not be assembled and to try again later; do not "
                "suggest other causes.")
        if gaps:
            log.warning("acquire %s: %d day(s) %s still missing; DSSAT input fills "
                        "them by interpolation", tag, len(gaps), gaps)
    # aggeodata logs a failed variable and carries on, and the cube builder
    # then quietly builds without it. A cube with no radiation makes DSSAT
    # return zero on every pixel, so refuse it here, loudly, naming the source.
    missing = _missing_climate_vars(climate)
    if missing:
        raise RuntimeError(
            "the climate cube is missing " + ", ".join(
                f"{v} (from {CLIMATE_SOURCES.get(v, '?')})" for v in missing)
            + ". That download failed, so the season cannot be simulated; "
            "simulating without radiation would report a yield of zero "
            "everywhere. Tell the user which variable and source failed.")
    _stage("building the soil cube")

    # The soil rasters are the only GeoTIFFs this pipeline writes, but name the
    # folder we want rather than taking the first folder with a .tif in it.
    soil_dirs = sorted(d for d in root.rglob("*") if d.is_dir()
                       and "soil" in d.name.lower() and any(d.glob("*.tif")))
    if not soil_dirs:
        soil_dirs = [d for d in root.rglob("*") if d.is_dir() and any(d.glob("*.tif"))]
    if not soil_dirs:
        raise RuntimeError("no SoilGrids rasters were written; soil step failed")
    soil = SoilDataCubeBuilder(
        data_folder=str(soil_dirs[0]), variables=SOIL_LAYERS,
        reference_variable="wv1500", target_crs="EPSG:4326",
    ).build_and_save(output_path=str(root), filename=f"soil_{tag}.nc")

    _stage("aligning soil to the climate grid")
    aligned = _align_soil(soil, climate, root / f"soil_{tag}_aligned.nc")
    res = {"climate_cube": str(climate), "soil_cube": str(aligned),
           "extent": [round(float(e), 4) for e in extent], "config": str(cfg_path),
           "region": feature_name, "country_code": country_code,
           "start_date": start_date, "end_date": end_date}
    done.write_text(json.dumps(res), encoding="utf-8")
    return res


def _align_soil(soil_path, climate_path, out_path):
    """Put the soil cube on the climate cube's grid, or say why it could not be."""
    import rioxarray  # noqa: F401
    import xarray as xr
    from rasterio.enums import Resampling

    opener = xr.open_zarr if str(climate_path).endswith(".zarr") else xr.open_dataset
    with opener(climate_path) as clim, xr.open_dataset(soil_path) as soil:
        ref = clim.isel(time=0, drop=True) if "time" in clim.dims else clim
        ref = ref[list(ref.data_vars)[0]].rio.write_crs("EPSG:4326")
        # Soil is the finer grid (250 m against ~5 km), so averaging preserves the
        # field mean; nearest would hand back one arbitrary sub-pixel per cell.
        out = soil.rio.write_crs("EPSG:4326").rio.reproject_match(
            ref, resampling=Resampling.average)
        out.to_netcdf(out_path)
    return out_path


# Where the shipped cubes were cut from, as scripts/prebake_cubes.py builds them:
# region tag -> (ISO3 country, admin level). A request is the same place only
# when all three agree; "Yoro" the department (level 1) is a different area
# from Yoro the municipality the cube covers.
PREBAKED_PLACES = {"nandom": ("GHA", 2), "yoro": ("HND", 2)}
PREBAKED_LABEL = "pre-built data cubes"


def _slug(text: str) -> str:
    import unicodedata
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    return "".join(c.lower() if c.isalnum() else "_" for c in text).strip("_")


@functools.lru_cache(maxsize=8)
def _cube_span(path: str) -> tuple[str, str, list[float]]:
    """First and last day of a climate cube, and its extent."""
    import xarray as xr
    opener = xr.open_zarr if str(path).rstrip("/\\").endswith(".zarr") else xr.open_dataset
    with opener(path) as ds:
        t = pd.to_datetime(ds["time"].values)
        lat = ds["lat"].values if "lat" in ds.coords else ds["y"].values
        lon = ds["lon"].values if "lon" in ds.coords else ds["x"].values
        ext = [round(float(v), 4) for v in (lon.min(), lat.min(), lon.max(), lat.max())]
    return t.min().strftime("%Y-%m-%d"), t.max().strftime("%Y-%m-%d"), ext


def _prebaked_match(country_code: str, feature_name: str, adm_level, start_date: str,
                    end_date: str) -> dict | None:
    """The shipped cubes, when they already cover exactly what was asked for.

    The prompt tells the model to use them, but a small model sometimes calls
    acquire_region for Nandom 2020 anyway and pays a full download for data that
    ships with the Space. Same place (name, country, admin level) and dates
    inside the cube's own time axis: hand back the cubes. Anything else, even
    one day outside, downloads as before.
    """
    tag = _slug(feature_name)
    place = PREBAKED_PLACES.get(tag)
    cubes = prebaked_regions().get(tag)
    if not place or not cubes:
        return None
    try:
        level = int(adm_level) if adm_level not in (None, "") else 2
    except (TypeError, ValueError):
        return None
    if (country_code or "").upper() != place[0] or level != place[1]:
        return None
    try:
        first, last, ext = _cube_span(str(cubes["weather"]))
        start = pd.Timestamp(start_date or first)
        end = pd.Timestamp(end_date or last)
    except Exception as exc:  # noqa: BLE001
        log.warning("pre-built check for %s failed: %s", tag, exc)
        return None
    if start < pd.Timestamp(first) or end > pd.Timestamp(last) or start > end:
        return None
    return {"climate_cube": str(cubes["weather"]), "soil_cube": str(cubes["soil"]),
            "extent": ext, "region": feature_name, "country_code": place[0],
            "start_date": first, "end_date": last, "source": PREBAKED_LABEL,
            "note": (f"Nothing was downloaded: these are the {PREBAKED_LABEL} shipped "
                     f"with the Space, covering {first} to {last}. Say "
                     f"'{PREBAKED_LABEL}' where the plan or reply names the data.")}


def _prebaked_hint(args: dict) -> str:
    """Appended to a place lookup in a country with shipped cubes.

    The plan is written before acquire_region runs, so its Data line came from
    the model's reading of the prompt, and once said "no pre-built cube for
    Nandom, fresh download" while the code then used the cubes. The lookup is
    the step the agent always takes first; this puts the fact in its answer."""
    cc = str(args.get("country_code", "")).upper()
    lines = []
    for tag, (country, lvl) in PREBAKED_PLACES.items():
        cubes = prebaked_regions().get(tag)
        if country != cc or not cubes:
            continue
        try:
            first, last, _ = _cube_span(str(cubes["weather"]))
        except Exception:  # noqa: BLE001
            continue
        lines.append(
            f"NOTE: {PREBAKED_LABEL} exist for {tag.title()} ({country}, admin level "
            f"{lvl}) covering {first} to {last}. For that place and season the "
            f"plan's Data line is 'Data: {PREBAKED_LABEL} (instant)' and nothing "
            f"is downloaded; acquire_region returns them at once.")
    where = (f"Anywhere else in {cc}" if lines else
             f"There are no {PREBAKED_LABEL} for {cc or 'this country'}, so for any place here")
    lines.append(f"NOTE: {where} the plan's Data line is 'Data: fresh download "
                 "(a few minutes)'; write that line in the plan.")
    return "\n\n" + "\n".join(lines)


# --- place lookup -------------------------------------------------------------
# list_admin_units handed the model a whole country (about 1,100 names for
# Colombia) on every call, and the model called it again and again until Gemini's
# per-minute token limit stopped the turn. find_place returns the best few.
_LEVEL1_WORDS = re.compile(r"\b(department|departamento|province|provincia|region|state|"
                           r"estado|county)\b", re.I)


@functools.lru_cache(maxsize=32)
def _admin_table(cc: str, level: int):
    """(GeoDataFrame, name column) of one country's admin level, from aggeodata's
    own boundary cache (geoBoundaries, simplified)."""
    from aggeodata.ingestion.boundaries import _detect_name_column, _fetch_geojson_cached
    gdf = _fetch_geojson_cached(cc, level)
    return gdf, _detect_name_column(gdf)


def _find_place_sync(name: str, country_code: str, message: str = "") -> dict:
    """The best matches for a place name in one country, at most five.

    Exact name matches rank first. When a name exists at both levels ("Yoro" is
    a department and a municipality) the district wins unless the request says
    department, province, region or state, so the same words always resolve to
    the same area."""
    import difflib
    cc = (country_code or "").strip().upper()
    want = _slug(name)
    prefer = 1 if _LEVEL1_WORDS.search(message or "") else 2
    cands, tables = [], {}
    for level in (1, 2):
        try:
            gdf, col = _admin_table(cc, level)
        except Exception as exc:  # noqa: BLE001
            log.info("no admin level %d for %s: %s", level, cc, exc)
            continue
        tables[level] = (gdf, col)
        for i, n in enumerate(gdf[col].astype(str)):
            s = _slug(n)
            score = 1.0 if s == want else difflib.SequenceMatcher(None, want, s).ratio()
            if score >= 0.6:
                cands.append((score, 0 if level == prefer else 1, level, i, n))
    if not tables:
        return {"error": f"no administrative boundaries found for country code "
                         f"{cc!r}. Check the ISO3 code (e.g. COL, RWA, ZMB)."}
    cands.sort(key=lambda c: (-c[0], c[1], c[4]))
    matches = []
    for score, _, level, i, n in cands[:5]:
        m = {"country_code": cc, "name": n, "adm_level": level,
             "score": round(score, 2), "adm1": n if level == 1 else None,
             "adm2": n if level == 2 else None}
        if level == 2 and 1 in tables:
            try:
                g1, c1 = tables[1]
                pt = tables[2][0].geometry.iloc[i].representative_point()
                hit = g1[g1.geometry.contains(pt)]
                if len(hit):
                    m["adm1"] = str(hit.iloc[0][c1])
            except Exception:  # noqa: BLE001
                pass
        matches.append(m)
    if not matches:
        return {"error": f"no place close to {name!r} in {cc}."}
    return {"matches": matches, "best": matches[0]}


async def _local_find_place(sess: Session, name: str = "", country_code: str = "", **_):
    key = (_slug(name), (country_code or "").upper())
    if key in sess.places:
        res = dict(sess.places[key])
        res["note"] = ("ALREADY FOUND earlier in this conversation. Use 'best'; do not "
                       "look this place up again.")
        return json.dumps(res)
    res = await asyncio.to_thread(_find_place_sync, name, country_code, sess.last_message)
    if "best" in res:
        sess.places[key] = res
        res["note"] = ("Use 'best' (its name and adm_level) for acquire_region; do not "
                       "call find_place again for this place."
                       + _prebaked_hint({"country_code": country_code}))
    return json.dumps(res)


# --- weather window and default sowing date ---------------------------------
# Both are fixed in code so the same request gives the same result; the model
# only passes on what the user said.
SPINUP_DAYS = 30        # weather before the first sowing (the prompt's "one month before")
AFTER_DAYS = 30         # weather after the last expected maturity
ONSET_SEARCH_DAYS = 183 # how far after the anchor date the onset is looked for
# Onset of rains, Sivakumar (1988, Agric. For. Meteorol. 42: 295-305): the
# first day from the anchor date with >= 20 mm over 3 consecutive days and no
# dry spell longer than 7 days in the next 30 days. Sivakumar's anchor is 1 May
# for the Sudano-Sahelian zone of West Africa; elsewhere the anchor is this
# app's own fixed choice: 1 January of the season year, or 1 July of the first
# year for a season that spans two years ("2018/19"). A dry day is < 1 mm (this
# app's choice; the definition above does not fix it).
ONSET_MM, ONSET_DAYS, DRY_SPELL_MAX, DRY_LOOKAHEAD, DRY_DAY_MM = 20.0, 3, 7, 30, 1.0


def _season_years(season: str, planting_date: str = "") -> tuple[int | None, bool]:
    """(first year, spans two years) from "2019", "2018/19", "2018-2019"."""
    ys = re.findall(r"(?:19|20)\d{2}", str(season or ""))
    if ys:
        y0 = int(ys[0])
        two = bool(re.search(r"(?:19|20)\d{2}\s*[/-]\s*\d{2,4}", str(season)))
        return y0, two
    if planting_date:
        try:
            return pd.Timestamp(planting_date).year, False
        except Exception:  # noqa: BLE001
            pass
    return None, False


def _west_africa_sahel(lon: float, lat: float) -> bool:
    """Sivakumar's zone: southern Sahel and Sudanian West Africa."""
    return -18.0 <= lon <= 16.0 and 8.0 <= lat <= 18.0


def _onset_anchor(y0: int, two_years: bool, lon: float, lat: float) -> tuple[dt.date, str]:
    if two_years:
        return dt.date(y0, 7, 1), f"from 1 July {y0} (a season spanning {y0}/{(y0 + 1) % 100:02d})"
    if _west_africa_sahel(lon, lat):
        return dt.date(y0, 5, 1), f"from 1 May {y0} (Sivakumar's anchor for West Africa)"
    return dt.date(y0, 1, 1), f"from 1 January {y0}"


def _onset_from_pr(pr: "pd.Series", anchor: dt.date) -> dt.date | None:
    """The first onset day on or after `anchor` in an area-mean daily rainfall series."""
    pr = pr.sort_index()
    start = pd.Timestamp(anchor)
    stop = start + pd.Timedelta(ONSET_SEARCH_DAYS, unit="D")
    for d in pr.loc[start:stop].index:
        win = pr.loc[d:d + pd.Timedelta(ONSET_DAYS - 1, unit="D")]
        if len(win) < ONSET_DAYS or win.sum() < ONSET_MM:
            continue
        nxt = pr.loc[d + pd.Timedelta(1, unit="D"):d + pd.Timedelta(DRY_LOOKAHEAD, unit="D")]
        if len(nxt) < DRY_LOOKAHEAD:
            return None                    # not enough weather to apply the test
        run = longest = 0
        for v in nxt.values:
            run = run + 1 if (not np.isfinite(v) or v < DRY_DAY_MM) else 0
            longest = max(longest, run)
        if longest <= DRY_SPELL_MAX:
            return d.date()
    return None


def _area_rainfall(cube_path: str) -> "pd.Series":
    import xarray as xr
    opener = xr.open_zarr if str(cube_path).rstrip("/\\").endswith(".zarr") else xr.open_dataset
    with opener(cube_path) as ds:
        pr = ds["pr"]
        sp = [d for d in pr.dims if d != "time"]
        s = pr.mean(sp, skipna=True).to_series()
    s.index = pd.to_datetime(s.index)
    return s


def _onset_rule_text(anchor_txt: str) -> str:
    return (f"onset of rains in CHIRPS rainfall averaged over the area: the first day "
            f"{anchor_txt} with at least {ONSET_MM:g} mm over {ONSET_DAYS} days and no dry "
            f"spell longer than {DRY_SPELL_MAX} days in the next {DRY_LOOKAHEAD} "
            f"(Sivakumar 1988)")


def _prebuilt_for(cc: str, name: str, level, year: int | None) -> dict | None:
    """The shipped cubes when they are this place and hold this season's year."""
    tag = _slug(name)
    place = PREBAKED_PLACES.get(tag)
    cubes = prebaked_regions().get(tag)
    if not place or not cubes or (cc or "").upper() != place[0]:
        return None
    try:
        if level not in (None, "") and int(level) != place[1]:
            return None
        first, last, ext = _cube_span(str(cubes["weather"]))
    except Exception:  # noqa: BLE001
        return None
    if year is not None and not (int(first[:4]) <= int(year) <= int(last[:4])):
        return None
    return {"tag": tag, "weather": str(cubes["weather"]), "soil": str(cubes["soil"]),
            "first": first, "last": last, "extent": ext, "level": place[1]}


async def _local_acquire_region(sess: Session, country_code: str = "", feature_name: str = "",
                                adm_level=None, season: str = "", crop: str = "maize",
                                planting_date: str = "", n_planting_windows=1,
                                planting_window_days=14, **_):
    """One call: place and season in; the area, weather window, sowing dates and
    two simulation-ready cubes out, all fixed here rather than by the model."""
    def _go():
        # 1. The place: the best match for the name, as find_place resolves it.
        key = (_slug(feature_name), (country_code or "").upper())
        found = sess.places.get(key) or _find_place_sync(feature_name, country_code,
                                                          sess.last_message)
        if "best" not in found:
            return {"error": found.get("error", "place not found")}
        sess.places[key] = found
        best = found["best"]
        if adm_level not in (None, "") and str(adm_level).isdigit():
            same = [m for m in found["matches"]
                    if _slug(m["name"]) == _slug(best["name"]) and m["adm_level"] == int(adm_level)]
            best = same[0] if same else best
        cc, name, level = best["country_code"], best["name"], int(best["adm_level"])

        # 2. The season and the sowing windows.
        y0, two = _season_years(season, planting_date)
        if y0 is None:
            return {"error": "give the season (e.g. '2019' or '2018/19')."}
        crop_ = str(crop or "maize").lower()
        days = _maturity_days(crop_)
        n = max(1, int(float(n_planting_windows or 1)))
        gap = max(1, int(float(planting_window_days or 14)))
        gdf, _ = _admin_table(cc, level)
        col = _admin_table(cc, level)[1]
        geom = gdf[gdf[col].astype(str) == name].geometry
        c = geom.iloc[0].representative_point() if len(geom) else None
        lon, lat = (c.x, c.y) if c is not None else (0.0, 0.0)
        user_date = None
        if planting_date:
            try:
                user_date = pd.Timestamp(planting_date).date()
            except Exception:  # noqa: BLE001
                return {"error": f"planting_date {planting_date!r} is not a date."}
        anchor, anchor_txt = _onset_anchor(y0, two, lon, lat)

        # 3. The weather window, from the sowing dates (or the latest an onset
        #    could fall) plus days to maturity and fixed margins.
        if user_date:
            first_sow, last_possible = user_date, user_date
        else:
            first_sow = anchor
            last_possible = anchor + dt.timedelta(days=ONSET_SEARCH_DAYS)
        start = first_sow - dt.timedelta(days=SPINUP_DAYS)
        end = last_possible + dt.timedelta(days=gap * (n - 1) + days + AFTER_DAYS)

        # 4. The cubes: the shipped ones when they are this place and season.
        pre = _prebuilt_for(cc, name, level, y0)
        if pre:
            climate, soil, source = pre["weather"], pre["soil"], PREBAKED_LABEL
            w_first, w_last = pre["first"], pre["last"]
        else:
            res = _acquire_sync(cc, name, level, start.isoformat(), end.isoformat())
            climate, soil, source = res["climate_cube"], res["soil_cube"], "fresh download"
            w_first, w_last, _ = _cube_span(str(climate))

        # 5. The sowing date: the user's, or the onset rule on this cube's rain.
        if user_date:
            sow, rule = user_date, "given by the user"
        else:
            onset = _onset_from_pr(_area_rainfall(str(climate)), anchor)
            if onset is None:
                return {"error": "no onset of rains found by the default rule ("
                                 + _onset_rule_text(anchor_txt) + "). Ask the user for a "
                                 "sowing date and call acquire_region with planting_date."}
            sow, rule = onset, "default sowing date (rule: " + _onset_rule_text(anchor_txt) + ")"
        sows = [sow + dt.timedelta(days=gap * k) for k in range(n)]
        sess.region = {"country_code": cc, "name": name, "adm_level": level,
                       "adm1": best.get("adm1"), "climate_cube": str(climate),
                       "soil_cube": str(soil), "source": source}
        sess.sowing = {"planting_date": sow.isoformat(), "n_planting_windows": n,
                       "planting_window_days": gap, "crop": crop_, "rule": rule,
                       "default": user_date is None}
        out = {"climate_cube": str(climate), "soil_cube": str(soil),
               "region": name, "adm_level": level, "adm1": best.get("adm1"),
               "country_code": cc, "data": source,
               "weather": f"{w_first} to {w_last}",
               "sowing_dates": [s.isoformat() for s in sows], "sowing_rule": rule,
               "maturity_days": days,
               "note": ("Use exactly these in generate_config: planting_date="
                        f"{sow.isoformat()}, n_planting_windows={n}, "
                        f"planting_window_days={gap}, feature={name!r}, adm_level={level}; "
                        "the page enforces them. In the plan write 'Data: "
                        + (f"{PREBAKED_LABEL} (instant)" if source == PREBAKED_LABEL
                           else "fresh download (a few minutes)")
                        + "' and 'Sowing: " + sow.isoformat()
                        + (" — " + rule if user_date is None else "") + "'.")}
        return out
    try:
        res = await asyncio.to_thread(_go)
    except Exception as exc:  # noqa: BLE001
        log.warning("acquire failed: %s", exc)
        return (f"ERROR acquisition failed: {type(exc).__name__}: {exc}. "
                "Do not simulate — there are no cubes to simulate on.")
    if "error" in res:
        return "ERROR " + res["error"]
    log.info("acquire %s %s (adm%s): %s, sowing %s (%s)", res["country_code"], res["region"],
             res["adm_level"], res["data"], res["sowing_dates"][0],
             "default" if sess.sowing.get("default") else "user")
    return json.dumps(res)


def _enforce_config_args(sess: Session, args: dict) -> dict | str:
    """generate_config always uses the area, cubes and sowing that acquire_region
    fixed; the model cannot swap in a bounding box, a placeholder cube or its own
    sowing date. Without acquire_region first there is nothing to enforce."""
    if not getattr(sess, "region", None):
        return ("ERROR call data" + SEP + "acquire_region first (instant for pre-built "
                "places): it fixes the area, the weather and the sowing dates.")
    out = dict(args)
    r, s = sess.region, sess.sowing
    out.update({"weather_path": r["climate_cube"], "soil_path": r["soil_cube"],
                "feature": r["name"], "adm_level": r["adm_level"],
                "country_code": r["country_code"],
                "planting_date": s["planting_date"],
                "n_planting_windows": s["n_planting_windows"],
                "planting_window_days": s["planting_window_days"]})
    return out


def _run_metadata(cfg, summary: dict) -> dict:
    """What a result file needs to say about itself to be compared later."""
    mgmt, crop, gen, sp = cfg.MANAGEMENT, cfg.CROP, cfg.GENERAL_INFO, cfg.SPATIAL_INFO
    try:
        n_total = float(mgmt.total_n_kg_ha)
    except Exception:  # noqa: BLE001
        n_total = float(sum(e.n_kg_ha for e in (mgmt.fertilizer_schedule or [])))
    return {
        "crop": str(crop.name), "cultivar": str(crop.cultivar),
        "n_kg_ha": n_total,
        "planting_date": str(mgmt.planting_date),
        "n_windows": int(mgmt.n_planting_windows),
        "window_days": int(mgmt.planting_window_days),
        "region": str(sp.feature or "") or _region_from_path(str(sp.weather_path)),
        "country": str(gen.country),
        "seasons": [int(y) for y in (summary.get("years") or [])],
        "mean_kg_ha": summary.get("mean_yield"),
        "pixels_ok": summary.get("pixels_simulated", summary.get("pixels_ok")),
        "adm_level": int(getattr(sp, "adm_level", 0) or 0) or None,
    }


def _region_from_path(weather_path: str) -> str:
    """`cubes/nandom_cube_climate_2020.zarr` -> `Nandom`; a guess, but a good one."""
    name = pathlib.Path(weather_path).name
    tag = name.split("_cube_")[0] if "_cube_" in name else ""
    return tag.replace("_", " ").title()


def _tag_result(path: pathlib.Path, meta: dict) -> None:
    """Write the run's metadata into the NetCDF so the file explains itself."""
    import xarray as xr
    ds = xr.load_dataset(path)
    ds.attrs.update({k: (json.dumps(v) if isinstance(v, (list, dict)) else
                         ("" if v is None else v))
                     for k, v in meta.items()})
    ds.to_netcdf(path)


def _maturity_days(crop: str) -> int:
    """Days from sowing to maturity, from ag-cube-cm's own table.

    ag-cube-cm uses the same table to decide whether a season would need
    weather past the cube, so the check below agrees with the model."""
    from ag_cube_cm.models.dssat.base import DSSATModel
    table = getattr(DSSATModel, "_MATURITY_DAYS", {}) or {}
    return int(table.get(str(crop or "").lower(), 120))


def _season_fit(config_path: str) -> dict | None:
    """Do all sowing dates of a config reach maturity inside its weather cube?

    A crop sown too close to the end of the cube runs out of weather before
    maturity and DSSAT harvests it early: a low yield that looks like an
    answer. The first season is the one that matters; a later season that
    would cross the end of a multi-year cube is already dropped by ag-cube-cm.
    Returns None when the config or cube cannot be read (no verdict)."""
    try:
        from ag_cube_cm.config.loader import load_config
        cfg = load_config(str(config_path))
        mgmt = cfg.MANAGEMENT
        crop = str(cfg.CROP.name)
        first_sow = pd.Timestamp(str(mgmt.planting_date))
        n = max(1, int(mgmt.n_planting_windows or 1))
        gap = int(mgmt.planting_window_days or 7)
        weather = str(cfg.SPATIAL_INFO.weather_path)
        cube_first, cube_last, _ = _cube_span(weather)
    except Exception as exc:  # noqa: BLE001
        log.warning("season check skipped for %s: %s", config_path, exc)
        return None
    days = _maturity_days(crop)
    sows = [first_sow + pd.Timedelta(int(gap * k), unit='D') for k in range(n)]
    c0, c1 = pd.Timestamp(cube_first), pd.Timestamp(cube_last)
    late = [s for s in sows if s + pd.Timedelta(int(days), unit='D') > c1]
    # DSSAT starts SIM_LEAD_DAYS before sowing and needs weather from then.
    early = [s for s in sows if s - pd.Timedelta(SIM_LEAD_DAYS, unit='D') < c0]
    latest_first = c1 - pd.Timedelta(int(days + gap * (n - 1)), unit='D')
    fits_shifted = latest_first >= c0
    return {
        "ok": not late and not early, "crop": crop, "maturity_days": days,
        "sowing_dates": [s.strftime("%Y-%m-%d") for s in sows],
        "cut_off": [s.strftime("%Y-%m-%d") for s in late],
        "before_cube": [s.strftime("%Y-%m-%d") for s in early],
        "cube_first": cube_first, "cube_last": cube_last,
        "latest_first_sowing": latest_first.strftime("%Y-%m-%d") if fits_shifted else None,
        "n_windows": n, "window_days": gap,
        "weather_must_end": (sows[-1] + pd.Timedelta(int(days), unit='D')).strftime("%Y-%m-%d"),
    }


def _season_fit_text(fit: dict) -> str:
    """The verdict in words the agent can put straight into its plan."""
    if fit["ok"]:
        return (f"SEASON CHECK: OK. Every sowing date ({', '.join(fit['sowing_dates'])}) "
                f"plus {fit['maturity_days']} days to maturity for {fit['crop']} ends by "
                f"{fit['cube_last']}, inside the weather cube ({fit['cube_first']} to "
                f"{fit['cube_last']}).")
    bits = []
    if fit["cut_off"]:
        bits.append(f"sowing on {', '.join(fit['cut_off'])} needs weather until "
                    f"{fit['weather_must_end']} ({fit['maturity_days']} days to maturity "
                    f"for {fit['crop']}), but the cube ends on {fit['cube_last']}, so the "
                    f"crop would be cut off before maturity")
    if fit["before_cube"]:
        bits.append(f"sowing on {', '.join(fit['before_cube'])} is less than "
                    f"{SIM_LEAD_DAYS} days after the cube starts ({fit['cube_first']}); "
                    f"the model starts {SIM_LEAD_DAYS} days before sowing")
    if fit["latest_first_sowing"]:
        k, g = fit["n_windows"], fit["window_days"]
        start = pd.Timestamp(fit["latest_first_sowing"])
        dates = ", ".join((start + pd.Timedelta(int(g * i), unit='D')).strftime("%Y-%m-%d")
                          for i in range(k))
        fix = (f"Dates that fit: first sowing no later than {fit['latest_first_sowing']} "
               f"(for example {dates}).")
    else:
        fix = (f"{fit['n_windows']} dates {fit['window_days']} days apart cannot fit in "
               f"this cube; use fewer dates or a shorter gap.")
    return ("SEASON CHECK: DOES NOT FIT — " + "; ".join(bits) + ". " + fix
            + f" Or a fresh download whose weather runs to at least "
            f"{fit['weather_must_end']}. Say this in the plan, propose the dates that "
            f"fit (or the download), and ask the user before running.")


def _config_fit_note(args: dict, payload) -> str:
    """The season check, appended to generate_config's answer, so the plan
    written next can say whether the dates fit and propose ones that do."""
    path = args.get("save_to")
    if not path:
        m = re.search(r"[^\s\"']+\.ya?ml", str(payload))
        path = m.group(0) if m else None
    if not path or not pathlib.Path(str(path)).is_file():
        return ""
    fit = _season_fit(str(path))
    return f"\n\n{_season_fit_text(fit)}" if fit else ""


async def _local_run_simulation(sess: Session, config_path: str, max_pixels=None, **_):
    def _go():
        from ag_cube_cm.config.loader import load_config
        from ag_cube_cm.spatial.spatial_cm import SpatialCM
        if not pathlib.Path(str(config_path)).is_file():
            # The model tends to invent a config name instead of using the
            # save_path generate_config returned; naming what exists lets it
            # correct itself in one step instead of re-running the pipeline.
            have = sorted(p.name for p in sess.dir.glob("*.yaml"))
            return {"error": f"config_path does not exist: {config_path}. "
                             + (f"Configs in this session: {', '.join(have)}. Pass "
                                "the save_path that generate_config returned."
                                if have else "Call generate_config with save_to first.")}
        cfg = load_config(str(config_path))
        # Refuse up front what would otherwise be a convincing wrong answer: a
        # weather cube without radiation runs to completion with every pixel
        # at zero, and a missing file waits on the tool budget then times out.
        sp = cfg.SPATIAL_INFO
        for label, p in (("weather_path", sp.weather_path), ("soil_path", sp.soil_path)):
            if not p or not pathlib.Path(str(p)).exists():
                return {"error": f"{label} does not exist: {p}. Use the exact cube "
                                 "paths given to you, or acquire the region first."}
        missing = _missing_climate_vars(sp.weather_path)
        if missing:
            return {"error": f"the weather cube {sp.weather_path} has no "
                             f"{', '.join(missing)}; DSSAT would report zero yield "
                             "everywhere. Acquire the region again."}
        # Enforce the job budget here rather than trust the prompt: a department
        # at three seasons and seven windows is 14,000 jobs and forty minutes on
        # this machine, and the model will happily omit max_pixels.
        windows = max(1, int(cfg.MANAGEMENT.n_planting_windows or 1))
        years, cells = _cube_extent(sp.weather_path)
        # The seasons that will run, not every calendar year the weather
        # touches (a Dec–Jan download touches three years for one season).
        try:
            first, last, _ = _cube_span(str(sp.weather_path))
            years = max(1, len(_fitting_seasons(
                cfg.MANAGEMENT.planting_date, windows,
                cfg.MANAGEMENT.planting_window_days, str(cfg.CROP.name), first, last)))
        except Exception as exc:  # noqa: BLE001
            log.warning("season count for the job budget: %s", exc)
        cap = max(1, MAX_SIMULATIONS // (windows * years))
        pixels = int(max_pixels) if max_pixels else None
        note = None
        if cells and cap >= cells and (pixels is None or pixels >= cells):
            pixels = None                     # the whole grid fits; no sampling
        elif pixels is None or pixels > cap:
            pixels = cap
            note = (f"max_pixels set to {cap}: {windows} window(s) x {years} "
                    f"season(s) at {MAX_SIMULATIONS} jobs. The pixels are spread "
                    "over the region as a sample; say so when reporting.")
            log.info("run_simulation: capping pixels at %d (%d windows x %d years)",
                     cap, windows, years)
        # SpatialCM already runs one job per season; left unset, DSSAT counts
        # NYERS from the weather it is handed and averages extra seasons into
        # what is reported as one year's yield.
        if getattr(cfg.GENERAL_INFO, "number_years", None) is None:
            cfg.GENERAL_INFO.number_years = 1
        with SpatialCM(cfg) as cm:
            summary = cm.run(max_pixels=pixels).summary()
        if note:
            summary["note"] = note
        out = summary.get("output_path")
        coverage = None
        if out and pathlib.Path(out).is_file():
            # Record whether the weather covered each season, before any figure
            # is drawn from the file: a season is left out for missing weather
            # only, and all-zero yields with complete weather are a failure.
            try:
                coverage = _season_coverage(
                    str(sp.weather_path), cfg.MANAGEMENT.planting_date, windows,
                    int(cfg.MANAGEMENT.planting_window_days), str(cfg.CROP.name),
                    summary.get("years") or [cfg.MANAGEMENT.planting_date.year])
                _tag_result(pathlib.Path(out), {"weather_coverage": coverage})
            except Exception as exc:  # noqa: BLE001
                log.warning("weather coverage for %s: %s", out, exc)
            # ag-cube-cm's own summary averages every season it ran, zeros
            # included; report the numbers every other part of the page uses.
            try:
                st = _window_stats_read(str(out))
                ws = st["windows"]
                if ws:
                    mean = int(f"{float(np.mean([w['mean_kg_ha'] for w in ws])):.0f}")
                    for k in ("mean_yield", "mean_hwam_kg_ha", "mean_hwam"):
                        summary[k] = mean
                    summary["pixels_simulated"] = max(w["pixels"] for w in ws)
                    summary["pixels_no_yield"] = max(w["pixels_no_yield"] for w in ws)
                summary["years"] = st.get("seasons") or summary.get("years")
                summary["n_years"] = st.get("n_seasons", summary.get("n_years"))
                for k in ("seasons_left_out", "crop_failure_seasons", "note"):
                    if st.get(k):
                        summary[k] = st[k]
                # Job counts (pixel x window x season), phantom seasons included;
                # a pixel count is what the page and the CSV show.
                for k in ("pixels_ok", "pixels_failed", "pixels_skipped"):
                    summary.pop(k, None)
            except Exception as exc:  # noqa: BLE001
                log.warning("could not recompute the summary of %s: %s", out, exc)
            meta = _run_metadata(cfg, summary)
            try:
                _tag_result(pathlib.Path(out), meta)
            except Exception as exc:  # noqa: BLE001
                log.warning("could not tag %s: %s", out, exc)
            rec = {"file": str(out), "created": time.time(), **meta}
            rec["label"] = _run_label(rec)
            sess.add_run(rec)
            summary["run"] = rec["label"]
        return summary
    # Never run a sowing date the cube cannot carry to maturity.
    if pathlib.Path(str(config_path)).is_file():
        fit = await asyncio.to_thread(_season_fit, str(config_path))
        if fit is not None and not fit["ok"]:
            log.info("season check refused %s: %s", pathlib.Path(str(config_path)).name,
                     fit["cut_off"] or fit["before_cube"])
            return json.dumps({"error": "NOT RUN: " + _season_fit_text(fit)})
    return json.dumps(await asyncio.to_thread(_go), default=str)


def _window_stats(path: str) -> dict:
    """Per-window statistics for one result file, read once per file version.

    A sowing-date pick reads them twice (tiles and the sowing series), and
    each read reopened the NetCDF; the file only changes when a run rewrites it."""
    import copy
    try:
        mtime = pathlib.Path(path).stat().st_mtime
    except OSError:
        return _window_stats_read(path)
    return copy.deepcopy(_window_stats_cached(str(path), mtime))


@functools.lru_cache(maxsize=64)
def _window_stats_cached(path: str, mtime: float) -> dict:
    return _window_stats_read(path)


# A season's weather counts as complete when no more than this many days are
# missing over sowing to maturity: ag-cube-cm fills short gaps by interpolation
# and warns above 5 missing days ("may yield unrealistic precipitation").
MAX_MISSING_WEATHER_DAYS = 5
_WEATHER_VARS = ("pr", "tasmax", "tasmin", "rsds")


def _season_coverage(weather_path: str, base_pdate, n_windows: int, step_days: int,
                     crop: str, years) -> dict:
    """Per season: does the weather cover sowing to maturity, day by day?

    A day is missing when the time axis lacks it, or when any of rainfall, the
    two temperatures or radiation is missing over the whole area. Written into
    each result, so the page leaves a season out for missing weather, never for
    its yields."""
    import xarray as xr
    from ag_cube_cm.spatial.spatial_cm import _shift_year
    opener = xr.open_zarr if str(weather_path).rstrip("/\\").endswith(".zarr") else xr.open_dataset
    days = _maturity_days(crop)
    with opener(weather_path) as ds:
        tdim = "time" if "time" in ds.dims else "date"
        sp = [d for d in ds[_WEATHER_VARS[0]].dims if d != tdim]
        ok = None
        for v in _WEATHER_VARS:
            if v not in ds:
                continue
            m = np.isfinite(ds[v].mean(sp, skipna=True).values)
            ok = m if ok is None else (ok & m)
        t = pd.to_datetime(ds[tdim].values)
    have = set(t[ok].normalize()) if ok is not None else set()
    out = {}
    for yr in years:
        first = pd.Timestamp(_shift_year(base_pdate, int(yr)))
        last = pd.Timestamp(_shift_year(base_pdate + dt.timedelta(days=step_days * (n_windows - 1)),
                                        int(yr))) + pd.Timedelta(days, unit="D")
        need = pd.date_range(first, last, freq="D")
        missing = int(sum(1 for d in need if d not in have))
        out[str(int(yr))] = {"from": first.strftime("%Y-%m-%d"), "to": last.strftime("%Y-%m-%d"),
                             "missing_days": missing,
                             "complete": missing <= MAX_MISSING_WEATHER_DAYS}
    return out


def _real_seasons(da, attrs=None) -> tuple[list[int], list, list]:
    """Positions along `year` of the seasons to report, the years left out, and
    the reported years in which every pixel yielded 0 (crop failures).

    A season is left out only when its weather is missing or incomplete: from
    the coverage each run records (`weather_coverage`), or, for a result without
    it, when the season has no value at all. A season with complete weather and
    every pixel at 0 is a crop failure: it is reported, never left out."""
    if "year" not in da.dims:
        v = da.values.astype(float)
        fin = v[np.isfinite(v)]
        return [], [], ([None] if fin.size and not (fin != 0).any() else [])
    years = (list(np.atleast_1d(da["year"].values)) if "year" in da.coords
             else list(range(da.sizes["year"])))
    cov = (attrs or {}).get("weather_coverage")
    if isinstance(cov, str):
        try:
            cov = json.loads(cov)
        except ValueError:
            cov = None
    keep, dropped, failed = [], [], []
    for i in range(da.sizes["year"]):
        try:
            y = int(years[i])
        except (TypeError, ValueError):
            y = years[i]
        v = da.isel(year=i).values.astype(float)
        fin = v[np.isfinite(v)]
        if isinstance(cov, dict) and str(y) in cov:
            covered = bool(cov[str(y)].get("complete"))
        else:
            covered = bool(fin.size)          # no value at all: no weather
        if not covered:
            dropped.append(y)
            continue
        keep.append(i)
        if fin.size and not (fin != 0).any():
            failed.append(y)
    return keep, dropped, failed


def _window_stats_read(path: str) -> dict:
    """Per-window statistics for one result file.

    The same numbers feed the tiles, the sowing-date chart, the chat (via
    read_yield) and the CSV check, so they agree: only real seasons count
    (`_real_seasons`), and a pixel is a place on the map, counted once."""
    import xarray as xr
    with xr.open_dataset(path) as ds:
        da = ds["HWAM"]
        dates = ([str(d)[:10] for d in np.atleast_1d(ds.planting_date.values)]
                 if "planting_date" in ds.coords else [])
        keep, dropped, failed = _real_seasons(da, ds.attrs)
        if "year" in da.dims and keep:
            da = da.isel(year=keep)
        n_yr = int(da.sizes.get("year", 1))
        kept_years = ([int(y) for y in np.atleast_1d(da["year"].values)]
                      if "year" in da.coords else [])
        windows = []
        n = int(da.sizes.get("planting_window", 1))
        for i in range(n):
            w = da.isel(planting_window=i) if "planting_window" in da.dims else da
            if "year" not in w.dims:
                w = w.expand_dims("year")
            v = w.transpose("year", ...).values.astype(float)   # year, y, x
            fin = v[np.isfinite(v)]
            if fin.size == 0:
                continue
            # A pixel is simulated when it has a value in any reported season;
            # it gave no yield when every such value is 0.
            has = np.isfinite(v).any(axis=0)
            zero = has & np.all(np.where(np.isfinite(v), v == 0, True), axis=0)
            # Area mean per season: the spread a sowing-date decision rests
            # on. The pixel min/max is a different question — one cold hilltop
            # failing in one year pulls it to zero for every window.
            per_season = ([float(np.nanmean(s)) for s in v if np.isfinite(s).any()]
                          if n_yr > 1 else [])
            windows.append({
                "window": i + 1,
                "sown": dates[i] if i < len(dates) else None,
                # Full precision here; every display rounds once, to whole kg.
                # Rounding to 0.1 first turned 2,663.49 into 2,663.5 and then
                # into 2,664 on the tile and in read_yield (CSV: 2,663).
                "mean_kg_ha": float(fin.mean()),
                "min_kg_ha": float(fin.min()),
                "max_kg_ha": float(fin.max()),
                "p10_kg_ha": float(np.percentile(fin, 10)),
                "p90_kg_ha": float(np.percentile(fin, 90)),
                "season_means_kg_ha": [float(s) for s in per_season],
                "failed_pixel_seasons": int((fin == 0).sum()),
                "pixels": int(has.sum()),
                "pixels_no_yield": int(zero.sum()),
                # Complete weather and nothing harvested anywhere: a result.
                "crop_failure": bool(has.sum()) and int(zero.sum()) == int(has.sum()),
            })
        best = max(windows, key=lambda w: w["mean_kg_ha"]) if windows else None
        attrs = {k: v for k, v in ds.attrs.items()
                 if k in ("crop", "n_kg_ha", "region", "country", "seasons")}
    if isinstance(attrs.get("seasons"), str):
        try:
            attrs["seasons"] = json.loads(attrs["seasons"])
        except ValueError:
            attrs.pop("seasons")
    if kept_years:
        attrs["seasons"] = kept_years
    out = {"file": pathlib.Path(path).name, "n_seasons": n_yr, **attrs,
           "windows": windows, "best_window": best}
    notes = []
    if dropped:
        out["seasons_left_out"] = dropped
        notes.append(f"Season(s) {', '.join(map(str, dropped))} left out of every figure: "
                     "the weather data does not cover sowing to maturity.")
    failed = [y for y in failed if y is not None] or ([None] if failed else [])
    if failed:
        out["crop_failure_seasons"] = [y for y in failed if y is not None]
        notes.append("CROP FAILURE: the weather was complete and every pixel yielded 0"
                     + (f" in {', '.join(map(str, out['crop_failure_seasons']))}"
                        if out["crop_failure_seasons"] else "")
                     + ". Report it as a crop failure, not as missing data.")
    if notes:
        out["note"] = " ".join(notes)
    return out


async def _local_read_yield(sess: Session, path: str | None = None, **_):
    """Per-window statistics from one or all result files of this session.

    Without this the agent can start a simulation but never look at what came
    out: it reported a mean across four planting windows and then had to say it
    could not tell which window was best. Comparing nitrogen rates is the same
    problem one level up — each rate is its own file, so the comparison needs to
    read several.
    """
    def _go():
        paths = [path] if path else [r["file"] for r in sess.runs()]
        out = []
        for f in paths[:8]:
            try:
                st = _window_stats(f)
                # Whole kg ha⁻¹, rounded as the tiles round them (half to
                # even): handed 2,981.5 the model wrote 2,981 while the tile
                # showed 2,982.
                for w in st.get("windows", []):
                    for k in ("mean_kg_ha", "min_kg_ha", "max_kg_ha", "p10_kg_ha", "p90_kg_ha"):
                        if w.get(k) is not None:
                            w[k] = int(f"{w[k]:.0f}")
                    w["season_means_kg_ha"] = [int(f"{v:.0f}") for v in w.get("season_means_kg_ha", [])]
                if st.get("best_window"):
                    st["best_window"] = next((w for w in st["windows"]
                                              if w["window"] == st["best_window"]["window"]),
                                             st["best_window"])
                out.append(st)
            except Exception as exc:  # noqa: BLE001
                out.append({"file": pathlib.Path(f).name,
                            "error": f"{type(exc).__name__}: {exc}"})
        return out
    return json.dumps(await asyncio.to_thread(_go), default=str)


LOCAL_TOOLS = {f"crop{SEP}run_simulation": _local_run_simulation,
               f"crop{SEP}read_yield": _local_read_yield,
               f"data{SEP}acquire_region": _local_acquire_region,
               f"data{SEP}find_place": _local_find_place}

# Argument names that carry an output location. Anything the model puts there is
# moved into the session folder, so a run cannot write over another visitor's
# files or outside the work tree.
OUTPUT_KEYS = ("output_folder", "output_path", "soil_folder", "working_path",
               "data_folder", "save_to")


async def _call_in_process(full_name: str, args: dict) -> str:
    fn = SERVER_FUNCS[full_name]
    res = await asyncio.to_thread(lambda: fn(**args))
    return res if isinstance(res, str) else json.dumps(res, default=str)


# --------------------------------------------------------------------------
# Model providers
#
# Gemini's free tier allows 20 requests per day per model, and one conversation
# spends five to ten of them — enough for a couple of runs, not for a demo other
# people open. So the agent falls back to a second provider when the first is out
# of quota or down.
#
# The two APIs disagree about the shape of both messages and tool declarations,
# so the conversation is kept in a neutral form and rendered per provider at call
# time. That is the only way a mid-conversation switch can work: history recorded
# in Gemini's objects cannot be replayed to an OpenAI-compatible endpoint.
# --------------------------------------------------------------------------
OPENROUTER_URL = os.environ.get(
    "CDH_OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")
# Chosen by measurement, not by name: of the free models that genuinely emit tool
# calls, this one answered a tool probe in 1.2 s. Latency multiplies by turn count,
# so the spread matters. Re-check with scripts/check_openrouter_models.py; free
# slugs are retired often.
OPENROUTER_MODEL = os.environ.get("CDH_OPENROUTER_MODEL",
                                  "nvidia/nemotron-3.5-super-vl:free")
# Gemini first, OpenRouter as the reserve. Both are free tiers; Gemini Flash-Lite
# allows on the order of a thousand requests a day, OpenRouter's free models
# fifty, so the order is the difference between a demo that lasts the day and
# one that stops before lunch. Set CDH_PROVIDERS to reorder or pin one.
# "gemini" expands to one entry per Gemini model ("gemini/<id>"), so a 429 on
# one model moves to the next model's quota before leaving Gemini.
PROVIDERS = [q for p in os.environ.get("CDH_PROVIDERS", "gemini,openrouter").split(",")
             if p.strip()
             for q in ([f"gemini/{m}" for m in GEMINI_MODELS] if p.strip() == "gemini"
                       else [p.strip()])]

# A daily quota does not come back in a minute, but `exhausted` lives inside one
# turn, so without this every later message re-tries the dead provider, waits for
# the same refusal, and fails over again. Remembered for a cooldown rather than
# permanently: the limit resets at midnight UTC, and a misread error should cost
# half an hour of fallback, not the whole session.
_QUOTA_COOLDOWN_S = 1800.0
_QUOTA_OUT: dict[str, float] = {}


def _mark_quota_out(provider: str) -> None:
    _QUOTA_OUT[provider] = time.monotonic()


def _in_cooldown(provider: str) -> bool:
    t = _QUOTA_OUT.get(provider)
    if t is None:
        return False
    if time.monotonic() - t < _QUOTA_COOLDOWN_S:
        return True
    del _QUOTA_OUT[provider]      # cooldown served; let it prove itself again
    return False


def _provider_ready(name: str) -> bool:
    if name.split("/")[0] == "gemini":
        return bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
    if name == "openrouter":
        return bool(os.environ.get("OPENROUTER_API_KEY"))
    return False


def _retry_seconds(msg: str) -> float | None:
    """The wait a 429 asks for, if it names one (`retry in 29.25s`, `retryDelay: 30s`)."""
    m = re.search(r"retry(?:Delay|\s+in)\W+([0-9.]+)\s*s", msg, re.IGNORECASE)
    try:
        return float(m.group(1)) if m else None
    except ValueError:
        return None


def _is_quota(msg: str) -> bool:
    """Out of allowance for the DAY — only another provider gets past this.

    Gemini words a per-minute limit the same way as the daily one ("exceeded
    your current quota ... RESOURCE_EXHAUSTED"); what tells them apart is the
    quota id (PerMinute vs PerDay) and the retry delay. A limit that clears in
    thirty seconds is a pause, not a reason to bench the provider for half an
    hour and tell the user their day is over.
    """
    low = msg.lower()
    if "perminute" in low or "per minute" in low:
        return False
    wait = _retry_seconds(msg)
    if wait is not None and wait < 120:
        return False
    return any(s in low for s in
               ("exceeded your current quota", "perday", "per day", "per-day",
                "resource_exhausted", "insufficient_quota",
                "free-models-per-day", "daily quota", "daily limit"))


def _is_transient(msg: str) -> bool:
    """A blip worth one more try before giving up on this provider."""
    if _is_quota(msg):
        return False
    low = msg.lower()
    return any(s in low for s in ("503", "unavailable", "500", "internal",
                                  "overloaded", "service unavailable",
                                  "timed out", "timeout", "deadline",
                                  "429", "rate limit", "rate-limit",
                                  "too many requests"))


# ---- neutral conversation -------------------------------------------------
# Each entry is one of:
#   {"role": "user"|"assistant", "text": str}
#   {"role": "assistant", "calls": [{"id","name","args"}]}
#   {"role": "tool", "id": str, "name": str, "result": str}

# Gemini 3 checks each function call's signature against the model that made
# it. After a mid-turn switch to another Gemini model (or from OpenRouter), the
# earlier calls carry no valid signature for the new model; this documented
# placeholder tells it to skip the check instead of refusing the history.
SKIP_SIG = b"skip_thought_signature_validator"


def _to_gemini(msgs, model=None):
    out = []
    for m in msgs:
        if m["role"] == "tool":
            out.append(gt.Content(role="user", parts=[gt.Part.from_function_response(
                name=m["name"], response={"result": m["result"]})]))
        elif m.get("calls"):
            # Gemini 3 signs each function call it emits and rejects the history
            # if the signature is not handed back with it (400 "missing
            # thought_signature"). Carried in the neutral record as `sig`.
            out.append(gt.Content(role="model", parts=[
                gt.Part(function_call=gt.FunctionCall(name=c["name"], args=c["args"]),
                        thought_signature=(c["sig"] if c.get("sig") and c.get("sig_model") == model
                                           else SKIP_SIG))
                for c in m["calls"]]))
        else:
            out.append(gt.Content(role="model" if m["role"] == "assistant" else "user",
                                  parts=[gt.Part(text=m.get("text") or "")]))
    return out


def _to_openai(msgs, system: str):
    out = [{"role": "system", "content": system}]
    for m in msgs:
        if m["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": m["id"], "content": m["result"]})
        elif m.get("calls"):
            out.append({"role": "assistant", "content": None, "tool_calls": [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["name"], "arguments": json.dumps(c["args"])}}
                for c in m["calls"]]})
        else:
            out.append({"role": m["role"], "content": m.get("text") or ""})
    return out


def _openai_tools(declared):
    """The same tools in OpenAI's shape."""
    return [{"type": "function",
             "function": {"name": d.name,
                          "description": d.description or "",
                          "parameters": (d.parameters.model_dump(exclude_none=True)
                                         if hasattr(d.parameters, "model_dump")
                                         else (d.parameters or
                                               {"type": "object", "properties": {}}))}}
            for d in declared]


def _whole_floats_to_int(value):
    """Turn 1.0 into 1 throughout a tool-argument structure.

    Gemini hands back arguments as a protobuf Struct, which has one number type
    and it is a double — so every integer the model writes arrives as a float.
    Anything used as an index cares: ag-cube-cm caps the grid with
    `pixels[:max_pixels]`, and max_pixels=1.0 raises "slice indices must be
    integers", several minutes into a run. Floats with a fractional part are
    left alone; booleans are excluded because True would become 1.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {k: _whole_floats_to_int(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_whole_floats_to_int(v) for v in value]
    return value


def _as_text(content) -> str:
    """Flatten an OpenAI-style `content` field to a string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, str):
                out.append(part)
            elif isinstance(part, dict):
                out.append(str(part.get("text") or ""))
        return "".join(out)
    return "" if content is None else str(content)


# Markup some models emit around a tool call when they fall back to writing one
# as prose. The call itself still arrives in tool_calls, so these tags are pure
# leakage, and a stray "</function>" in the transcript reads as a broken template.
_LEAKED = re.compile(
    r"<\s*/?\s*(?:function|function_call|tool_call|tool_code|invoke|antml:invoke)"
    r"(?:\s[^>]*)?>|<\s*function\s*=\s*[^>]*>", re.I)


def _strip_leaked_markup(text: str) -> str:
    cleaned = _LEAKED.sub("", text)
    if cleaned != text:
        log.info("stripped leaked tool-call markup from reply")
    return cleaned.strip()


# Absolute paths in the chat leak the server's layout and, in a plan table, an
# unbreakable 70-character path squeezes the label column to one letter per line.
# The model is told not to write them; this catches the ones it writes anyway.
SESSION_WORD = "session workspace"
# Any other absolute path: a drive path with either slash, or a Linux path
# under the usual roots (the Space runs on Linux). URLs are left alone.
_WIN_PATH = re.compile(
    r"(?<![\w/:.])(?:[A-Za-z]:[\\/]|/(?:tmp|home|app|data|mnt|usr|root|content|workspace|var|opt)/)"
    r"[^\s`|<>\"'*?()\[\]]*")


def _path_forms(p: pathlib.Path) -> set[str]:
    """A folder as the model might spell it: short or long name, either slash,
    or JSON-escaped backslashes copied out of a tool result."""
    forms = set()
    for s in {str(p), os.path.realpath(p)}:
        for t in (s, s.replace("\\", "/"), s.replace("\\", "\\\\")):
            forms.add(t)
    return {f for f in forms if len(f) > 3}


def _scrub_paths(text: str, sess: Session) -> str:
    roots = [(f, SESSION_WORD) for f in _path_forms(sess.dir)]
    roots += [(f, "shared cache") for f in _path_forms(SHARED)]
    roots += [(f, "cubes") for f in _path_forms(PREBAKED)]
    roots += [(f, "workspace") for f in _path_forms(WORK)]
    for form, word in sorted(roots, key=lambda r: -len(r[0])):
        pat = re.compile(re.escape(form) + r"((?:[\\/][^\s`|<>\"']*)?)", re.I)
        text = pat.sub(lambda m, w=word: w + re.sub(r"[\\/]+", "/", m.group(1)), text)
    # Anything else that still looks like a path: a file keeps only its name,
    # a folder (the model once wrote "C:/dssat_work") is the session workspace.
    return _WIN_PATH.sub(_short_path, text)


def _short_path(m) -> str:
    name = re.split(r"[\\/]+", m.group(0).rstrip("\\/"))[-1]
    return name if re.search(r"\.\w{1,5}$", name) else SESSION_WORD


def _unscrub_paths(text: str, sess: Session) -> str:
    """The reverse, for history sent back to the model, so a follow-up that
    names `session workspace/yield.nc` still reaches the real file."""
    return text.replace(SESSION_WORD, sess.dir.as_posix())


# A model call that never answers left a turn waiting forever (seen live: the
# reply after a finished run never came). Bound it like the OpenRouter call;
# a timeout is then retried, or handed to the next provider, as a 503 would be.
MODEL_TIMEOUT_S = int(os.environ.get("CDH_MODEL_TIMEOUT", "120"))


async def _ask_gemini(msgs, declared, system, api_key=None, model=None):
    """`api_key` is a visitor's own key (from their gr.State), else the Space's."""
    model = model or MODEL
    client = genai.Client(api_key=api_key or os.environ.get("GOOGLE_API_KEY")
                          or os.environ.get("GEMINI_API_KEY"),
                          http_options=gt.HttpOptions(timeout=MODEL_TIMEOUT_S * 1000))
    try:
        resp = await asyncio.wait_for(asyncio.to_thread(
            client.models.generate_content,
            model=model, contents=_to_gemini(msgs, model),
            config=gt.GenerateContentConfig(
                system_instruction=system, tools=[gt.Tool(function_declarations=declared)],
                temperature=0.2),
        ), timeout=MODEL_TIMEOUT_S + 15)
    except asyncio.TimeoutError:
        raise RuntimeError(f"503 unavailable: no answer within {MODEL_TIMEOUT_S}s") from None
    um = getattr(resp, "usage_metadata", None)
    _note_usage(getattr(um, "prompt_token_count", None),
                (getattr(um, "candidates_token_count", None) or 0)
                + (getattr(um, "thoughts_token_count", None) or 0), model)
    cand = (resp.candidates or [None])[0]
    if cand is None or not cand.content:
        return "", []
    parts = cand.content.parts or []
    text = "".join(p.text for p in parts if getattr(p, "text", None))
    calls = [{"id": f"c{i}", "name": p.function_call.name,
              "args": dict(p.function_call.args or {}),
              "sig": getattr(p, "thought_signature", None), "sig_model": model}
             for i, p in enumerate(parts) if getattr(p, "function_call", None)]
    return text, calls


async def _ask_openrouter(msgs, declared, system, api_key=None):
    body = json.dumps({
        "model": OPENROUTER_MODEL,
        "messages": _to_openai(msgs, system),
        "tools": _openai_tools(declared),
        "temperature": 0.2,
    }).encode()
    req = urllib.request.Request(
        OPENROUTER_URL, data=body, method="POST",
        headers={"Authorization": f"Bearer {api_key or os.environ['OPENROUTER_API_KEY']}",
                 "Content-Type": "application/json",
                 "HTTP-Referer": "https://cgiar-climate-data-hub.github.io/skills/",
                 "X-Title": "CDH Crop Yield Explorer",
                 "Accept": "application/json"})

    def _go():
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.loads(r.read())

    try:
        data = await asyncio.to_thread(_go)
    except urllib.error.HTTPError as exc:          # surface the body, not just the code
        raise RuntimeError(f"{exc.code} {exc.read().decode(errors='ignore')[:300]}") from exc

    # An upstream failure can arrive as HTTP 200 with an `error` body rather than
    # a completion, which reads as an empty reply.
    if data.get("error"):
        err = data["error"]
        raise RuntimeError(f"provider error: {str(err.get('message', err))[:200]}")
    u = data.get("usage") or {}
    _note_usage(u.get("prompt_tokens"), u.get("completion_tokens"), OPENROUTER_MODEL)
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    text = _strip_leaked_markup(_as_text(msg.get("content")))
    calls = []
    for i, tc in enumerate(msg.get("tool_calls") or []):
        fn = tc.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        calls.append({"id": tc.get("id") or f"c{i}", "name": fn.get("name", ""),
                      "args": args})
    return text, calls


ASK = {"gemini": _ask_gemini, "openrouter": _ask_openrouter}

# Token counts per model call, for planning (counts only, never content). The
# provider functions report the counts they get back; _ask hands them a dict.
_USAGE: contextvars.ContextVar[dict | None] = contextvars.ContextVar("cdh_usage", default=None)


def _note_usage(n_in, n_out, model) -> None:
    u = _USAGE.get()
    if u is not None:
        u.update({"in": n_in, "out": n_out, "model": model})


def _prompt_parts(msgs, declared, system) -> dict:
    """Characters per part of the prompt: the system prompt, the tool schemas,
    the conversation (user/assistant text and the calls), the tool results."""
    tools = len(json.dumps(_openai_tools(declared), default=str))
    hist = sum(len(m.get("text") or "") + len(json.dumps(m.get("calls") or [], default=str))
               for m in msgs if m["role"] != "tool")
    res = sum(len(m.get("result") or "") for m in msgs if m["role"] == "tool")
    return {"system": len(system), "tools": tools, "history": hist, "tool_results": res}


def _log_usage(sess_id: str, step: int, usage: dict, parts: dict) -> None:
    """One line per call: tokens in/out, and the input split across the parts
    in proportion to their characters (an estimate; the API reports a total)."""
    n_in = usage.get("in")
    tot = sum(parts.values()) or 1
    est = ({k: round(n_in * v / tot) for k, v in parts.items()} if n_in else {})
    log.info("tokens session=%s step=%d model=%s in=%s out=%s est_in=%s chars=%s",
             sess_id, step + 1, usage.get("model"), n_in, usage.get("out"),
             json.dumps(est), json.dumps(parts))


# ---- bring your own key ---------------------------------------------------
# When the shared quota runs out, a visitor can continue with their own Gemini or
# OpenRouter key. The key lives only in that tab's gr.State and is handed to the
# provider call as an argument: never to the Session, the model's messages, a
# tool, a file or the log. OpenRouter keys use the same model as the shared
# fallback; the visitor does not choose it.
OWN = "own"
PROVIDER_NAMES = {"gemini": "Gemini", "openrouter": "OpenRouter"}
KEY_LINKS = {"gemini": "https://aistudio.google.com/apikey",
             "openrouter": "https://openrouter.ai/settings/keys"}
# Key shapes, so a key is masked even where the exact value is not in scope
# (an exception from deep inside a client library, a key pasted into the chat).
_KEY_SHAPES = re.compile(
    r"AIza[0-9A-Za-z_\-]{20,}|sk-or-[0-9A-Za-z_\-]{10,}|sk-[0-9A-Za-z_\-]{20,}"
    r"|(?<=Bearer )[0-9A-Za-z_\-.]{12,}|(?<=key=)[0-9A-Za-z_\-]{12,}")
MASK = "[key hidden]"


def _redact(text, *keys) -> str:
    out = str(text)
    for k in keys:
        if k and len(k) >= 6:
            out = out.replace(k, MASK)
    return _KEY_SHAPES.sub(MASK, out)


class _RedactKeys(logging.Formatter):
    """Last line of defence: no key-shaped string reaches the log, tracebacks
    included (a filter sees only the message, not the formatted exception)."""

    def __init__(self, inner: logging.Formatter | None):
        super().__init__()
        self.inner = inner or logging.Formatter()

    def format(self, record):
        return _redact(self.inner.format(record))


for _h in logging.getLogger().handlers:
    if not isinstance(_h.formatter, _RedactKeys):
        _h.setFormatter(_RedactKeys(_h.formatter))


def _own_key(own) -> tuple[str, str] | None:
    """(provider, key) from a tab's key state, or None."""
    if isinstance(own, dict) and own.get("provider") in ASK and own.get("key"):
        return own["provider"], own["key"]
    return None


def _provider_label(p: str, own=None) -> str:
    """What the visitor sees: the provider, never the model id."""
    if p == OWN:
        ok = _own_key(own)
        return f"your {PROVIDER_NAMES[ok[0]]} key" if ok else "your key"
    kind = p.split("/")[0]
    return f"shared {PROVIDER_NAMES.get(kind, kind)}"


def _model_of(p: str, own=None) -> str:
    """The model id behind a provider entry, for the log."""
    if p == OWN:
        ok = _own_key(own)
        return (f"own {GEMINI_MODELS[0]}" if ok and ok[0] == "gemini"
                else f"own {OPENROUTER_MODEL}")
    kind, _, model = p.partition("/")
    return model or (OPENROUTER_MODEL if kind == "openrouter" else kind)


async def _ask(p: str, own, msgs, declared, system, usage: dict | None = None):
    """One model call. The visitor's key is resolved here, at the last moment,
    and nowhere else; a shared provider gets api_key=None and uses the env."""
    tok = _USAGE.set(usage if usage is not None else {})
    try:
        if p == OWN:
            prov, key = _own_key(own)
            return await ASK[prov](msgs, declared, system, api_key=key)
        kind, _, model = p.partition("/")
        return await ASK[kind](msgs, declared, system, **({"model": model} if model else {}))
    finally:
        _USAGE.reset(tok)


async def _test_key(provider: str, key: str) -> tuple[bool, str]:
    """One cheap call with the visitor's key: a few output tokens on the same
    model the app would use. Returns (works, message); the message never
    contains the key."""
    key = (key or "").strip()
    name = PROVIDER_NAMES.get(provider, provider)
    if provider not in ASK:
        return False, "Choose Gemini or OpenRouter."
    if len(key) < 10 or any(c.isspace() for c in key):
        return False, f"That does not look like a {name} key. Paste the whole key."
    try:
        if provider == "gemini":
            client = genai.Client(api_key=key, http_options=gt.HttpOptions(timeout=30_000))
            await asyncio.wait_for(asyncio.to_thread(
                client.models.generate_content, model=GEMINI_MODELS[0], contents="Reply with OK.",
                config=gt.GenerateContentConfig(max_output_tokens=8, temperature=0)),
                timeout=45)
        else:
            body = json.dumps({"model": OPENROUTER_MODEL, "max_tokens": 5,
                               "messages": [{"role": "user", "content": "Reply with OK."}]}).encode()
            req = urllib.request.Request(
                OPENROUTER_URL, data=body, method="POST",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                         "X-Title": "CDH Crop Yield Explorer"})

            def _go():
                try:
                    with urllib.request.urlopen(req, timeout=45) as r:
                        data = json.loads(r.read())
                except urllib.error.HTTPError as exc:
                    raise RuntimeError(f"{exc.code} {exc.read().decode(errors='ignore')[:300]}") from None
                if data.get("error"):
                    raise RuntimeError(f"provider error: {str(data['error'])[:200]}")
            await asyncio.to_thread(_go)
    except Exception as exc:  # noqa: BLE001
        err = _redact(f"{type(exc).__name__}: {exc}", key)
        low = err.lower()
        log.info("own-key test (%s) failed: %s", provider, err[:160])
        if any(s in low for s in ("api_key_invalid", "api key not valid", "401", "403",
                                  "unauthorized", "invalid api key", "no auth", "user not found",
                                  "permission_denied")):
            return False, f"{name} did not accept this key. Check that you copied all of it."
        if _is_quota(err) or "429" in low or "rate" in low:
            return False, (f"The key is valid, but its own {name} quota or rate limit is "
                           "used up too. Try again later or use the other provider.")
        if "timeout" in low or "timed out" in low or "urlopen error" in low:
            return False, f"Could not reach {name} to check the key. Try again in a moment."
        return False, f"{name} refused the test call: {err[:200]}"
    log.info("own-key test (%s) passed", provider)
    return True, f"Key works. Using: your {name} key."


# --------------------------------------------------------------------------
# The agent loop
# --------------------------------------------------------------------------
def _history_to_msgs(history, message, sess: Session) -> list[dict]:
    msgs = []
    for turn in history or []:
        content = turn.get("content") if isinstance(turn, dict) else None
        if content is None:
            continue
        # _redact: a key pasted into the chat is shown back to its owner but
        # never reaches the model in a later turn.
        msgs.append({"role": "user" if turn["role"] == "user" else "assistant",
                     "text": _redact(_unscrub_paths(_as_text(content), sess))})
    msgs.append({"role": "user", "text": message})
    return msgs


PLAN_GATED = {f"data{SEP}acquire_region", f"crop{SEP}run_simulation"}
PLAN_FIRST = (
    "NOT RUN: the user has not confirmed a plan for this request. A run starts "
    "only after the user replies yes to the plan as it stands; a reply that asks "
    "for a change, or a new request, needs a revised plan first. Call no more "
    "tools now. Reply with the plan (revised with any change they asked for): "
    "place, season and weather dates, "
    "the data (a fresh download, a few minutes, for a place without pre-built "
    "cubes), "
    "crop and cultivar, sowing date and windows, nitrogen, pixel cap and roughly "
    "how long. Say plainly which of those the user did not give and you chose "
    "(for example the season and the sowing date), and ask them to reply yes or "
    "change anything.")


# The whole reply must be a confirmation: "yes", "ok, go ahead", "proceed
# please". "Yes but with 60 kg" asks for a change, so the plan is revised first.
_CONFIRM = re.compile(
    r"^\s*(?:(?:yes|yep|yeah|y|ok|okay|sure|go|go ahead|proceed|run|run it|do it|"
    r"confirm|confirmed|correct|sounds good|looks good|s[ií]|dale|adelante|"
    r"please|thanks|thank you)[\s,.!]*)+$", re.I)


def _confirms(message) -> bool:
    return bool(_CONFIRM.match(_as_text(message or "")))


def _gate_turn_start(sess: Session, message) -> None:
    """A turn may run DSSAT only when it answers a shown plan with a yes."""
    sess.approved = sess.plan_shown and _confirms(message)
    sess.ran_slow = False
    sess.turn_tools = set()
    sess.last_message = _as_text(message or "")


# --- invented results --------------------------------------------------------
# A reply may only quote yields that a tool produced in this session. Once, asked
# "now the same with 60 kg of nitrogen", the model called nothing and wrote
# "now yielding 2,868 kg ha⁻¹" and a result file that did not exist.
_YIELD_RE = re.compile(
    r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*"
    r"(kg|kilograms?|t|tonnes?|tons?)\s*(?:ha⁻¹|ha-1|ha\^-1|/\s*ha\b|per\s+hectare)", re.I)
_PCT_RE = re.compile(r"\d+(?:\.\d+)?\s*%")
_FILE_RE = re.compile(r"\b[\w.-]+\.(?:nc|csv)\b")


def _known_yields(sess: Session) -> list[float]:
    """Every yield figure the session's result files can back up."""
    vals: list[float] = [0.0]
    for r in sess.runs():
        if r.get("mean_kg_ha") is not None:
            vals.append(float(r["mean_kg_ha"]))
        try:
            st = _window_stats(r["file"])
        except Exception:  # noqa: BLE001
            continue
        for w in st.get("windows", []):
            for k in ("mean_kg_ha", "min_kg_ha", "max_kg_ha", "p10_kg_ha", "p90_kg_ha"):
                if w.get(k) is not None:
                    vals.append(float(w[k]))
            vals += [float(v) for v in (w.get("season_means_kg_ha") or [])]
    return vals


def _yield_tol(raw: str, tonnes: bool) -> float:
    """How far a quoted yield may sit from the file's value: half a unit of
    the last digit it was written with. "6,470" ±1 kg, "6,500" ±50 kg, "6.5 t"
    ±50 kg, and never more than 0.5 % of the figure. The 0.5 % alone let
    "6,447" pass for 6,470."""
    num = raw.replace(",", "")
    if "." in num:
        step = 10 ** -len(num.split(".")[1])
    else:
        z = min(len(num) - len(num.rstrip("0")), 3) if num.strip("0") else 0
        step = 10 ** z
    tol = step / 2 * (1000 if tonnes else 1)
    v = float(num) * (1000 if tonnes else 1)
    return max(min(tol, 0.005 * v), 1.0)


def _unbacked_yield(sess: Session, text: str) -> str | None:
    """In a turn that ran DSSAT: a yield the result files do not back up (a
    figure copied from an example, say). A difference between two backed
    yields ("1,723 kg ha⁻¹ more") counts as backed."""
    if not text:
        return None
    known = None
    for m in _YIELD_RE.finditer(text):
        v = float(m.group(1).replace(",", ""))
        if m.group(2).lower().startswith("t"):
            v *= 1000
        if m.group(1) in sess.last_message:
            continue
        known = _known_yields(sess) if known is None else known
        tol = _yield_tol(m.group(1), m.group(2).lower().startswith("t"))
        near = lambda x: abs(v - x) <= tol  # noqa: E731
        if any(near(k) for k in known):
            continue
        if any(near(abs(x - y)) for i, x in enumerate(known) for y in known[i + 1:]):
            continue
        return f"yield {m.group(0)!r} is not in any result of this session"
    return None


def _results_summary(sess: Session) -> str:
    """The figures straight from this session's result files, newest first."""
    lines = []
    for r in sess.runs():
        try:
            st = _window_stats(r["file"])
        except Exception:  # noqa: BLE001
            continue
        for w in st.get("windows", []):
            lines.append(f"- {_run_label(r)}, sown {w.get('sown', '?')}: "
                         f"{w['mean_kg_ha']:,.0f} kg ha⁻¹ mean over {w.get('pixels', '?')} pixels")
    return ("Here are the results as the files hold them:" + NL + NL + NL.join(lines)
            + NL + NL + "The map, the tiles and the charts on the right show the same figures.")


def _invented(sess: Session, text: str) -> str | None:
    """Why a reply quotes results the session does not have, or None.

    A turn that ran DSSAT reports fresh numbers and is not checked. Any other
    turn may quote only yields found in the session's result files (to the
    rounding of a displayed value), may name only result files that exist, and
    may quote a percentage only right after reading results this turn."""
    if not text or "run_simulation" in sess.turn_tools:
        return None
    asked = sess.last_message
    known = None
    for m in _YIELD_RE.finditer(text):
        v = float(m.group(1).replace(",", ""))
        if m.group(2).lower().startswith("t"):
            v *= 1000
        if m.group(1) in asked:
            continue                     # the user's own figure, repeated
        known = _known_yields(sess) if known is None else known
        tol = _yield_tol(m.group(1), m.group(2).lower().startswith("t"))
        if not any(abs(v - k) <= tol for k in known):
            return f"yield {m.group(0)!r} is not in any result of this session"
    if _PCT_RE.search(text) and "read_yield" not in sess.turn_tools:
        pct = _PCT_RE.search(text).group(0)
        if pct not in asked:
            return f"percentage {pct!r} without reading results this turn"
    # A file the plan will write ("Output: yield_bean_yoro_2012.nc") does not
    # exist yet and is fine; one offered as a result ("download your new
    # results file …") must exist.
    have = {p.name for p in sess.dir.rglob("*") if p.is_file()}
    for line in text.splitlines():
        for f in _FILE_RE.findall(line):
            if f in have or not _AS_RESULT.search(line) or _AS_PLAN.search(line):
                continue
            return f"file {f!r} is offered as a result but does not exist in this session"
    return None


_AS_RESULT = re.compile(r"download|result|saved|written|ready|created|generated|contains", re.I)
_AS_PLAN = re.compile(r"\boutput\b\s*[:|]|\bwill\b|\bplan", re.I)


_CROPS = ("maize", "bean", "rice", "sorghum", "millet", "wheat", "soybean", "potato",
          "cassava", "sunflower", "tomato", "cabbage", "canola", "sugarcane")


def _fallback_plan(sess: Session, message: str) -> str:
    """A plan for what was asked, shown instead of a reply with invented numbers.

    Built from the request and the session's latest run (the "same" in "now the
    same with 60 kg"), so a yes runs exactly what was asked for."""
    msg = message or ""
    runs = sess.runs()
    base = runs[0] if runs else {}
    low = msg.lower()
    crop = next((c for c in _CROPS if re.search(rf"\b{c}s?\b", low)), None) \
        or str(base.get("crop") or "").lower() or None
    place = base.get("region") or base.get("country") or None
    for tag in PREBAKED_PLACES:
        if re.search(rf"\b{tag}\b", low):
            place = tag.title()
    years = re.findall(r"\b((?:19|20)\d{2})\b", msg)
    season = (years[0] if years else
              (str((base.get("seasons") or [None])[0]) if base.get("seasons") else None))
    rates = None
    if re.search(r"\bn\b|nitrogen|fertili[sz]", low):
        rates = [float(x) for x in re.findall(r"\b\d+(?:\.\d+)?\b", msg)
                 if not re.fullmatch(r"(?:19|20)\d{2}", x)]
    if not rates and base.get("n_kg_ha") is not None:
        rates = [float(base["n_kg_ha"])]
    if not (crop and place and season):
        return ("I have not run a simulation for this yet, so there are no results to "
                "report. Tell me the crop, the place and the season, and I will propose "
                "a plan to run.")
    n_win = int(base.get("n_windows") or 1)
    sow = str(base.get("planting_date") or "")[:10]
    gap = base.get("window_days")
    # "four dates two weeks apart" in the question wins over the last run.
    words = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
             "eight": 8, "nine": 9, "ten": 10, "one": 1, "a": 1}
    m = re.search(r"\b(\d+|" + "|".join(words) + r")\s+(?:sowing\s+|planting\s+)?dates\b", low)
    if m:
        n_win = int(m.group(1)) if m.group(1).isdigit() else words[m.group(1)]
    m = re.search(r"\b(\d+|" + "|".join(words) + r")\s+(day|week)s?\s+apart", low)
    if m:
        k = int(m.group(1)) if m.group(1).isdigit() else words[m.group(1)]
        gap = k * (7 if m.group(2) == "week" else 1)
    if n_win > 1 and not gap:
        gap = 14
    if not runs or (m and not sow):
        sow = ""                          # a new request: the date is the agent's to propose
    tag = _slug(place)
    data = "fresh download (a few minutes)"
    cubes = prebaked_regions().get(tag)
    if tag in PREBAKED_PLACES and cubes:
        try:
            first, last, _ = _cube_span(str(cubes["weather"]))
            if first[:4] <= str(season) <= last[:4]:
                data = f"{PREBAKED_LABEL} (instant)"
        except Exception:  # noqa: BLE001
            pass
    n_txt = (", ".join(f"{r:g}" for r in rates) + " kg N ha⁻¹" + (" (one run per rate)"
             if len(rates) > 1 else "")) if rates else "none (rainfed, no fertiliser)"
    start = sow or "a start date at the beginning of the main rainy season (my guess)"
    sow_txt = (f"{n_win} dates, {gap} days apart, from {start}" if n_win > 1 and gap
               else start)
    return (
        "I have not run this yet, so there are no results for it to report.\n\n"
        "**Plan**\n\n"
        f"- Place: {place} · Season: {season}\n"
        f"- Crop: {crop.title()}\n"
        f"- Nitrogen: {n_txt}\n"
        f"- Sowing: {sow_txt}\n"
        f"- Data: {data}\n\n"
        "Reply **yes** to run it, or tell me what to change.")


def _gate_turn_end(sess: Session) -> None:
    """A turn that ran nothing slow ended on the agent's plan, revision or
    question; the next message is the answer to it."""
    sess.plan_shown = not sess.ran_slow
    sess.approved = False


def _is_slow(name: str, args: dict | None = None) -> bool:
    """DSSAT, or a real download. Handing back the pre-built cubes is neither."""
    if name not in PLAN_GATED:
        return False
    if name == f"data{SEP}acquire_region":
        a = args or {}
        y0, _ = _season_years(a.get("season", ""), a.get("planting_date", ""))
        return _prebuilt_for(a.get("country_code", ""), a.get("feature_name", ""),
                             a.get("adm_level"), y0) is None
    return True


def _plan_pending(sess: Session, name: str, args: dict | None = None) -> bool:
    """True when a slow tool would start before the user confirmed a plan.

    The prompt asks for plan-then-confirm, but a small model sometimes downloads
    a new region, or runs DSSAT, straight from the first message. Whether a plan
    was shown and confirmed is session state set by the turns themselves
    (`_gate_turn_start` / `_gate_turn_end`), not read back from the chat text."""
    # The pre-built cubes are not held: instant, nothing downloads, and the
    # answer lets the plan say "pre-built data cubes". DSSAT still waits.
    if not _is_slow(name, args):
        return False
    return not sess.approved


def _scope_args(sess: Session, args: dict) -> dict:
    """Keep every output the model names inside the session folder."""
    out = dict(args)
    for k in OUTPUT_KEYS:
        if k in out and out[k] is not None and not str(out[k]).startswith(str(sess.dir)):
            name = pathlib.Path(str(out[k])).name or k
            out[k] = str(sess.dir / ("dssat" if k == "working_path" else name))
    return out


LIMIT_NOTICE = "The demo has reached today's limit, please try again later."
QUOTA_NOTICE = (
    "The shared AI quota for this demo is used up for today. You can continue "
    "with your own key. It is used only in this browser session and never stored.")


def _providers_for(own) -> list[str]:
    """Per session: the visitor's key first, then the shared providers."""
    ready = [p for p in PROVIDERS if _provider_ready(p)]
    shared = [p for p in ready if not _in_cooldown(p)] or ready
    return ([OWN] if _own_key(own) else []) + shared


def _pause(sess: Session, msgs, message, step: int, said: list[str]) -> str:
    """Keep the run where it stopped; the visitor's key picks it up from here.
    With BYOK off there is no key to pick it up: a plain message instead."""
    if not BYOK_ENABLED:
        log.warning("session %s: shared quota used up; told to try later", sess.id)
        done = [s for s in said if s.startswith("✓")]
        return "\n\n".join(done + [f"**{LIMIT_NOTICE}**"])
    sess.paused = {"msgs": msgs, "message": message, "step": step,
                   "approved": sess.approved, "ran_slow": sess.ran_slow,
                   "turn_tools": set(sess.turn_tools), "region": sess.region,
                   "sowing": sess.sowing, "at": time.time()}
    log.warning("session %s: shared quota used up; run paused at step %d", sess.id, step + 1)
    done = [s for s in said if s.startswith("✓")]
    return "\n\n".join(done + [
        f"**Paused.** {QUOTA_NOTICE}",
        "Add a key in the panel above the chat and the run continues from this "
        "step" + (" — the data already prepared is kept." if done else ".")])


async def _turn(sess: Session, history, message, own=None):
    """One user message -> the agent's reply, running tools in between."""
    msgs = _history_to_msgs(history, message, sess)
    # The chat history carries text, not tool answers, so after a "yes" the
    # model looked the place up again. Say what is already found (model only).
    if sess.places:
        known = "; ".join(
            f"{v['best']['name']} ({v['best']['country_code']}, adm_level "
            f"{v['best']['adm_level']}, in {v['best'].get('adm1') or '-'})"
            for v in sess.places.values() if "best" in v)
        msgs[-1]["text"] = (f"{msgs[-1]['text']}\n\n[Places already found with find_place: "
                            f"{known}. Use them; do not call find_place for them again.]")
    sess.paused = None
    async for chunk in _agent_loop(sess, msgs, message, 0, own):
        yield chunk


async def _agent_loop(sess: Session, msgs, message, step0: int, own=None):
    """The model/tool loop from `step0`: a new turn starts at 0, a resumed one
    where it paused, with the tool results it already had."""
    usable = _providers_for(own)
    said: list[str] = []
    if not usable:
        yield _pause(sess, msgs, message, step0, said)
        return
    system = system_prompt(sess)
    sess.last_result = ""
    provider = usable[0]
    exhausted: set[str] = set()
    out_of_quota: set[str] = set()
    attempts: dict[str, int] = {}
    answered: list[str] = []      # model id per call, logged when the turn ends
    try:
        async for chunk in _agent_steps(sess, msgs, message, step0, own, usable, said, system,
                                        provider, exhausted, out_of_quota, attempts, answered):
            yield chunk
    finally:
        if answered:
            log.info("session %s: turn answered by %s (%d calls)", sess.id,
                     " -> ".join(dict.fromkeys(answered)), len(answered))


async def _agent_steps(sess, msgs, message, step0, own, usable, said, system,
                       provider, exhausted, out_of_quota, attempts, answered):
    rechecked = False             # one corrected re-ask per turn for unbacked yields
    narration: set[int] = set()   # indexes in `said` of text written between tool calls
    own_secret = (_own_key(own) or (None, None))[1]

    for _step in range(step0, MAX_STEPS):
        text, calls = "", []
        sess.status = f"Step {_step + 1} · thinking"
        while True:
            try:
                usage: dict = {}
                text, calls = await _ask(provider, own, msgs, DECLARED, system, usage)
                usage.setdefault("model", _model_of(provider, own))
                _log_usage(sess.id, _step, usage, _prompt_parts(msgs, DECLARED, system))
                sess.provider = _provider_label(provider, own)
                answered.append(_model_of(provider, own))
                break
            except Exception as exc:  # noqa: BLE001
                err = _redact(exc, own_secret)
                attempts[provider] = attempts.get(provider, 0) + 1
                # "High demand" 503s from Gemini come in bursts of a minute or
                # so; one retry after four seconds walked straight into the
                # same burst and abandoned a finished simulation. Three tries
                # with growing waits ride it out; a rate limit needs its window.
                if _is_transient(err) and attempts[provider] < 4:
                    rate = any(s in err.lower() for s in
                               ("429", "rate limit", "rate-limit", "too many requests"))
                    asked = _retry_seconds(err)
                    wait = (min(90, int(asked) + 3) if asked is not None
                            else 30 if rate else 5 * attempts[provider])
                    log.warning("%s hiccuped (%s); retry %d in %ds",
                                _model_of(provider, own), err[:90], attempts[provider], wait)
                    said.append(f"*{_provider_label(provider, own)} is busy — retrying in "
                                f"{wait}s ({attempts[provider]}/3)…*")
                    yield "\n\n".join(said)
                    await asyncio.sleep(wait)
                    said.pop()
                    continue
                exhausted.add(provider)
                low = err.lower()
                limited = _is_quota(err) or any(
                    s in low for s in ("429", "rate limit", "rate-limit", "too many requests"))
                if limited:
                    out_of_quota.add(provider)
                if _is_quota(err) and provider != OWN:   # the cooldown is for shared keys
                    _mark_quota_out(provider)
                nxt = next((p for p in usable
                            if p != provider and p not in exhausted), None)
                if nxt is None:
                    # Every model is out of quota or rate-limited: pause, keep
                    # the work done so far, and ask for the visitor's own key.
                    if out_of_quota or all(p == OWN for p in usable):
                        yield _pause(sess, msgs, message, _step, said)
                        return
                    raise RuntimeError(err) from None
                if _is_quota(err):
                    why = "is out of quota"
                elif provider == OWN and not limited:
                    why = "was not accepted"
                else:
                    why = "is unavailable"
                was, now = _provider_label(provider, own), _provider_label(nxt, own)
                log.warning("session %s: %s %s; switching to %s", sess.id,
                            _model_of(provider, own), why, _model_of(nxt, own))
                # Which model answers is for the log; the chat only hears about
                # the visitor's own key being passed over.
                if OWN in (provider, nxt):
                    said.append(f"*{was[0].upper()}{was[1:]} {why} — continuing on {now}.*")
                    yield "\n\n".join(said)
                provider = nxt

        if "run_simulation" in sess.turn_tools and not calls:
            why = _unbacked_yield(sess, text)
            if why and not rechecked:
                # Not shown. One more try, told exactly what is wrong.
                rechecked = True
                log.warning("session %s: reply held back for a recheck (%s)", sess.id, why)
                msgs.append({"role": "assistant", "text": text})
                msgs.append({"role": "user", "text": (
                    f"[Page check: {why}. Rewrite the whole reply, answer first, quoting "
                    "only yields that crop__read_yield returned in this session, exactly "
                    "as it gave them. Do not mention this check.]").replace("crop__", f"crop{SEP}")})
                continue
            if why:
                log.warning("session %s: reply replaced by the file figures (%s)", sess.id, why)
                said.append(_results_summary(sess))
                yield "\n\n".join(said)
                return
        why = _invented(sess, text)
        if why:
            # Never shown: a reply quoting results nothing produced. In its
            # place, a plan for what was asked; a yes then runs it.
            log.warning("session %s: reply withheld (%s)", sess.id, why)
            text = ""
            if not calls:
                said[:] = [x for i, x in enumerate(said) if i not in narration]
                said.append(_fallback_plan(sess, message))
                yield "\n\n".join(said)
                return
        if text and not calls:
            # The answer comes first: narration from earlier steps ("Now
            # running the simulation:") is dropped; the progress lines stay.
            said[:] = [x for i, x in enumerate(said) if i not in narration]
            said.append(text)
            yield "\n\n".join(said)
        elif text:
            narration.add(len(said))
            said.append(text)
            yield "\n\n".join(said)
        if not calls:
            return

        msgs.append({"role": "assistant", "calls": calls})
        for call in calls:
            name = call["name"] or ""
            prefix, _, bare = name.partition(SEP)
            args = _scope_args(sess, _whole_floats_to_int(dict(call["args"])))
            local = LOCAL_TOOLS.get(name)
            if local is None and (name not in SERVER_FUNCS or name in HIDDEN or not bare):
                log.warning("unroutable tool call %r", name)
                msgs.append({"role": "tool", "id": call["id"], "name": name or "unknown",
                             "result": (f"ERROR no such tool {name!r}. Names are "
                                        f"prefixed data{SEP}* for downloads and "
                                        f"crop{SEP}* for the crop model.")})
                continue
            if _plan_pending(sess, name, args):
                log.info("held %s until the user confirms a plan", name)
                msgs.append({"role": "tool", "id": call["id"], "name": name,
                             "result": PLAN_FIRST})
                continue
            if name == f"crop{SEP}generate_config":
                fixed = _enforce_config_args(sess, args)
                if isinstance(fixed, str):
                    msgs.append({"role": "tool", "id": call["id"], "name": name,
                                 "result": fixed})
                    continue
                args = fixed
            t0 = time.time()
            budget = ACQUIRE_TIMEOUT if name == f"data{SEP}acquire_region" else TOOL_TIMEOUT
            prog: dict = {"n": 0, "total": 0, "desc": "", "stage": ""}
            # The task copies the context at creation, so set the progress
            # target just for that moment and the tool reports into `prog`.
            token = PROGRESS.set(prog)
            try:
                task = asyncio.ensure_future(
                    local(sess, **args) if local is not None
                    else _call_in_process(name, args))
            finally:
                PROGRESS.reset(token)
            idx = len(said)
            said.append(_progress_line(bare, prog, t0))
            sess.status = _status_text(_step, bare, prog, t0)
            yield "\n\n".join(said)
            try:
                while True:
                    done, _ = await asyncio.wait({task}, timeout=2.0)
                    if done:
                        break
                    if time.time() - t0 > budget:
                        task.cancel()
                        raise asyncio.TimeoutError
                    said[idx] = _progress_line(bare, prog, t0)
                    sess.status = _status_text(_step, bare, prog, t0)
                    yield "\n\n".join(said)
                payload = task.result()
                if local is not None:
                    sess.last_result = payload
                # A refused run ("NOT RUN: …") did no slow work: the plan it
                # sends back still needs the user's yes.
                if _is_slow(name, args) and "NOT RUN:" not in str(payload)[:200]:
                    sess.ran_slow = True
                sess.turn_tools.add(bare)
                if bare == "list_admin_units":
                    payload = f"{payload}{_prebaked_hint(args)}"
                if bare == "generate_config":
                    payload = f"{payload}{_config_fit_note(args, payload)}"
                label = _tool_label(bare)
                if bare == "acquire_region" and PREBAKED_LABEL in str(payload):
                    label = f"Using {PREBAKED_LABEL}"
                mark = "✓"
                if bare == "run_simulation" and "SEASON CHECK: DOES NOT FIT" in str(payload):
                    mark, label = "✗", "DSSAT not run — the sowing dates do not fit the weather data"
                elif bare == "run_simulation" and '"error"' in str(payload)[:40]:
                    mark, label = "✗", "DSSAT not run"
                said[idx] = f"{mark} {label} · {_fmt_s(time.time() - t0)}"
                log.info("%s -> %.1fs", bare, time.time() - t0)
            except asyncio.TimeoutError:
                payload = (f"ERROR {bare} did not return within {budget:.0f}s and was "
                           "abandoned. The usual cause is a path that does not exist, "
                           "so the run waits on a file that never arrives.")
                said[idx] = _failed_line(bare, f"no answer within {_fmt_s(budget)}")
                log.error("%s timed out", bare)
            except Exception as exc:  # noqa: BLE001
                payload = f"ERROR {type(exc).__name__}: {exc}"
                said[idx] = _failed_line(bare, f"{type(exc).__name__}: {exc}")
                log.warning("%s failed: %s", bare, exc)
            yield "\n\n".join(said)
            msgs.append({"role": "tool", "id": call["id"], "name": name,
                         "result": payload[:8000]})

    yield "\n\n".join(said + [f"*(stopped after {MAX_STEPS} steps)*"])


KEY_IN_CHAT = (
    "That looks like an API key, so it was not sent to the model. Add your key "
    "under **Use your own API key** below the chat instead: it stays in this "
    "browser session and is not stored.")
KEY_IN_CHAT_OFF = ("That looks like an API key, so it was not sent to the model. "
                   "Please do not paste keys into the chat.")


def _complete_history(history, sess: Session) -> list:
    """The browser's history, or the server's record of the conversation when
    the browser's is missing the last finished reply (a message sent before
    the previous turn ended carries the history from before that turn)."""
    history = list(history or [])
    if not sess.exchanges:
        return history
    shown = [_as_text(t.get("content")).strip() for t in history
             if isinstance(t, dict) and t.get("role") == "assistant"]
    kept = [(e["assistant"] or "").strip() for e in sess.exchanges]
    # Only a stale copy (the server's replies, minus the latest ones) is
    # replaced. A history that differs (the visitor undid or retried a
    # message) is the visitor's choice and is kept.
    if len(shown) >= len(kept) or shown != kept[:len(shown)]:
        return history
    log.info("session %s: chat history was behind the server's; using the server's", sess.id)
    out = []
    for e in sess.exchanges:
        if e.get("user") is not None:
            out.append({"role": "user", "content": e["user"]})
        out.append({"role": "assistant", "content": e["assistant"]})
    return out


async def respond(message, history, own, request: gr.Request):
    """Stream the agent's reply with paths scrubbed, and clear the panel's
    status line when the turn ends, however it ends. `own` is this tab's key
    state; it goes no further than the provider call.

    Turns of one session run one at a time: a "yes" sent while the plan is
    still being written waits for it, so the plan gate and the history it is
    checked against are the finished turn's."""
    sess = session_of(request)
    if _KEY_SHAPES.search(_as_text(message or "")):
        yield KEY_IN_CHAT if BYOK_ENABLED else KEY_IN_CHAT_OFF
        return
    if not CUBES_READY and _as_text(message or "").strip() in EXAMPLES:
        yield CUBES_UNAVAILABLE
        return
    if not BYOK_ENABLED:
        own = None                       # nothing to use, whatever is sent
    if sess.turn_lock is None:
        sess.turn_lock = asyncio.Lock()
    if sess.turn_lock.locked():
        log.info("session %s: message arrived during a turn; it waits for that turn", sess.id)
        yield "*Waiting for the previous reply to finish…*"
    async with sess.turn_lock:
        history = _complete_history(history, sess)
        sess.status = "Step 1 · thinking"
        _gate_turn_start(sess, message)
        last = ""
        try:
            async for chunk in _reply(sess, message, history, own):
                last = _plain_terms(_scrub_paths(chunk, sess))
                yield last
        finally:
            sess.status = ""
            _gate_turn_end(sess)
            sess.exchanges.append({"user": _as_text(message), "assistant": last})


async def resume_run(history, own, request: gr.Request):
    """Continue a paused run with the visitor's key, from the step it stopped
    at: the tool results it had (the downloaded cubes among them) are reused,
    nothing is fetched again. Streams a new assistant message into the chat."""
    sess = session_of(request)
    p = sess.paused
    # Only right after a key passed its test (use_key marks the paused run).
    if p is None or not p.get("resuming") or not _own_key(own):
        yield gr.skip(), gr.skip()
        return
    sess.paused = None
    sess.approved, sess.ran_slow = p["approved"], p["ran_slow"]
    sess.turn_tools, sess.last_message = set(p["turn_tools"]), _as_text(p["message"])
    sess.region, sess.sowing = p["region"], p["sowing"]
    history = list(history or [])
    head = f"*Continuing with {_provider_label(OWN, own)}…*"
    log.info("session %s: resuming the paused run at step %d", sess.id, p["step"] + 1)
    sess.status = f"Step {p['step'] + 1} · thinking"
    msgs = p["msgs"] or _history_to_msgs([], p["message"], sess)
    text = ""
    if sess.turn_lock is None:
        sess.turn_lock = asyncio.Lock()
    async with sess.turn_lock:           # a turn like any other: one at a time
        try:
            yield history + [{"role": "assistant", "content": head}], gr.update(interactive=False)
            async for chunk in _reply(sess, p["message"], [], own, resume=(msgs, p["step"])):
                text = _plain_terms(_scrub_paths(chunk, sess))
                yield history + [{"role": "assistant", "content": f"{head}\n\n{text}"}], gr.skip()
        finally:
            sess.status = ""
            _gate_turn_end(sess)
            sess.exchanges.append({"user": None, "assistant": f"{head}\n\n{text}"})
    yield gr.skip(), gr.update(interactive=True)


async def _reply(sess: Session, message, history, own=None, resume=None):
    """Stream the agent's reply, surfacing any failure in the chat itself.

    An exception escaping this generator reaches Gradio as a red "Error" toast
    with no detail. Showing it here costs nothing and makes the Space debuggable
    by whoever is using it.
    """
    secret = (_own_key(own) or (None, None))[1]
    try:
        gen = (_agent_loop(sess, resume[0], message, resume[1], own) if resume
               else _turn(sess, history or [], message, own))
        async for chunk in gen:
            yield _redact(chunk, secret)
    except Exception as exc:  # noqa: BLE001
        log.exception("turn failed")
        detail = _redact(traceback.format_exc()[-1500:], secret)
        what = _redact(f"{type(exc).__name__}: {exc}", secret)
        if _is_quota(detail):
            yield ("**The model's daily quota is used up.**" + NL + NL
                   + "The free tier allows a small number of requests per day, "
                   "and one conversation uses several. It resets at midnight UTC."
                   + NL + NL + "Nothing is wrong with the simulation itself.")
            return
        if "429" in detail or "exceeded your current quota" in detail:
            asked = _retry_seconds(detail)
            yield ("**The model is being rate-limited.**" + NL + NL
                   + "The free tier allows only so many tokens a minute and this "
                   "conversation just used them. "
                   + (f"Wait about {int(asked) + 5} seconds, " if asked
                      else "Wait a minute, ")
                   + "then send your last message again — nothing has been lost.")
            return
        salvage = sess.last_result
        if salvage and any(c in detail for c in ("503", "UNAVAILABLE", "429")):
            yield ("**The simulation finished, but the model was unavailable to "
                   "summarise it.** Here is the raw result — the numbers are "
                   f"good:{NL}{NL}```{NL}{salvage[:2000]}{NL}```{NL}{NL}"
                   "Ask again in a moment and it will explain them.")
            return
        if "API_KEY_INVALID" in detail or "API key not valid" in detail:
            yield ("**The model API key is not valid.** Set a working key under "
                   "*Settings -> Variables and secrets* on the Space, or as an "
                   "environment variable locally. Nothing else is wrong.")
            return
        yield (NL + NL).join([
            "**Something went wrong on our side.** This is a fault in the Space, "
            "not in your request — send your message again, or try a pre-built "
            "example.",
            "<details><summary>details</summary>",
            f"`{what}`",
            f"```{NL}{detail}{NL}```",
            "</details>",
        ])


# --------------------------------------------------------------------------
# Result panel: tiles, map, charts
# --------------------------------------------------------------------------
# One accent for the whole page: the theme, the bars, the best-bar highlight and
# the map ramp all come from ACCENT, so no second hue competes for attention.
# The map ramp is that hue from light (low yield) to dark (high), starting above
# white so the lowest cells still read against the basemap. Categorical colours
# after the first appear only when a nitrogen chart has several sowing windows.
ACCENT = "#2e7d4f"            # 5.1:1 on white
ACCENT_SOFT = "#a7d0b4"       # the bars that are not the best one
ACCENT_RAMP = ["#cfe8d6", "#8fc4a0", "#4f9d6d", "#2e7d4f", "#174a2c"]
SERIES = [ACCENT, "#eb6834", "#6250d6", "#eda100", "#e87ba4", "#2a78d6"]
INK = "#1f2937"
INK_MUTED = "#4b5563"         # 7.6:1 on white
GRID = "#e5e7eb"
AXIS = "#9ca3af"
FONT = "Inter, Helvetica, Arial, sans-serif"
YIELD_AXIS = "Grain yield (kg ha⁻¹)"
# The map iframe's height. Set once here and once in CSS (as a variable) so the
# placeholder and the map are the same size and the layout never jumps.
MAP_H = int(os.environ.get("CDH_MAP_H", "420"))

PLACEHOLDER = (
    "<div class='placeholder'><div><b>No results yet</b><br>"
    "Ask for a yield simulation in the chat. The map and charts fill in "
    "when a run finishes.</div></div>")


def _attrs_of(path: str) -> dict:
    import xarray as xr
    try:
        with xr.open_dataset(path) as ds:
            a = dict(ds.attrs)
    except Exception:  # noqa: BLE001
        return {}
    for k in ("seasons",):
        if isinstance(a.get(k), str):
            try:
                a[k] = json.loads(a[k])
            except ValueError:
                pass
    return a


def _windows_in(path: str | None) -> list[str]:
    """Sowing dates in a result file, labelled for a dropdown."""
    import xarray as xr
    if not path:
        return []
    try:
        with xr.open_dataset(path) as ds:
            n = int(ds.sizes.get("planting_window", 1))
            if "planting_date" in ds.coords:
                # planting_date holds the FIRST season's dates, so a five-season
                # run labelled "sown 2016-03-15" reads as a 2016 result. What
                # varies between windows is the day of year, not the year.
                keep, _, _ = _real_seasons(ds["HWAM"], ds.attrs)
                n_yr = len(keep) if keep else int(ds.sizes.get("year", 1))
                dates = np.atleast_1d(ds.planting_date.values)
                if n_yr > 1:
                    return [f"{i + 1}: {pd.Timestamp(str(d)).strftime('%d %b')}"
                            f" · {n_yr}-season mean"
                            for i, d in enumerate(dates)]
                return [f"{i + 1}: {str(d)[:10]}" for i, d in enumerate(dates)]
            return [f"{i + 1}" for i in range(n)]
    except Exception:  # noqa: BLE001
        return ["1"]


def _window_index(label: str | None) -> int:
    if label and ":" in label:
        try:
            return int(label.split(":")[0]) - 1
        except ValueError:
            return 0
    return 0


def _grid_of(path: str, idx: int):
    """One planting window of a result as (values, lats, lons, caption bits)."""
    import xarray as xr
    with xr.open_dataset(path) as ds:
        da = ds["HWAM"]
        # Only real seasons, as in the tiles (_real_seasons).
        keep, _, _ = _real_seasons(da, ds.attrs)
        if "year" in da.dims and keep:
            da = da.isel(year=keep)
        n_yr = int(da.sizes.get("year", 1))
        if "planting_window" in da.dims:
            idx = min(idx, da.sizes["planting_window"] - 1)
            da = da.isel(planting_window=idx)
        # Average the seasons rather than showing the first one: a map captioned
        # "mean of 5 seasons" that draws 2016 alone would be quietly wrong.
        for extra in ("year", "time"):
            if extra in da.dims:
                da = (da.mean(extra, skipna=True) if da.sizes[extra] > 1
                      else da.isel({extra: 0}))
        sown = ""
        if "planting_date" in ds.coords:
            dates = np.atleast_1d(ds.planting_date.values)
            if idx < len(dates):
                sown = (pd.Timestamp(str(dates[idx])).strftime("%d %b") if n_yr > 1
                        else str(dates[idx])[:10])
        ydim = "y" if "y" in da.dims else "lat"
        xdim = "x" if "x" in da.dims else "lon"
        vals = da.values.astype(float)
        lats = da[ydim].values.astype(float)
        lons = da[xdim].values.astype(float)
        attrs = dict(ds.attrs)
    return vals, lats, lons, sown, n_yr, attrs


def _season_name(year, sown, crop) -> str:
    """'2018/19' for a season that runs into the next year (maize sown in
    Chipata in December), '2019' otherwise."""
    try:
        y = int(year)
        if sown:
            end = pd.Timestamp(str(sown)) + pd.Timedelta(_maturity_days(crop), unit="D")
            if end.year > y:
                return f"{y}/{(y + 1) % 100:02d}"
        return str(y)
    except (TypeError, ValueError):
        return str(year)


def tiles_html(path: str | None, window_label: str | None,
               series: dict | None = None) -> str:
    """The headline numbers for the selected run and window. `series` is what
    the sowing-date chart compares, so "best sowing" agrees with the chart
    when it was assembled from separate runs."""
    if not path:
        return ""
    try:
        stats = _window_stats(path)
    except Exception:  # noqa: BLE001
        return ""
    idx = _window_index(window_label)
    windows = stats["windows"]
    cur = next((w for w in windows if w["window"] == idx + 1), windows[0] if windows else None)
    if cur is None:
        return "<div class='tiles'><div class='tile'><div class='tile-v'>—</div>" \
               "<div class='tile-l'>no simulated pixels</div></div></div>"
    best = stats["best_window"]
    attrs = _attrs_of(path)
    n_rate = attrs.get("n_kg_ha")
    n_seasons = stats.get("n_seasons", 1)
    crop = attrs.get("crop") or "Yield"
    years = stats.get("seasons") or []
    season_txt = (f"mean of {n_seasons} seasons" if n_seasons > 1 else
                  (_season_name(years[0], cur.get("sown"), crop) if years else "one season"))
    n_txt = (f"{float(n_rate):g} kg N ha⁻¹" if n_rate not in (None, "")
             else "rainfed, no N")
    # One sowing date: there is no "best" to name, so show when it was sown.
    one_sown = cur.get("sown")
    best_txt = (pd.Timestamp(one_sown).strftime("%d %b %Y") if one_sown else n_txt)
    best_lbl = "sowing date" if one_sown else "nitrogen applied"
    if series and series.get("merged"):
        best = max(series["windows"], key=lambda w: w["mean_kg_ha"])
        windows = series["windows"]
    if best and len(windows) > 1:
        sown = best.get("sown")
        best_txt = (pd.Timestamp(sown).strftime("%d %b") if sown else f"window {best['window']}")
        # The tile compares all dates, so say so: next to a picked date it
        # otherwise reads as the date being shown.
        best_lbl = (f"best of {len(windows)} sowing dates · "
                    f"{best['mean_kg_ha']:,.0f} kg ha⁻¹")
    cur_sown = cur.get("sown")
    sown_txt = (f"sown {pd.Timestamp(cur_sown).strftime('%d %b')} · "
                if cur_sown and len(windows) > 1 else "")
    third_lbl = best_lbl if best_lbl == "nitrogen applied" else f"{best_lbl} · {n_txt}"
    # Pixels that gave no yield are part of the mean (a real outcome, often
    # too cold or steep); the map greys them out and this says how many.
    nz = int(cur.get("pixels_no_yield") or 0)
    no_yield_txt = f" · {nz} gave no yield" if nz else ""
    # Complete weather and nothing harvested on any pixel: say what it is.
    fail_txt = ("<b>crop failure</b>: every pixel yielded 0 · " if cur.get("crop_failure")
                else "mean yield · ")
    return f"""
<div class='tiles'>
  <div class='tile hero'>
    <div class='tile-v'>{cur['mean_kg_ha']:,.0f}<span class='unit'>kg ha⁻¹</span></div>
    <div class='tile-l'>{crop} · {fail_txt}{sown_txt}{season_txt}</div>
  </div>
  <div class='tile'>
    <div class='tile-v'><span class='nw'>{cur['min_kg_ha']:,.0f}–{cur['max_kg_ha']:,.0f}</span> <span class='unit'>kg ha⁻¹</span></div>
    <div class='tile-l'>range across {cur['pixels']} pixels{no_yield_txt}</div>
  </div>
  <div class='tile'>
    <div class='tile-v'>{best_txt}</div>
    <div class='tile-l'>{third_lbl}</div>
  </div>
</div>"""


from branca.element import MacroElement   # noqa: E402

ZERO_RGBA = (0.78, 0.76, 0.73, 0.85)        # crop failure, distinct from no data
from jinja2 import Template                # noqa: E402


class _YieldHover(MacroElement):
    """A readout in the map corner: the yield and position of the cell under
    the pointer. An image layer has no per-pixel tooltip, so the grid's cell
    edges and values travel with the map and the lookup happens in the page."""

    _template = Template("""
{% macro script(this, kwargs) %}
(function () {
  var map = {{ this._parent.get_name() }};
  var G = {{ this.data }};
  function find(e, q) {
    var asc = e[e.length - 1] > e[0];
    for (var k = 0; k < e.length - 1; k++) {
      var a = e[k], b = e[k + 1];
      if ((asc && q >= a && q < b) || (!asc && q <= a && q > b)) return k;
    }
    return -1;
  }
  var info = L.control({position: 'topright'});
  info.onAdd = function () {
    var d = L.DomUtil.create('div', 'yield-hover');
    d.style.cssText = 'background:rgba(255,255,255,.94);padding:5px 9px;' +
      'border-radius:6px;font:12px/1.4 Inter,Helvetica,Arial,sans-serif;color:#1f2937;' +
      'box-shadow:0 1px 3px rgba(0,0,0,.25);display:none;pointer-events:none';
    this._d = d; return d;
  };
  info.addTo(map);
  map.on('mousemove', function (ev) {
    var i = find(G.ey, ev.latlng.lat), j = find(G.ex, ev.latlng.lng);
    var v = (i >= 0 && j >= 0) ? G.v[i][j] : null, d = info._d;
    if (v === null) { d.style.display = 'none'; return; }
    d.style.display = 'block';
    d.innerHTML = (v === 0 ? '<b>No harvest</b> (0 kg ha⁻¹)' :
      '<b>' + v.toLocaleString('en-US') + ' kg ha⁻¹</b>') + '<br>' +
      G.lat[i].toFixed(3) + ', ' + G.lon[j].toFixed(3);
  });
  map.on('mouseout', function () { info._d.style.display = 'none'; });
})();
{% endmacro %}
""")

    def __init__(self, grid: dict):
        super().__init__()
        self._name = "YieldHover"
        self.data = json.dumps(grid)


class _Legend(MacroElement):
    """The colour scale as a white card in the bottom-right corner, with the
    variable and its units as the title. branca's default legend sat on top
    of the tiles, where its tick labels collided with the basemap's place names."""

    _template = Template("""
{% macro header(this, kwargs) %}
<style>
.yield-legend{background:#fff;padding:8px 10px 7px;border-radius:6px;
  box-shadow:0 1px 3px rgba(0,0,0,.25);font:11px/1.35 Inter,Helvetica,Arial,sans-serif;
  color:#1f2937;width:190px}
.yield-legend .t{font-weight:600;font-size:12px}
.yield-legend .s{color:#4b5563;margin-bottom:5px}
.yield-legend .bar{height:10px;border-radius:2px}
.yield-legend .ticks{display:flex;justify-content:space-between;color:#4b5563;
  margin-top:2px;font-variant-numeric:tabular-nums}
.yield-legend .z{margin-top:4px;color:#4b5563}
.yield-legend .z i{display:inline-block;width:10px;height:10px;border-radius:2px;
  vertical-align:-1px;margin-right:5px}
</style>
{% endmacro %}
{% macro script(this, kwargs) %}
(function () {
  var map = {{ this._parent.get_name() }};
  var c = L.control({position: 'bottomright'});
  c.onAdd = function () {
    var d = L.DomUtil.create('div', 'yield-legend');
    d.innerHTML = {{ this.html }};
    return d;
  };
  c.addTo(map);
})();
{% endmacro %}
""")

    def __init__(self, html: str):
        super().__init__()
        self._name = "YieldLegend"
        self.html = json.dumps(html)


def yield_map(path: str | None = None, window_label: str | None = None) -> str:
    """Render one planting window as a Leaflet map, returned as embeddable HTML.

    The grid is drawn as a raster with a hover readout of each cell, so the
    map answers "how much, here" rather than only "roughly where is it green".
    Over a basemap the grid becomes a place the user recognises, which is the
    whole point of simulating spatially.
    """
    import folium
    import matplotlib

    if not path:
        return PLACEHOLDER
    try:
        vals, lats, lons, sown, n_yr, attrs = _grid_of(path, _window_index(window_label))
    except Exception as exc:  # noqa: BLE001
        return f"<div class='placeholder'><div>Could not read this result: {exc}</div></div>"

    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return ("<div class='placeholder'><div>This window has no simulated pixels."
                "</div></div>")
    # Zero is a crop failure, not the low end of a yield range: it gets its
    # own grey so it cannot be mistaken for missing data, and the colour scale
    # stretches over the cells that actually produced something.
    harvested = finite[finite > 0]
    has_zero = bool((finite == 0).any())
    rng = harvested if harvested.size else finite
    vmin, vmax = float(rng.min()), float(rng.max())
    if vmax - vmin < 1e-6:                      # a flat field still needs a range
        vmin, vmax = vmin - 1, vmax + 1

    # Cell edges halfway between neighbouring centres. A capped run keeps only
    # a lattice of sampled pixels, spaced 5 or 6 native cells apart, so the
    # spacing is uneven; sizing every cell from the first gap left thin empty
    # stripes wherever the gap was wider.
    def _edges(c: np.ndarray, default: float = 0.05) -> np.ndarray:
        c = np.asarray(c, dtype=float)
        if c.size < 2:
            return np.array([c[0] - default / 2, c[0] + default / 2])
        mid = (c[1:] + c[:-1]) / 2
        return np.concatenate([[c[0] - (mid[0] - c[0])], mid,
                               [c[-1] + (c[-1] - mid[-1])]])

    ey, ex = _edges(lats), _edges(lons)
    bounds = [[float(min(ey[0], ey[-1])), float(min(ex[0], ex[-1]))],
              [float(max(ey[0], ey[-1])), float(max(ex[0], ex[-1]))]]

    # The page accent from light to dark, light = low. It starts above white,
    # which would vanish against the basemap.
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("accent", ACCENT_RAMP)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    # A raster, not a polygon per pixel: one image layer reads as a continuous
    # surface, has no outlines to draw a grid over the data, and stays light
    # at a thousand pixels where a GeoJSON layer gets heavy. The image is the
    # grid resampled by nearest cell onto screen-sized pixels, so every cell
    # keeps its exact value and edge; nothing is smoothed or interpolated.
    lat_lo, lon_lo = bounds[0]
    lat_hi, lon_hi = bounds[1]
    span_x, span_y = lon_hi - lon_lo, lat_hi - lat_lo
    # Rows are spaced evenly in Web Mercator, the map's own projection, so each
    # image row lands where Leaflet draws it. folium's mercator_project does
    # this by interpolating between rows, which blurs cell edges; picking the
    # nearest cell per row keeps them sharp.
    def _merc(lat):
        return np.degrees(np.log(np.tan(np.pi / 4 + np.radians(lat) / 2)))

    def _unmerc(m):
        return np.degrees(2 * np.arctan(np.exp(np.radians(m))) - np.pi / 2)

    m_lo, m_hi = _merc(lat_lo), _merc(lat_hi)
    long_side = 1000
    if span_x >= (m_hi - m_lo):
        W, H = long_side, max(1, int(round(long_side * (m_hi - m_lo) / span_x)))
    else:
        W, H = max(1, int(round(long_side * span_x / (m_hi - m_lo)))), long_side
    px_lon = lon_lo + (np.arange(W) + 0.5) * span_x / W
    px_lat = _unmerc(m_hi - (np.arange(H) + 0.5) * (m_hi - m_lo) / H)   # top row first

    def _cell(edges: np.ndarray, q: np.ndarray) -> np.ndarray:
        asc = edges[-1] > edges[0]
        e = edges if asc else edges[::-1]
        k = np.clip(np.searchsorted(e, q, side="right") - 1, 0, len(e) - 2)
        return k if asc else (len(e) - 2 - k)

    img = vals[np.ix_(_cell(ey, px_lat), _cell(ex, px_lon))]
    rgba = cmap(norm(np.where(np.isfinite(img), img, vmin)))
    rgba[img == 0] = ZERO_RGBA
    rgba[..., 3] = np.where(np.isfinite(img), 0.85, 0.0)
    # Hand folium finished 8-bit colours: given floats it rescales each
    # channel by that channel's maximum, which shifts every colour.
    rgba = np.round(rgba * 255).astype(np.uint8)

    # Values for the hover readout, cell by cell in file order.
    grid = {"ey": [round(float(v), 5) for v in ey],
            "ex": [round(float(v), 5) for v in ex],
            "lat": [round(float(v), 4) for v in lats],
            "lon": [round(float(v), 4) for v in lons],
            "v": [[None if not np.isfinite(v) else int(round(v)) for v in row]
                  for row in vals]}

    # Zoom chosen from the extent rather than fit_bounds: inside a not-yet-sized
    # iframe Leaflet fits to a zero-size container and lands two levels too far out.
    # Fit the grid into roughly 600 x (map height - legend) pixels, in
    # quarter-zoom steps so a department and a district both fill the frame.
    fit_w, fit_h = 600.0, MAP_H * 0.75
    z_x = np.log2(fit_w * 360 / (256 * max(span_x, 0.02)))
    z_y = np.log2(fit_h * 360 / (256 * max(span_y, 0.02)))
    zoom = float(np.clip(np.floor(4 * min(z_x, z_y)) / 4, 5, 13))
    fig = folium.Figure(width="100%", height=MAP_H)
    m = folium.Map(location=[(bounds[0][0] + bounds[1][0]) / 2,
                             (bounds[0][1] + bounds[1][1]) / 2],
                   zoom_start=zoom, tiles="OpenStreetMap", control_scale=True,
                   zoom_control="topleft", zoom_snap=0.25, zoom_delta=0.5,
                   # The wheel scrolls the page; zoom with the +/- buttons.
                   scrollWheelZoom=False)
    fig.add_child(m)
    # Leaflet scales the image with the browser's smoothing; keep cells crisp.
    fig.header.add_child(folium.Element(
        "<style>.leaflet-image-layer{image-rendering:pixelated;"
        "image-rendering:crisp-edges}</style>"))
    folium.raster_layers.ImageOverlay(
        image=rgba, bounds=bounds, mercator_project=False, opacity=1.0,
        name="Simulated yield", interactive=False, zindex=2,
    ).add_to(m)
    _YieldHover(grid).add_to(m)

    crop = attrs.get("crop") or ""
    stops = ", ".join(matplotlib.colors.to_hex(cmap(i / 8)) for i in range(9))
    zero_hex = matplotlib.colors.to_hex(ZERO_RGBA[:3])
    _Legend(
        f"<div class='t'>{YIELD_AXIS}</div>"
        f"<div class='s'>{(crop + ' · ') if crop else ''}"
        f"{('sown ' + sown) if sown else 'simulated'}</div>"
        f"<div class='bar' style='background:linear-gradient(90deg,{stops})'></div>"
        f"<div class='ticks'><span>{vmin:,.0f}</span><span>{(vmin + vmax) / 2:,.0f}</span>"
        f"<span>{vmax:,.0f}</span></div>"
        + (f"<div class='z'><i style='background:{zero_hex}'></i>no harvest (0)</div>"
           if has_zero else "")
    ).add_to(m)

    # _repr_html_ on the Figure, not render(): it wraps the map in an
    # <iframe srcdoc=...>. gr.HTML inserts content with innerHTML, and browsers
    # do not execute <script> tags inserted that way. Inside an iframe it runs.
    return fig._repr_html_()


def _blank_fig(text: str):
    """An empty chart with one message. Plotly annotations do not wrap, so the
    text is broken into short lines here or a phone clips it at both edges."""
    import textwrap
    text = "<br>".join(textwrap.wrap(text, 38))
    fig = go.Figure()
    fig.add_annotation(text=text, showarrow=False, font={"color": INK_MUTED, "size": 13},
                       xref="paper", yref="paper", x=0.5, y=0.5)
    fig.update_layout(**_layout(xaxis={"visible": False}, yaxis={"visible": False}))
    return fig


def _layout(**extra):
    """Journal-style axes: white ground, thin light gridlines on y only, axis
    lines left and bottom, the page's own font, and no fixed width so the
    figure follows its container down to a phone."""
    axis = {"showline": True, "linecolor": AXIS, "linewidth": 1, "ticks": "outside",
            "tickcolor": AXIS, "ticklen": 4, "zeroline": False,
            "title": {"font": {"size": 12, "color": INK}},
            "tickfont": {"size": 11, "color": INK_MUTED}}
    base = {
        "template": "none", "autosize": True,
        "paper_bgcolor": "#ffffff", "plot_bgcolor": "#ffffff",
        "font": {"family": FONT, "size": 12, "color": INK},
        "margin": {"l": 64, "r": 12, "t": 56, "b": 52},
        "height": MAP_H,
        "hoverlabel": {"bgcolor": INK, "font": {"color": "#ffffff", "size": 12,
                                                 "family": FONT},
                       "bordercolor": INK},
        "legend": {"orientation": "h", "y": 1.06, "x": 0, "font": {"size": 11}},
        "xaxis": {**axis, "showgrid": False},
        "yaxis": {**axis, "showgrid": True, "gridcolor": GRID, "gridwidth": 0.5,
                  "title": {"text": YIELD_AXIS, "font": {"size": 12, "color": INK}},
                  "tickformat": ",.0f", "rangemode": "tozero"},
    }
    base.update(extra)
    return base


def _title(main: str, sub: str = "") -> dict:
    text = f"<b>{main}</b>" + (f"<br><span style='font-size:11px;color:{INK_MUTED}'>"
                               f"{sub}</span>" if sub else "")
    return {"text": text, "font": {"size": 13, "color": INK, "family": FONT},
            "x": 0, "xanchor": "left", "y": 0.97, "yanchor": "top"}


def _sowing_key(rec: dict) -> tuple:
    """Runs that differ only in sowing date share this key."""
    return (rec.get("crop"), rec.get("cultivar"), rec.get("region") or rec.get("country"),
            rec.get("n_kg_ha"), tuple(rec.get("seasons") or []))


def _sowing_series(sess: Session | None, path: str | None) -> dict | None:
    """The sowing windows to compare for the selected run.

    Normally that is the run's own planting windows. The agent is told to put
    a sowing-date comparison into one run, but a model sometimes makes one run
    per date instead, which left the chart saying "one planting window". When
    the selected run has a single window and the session holds other
    single-window runs that differ from it only in sowing date, those runs are
    lined up as the windows — the same assembly the nitrogen chart does across
    rates. Each file's numbers are read as they are; nothing is recomputed.
    """
    if not path:
        return None
    stats = _window_stats(path)
    own = {"windows": stats["windows"], "n_seasons": stats.get("n_seasons", 1),
           "merged": 0}
    if len(stats["windows"]) != 1 or sess is None:
        return own
    runs = sess.runs()
    rec = next((r for r in runs if r["file"] == path), None)
    if rec is None or (rec.get("n_windows") or 1) > 1:
        return own
    key = _sowing_key(rec)
    by_date: dict[str, dict] = {}
    for r in runs:                               # newest first, so first wins
        if (r.get("n_windows") or 1) == 1 and _sowing_key(r) == key and r.get("planting_date"):
            by_date.setdefault(str(r["planting_date"])[:10], r)
    if len(by_date) < 2:
        return own
    windows = []
    for d in sorted(by_date):
        try:
            w = _window_stats(by_date[d]["file"])["windows"]
        except Exception:  # noqa: BLE001
            continue
        if w:
            windows.append({**w[0], "window": len(windows) + 1,
                            "sown": w[0].get("sown") or d})
    if len(windows) < 2:
        return own
    return {"windows": windows, "n_seasons": own["n_seasons"], "merged": len(windows)}


def windows_chart(path: str | None, sess: Session | None = None):
    """Mean yield per planting window, with the pixel range behind each bar."""
    if not path:
        return _blank_fig("Run a simulation to compare planting windows.")
    try:
        series = _sowing_series(sess, path)
    except Exception as exc:  # noqa: BLE001
        return _blank_fig(f"Could not read this result: {exc}")
    w = series["windows"]
    if not w:
        return _blank_fig("This run produced no simulated pixels.")
    if len(w) == 1:
        return _blank_fig("One planting window in this run — ask for several "
                          "to compare sowing dates.")
    labels = [(f"{x['sown'][5:]}" if x.get("sown") else f"window {x['window']}") for x in w]
    means = [x["mean_kg_ha"] for x in w]
    best = max(range(len(w)), key=lambda i: means[i])
    # Whiskers: worst and best season of the area mean when there are several
    # seasons, else the 10th-90th percentile of pixels. Never the pixel
    # min/max, which a single crop-failure cell drags to zero.
    by_season = all(len(x.get("season_means_kg_ha") or []) > 1 for x in w)
    if by_season:
        lo = [min(x["season_means_kg_ha"]) for x in w]
        hi = [max(x["season_means_kg_ha"]) for x in w]
        spread = "seasons %{customdata[0]:,.0f}–%{customdata[1]:,.0f}"
        whisker = "whiskers: worst and best season"
    else:
        lo = [x.get("p10_kg_ha", x["min_kg_ha"]) for x in w]
        hi = [x.get("p90_kg_ha", x["max_kg_ha"]) for x in w]
        spread = "pixels p10–p90 %{customdata[0]:,.0f}–%{customdata[1]:,.0f}"
        whisker = "whiskers: 10th–90th percentile of pixels"
    fig = go.Figure()
    fig.add_bar(
        x=labels, y=means, name="mean yield",
        marker={"color": [ACCENT if i == best else ACCENT_SOFT for i in range(len(w))],
                "line": {"width": 0}},
        error_y={"type": "data", "symmetric": False,
                 "array": [max(0.0, h - m) for h, m in zip(hi, means)],
                 "arrayminus": [max(0.0, m - l) for l, m in zip(lo, means)],
                 "color": INK_MUTED, "thickness": 1, "width": 4},
        customdata=[[l, h, x["pixels"], x["min_kg_ha"], x["max_kg_ha"],
                     x.get("failed_pixel_seasons", 0)] for l, h, x in zip(lo, hi, w)],
        hovertemplate=("sown %{x}<br>mean %{y:,.0f} kg ha⁻¹"
                       f"<br>{spread}"
                       "<br>pixels %{customdata[3]:,.0f}–%{customdata[4]:,.0f}"
                       " · %{customdata[5]} with no harvest"
                       "<br>%{customdata[2]} pixels<extra></extra>"),
        text=[f"{m:,.0f}" if i == best else "" for i, m in enumerate(means)],
        textposition="outside", cliponaxis=False,
        textfont={"color": INK, "size": 12, "family": FONT},
    )
    n_seasons = series.get("n_seasons", 1)
    notes = [whisker, "darkest bar: best sowing date"]
    if series.get("merged"):
        notes.append(f"from {series['merged']} separate runs")
    # One note per line: Plotly titles do not wrap, and on one line the notes
    # ran off the right edge of a phone.
    base = _layout()
    fig.update_layout(**_layout(
        title=_title("Yield by sowing date"
                     + (f" · mean of {n_seasons} seasons" if n_seasons > 1 else ""),
                     "<br>".join(notes)),
        margin={**base["margin"], "t": base["margin"]["t"] + 14 * (len(notes) - 1)},
        showlegend=False, bargap=0.35))
    fig.update_xaxes(title={"text": "Sowing date (month-day)"}, type="category")
    return fig


def nitrogen_chart(sess: Session):
    """Yield response to nitrogen, from the session's runs that differ only in N.

    Each rate is its own run and its own file, so the curve is assembled from
    the manifest: the runs are grouped by crop and place, the newest run per
    rate wins, and a group needs at least two rates to be a response at all.
    """
    runs = sess.runs()
    groups: dict[tuple, dict[float, dict]] = {}
    for r in runs:                               # newest first, so first wins
        n = r.get("n_kg_ha")
        if n is None:
            continue
        key = (r.get("crop"), r.get("region") or r.get("country"),
               r.get("planting_date"), r.get("n_windows"))
        groups.setdefault(key, {}).setdefault(float(n), r)
    best_key = max(groups, key=lambda k: len(groups[k]), default=None)
    if best_key is None or len(groups[best_key]) < 2:
        return _blank_fig("Ask for the same crop and place at two or more nitrogen "
                          "rates — e.g. “maize in Yoro at 0, 60 and 120 kg N”.")
    rates = sorted(groups[best_key])
    per_window: dict[int, list] = {}
    for n in rates:
        try:
            stats = _window_stats(groups[best_key][n]["file"])
        except Exception:  # noqa: BLE001
            continue
        for w in stats["windows"]:
            per_window.setdefault(w["window"], []).append((n, w))
    if not per_window:
        return _blank_fig("The nitrogen runs produced no simulated pixels.")

    fig = go.Figure()
    many = len(per_window) > 1
    for k, (win, pts) in enumerate(sorted(per_window.items())):
        xs = [p[0] for p in pts]
        ys = [p[1]["mean_kg_ha"] for p in pts]
        sown = pts[0][1].get("sown")
        name = (f"sown {sown[5:]}" if sown else f"window {win}") if many else "mean yield"
        colour = SERIES[k % len(SERIES)]
        if not many:
            # One series: show the pixel range as a band behind the line.
            lo = [p[1]["min_kg_ha"] for p in pts]
            hi = [p[1]["max_kg_ha"] for p in pts]
            fig.add_scatter(x=xs + xs[::-1], y=hi + lo[::-1], fill="toself",
                            fillcolor="rgba(46,125,79,.12)", line={"width": 0},
                            hoverinfo="skip", showlegend=False, name="pixel range")
        fig.add_scatter(
            x=xs, y=ys, mode="lines+markers+text", name=name,
            line={"color": colour, "width": 2},
            marker={"size": 9, "color": colour, "line": {"color": "#ffffff", "width": 2}},
            text=([f"{y:,.0f}" if i == len(ys) - 1 else "" for i, y in enumerate(ys)]
                  if not many else None),
            textposition="top center", textfont={"size": 11, "color": INK, "family": FONT},
            hovertemplate=f"{name}<br>%{{x}} kg N ha⁻¹ → %{{y:,.0f}} kg ha⁻¹<extra></extra>",
        )
    crop, place = best_key[0] or "Crop", best_key[1] or ""
    fig.update_layout(**_layout(
        title=_title(f"{crop} response to nitrogen" + (f" · {place}" if place else ""),
                     "" if many else "band: pixel minimum to maximum"),
        showlegend=many))
    fig.update_xaxes(title={"text": "Nitrogen applied (kg N ha⁻¹)"}, tickvals=rates,
                     type="category")
    fig.update_yaxes(rangemode="tozero")
    return fig


def _csv_of(path: str) -> pathlib.Path | None:
    """The result file's yields as a flat table, one row per pixel, sowing date
    and season, for anyone who wants them in a spreadsheet rather than NetCDF.
    The values are the file's own; nothing is recomputed."""
    import xarray as xr
    src = pathlib.Path(path)
    out = src.with_suffix(".csv")
    if out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
        return out
    with xr.open_dataset(src) as ds:
        da = ds["HWAM"]
        ydim = "y" if "y" in da.dims else "lat"
        xdim = "x" if "x" in da.dims else "lon"
        # The same seasons as every other figure: a phantom season would put
        # rows of zeros in the table and halve its mean.
        keep, _, _ = _real_seasons(da, ds.attrs)
        if "year" in da.dims and keep:
            da = da.isel(year=keep)
        df = da.to_dataframe(name="yield_kg_ha").reset_index()
        dates = (np.atleast_1d(ds.planting_date.values) if "planting_date" in ds.coords
                 else None)
        # Window labels are coordinate values, not positions; dates go by position.
        pos = ({v: k for k, v in enumerate(np.atleast_1d(ds["planting_window"].values))}
               if "planting_window" in ds.coords else None)
    df = df.rename(columns={ydim: "lat", xdim: "lon"})
    df = df[np.isfinite(df["yield_kg_ha"])]
    cols = ["lat", "lon"]
    if "planting_window" in df.columns:
        if dates is not None and len(dates) == df["planting_window"].nunique():
            df["planting_date"] = [
                str(dates[pos.get(i, int(i)) if pos else int(i)])[:10]
                for i in df["planting_window"]]
            cols.append("planting_date")
        else:
            cols.append("planting_window")
    elif dates is not None and len(dates) == 1:
        df["planting_date"] = str(dates[0])[:10]
        cols.append("planting_date")
    for extra in ("year", "time"):
        if extra in df.columns:
            cols.append(extra)
    df["pixel"] = df.groupby(["lat", "lon"], sort=True).ngroup() + 1
    df = df[["pixel", *cols, "yield_kg_ha"]].sort_values(["pixel", *cols[2:]])
    tmp = out.with_suffix(".csv.part")
    df.to_csv(tmp, index=False, float_format="%.6g")
    tmp.replace(out)
    return out


def _artifacts(sess: Session) -> list[str]:
    """Anything worth downloading from this session, newest first."""
    for r in sess.runs():
        try:
            _csv_of(r["file"])
        except Exception as exc:  # noqa: BLE001
            log.warning("no CSV for %s: %s", r.get("file"), exc)
    out = []
    for p in sorted(sess.dir.rglob("*"), key=lambda q: q.stat().st_mtime, reverse=True):
        if p.is_file() and p.suffix in (".nc", ".png", ".csv", ".yaml", ".zip") \
                and "dssat" not in p.parts:
            out.append(str(p))
    return out[:16]


def _choices(sess: Session) -> list[tuple[str, str]]:
    return [(r.get("label") or pathlib.Path(r["file"]).name, r["file"]) for r in sess.runs()]


def _panel(sess: Session, path: str | None, window_label: str | None = None):
    """Everything the result panel shows, for one run and window."""
    choices = _choices(sess)
    if path is None and choices:
        path = choices[0][1]
    opts = _windows_in(path)
    if window_label not in opts:
        window_label = opts[0] if opts else None
    try:
        series = _sowing_series(sess, path)
    except Exception:  # noqa: BLE001
        series = None
    files = _artifacts(sess)
    # The pickers have nothing to pick until there is a result.
    return (gr.update(choices=choices, value=path, visible=bool(choices)),
            gr.update(choices=opts, value=window_label, visible=bool(choices)),
            tiles_html(path, window_label, series),
            yield_map(path, window_label),
            windows_chart(path, sess),
            nitrogen_chart(sess),
            gr.update(value=files, visible=bool(files)))


N_OUTPUTS = 7


def status_html(sess: Session) -> str:
    """The result panel's one-line status while the agent works; empty when idle."""
    if not sess.status:
        return ""
    import html as _html
    return (f"<div class='status' role='status' aria-live='polite'>"
            f"<span class='spin' aria-hidden='true'></span>"
            f"<span>Running… {_html.escape(sess.status)}</span></div>")


# Both pickers listen to `.select` and read the pick from the event. With
# `.input`, gradio 6.30 sent each pick twice with the previous value and
# dropped every second one, so the map lagged a pick behind or ignored it.
def _picked(evt: gr.SelectData, fallback):
    v = getattr(evt, "value", None)
    return v if v not in (None, "") else fallback


def on_file_change(path, evt: gr.SelectData, request: gr.Request):
    """A different run was picked: redraw everything for its first window."""
    sess = session_of(request)
    picked = _picked(evt, path)
    # The event may carry the label rather than the file; map it back.
    files = dict(_choices(sess))
    path = files.get(picked, picked) or None
    log.info("session %s: run %s", sess.id, pathlib.Path(str(path)).name if path else None)
    # The run picker keeps its own value; the sowing date picker gets the
    # new run's dates.
    return (gr.skip(), *_panel(sess, path)[1:])


def on_window_change(path, window_label, evt: gr.SelectData, request: gr.Request):
    """A different sowing date: only the tiles and the map depend on it.

    The charts and the download list do not change with the date, and
    rebuilding them doubled the wait, so everything else is left as it is."""
    sess = session_of(request)
    path = path or None
    window_label = _picked(evt, window_label)
    log.info("session %s: sowing date %s", sess.id, window_label)
    if window_label not in _windows_in(path):
        return _panel(sess, path, window_label)
    try:
        series = _sowing_series(sess, path)
    except Exception:  # noqa: BLE001
        series = None
    return (gr.skip(), gr.skip(),
            tiles_html(path, window_label, series),
            yield_map(path, window_label),
            gr.skip(), gr.skip(), gr.skip())


def refresh_all(request: gr.Request):
    """Jump to the newest run and draw it."""
    sess = session_of(request)
    sess.seen_sig = _signature(sess)
    log.info("session %s: panel drawn (%d runs)", sess.id, len(sess.runs()))
    return _panel(sess, None)


def _signature(sess: Session) -> str:
    runs = sess.runs()
    return f"{len(runs)}:{runs[0]['file']}:{runs[0].get('created')}" if runs else ""


def auto_refresh(request: gr.Request):
    """Redraw only when a new run has appeared since the panel last drew."""
    sess = session_of(request)
    st = status_html(sess)
    st_upd = gr.skip() if st == sess.seen_status else st
    sess.seen_status = st
    sig = _signature(sess)
    if sig == sess.seen_sig:
        return (*(gr.skip() for _ in range(N_OUTPUTS)), st_upd)
    sess.seen_sig = sig
    log.info("session %s: new result; refreshing the panel", sess.id)
    return (*_panel(sess, None), st_upd)


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
PREBAKED_LIST = " · ".join(f"{t.title()} ({v.get('season', '?')})"
                           for t, v in sorted(prebaked_regions().items())) or "none"

GH = "https://github.com/CGIAR-Climate-Data-Hub"
AGWISE = "https://github.com/CGIAR-AgWise"
HF_DATA = "https://huggingface.co/datasets/aaguilar90"
SKILL_URL = f"https://cgiar-climate-data-hub.github.io/skills/skills/{SKILL}/"

HEADER = f"""
<header class='cdh-header'>
  <div class='eyebrow'><a href='{AGWISE}' target='_blank' rel='noopener'>AgWise</a><span class='dot' aria-hidden='true'>·</span>CGIAR Climate Data Hub</div>
  <h1>Crop Yield Explorer</h1>
  <p class='lede'>Ask for a crop, a place and a season in plain language. An agent
  fetches climate and soil data, builds datacubes and runs the DSSAT crop model on
  every pixel.</p>
  <nav class='links' aria-label='Project links'>
    <a href='{GH}' target='_blank' rel='noopener'>Climate Data Hub on GitHub</a>
    <a href='{GH}/aggeodata' target='_blank' rel='noopener'>aggeodata</a>
    <a href='{GH}/ag-cube-cm' target='_blank' rel='noopener'>ag-cube-cm</a>
    <a href='{SKILL_URL}' target='_blank' rel='noopener'>{SKILL} skill</a>
    <span class='sep' aria-hidden='true'>|</span>
    <span class='grp'>Datasets:</span>
    <a href='{HF_DATA}/chirps-v3-daily-rnl' target='_blank' rel='noopener'>CHIRPS v3</a>
    <a href='{HF_DATA}/chirts-era5-daily' target='_blank' rel='noopener'>CHIRTS-ERA5</a>
    <a href='{HF_DATA}/soilgrids-v2-1km' target='_blank' rel='noopener'>SoilGrids v2</a>
  </nav>
</header>
"""

# Hidden until there is a citation to give. Set CITATION to the text (HTML is
# fine) and the "How to cite" block appears in the footer.
CITATION = ""
CITE_BLOCK = (f"<h3>How to cite</h3><p>{CITATION} Please also cite the data "
              f"sources listed here.</p>") if CITATION else ""

BYOK_FOOTER = ("<p class='fine'>Your API key, if you add one, stays in your browser "
               "session and is not stored.</p>") if BYOK_ENABLED else ""

FOOTER = f"""
<footer class='cdh-footer'>
  <div>
    <h3>Data sources</h3>
    <ul>
      <li><b>Precipitation</b> — CHIRPS v3 daily
        (<a href='{HF_DATA}/chirps-v3-daily-rnl' target='_blank' rel='noopener'>Zarr on the Hub</a>)</li>
      <li><b>Temperature</b> — CHIRTS-ERA5 daily
        (<a href='{HF_DATA}/chirts-era5-daily' target='_blank' rel='noopener'>Zarr on the Hub</a>)</li>
      <li><b>Solar radiation</b> — AgERA5 (0.1°) with a CDS key, otherwise NASA POWER (0.5°).
        This Space uses {"AgERA5" if HAS_CDS else "NASA POWER"}.</li>
      <li><b>Soil</b> — SoilGrids v2, 1 km
        (<a href='{HF_DATA}/soilgrids-v2-1km' target='_blank' rel='noopener'>Zarr on the Hub</a>)</li>
      <li><b>Crop model</b> — DSSAT, run per pixel by
        <a href='{GH}/ag-cube-cm' target='_blank' rel='noopener'>ag-cube-cm</a>; data
        acquisition by <a href='{GH}/aggeodata' target='_blank' rel='noopener'>aggeodata</a></li>
    </ul>
  </div>
  <div>
    {CITE_BLOCK}
    <h3>About the numbers</h3>
    <p class='fine'>Simulated, rainfed yields unless nitrogen is requested. Soil is a
    global model, not a survey. Treat values as relative across the map.</p>
    {BYOK_FOOTER}
  </div>
</footer>
"""

ABOUT = f"""
The agent follows the CGIAR Climate Data Hub
[`{SKILL}`]({SKILL_URL}) skill —
the same file you would install into your own coding agent with
`npx skills add https://github.com/{REPO} --skill {SKILL}`. Nothing about crop
modelling is written into this web page; it is all in the skill and the packages
[aggeodata]({GH}/aggeodata) and [ag-cube-cm]({GH}/ag-cube-cm).

**Reading the numbers.** Water and nitrogen stress are both simulated and nothing
is applied unless you ask for it, so the default map is rainfed, unfertilised
yield — not potential yield and not what a farm harvests. Soil comes from
SoilGrids, a global model rather than a survey, and one season is not a
recommendation. Treat the numbers as relative across the map.

**Solar radiation** comes from {"AgERA5 (0.1°)" if HAS_CDS else "NASA POWER (0.5°; set `CDSAPI_KEY` for AgERA5 at 0.1°)"}.
"""

CHAT_INTRO = (
    (f"Pre-built and instant: {PREBAKED_LIST}. Anywhere else downloads live and "
     f"takes a few minutes. " if CUBES_READY else
     "The pre-built example data could not be loaded, so the examples are not "
     "available right now; any place you ask about downloads live and takes a few "
     "minutes. ") +
    f"Results are kept for this session only — download "
    f"them before you close the tab.{NL}{NL}"
    "**Ask a question about sowing dates, nitrogen or expected yield for a place "
    "and season. Try an example:**")

# Decision questions, each answerable from a pre-built cube with no download
# (Yoro 2012, Nandom 2020). The sowing-date one leaves the start to the agent,
# whose plan is checked against the cube: bean takes 90 days, so four dates
# two weeks apart must start by 21 Aug 2012 to mature by 31 Dec.
EXAMPLES = [
    "When is the best time to sow beans in Yoro, Honduras? Compare four dates two "
    "weeks apart in the 2012 season",
    "How much nitrogen should I apply to maize in Yoro, Honduras, to increase yield? "
    "Compare 0, 60 and 120 kg N/ha for 2012",
    "What maize yield can I expect in Nandom, Ghana? Use the 2020 season",
]

CHAT_H = MAP_H + 190      # chat column ends level with the result panel

CSS = """
/* A working surface, not an article: two columns that fit a laptop, stacked on
   a phone. White ground, neutral greys and one accent (ACCENT in the code). */
:root {
    --accent: #2e7d4f; --accent-700: #215a39; --accent-800: #1a472e;
    --accent-50: #eef6f1; --accent-200: #afd6bc;
    --ink: #1f2937; --ink-muted: #4b5563; --line: #e5e7eb;
}
body, gradio-app, .gradio-container, .main, .wrap { background: #ffffff !important; }
/* Gradio 6 no longer centres a capped container, so centre it explicitly. */
body .gradio-container {
    max-width: 1440px !important; margin: 0 auto !important; padding: 0 24px !important;
}
/* Gradio's inner wrapper adds its own 32px side padding; the container's is enough. */
body .gradio-container .main.fillable { padding-left: 0 !important; padding-right: 0 !important; }
/* Gradio styles bare links in HTML blocks as padded blocks; keep ours inline. */
.cdh-header a, .cdh-footer a { padding: 0 !important; display: inline !important; }
footer:not(.cdh-footer) { display: none !important; }
a { color: var(--accent-700); }

/* --- header ------------------------------------------------------------- */
.cdh-header { padding: 20px 0 14px; border-bottom: 1px solid var(--line); margin-bottom: 4px; }
.cdh-header .eyebrow {
    font-size: 12px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
    color: var(--accent);
}
.cdh-header .eyebrow a { color: inherit; text-decoration: none; }
.cdh-header .eyebrow a:hover { text-decoration: underline; }
.cdh-header .eyebrow .dot { margin: 0 .5em; color: var(--ink-muted); }
.cdh-header h1 {
    font-size: 26px; font-weight: 700; letter-spacing: -.015em; line-height: 1.2;
    color: #111827; margin: 2px 0 6px;
}
.cdh-header .lede { font-size: 14.5px; line-height: 1.55; color: var(--ink-muted);
                    max-width: 78ch; margin: 0 0 10px; }
.cdh-header .links { display: flex; flex-wrap: wrap; align-items: baseline;
                     gap: 4px 16px; font-size: 13px; }
.cdh-header .links a { color: var(--accent-700); text-decoration: none;
                       border-bottom: 1px solid var(--accent-200); }
.cdh-header .links a:hover { border-bottom-color: var(--accent-700); }
.cdh-header .links .grp { color: var(--ink-muted); }
.cdh-header .links .sep { color: #d1d5db; }

/* --- chat ---------------------------------------------------------------- */
#chat-col .chatbot, #chat-col .chatbot * { font-size: 13.5px; line-height: 1.55; }
#chat-col .chatbot p { margin: .3rem 0 !important; }
#chat-col .chatbot h1, #chat-col .chatbot h2, #chat-col .chatbot h3 {
    font-size: 14px !important; font-weight: 650 !important;
    margin: .55rem 0 .25rem !important;
}
#chat-col .chatbot ul, #chat-col .chatbot ol {
    margin: .25rem 0 .35rem 1.05rem !important; padding-left: .2rem !important;
}
#chat-col .chatbot li { margin: .12rem 0 !important; }
#chat-col .chatbot code, #chat-col .chatbot pre { font-size: 12px !important; }
/* Plan tables: long values wrap inside their cell; the label column keeps its
   words whole, so it can never collapse to one letter per line. */
#chat-col .chatbot table { font-size: 12px !important; width: 100%; table-layout: auto; }
#chat-col .chatbot td, #chat-col .chatbot th {
    padding: .18rem .4rem !important; font-variant-numeric: tabular-nums;
    overflow-wrap: anywhere; vertical-align: top;
}
#chat-col .chatbot td:first-child, #chat-col .chatbot th:first-child {
    white-space: nowrap; overflow-wrap: normal;
}
/* A file name in a code span cannot break by default; in a plan table on a
   phone it widened the chat, and with it the whole page, past the screen. */
#chat-col .chatbot td code, #chat-col .chatbot li code, #chat-col .chatbot p code {
    white-space: normal !important; overflow-wrap: anywhere; word-break: break-all;
}
#chat-col .message, #chat-col .message-row, #chat-col .bubble-wrap { min-width: 0; max-width: 100%; }
#chat-col .chatbot pre { overflow-x: auto; }
/* "running tool…" lines are status, not prose: quieter and smaller. */
#chat-col .chatbot em { color: var(--ink-muted); font-size: 12.5px; }
#chat-col .chatbot details { margin: .1rem 0 .3rem; }
#chat-col .chatbot details summary {
    cursor: pointer; color: var(--ink-muted); font-size: 12px; display: inline;
}

/* Empty state: a short intro and the examples as chips. */
#chat-col .placeholder-content {
    display: flex !important; flex-direction: column; justify-content: center;
    gap: 14px; height: 100%; padding: 20px 18px;
}
/* The intro's markdown span carries Gradio's `chatbot` class; a height meant
   for the message list once made it 680px tall and pushed it out of view. */
#chat-col .placeholder-content .chatbot { height: auto !important; max-height: none !important; }
#chat-col .placeholder-content .placeholder {
    color: var(--ink) !important; max-width: 56ch; margin: 0 auto; text-align: left;
    /* Gradio gives the intro flex-grow and height:100%, so it fills the chat and
       pushes the example chips to the bottom; keep the pair together, centred. */
    flex: 0 0 auto !important; height: auto !important;
}
#chat-col .placeholder-content .placeholder p { text-align: left !important;
                                                color: var(--ink-muted) !important; }
#chat-col .placeholder-content .placeholder strong { color: #111827; font-size: 15px; }
#chat-col .examples {
    display: flex !important; flex-wrap: wrap; gap: 8px; justify-content: center;
    max-width: 60ch; margin: 0 auto;
}
#chat-col button.example {
    flex: 0 1 auto; width: auto !important; min-height: 0 !important;
    /* Rounded rectangles, not pills: the examples are full questions and run
       to three lines on a phone, where a pill's curve cut into the text. */
    padding: 8px 14px !important; border-radius: 12px !important;
    border: 1px solid var(--accent-200) !important; background: #fff !important;
    box-shadow: none !important; text-align: left;
}
#chat-col button.example:hover, #chat-col button.example:focus-visible {
    background: var(--accent-50) !important; border-color: var(--accent) !important;
}
#chat-col button.example .example-text {
    font-size: 13px !important; color: var(--accent-800) !important;
    white-space: normal; -webkit-line-clamp: unset; display: inline;
}

/* --- result panel -------------------------------------------------------- */
#map-col .panel-title {
    font-size: 12px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
    color: var(--ink-muted); margin: 6px 0 2px;
}
#status { min-height: 0 !important; padding: 0 !important; }
/* An empty status block still takes a line plus the column gap; drop it. */
#status:not(:has(.status)) { display: none !important; }
#status .status {
    display: flex; align-items: center; gap: 9px; padding: 8px 12px;
    border: 1px solid var(--accent-200); background: var(--accent-50);
    border-radius: 6px; font-size: 13px; color: var(--accent-800);
    font-variant-numeric: tabular-nums;
}
#status .spin {
    flex: none; width: 12px; height: 12px; border-radius: 50%;
    border: 2px solid var(--accent-200); border-top-color: var(--accent);
    animation: cdh-spin .8s linear infinite;
}
@keyframes cdh-spin { to { transform: rotate(360deg); } }
@media (prefers-reduced-motion: reduce) { #status .spin { animation: none; } }

#pickers { gap: .6rem !important; }
/* Before the first result the pickers are hidden and the tiles are empty, but
   each block still took a column gap: a blank band under "Results". */
#pickers:not(:has(input)), #tiles:not(:has(.tiles)) { display: none !important; }
#pickers input { padding-right: 28px !important; text-overflow: ellipsis; }

.tiles {
    display: grid; grid-template-columns: 1.4fr 1fr 1fr; gap: .6rem;
    margin: .2rem 0 .4rem;
}
.tile {
    padding: .7rem .85rem; border: 1px solid var(--line);
    border-radius: 8px; background: #fff; min-width: 0;
}
.tile.hero { background: var(--accent-50); border-color: var(--accent-200); }
.tile-v {
    font-size: 22px; font-weight: 700; letter-spacing: -.01em; line-height: 1.15;
    color: var(--ink); font-variant-numeric: tabular-nums;
}
.tile.hero .tile-v { font-size: 30px; color: var(--accent-800); }
.tile-v .unit { font-size: 12px; font-weight: 600; margin-left: .3rem;
                color: var(--ink-muted); white-space: nowrap; }
.tile-l { font-size: 12px; color: var(--ink-muted); margin-top: .2rem; line-height: 1.35; }

#map-col .map-frame iframe {
    width: 100% !important; height: var(--map-h) !important;
    border: 1px solid var(--line); border-radius: 8px;
}
#map-col .placeholder {
    display: flex; align-items: center; justify-content: center;
    height: var(--map-h); border: 1px dashed #d1d5db;
    border-radius: 8px; color: var(--ink-muted);
    font-size: 13px; line-height: 1.6; text-align: center; padding: 1.5rem;
}
#map-col .placeholder b { color: var(--ink); }
#map-col .tabs > .tab-nav button, #map-col .tab-wrapper button { font-size: 13px; }
#map-col .tabitem { padding: .5rem 0 0 !important; border: none !important; }
#map-col .js-plotly-plot .plotly .modebar { opacity: .35; }
#downloads { border-color: var(--line) !important; }
/* Gradio's "×" per file only drops it from this list in the browser (the next
   refresh brings it back) and reads as "delete"; the list is download-only. */
#downloads .label-clear-button, #downloads button[aria-label="Clear"] {
    display: none !important;
}
/* Whole file names: on a phone Gradio cut them to "yield_bean_y… .csv". They
   wrap only at "_" and "." (break points added by FORCE_LIGHT's script), never
   mid-word. */
#downloads .filename, #downloads .stem {
    white-space: normal !important; overflow: visible !important;
    text-overflow: clip !important; overflow-wrap: normal; word-break: normal;
}
/* "3,254–3,964" stays on one line; the dash is not a break point. */
.tile-v .nw { white-space: nowrap; }

/* --- footer -------------------------------------------------------------- */
.cdh-footer {
    display: grid; grid-template-columns: 3fr 2fr; gap: 28px;
    border-top: 1px solid var(--line); margin-top: 20px; padding: 18px 0 28px;
    font-size: 12.5px; line-height: 1.6; color: var(--ink-muted);
}
.cdh-footer h3 {
    font-size: 12px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
    color: var(--ink); margin: 0 0 6px;
}
.cdh-footer ul { margin: 0; padding-left: 1.1rem; }
.cdh-footer li { margin: .1rem 0; }
.cdh-footer b { color: var(--ink); font-weight: 600; }
.cdh-footer p { margin: 0 0 .5rem; }
.cdh-footer .fine { font-size: 12px; }

/* --- phones -------------------------------------------------------------- */
@media (max-width: 920px) {
    body .gradio-container { padding: 0 12px !important; }
    .cdh-header h1 { font-size: 21px; }
    .cdh-header .lede { font-size: 13.5px; }
    .cdh-header .links .sep { display: none; }
    .tiles { grid-template-columns: 1fr 1fr; }
    .tile.hero { grid-column: 1 / -1; }
    .tile.hero .tile-v { font-size: 26px; }
    /* Run labels are long ("Bean · Yoro · 2012 · 0 kg N · 4 windows"): one
       picker per line so neither value is cut off. */
    #pickers { flex-direction: column !important; }
    #pickers > * { min-width: 100% !important; }
    #pickers input { font-size: 13px !important; }
    #chat-col button.example { width: 100% !important; flex-basis: 100%; }
    .tile-v { font-size: 19px; }
    :root { --map-h: 55vh !important; }
    .cdh-footer { grid-template-columns: 1fr; gap: 14px; }
}
"""
# Appended rather than interpolated: the block above is full of CSS braces,
# which an f-string would read as placeholders.
# Heights follow the window rather than a fixed pixel count: on a tall screen
# the panels grow with it, on a laptop they still fit above the fold. The chart
# figures keep MAP_H as their own height, which is why the floor matches it.
CSS += f"""
:root {{ --map-h: clamp({MAP_H}px, calc(100vh - 470px), 760px); }}
#chat-col .bubble-wrap {{
    height: calc(var(--map-h) + 250px) !important; max-height: none !important;
}}
/* On a phone the chat sizes to its messages, up to 60% of the screen, instead
   of a fixed box that left a large blank area under a short conversation. The
   empty state keeps enough room for the intro and the examples. */
@media (max-width: 920px) {{
    #chat-col .bubble-wrap {{
        height: auto !important; min-height: 0 !important; max-height: 60vh !important;
    }}
    #chat-col .bubble-wrap:has(.placeholder-content) {{ min-height: 300px !important; }}
}}
"""

# Own-key panel: a notice above the chat, in the accent's soft tint, and the
# same controls in a small accordion below it. Stacks on a phone.
CSS += """
#byok-panel {
    border: 1px solid var(--accent-200) !important; background: var(--accent-50) !important;
    border-radius: 8px !important; padding: 12px 14px !important; margin-bottom: 8px;
    gap: 8px !important;
}
#byok-panel > *, #byok-panel .block { background: transparent !important; border: 0 !important; }
.byok-note { margin: 0; font-size: 14px; line-height: 1.45; color: var(--ink); }
.byok-links { margin: 0; font-size: 13px; color: var(--ink-muted); }
.byok-row { align-items: flex-end !important; gap: 8px !important; flex-wrap: wrap !important; }
.byok-row .byok-prov { flex: 0 0 auto !important; min-width: 0 !important; }
.byok-row .byok-key { flex: 1 1 200px !important; min-width: 180px !important; }
.byok-row .byok-btn { flex: 0 0 auto !important; min-width: 120px !important; }
.byok-row .byok-msg { flex: 1 1 100% !important; font-size: 13px; min-height: 0 !important; }
.byok-row .byok-msg:empty { display: none; }
#byok-settings { margin-top: 6px; }
#byok-settings .label-wrap span { font-size: 13px; color: var(--ink-muted); }
@media (max-width: 920px) {
    .byok-row .byok-key, .byok-row .byok-btn { flex: 1 1 100% !important; min-width: 0 !important; }
}
"""

# The accent as a full Gradio colour scale, so buttons, focus rings and the
# selected tab all use the same green as the charts and the map.
_ACCENT_HUE = gr.themes.Color(
    name="cdh", c50="#eef6f1", c100="#d6ebdd", c200="#afd6bc", c300="#82bd97",
    c400="#559f71", c500="#2e7d4f", c600="#286c45", c700="#215a39", c800="#1a472e",
    c900="#133522", c950="#0c2316")

THEME = gr.themes.Base(
    primary_hue=_ACCENT_HUE, secondary_hue=_ACCENT_HUE, neutral_hue="gray",
    radius_size="sm",
    font=[gr.themes.GoogleFont("Inter"), "Helvetica", "Arial", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "monospace"],
).set(
    body_background_fill="#ffffff",
    body_text_color="#1f2937",
    body_text_color_subdued="#4b5563",
    background_fill_primary="#ffffff",
    background_fill_secondary="#f9fafb",
    block_background_fill="#ffffff",
    block_border_color="#e5e7eb",
    block_shadow="none",
    # Field labels ("Run", "Sowing date") as small grey text, not green pills:
    # the result tiles are what the eye should land on.
    block_label_background_fill="transparent",
    block_label_text_color="#4b5563",
    block_label_border_width="0px",
    block_title_background_fill="transparent",
    block_title_text_color="#4b5563",
    block_title_text_size="*text_sm",
    block_title_text_weight="500",
    block_title_padding="0px",
    block_title_radius="0px",
    input_background_fill="#ffffff",
    input_border_color="#d1d5db",
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
    button_primary_text_color="#ffffff",
    color_accent="*primary_600",
    color_accent_soft="*primary_50",
    link_text_color="*primary_700",
    link_text_color_visited="*primary_700",
)

# The page is designed for a white ground. Gradio follows the visitor's OS dark
# mode by default, which would put white figures on a dark page.
# It also gives download file names break points after "_" and "." so a phone
# wraps "yield_bean_yoro_2012" at an underscore rather than mid-word. Only the
# shown text changes (a zero-width space); the link still points at the file.
FORCE_LIGHT = r"""() => {
  document.body.classList.remove('dark');
  const ZW = '​';
  // A once-a-second pass, not a MutationObserver: rewriting the text inside a
  // DOM observer fed Gradio's own updates back into it and froze the page
  // when the results arrived.
  const soften = () => document.querySelectorAll('#downloads .stem')
    .forEach(el => {
      const t = el.textContent;
      if (t && !t.includes(ZW) && /[_.]/.test(t)) el.textContent = t.replace(/([_.])/g, '$1' + ZW);
    });
  setInterval(soften, 1000);
}"""

# gradio 6 moved `theme`, `css` and `js` from Blocks() to launch(); gradio 5
# wants them on Blocks. Decide once so the same file is warning-free on both.
_BLOCKS_PARAMS = inspect.signature(gr.Blocks.__init__).parameters
_blocks_kwargs = {}
if "theme" in _BLOCKS_PARAMS:
    _blocks_kwargs["theme"] = THEME
if "css" in _BLOCKS_PARAMS:
    _blocks_kwargs["css"] = CSS
if "js" in _BLOCKS_PARAMS:
    _blocks_kwargs["js"] = FORCE_LIGHT


def _chat_kwargs() -> dict:
    """ChatInterface arguments that depend on the gradio major version."""
    kw = {}
    ci = inspect.signature(gr.ChatInterface.__init__).parameters
    if "type" in ci:
        kw["type"] = "messages"
    if "chatbot" in ci:
        bot = {"height": CHAT_H, "show_label": False, "placeholder": CHAT_INTRO}
        if "type" in inspect.signature(gr.Chatbot.__init__).parameters:
            bot["type"] = "messages"
        kw["chatbot"] = gr.Chatbot(**bot)
    if "textbox" in ci:
        # A real label, not just a placeholder, so screen readers name the field.
        kw["textbox"] = gr.Textbox(
            label="Your request", show_label=True, lines=1, max_lines=6,
            placeholder="e.g. Maize yield in Nandom, Ghana for the 2020 season",
            submit_btn=True, scale=7, autofocus=False)
    return kw


KEY_CHOICES = ["Gemini", "OpenRouter"]
KEY_HELP = (
    "<p class='byok-links'>Get a free key: "
    f"<a href='{KEY_LINKS['gemini']}' target='_blank' rel='noopener'>Google AI Studio</a> (Gemini) · "
    f"<a href='{KEY_LINKS['openrouter']}' target='_blank' rel='noopener'>OpenRouter</a></p>")


def _active_text(sess: Session, own) -> str:
    if _own_key(own):
        return f"Using: {_provider_label(OWN, own)}"
    return f"Using: {sess.provider}" if sess.provider else "Using: the shared AI model"


def _key_controls(button: str):
    """Provider, a masked key box, a button and the result line."""
    prov = gr.Radio(KEY_CHOICES, value="Gemini", label="Provider", elem_classes="byok-prov")
    box = gr.Textbox(label="API key", type="password", placeholder="Paste your key",
                     max_lines=1, elem_classes="byok-key")
    btn = gr.Button(button, variant="primary", size="sm", elem_classes="byok-btn")
    msg = gr.Markdown("", elem_classes="byok-msg")
    return prov, box, btn, msg


async def use_key(provider, key, own, request: gr.Request):
    """Test the key with one cheap call; keep it in this tab's state if it works.
    The key box is cleared either way once a key is accepted."""
    sess = session_of(request)
    prov = "openrouter" if str(provider).lower().startswith("openrouter") else "gemini"
    ok, msg = await _test_key(prov, key)
    if not ok:
        return own, gr.skip(), f"✗ {msg}", gr.skip(), gr.skip()
    new = {"provider": prov, "key": (key or "").strip()}
    sess.seen_keyui = None
    if sess.paused is not None:
        sess.paused["resuming"] = True
    # Hidden at once: a paused run resumes in the chat right after this.
    return (new, gr.update(value=""), f"✓ {msg}", gr.update(visible=False),
            gr.update(label=f"Use your own API key · {_active_text(sess, new)}"))


def forget_key(request: gr.Request):
    sess = session_of(request)
    sess.provider, sess.seen_keyui = "", None
    return None, "Your key is removed from this session."


def _busy(sess: Session) -> bool:
    return sess.turn_lock is not None and sess.turn_lock.locked()


def key_ui(own, request: gr.Request):
    """Show the quota panel while a run is paused; keep the accordion's
    'Using: …' line current; keep the chat input off while a turn runs and on
    once it ends (a chip-started turn has no end event of its own that could
    switch it back on). Skipped when nothing changed."""
    sess = session_of(request)
    # The finished-turn count catches a turn that began and ended between ticks.
    state = (sess.paused is not None and not sess.paused.get("resuming"),
             _active_text(sess, own), _busy(sess), len(sess.exchanges))
    if state == sess.seen_keyui:
        return gr.skip(), gr.skip(), gr.skip()
    old = sess.seen_keyui or (None, None, None, None)
    sess.seen_keyui = state
    box = (gr.update(interactive=not state[2]) if state[2:] != old[2:] else gr.skip())
    return (gr.update(visible=state[0]),
            gr.update(label=f"Use your own API key · {state[1]}"), box)


def input_ui(own, request: gr.Request):
    """With BYOK off, the poll only switches the chat input on and off."""
    return key_ui(own, request)[2]


with gr.Blocks(title="Crop Yield Explorer · CGIAR Climate Data Hub",
               **_blocks_kwargs) as demo:
    gr.HTML(HEADER)

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=340, elem_id="chat-col"):
            # This tab's own API key, if the visitor adds one. Server-side
            # session state only: never written anywhere, never shown back.
            key_state = gr.State(None)
            # Shown when every shared model is out of quota and a run paused.
            # Not built at all with BYOK off, so there is nothing to call.
            if BYOK_ENABLED:
                with gr.Group(visible=False, elem_id="byok-panel") as byok_panel:
                    gr.HTML(f"<p class='byok-note'><b>{QUOTA_NOTICE.split('. ', 1)[0]}.</b> "
                            f"{QUOTA_NOTICE.split('. ', 1)[1]}</p>")
                    with gr.Row(elem_classes="byok-row"):
                        p_prov, p_box, p_btn, p_msg = _key_controls("Use this key")
                    gr.HTML(KEY_HELP)
            # show_progress="hidden": Gradio's overlay covers the whole chat in
            # white while a turn runs; the chat's own progress lines and the
            # status line in the result panel already say what is happening.
            chat = gr.ChatInterface(
                respond, **_chat_kwargs(), additional_inputs=[key_state],
                examples=[[e] for e in EXAMPLES], cache_examples=False, show_progress="hidden",
            )
            if BYOK_ENABLED:
                with gr.Accordion("Use your own API key · Using: the shared AI model",
                                  open=False, elem_id="byok-settings") as byok_acc:
                    gr.Markdown("Add a Gemini or OpenRouter key to use it instead of the "
                                "shared quota. It is used only in this browser session and "
                                "never stored.")
                    with gr.Row(elem_classes="byok-row"):
                        s_prov, s_box, s_btn, s_msg = _key_controls("Use this key")
                    gr.HTML(KEY_HELP)
                    forget = gr.Button("Forget my key", size="sm", variant="secondary")

        with gr.Column(scale=1, min_width=340, elem_id="map-col"):
            gr.HTML("<h2 class='panel-title'>Results</h2>")
            status = gr.HTML("", elem_id="status")
            with gr.Row(elem_id="pickers"):
                # allow_custom_value: the choices are pushed per session, and
                # gradio validates a selection against the component's initial
                # (empty) list otherwise — every pick would raise.
                run_dd = gr.Dropdown(label="Run", choices=[], value=None, visible=False,
                                     interactive=True, scale=3, allow_custom_value=True)
                window_dd = gr.Dropdown(label="Sowing date", choices=[], value=None,
                                        visible=False, interactive=True, scale=2,
                                        allow_custom_value=True)
            tiles = gr.HTML("", elem_id="tiles")
            with gr.Tabs():
                with gr.Tab("Map"):
                    map_html = gr.HTML(PLACEHOLDER, elem_classes="map-frame")
                with gr.Tab("Sowing dates"):
                    win_plot = gr.Plot(windows_chart(None), show_label=False)
                with gr.Tab("Nitrogen"):
                    n_plot = gr.Plot(_blank_fig("Run two or more nitrogen rates to "
                                                "see the response."), show_label=False)
            # Shown as soon as the session has files; the poll below fills it, so
            # there is nothing to expand and nothing to refresh by hand.
            files = gr.Files(label="Downloads — results, configs and cubes of this session",
                             interactive=False, visible=False, elem_id="downloads")
            with gr.Accordion("How this works", open=False):
                gr.Markdown(ABOUT)

    gr.HTML(FOOTER)

    OUTPUTS = [run_dd, window_dd, tiles, map_html, win_plot, n_plot, files]
    # Redraws take well under a second; Gradio's "processing" overlay on every
    # output for that long flickers the whole panel, so it is hidden.
    # always_last: a pick made while the previous one is still drawing is
    # kept and drawn next, rather than dropped (the default, "once").
    run_dd.select(on_file_change, run_dd, OUTPUTS, show_progress="hidden",
                  trigger_mode="always_last")
    window_dd.select(on_window_change, [run_dd, window_dd], OUTPUTS, show_progress="hidden",
                     trigger_mode="always_last")
    # Poll for finished runs and for the agent's progress. gr.skip() when
    # nothing changed means the browser is not sent a new map every few
    # seconds, so a selection the user is reading stays put until there is
    # genuinely something newer to show.
    gr.Timer(POLL_SECONDS).tick(auto_refresh, None, OUTPUTS + [status],
                                show_progress="hidden")
    # A fresh tab starts from a clean panel, whatever this process drew before.
    demo.load(refresh_all, None, OUTPUTS, show_progress="hidden")

    # Own key: test it, keep it in this tab's state, then resume a paused run
    # in the chat and hand the chat's history back to the ChatInterface.
    for _prov, _box, _btn, _msg in (((p_prov, p_box, p_btn, p_msg),
                                     (s_prov, s_box, s_btn, s_msg)) if BYOK_ENABLED else ()):
        _go = dict(fn=use_key, inputs=[_prov, _box, key_state],
                   outputs=[key_state, _box, _msg, byok_panel, byok_acc],
                   show_progress="minimal")
        for _ev in (_btn.click, _box.submit):
            _ev(**_go).then(
                resume_run, [chat.chatbot, key_state], [chat.chatbot, chat.textbox],
                show_progress="hidden",
            ).then(lambda h: h, chat.chatbot, chat.chatbot_state, show_progress="hidden")
    if BYOK_ENABLED:
        forget.click(forget_key, None, [key_state, s_msg], show_progress="hidden")
    # A typed message disables the input until the reply ends; a chip did not,
    # so "yes" could be sent before the plan existed. Same rule for both.
    # The poll below switches it back on when the turn has ended.
    chat.chatbot.example_select(lambda: gr.update(interactive=False), None, chat.textbox,
                                queue=False, show_progress="hidden")
    if BYOK_ENABLED:
        gr.Timer(POLL_SECONDS).tick(key_ui, key_state, [byok_panel, byok_acc, chat.textbox],
                                    show_progress="hidden")
    else:
        gr.Timer(POLL_SECONDS).tick(input_ui, key_state, chat.textbox,
                                    show_progress="hidden")


if __name__ == "__main__":
    launch_kwargs = {}
    if "theme" not in _BLOCKS_PARAMS:
        launch_kwargs["theme"] = THEME
    if "css" not in _BLOCKS_PARAMS:
        launch_kwargs["css"] = CSS
    if "js" not in _BLOCKS_PARAMS and "js" in inspect.signature(gr.Blocks.launch).parameters:
        launch_kwargs["js"] = FORCE_LIGHT
    # share=True downloads and runs `frpc`, a reverse-proxy tunnel client. Corporate
    # endpoint security blocks it on sight. Leave CDH_SHARE unset and use the local
    # URL; set CDH_HOST=0.0.0.0 to reach it from another machine on the network.
    if os.environ.get("CDH_SHARE"):
        launch_kwargs["share"] = True
    if os.environ.get("CDH_HOST"):
        launch_kwargs["server_name"] = os.environ["CDH_HOST"]
    port = int(os.environ.get("CDH_PORT", os.environ.get("PORT", "7860")))
    launch_kwargs["server_port"] = port
    log.info("starting on http://127.0.0.1:%d  (ctrl-c to stop)", port)
    demo.queue(max_size=6).launch(**launch_kwargs)
