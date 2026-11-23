# Crop Yield Explorer — frontend changelog (option A: Gradio redesign)

Not deployed. Nothing has been pushed to the Hugging Face Space.

## Changed files

| File | What changed |
|---|---|
| `app_agent.py` | All bug fixes and UI changes below |
| `README.md` | `python_version: "3.12"`; datasets `aaguilar90/soilgrids-v2-1km` and `aaguilar90/cdh-demo-cubes` added to the header; "Pre-built cubes" section rewritten for the dataset; `CDH_CUBES_REVISION` row; `CDH_MODEL` default corrected to 3.5 |
| `requirements.txt` | Every package pinned to the version tested in the `hfspace` env; the two CDH packages pinned to commits |
| `.gitignore` | `cubes/` added |
| `scripts/deploy_space.py` | `cubes/*` excluded from the upload |
| `scripts/prebake_cubes.py` | Docstring: upload to the dataset instead of committing |
| `CHANGELOG.md` | This file (new) |

No Playwright in `hfspace` or `requirements.txt` (screenshots use the base env).

### Pinned commits

- **ag-cube-cm**: `2a34217dd005b67b5f8a46be1c6009acce14d470`
  (`ag-cube-cm[mcp] @ git+https://github.com/CGIAR-Climate-Data-Hub/ag-cube-cm.git@2a34217dd005b67b5f8a46be1c6009acce14d470`)
- **aggeodata**: `2e7cdb70075da08ee16772e77ef744560ed1d9be`
  (`aggeodata[download,mcp] @ git+https://github.com/CGIAR-Climate-Data-Hub/aggeodata.git@2e7cdb70075da08ee16772e77ef744560ed1d9be`)

Other pins: gradio 6.30.0, matplotlib 3.11.2, folium 0.20.0, branca 0.8.2,
plotly 7.1.0, numpy 2.5.4, pandas 2.3.3, xarray 2026.9.0, python-dotenv 1.2.4,
mcp 1.30.0, google-genai 2.31.0. `pip check` clean.

## Bug fixes

00. **The same request gave different results** (Chipata 2018/19 maize: 861 and
    1,396 kg ha⁻¹), because the model chose the area, the weather window and the
    sowing date. These are now fixed in code by `acquire_region`, which every
    request calls (instantly for pre-built places):
    - **Area**: always the boundary of the place named, clipped (never the
      bounding box). The place is resolved by `find_place` (below), so the same
      words resolve to the same unit; a name at both levels ("Yoro") is the
      district unless the request says department, province, region or state.
    - **Weather window**: from the sowing dates plus the crop's days to
      maturity (ag-cube-cm's table), with 30 days before the first sowing and 30
      after the last maturity. With a default sowing date the window covers the
      latest date the onset rule can return. The model's dates are no longer
      accepted; downloads are cached by their exact window.
    - **Default sowing date** (user gave none): onset of rains on CHIRPS
      rainfall averaged over the area, Sivakumar (1988): the first day with
      ≥ 20 mm over 3 days and no dry spell longer than 7 days in the next 30.
      Search from 1 May in the Sudano-Sahelian zone (Sivakumar's anchor),
      otherwise from 1 January of the season year, or 1 July of the first year
      for a two-year season ("2018/19"); a dry day is < 1 mm. The anchors outside
      West Africa and the dry-day threshold are this app's choices, not
      Sivakumar's, and say so. The plan writes "Sowing: <date> — default sowing
      date (rule: …)". A sowing date the user gives is used as given.
    - `generate_config` is forced onto those values (cubes, boundary, sowing
      dates); called before `acquire_region` it is refused.
000. **Seasons are left out for missing weather only.** Each run records, per
    season, whether the weather covers sowing to maturity day by day (all of
    rainfall, both temperatures and radiation; at most 5 missing days, the gap
    ag-cube-cm will interpolate). A season with incomplete weather is left out
    and named; a season with complete weather and every pixel at 0 is kept and
    reported as a **crop failure** in the tiles ("crop failure: every pixel
    yielded 0") and the reply. (Previously any all-zero season was left out.)
0000. **Place lookup**: `find_place` returns the best five matches with
    country, province and district, instead of `list_admin_units`' whole
    country (~1,100 names for Colombia, which hit Gemini's per-minute token
    limit). A second lookup of the same place returns the first answer with
    "already found; do not look up again"; `list_admin_units` is hidden from the
    agent. Each turn also tells the model which places are already found (the
    chat history holds text, not tool answers, so after a "yes" it looked the
    place up again); a run now makes one lookup.
00000. **The page could freeze when results arrived** (found in the live
    tests): the script that adds break points to download file names watched
    the page with a MutationObserver, and rewriting the names inside it fed
    Gradio's own updates back in. It now runs as a once-a-second pass that
    changes nothing once a name is done; a page given a finished result stays
    responsive.

0. **CRITICAL — fresh downloads reported about half the real yield.** The weather
   for a season is downloaded from December of the year before to January of the
   year after, and ag-cube-cm kept every season whose sowing + 200 days fell in a
   calendar year present in the weather, so a one-season request ran three
   seasons: the year before (only December's weather) and the year after (only
   January's) yielded 0 on every pixel, and the tiles, map, chat and chart
   averaged them in ("mean of 3 seasons"). Palmira (COL) maize 2019 showed
   974 kg ha⁻¹ instead of 1,949; Musanze (RWA) beans 2021 1,467 / 1,464 / 1,059
   instead of 2,935 / 2,928 / 2,118.
   - Seasons: only those whose every sowing date plus the crop's days to
     maturity (ag-cube-cm's table) lies inside the weather are simulated; a
     one-season request simulates one season. The job budget counts those
     seasons, not every calendar year the weather touches.
   - Safety net: a season in which every pixel is 0 or missing is left out of
     every figure (tiles, map, chart, CSV, the chat's results, the summary handed
     to the agent) and named as left out.
   - Counts: a pixel is counted once, as in the CSV; "pixels_ok/failed" job
     counts (which counted phantom seasons) are no longer reported.
   - Pixels with no yield in the requested season are counted ("14 gave no
     yield" on the range tile, and in the chat), greyed out on the map, and
     included in the mean as the real outcome they are.
   - Verified on the user's own Palmira and Musanze files: tile, map, chat and
     CSV give 1,949 and 2,935 / 2,928 / 2,118, with 48 and 23 pixels (14 and 2
     with no yield).
   - The chat gets whole kg ha⁻¹ rounded as the tiles round them (a reply once
     said 2,981 for a tile's 2,982), and a season that runs into the next year
     is named "2018/19" on the tile and in the reply.

1. **Empty sowing-date chart.** The agent sometimes ran one simulation per
   sowing date, so the chart said "one planting window". (a) Prompt rule: a
   sowing-date comparison is one run with `n_planting_windows` /
   `planting_window_days`. (b) Fallback: when the selected run has one window and
   the session holds runs that differ only in sowing date, the chart and the
   "best sowing" tile line them up. The files' numbers are read as they are.
2. **Pre-built cubes re-downloaded (Nandom 2020).** Enforced in code: when
   `acquire_region` asks for a shipped cube's place (name, country, admin level)
   and dates inside the cube's time axis, it returns the shipped cubes with no
   download. Anything outside downloads as before. The plan shows a "Data:"
   line and the progress line reads "Using pre-built data cubes". The plan is
   written before `acquire_region` runs, and once said "no pre-built cube for
   Nandom, fresh download" while the code then used the cubes; the place
   lookup (`list_admin_units`) now carries a note naming the shipped cubes and
   their dates for that country, so the plan's Data line comes from a tool
   answer. Internal mode names (`with_cubes`, `full_pipeline`) are shown in
   plain words.
3. **Downloads started before any plan (Aipe, Huila).** DSSAT and real downloads
   are held until the user confirms a plan. The gate is session state, not chat
   text: `plan_shown` is set when a turn ends without DSSAT or a download having
   run (that reply was the plan or a revision); `approved` is set for one turn
   when the user's whole reply is a confirmation ("yes", "ok", "proceed", "go
   ahead", "sí"…). Anything else, including "yes but with 60 kg", a new request
   or a follow-up, stays held and the agent revises the plan. Pre-built cubes
   are picked without asking; DSSAT still waits for the yes. (An earlier,
   text-based version looked for "?" in the plan and blocked DSSAT after "yes"
   on a plan ending "Please reply yes if…"; replaced.)
4. **Absolute paths in the chat and plan table.** Paths are rewritten to
   "session workspace/…" (or "cubes/…") before display, and mapped back for the
   model. The prompt also forbids paths. Any other absolute path the model
   writes, with either slash and on Linux too (the Space runs on Linux), shows
   only its file name, or "session workspace" for a folder; URLs are untouched.
   (Found live: a plan offered "Working path: C:/dssat_work (please ensure this
   folder exists)". The prompt now also says the page manages the workspace and
   working paths stay out of the plan.)
5. **Plan table label column collapsed** to one letter per line; long values now
   wrap and the label column keeps its words whole.
6. **Chat intro invisible** (blank chat on load). A height rule meant for the
   message list also matched the intro; scoped to the message list.
7. **Chat went fully white while the agent ran** (Gradio's opaque progress
   overlay). Hidden on the chat and on the panel's refresh events; the chat's
   progress lines and the result status line remain.
8. **Map zoomed when scrolling the page.** Scroll-wheel zoom is off; +/- buttons stay.
9. **Sowing date and Run pickers lagged or ignored picks.** With Gradio 6.30's
   `.input` event each pick reached the server twice with the previous value,
   and every second pick was dropped, so the map showed the previous date or
   none (likely behind the "picked 26 Jun, shows 12 Jun" report). Both pickers
   now use `.select` with the value from the event, `trigger_mode="always_last"`
   (a pick made during a redraw is drawn next, not dropped), and a sowing-date
   pick redraws only the tiles and the map. Per-file statistics are cached by
   file and modification time. Verified: four date picks in a row and four run
   switches each drew the right date and run; a redraw takes 1.3–4.6 s
   (faster once warm).
10. **Phone page wider than the screen** when the chat held a long file name in a
    plan table: code spans in the chat now wrap. Fresh 390 px load after a run:
    page 390 px wide, chart 340 px in a 342 px column.
11. **Tile labels read as the picked date** ("12 Jun · best sowing" next to a
    26 Jun pick): the hero tile names the date shown, the third tile says
    "best of 4 sowing dates".
12. **CSV export dates** for files whose window numbers do not start at 0 (found
    by the offline checks before release).
13. **Invented results are never shown.** In a turn that ran no simulation, a
    reply is withheld if it quotes a yield not found in this session's result
    files (exact to the displayed rounding; kg ha⁻¹, kg/ha or t/ha), a
    percentage without reading results in that turn, or a .nc/.csv file the
    session does not have offered as a result ("download", "results", "ready"…).
    A plan naming the file it will write ("Output: yield_bean_yoro_2012.nc") is
    not a claim and passes; the first live version withheld such plans, and the
    fallback then lost "four dates two weeks apart", now read from the question. In its place the page shows a plan for what was asked,
    built from the request and the session's latest run (e.g. "Yoro · 2012 · Bean ·
    60 kg N ha⁻¹ · 4 dates from 2012-05-15 · Data: pre-built data cubes (instant).
    Reply yes to run it"); a yes then runs it. Prompt rule: never state a yield,
    a percentage change or a result file that did not come from a tool result in
    this session.
14. **Sowing dates cut off by the end of the cube.** Before DSSAT runs, every
    sowing date plus the crop's days to maturity must end inside the weather
    cube. Days to maturity come from ag-cube-cm's own table
    (`DSSATModel._MATURITY_DAYS`: bean 90, maize 120, …), the one it uses to drop
    seasons. The check is appended to `generate_config`'s answer, so the plan
    says whether the dates fit and proposes ones that do (latest first sowing)
    or a download that covers them; `run_simulation` refuses dates that do not
    fit ("NOT RUN"), which leaves the plan open for a yes; its progress line reads
    "✗ DSSAT not run — the sowing dates do not fit the weather data" (a run that
    failed for another reason reads "✗ DSSAT not run", not "✓"). Example: bean in Yoro
    2012 sown 1 Sep–13 Oct is refused (13 Oct needs weather to 11 Jan 2013); the
    latest start for four dates two weeks apart is 21 Aug.
15. **A turn could hang forever** when the Gemini call never answered (seen live:
    the final reply after a finished run never came). Gemini calls now time out
    after 120 s (`CDH_MODEL_TIMEOUT`), like the OpenRouter call (180 s), and a
    timeout is retried or handed to the next provider like a 503.
16. **Free model chain.** A 429 or quota error moves to the next model, each with
    its own quota: `gemini-3.5-flash-lite` → `gemini-3.1-flash-lite` → shared
    OpenRouter (`nvidia/nemotron-3.5-super-vl:free`). When all three are out, the
    chat says "The demo has reached today's limit, please try again later."
    (or, with BYOK on, opens the own-key panel). The IDs
    are exact matches from the Gemini API model list; no Gemma. A turn stays on
    one model unless that model fails. A model that runs out of quota is skipped
    for 30 minutes, per model. A switch keeps the conversation, the tool results,
    the plan gate and the session state. Gemini 3 function-call signatures are
    only handed back to the model that made them; after a switch, the documented
    `skip_thought_signature_validator` placeholder is sent instead. The log records
    which model answered each turn (`turn answered by …`) and each switch; the chat
    does not mention models. `CDH_MODEL` picks the first model and
    `CDH_GEMINI_FALLBACKS` the rest.
17. **A reply after a run quoted a yield that no file has** (found live on 3.5
    Flash Lite: "against 6,447 kg ha⁻¹ with no fertilizer" while the 0 N run gave
    6,470). The model had copied 6,447 from a real-number example in the prompt,
    and the invented-results check skipped every turn that ran DSSAT. Fixes: the
    prompt's example answers now use placeholders (`<mean> kg ha⁻¹`). Turns that
    ran DSSAT are checked too: a yield the result files do not back up (a
    difference of two backed yields counts as backed) is held back, and the model
    is asked once more with the reason. If it is still wrong, the reply is
    replaced by the figures from the files. The tolerance now follows how the
    number is written (whole kg ±1, "6,500" or "6.5 t" up to ±0.5 %); the flat
    0.5 % had let 6,447 pass for 6,470.
18. **"yes" brought back the plan instead of running it** (found live on the
    Nandom chip with 3.1, and behind the first failed nitrogen run on 3.5). "yes"
    was recognised; the model never saw its own plan. A plan turn that waited
    out 503 retries left the chat input enabled when started from a chip (a typed
    message disables it). A "yes" sent during that wait reached the server with
    the history from before the plan, while the plan turn was still running. The
    "yes" turn's first call held 144 characters of history instead of about
    1,100 (logged by the token counter), so the model wrote the plan again.
    Fixes:
    - Turns of one session run one at a time; a message that arrives early says
      "Waiting for the previous reply to finish…" and runs after it, so the plan
      gate reads the finished turn.
    - The server keeps a record of each finished exchange. A history that is a
      stale copy of it (missing the latest replies) is replaced by the record;
      a history that differs (an undo or retry) is kept.
    - A chip-started turn now disables the input like a typed one; the
      4-second poll switches it back on when the turn ends.
    - The server record is kept per session: one session's history can never
      fill another's (offline check, and live with two sessions at once).
    Offline check of the exact case: 15 checks.
19. **Yields without thousands separators** in the model's text ("8193 kg ha⁻¹",
    a table cell "2609", "max 3801"). Separators are now added in three places:
    - any figure of four or more digits followed by a yield unit ("8,193",
      "4,585–9,783");
    - a figure after min / max / mean ("max 3,801");
    - the cells of yield columns in a table (headers with kg, yield, mean, min
      or max).

    Years, dates, seasons, pixel counts, nitrogen rates and file names are left
    alone. The tiles and charts already used separators.
20. **A mean shown 1 kg too high** (found in the final smoke test: Yoro bean sown
    27 May, CSV mean 2,663.49, shown as 2,664 on the tile and given to the model
    as 2,664). The per-date statistics were rounded to 0.1 kg (2,663.5) and then
    again to whole kg (2,664). They now keep full precision and are rounded once,
    where they are shown. This predates this round and could affect any figure
    within 0.05 of a half.
21. **Narration before the answer** (seen on the OpenRouter fallback: "Now running
    the simulation:", "The config wasn't saved — let me fix that." above the
    result). Text a model writes between tool calls is shown while it works, but
    dropped from the final message, which keeps the progress lines and then the
    answer.

## Pre-built cubes moved to a Hub dataset

The Nandom 2020 and Yoro 2012 cubes are no longer part of the code repository.
They live in the dataset `aaguilar90/cdh-demo-cubes` (public; 164 files, same
`<place>_cube_<kind>.zarr/` layout as the old `cubes/` folder).

- At start-up the app downloads them with `huggingface_hub.snapshot_download`
  (`repo_type="dataset"`), pinned to commit
  `cfc905dd1d078a837d261a051815a9c1e303f234` (`CDH_CUBES_REVISION`), into
  `<work>/cdh_agent/cubes` (`CDH_CUBES_DIR` overrides it). A `.revision` marker
  is written only after a complete download, so a restart reuses the folder
  with no network call; a partial or older download is cleared first. The
  download's per-file HTTP logging is silenced (one log line instead of ~500).
- If the download fails, the app still starts. The intro says the pre-built
  examples are unavailable, clicking an example shows: "The pre-built data for
  the examples could not be loaded when the demo started, so this example is
  not available right now. You can still ask about any place and season: its
  data will be downloaded fresh, which takes a few minutes." No model call is
  made for it. Typed requests (Nandom 2020 included) go to a fresh download.
- `cubes/` is in `.gitignore` and excluded by `scripts/deploy_space.py`. Your
  local copies are untouched.
- `git rm --cached` was not needed: the cubes were never tracked. The Space
  folder is untracked inside the parent `scripts` repository, and
  `git log --all --stat -- cubes` (and a repo-wide search for `*_cube_*`) is
  empty. No Crop Yield Explorer Space exists on the Hub yet, so the cubes are
  not in any Space history either.

## Bring your own key: built, off by default, pending live tests

`BYOK_ENABLED` (environment variable, default off). With it off, the shipping
setting, the key panel, the settings accordion, their events and the footer
line are not built at all. A run that finds every shared model out of quota
ends with "The demo has reached today's limit, please try again later." With
`BYOK_ENABLED=1`, the feature works as described below. Its live tests are to
be run on a private duplicate Space before it is turned on.

- When every shared model is rate-limited or out of quota, the run **pauses**
  rather than failing. A notice panel opens above the chat (a hidden `gr.Group`,
  not a browser pop-up): "The shared AI quota for this demo is used up for today.
  You can continue with your own key. It is used only in this browser session
  and never stored." It offers a provider choice (Gemini / OpenRouter), a
  password-type box, "Use this key", and links to Google AI Studio and OpenRouter.
- "Use this key" makes one cheap test call (a few output tokens on the same
  model the app uses). It shows "Key works" or a plain error: key not accepted,
  key valid but its own quota used up, or provider not reachable. A working key
  resumes the paused run in the chat from the step where it stopped, using the
  tool results it already had, so nothing is downloaded again.
- A small accordion under the chat, "Use your own API key · Using: …", lets a
  visitor add a key before any limit is hit. It also offers "Forget my key", and
  its title shows the active provider ("Using: your Gemini key", "Using: shared
  Gemini").
- Order per session: the visitor's key → shared Gemini models → shared
  OpenRouter → ask for a key. A rejected own key falls back to the shared models
  (the chat says so). An OpenRouter key uses the same model as the shared
  fallback; the visitor cannot pick the model.
- Security:
  - The key is held only in that tab's `gr.State`. It is passed as an argument
    to the provider call and nowhere else: not to the session object, the
    model's messages, tool arguments, files, configs or the log.
  - The key box is masked and cleared once a key is accepted.
  - Key-shaped strings (and the exact key) are masked as "[key hidden]" in
    error text and "details" shown in the chat. A formatter on the log handlers
    does the same for every log line, tracebacks included.
  - A key pasted into the chat is not sent to the model, and is masked in the
    history sent on later turns.
- Footer: "Your API key, if you add one, stays in your browser session and is
  not stored."
- No test hook in the app: the live tests fake quota errors with a scratchpad
  launcher that wraps the provider calls.

## UI improvements

- **Look**: white ground, neutral greys, one accent (#2e7d4f) across theme,
  bars, best-bar highlight and map ramp. Field labels ("Run", "Sowing date") as
  small grey text; result tiles are the most prominent element.
- **Header**: title, one-line description, links (Climate Data Hub on GitHub,
  aggeodata, ag-cube-cm, the skill, the three Hub datasets).
- **Footer**: data sources (CHIRPS v3, CHIRTS-ERA5, AgERA5 / NASA POWER,
  SoilGrids v2, DSSAT via ag-cube-cm) and "About the numbers". "How to cite"
  is hidden until a `CITATION` value is set in `app_agent.py`.
- **Empty state**: intro plus example chips kept together in the middle of the
  chat (Gradio stretched the intro to full height, pushing the chips to the
  bottom); "Run" and "Sowing date" dropdowns and the empty tiles hidden until a
  result exists, so there is no blank band under "Results".
- **Progress**: "Running… step x" status line in the result panel. The chat
  shows plain labels ("Checking the location", "Running DSSAT", "Reading the
  results", "Using pre-built data cubes") instead of tool names; "with_cubes"
  and tool names the model writes are replaced with plain words.
- **Errors**: a friendly line first; technical detail in a collapsed "details".
- **Tiles**: the hero tile names the sowing date shown ("sown 26 Jun"). The third
  tile says "best of 4 sowing dates · 2,693 kg ha⁻¹", or for a single run shows
  its sowing date. The range tile has units.
- **Units**: kg ha⁻¹ and kg N ha⁻¹ in tiles, map hover, charts and prompt; any
  "kg/ha" in a reply is rewritten.
- **Charts**: journal style (white, thin y gridlines, axis titles with units,
  page font, autosize). Best bar in the accent, others in a light tint.
  Messages on empty charts wrap so a phone does not clip them.
- **Map**: legend bottom-right on a white card with title and units; accent ramp.
- **Downloads**: shown directly once files exist (no accordion, no refresh
  button). The "×" is hidden: it only dropped the file from the browser's list
  and came back on the next refresh. Each result also gets a CSV (pixel, lat,
  lon, planting_date, year, yield_kg_ha) with the same values as the .nc.
- **Mobile (390px)**: 12px gutters; pickers stacked full width with 13px text;
  chat sizes to its messages up to 60% of the screen; tile captions wrap; chart
  notes one per line; download file names wrap instead of being cut.
- **Run labels** say "4 dates" instead of "4 windows" (shorter, and matches the
  "Sowing date" picker).
- **Decision questions.** The start screen asks "Ask a question about sowing
  dates, nitrogen or expected yield for a place and season. Try an example:"
  with three chips, each answerable from a pre-built cube with no download:
  best sowing time for beans in Yoro 2012 (four dates two weeks apart), how much
  nitrogen for maize in Yoro 2012 (0, 60, 120 kg N/ha), expected maize yield in
  Nandom 2020. The prompt maps "best time to sow" to one run with several sowing
  windows, "how much nitrogen" to one run per rate (0, 60, 120 by default) and
  "what yield can I expect" to one run; missing years or dates are proposed in
  the plan. The final answer leads with the answer in one sentence ("In 2012,
  the 12 June sowing gave the highest mean yield: 2,693 kg ha⁻¹."), then the
  details, and keeps the one-season caveat. Plan + yes before DSSAT unchanged.
- **Phone wrapping**: download file names break only at "_" or "."; the range
  tile keeps "3,254–3,964" on one line.
- **Plain progress words**: "CHIRPS (HF) pr 2/3" reads "Downloading rainfall 2/3",
  "Stacking dates" reads "Building the climate cube" (likewise temperature,
  solar radiation and the soil cube).
- **Data line for a fresh download**: the place lookup tells the agent, for any
  place without pre-built cubes, to write "Data: fresh download (a few
  minutes)" in the plan.
- **Accessibility**: chat input has a real label ("Your request"); subdued text
  darkened to #4b5563.

## Verification

Local test instance on port 7861 (keys from the gitignored `.env`, own work
folder), Playwright from the base conda env.

- Offline checks, all passing: plan gate 35, pre-built match 17, UI 19, pre-built
  note 5, invented results + season check 33 (including the exact "now the same
  with 60 kg" reply), path scrubbing 11, model-call timeout 5.
- Live, final code:
  - Chip 1 (sowing dates, Yoro 2012): plan, yes, one run with 4 dates; answer
    first. A "now the same with 60 kg" follow-up got a revised plan; yes ran it;
    no reply invented numbers.
  - Chip 2 (nitrogen, Yoro 2012): one yes, three runs; 0 → 6,447,
    60 → 7,775, 120 → 8,103 kg ha⁻¹; answer first.
  - Chip 3 (Nandom 2020): "Data: pre-built data cubes (instant)", no download;
    "In 2020, maize in Nandom averaged 3,699 kg ha⁻¹".
  - Bean sowing from 1 September 2012: the first plan says those dates would be
    cut off and proposes 24 Jul–4 Sep (or a download); yes ran the dates that fit.
  - 390 px: no horizontal scroll; chips wrap inside the screen; range tile on
    one line; download names break at "_".
  - In every run: chat stayed visible (no overlay), no tool names, no kg/ha, no
    file-system paths.

### Fresh-download verification (yield halving fix)

Offline: 56 checks on the user's own Palmira and Musanze files (tile = map =
chat = CSV for the requested season and date; pixel and no-yield counts; season
selection for Palmira, Musanze, Chipata 2018/19, a single-year and a five-year
cube; a real Nandom run's summary), plus 5 rounding checks; all earlier suites
still pass (169 checks).

Live on 7861 (fresh downloads, final code):

| Run | Weather | Seasons run | Chat | Tile | CSV | Pixels (no yield) |
|---|---|---|---|---|---|---|
| Palmira maize 2019 | 2019-03-01 – 2019-12-31 | 2019 | 1,938 | 1,938 | 1,938.2 | 55 (16) |
| Palmira maize 2019, the reported window | 2018-12-01 – 2020-01-31 | 2019 | 1,939 | 1,939 | 1,938.6 | 55 (16) |
| Musanze beans 2021, 3 dates | 2021-01-01 – 2021-12-31 | 2021 | 2,981 / 2,971 / 2,143 | 2,982 / 2,972 / 2,143 | 2,981.5 / 2,971.5 / 2,143.1 | 24 (2) |
| Chipata maize 2018/19 | 2018-09-01 – 2019-07-31 | 2018/19 | 1,396 | 1,396 | 1,396.2 | 80 (0) |

The Musanze run predates the rounding fix (chat 2,981 vs tile 2,982); the
later runs agree to the unit. Figures differ slightly from the 7860 runs because
the agent chose a different weather window, sowing date or clipping each time.

### Reproducibility verification (area, window and sowing fixed in code)

Offline: 38 checks (place lookup for Palmira, Musanze, Chipata and Yoro as
district or department; the onset rule on synthetic rain; Nandom asked twice
gives the same sowing, cubes and area; generate_config forced onto them; two
real Nandom runs give the same mean and pixels; weather coverage recorded;
crop failure vs incomplete weather). All earlier suites pass (225 checks).

Live on 7861, each request made twice in fresh sessions (fresh download on the
first, the same cached window on the second):

| Request | Sowing (default rule) | Weather window | Run 1 | Run 2 | Pixels (no yield) |
|---|---|---|---|---|---|
| Palmira maize 2019 | 2019-02-05 | 2018-12-02 – 2019-11-30 | 2,897 | 2,897 | 48 (14) both |
| Musanze beans 2021, 3 dates | 2021-02-04, 02-18, 03-04 | 2020-12-02 – 2021-11-28 | 3,209 / 3,049 / 2,927 | 3,209 / 3,049 / 2,927 | 23 (2) both |
| Chipata maize 2018/19 | 2018-11-17 | 2018-06-01 – 2019-05-30 | 1,296 | 1,296 | 80 (0) both |

Chat, tile and CSV agree in every run (CSV 2,896.9; 3,209.2 / 3,048.8 / 2,927.0;
1,296.3). One season simulated each time. The reply's wording varies between
runs; the numbers do not. Results differ from earlier rounds because the
sowing date now comes from the onset rule rather than the model's guess (e.g.
Palmira 5 Feb instead of 1 Apr).

### Model chain, token use and own-key verification

Offline (all passing, 288 checks in total):
- Model chain (17 checks): order, no Gemma, 3.1 → 3.5 on a quota error in the
  middle of a turn with the tool run once and the gate kept, signatures,
  cooldown, OpenRouter, pause, and the log lines.
- Own key (39 checks): pause, panel, invalid and valid key, resume with no
  repeated tool, two sessions at once, a rejected own key, rate limits, and
  redaction in the chat, log and files.
- Unbacked yield after a run (7 checks).
- All earlier suites still pass.

Live on 7861 with `gemini-3.5-flash-lite` forced as the main model:

| Test | Result |
|---|---|
| Nandom chip | Pass: pre-built cubes, plan, no DSSAT before "yes". Answer first, "In 2020, maize in Nandom averaged 3,529 kg ha⁻¹"; the range 3,092–4,036 is the same in the chat and the tile. CSV 3,529, 15 pixels. |
| Yoro sowing-date chip | Pass: 4 dates, answer first, "the 10 June sowing … 2,806 kg ha⁻¹". CSV 2,609 / 2,663 / 2,806 / 2,757. |
| Plan gate with "yes" | Pass in every run: no DSSAT before "yes". |
| Invented-results check ("now the same with 60 kg") | Pass: revised plan, no numbers, no DSSAT. |
| Nitrogen chip, 1st try | **Fail**: the plan dropped the rates ("Fertilizer: none"), and only 0 N was run. Its numbers matched the CSV (6,470). |
| Nitrogen chip, 2nd try | **Fail**: three rates, but the first line said "against 6,447 kg ha⁻¹" (CSV 6,470). This led to fix 17. |
| Nitrogen chip, after fix 17, ×2 | Pass: 0 / 60 / 120 → 6,470 / 7,894 / 8,193, answer first, all equal to the CSV. Minor: the model wrote "8193" without the thousands comma. |
| Palmira fresh download | Pass: 2,897 kg ha⁻¹, 48 pixels, 14 with no yield, equal to the CSV and the earlier round. A real 503 on 3.5 in the middle of the turn moved it to 3.1, and the run finished. |

Live on 7861 with 3.1 first (the order at the time):
- Simulated quota error on 3.1 in the middle of a run: **not completed**. In
  that run "yes" brought the plan back (fix 18), so the quota flag never fell
  inside a run. Rerun after fix 18: not possible yet, because Google returned
  503 "high demand" on every 3.1 call (also on a direct one-call check).
- Real 503s on 3.1 switched sessions to 3.5 (and on to OpenRouter) with no
  loss.

Plan + "yes" after fix 18 (Nandom chip, fresh session each time):
- On 3.5: 3/3 pass. "yes" ran DSSAT once, the plan was not repeated, and the
  "yes" turn's history held the plan (966–1,031 characters). The answer came
  first ("In 2020, maize in Nandom averaged 3,529 kg ha⁻¹"), and the CSV gives
  3,529.
- On 3.1: not done. Every call fell through to 3.5 and OpenRouter on 503s. The
  gate held in all three runs and the numbers matched, but these do not count
  as 3.1 runs.

The own-key live tests were stopped twice part-way when the machine ran low on
memory; they move to the private duplicate Space (see Open items).

### Shipping configuration (BYOK off, 3.5 → 3.1 → OpenRouter)

Run from `app_agent.py` itself on 7861 (no test launcher, `BYOK_ENABLED` unset).

Offline: 337 checks pass in all. They include BYOK off and session isolation
(17), thousands separators and rounding (12), narration (3), the "yes" order
(15), and the model chain (17, including "try again later" when all three
models are out).

Session isolation, live: two browser sessions at once.
- Session A: the Nandom chip. Session B: typed "Beans in Yoro, Honduras, 2012
  season, one sowing date". Both use pre-built cubes, so there was no download.
- Each plan named only its own place, and each "yes" ran its own DSSAT run
  (2 in total).
- A's page never mentioned Yoro or beans; B's never mentioned Nandom or Ghana.
- Answers: A "In 2020, maize in Nandom, Ghana averaged 3,529 kg ha⁻¹" (CSV
  3,529); B "In 2012, beans in Yoro averaged 2,609 kg ha⁻¹" (CSV 2,609).
- Tiles, downloads and session folders each held only their own files. The
  history on each "yes" turn held its own plan (1,110 and 816 characters), and
  no stale-history repair was needed.

Final smoke test, live on the final code (Yoro sowing-date chip, then "yes"):
**pass**.
- The answer came first: "In 2012, the 2012-06-10 sowing gave the highest mean
  yield: 2,806 kg ha⁻¹."
- All four means match the CSV: 2,609 / 2,663 / 2,806 / 2,757.
- No DSSAT before "yes", one run after it, and one place lookup.
- No paths shown; no key panel and no key line in the footer.
- 7861 was stopped afterwards.

Token use per model call (counts only; split estimated from characters):

| Run | Calls | Input avg / max | Output avg / max | System prompt | Tool schemas | History | Tool results |
|---|---|---|---|---|---|---|---|
| Yoro sowing dates (plan + yes) | 8 | 15,892 / 17,960 | 193 / 509 | 67 % | 25 % | 4 % | 3 % |
| Palmira fresh download (plan + yes) | 8 | 16,008 / 18,094 | 190 / 738 | 66 % | 25 % | 5 % | 4 % |

The system prompt (36 k characters, about 10.6 k tokens) is resent on every
call and is the largest part; the 14 tool schemas (about 4 k tokens) come
next. Nothing has been changed based on this yet.

### Pre-built cubes from the dataset

Offline (`cubes_fetch_check`, 15 checks): the download at the pinned commit
gives the same 164 files as the local `cubes/`, every one byte-identical; the
app, system prompt and plan read them from the download folder; a restart with
`HF_HUB_OFFLINE=1` reuses the folder; with an unreachable revision the app
still imports, lists no pre-built places, answers every example with the
message above, sends Nandom 2020 to a fresh download, and leaves no partial
folder. All 352 offline checks pass.

Live on 7861, run from `app_agent.py` with a fresh work folder, so the cubes
were downloaded at start-up:
- **Nandom chip, then "yes"**: "In the 2020 season, maize in Nandom, Ghana,
  averaged 3,529 kg ha⁻¹." CSV 3,529 (sown 2020-05-09, 15 px), as before.
- **Yoro sowing-date chip, then "yes"**: "In the 2012 season, the 2012-06-10
  sowing gave the highest mean yield: 2,806 kg ha⁻¹." Means 2,609 / 2,663 /
  2,806 / 2,757, matching the CSV and the earlier runs.
- Both: `acquire_region` returned "pre-built data cubes" in 0.1 s (no fresh
  download); the run configs point at `cdh_agent/cubes/…` in the work folder;
  no DSSAT before "yes", one run after; no paths shown.
- During the Nandom "yes" turn, Gemini 3.5 returned 503 twice and the run
  continued on 3.1 with the right answer. This was an outage switch, not a
  quota switch, so the quota-switch test below is still open.
- 7861 was stopped afterwards.

### Why yields changed since earlier rounds

Each case was rerun with the current code (clipped district, weather window
from the season) but the old sowing date:

| Case | Earlier | Now | Current code with the old sowing date | Cause |
|---|---|---|---|---|
| Nandom maize 2020 | 3,699 (sown 15 May, 15 px) | 3,529 (sown 9 May, 15 px) | 3,699 (15 May) | Sowing date only (onset rule: 9 May) |
| Palmira maize 2019 | 1,949 (sown 1 Apr, 48 px, 14 no yield) | 2,897 (sown 5 Feb, 48 px, 14 no yield) | 1,948 (1 Apr, 48 px, 14 no yield) | Sowing date (onset rule: 5 Feb); the 1 kg gap to 1,949 comes from the weather window |
| Yoro maize 2012, 0 N | 6,446 (sown 15 May, 140 px, bounding box) | 6,470 (sown 13 May, 96 px, district) | 6,504 (15 May, 96 px) | Both: clipping to the district (6,446 → 6,504), then the sowing date (6,504 → 6,470) |

The earlier sowing dates were the model's own picks (15 May, 1 Apr). They now
come from the onset-of-rains rule stated in the plan. Yoro's area changed when
runs were clipped to the district boundary instead of a bounding box (the
reproducibility fix).

## Open items

Remaining tests:
- **3.1 quota switch**: a simulated quota error on one Gemini model during a
  running job, live (passes offline; blocked by Google's 503s on 3.1). With the
  new order this is now 3.5 → 3.1.
- **Plan + "yes" ×3 on 3.1** (3.5 passed).
- **BYOK live tests**, on a private duplicate Space with `BYOK_ENABLED=1`:
  - pause and resume with no new download;
  - an invalid key;
  - the panel at 390 px;
  - two sessions at once (A's key never used by B);
  - a scan of logs, result files, configs and this changelog for the key.

  The offline BYOK checks (41) pass.

- Citation text: hidden until you provide it.
- The "Climate Data Hub" header link points to the GitHub org; no official CDH
  site URL was found in the repo.
- Follow-ups now need a short plan and a "yes" too (the earlier follow-up
  exemption was removed with the text-based gate).
- A local `.env` (gitignored, excluded by `scripts/deploy_space.py`) now holds
  the keys for local test instances.
