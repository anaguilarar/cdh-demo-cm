---
title: Crop Yield Explorer
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app_agent.py
pinned: false
license: mit
short_description: Ask for a crop yield in plain language; DSSAT runs per pixel
datasets:
  - aaguilar90/cdh-demo-cubes
  - aaguilar90/chirps-v3-daily-rnl
  - aaguilar90/chirts-era5-daily
  - aaguilar90/soilgrids-v2-1km
---

# Crop Yield Explorer

Ask for a crop, a place and a season in plain language. An agent finds the
area, fetches climate and soil, builds datacubes, and runs the DSSAT crop model
pixel by pixel. The page shows a yield map with a value on every pixel, a chart
of yield by sowing date, and — when you ask for several nitrogen rates — the
nitrogen response curve.

Two packages do the work:

* [aggeodata](https://github.com/CGIAR-Climate-Data-Hub/aggeodata) — climate and
  soil acquisition. Precipitation and temperature are read from analysis-ready
  Zarr datacubes on the Hub, so a season is a handful of indexed reads rather
  than hundreds of per-day downloads.
* [ag-cube-cm](https://github.com/CGIAR-Climate-Data-Hub/ag-cube-cm) — the crop
  model. Bundles the DSSAT binary; no separate install.

## Two apps, one Space

| File | What it is | Needs a model key |
|---|---|---|
| `app_agent.py` | **The agent** (default). `SKILL.md` is fetched from the skills repo and *is* the behaviour; the packages' own MCP tools do the work. | Yes |
| `app.py` | A **form**. The same pipeline with sliders instead of a chat — deterministic, no LLM. | No |

Switch by changing `app_file:` in the frontmatter above.

The agent version is the one that demonstrates the argument: nothing in
`app_agent.py` knows what CHIRPS is, which CRS SoilGrids ships in, or that soil
has to be averaged onto the climate grid. That lives in the skill and the
packages, so the page cannot drift from them. It is the same `SKILL.md` a
researcher installs with `npx skills add`, running unmodified.

## Secrets and settings

Under **Settings → Variables and secrets** on the Space (or environment
variables locally). At least one model key is required.

| Name | Kind | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` | Secret | Primary provider ([Gemini key](https://aistudio.google.com/apikey), free). Flash-Lite allows on the order of a thousand requests a day on the free tier — about a hundred conversations. |
| `OPENROUTER_API_KEY` | Secret | Reserve provider. Free models with tool calling, ~50 requests/day; used when Gemini is out of quota or down. |
| `CDSAPI_KEY` | Secret | Recommended. A free [Copernicus CDS](https://cds.climate.copernicus.eu/) token enables AgERA5 solar radiation at 0.1° (an existing `~/.cdsapirc` also counts locally). Without it the app falls back to NASA POWER at 0.5°, which needs no credentials. |

Optional, all with working defaults:

| Name | Default | Purpose |
|---|---|---|
| `CDH_PROVIDERS` | `gemini,openrouter` | Provider order; a single name pins one |
| `CDH_OPENROUTER_MODEL` | `nvidia/nemotron-3.5-super-vl:free` | OpenRouter model slug; check with `scripts/check_openrouter_models.py` |
| `CDH_MODEL` | `gemini-3.5-flash-lite` | Gemini model id; check with `scripts/check_gemini_models.py` |
| `CDH_SOIL_BACKEND` | `hf` | SoilGrids v2 at 1 km from the Hub Zarr store; `isric` uses ISRIC's 250 m services |
| `CDH_RSDS_SOURCE` | `agera5` with a CDS key, else `nasa_power` | Solar radiation source; set it to pin one |
| `CDH_SKILL` | `spatial-crop-modeler` | Skills loaded into the system prompt. `climate-data-download` and `soil-data-download` can be added, at roughly three times the tokens per request |
| `CDH_SKILL_URL` | raw `main` | Pin the first skill to a tag or SHA for a stable demo |
| `CDH_MAX_STEPS` | `30` | Tool-call ceiling per message |
| `CDH_MAX_SIMULATIONS` | `1200` | Job budget quoted to the agent (pixels × windows × seasons) |
| `CDH_NCORES` | `2` | Matches CPU Basic |
| `CDH_SESSION_TTL` | `21600` | Seconds before an idle session's files are removed |
| `CDH_CUBES_REVISION` | pinned commit | Commit of `aaguilar90/cdh-demo-cubes` to download the pre-built cubes from |

Because `CDH_SKILL_URL` tracks `main` by default, edits to the published skill
reach the Space on its next restart. Pin it when the demo needs to be
reproducible.

## Sessions and files

Every browser tab gets its own folder under the work tree. Result files,
configs and the yield map are scoped to that session: two people using the
Space at once never see each other's runs, and the sessions folder is wiped on
every start, so a redeploy never shows a previous visitor's files. Acquired
cubes are shared and cached by region and season, so a second request for the
same district is instant whoever makes it.

Each result file carries the crop, cultivar, nitrogen rate, sowing date and
seasons it was produced with, written as NetCDF attributes. That is what lets
the page group runs into a nitrogen response, and what makes a downloaded file
explain itself.

## Pre-built cubes

Nandom (Ghana, 2020) and Yoro (Honduras, 2012) are pre-built — about 1 MB in
Zarr. Without them the first visitor pays a 3–8 minute download; with them the
demo people actually click runs in about a minute, and the download path stays
reachable for anywhere else.

They are not in this repository. They live in the Hub dataset
[aaguilar90/cdh-demo-cubes](https://huggingface.co/datasets/aaguilar90/cdh-demo-cubes),
and the app downloads them at start-up, pinned to one commit
(`CDH_CUBES_REVISION`), so a rebuilt Space always reads the same data. If the
download fails the app still starts: the examples say the pre-built data is
unavailable, and any request downloads fresh as for any other place.

```bash
python scripts/prebake_cubes.py   # builds cubes/ locally (gitignored)
```

Then upload `cubes/` to the dataset and set `CUBES_REVISION` default in
`app_agent.py` to the new commit hash.

## Running locally

```bash
pip install -r requirements.txt
```

**PowerShell** (Windows):

```powershell
$env:GOOGLE_API_KEY = "..."               # or OPENROUTER_API_KEY
$env:CDSAPI_KEY = "..."                   # optional, AgERA5 radiation
python app_agent.py
```

**bash / zsh**:

```bash
export GOOGLE_API_KEY="..."
python app_agent.py
```

Gradio serves it at `http://127.0.0.1:7860`. Set `CDH_PORT` to change the port
and `CDH_HOST=0.0.0.0` to reach it from another machine on the network. Keep
`CDH_NCORES=2` while testing so the timings match the free tier.

> Never paste a key into a shared terminal, a commit, or a chat. A gitignored
> `.env` is read automatically; a Space is a public git repo and the Hub scans
> for secrets.

## Deploying to a Space

```bash
python scripts/deploy_space.py <namespace>/crop-yield-explorer
```

The script uploads this folder (minus caches and local files) with
`huggingface_hub`, which handles the binary Zarr chunks without a manual LFS
setup, creates the Space if it does not exist, and prints the URL. Set the
secrets above in the Space settings afterwards.

## Running within the free tier

CPU Basic gives 2 vCPU and 16 GB RAM. Memory is never the constraint — **data
acquisition is**. On a district, one season:

| Stage | Cold | Cached |
|-------|------|--------|
| Climate download + datacube | ~3 min | 0 s |
| SoilGrids download + datacube | ~3 min | 0 s |
| DSSAT, 60 simulations | ~6 s | ~6 s |

The crop model is the cheap part. Cubes are therefore cached by region and
season; the first run on a new area takes several minutes, every run after it
on the same area returns in seconds. The cache lives on the Space's ephemeral
disk and is lost when the Space sleeps.

## Reading the output

Both water and nitrogen stress are simulated, and no fertiliser is applied
unless you ask for it, so the default map shows **rainfed, unfertilised yield**
— not potential yield and not observed farm yield. Initial soil nitrogen is
derived from SoilGrids total N, which fixes the depth profile and the contrast
between soils but anchors the absolute level on an assumption; treat the
numbers as relative across the map rather than absolute.

## Data sources

| Variable | Source | Credentials |
|----------|--------|-------------|
| `pr` | CHIRPS v3 daily (Zarr on the Hub) | none |
| `tasmax` / `tasmin` | CHIRTS-ERA5 daily (Zarr on the Hub) | none |
| `rsds` | AgERA5 via the ECMWF ARCO store (0.1°), or NASA POWER (0.5°) as the fallback | `CDSAPI_KEY` for AgERA5 |
| soil | SoilGrids v2 at 1 km (Zarr on the Hub, `aaguilar90/soilgrids-v2-1km`) | none |

**Solar radiation is not optional.** DSSAT's CERES models read `SRAD` from the
weather file and do not derive it from the temperature range; without it every
pixel returns a yield of zero.
