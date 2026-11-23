# -*- coding: utf-8 -*-
"""Build the demo cubes the Space downloads from aaguilar90/cdh-demo-cubes.

A district's climate and soil cubes come to roughly 6 MB, which is small enough
to commit. That matters more than it sounds: measured on this pipeline, a cold
region costs 3-8 minutes of downloading and under a minute of DSSAT, so without
pre-built cubes the first visitor pays for everyone. With them, the demo everyone
actually clicks runs in about a minute and the download path stays reachable for
anyone who asks for somewhere else.

Run locally, then upload `cubes/` to the dataset (it is gitignored here) and
point CUBES_REVISION in app_agent.py at the new commit:

    python scripts/prebake_cubes.py

Needs aggeodata with the [download] extra, and a CDS key if any variable is
sourced from AgERA5.
"""
from __future__ import annotations

import pathlib
import shutil
import sys
import time

import yaml

OUT = pathlib.Path(__file__).resolve().parent.parent / "cubes"
# DSSAT's DSSATPRO parser is whitespace-delimited, so a space anywhere in a path
# it touches corrupts the run. Keep the staging area clear of them too.
STAGE = pathlib.Path("D:/tmp/cdh_prebake") if sys.platform == "win32" \
    else pathlib.Path("/tmp/cdh_prebake")

REGIONS = [
    # tag,      ISO3,  admin unit, level, season
    ("nandom",  "GHA", "Nandom",   2,     2020),
    ("yoro",    "HND", "Yoro",     2,     2012),
]

SOIL_VARS = ["clay", "sand", "silt", "bdod", "cfvo", "soc", "phh2o",
             "wv0010", "wv0033", "wv1500"]
SOIL_DEPTHS = ["0-5", "5-15", "15-30", "30-60"]
IGH = "+proj=igh +lat_0=0 +lon_0=0 +datum=WGS84 +units=m +no_defs"


def extent(iso3: str, name: str, level: int) -> list[float]:
    from aggeodata.ingestion.boundaries import get_admin_boundary
    gdf = get_admin_boundary(iso3, name, level)
    return [float(v) for v in gdf.total_bounds]


def climate(bbox, tag, year, work: pathlib.Path) -> pathlib.Path:
    from aggeodata.pipelines.datacube import run_datacube
    from aggeodata.pipelines.download import run_download
    cfg = {
        "dates": {"starting_date": f"{year}-01-01", "ending_date": f"{year}-12-31"},
        "spatial_info": {"spatial_file": None, "extent": [float(v) for v in bbox]},
        "climate": {"variables": {
            "pr":     {"source": "chirps", "use_zarr": True},
            # Only the era5 mirror has a Zarr store; pairing use_zarr with
            # chirts_source "chirts" fails partway through a run.
            "tasmax": {"source": "chirts", "use_zarr": True, "chirts_source": "era5"},
            "tasmin": {"source": "chirts", "use_zarr": True, "chirts_source": "era5"},
            "rsds":   {"source": "agera5", "use_zarr": True, "agera5_chunking": "geo"},
        }},
        "soil": {"enabled": False},
        "general": {
            "suffix": tag, "use_zarr": True, "ncores": 4, "task": "download",
            "reference_variable": "pr", "agera5_version": "2_0",
            "nasa_power_backend": "s3", "target_crs": "EPSG:4326",
            "download_format": "zarr", "output_format": "zarr",
            "zarr_codec": "zstd", "zarr_quantize": True, "zarr_sharded": False,
        },
        "paths": {"output_path": str(work)},
    }
    p = work / f"{tag}.yaml"
    p.write_text(yaml.dump(cfg, sort_keys=False), encoding="utf-8")
    run_download(str(p))
    run_datacube(str(p))
    cubes = [q for q in work.rglob("*.zarr") if q.is_dir() and q.name.startswith("climate_")]
    if not cubes:
        raise RuntimeError(f"no climate cube produced for {tag}")
    return max(cubes, key=lambda q: q.stat().st_mtime)


def _usable(path: pathlib.Path) -> bool:
    """Is this GeoTIFF present AND readable?

    Existence is not enough: a dropped connection leaves a truncated file behind,
    which passes an `is_file()` check and then fails much later as
    "not recognized as being in a supported file format". Opening it here costs
    milliseconds and turns that into a refetch.
    """
    if not path.is_file() or path.stat().st_size < 1024:
        return False
    try:
        import rasterio
        with rasterio.open(path) as src:
            return src.width > 0 and src.height > 0
    except Exception:  # noqa: BLE001
        return False


def missing_tiles(tiles: pathlib.Path) -> list[tuple[str, str]]:
    """Which (variable, depth) GeoTIFFs are absent or unreadable."""
    out = []
    for v in SOIL_VARS:
        for d in SOIL_DEPTHS:
            f = tiles / f"{v}_{d}cm_mean.tif"
            if not _usable(f):
                f.unlink(missing_ok=True)   # drop the truncated file so it refetches
                out.append((v, d))
    return out


def ensure_tiles(bbox, tiles: pathlib.Path, attempts: int = 3) -> None:
    """Download every tile, retrying the ones ISRIC drops.

    maps.isric.org times out intermittently, and a single missing tile does not
    fail loudly: SoilDataCubeBuilder collects a ragged list of depths and dies
    inside numpy with "inhomogeneous shape", naming neither the variable nor the
    depth. Checking here turns a confusing crash sixty lines away into a retry,
    and failing here names exactly what is missing.
    """
    from aggeodata.ingestion.soil import SoilGridsDownloader
    for attempt in range(1, attempts + 1):
        gaps = missing_tiles(tiles)
        if not gaps:
            return
        if attempt == 1 and len(gaps) == len(SOIL_VARS) * len(SOIL_DEPTHS):
            print(f"  downloading {len(gaps)} soil tiles ...")
            SoilGridsDownloader(soil_layers=SOIL_VARS, depths=SOIL_DEPTHS,
                                output_folder=str(tiles)).download(boundaries=list(bbox))
            continue
        print(f"  attempt {attempt}: refetching {len(gaps)} missing tile(s): "
              + ", ".join(f"{v}@{d}" for v, d in gaps[:6])
              + (" ..." if len(gaps) > 6 else ""))
        for var, depth in gaps:
            try:
                SoilGridsDownloader(soil_layers=[var], depths=[depth],
                                    output_folder=str(tiles)).download(boundaries=list(bbox))
            except Exception as exc:  # noqa: BLE001
                print(f"    {var}@{depth}: {type(exc).__name__}")
        time.sleep(2 * attempt)        # ISRIC rate-limits; back off between rounds

    gaps = missing_tiles(tiles)
    if gaps:
        raise RuntimeError(
            f"{len(gaps)} SoilGrids tile(s) still missing after {attempts} attempts: "
            + ", ".join(f"{v}@{d}cm" for v, d in gaps)
            + ". maps.isric.org is intermittent — rerun this script; tiles already "
              "on disk are kept, so it resumes."
        )


def soil(bbox, tag, work: pathlib.Path) -> pathlib.Path:
    from aggeodata.transform.soil_cube import SoilDataCubeBuilder
    tiles = work / "soil_raw"
    tiles.mkdir(parents=True, exist_ok=True)
    ensure_tiles(bbox, tiles)
    print(f"  {len(list(tiles.glob('*.tif')))} tiles complete")
    # crs and target_crs are passed explicitly: SoilGrids ships in Interrupted
    # Goode Homolosine, and a cube left in metres pairs with no climate cube —
    # it fails later in ways that look like missing data rather than a CRS bug.
    # wv1500 is the skill's reference (1 km, so texture is averaged down rather
    # than water being upsampled). The builder falls back silently to whatever it
    # finds first if the tile is absent, so assert it is there.
    if not _usable(tiles / "wv1500_0-5cm_mean.tif"):
        raise RuntimeError("wv1500 tiles missing — the cube would be built on the "
                           "250 m grid instead, upsampling the water layers")
    return pathlib.Path(SoilDataCubeBuilder(
        data_folder=str(tiles), variables=SOIL_VARS, reference_variable="wv1500",
        crs=IGH, target_crs="EPSG:4326",
    ).build_and_save(output_path=str(work), filename=f"soil_{tag}.zarr",
                     output_format="zarr", codec="zstd"))


def align(weather: pathlib.Path, soil_path: pathlib.Path, dest: pathlib.Path):
    """Average soil onto the climate grid — they must match exactly.

    ag-cube-cm intersects the two validity masks, so grids that differ anywhere
    simulate zero pixels and the run reports success having done nothing.
    """
    import numpy as np
    import rioxarray  # noqa: F401
    import xarray as xr
    from rasterio.enums import Resampling

    def _yx(ds):
        ren = {s: d for s, d in (("lat", "y"), ("latitude", "y"),
                                 ("lon", "x"), ("longitude", "x")) if s in ds.dims}
        return ds.rename(ren) if ren else ds

    w = _yx(xr.open_zarr(str(weather), consolidated=True, decode_coords="all"))
    s = _yx(xr.open_zarr(str(soil_path), consolidated=True, decode_coords="all"))
    tdim = next((d for d in ("date", "time") if d in w.dims), None)
    ref = (w.isel({tdim: 0}) if tdim else w)
    ref = ref[[v for v in ref.data_vars if v != "spatial_ref"][:1]]
    ref = ref.rio.set_spatial_dims(x_dim="x", y_dim="y").rio.write_crs("EPSG:4326")
    s = s.rio.set_spatial_dims(x_dim="x", y_dim="y").rio.write_crs("EPSG:4326")
    # average, not nearest: going 250 m -> 5 km, nearest keeps one source pixel
    # per cell and discards the other ~400.
    out = s.rio.reproject_match(ref, resampling=Resampling.average).load()
    if dest.exists():
        shutil.rmtree(dest)
    out = out.chunk({"depth": 1})
    # The source store's chunk encoding no longer describes this array, and zarr
    # refuses the write rather than risk corrupting it.
    for v in out.variables:
        out[v].encoding = {}
    out.to_zarr(dest, mode="w", consolidated=True)
    exact = bool(np.array_equal(out["y"].values, w["y"].values)
                 and np.array_equal(out["x"].values, w["x"].values))
    w.close(); s.close()
    if not exact:
        raise RuntimeError(f"{dest.name} did not land on the weather grid")
    return dest


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for tag, iso3, name, level, year in REGIONS:
        work = STAGE / tag
        work.mkdir(parents=True, exist_ok=True)
        dest_w = OUT / f"{tag}_cube_climate_{year}.zarr"
        dest_s = OUT / f"{tag}_cube_soil.zarr"
        if dest_w.is_dir() and dest_s.is_dir():
            print(f"{tag}: already built, skipping")
            continue

        print(f"\n=== {name}, {iso3} ({year}) ===")
        t0 = time.time()
        bbox = extent(iso3, name, level)
        print(f"  extent {[round(v, 3) for v in bbox]}")

        w = climate(bbox, tag, year, work)
        print(f"  climate {w.name}")
        s = soil(bbox, tag, work)
        print(f"  soil    {s.name}")

        if dest_w.exists():
            shutil.rmtree(dest_w)
        shutil.copytree(w, dest_w)
        align(dest_w, s, dest_s)

        mb = sum(f.stat().st_size for d in (dest_w, dest_s)
                 for f in d.rglob("*") if f.is_file()) / 1e6
        print(f"  -> cubes/ {mb:.1f} MB in {(time.time() - t0) / 60:.1f} min")

    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e6
    print(f"\ncubes/ is {total:.1f} MB — commit it so the Space ships with them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
