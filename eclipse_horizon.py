#!/usr/bin/env python3
"""
eclipse_horizon.py — SRTM-based horizon visibility checker for the 2026 Spanish eclipse.

OVERVIEW
--------
This script downloads SRTM 1-arc-second (~30 m) elevation tiles and computes
the terrain horizon profile from any candidate observation point.  It then
compares the horizon with the sun's position during the August 12 2026 total
solar eclipse to tell you whether the sun will be blocked by terrain.

NEW IN v0.2 — TREE CANOPY ANALYSIS & ECLIPSE DURATION
------------------------------------------------------
  * Computes totality duration for any location within the path (NASA/Espenak data)
  * Optionally overlays Meta/WRI Global Canopy Height data (1 m resolution) onto
    the SRTM terrain model to warn about tree obstruction.
  * Requires rasterio for canopy height (falls back gracefully without it).
  * CLI with argparse: analyze any lat/lon from the command line.

NEW IN v0.2.2 — SOLAR DISK VISIBILITY & PATH FIX
-------------------------------------------------
  * Solar disk angular size (0.53° diameter) now modelled — reports full/partial/blocked.
  * Path extended to sunset terminator: covers Valencia, Ibiza, Mallorca.
  * Point-in-polygon path test replaces distance heuristic (fixes false negatives
    near the tapering eastern end of the path).

SETUP
-----
    pip install numpy matplotlib requests
    pip install rasterio          # optional: enables canopy height analysis

CANOPY HEIGHT DATA
------------------
Uses Meta/WRI Global Canopy Height Map (Cloud-Optimized GeoTIFF on AWS S3).
No download required - reads only the needed pixels via HTTP range requests.
Source: s3://dataforgood-fb-data/forests/v1/alsgedi_global_v6_float/chm/

QUICK START
-----------
    from eclipse_horizon import EclipseHorizonChecker

    checker = EclipseHorizonChecker()
    checker.analyze_site("Castillo de Monzon", lat=42.1245, lon=-4.4983, observer_height=5.0)

Eclipse Predictions by Fred Espenak, NASA's GSFC.
Canopy height data: Meta and World Resources Institute (WRI), 2024.
"""

import gzip
import math
from pathlib import Path
from typing import Optional, cast

import numpy as np
import requests

# ---------------------------------------------------------------------------
# Optional rasterio import (needed only for canopy height)
# ---------------------------------------------------------------------------
try:
    import rasterio
    import rasterio.windows
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

# ---------------------------------------------------------------------------
# Eclipse parameters - defaults for the Palencia area
# Overridden per-site by compute_eclipse_circumstances()
# ---------------------------------------------------------------------------
ECLIPSE: dict[str, float | str] = {
    "sun_azimuth_deg": 280.0,
    "sun_altitude_deg": 9.0,
    "azimuth_tolerance_deg": 15.0,
    "date": "2026-08-12",
    "time_local": "20:30 CEST",
    "totality_sec": 104,
}

EARTH_RADIUS_M = 6_371_000.0
REFRACTION_K = 0.13
MAX_RAY_DISTANCE_M = 80_000
RAY_STEP_M = 30
CACHE_DIR = Path("srtm_cache")
SOLAR_RADIUS_DEG = 0.267  # angular radius of the sun (~0.53 deg diameter)


# ---------------------------------------------------------------------------
# NASA/Espenak Eclipse Path Table - 2026 Aug 12
# Source: https://eclipse.gsfc.nasa.gov/SEpath/SEpath2001/SE2026Aug12Tpath.html
# Fields: UT hours, centerline lat/lon, north/south limit lat/lon,
#         sun altitude/azimuth, path width (km), centerline duration (seconds)
# ---------------------------------------------------------------------------
ECLIPSE_PATH_TABLE: list[dict[str, float]] = [
    {"ut": 17.70, "c_lat": 67.21, "c_lon": -26.03, "n_lat": 67.67, "n_lon": -22.71, "s_lat": 66.71, "s_lon": -29.13, "alt": 26, "az": 246, "width": 291, "dur": 138.1},
    {"ut": 17.77, "c_lat": 65.17, "c_lon": -25.21, "n_lat": 65.59, "n_lon": -22.12, "s_lat": 64.71, "s_lon": -28.11, "alt": 26, "az": 248, "width": 294, "dur": 138.2},
    {"ut": 17.83, "c_lat": 63.17, "c_lon": -24.29, "n_lat": 63.56, "n_lon": -21.38, "s_lat": 62.75, "s_lon": -27.03, "alt": 26, "az": 251, "width": 298, "dur": 137.9},
    {"ut": 17.90, "c_lat": 61.20, "c_lon": -23.28, "n_lat": 61.55, "n_lon": -20.51, "s_lat": 60.82, "s_lon": -25.91, "alt": 25, "az": 254, "width": 302, "dur": 137.1},
    {"ut": 17.97, "c_lat": 59.25, "c_lon": -22.17, "n_lat": 59.55, "n_lon": -19.50, "s_lat": 58.90, "s_lon": -24.71, "alt": 25, "az": 257, "width": 305, "dur": 136.0},
    {"ut": 18.03, "c_lat": 57.30, "c_lon": -20.95, "n_lat": 57.57, "n_lon": -18.35, "s_lat": 56.99, "s_lon": -23.42, "alt": 24, "az": 260, "width": 309, "dur": 134.5},
    {"ut": 18.10, "c_lat": 55.34, "c_lon": -19.59, "n_lat": 55.57, "n_lon": -17.03, "s_lat": 55.08, "s_lon": -22.03, "alt": 23, "az": 263, "width": 313, "dur": 132.5},
    {"ut": 18.17, "c_lat": 53.37, "c_lon": -18.06, "n_lat": 53.55, "n_lon": -15.50, "s_lat": 53.15, "s_lon": -20.49, "alt": 22, "az": 266, "width": 316, "dur": 130.0},
    {"ut": 18.23, "c_lat": 51.36, "c_lon": -16.30, "n_lat": 51.48, "n_lon": -13.71, "s_lat": 51.19, "s_lon": -18.76, "alt": 20, "az": 269, "width": 319, "dur": 127.0},
    {"ut": 18.30, "c_lat": 49.29, "c_lon": -14.24, "n_lat": 49.33, "n_lon": -11.55, "s_lat": 49.18, "s_lon": -16.77, "alt": 18, "az": 272, "width": 319, "dur": 123.3},
    {"ut": 18.37, "c_lat": 47.10, "c_lon": -11.72, "n_lat": 47.04, "n_lon":  -8.80, "s_lat": 47.08, "s_lon": -14.40, "alt": 16, "az": 275, "width": 318, "dur": 118.8},
    {"ut": 18.40, "c_lat": 45.94, "c_lon": -10.19, "n_lat": 45.80, "n_lon":  -7.08, "s_lat": 45.98, "s_lon": -13.01, "alt": 14, "az": 277, "width": 315, "dur": 116.1},
    {"ut": 18.43, "c_lat": 44.71, "c_lon":  -8.40, "n_lat": 44.46, "n_lon":  -4.95, "s_lat": 44.83, "s_lon": -11.42, "alt": 13, "az": 278, "width": 311, "dur": 113.0},
    {"ut": 18.47, "c_lat": 43.37, "c_lon":  -6.19, "n_lat": 42.91, "n_lon":  -2.09, "s_lat": 43.61, "s_lon":  -9.55, "alt": 10, "az": 281, "width": 304, "dur": 109.3},
    {"ut": 18.50, "c_lat": 41.82, "c_lon":  -3.19, "n_lat": 40.67, "n_lon":   3.30, "s_lat": 42.26, "s_lon":  -7.24, "alt":  8, "az": 283, "width": 294, "dur": 104.6},
    {"ut": 18.53, "c_lat": 39.41, "c_lon":   2.95, "n_lat": 39.71, "n_lon":   6.34, "s_lat": 40.68, "s_lon":  -4.04, "alt":  2, "az": 288, "width": 270, "dur":  95.8},
    # Sunset terminator — geometric end of totality path (NASA Limits row)
    {"ut": 18.53, "c_lat": 38.68, "c_lon":   5.42, "n_lat": 39.71, "n_lon":   6.34, "s_lat": 37.69, "s_lon":   4.54, "alt":  0, "az": 290, "width": 262, "dur":  92.8},
]


# ---------------------------------------------------------------------------
# Duration & sun position computation
# ---------------------------------------------------------------------------

def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two geographic points."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return EARTH_RADIUS_M * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _point_to_segment_distance(px: float, py: float,
                                ax: float, ay: float,
                                bx: float, by: float) -> tuple[float, float]:
    """Distance (km) from point P to line segment AB, and fractional t along AB."""
    mid_lat = math.radians((ay + by + py) / 3)
    cos_lat = math.cos(mid_lat)
    kpd_lat = 111.32
    kpd_lon = 111.32 * cos_lat

    pxk, pyk = px * kpd_lon, py * kpd_lat
    axk, ayk = ax * kpd_lon, ay * kpd_lat
    bxk, byk = bx * kpd_lon, by * kpd_lat

    dx, dy = bxk - axk, byk - ayk
    seg2 = dx * dx + dy * dy
    if seg2 < 1e-12:
        return math.hypot(pxk - axk, pyk - ayk), 0.0

    t = max(0.0, min(1.0, ((pxk - axk) * dx + (pyk - ayk) * dy) / seg2))
    proj_x = axk + t * dx
    proj_y = ayk + t * dy
    return math.hypot(pxk - proj_x, pyk - proj_y), t


def _point_in_eclipse_path(lat: float, lon: float) -> bool:
    """Ray-casting point-in-polygon test against the eclipse path outline."""
    north = [(r["n_lat"], r["n_lon"]) for r in ECLIPSE_PATH_TABLE]
    south = [(r["s_lat"], r["s_lon"]) for r in ECLIPSE_PATH_TABLE]
    poly = north + south[::-1]

    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        yi, xi = poly[i]
        yj, xj = poly[j]
        if ((yi > lat) != (yj > lat)) and \
           (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _solar_disk_status(margin_deg: float) -> str:
    """Classify solar disk visibility based on clearance margin.

    Returns 'full', 'partial', or 'blocked'.
      full    — entire disk clears the horizon (margin > +0.267°)
      partial — centre visible but lower limb grazes terrain/trees
      blocked — geometric centre below horizon
    """
    if margin_deg > SOLAR_RADIUS_DEG:
        return "full"
    elif margin_deg > -SOLAR_RADIUS_DEG:
        return "partial"
    else:
        return "blocked"


def compute_eclipse_circumstances(lat: float, lon: float) -> dict[str, float | bool | str]:
    """
    Compute eclipse circumstances for any location using the NASA path table.

    Returns dict with: in_totality, duration_sec, duration_str,
    sun_altitude_deg, sun_azimuth_deg, distance_from_centerline_km,
    path_half_width_km, centerline_duration_sec
    """
    table = ECLIPSE_PATH_TABLE
    best_dist = float("inf")
    best_i = 0
    best_t = 0.0

    for i in range(len(table) - 1):
        a, b = table[i], table[i + 1]
        d, t = _point_to_segment_distance(
            lon, lat, a["c_lon"], a["c_lat"], b["c_lon"], b["c_lat"])
        if d < best_dist:
            best_dist, best_i, best_t = d, i, t

    a, b = table[best_i], table[best_i + 1]
    lerp = lambda k: a[k] + best_t * (b[k] - a[k])

    c_lat, c_lon = lerp("c_lat"), lerp("c_lon")
    sun_alt, sun_az = lerp("alt"), lerp("az")
    path_width_km = lerp("width")
    cl_dur = lerp("dur")

    n_lat, n_lon = lerp("n_lat"), lerp("n_lon")
    s_lat, s_lon = lerp("s_lat"), lerp("s_lon")

    dist_to_n = _haversine_distance(lat, lon, n_lat, n_lon) / 1000
    dist_to_s = _haversine_distance(lat, lon, s_lat, s_lon) / 1000
    dist_n_s = _haversine_distance(n_lat, n_lon, s_lat, s_lon) / 1000

    half_w = dist_n_s / 2.0
    actual_dist = _haversine_distance(lat, lon, c_lat, c_lon) / 1000

    in_totality = _point_in_eclipse_path(lat, lon)

    if in_totality and half_w > 0:
        ratio = min(actual_dist / half_w, 0.999)
        dur = cl_dur * math.sqrt(1.0 - ratio * ratio)
    else:
        dur = 0.0

    # Latitude-based sun altitude refinement
    sun_alt_adj = sun_alt + (lat - c_lat) * 0.15

    return {
        "in_totality": in_totality,
        "duration_sec": dur,
        "duration_str": f"{int(dur // 60)}m{dur % 60:04.1f}s",
        "sun_altitude_deg": sun_alt_adj,
        "sun_azimuth_deg": sun_az,
        "distance_from_centerline_km": actual_dist,
        "path_half_width_km": half_w,
        "centerline_duration_sec": cl_dur,
    }


# ---------------------------------------------------------------------------
# SRTM tile I/O
# ---------------------------------------------------------------------------

def tile_filename(lat: int, lon: int) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}.hgt"


def download_tile(lat: int, lon: int, cache_dir: Path = CACHE_DIR) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    fname = tile_filename(lat, lon)
    local_path = cache_dir / fname

    if local_path.exists() and local_path.stat().st_size > 1_000_000:
        return local_path

    ns = "N" if lat >= 0 else "S"
    tile_name = f"{ns}{abs(lat):02d}{'E' if lon >= 0 else 'W'}{abs(lon):03d}"

    urls = [
        f"https://opentopography.s3.sdsc.edu/raster/SRTM_GL1/SRTM_GL1_srtm/{tile_name}.hgt",
        f"https://s3.amazonaws.com/elevation-tiles-prod/skadi/{ns}{abs(lat):02d}/{tile_name}.hgt.gz",
    ]

    for url in urls:
        print(f"  Downloading {fname} from {url.split('/')[2]} ...", end=" ", flush=True)
        try:
            r = requests.get(url, timeout=120, stream=True)
            if r.status_code != 200:
                print(f"HTTP {r.status_code}, trying next mirror.")
                continue
            data = r.content
            if url.endswith(".gz"):
                data = gzip.decompress(data)
            local_path.write_bytes(data)
            print(f"OK ({len(data) / 1e6:.1f} MB)")
            return local_path
        except Exception as e:
            print(f"failed ({e}), trying next mirror.")

    raise FileNotFoundError(f"Could not download tile {fname}. Place it manually in {cache_dir}/")


def load_tile(lat: int, lon: int, cache_dir: Path = CACHE_DIR) -> np.ndarray:
    path = download_tile(lat, lon, cache_dir)
    data = np.frombuffer(path.read_bytes(), dtype=">i2")
    side = int(math.sqrt(len(data)))
    if side not in (3601, 1201):
        raise ValueError(f"Unexpected tile size {len(data)} (side={side})")
    return data.reshape((side, side))


class SrtmElevationModel:
    """Lazy-loading mosaic of SRTM tiles with bilinear interpolation."""

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self._tiles: dict[tuple[int, int], np.ndarray | None] = {}

    def _ensure_tile(self, lat_floor: int, lon_floor: int) -> np.ndarray | None:
        key = (lat_floor, lon_floor)
        if key not in self._tiles:
            try:
                self._tiles[key] = load_tile(lat_floor, lon_floor, self.cache_dir)
            except FileNotFoundError as e:
                print(f"\n  Warning: {e}\n  Treating as flat terrain (0 m).\n")
                self._tiles[key] = None
        return self._tiles[key]

    def get_elevation(self, lat: float, lon: float) -> float:
        lat_floor = math.floor(lat)
        lon_floor = math.floor(lon)
        tile = self._ensure_tile(lat_floor, lon_floor)
        if tile is None:
            return 0.0

        n = tile.shape[0] - 1
        y_frac = lat - lat_floor
        x_frac = lon - lon_floor
        row = (1.0 - y_frac) * n
        col = x_frac * n

        r0, c0 = int(math.floor(row)), int(math.floor(col))
        r1, c1 = min(r0 + 1, n), min(c0 + 1, n)
        dr, dc = row - r0, col - c0

        z00, z01 = float(tile[r0, c0]), float(tile[r0, c1])
        z10, z11 = float(tile[r1, c0]), float(tile[r1, c1])

        if all(z > -1000 for z in [z00, z01, z10, z11]):
            return (z00 * (1 - dr) * (1 - dc) + z01 * (1 - dr) * dc +
                    z10 * dr * (1 - dc) + z11 * dr * dc)
        vals = [z for z in [z00, z01, z10, z11] if z > -1000]
        return sum(vals) / len(vals) if vals else 0.0

    def preload_region(self, lat_min: float, lat_max: float,
                       lon_min: float, lon_max: float) -> None:
        for la in range(math.floor(lat_min), math.floor(lat_max) + 1):
            for lo in range(math.floor(lon_min), math.floor(lon_max) + 1):
                self._ensure_tile(la, lo)


# ---------------------------------------------------------------------------
# Canopy Height Model (Meta/WRI) - optional, requires rasterio
# ---------------------------------------------------------------------------

def _lat_lon_to_quadkey(lat: float, lon: float, zoom: int = 9) -> str:
    """Convert lat/lon to a Bing Maps quadkey at the given zoom level."""
    lat_rad = math.radians(lat)
    n = 2 ** zoom
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2 * n)
    x = max(0, min(n - 1, x))
    y = max(0, min(n - 1, y))

    quadkey = ""
    for i in range(zoom, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if x & mask:
            digit += 1
        if y & mask:
            digit += 2
        quadkey += str(digit)
    return quadkey


class CanopyHeightModel:
    """
    Access Meta/WRI Global Canopy Height data via Cloud-Optimized GeoTIFF.

    Reads only needed pixels via HTTP range requests - no bulk download.
    Data is 1 m resolution canopy height in metres (float32), EPSG:3857.
    Source: s3://dataforgood-fb-data/forests/v1/alsgedi_global_v6_float/chm/
    """

    BASE_URL = "https://dataforgood-fb-data.s3.amazonaws.com/forests/v1/alsgedi_global_v6_float/chm"

    def __init__(self) -> None:
        self._open_datasets: dict[str, object] = {}
        self._failed_quadkeys: set[str] = set()

    def _get_dataset(self, quadkey: str) -> object | None:
        if quadkey in self._failed_quadkeys:
            return None
        if quadkey not in self._open_datasets:
            url = f"{self.BASE_URL}/{quadkey}.tif"
            try:
                env = rasterio.Env(
                    GDAL_HTTP_UNSAFESSL="YES",
                    AWS_NO_SIGN_REQUEST="YES",
                    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                )
                env.__enter__()
                ds = rasterio.open(url)
                self._open_datasets[quadkey] = ds
            except Exception as e:
                print(f"  Warning: Cannot open canopy tile {quadkey}: {e}")
                self._failed_quadkeys.add(quadkey)
                return None
        return self._open_datasets[quadkey]

    @staticmethod
    def _to_mercator(lat: float, lon: float) -> tuple[float, float]:
        """Convert WGS84 to EPSG:3857 Web Mercator."""
        x = lon * 20037508.34 / 180.0
        lat_rad = math.radians(lat)
        y = math.log(math.tan(math.pi / 4 + lat_rad / 2)) * 20037508.34 / math.pi
        return x, y

    def get_canopy_height(self, lat: float, lon: float) -> float:
        """Return canopy height (metres) at the given coordinate."""
        quadkey = _lat_lon_to_quadkey(lat, lon, zoom=9)
        ds = self._get_dataset(quadkey)
        if ds is None:
            return 0.0

        try:
            x_merc, y_merc = self._to_mercator(lat, lon)
            row, col = ds.index(x_merc, y_merc)  # type: ignore[union-attr]
            if 0 <= row < ds.height and 0 <= col < ds.width:  # type: ignore[union-attr]
                window = rasterio.windows.Window(col, row, 1, 1)
                data = ds.read(1, window=window)  # type: ignore[union-attr]
                val = float(data[0, 0])
                if val < 0 or val > 100 or np.isnan(val):
                    return 0.0
                return val
            return 0.0
        except Exception:
            return 0.0

    def close(self) -> None:
        for ds in self._open_datasets.values():
            if hasattr(ds, "close"):
                ds.close()  # type: ignore[union-attr]
        self._open_datasets.clear()


# ---------------------------------------------------------------------------
# Horizon computation - with optional canopy height
# ---------------------------------------------------------------------------

def destination_point(lat: float, lon: float, azimuth_deg: float,
                      distance_m: float) -> tuple[float, float]:
    """Destination point on a sphere (Vincenty direct, spherical)."""
    p1 = math.radians(lat)
    l1 = math.radians(lon)
    th = math.radians(azimuth_deg)
    d = distance_m / EARTH_RADIUS_M
    p2 = math.asin(math.sin(p1) * math.cos(d) + math.cos(p1) * math.sin(d) * math.cos(th))
    l2 = l1 + math.atan2(math.sin(th) * math.sin(d) * math.cos(p1),
                          math.cos(d) - math.sin(p1) * math.sin(p2))
    return math.degrees(p2), math.degrees(l2)


def compute_horizon_angle(dem: SrtmElevationModel,
                          lat: float, lon: float,
                          azimuth_deg: float,
                          observer_height_m: float = 2.0,
                          max_distance_m: float = MAX_RAY_DISTANCE_M,
                          step_m: float = RAY_STEP_M,
                          chm: Optional[CanopyHeightModel] = None,
                          canopy_scan_m: float = 500.0,
                          ) -> tuple[float, float, float, float]:
    """
    Trace a ray and return both terrain-only and terrain+canopy horizon angles.

    At 9 deg sun altitude, a 25 m tree blocks only within ~160 m,
    so canopy_scan_m=500 is conservative.

    Returns: (terrain_angle, terrain_dist, canopy_angle, canopy_dist)
    """
    obs_elev = dem.get_elevation(lat, lon) + observer_height_m
    R_eff = EARTH_RADIUS_M / (1.0 - REFRACTION_K)

    max_t_ang, max_t_dist = -90.0, 0.0
    max_c_ang, max_c_dist = -90.0, 0.0

    dist = step_m
    while dist <= max_distance_m:
        plat, plon = destination_point(lat, lon, azimuth_deg, dist)
        target_elev = dem.get_elevation(plat, plon)

        dh = target_elev - obs_elev
        curve_drop = (dist * dist) / (2.0 * R_eff)
        app_dh = dh - curve_drop
        ang = math.degrees(math.atan2(app_dh, dist))

        if ang > max_t_ang:
            max_t_ang, max_t_dist = ang, dist

        # Canopy within scan radius
        if chm is not None and dist <= canopy_scan_m:
            ch = chm.get_canopy_height(plat, plon)
            if ch > 0.5:
                c_ang = math.degrees(math.atan2(app_dh + ch, dist))
                if c_ang > max_c_ang:
                    max_c_ang, max_c_dist = c_ang, dist
            elif ang > max_c_ang:
                max_c_ang, max_c_dist = ang, dist
        elif ang > max_c_ang:
            max_c_ang, max_c_dist = ang, dist

        if dist > 10_000 and ang < max_t_ang - 5:
            dist += step_m * 4
        else:
            dist += step_m

    if chm is None:
        max_c_ang, max_c_dist = max_t_ang, max_t_dist

    return max_t_ang, max_t_dist, max_c_ang, max_c_dist


def compute_horizon_profile(dem: SrtmElevationModel,
                            lat: float, lon: float,
                            observer_height_m: float = 2.0,
                            azimuth_start: float = 0.0,
                            azimuth_end: float = 360.0,
                            azimuth_step: float = 1.0,
                            max_distance_m: float = MAX_RAY_DISTANCE_M,
                            chm: Optional[CanopyHeightModel] = None,
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                       np.ndarray, np.ndarray]:
    """
    Compute horizon profile. Returns:
    (azimuths, terrain_angles, terrain_dists, canopy_angles, canopy_dists)
    """
    azimuths = np.arange(azimuth_start, azimuth_end, azimuth_step)
    t_ang = np.zeros_like(azimuths)
    t_dst = np.zeros_like(azimuths)
    c_ang = np.zeros_like(azimuths)
    c_dst = np.zeros_like(azimuths)

    total = len(azimuths)
    for i, az in enumerate(azimuths):
        if i % 30 == 0:
            print(f"\r  Computing horizon: {i}/{total} ({az:.0f} deg) ...", end="", flush=True)
        ta, td, ca, cd = compute_horizon_angle(
            dem, lat, lon, az, observer_height_m, max_distance_m, chm=chm)
        t_ang[i], t_dst[i], c_ang[i], c_dst[i] = ta, td, ca, cd
    print(f"\r  Computing horizon: {total}/{total} - done.           ")
    return azimuths, t_ang, t_dst, c_ang, c_dst


# ---------------------------------------------------------------------------
# Eclipse-specific analysis
# ---------------------------------------------------------------------------

def check_eclipse_visibility(dem: SrtmElevationModel,
                             lat: float, lon: float,
                             observer_height_m: float = 2.0,
                             chm: Optional[CanopyHeightModel] = None,
                             ) -> dict:
    """Check whether the eclipse sun will be visible from a candidate site."""
    circ = compute_eclipse_circumstances(lat, lon)
    sun_az = circ["sun_azimuth_deg"]
    sun_alt = circ["sun_altitude_deg"]
    half_w = cast(float, ECLIPSE["azimuth_tolerance_deg"])

    azimuths, t_ang, t_dst, c_ang, c_dst = compute_horizon_profile(
        dem, lat, lon, observer_height_m,
        azimuth_start=sun_az - half_w, azimuth_end=sun_az + half_w,
        azimuth_step=0.5, chm=chm)

    idx = np.argmin(np.abs(azimuths - sun_az))
    t_margin = sun_alt - t_ang[idx]
    c_margin = sun_alt - c_ang[idx]

    # Effective margin accounts for canopy when available
    effective_margin = c_margin if chm is not None else t_margin
    disk_status = _solar_disk_status(effective_margin)

    return {
        "visible_terrain": t_margin > 0,
        "visible_canopy": c_margin > 0,
        "visible": c_margin > 0 if chm is not None else t_margin > 0,
        "sun_altitude_deg": sun_alt,
        "sun_azimuth_deg": sun_az,
        "terrain_horizon_deg": t_ang[idx],
        "canopy_horizon_deg": c_ang[idx],
        "terrain_margin_deg": t_margin,
        "canopy_margin_deg": c_margin,
        "disk_status": disk_status,
        "worst_terrain_deg": t_ang.max(),
        "worst_terrain_az": azimuths[t_ang.argmax()],
        "worst_canopy_deg": c_ang.max(),
        "worst_canopy_az": azimuths[c_ang.argmax()],
        "profile_azimuths": azimuths,
        "profile_terrain_angles": t_ang,
        "profile_canopy_angles": c_ang,
        "eclipse": circ,
    }


# ---------------------------------------------------------------------------
# High-level analysis & plotting
# ---------------------------------------------------------------------------

class EclipseHorizonChecker:
    """Convenient wrapper for eclipse visibility analysis."""

    def __init__(self, cache_dir: str | Path = CACHE_DIR, use_canopy: bool = True):
        self.dem = SrtmElevationModel(Path(cache_dir))
        self.chm: Optional[CanopyHeightModel] = None

        if use_canopy and HAS_RASTERIO:
            self.chm = CanopyHeightModel()
            print("  Tree canopy analysis enabled (Meta/WRI CHM via rasterio)")
        elif use_canopy:
            print("  Note: rasterio not installed - canopy analysis disabled.")
            print("        Install with: pip install rasterio")

    def analyze_site(self, name: str, lat: float, lon: float,
                     observer_height: float = 2.0,
                     full_profile: bool = True,
                     save_plot: bool = True) -> dict:
        """Run a complete eclipse visibility analysis for one site."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        print(f"\n{'='*70}")
        print(f" ECLIPSE VISIBILITY ANALYSIS: {name}")
        print(f"{'='*70}")

        ground_elev = self.dem.get_elevation(lat, lon)
        lon_lbl = f"{abs(lon):.5f} deg {'W' if lon < 0 else 'E'}"
        print(f"  Location:  {lat:.5f} deg N, {lon_lbl}")
        print(f"  Ground elevation: {ground_elev:.0f} m | Observer +{observer_height:.1f} m")

        # Eclipse circumstances
        ec = compute_eclipse_circumstances(lat, lon)
        print()
        if ec["in_totality"]:
            print(f"  TOTALITY: {ec['duration_str']}  "
                  f"({ec['distance_from_centerline_km']:.1f} km from centerline)")
        else:
            print(f"  OUTSIDE TOTALITY PATH  "
                  f"({ec['distance_from_centerline_km']:.1f} km from centerline)")
        print(f"  Sun: az={ec['sun_azimuth_deg']:.1f} deg, alt={ec['sun_altitude_deg']:.1f} deg")
        print()

        # Horizon analysis
        result = check_eclipse_visibility(self.dem, lat, lon, observer_height, self.chm)

        print(f"  Terrain horizon at sun azimuth: {result['terrain_horizon_deg']:.2f} deg")
        print(f"  Terrain clearance: {result['terrain_margin_deg']:+.2f} deg")

        if result["visible_terrain"]:
            print(f"  >> TERRAIN OK - {result['terrain_margin_deg']:.1f} deg clearance")
        else:
            print(f"  >> TERRAIN BLOCKS SUN by {abs(result['terrain_margin_deg']):.1f} deg")

        has_canopy = self.chm is not None
        if has_canopy:
            extra = result["canopy_horizon_deg"] - result["terrain_horizon_deg"]
            print(f"\n  Canopy horizon at sun azimuth: {result['canopy_horizon_deg']:.2f} deg")
            print(f"  Canopy clearance: {result['canopy_margin_deg']:+.2f} deg")
            if extra > 0.1:
                print(f"  Trees add {extra:.1f} deg to horizon")
            if result["visible_terrain"] and not result["visible_canopy"]:
                print(f"  >> WARNING: Trees block the sun!")
            elif not result["visible_terrain"]:
                pass  # already reported terrain blocked
            else:
                print(f"  >> TREES OK - {result['canopy_margin_deg']:.1f} deg clearance")

        # Solar disk status
        ds = result["disk_status"]
        if result["visible"]:
            if ds == "full":
                print(f"\n  ☉ FULL CORONA CLEAR — entire solar disk above horizon")
            elif ds == "partial":
                print(f"\n  ⊘ LOWER LIMB CLIPPED — centre visible, edge grazes horizon")

        # Full profile
        az_f = t_f = c_f = None
        if full_profile:
            print("\n  Computing full 360 deg horizon profile ...")
            az_f, t_f, _, c_f, _ = compute_horizon_profile(
                self.dem, lat, lon, observer_height,
                azimuth_start=0, azimuth_end=360, azimuth_step=1.0, chm=self.chm)

        # --- Plotting ---
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        dur_lbl = ec['duration_str'] if ec['in_totality'] else "outside path"
        fig.suptitle(f"Eclipse Horizon: {name}\n"
                     f"({lat:.4f} N, {abs(lon):.4f} {'W' if lon < 0 else 'E'}, "
                     f"elev {ground_elev:.0f} m | Totality: {dur_lbl})",
                     fontsize=13, fontweight="bold")

        ax = axes[0]
        az_b = result["profile_azimuths"]
        tb = result["profile_terrain_angles"]
        cb = result["profile_canopy_angles"]

        ax.fill_between(az_b, 0, tb, alpha=0.3, color="saddlebrown", label="Terrain")
        ax.plot(az_b, tb, "saddlebrown", linewidth=2)
        if has_canopy and np.any(cb > tb + 0.05):
            ax.fill_between(az_b, tb, cb, alpha=0.4, color="forestgreen", label="Canopy")
            ax.plot(az_b, cb, "darkgreen", linewidth=1.5, linestyle="--")

        ax.axhline(result["sun_altitude_deg"], color="gold", linewidth=2,
                   linestyle="--", label=f"Sun alt ({result['sun_altitude_deg']:.1f} deg)")
        ax.axhline(result["sun_altitude_deg"] + SOLAR_RADIUS_DEG, color="gold",
                   linewidth=0.8, linestyle=":", alpha=0.5, label="Upper/lower limb")
        ax.axhline(result["sun_altitude_deg"] - SOLAR_RADIUS_DEG, color="gold",
                   linewidth=0.8, linestyle=":", alpha=0.5)
        ax.axvline(result["sun_azimuth_deg"], color="orange", linewidth=1,
                   linestyle=":", alpha=0.7)
        ax.set_xlabel("Azimuth (deg)")
        ax.set_ylabel("Elevation angle (deg)")
        ax.set_title("Eclipse azimuth band")
        ax.legend(fontsize=9)
        ymax = max(12, cb.max() + 2, tb.max() + 2)
        ax.set_ylim(bottom=min(-1, tb.min() - 1), top=ymax)
        ax.grid(True, alpha=0.3)

        ax2 = axes[1]
        if az_f is not None and t_f is not None and c_f is not None:
            ax2.fill_between(az_f, 0, t_f, alpha=0.3, color="saddlebrown")
            ax2.plot(az_f, t_f, "saddlebrown", linewidth=1.5, label="Terrain")
            if has_canopy and np.any(c_f > t_f + 0.05):
                ax2.fill_between(az_f, t_f, c_f, alpha=0.3, color="forestgreen", label="Canopy")
            ax2.axhline(result["sun_altitude_deg"], color="gold", linewidth=2, linestyle="--")
            ax2.plot(result["sun_azimuth_deg"], result["sun_altitude_deg"],
                     "o", color="gold", markersize=14, markeredgecolor="darkorange",
                     markeredgewidth=2, zorder=5, label="Sun")
            tol = cast(float, ECLIPSE["azimuth_tolerance_deg"])
            ax2.axvspan(result["sun_azimuth_deg"] - tol,
                        result["sun_azimuth_deg"] + tol,
                        alpha=0.1, color="orange")
            ax2.set_xlabel("Azimuth (deg)")
            ax2.set_ylabel("Elevation angle (deg)")
            ax2.set_title("Full 360 deg horizon")
            ax2.set_xlim(0, 360)
            ax2.set_xticks(range(0, 361, 45))
            ax2.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"])
            ax2.legend(fontsize=9, loc="upper left")
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, "(skipped)", ha="center", va="center", transform=ax2.transAxes)

        plt.tight_layout()
        if save_plot:
            slug = name.lower().replace(" ", "_").replace("/", "_")
            path = f"horizon_{slug}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"\n  Plot saved: {path}")
        plt.close(fig)

        result["ground_elevation_m"] = ground_elev
        result["name"] = name
        return result

    def compare_sites(self, sites: dict[str, tuple[float, float]],
                      observer_height: float = 2.0,
                      save_plot: bool = True) -> list[dict]:
        """Analyze and compare multiple candidate sites."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        results = []
        for name, (lat, lon) in sites.items():
            r = self.analyze_site(name, lat, lon, observer_height,
                                  full_profile=False, save_plot=False)
            results.append(r)

        results.sort(key=lambda r: r["terrain_margin_deg"], reverse=True)

        has_canopy = self.chm is not None
        print(f"\n{'='*90}")
        print(" SITE COMPARISON - ranked by eclipse sun clearance")
        print(f"{'='*90}")

        hdr = f" {'#':<3} {'Site':<32} {'Elev':>5} {'Totality':>9} {'Horizon':>8} {'Margin':>8}"
        if has_canopy:
            hdr += f" {'Canopy':>8} {'C.Marg':>8}"
        hdr += f" {'Disk':>8} {'OK?':>7}"
        print(hdr)
        print(f" {'-'*100 if has_canopy else '-'*88}")

        for i, r in enumerate(results, 1):
            ec = r["eclipse"]
            dur = ec["duration_str"] if ec["in_totality"] else "-"
            line = (f" {i:<3} {r['name']:<32} {r['ground_elevation_m']:>5.0f} "
                    f"{dur:>9} {r['terrain_horizon_deg']:>8.2f} "
                    f"{r['terrain_margin_deg']:>+8.2f}")
            if has_canopy:
                line += (f" {r['canopy_horizon_deg']:>8.2f} "
                         f"{r['canopy_margin_deg']:>+8.2f}")
            ds = r.get("disk_status", "?")
            line += f" {ds:>8}"
            vis = "YES" if r["visible"] else "NO"
            if has_canopy and r["visible_terrain"] and not r["visible_canopy"]:
                vis = "TREE!"
            line += f" {vis:>7}"
            print(line)
        print()

        # Bar chart
        fig, ax = plt.subplots(figsize=(12, max(4, len(results) * 0.8 + 1)))
        labels = [f"{r['name']}\n({r['eclipse']['duration_str'] if r['eclipse']['in_totality'] else 'outside'})"
                  for r in results]
        margins = [r["terrain_margin_deg"] for r in results]
        colors = ["forestgreen" if m > 0 else "firebrick" for m in margins]

        bars = ax.barh(labels, margins, color=colors, edgecolor="black", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_xlabel("Clearance margin (degrees)")
        ax.set_title("Eclipse Sun Visibility - 2026 Aug 12\n(with totality duration per site)")
        ax.invert_yaxis()

        for bar, m in zip(bars, margins):
            ax.text(m + (0.1 if m >= 0 else -0.1),
                    bar.get_y() + bar.get_height() / 2,
                    f"{m:+.1f} deg", va="center",
                    ha="left" if m >= 0 else "right",
                    fontsize=10, fontweight="bold")

        ax.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()
        if save_plot:
            fig.savefig("eclipse_site_comparison.png", dpi=150, bbox_inches="tight")
            print("  Comparison plot saved: eclipse_site_comparison.png")
        plt.close(fig)

        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Eclipse Horizon Checker - 12 August 2026 total solar eclipse",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s                                    # Default Palencia sites\n"
            "  %(prog)s --lat 41.05 --lon -0.86 --name Alcaniz\n"
            "  %(prog)s --no-canopy                        # Skip tree analysis\n"
            "  %(prog)s --height 5 --lat 42.12 --lon -4.50 # Standing on wall\n"
        ),
    )
    parser.add_argument("--lat", type=float, help="Latitude of site")
    parser.add_argument("--lon", type=float, help="Longitude of site")
    parser.add_argument("--name", type=str, default="Custom site", help="Site name")
    parser.add_argument("--height", type=float, default=2.0,
                        help="Observer height above ground (m, default 2.0)")
    parser.add_argument("--no-canopy", action="store_true",
                        help="Disable canopy height analysis")
    parser.add_argument("--no-full-profile", action="store_true",
                        help="Skip full 360 deg profile")

    args = parser.parse_args()

    print("=" * 70)
    print("  ECLIPSE HORIZON CHECKER v0.2.2 - 12 August 2026")
    print("  Features: totality duration | terrain | canopy | solar disk")
    print("=" * 70)

    checker = EclipseHorizonChecker(use_canopy=not args.no_canopy)

    if args.lat is not None and args.lon is not None:
        checker.analyze_site(args.name, args.lat, args.lon,
                             observer_height=args.height,
                             full_profile=not args.no_full_profile)
    else:
        sites = {
            "Castillo de Monzon de Campos":      (42.1245, -4.4983),
            "Cristo del Otero (Palencia)":        (42.0230, -4.5195),
            "Torremomojon (Estrella de Campos)":  (41.8940, -4.6775),
            "Castillo Fuentes de Valdepero":      (42.0795, -4.4575),
            "Castillo de Ampudia":                (41.9170, -4.7815),
        }
        for name, (lat, lon) in sites.items():
            checker.analyze_site(name, lat, lon, observer_height=args.height,
                                 full_profile=not args.no_full_profile)
        checker.compare_sites(sites, observer_height=args.height)

    print("\nDone! Check .png files for horizon plots.")


if __name__ == "__main__":
    main()
