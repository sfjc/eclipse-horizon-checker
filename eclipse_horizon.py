#!/usr/bin/env python3
"""
eclipse_horizon.py — SRTM-based horizon visibility checker for the 2026 Spanish eclipse.

OVERVIEW
--------
This script downloads SRTM 1-arc-second (~30 m) elevation tiles and computes
the terrain horizon profile from any candidate observation point.  It then
compares the horizon with the sun's position during the August 12 2026 total
solar eclipse to tell you whether the sun will be blocked by terrain.

SETUP (one-time)
----------------
    pip install numpy matplotlib requests

SRTM DATA
----------
The script auto-downloads the required 1°×1° HGT tiles from the public
OpenTopography AWS mirror (no account needed):

    https://opentopography.s3.sdsc.edu/raster/SRTM_GL1/SRTM_GL1_srtm/<tile>.hgt

Tiles are cached in a local ./srtm_cache/ directory.  Each tile is ~25 MB
(uncompressed SRTM1, 3601×3601 signed-16-bit big-endian).

If the automatic download fails (corporate firewall, etc.), you can manually
place .hgt files in ./srtm_cache/.  Alternative sources:

  • USGS EarthExplorer  https://earthexplorer.usgs.gov  (free account required)
  • OpenTopography      https://portal.opentopography.org/raster?opentopoID=OTSRTM.082015.4326.1
  • Viewfinder Panoramas  https://viewfinderpanoramas.org/dem3.html  (3-arc-sec)
  • CGIAR-CSI            https://csidotinfo.wordpress.com/data/srtm-90m-digital-elevation-database-v4-1/

Tile naming: a tile named N42W005.hgt covers latitudes 42°–43° N and
longitudes 5°–4° W.

QUICK START
-----------
    from eclipse_horizon import EclipseHorizonChecker

    checker = EclipseHorizonChecker()

    # Full analysis for a candidate site
    checker.analyze_site(
        name="Castillo de Monzón de Campos",
        lat=42.1245, lon=-4.4983,
        observer_height=5.0,       # metres above ground (e.g. standing on wall)
    )

    # Compare several candidates
    checker.compare_sites({
        "Monzón de Campos":     (42.1245, -4.4983),
        "Cristo del Otero":     (42.0230, -4.5195),
        "Torremormojón":        (41.8940, -4.6775),
        "Fuentes de Valdepero": (42.0795, -4.4575),
        "Ampudia":              (41.9170, -4.7815),
    })

ECLIPSE PARAMETERS (Palencia area)
----------------------------------
    Date/time :  12 Aug 2026, ~20:29–20:31 CEST  (totality)
    Sun azimuth:  ~280°  (WNW)
    Sun altitude: ~9° above horizon
    Duration:     ~1 min 44 sec

Author: Generated for eclipse planning near Grijota, Palencia, Spain.
"""

import struct
import math
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Configuration — Eclipse parameters for the Palencia area
# ---------------------------------------------------------------------------
ECLIPSE = {
    "sun_azimuth_deg": 280.0,       # degrees from north, clockwise
    "sun_altitude_deg": 9.0,        # degrees above horizon at max eclipse
    "azimuth_tolerance_deg": 15.0,  # half-width of azimuth band to check
    "date": "2026-08-12",
    "time_local": "20:30 CEST",
    "totality_sec": 104,
}

# Earth parameters
EARTH_RADIUS_M = 6_371_000.0
# Standard atmospheric refraction coefficient (k ≈ 0.13)
# Effective earth radius = R / (1 - k) ≈ R × 1.15
REFRACTION_K = 0.13

# How far out to trace each ray (metres).  50 km is more than enough for
# terrain at 9° altitude; even a 1000 m peak only subtends 9° at ~6.3 km.
MAX_RAY_DISTANCE_M = 80_000
RAY_STEP_M = 30  # step size along each ray (matches SRTM1 resolution)

CACHE_DIR = Path("srtm_cache")

# ---------------------------------------------------------------------------
# SRTM tile I/O
# ---------------------------------------------------------------------------

def tile_filename(lat: int, lon: int) -> str:
    """Return the standard HGT filename for a 1°×1° tile."""
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}.hgt"


def download_tile(lat: int, lon: int, cache_dir: Path = CACHE_DIR) -> Path:
    """
    Download an SRTM1 HGT tile if not already cached.

    Tries multiple public mirrors in order:
      1. OpenTopography S3 (no auth)
      2. elevation-tiles-prod S3 (gzipped, no auth)

    Returns the path to the local .hgt file, or raises FileNotFoundError.
    """
    import requests, gzip, io

    cache_dir.mkdir(parents=True, exist_ok=True)
    fname = tile_filename(lat, lon)
    local_path = cache_dir / fname

    if local_path.exists() and local_path.stat().st_size > 1_000_000:
        return local_path

    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    tile_name = f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}"

    # Mirror list (order of preference)
    urls = [
        # OpenTopography — raw HGT (≈25 MB)
        f"https://opentopography.s3.sdsc.edu/raster/SRTM_GL1/SRTM_GL1_srtm/{tile_name}.hgt",
        # AWS elevation tiles — gzipped
        f"https://s3.amazonaws.com/elevation-tiles-prod/skadi/{ns}{abs(lat):02d}/{tile_name}.hgt.gz",
    ]

    for url in urls:
        print(f"  Downloading {fname} from {url.split('/')[2]} …", end=" ", flush=True)
        try:
            r = requests.get(url, timeout=120, stream=True)
            if r.status_code != 200:
                print(f"HTTP {r.status_code}, trying next mirror.")
                continue

            data = r.content
            if url.endswith(".gz"):
                data = gzip.decompress(data)

            local_path.write_bytes(data)
            size_mb = len(data) / 1e6
            print(f"OK ({size_mb:.1f} MB)")
            return local_path

        except Exception as e:
            print(f"failed ({e}), trying next mirror.")
            continue

    raise FileNotFoundError(
        f"Could not download tile {fname}.\n"
        f"Please download it manually and place it in {cache_dir}/\n"
        f"Sources:\n"
        f"  • https://earthexplorer.usgs.gov (search SRTM 1 Arc-Second)\n"
        f"  • https://portal.opentopography.org/raster?opentopoID=OTSRTM.082015.4326.1\n"
        f"  • https://viewfinderpanoramas.org/dem3.html\n"
    )


def load_tile(lat: int, lon: int, cache_dir: Path = CACHE_DIR) -> np.ndarray:
    """Load an SRTM1 HGT tile as a (3601, 3601) int16 numpy array."""
    path = download_tile(lat, lon, cache_dir)
    data = np.frombuffer(path.read_bytes(), dtype=">i2")  # big-endian int16

    # SRTM1 = 3601×3601, SRTM3 = 1201×1201
    side = int(math.sqrt(len(data)))
    if side not in (3601, 1201):
        raise ValueError(f"Unexpected tile size {len(data)} (side={side})")

    return data.reshape((side, side))


class SrtmElevationModel:
    """
    Lazy-loading mosaic of SRTM tiles, with bilinear interpolation.
    """

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self._tiles: dict[tuple[int, int], np.ndarray] = {}

    def _ensure_tile(self, lat_floor: int, lon_floor: int) -> Optional[np.ndarray]:
        key = (lat_floor, lon_floor)
        if key not in self._tiles:
            try:
                self._tiles[key] = load_tile(lat_floor, lon_floor, self.cache_dir)
            except FileNotFoundError as e:
                print(f"\n  ⚠  {e}\n  Treating as flat terrain (0 m).\n")
                self._tiles[key] = None
        return self._tiles[key]

    def get_elevation(self, lat: float, lon: float) -> float:
        """Return interpolated elevation (metres) at a geographic coordinate."""
        lat_floor = math.floor(lat)
        lon_floor = math.floor(lon)
        tile = self._ensure_tile(lat_floor, lon_floor)

        if tile is None:
            return 0.0  # missing tile — assume flat/sea level

        side = tile.shape[0]
        n = side - 1  # 3600 for SRTM1

        # Fractional position within tile (note: rows go top-to-bottom = N-to-S)
        y_frac = (lat - lat_floor)        # 0 at southern edge, 1 at northern edge
        x_frac = (lon - lon_floor)        # 0 at western edge,  1 at eastern edge

        # Pixel coordinates (row 0 = northern edge)
        row = (1.0 - y_frac) * n
        col = x_frac * n

        r0 = int(math.floor(row))
        c0 = int(math.floor(col))
        r1 = min(r0 + 1, n)
        c1 = min(c0 + 1, n)
        dr = row - r0
        dc = col - c0

        z00 = float(tile[r0, c0])
        z01 = float(tile[r0, c1])
        z10 = float(tile[r1, c0])
        z11 = float(tile[r1, c1])

        # Handle SRTM voids (-32768)
        vals = [z for z in [z00, z01, z10, z11] if z > -1000]
        if not vals:
            return 0.0

        # Bilinear interpolation
        elev = (z00 * (1 - dr) * (1 - dc) +
                z01 * (1 - dr) * dc +
                z10 * dr * (1 - dc) +
                z11 * dr * dc)
        return elev

    def preload_region(self, lat_min: float, lat_max: float,
                       lon_min: float, lon_max: float):
        """Pre-download all tiles covering a bounding box."""
        for lat in range(math.floor(lat_min), math.floor(lat_max) + 1):
            for lon in range(math.floor(lon_min), math.floor(lon_max) + 1):
                self._ensure_tile(lat, lon)


# ---------------------------------------------------------------------------
# Horizon computation
# ---------------------------------------------------------------------------

def destination_point(lat: float, lon: float, azimuth_deg: float,
                      distance_m: float) -> tuple[float, float]:
    """
    Given a starting point, azimuth and distance, return the destination
    point on a sphere (Vincenty direct formula, spherical).
    """
    φ1 = math.radians(lat)
    λ1 = math.radians(lon)
    θ = math.radians(azimuth_deg)
    δ = distance_m / EARTH_RADIUS_M  # angular distance

    φ2 = math.asin(math.sin(φ1) * math.cos(δ) +
                    math.cos(φ1) * math.sin(δ) * math.cos(θ))
    λ2 = λ1 + math.atan2(math.sin(θ) * math.sin(δ) * math.cos(φ1),
                          math.cos(δ) - math.sin(φ1) * math.sin(φ2))

    return math.degrees(φ2), math.degrees(λ2)


def compute_horizon_angle(dem: SrtmElevationModel,
                          lat: float, lon: float,
                          azimuth_deg: float,
                          observer_height_m: float = 2.0,
                          max_distance_m: float = MAX_RAY_DISTANCE_M,
                          step_m: float = RAY_STEP_M) -> tuple[float, float]:
    """
    Trace a ray from (lat, lon) at the given azimuth and return the
    maximum terrain elevation angle (the horizon angle) and the distance
    at which it occurs.

    Accounts for earth curvature and standard atmospheric refraction.

    Returns
    -------
    horizon_angle_deg : float
        Maximum elevation angle (degrees) of terrain along the ray.
        Positive = terrain protrudes above geometric horizon.
    horizon_distance_m : float
        Distance to the horizon-defining terrain point.
    """
    obs_elev = dem.get_elevation(lat, lon) + observer_height_m

    # Effective earth radius (with refraction)
    R_eff = EARTH_RADIUS_M / (1.0 - REFRACTION_K)

    max_angle = -90.0
    max_dist = 0.0

    dist = step_m
    while dist <= max_distance_m:
        plat, plon = destination_point(lat, lon, azimuth_deg, dist)
        target_elev = dem.get_elevation(plat, plon)

        # Height difference
        dh = target_elev - obs_elev

        # Earth curvature drop at distance d:  drop ≈ d² / (2·R_eff)
        curvature_drop = (dist * dist) / (2.0 * R_eff)

        # Apparent height difference (target appears lower by curvature_drop)
        apparent_dh = dh - curvature_drop

        # Elevation angle
        angle_rad = math.atan2(apparent_dh, dist)
        angle_deg = math.degrees(angle_rad)

        if angle_deg > max_angle:
            max_angle = angle_deg
            max_dist = dist

        # Optimisation: if we're far out and terrain is very low, increase step
        if dist > 10_000 and angle_deg < max_angle - 5:
            dist += step_m * 4
        else:
            dist += step_m

    return max_angle, max_dist


def compute_horizon_profile(dem: SrtmElevationModel,
                            lat: float, lon: float,
                            observer_height_m: float = 2.0,
                            azimuth_start: float = 0.0,
                            azimuth_end: float = 360.0,
                            azimuth_step: float = 1.0,
                            max_distance_m: float = MAX_RAY_DISTANCE_M,
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the full horizon profile around an observer.

    Returns
    -------
    azimuths : ndarray  — azimuth angles in degrees
    horizon_angles : ndarray — horizon elevation angle at each azimuth (degrees)
    horizon_distances : ndarray — distance to horizon-defining point (metres)
    """
    azimuths = np.arange(azimuth_start, azimuth_end, azimuth_step)
    horizon_angles = np.zeros_like(azimuths)
    horizon_distances = np.zeros_like(azimuths)

    total = len(azimuths)
    for i, az in enumerate(azimuths):
        if i % 30 == 0:
            print(f"\r  Computing horizon: {i}/{total} ({az:.0f}°) …", end="", flush=True)
        angle, dist = compute_horizon_angle(
            dem, lat, lon, az, observer_height_m, max_distance_m)
        horizon_angles[i] = angle
        horizon_distances[i] = dist
    print(f"\r  Computing horizon: {total}/{total} — done.           ")

    return azimuths, horizon_angles, horizon_distances


# ---------------------------------------------------------------------------
# Eclipse-specific analysis
# ---------------------------------------------------------------------------

def check_eclipse_visibility(dem: SrtmElevationModel,
                             lat: float, lon: float,
                             observer_height_m: float = 2.0,
                             sun_az: float = ECLIPSE["sun_azimuth_deg"],
                             sun_alt: float = ECLIPSE["sun_altitude_deg"],
                             half_width: float = ECLIPSE["azimuth_tolerance_deg"],
                             ) -> dict:
    """
    Check whether the eclipse sun will be visible from a candidate site.

    Returns a dict with:
      visible: bool — is the sun above the terrain horizon?
      sun_altitude: float — degrees
      horizon_at_sun_az: float — terrain horizon angle at the sun's azimuth
      margin: float — degrees of clearance (positive = sun visible)
      worst_horizon_in_band: float — max horizon angle in az ± half_width
      profile: (azimuths, angles, distances) — detailed profile in band
    """
    # Compute horizon in the band around the sun azimuth
    az_min = sun_az - half_width
    az_max = sun_az + half_width

    azimuths, angles, dists = compute_horizon_profile(
        dem, lat, lon, observer_height_m,
        azimuth_start=az_min, azimuth_end=az_max, azimuth_step=0.5,
        max_distance_m=MAX_RAY_DISTANCE_M,
    )

    # Exact horizon at sun azimuth
    idx_sun = np.argmin(np.abs(azimuths - sun_az))
    horizon_at_sun = angles[idx_sun]

    # Worst case in band
    worst_idx = np.argmax(angles)
    worst_horizon = angles[worst_idx]
    worst_az = azimuths[worst_idx]

    margin = sun_alt - horizon_at_sun
    visible = margin > 0

    return {
        "visible": visible,
        "sun_altitude_deg": sun_alt,
        "sun_azimuth_deg": sun_az,
        "horizon_at_sun_az_deg": horizon_at_sun,
        "margin_deg": margin,
        "worst_horizon_in_band_deg": worst_horizon,
        "worst_horizon_azimuth_deg": worst_az,
        "profile_azimuths": azimuths,
        "profile_angles": angles,
        "profile_distances": dists,
    }


# ---------------------------------------------------------------------------
# High-level analysis & plotting
# ---------------------------------------------------------------------------

class EclipseHorizonChecker:
    """
    Convenient wrapper for eclipse visibility analysis.
    """

    def __init__(self, cache_dir: str | Path = CACHE_DIR):
        self.dem = SrtmElevationModel(Path(cache_dir))

    def analyze_site(self, name: str, lat: float, lon: float,
                     observer_height: float = 2.0,
                     full_profile: bool = True,
                     save_plot: bool = True) -> dict:
        """
        Run a complete eclipse visibility analysis for one site.
        Prints a report and optionally saves plots.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        print(f"\n{'='*65}")
        print(f" ECLIPSE VISIBILITY ANALYSIS: {name}")
        print(f"{'='*65}")

        ground_elev = self.dem.get_elevation(lat, lon)
        print(f"  Location:  {lat:.5f}°N, {abs(lon):.5f}°W")
        print(f"  Ground elevation: {ground_elev:.0f} m (SRTM)")
        print(f"  Observer height above ground: {observer_height:.1f} m")
        print(f"  Total observer altitude: {ground_elev + observer_height:.0f} m")
        print()

        # Eclipse-band analysis
        result = check_eclipse_visibility(
            self.dem, lat, lon, observer_height)

        print(f"  Sun at eclipse:  az={result['sun_azimuth_deg']:.1f}°  "
              f"alt={result['sun_altitude_deg']:.1f}°")
        print(f"  Terrain horizon at sun azimuth: "
              f"{result['horizon_at_sun_az_deg']:.2f}°")
        print(f"  Clearance margin: {result['margin_deg']:.2f}°")

        if result["visible"]:
            print(f"\n  ✅  SUN IS VISIBLE — {result['margin_deg']:.1f}° "
                  f"of clearance above terrain.")
        else:
            print(f"\n  ❌  SUN IS BLOCKED — terrain is "
                  f"{abs(result['margin_deg']):.1f}° above the sun's position.")

        print(f"\n  Worst horizon in ±{ECLIPSE['azimuth_tolerance_deg']:.0f}° band: "
              f"{result['worst_horizon_in_band_deg']:.2f}° "
              f"at azimuth {result['worst_horizon_azimuth_deg']:.1f}°")

        # Full 360° profile
        if full_profile:
            print(f"\n  Computing full 360° horizon profile …")
            az_full, ang_full, dist_full = compute_horizon_profile(
                self.dem, lat, lon, observer_height,
                azimuth_start=0, azimuth_end=360, azimuth_step=1.0)
        else:
            az_full = ang_full = dist_full = None

        # --- Plotting ---
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f"Eclipse Horizon Analysis: {name}\n"
                     f"({lat:.4f}°N, {abs(lon):.4f}°W, "
                     f"ground elev {ground_elev:.0f} m)",
                     fontsize=13, fontweight="bold")

        # Left panel: eclipse azimuth band detail
        ax = axes[0]
        az_band = result["profile_azimuths"]
        ang_band = result["profile_angles"]

        ax.fill_between(az_band, 0, ang_band, alpha=0.3, color="saddlebrown",
                        label="Terrain horizon")
        ax.plot(az_band, ang_band, "saddlebrown", linewidth=2)
        ax.axhline(result["sun_altitude_deg"], color="gold", linewidth=2,
                   linestyle="--", label=f"Sun altitude ({result['sun_altitude_deg']:.1f}°)")
        ax.axvline(result["sun_azimuth_deg"], color="orange", linewidth=1,
                   linestyle=":", alpha=0.7, label=f"Sun azimuth ({result['sun_azimuth_deg']:.0f}°)")
        ax.set_xlabel("Azimuth (°)")
        ax.set_ylabel("Elevation angle (°)")
        ax.set_title("Eclipse azimuth band (detail)")
        ax.legend(fontsize=9)
        ax.set_ylim(bottom=min(-1, ang_band.min() - 1),
                    top=max(12, ang_band.max() + 2))
        ax.grid(True, alpha=0.3)

        # Shade the clearance zone
        for az_val, ang_val in zip(az_band, ang_band):
            if ang_val < result["sun_altitude_deg"]:
                ax.fill_between([az_val - 0.25, az_val + 0.25],
                                ang_val, result["sun_altitude_deg"],
                                color="lightgreen", alpha=0.02)

        # Right panel: full 360° horizon (if computed)
        ax2 = axes[1]
        if az_full is not None:
            ax2.fill_between(az_full, 0, ang_full, alpha=0.3, color="saddlebrown")
            ax2.plot(az_full, ang_full, "saddlebrown", linewidth=1.5,
                     label="Terrain horizon")
            ax2.axhline(result["sun_altitude_deg"], color="gold", linewidth=2,
                        linestyle="--", label=f"Sun altitude ({result['sun_altitude_deg']:.1f}°)")

            # Mark sun position
            ax2.plot(result["sun_azimuth_deg"], result["sun_altitude_deg"],
                     "o", color="gold", markersize=14, markeredgecolor="darkorange",
                     markeredgewidth=2, label="Sun at eclipse", zorder=5)

            # Shade eclipse band
            ax2.axvspan(result["sun_azimuth_deg"] - ECLIPSE["azimuth_tolerance_deg"],
                        result["sun_azimuth_deg"] + ECLIPSE["azimuth_tolerance_deg"],
                        alpha=0.1, color="orange", label="Eclipse azimuth band")

            ax2.set_xlabel("Azimuth (°)  [N=0, E=90, S=180, W=270]")
            ax2.set_ylabel("Elevation angle (°)")
            ax2.set_title("Full 360° horizon profile")
            ax2.set_xlim(0, 360)
            ax2.set_xticks(range(0, 361, 45))
            ax2.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"])
            ax2.set_ylim(bottom=min(-2, ang_full.min() - 1),
                         top=max(15, ang_full.max() + 2))
            ax2.legend(fontsize=9, loc="upper left")
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, "(full_profile=False;\n 360° plot skipped)",
                     ha="center", va="center", transform=ax2.transAxes)

        plt.tight_layout()
        if save_plot:
            slug = name.lower().replace(" ", "_").replace("/", "_")
            plot_path = f"horizon_{slug}.png"
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            print(f"\n  📊 Plot saved: {plot_path}")
        plt.close(fig)

        result["ground_elevation_m"] = ground_elev
        result["name"] = name
        return result

    def compare_sites(self, sites: dict[str, tuple[float, float]],
                      observer_height: float = 2.0,
                      save_plot: bool = True) -> list[dict]:
        """
        Analyze and compare multiple candidate sites.

        Parameters
        ----------
        sites : dict  name → (lat, lon)
        observer_height : float, metres above ground

        Returns sorted list of result dicts (best first).
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        results = []
        for name, (lat, lon) in sites.items():
            r = self.analyze_site(name, lat, lon, observer_height,
                                  full_profile=False, save_plot=False)
            results.append(r)

        # Sort by margin (best = most clearance)
        results.sort(key=lambda r: r["margin_deg"], reverse=True)

        # Summary table
        print(f"\n{'='*75}")
        print(f" SITE COMPARISON — ranked by eclipse sun clearance")
        print(f"{'='*75}")
        print(f" {'Rank':<5} {'Site':<30} {'Elev':>6} {'Horizon':>8} "
              f"{'Margin':>8} {'Visible':>8}")
        print(f" {'':5} {'':30} {'(m)':>6} {'(°)':>8} {'(°)':>8} {'':>8}")
        print(f" {'-'*5} {'-'*30} {'-'*6} {'-'*8} {'-'*8} {'-'*8}")

        for i, r in enumerate(results, 1):
            vis = "✅ YES" if r["visible"] else "❌ NO"
            print(f" {i:<5} {r['name']:<30} {r['ground_elevation_m']:>6.0f} "
                  f"{r['horizon_at_sun_az_deg']:>8.2f} "
                  f"{r['margin_deg']:>+8.2f} {vis:>8}")
        print()

        # Comparison bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        names = [r["name"] for r in results]
        margins = [r["margin_deg"] for r in results]
        colors = ["forestgreen" if m > 0 else "firebrick" for m in margins]

        bars = ax.barh(names, margins, color=colors, edgecolor="black", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_xlabel("Clearance margin (degrees above terrain horizon)")
        ax.set_title("Eclipse Sun Visibility Comparison\n"
                     f"Sun at {ECLIPSE['sun_azimuth_deg']:.0f}° azimuth, "
                     f"{ECLIPSE['sun_altitude_deg']:.0f}° altitude")
        ax.invert_yaxis()

        for bar, margin in zip(bars, margins):
            x_pos = margin + (0.1 if margin >= 0 else -0.1)
            ha = "left" if margin >= 0 else "right"
            ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                    f"{margin:+.1f}°", va="center", ha=ha, fontsize=10,
                    fontweight="bold")

        ax.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()

        if save_plot:
            fig.savefig("eclipse_site_comparison.png", dpi=150, bbox_inches="tight")
            print(f"  📊 Comparison plot saved: eclipse_site_comparison.png")
        plt.close(fig)

        return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Run the analysis for the five candidate sites near Grijota."""

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  ECLIPSE HORIZON CHECKER — 12 August 2026, Palencia, Spain ║")
    print("║  Sun at ~280° azimuth (WNW), ~9° altitude at totality      ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    checker = EclipseHorizonChecker()

    sites = {
        "Castillo de Monzón de Campos":     (42.1245, -4.4983),
        "Cristo del Otero (Palencia)":       (42.0230, -4.5195),
        "Torremormojón (Estrella de Campos)":(41.8940, -4.6775),
        "Castillo Fuentes de Valdepero":     (42.0795, -4.4575),
        "Castillo de Ampudia":               (41.9170, -4.7815),
    }

    # Individual analysis with full 360° profiles
    for name, (lat, lon) in sites.items():
        checker.analyze_site(name, lat, lon, observer_height=2.0,
                             full_profile=True, save_plot=True)

    # Comparative summary
    checker.compare_sites(sites, observer_height=2.0, save_plot=True)

    print("\n✨ Done!  Check the .png files for horizon plots.")
    print("   Adjust observer_height if viewing from a castle wall/tower.\n")


if __name__ == "__main__":
    main()
