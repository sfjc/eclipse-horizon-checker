#!/usr/bin/env python3
"""
prepare_canopy.py — Download and prepare canopy height data for the web app.

Downloads Meta/WRI Global Canopy Height tiles covering the 2026 eclipse path
across Spain, downsamples to ~30 m resolution, and exports a directory of small
1x1 degree PNG tiles that the web app loads on demand.

PREREQUISITES
    pip install numpy rasterio Pillow

USAGE
    python prepare_canopy.py

    Creates:
      canopy_tiles/           — directory of 1x1 degree PNG tiles
      canopy_tiles/tiles.json — metadata listing available tiles

    Copy the entire canopy_tiles/ folder alongside index.html when deploying.

    Optionally specify a custom bounding box or resolution:
      python prepare_canopy.py --lat-min 41 --lat-max 43 --lon-min -6 --lon-max -3
      python prepare_canopy.py --resolution 30

DATA SOURCE
    Meta/WRI Global Canopy Height Map (2024)
    s3://dataforgood-fb-data/forests/v1/alsgedi_global_v6_float/chm/
    1 m resolution, Cloud-Optimized GeoTIFF, EPSG:3857

OUTPUT FORMAT
    PNG tiles where each pixel = canopy height in metres (0-255, grayscale).
    Tiles named canopy_{lat}_{lon}.png (SW corner, signed integers).
    Empty tiles (all zeros) are omitted to save space.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

try:
    import rasterio
    from rasterio.windows import from_bounds
except ImportError:
    print("ERROR: rasterio is required. Install with: pip install rasterio")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow is required. Install with: pip install Pillow")
    sys.exit(1)


# -- Defaults ---------------------------------------------------------------
DEFAULT_LAT_MIN = 38.5
DEFAULT_LAT_MAX = 44.5
DEFAULT_LON_MIN = -10.5
DEFAULT_LON_MAX = 5.0
OUTPUT_RESOLUTION_M = 30

CHM_BASE_URL = "https://dataforgood-fb-data.s3.amazonaws.com/forests/v1/alsgedi_global_v6_float/chm"


def lat_lon_to_quadkey(lat, lon, zoom=9):
    """Convert a WGS84 coordinate to a Bing Maps quadkey."""
    lat_rad = math.radians(lat)
    n = 2 ** zoom
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2 * n)
    x = max(0, min(n - 1, x))
    y = max(0, min(n - 1, y))
    qk = ""
    for i in range(zoom, 0, -1):
        d = 0
        mask = 1 << (i - 1)
        if x & mask:
            d += 1
        if y & mask:
            d += 2
        qk += str(d)
    return qk


def quadkey_bounds(qk):
    """Return (lat_min, lon_min, lat_max, lon_max) for a quadkey."""
    zoom = len(qk)
    x = y = 0
    for i, c in enumerate(qk):
        mask = 1 << (zoom - 1 - i)
        d = int(c)
        if d & 1:
            x |= mask
        if d & 2:
            y |= mask
    n = 2 ** zoom
    lon_min = x / n * 360 - 180
    lon_max = (x + 1) / n * 360 - 180
    lat_max = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    lat_min = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return lat_min, lon_min, lat_max, lon_max


def to_mercator(lat, lon):
    """Convert WGS84 to EPSG:3857 Web Mercator coordinates."""
    x = lon * 20037508.34 / 180.0
    lat_rad = math.radians(lat)
    y = math.log(math.tan(math.pi / 4 + lat_rad / 2)) * 20037508.34 / math.pi
    return x, y


def find_quadkeys(lat_min, lat_max, lon_min, lon_max, zoom=9):
    """Find all unique quadkeys that cover a bounding box."""
    qks = set()
    step = 0.3
    lat = lat_min
    while lat <= lat_max:
        lon = lon_min
        while lon <= lon_max:
            qks.add(lat_lon_to_quadkey(lat, lon, zoom))
            lon += step
        lat += step
    return sorted(qks)


def read_canopy_tile(quadkey, bbox_merc, out_width, out_height):
    """Read a portion of a Meta/WRI canopy height COG tile (downsampled)."""
    url = f"{CHM_BASE_URL}/{quadkey}.tif"
    try:
        env = rasterio.Env(
            GDAL_HTTP_UNSAFESSL="YES",
            AWS_NO_SIGN_REQUEST="YES",
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
            GDAL_HTTP_CONNECTTIMEOUT="30",
            GDAL_HTTP_TIMEOUT="120",
        )
        with env:
            with rasterio.open(url) as ds:
                xmin, ymin, xmax, ymax = bbox_merc
                try:
                    window = from_bounds(xmin, ymin, xmax, ymax, ds.transform)
                except Exception:
                    return None

                win_col_off = max(0, int(window.col_off))
                win_row_off = max(0, int(window.row_off))
                win_width = min(int(window.width), ds.width - win_col_off)
                win_height = min(int(window.height), ds.height - win_row_off)

                if win_width <= 0 or win_height <= 0:
                    return None

                clamped = rasterio.windows.Window(
                    win_col_off, win_row_off, win_width, win_height)

                data = ds.read(
                    1,
                    window=clamped,
                    out_shape=(out_height, out_width),
                    resampling=rasterio.enums.Resampling.average,
                )
                data = data.astype(np.float32)
                data[data < 0] = 0
                data[data > 100] = 0
                data[np.isnan(data)] = 0
                return data

    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Prepare canopy height data for the eclipse web app")
    parser.add_argument("--lat-min", type=float, default=DEFAULT_LAT_MIN)
    parser.add_argument("--lat-max", type=float, default=DEFAULT_LAT_MAX)
    parser.add_argument("--lon-min", type=float, default=DEFAULT_LON_MIN)
    parser.add_argument("--lon-max", type=float, default=DEFAULT_LON_MAX)
    parser.add_argument("--resolution", type=float, default=OUTPUT_RESOLUTION_M,
                        help="Output resolution in metres (default: 30)")
    parser.add_argument("--output-dir", type=str, default="canopy_tiles",
                        help="Output directory (default: canopy_tiles)")
    args = parser.parse_args()

    lat_min, lat_max = args.lat_min, args.lat_max
    lon_min, lon_max = args.lon_min, args.lon_max
    res_m = args.resolution

    # Snap bounds outward to integer degrees (tile boundaries)
    tile_lat_min = math.floor(lat_min)
    tile_lat_max = math.ceil(lat_max)
    tile_lon_min = math.floor(lon_min)
    tile_lon_max = math.ceil(lon_max)

    print("=" * 65)
    print("  CANOPY HEIGHT DATA PREPARATION (tiled)")
    print(f"  Bounding box: {lat_min}-{lat_max} N, {lon_min}-{lon_max} E")
    print(f"  Tile grid: {tile_lat_min}-{tile_lat_max} N, {tile_lon_min}-{tile_lon_max} E")
    print(f"  Resolution: {res_m} m")
    print("=" * 65)

    # Calculate total mosaic dimensions
    lat_extent_m = (tile_lat_max - tile_lat_min) * 111320
    mid_lat = (tile_lat_min + tile_lat_max) / 2
    lon_extent_m = (tile_lon_max - tile_lon_min) * 111320 * math.cos(math.radians(mid_lat))
    out_width = int(lon_extent_m / res_m)
    out_height = int(lat_extent_m / res_m)
    print(f"\n  Full mosaic: {out_width} x {out_height} pixels")

    # Build mosaic
    output = np.zeros((out_height, out_width), dtype=np.float32)

    quadkeys = find_quadkeys(tile_lat_min, tile_lat_max, tile_lon_min, tile_lon_max)
    print(f"  Quadkeys to check: {len(quadkeys)}\n")

    success = 0
    skipped = 0
    for i, qk in enumerate(quadkeys):
        qk_lat_min, qk_lon_min, qk_lat_max, qk_lon_max = quadkey_bounds(qk)

        overlap_lat_min = max(tile_lat_min, qk_lat_min)
        overlap_lat_max = min(tile_lat_max, qk_lat_max)
        overlap_lon_min = max(tile_lon_min, qk_lon_min)
        overlap_lon_max = min(tile_lon_max, qk_lon_max)

        if overlap_lat_min >= overlap_lat_max or overlap_lon_min >= overlap_lon_max:
            skipped += 1
            continue

        col_start = int((overlap_lon_min - tile_lon_min) / (tile_lon_max - tile_lon_min) * out_width)
        col_end = int((overlap_lon_max - tile_lon_min) / (tile_lon_max - tile_lon_min) * out_width)
        row_start = int((tile_lat_max - overlap_lat_max) / (tile_lat_max - tile_lat_min) * out_height)
        row_end = int((tile_lat_max - overlap_lat_min) / (tile_lat_max - tile_lat_min) * out_height)

        tile_w = max(1, col_end - col_start)
        tile_h = max(1, row_end - row_start)

        ox_min, oy_min = to_mercator(overlap_lat_min, overlap_lon_min)
        ox_max, oy_max = to_mercator(overlap_lat_max, overlap_lon_max)

        print(f"  [{i+1}/{len(quadkeys)}] {qk} ", end="", flush=True)
        data = read_canopy_tile(qk, (ox_min, oy_min, ox_max, oy_max), tile_w, tile_h)

        if data is not None and data.max() > 0:
            h, w = data.shape
            target_h = min(h, row_end - row_start)
            target_w = min(w, col_end - col_start)
            output[row_start:row_start + target_h,
                   col_start:col_start + target_w] = data[:target_h, :target_w]
            print(f"OK (max: {data.max():.0f} m)")
            success += 1
        else:
            print("no data / empty")
            skipped += 1

    print(f"\n  Tiles with data: {success}")
    print(f"  Tiles skipped: {skipped}")

    output = np.clip(output, 0, 255).astype(np.uint8)
    nonzero = np.count_nonzero(output)
    total = output.size
    print(f"  Non-zero pixels: {nonzero} / {total} ({100*nonzero/total:.1f}%)")
    print(f"  Max canopy height: {output.max()} m")

    # -- Slice into 1x1 degree tiles ----------------------------------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    n_lat = tile_lat_max - tile_lat_min
    n_lon = tile_lon_max - tile_lon_min
    rows_per_tile = out_height // n_lat
    cols_per_tile = out_width // n_lon

    print(f"\n  Slicing into {n_lat} x {n_lon} = {n_lat * n_lon} potential tiles")
    print(f"  Each tile: {cols_per_tile} x {rows_per_tile} px")

    tile_list = []
    total_size = 0

    for lat_i in range(n_lat):
        tile_lat = tile_lat_max - 1 - lat_i  # SW corner latitude
        r_start = lat_i * rows_per_tile
        r_end = (lat_i + 1) * rows_per_tile if lat_i < n_lat - 1 else out_height

        for lon_i in range(n_lon):
            tile_lon = tile_lon_min + lon_i  # SW corner longitude
            c_start = lon_i * cols_per_tile
            c_end = (lon_i + 1) * cols_per_tile if lon_i < n_lon - 1 else out_width

            patch = output[r_start:r_end, c_start:c_end]

            if np.count_nonzero(patch) == 0:
                continue

            fname = f"canopy_{tile_lat}_{tile_lon}.png"
            img = Image.fromarray(patch, mode="L")
            img.save(out_dir / fname, optimize=True)
            fsize = (out_dir / fname).stat().st_size
            total_size += fsize

            tile_list.append({
                "lat": tile_lat,
                "lon": tile_lon,
                "file": fname,
                "kb": round(fsize / 1024, 1),
                "max_h": int(patch.max()),
            })

    # Save metadata
    meta = {
        "lat_min": tile_lat_min,
        "lat_max": tile_lat_max,
        "lon_min": tile_lon_min,
        "lon_max": tile_lon_max,
        "tile_size_deg": 1,
        "resolution_m": res_m,
        "tile_width": cols_per_tile,
        "tile_height": rows_per_tile,
        "encoding": "grayscale_uint8_metres",
        "source": "Meta/WRI Global Canopy Height Map 2024",
        "tiles": tile_list,
    }
    json_path = out_dir / "tiles.json"
    json_path.write_text(json.dumps(meta, indent=2))

    print(f"\n  Saved {len(tile_list)} non-empty tiles to {out_dir}/")
    print(f"  Total size: {total_size / 1024:.0f} KB ({total_size / 1024 / 1024:.1f} MB)")
    print(f"  Metadata: {json_path}")
    print(f"  Empty tiles skipped: {n_lat * n_lon - len(tile_list)}")

    if tile_list:
        tile_list.sort(key=lambda t: t["kb"], reverse=True)
        print(f"\n  Largest tiles:")
        for t in tile_list[:5]:
            print(f"    {t['file']:30s}  {t['kb']:6.1f} KB  (max {t['max_h']} m)")

    print(f"\n  Copy the {out_dir}/ folder alongside index.html to deploy.")
    print("  Done!")


if __name__ == "__main__":
    main()
