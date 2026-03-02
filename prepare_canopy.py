#!/usr/bin/env python3
"""
prepare_canopy.py — Download and prepare canopy height data for the web app.

Downloads Meta/WRI Global Canopy Height tiles covering the 2026 eclipse path
across Spain, downsamples to ~100 m resolution, and exports a single PNG file
that the web app can load without CORS issues.

PREREQUISITES
    pip install numpy rasterio Pillow

USAGE
    python prepare_canopy.py

    This creates two files in the current directory:
      canopy_spain.png   — canopy height map (R channel = height in metres)
      canopy_spain.json  — bounding box metadata for the web app

    Copy both files alongside index.html when deploying.

    Optionally specify a custom bounding box:
      python prepare_canopy.py --lat-min 41 --lat-max 43 --lon-min -6 --lon-max -3

DATA SOURCE
    Meta/WRI Global Canopy Height Map (2024)
    s3://dataforgood-fb-data/forests/v1/alsgedi_global_v6_float/chm/
    1 m resolution, Cloud-Optimized GeoTIFF, EPSG:3857

OUTPUT FORMAT
    PNG image where each pixel's R channel = canopy height in metres (0–255).
    G and B channels are zero. This makes decoding trivial in JavaScript:
        const height = imageData.data[pixelIndex * 4];  // R channel
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

try:
    import rasterio  # type: ignore[import-not-found]
    from rasterio.windows import from_bounds  # type: ignore[import-not-found]
except ImportError:
    print("ERROR: rasterio is required. Install with: pip install rasterio")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow is required. Install with: pip install Pillow")
    sys.exit(1)


# ── Defaults: bounding box covering the eclipse path across Spain ──────────
DEFAULT_LAT_MIN = 38.5
DEFAULT_LAT_MAX = 44.5
DEFAULT_LON_MIN = -10.5
DEFAULT_LON_MAX = 5.0

# Output resolution in metres (100 m is a good balance)
OUTPUT_RESOLUTION_M = 100

# Meta/WRI CHM base URL (public S3, no auth needed)
CHM_BASE_URL = "https://dataforgood-fb-data.s3.amazonaws.com/forests/v1/alsgedi_global_v6_float/chm"


def lat_lon_to_quadkey(lat: float, lon: float, zoom: int = 9) -> str:
    """Convert a WGS84 coordinate to a Bing Maps quadkey."""
    lat_rad = math.radians(lat)
    n = 2**zoom
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


def quadkey_bounds(qk: str) -> tuple[float, float, float, float]:
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
    n = 2**zoom
    lon_min = x / n * 360 - 180
    lon_max = (x + 1) / n * 360 - 180
    lat_max = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    lat_min = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return lat_min, lon_min, lat_max, lon_max


def to_mercator(lat: float, lon: float) -> tuple[float, float]:
    """Convert WGS84 to EPSG:3857 Web Mercator coordinates."""
    x = lon * 20037508.34 / 180.0
    lat_rad = math.radians(lat)
    y = math.log(math.tan(math.pi / 4 + lat_rad / 2)) * 20037508.34 / math.pi
    return x, y


def find_quadkeys(
    lat_min: float, lat_max: float, lon_min: float, lon_max: float, zoom: int = 9
) -> list[str]:
    """Find all unique quadkeys that cover a bounding box."""
    qks = set()
    step = 0.3  # degrees — fine enough to hit all tiles at zoom 9
    lat = lat_min
    while lat <= lat_max:
        lon = lon_min
        while lon <= lon_max:
            qks.add(lat_lon_to_quadkey(lat, lon, zoom))
            lon += step
        lat += step
    return sorted(qks)


def read_canopy_tile(
    quadkey: str,
    bbox_merc: tuple[float, float, float, float],
    out_width: int,
    out_height: int,
) -> np.ndarray | None:
    """
    Read a portion of a Meta/WRI canopy height COG tile.

    Uses rasterio's windowed reading to download only the pixels we need.
    Returns a 2D float32 array, or None if the tile doesn't exist.
    """
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
                # Window in the tile's coordinate system
                xmin, ymin, xmax, ymax = bbox_merc
                try:
                    window = from_bounds(xmin, ymin, xmax, ymax, ds.transform)
                except Exception:
                    return None

                # Clamp window to valid range
                win_col_off = max(0, int(window.col_off))
                win_row_off = max(0, int(window.row_off))
                win_width = min(int(window.width), ds.width - win_col_off)
                win_height = min(int(window.height), ds.height - win_row_off)

                if win_width <= 0 or win_height <= 0:
                    return None

                clamped = rasterio.windows.Window(
                    win_col_off, win_row_off, win_width, win_height
                )

                # Read at reduced resolution
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
        # Tile doesn't exist or network error — not all quadkeys have data
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Prepare canopy height data for the eclipse web app"
    )
    parser.add_argument("--lat-min", type=float, default=DEFAULT_LAT_MIN)
    parser.add_argument("--lat-max", type=float, default=DEFAULT_LAT_MAX)
    parser.add_argument("--lon-min", type=float, default=DEFAULT_LON_MIN)
    parser.add_argument("--lon-max", type=float, default=DEFAULT_LON_MAX)
    parser.add_argument(
        "--resolution",
        type=float,
        default=OUTPUT_RESOLUTION_M,
        help="Output resolution in metres (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="canopy_spain",
        help="Output filename prefix (default: canopy_spain)",
    )
    args = parser.parse_args()

    lat_min, lat_max = args.lat_min, args.lat_max
    lon_min, lon_max = args.lon_min, args.lon_max
    res_m = args.resolution

    print("=" * 65)
    print("  CANOPY HEIGHT DATA PREPARATION")
    print(f"  Bounding box: {lat_min}–{lat_max}°N, {lon_min}–{lon_max}°E")
    print(f"  Resolution: {res_m} m")
    print("=" * 65)

    # Calculate output image dimensions
    lat_extent_m = (lat_max - lat_min) * 111320
    mid_lat = (lat_min + lat_max) / 2
    lon_extent_m = (lon_max - lon_min) * 111320 * math.cos(math.radians(mid_lat))

    out_width = int(lon_extent_m / res_m)
    out_height = int(lat_extent_m / res_m)
    print(f"\n  Output image: {out_width} × {out_height} pixels")

    # Create output array
    output = np.zeros((out_height, out_width), dtype=np.float32)

    # Find all quadkeys covering the bbox
    quadkeys = find_quadkeys(lat_min, lat_max, lon_min, lon_max)
    print(f"  Quadkeys to check: {len(quadkeys)}")
    print()

    # Convert output bbox to Web Mercator
    x_min, y_min = to_mercator(lat_min, lon_min)
    x_max, y_max = to_mercator(lat_max, lon_max)

    # Process each quadkey
    success = 0
    skipped = 0
    for i, qk in enumerate(quadkeys):
        qk_lat_min, qk_lon_min, qk_lat_max, qk_lon_max = quadkey_bounds(qk)

        # Find the overlap between this tile and our output bbox
        overlap_lat_min = max(lat_min, qk_lat_min)
        overlap_lat_max = min(lat_max, qk_lat_max)
        overlap_lon_min = max(lon_min, qk_lon_min)
        overlap_lon_max = min(lon_max, qk_lon_max)

        if overlap_lat_min >= overlap_lat_max or overlap_lon_min >= overlap_lon_max:
            skipped += 1
            continue

        # Output pixel range for this tile's contribution
        col_start = int((overlap_lon_min - lon_min) / (lon_max - lon_min) * out_width)
        col_end = int((overlap_lon_max - lon_min) / (lon_max - lon_min) * out_width)
        row_start = int((lat_max - overlap_lat_max) / (lat_max - lat_min) * out_height)
        row_end = int((lat_max - overlap_lat_min) / (lat_max - lat_min) * out_height)

        tile_w = max(1, col_end - col_start)
        tile_h = max(1, row_end - row_start)

        # Convert overlap bounds to Mercator for the COG read
        ox_min, oy_min = to_mercator(overlap_lat_min, overlap_lon_min)
        ox_max, oy_max = to_mercator(overlap_lat_max, overlap_lon_max)

        print(f"  [{i + 1}/{len(quadkeys)}] {qk} ", end="", flush=True)
        data = read_canopy_tile(qk, (ox_min, oy_min, ox_max, oy_max), tile_w, tile_h)

        if data is not None and data.max() > 0:
            # Place into output array
            h, w = data.shape
            target_h = min(h, row_end - row_start)
            target_w = min(w, col_end - col_start)
            output[
                row_start : row_start + target_h, col_start : col_start + target_w
            ] = data[:target_h, :target_w]
            print(f"OK (max height: {data.max():.0f} m)")
            success += 1
        else:
            print("no data / empty")
            skipped += 1

    print(f"\n  Tiles with data: {success}")
    print(f"  Tiles skipped: {skipped}")

    # Clamp to uint8 (0–255 m range, 1 m precision)
    output = np.clip(output, 0, 255).astype(np.uint8)

    nonzero = np.count_nonzero(output)
    total = output.size
    print(f"  Non-zero pixels: {nonzero} / {total} ({100 * nonzero / total:.1f}%)")
    print(f"  Max canopy height: {output.max()} m")

    # Save as PNG (R channel = height)
    png_path = Path(f"{args.output}.png")
    img = Image.fromarray(output, mode="L")  # grayscale
    img.save(png_path, optimize=True)
    png_size = png_path.stat().st_size
    print(f"\n  Saved: {png_path} ({png_size / 1024:.0f} KB)")

    # Save metadata
    meta = {
        "lat_min": lat_min,
        "lat_max": lat_max,
        "lon_min": lon_min,
        "lon_max": lon_max,
        "width": out_width,
        "height": out_height,
        "resolution_m": res_m,
        "encoding": "grayscale_uint8_metres",
        "source": "Meta/WRI Global Canopy Height Map 2024",
        "description": "Pixel value = canopy height in metres (0-255)",
    }
    json_path = Path(f"{args.output}.json")
    json_path.write_text(json.dumps(meta, indent=2))
    print(f"  Saved: {json_path}")

    print("\n  Copy both files alongside index.html to enable canopy analysis.")
    print("  Done!")


if __name__ == "__main__":
    main()
