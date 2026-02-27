# 🌑 Eclipse Horizon Checker

**Will terrain block your view of the [2026 total solar eclipse](https://en.wikipedia.org/wiki/Solar_eclipse_of_August_12,_2026)?**

On **12 August 2026**, a total solar eclipse crosses northern Spain near sunset. The sun will be just 2–13° above the horizon — even a small hill to the west-northwest could block your view of totality.

This tool uses satellite elevation data (SRTM, ~30 m resolution) to compute the terrain horizon from any candidate observation point and tells you whether the eclipsed sun will be visible or blocked.

> **[→ Launch the web app](https://sfjc.github.io/eclipse-horizon-checker/)** *(replace with your GitHub Pages URL)*

![Screenshot](screenshot.png)

---

## 🌐 Web App (for everyone)

The web app runs entirely in your browser — no installation, no server, no account needed.

1. **Open the app** at your GitHub Pages URL
2. **Click the map** to drop a pin at a candidate viewing location — or paste coordinates from Google Maps
3. **Set your observer height** (2 m for flat ground, 5–10 m on a castle wall)
4. **Click "Analyze all pins"** — the app downloads elevation tiles and computes the horizon profile
5. **Read your results** — green = sun visible, red = blocked by terrain

The app automatically computes the correct sun position (altitude and azimuth) for each location based on NASA eclipse path data.

### Features

- Eclipse path of totality drawn on the map (centerline + limits)
- Automatic sun position calculation for any point along the path
- Terrain horizon profile chart for each pin
- Multi-site comparison ranked by clearance margin
- Shareable URLs: `?lat=42.12&lon=-4.50&name=My+Hill&h=5`
- Works on mobile

### Hosting it yourself

The web app is a single `index.html` file. To host it:

1. Fork this repository
2. Go to **Settings → Pages → Source → Deploy from branch → main**
3. Your app will be live at `https://yourusername.github.io/eclipse-horizon-checker/`

No build step, no dependencies, no server.

---

## 🐍 Python Script (for power users)

The Python script offers higher-resolution analysis with SRTM1 (30 m) tiles, full 360° horizon profiles, and publication-quality plots.

### Setup

```bash
pip install numpy matplotlib requests
python eclipse_horizon.py
```

On first run, the script downloads the needed SRTM tiles (~25 MB each) into `./srtm_cache/`.

### Usage

```python
from eclipse_horizon import EclipseHorizonChecker

checker = EclipseHorizonChecker()

# Analyze a single site
checker.analyze_site("Castillo de Monzón", lat=42.1245, lon=-4.4983, observer_height=5.0)

# Compare multiple candidates
checker.compare_sites({
    "Monzón de Campos":  (42.1245, -4.4983),
    "Cristo del Otero":  (42.0230, -4.5195),
    "Torremormojón":     (41.8940, -4.6775),
})
```

### Output

- Per-site horizon plots (360° panoramic + eclipse band detail)
- Ranked comparison bar chart
- Console report with exact clearance margins

See [`PYTHON_README.md`](PYTHON_README.md) for full documentation.

---

## How it works

For each azimuth direction around the observer:

1. **Ray tracing** — steps outward in ~30–40 m increments, sampling terrain elevation
2. **Earth curvature correction** — accounts for the drop of the horizon at distance (d²/2R)
3. **Atmospheric refraction** — standard refraction coefficient (k=0.13) effectively increases Earth's radius
4. **Horizon angle** — the maximum apparent elevation angle of terrain along each ray
5. **Eclipse comparison** — compares the horizon at the sun's azimuth (~280°) with the sun's altitude (~2–13°)

**Positive margin** = sun clears the terrain. **Negative margin** = terrain blocks the sun.

## Eclipse parameters across Spain

| Region | Sun altitude | Sun azimuth | Notes |
|--------|-------------|-------------|-------|
| North coast (A Coruña–Bilbao) | 10–13° | 278° | Most forgiving |
| Meseta Norte (Palencia, Valladolid) | 8–10° | 281° | Good balance |
| South edge (near Madrid) | 6–8° | 283° | More critical |
| Mediterranean / Balearics | 2–5° | 286° | Very low sun — terrain critical |

## Accuracy notes

- **Resolution**: ~30–40 m horizontal. Individual buildings and trees are not resolved.
- **Surface model**: SRTM measures canopy/roof tops (radar surface model), not bare earth. Forests appear as extra terrain height.
- **Recommended**: Always verify results with a site visit. Check the actual western horizon with your own eyes, ideally at the same time of day.

## Data sources

- **Eclipse path**: Fred Espenak, NASA/GSFC ([eclipse.gsfc.nasa.gov](https://eclipse.gsfc.nasa.gov/SEpath/SEpath2001/SE2026Aug12Tpath.html))
- **Elevation (web app)**: [AWS Terrain Tiles](https://registry.opendata.aws/terrain-tiles/) (Mapzen/Tilezen, derived from SRTM/GMTED)
- **Elevation (Python)**: [SRTM 1 Arc-Second](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-digital-elevation-shuttle-radar-topography-mission-srtm-1) via OpenTopography
- **Map tiles**: © [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors, © [CARTO](https://carto.com/)

## License

MIT. Eclipse predictions by Fred Espenak, NASA's GSFC — used with permission.
