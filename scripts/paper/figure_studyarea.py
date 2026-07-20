"""
Study Area Figure - Belo Horizonte

Builds Figure 2 of the paper (sections/study_setting.tex,
\\label{fig:studyarea}): the currently active ovitrap network plotted
over a real, current basemap of Belo Horizonte (OpenStreetMap tiles
via folium), screenshotted to a static PNG with selenium/headless
Chrome for embedding in the LaTeX build.

The dengue-incidence-per-sector map that used to share this figure
was split out to figure_dengue_incidence.py -- reserved for a later
figure, not currently wired into any section.

Inputs:
  data/processed/ovitraps_data.csv
  data/complementar/MG_Municipios_2022/MG_Municipios_2022.shp
Output:
  6a441e20c1f1a66c183b3c38/Figures/figure_studyarea.png
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

warnings.filterwarnings("ignore")

# =============================================================================
# CONSTANTS
# =============================================================================

OVITRAPS_CSV = Path("data/processed/ovitraps_data.csv")
MUNICIPIOS_SHP = Path("data/complementar/MG_Municipios_2022/MG_Municipios_2022.shp")

FIGURES_DIR = Path("6a441e20c1f1a66c183b3c38/Figures")
HTML_OUTPUT_PATH = FIGURES_DIR / "figure_studyarea.html"
PNG_OUTPUT_PATH = FIGURES_DIR / "figure_studyarea.png"

MUNICIPALITY_NAME = "Belo Horizonte"
TRAP_COLOR = "#08519c"
BOUNDARY_COLOR = "#222222"
TILE_STYLE = "OpenStreetMap"
MAP_PADDING_FRAC = 0.02
SCREENSHOT_SIZE = (1200, 1300)  # (width, height) px
TILE_LOAD_WAIT_SECONDS = 3

# =============================================================================
# DATA LOADING
# =============================================================================


def load_municipality_boundary() -> gpd.GeoDataFrame:
    """Load and isolate the Belo Horizonte municipality boundary.

    Returns:
        Single-row GeoDataFrame with the BH polygon in EPSG:4326.
    """
    muni = gpd.read_file(MUNICIPIOS_SHP)
    bh = muni[muni["NM_MUN"].str.contains(MUNICIPALITY_NAME, case=False, na=False)]
    return bh.to_crs(4326)


def load_active_traps() -> pd.DataFrame:
    """Load one row per ovitrap that was still active in the last
    calendar year present in the dataset.

    "Active" is defined as having at least one collection (dt_col)
    recorded in the same calendar year as the dataset's most recent
    collection date.

    Returns:
        DataFrame with columns [narmad, latitude, longitude], one row
        per active trap.
    """
    ov = pd.read_csv(
        OVITRAPS_CSV,
        usecols=["narmad", "dt_col", "latitude", "longitude"],
        low_memory=False,
    )
    ov["dt_col"] = pd.to_datetime(ov["dt_col"], errors="coerce")

    last_seen = ov.groupby("narmad")["dt_col"].max()
    last_year = last_seen.dt.year.max()
    active_narmad = last_seen[last_seen.dt.year == last_year].index

    return ov[ov["narmad"].isin(active_narmad)].drop_duplicates("narmad")


# =============================================================================
# MAP BUILDING
# =============================================================================


def build_map(boundary: gpd.GeoDataFrame, traps: pd.DataFrame) -> folium.Map:
    """Build the folium map: real basemap + BH outline + trap markers.

    Args:
        boundary: BH municipality polygon, from load_municipality_boundary().
        traps: Active-trap coordinates, from load_active_traps().

    Returns:
        A folium.Map ready to be saved to HTML.
    """
    minx, miny, maxx, maxy = boundary.total_bounds
    pad_x = (maxx - minx) * MAP_PADDING_FRAC
    pad_y = (maxy - miny) * MAP_PADDING_FRAC

    fmap = folium.Map(tiles=TILE_STYLE, control_scale=True, zoom_control=False)
    fmap.fit_bounds([[miny - pad_y, minx - pad_x], [maxy + pad_y, maxx + pad_x]])

    folium.GeoJson(
        boundary.geometry,
        style_function=lambda _: {
            "color": BOUNDARY_COLOR,
            "weight": 2,
            "fillOpacity": 0,
        },
    ).add_to(fmap)

    for lat, lon in zip(traps["latitude"], traps["longitude"]):
        folium.CircleMarker(
            location=(lat, lon),
            radius=1.5,
            color=TRAP_COLOR,
            fill=True,
            fill_color=TRAP_COLOR,
            fill_opacity=0.85,
            weight=0,
        ).add_to(fmap)

    return fmap


def screenshot_map(html_path: Path, png_path: Path, size: tuple[int, int]) -> None:
    """Render the saved HTML map in headless Chrome and screenshot it.

    Args:
        html_path: Path to the folium-generated HTML file.
        png_path: Where to save the screenshot.
        size: (width, height) of the browser window / screenshot, in px.
    """
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument(f"--window-size={size[0]},{size[1]}")

    driver = webdriver.Chrome(options=options)
    try:
        driver.get(f"file://{html_path.resolve()}")
        time.sleep(TILE_LOAD_WAIT_SECONDS)  # let map tiles finish loading
        driver.save_screenshot(str(png_path))
    finally:
        driver.quit()


def build_figure() -> None:
    """Assemble the map, save its HTML, and screenshot it to PNG."""
    boundary = load_municipality_boundary()
    traps = load_active_traps()

    fmap = build_map(boundary, traps)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fmap.save(str(HTML_OUTPUT_PATH))
    screenshot_map(HTML_OUTPUT_PATH, PNG_OUTPUT_PATH, SCREENSHOT_SIZE)

    print(f"Active traps: {len(traps)}")
    print(f"Saved interactive map to {HTML_OUTPUT_PATH}")
    print(f"Saved screenshot to {PNG_OUTPUT_PATH}")


if __name__ == "__main__":
    build_figure()
