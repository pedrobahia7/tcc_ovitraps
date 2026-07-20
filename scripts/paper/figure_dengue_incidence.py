"""
Dengue incidence choropleth - Belo Horizonte

Standalone figure: mean Empirical-Bayes-smoothed dengue rate per IBGE
population sector, averaged over the full biweekly series. Split out
of figure_studyarea.py -- not currently referenced by any section in
6a441e20c1f1a66c183b3c38, kept here for a future figure.

Inputs:
  data/processed/dengue_per_capita.csv
  data/processed/bh_sectors_2022_with_populations.geojson
  data/complementar/MG_Municipios_2022/MG_Municipios_2022.shp
Output:
  6a441e20c1f1a66c183b3c38/Figures/figure_dengue_incidence.pdf
"""

from __future__ import annotations

import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings("ignore")

# =============================================================================
# CONSTANTS
# =============================================================================

SECTORS_GEOJSON = Path("data/processed/bh_sectors_2022_with_populations.geojson")
PER_CAPITA_CSV = Path("data/processed/dengue_per_capita.csv")
MUNICIPIOS_SHP = Path("data/complementar/MG_Municipios_2022/MG_Municipios_2022.shp")

OUTPUT_PATH = Path("6a441e20c1f1a66c183b3c38/Figures/figure_dengue_incidence.pdf")

MUNICIPALITY_NAME = "Belo Horizonte"
COLORBAR_PERCENTILE_CAP = 0.98
FIGSIZE = (5, 4.3)

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


def load_sector_incidence() -> gpd.GeoDataFrame:
    """Load IBGE sectors with the mean EB-smoothed dengue rate per
    sector, averaged over the full biweekly series.

    Returns:
        GeoDataFrame of sectors with an added mean_eb_rate_per_1000
        column.
    """
    per_capita = pd.read_csv(
        PER_CAPITA_CSV, usecols=["sector_id", "eb_rate_per_1000"], low_memory=False
    )
    per_capita["sector_id"] = per_capita["sector_id"].astype(str)
    mean_eb = (
        per_capita.groupby("sector_id")["eb_rate_per_1000"]
        .mean()
        .rename("mean_eb_rate_per_1000")
    )

    sectors = gpd.read_file(SECTORS_GEOJSON)
    sectors["sector_id"] = sectors["CD_SETOR"].astype(str)
    return sectors.merge(mean_eb, left_on="sector_id", right_index=True, how="left")


# =============================================================================
# PLOTTING
# =============================================================================


def plot_incidence_panel(ax: plt.Axes, boundary: gpd.GeoDataFrame, sectors: gpd.GeoDataFrame) -> None:
    """Draw the mean dengue-incidence choropleth panel onto ax."""
    vmax = sectors["mean_eb_rate_per_1000"].quantile(COLORBAR_PERCENTILE_CAP)
    sectors.plot(
        column="mean_eb_rate_per_1000",
        ax=ax,
        cmap="YlOrRd",
        vmin=0,
        vmax=vmax,
        linewidth=0.02,
        edgecolor="black",
        legend=True,
        legend_kwds={
            "label": "Mean EB dengue rate\n(cases per 1,000 pop.)\n"
            f"[capped at {int(COLORBAR_PERCENTILE_CAP * 100)}th pct.]",
            "shrink": 0.75,
            "extend": "max",
        },
        missing_kwds={"color": "lightgrey"},
    )
    boundary.boundary.plot(ax=ax, color="black", linewidth=0.8)
    ax.set_title("Mean dengue incidence by sector")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


def build_figure() -> None:
    """Assemble the panel and save the figure to OUTPUT_PATH."""
    boundary = load_municipality_boundary()
    sectors = load_sector_incidence()

    mpl.rcParams.update({"font.size": 9})
    fig, ax = plt.subplots(figsize=FIGSIZE)
    plot_incidence_panel(ax, boundary, sectors)

    plt.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, bbox_inches="tight")
    print(f"Sectors with dengue data: {sectors['mean_eb_rate_per_1000'].notna().sum()}/{len(sectors)}")
    print(f"Saved figure to {OUTPUT_PATH}")


if __name__ == "__main__":
    build_figure()
